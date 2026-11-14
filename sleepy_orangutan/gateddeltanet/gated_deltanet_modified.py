"""
Gated DeltaNet layer with optional **v_zero** / **q_one** flags on both chunk and fused recurrent paths.

- **Both false:** standard gated delta (same as :class:`GatedDeltaNet`).
- **v_zero only:** recurrent uses **v = 0** (no value injection); **q** is unchanged.
  The layer skips **v_proj**, **v** short-conv, and SiLU on **v** (same math as zeroing **v** inside the kernel; **v_proj** gets no grad).
- **q_one only:** full **v** in the state; readout uses **q = 1** (sum over key dim in the op).
- **Both true:** **v = 0** and **q = 1** readout.

Chunk mode uses purpose-built :func:`CHUNK_GDN_MODIFIED` autograd nodes (fixed ``v_zero`` / ``q_one``; when
``v_zero``, ``v`` is not saved for backward). Fused inference uses ``fused_recurrent_gdn*`` entry points
(``q_ones_v_zero`` drops the ``q`` argument and uses the kernel's compact path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange

from sleepy_orangutan.gateddeltanet.fla.layers.gated_deltanet import GatedDeltaNet
from sleepy_orangutan.gateddeltanet.fla.layers.utils import (
    get_layer_cache,
    get_unpad_data,
    index_first_axis,
    pad_input,
    update_layer_cache,
)
from sleepy_orangutan.gateddeltanet.fla.ops.gated_delta_rule import (
    fused_recurrent_gated_delta_rule_q_ones_v_zero,
    fused_recurrent_gdn,
    fused_recurrent_gdn_q_one,
    fused_recurrent_gdn_v_zero,
)
from sleepy_orangutan.gateddeltanet.fla.ops.gated_delta_rule.chunk_gdn_modified import (
    CHUNK_GDN_MODIFIED,
)
from sleepy_orangutan.gateddeltanet.fla.ops.gated_delta_rule.gate import fused_gdn_gate

if TYPE_CHECKING:
    from typing import Any

    from transformers.processing_utils import Unpack

    Cache = Any


class GatedDeltaNetModified(GatedDeltaNet):
    """
    Same as :class:`GatedDeltaNet`, plus ``v_zero`` and ``q_one`` for four recurrent variants
    (full / v-only / q-only / both), in either ``chunk`` or ``fused_recurrent`` ``mode``.
    """

    def __init__(
        self,
        *args,
        v_zero: bool = False,
        q_one: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.v_zero = v_zero
        self.q_one = q_one

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training and not (self.v_zero or self.q_one):
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        b, tlen, _ = hidden_states.shape
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            if self.v_zero:
                conv_state_v = None
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            if not self.v_zero:
                v_flat, conv_state_v = self.v_conv1d(
                    x=self.v_proj(hidden_states),
                    cache=conv_state_v,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            if not self.v_zero:
                v_flat = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k))
        if self.v_zero:
            # Chunk op uses V_ZERO + empty v buffer (no HBM read / memset); see fla chunk + wy_fast.
            v = torch.empty(
                b,
                tlen,
                self.num_v_heads,
                self.head_v_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                requires_grad=False,
            )
        else:
            v = rearrange(v_flat, "... (h d) -> ... h d", d=self.head_v_dim)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        g_raw = self.a_proj(hidden_states)
        if mode == "chunk":
            chunk_fn = CHUNK_GDN_MODIFIED[(self.v_zero, self.q_one)]
            o, recurrent_state = chunk_fn(
                q=q,
                k=k,
                v=v,
                g=g_raw,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
            )
        elif mode == "fused_recurrent":
            g = fused_gdn_gate(g=g_raw, A_log=self.A_log, dt_bias=self.dt_bias)
            if self.v_zero and self.q_one:
                o, recurrent_state = fused_recurrent_gated_delta_rule_q_ones_v_zero(
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                    use_exp2=False,
                    transpose_state_layout=False,
                )
            elif self.v_zero:
                o, recurrent_state = fused_recurrent_gdn_v_zero(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                    use_exp2=False,
                    transpose_state_layout=False,
                )
            elif self.q_one:
                o, recurrent_state = fused_recurrent_gdn_q_one(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                    use_exp2=False,
                    transpose_state_layout=False,
                )
            else:
                o, recurrent_state = fused_recurrent_gdn(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                    use_exp2=False,
                    transpose_state_layout=False,
                )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        if self.use_gate:
            g_out = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
