# Purpose-built chunk autograd paths for :class:`GatedDeltaNetModified`.
#
# Each variant fixes ``v_zero`` / ``q_one`` at compile time (no runtime flags on the autograd node).
# When ``v_zero``, ``v`` is not stored for backward: only ``ctx.v_shape`` is kept and an empty buffer is
# reconstructed for ``chunk_gated_delta_rule_bwd``; ``dv`` is returned as ``None`` to skip useless work.

from __future__ import annotations

from typing import Callable, Type

import torch

from sleepy_orangutan.gateddeltanet.fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from sleepy_orangutan.gateddeltanet.fla.ops.cp import FLACPContext
from sleepy_orangutan.gateddeltanet.fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)
from sleepy_orangutan.gateddeltanet.fla.ops.utils.index import prepare_chunk_indices
from sleepy_orangutan.gateddeltanet.fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _make_chunk_gdn_modified_class(*, v_zero: bool, q_one: bool, class_name: str) -> Type[torch.autograd.Function]:
    save_v_tensor = not v_zero

    class _ChunkGdnModifiedVariant(torch.autograd.Function):
        @staticmethod
        @input_guard
        @autocast_custom_fwd
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: torch.LongTensor | None,
            cu_seqlens_cpu: torch.LongTensor | None,
            use_qk_l2norm_in_kernel: bool,
            cp_context: FLACPContext | None,
            transpose_state_layout: bool,
            use_gate_in_kernel: bool,
            A_log: torch.Tensor | None,
            dt_bias: torch.Tensor | None,
            use_exp2: bool,
        ):
            q_rstd, k_rstd = None, None
            if use_qk_l2norm_in_kernel:
                q, q_rstd = l2norm_fwd(q)
                k, k_rstd = l2norm_fwd(k)

            chunk_indices = (
                prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
                if cu_seqlens is not None
                else None
            )
            g_raw = g
            g, o, A, final_state, initial_state, g_input = chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
                chunk_indices=chunk_indices,
                transpose_state_layout=transpose_state_layout,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                v_zero=v_zero,
                q_one=q_one,
                use_exp2=use_exp2,
            )
            if save_v_tensor:
                ctx.save_for_backward(
                    q,
                    q_rstd,
                    k,
                    k_rstd,
                    v,
                    g,
                    beta,
                    A,
                    initial_state,
                    cu_seqlens,
                    chunk_indices,
                    g_input,
                    A_log,
                    dt_bias,
                    g_raw,
                )
            else:
                ctx.v_shape = v.shape
                ctx.save_for_backward(
                    q,
                    q_rstd,
                    k,
                    k_rstd,
                    g,
                    beta,
                    A,
                    initial_state,
                    cu_seqlens,
                    chunk_indices,
                    g_input,
                    A_log,
                    dt_bias,
                    g_raw,
                )
            ctx.scale = scale
            ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
            ctx.cp_context = cp_context
            ctx.transpose_state_layout = transpose_state_layout
            ctx.use_gate_in_kernel = use_gate_in_kernel
            ctx.v_zero = v_zero
            ctx.q_one = q_one
            ctx.use_exp2 = use_exp2
            ctx.output_final_state = output_final_state
            ctx.save_v_tensor = save_v_tensor
            return o.to(q.dtype), final_state

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
            if ctx.save_v_tensor:
                (
                    q,
                    q_rstd,
                    k,
                    k_rstd,
                    v,
                    g,
                    beta,
                    A,
                    initial_state,
                    cu_seqlens,
                    chunk_indices,
                    g_input,
                    A_log,
                    dt_bias,
                    g_raw,
                ) = ctx.saved_tensors
            else:
                (
                    q,
                    q_rstd,
                    k,
                    k_rstd,
                    g,
                    beta,
                    A,
                    initial_state,
                    cu_seqlens,
                    chunk_indices,
                    g_input,
                    A_log,
                    dt_bias,
                    g_raw,
                ) = ctx.saved_tensors
                v = torch.empty(
                    ctx.v_shape,
                    device=q.device,
                    dtype=q.dtype,
                    requires_grad=False,
                )

            dq, dk, dv, db, dg, dh0, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A=A,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                cu_seqlens=cu_seqlens,
                cp_context=ctx.cp_context,
                chunk_indices=chunk_indices,
                transpose_state_layout=ctx.transpose_state_layout,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                g_input=g_input,
                A_log=A_log,
                dt_bias=dt_bias,
                v_zero=ctx.v_zero,
                q_one=ctx.q_one,
                use_exp2=ctx.use_exp2,
            )
            if ctx.use_qk_l2norm_in_kernel:
                dq = l2norm_bwd(q, q_rstd, dq)
                dk = l2norm_bwd(k, k_rstd, dk)
            dg_out = dg.to(g_raw) if dg is not None else None

            if ctx.save_v_tensor:
                dv_out = dv.to(v) if dv is not None else None
            else:
                dv_out = None

            # One slot per :meth:`forward` arg after ``ctx`` (17 total; no ``v_zero`` / ``q_one`` tensors).
            return (
                dq.to(q) if dq is not None else None,
                dk.to(k) if dk is not None else None,
                dv_out,
                dg_out,
                db.to(beta) if db is not None else None,
                None,
                dh0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                dA_log,
                ddt_bias,
                None,
            )

    _ChunkGdnModifiedVariant.__name__ = class_name
    _ChunkGdnModifiedVariant.__qualname__ = class_name
    return _ChunkGdnModifiedVariant


ChunkGdnModifiedFull = _make_chunk_gdn_modified_class(v_zero=False, q_one=False, class_name="ChunkGdnModifiedFull")
ChunkGdnModifiedVZero = _make_chunk_gdn_modified_class(v_zero=True, q_one=False, class_name="ChunkGdnModifiedVZero")
ChunkGdnModifiedQOne = _make_chunk_gdn_modified_class(v_zero=False, q_one=True, class_name="ChunkGdnModifiedQOne")
ChunkGdnModifiedV0Q1 = _make_chunk_gdn_modified_class(v_zero=True, q_one=True, class_name="ChunkGdnModifiedV0Q1")


def _chunk_gdn_modified_common(
    Fn: Type[torch.autograd.Function],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.LongTensor | None,
    cu_seqlens_cpu: torch.LongTensor | None,
    cp_context: FLACPContext | None,
    transpose_state_layout: bool,
    use_gate_in_kernel: bool,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    use_exp2: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q and k must have the same number of heads, "
            f"but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}",
        )
    H, HV = q.shape[2], v.shape[2]
    if HV % H != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by "
            f"num_heads (H={H}), but got HV % H = {HV % H}",
        )
    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True."
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return Fn.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
        transpose_state_layout,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_exp2,
    )


@torch.compiler.disable
def chunk_gdn_modified_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_exp2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _chunk_gdn_modified_common(
        ChunkGdnModifiedFull,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        cp_context,
        transpose_state_layout,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_exp2,
    )


@torch.compiler.disable
def chunk_gdn_modified_v_zero(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_exp2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _chunk_gdn_modified_common(
        ChunkGdnModifiedVZero,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        cp_context,
        transpose_state_layout,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_exp2,
    )


@torch.compiler.disable
def chunk_gdn_modified_q_one(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_exp2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _chunk_gdn_modified_common(
        ChunkGdnModifiedQOne,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        cp_context,
        transpose_state_layout,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_exp2,
    )


@torch.compiler.disable
def chunk_gdn_modified_v0q1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_exp2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _chunk_gdn_modified_common(
        ChunkGdnModifiedV0Q1,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        cp_context,
        transpose_state_layout,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_exp2,
    )


CHUNK_GDN_MODIFIED: dict[tuple[bool, bool], Callable[..., tuple[torch.Tensor, torch.Tensor | None]]] = {
    (False, False): chunk_gdn_modified_full,
    (True, False): chunk_gdn_modified_v_zero,
    (False, True): chunk_gdn_modified_q_one,
    (True, True): chunk_gdn_modified_v0q1,
}

__all__ = [
    "CHUNK_GDN_MODIFIED",
    "ChunkGdnModifiedFull",
    "ChunkGdnModifiedQOne",
    "ChunkGdnModifiedV0Q1",
    "ChunkGdnModifiedVZero",
    "chunk_gdn_modified_full",
    "chunk_gdn_modified_q_one",
    "chunk_gdn_modified_v0q1",
    "chunk_gdn_modified_v_zero",
]
