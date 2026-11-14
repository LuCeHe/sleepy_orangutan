"""
PyTorch-only Gated DeltaNet for CPU / no-Triton environments.

Same parameterization as :class:`~fla.layers.GatedDeltaNet` (projections, gates, head layout),
but forward uses :func:`naive_recurrent_gated_delta_rule` and plain ``silu`` on Q/K/V
(no short conv, no chunk/Triton kernels).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sleepy_orangutan.gateddeltanet.fla.ops.gated_delta_rule.naive import (
    naive_recurrent_gated_delta_rule,
)


def _naive_gdn_gate(
    g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``g = -A_log.exp() * softplus(g + dt_bias)`` (matches FLA ``naive_gdn_gate``)."""
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float()
    return (-a_log.float().exp() * F.softplus(g)).to(g.dtype)


def _rms_norm_gated(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with multiplicative SiLU gate (approximates ``FusedRMSNormGated``)."""
    x = x * F.silu(g)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * weight


class GatedDeltaNetNaive(nn.Module):
    """CPU-safe GDN: naive recurrent core, no Triton / causal_conv1d."""

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int | None = None,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs  # use_short_conv / mode not supported on naive path

        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = int(num_heads * head_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx
        self.norm_eps = norm_eps

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        a = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.a_log = nn.Parameter(torch.log(a))
        self.a_log._no_weight_decay = True
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        else:
            self.g_proj = None
            self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        del attention_mask, kwargs
        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))
        v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = _naive_gdn_gate(self.a_proj(hidden_states), self.a_log, self.dt_bias)
        g = g.to(dtype=hidden_states.dtype)
        if self.num_v_heads != self.num_heads:
            repeat = self.num_v_heads // self.num_heads
            q = q.repeat_interleave(repeat, dim=-2)
            k = k.repeat_interleave(repeat, dim=-2)

        o, _ = naive_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
        )

        if self.use_gate:
            gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = _rms_norm_gated(o, gate, self.o_norm_weight, self.norm_eps)
        else:
            o = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            o = o * self.o_norm_weight

        o = rearrange(o, "... h d -> ... (h d)")
        return self.o_proj(o), None, None
