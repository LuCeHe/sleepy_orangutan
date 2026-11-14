"""
Naive gated linear attention (CPU / no-Triton).

Causal linear attention with a matrix-valued state and multiplicative decay gate:

    h_t = α_t ⊙ h_{t-1} + k_t ⊗ v_t
    o_t = q_t · h_t

Q/K/V are projected from ``x``, passed through causal depthwise short conv + SiLU,
then L2-normalized (linear-attention convention). α_t = σ(W_g x_t) per head.

Pure PyTorch — intended for laptop ablations and copy-paste benchmarks.

Optional ``notes`` flags (``NaiveLinearAttentionSSM(notes=...)``):
- ``minussqk``: ``abs(-QK)`` then decay then ``@ V`` (linear attention).
- ``meanminussqk``: ``abs(mean(QK, keys)-QK) @ V`` (linear attention).

On native Qwen3 attention (``delta:none`` / QW3), the same flags use **softmax** on
``QK`` then post-process (manual path; cannot run inside SDPA/Flash kernels):
``(1 - softmax(QK)) @ V`` and ``abs(mean(softmax(QK)) - softmax(QK)) @ V``.
``2spda`` uses two SDPA calls: ``SDPA(Q,K,V)/2 + SDPA(Q,-K,V)/2``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal Conv1d on the channel axis of ``(B, T, C)``."""

    def __init__(self, channels: int, kernel_size: int, bias: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=bias,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = self.conv(x.transpose(1, 2))
        y = y[..., : x.shape[1]]
        return y.transpose(1, 2)


def _qk_mode_from_notes(notes: str) -> str:
    if "meanminussqk" in notes:
        return "meanminussqk"
    if "minussqk" in notes:
        return "minussqk"
    return "default"


def _qk_logits_to_weights(logits: torch.Tensor, qk_mode: str) -> torch.Tensor:
    if qk_mode == "minussqk":
        return (-logits).abs()
    if qk_mode == "meanminussqk":
        return (logits.mean(dim=-1, keepdim=True) - logits).abs()
    return logits


def _apply_softmax_qk_mode(
    attn_probs: torch.Tensor,
    qk_mode: str,
    *,
    key_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Post-softmax Q/K variants for causal softmax attention (Qwen3)."""
    if qk_mode == "minussqk":
        out = 1.0 - attn_probs
    elif qk_mode == "meanminussqk":
        out = (attn_probs.mean(dim=-1, keepdim=True) - attn_probs).abs()
    else:
        return attn_probs
    if key_valid is not None:
        out = out.masked_fill(~key_valid, 0.0)
    return out


def naive_gated_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Recurrent gated linear attention.

    Args:
        q, k: ``(B, T, H, K)``
        v: ``(B, T, H, V)``
        decay: ``(B, T, H)`` in ``(0, 1)`` (multiplicative retention per step)
    Returns:
        o: ``(B, T, H, V)``
    """
    b, t, h, k_dim = q.shape
    if scale is None:
        scale = k_dim ** -0.5
    q = q * scale

    out = v.new_zeros(b, t, h, v.shape[-1])
    state = v.new_zeros(b, h, k_dim, v.shape[-1])
    for i in range(t):
        alpha = decay[:, i].view(b, h, 1, 1)
        state = state * alpha + torch.einsum("bhk,bhv->bhkv", k[:, i], v[:, i])
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q[:, i], state)
    return out


def naive_gated_linear_attention_mult(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    *,
    scale: float | None = None,
    qk_mode: str = "default",
) -> torch.Tensor:
    """
    Gated linear attention via causal lower-triangular matmul (equivalent to
    :func:`naive_gated_linear_attention` when ``qk_mode`` is ``default``).

    For each head, ``o = W V`` with ``W = f(Q K^T) ⊙ D`` on the lower triangle
    (``D[t,s] = prod_{r=s+1}^t α_r``, ``D[t,t]=1``). ``f`` is identity unless
    ``qk_mode`` is ``minussqk`` (``abs(-QK)``) or ``meanminussqk``
    (``abs(mean(QK)-QK)``). Complexity O(T²) in sequence length.
    """
    b, t, h, k_dim = q.shape
    if scale is None:
        scale = k_dim ** -0.5
    q = q * scale

    q_h = q.transpose(1, 2)
    k_h = k.transpose(1, 2)
    v_h = v.transpose(1, 2)
    decay_h = decay.transpose(1, 2).clamp(min=1e-8)

    logits = torch.matmul(q_h, k_h.transpose(-2, -1))
    weights = _qk_logits_to_weights(logits, qk_mode)
    log_prefix = torch.cumsum(torch.log(decay_h), dim=-1)
    diff = log_prefix.unsqueeze(-1) - log_prefix.unsqueeze(-2)
    tril = torch.tril(torch.ones(t, t, device=q.device, dtype=torch.bool))
    decay_mat = torch.exp(diff.masked_fill(~tril.view(1, 1, t, t), float("-inf")))
    weights = weights * decay_mat
    out = torch.matmul(weights, v_h)
    return out.transpose(1, 2)


class NaiveLinearAttentionSSM(nn.Module):
    """
    One layer: short conv on Q/K/V + gated linear attention + output gate + ``o_proj``.

    Interface matches other analyzer blocks: ``forward(x) -> (y, None, None)``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        expand_v: float = 2.0,
        conv_size: int = 4,
        use_output_gate: bool = True,
        norm_eps: float = 1e-5,
        notes: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_v_dim = int(round(head_dim * expand_v))
        self.key_dim = num_heads * head_dim
        self.value_dim = num_heads * self.head_v_dim
        self.norm_eps = norm_eps
        self.use_output_gate = use_output_gate
        self.qk_mode = _qk_mode_from_notes(notes)

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.q_conv = CausalDepthwiseConv1d(self.key_dim, conv_size)
        self.k_conv = CausalDepthwiseConv1d(self.key_dim, conv_size)
        self.v_conv = CausalDepthwiseConv1d(self.value_dim, conv_size)

        if use_output_gate:
            self.z_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        else:
            self.z_proj = None
            self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    _attention_fn = staticmethod(naive_gated_linear_attention)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        del attention_mask, kwargs

        q = F.silu(self.q_conv(self.q_proj(hidden_states)))
        k = F.silu(self.k_conv(self.k_proj(hidden_states)))
        v = F.silu(self.v_conv(self.v_proj(hidden_states)))

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", h=self.num_heads, d=self.head_v_dim)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        decay = torch.sigmoid(self.g_proj(hidden_states))

        if self.qk_mode != "default":
            o = naive_gated_linear_attention_mult(q, k, v, decay, qk_mode=self.qk_mode)
        else:
            o = self._attention_fn(q, k, v, decay)

        if self.use_output_gate and self.z_proj is not None:
            z = rearrange(
                F.silu(self.z_proj(hidden_states)),
                "... (h d) -> ... h d",
                h=self.num_heads,
                d=self.head_v_dim,
            )
            o = o * z
        o = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
        o = o * self.o_norm_weight

        o = rearrange(o, "... h d -> ... (h d)")
        return self.o_proj(o), None, None


class NaiveLinearAttentionSSMMult(NaiveLinearAttentionSSM):
    """Same as :class:`NaiveLinearAttentionSSM` but uses causal matmul instead of a time loop."""

    _attention_fn = staticmethod(naive_gated_linear_attention_mult)


def test_naive_gated_linear_attention_mult_matches_recurrence() -> None:
    torch.manual_seed(0)
    b, t, h, k_dim, v_dim = 2, 7, 3, 8, 16
    q = torch.randn(b, t, h, k_dim)
    k = torch.randn(b, t, h, k_dim)
    v = torch.randn(b, t, h, v_dim)
    decay = torch.sigmoid(torch.randn(b, t, h))
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    y_loop = naive_gated_linear_attention(q, k, v, decay)
    y_mat = naive_gated_linear_attention_mult(q, k, v, decay)
    assert torch.allclose(y_loop, y_mat, rtol=1e-4, atol=1e-5)
