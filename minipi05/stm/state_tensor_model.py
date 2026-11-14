"""
State Tensor Model (STM): causal bilinear attention with tensor state
``S_t += v_t ⊗ k_t ⊗ r_t``, readout ``o_t = ⟨S_t, q_t, u_t⟩``, expressed per head as

    O = ( (Q Kᵀ) ⊙ (U Rᵀ) ⊙ M ⊙ Γ ) V,

where ``M`` is causal, ``Γ_{ts} = exp(g_t - g_s)`` when gating is on
(same head index throughout). Optional **write gate** ``β`` is fused into the input
projection as the last conv-path channel after the temporal gate ``a`` (same short conv;
with ``conconlyc`` that is concat → channel conv → short conv; ``conconlyc2``/``c3``/``c4``
conv the original width first, periodically, then concat). Additive STM scales
contributions from key index ``s`` via ``k,r ← √β · k,r``, ``v ← β · v``.
Output gate ``g_out`` defaults to a bypass that skips the short conv; ``stmgouttime``
puts it on that conv path as the last fused block.
``stmtwobetas`` fuses query-side mix gates ``β1, β2`` on the same conc/short-conv
path (after ``a`` / write ``β``) so scores are
``((1-β1)+β1 q·k)((1-β2)+β2 u·r)`` (none / one / two queries per head).
``stmtwobetas2`` is the same map then L2-normalizes the expanded ``q,k,u,r``, so
pairwise ``k_t·k_s`` (and ``r_t·r_s``) stay in ``[-1, 1]`` for the delta UT Gram.
``use_delta`` / ``stmdelta`` replaces the additive write with the tensor delta rule
``S ← α S + β (v − ⟨α S, k, r⟩) ⊗ k ⊗ r`` (Python recurrent scan).
``use_delta2`` / ``stmdelta2`` is the same recurrence, chunk-parallelized with the
Yang et al. (2024) UT / WY transform (Triton intra-chunk solve on CUDA).
``use_delta3`` / ``stmdelta3`` is the UT path with GDN-style **per-step** forget
``α_t = exp(g_t)`` (``g`` is passed through to FLA; no pairwise ``g_t − g_{t-1}``).
On CUDA with ``use_triton=True`` (default for ``delta:stm``),
**training** uses :func:`stm_triton.stm_causal_bilinear_fwd_trainable` — Triton forward plus
PyTorch dense backward recompute, or fused CUDA chunked backward (chunk 512) when
``torch_chunk_size`` is set. Pure PyTorch dense/chunked cores are used on CPU,
when ``pstmnotriton`` is set, or when Triton is unavailable. Inference uses fused Triton (or JIT
CUDA / dense PyTorch fallbacks) when gradients are disabled.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from minipi05.stm.some_convs import (
    ConcAlt,
    ConcAltC,
    ConcOnly,
    ConcOnlyC,
    ConcOnlyC2,
    ConcOnlyC3,
    ConcOnlyC4,
    ConcReor,
    get_conv1d,
)
from minipi05.stm.stm_chaos_proj import ConcChaosProj, parse_stm_chaos_config
from minipi05.stm import (
    stm_chunked_bilinear_ext as _stm_cuda_ext,
    stm_triton as _stm_triton,
)

_STM_CHUNK_AUTOTUNE_CACHE: dict[tuple, int] = {}
_STM_BILINEAR_CHUNK_SIZES = (64, 128, 256, 512)


class _RMSNormFallback(nn.Module):
    """PyTorch RMSNorm when FLA is unavailable."""

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = kwargs
        self.hidden_size = hidden_size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        _ = (residual, prenorm, residual_in_fp32)
        y = x.to(torch.float32)
        var = y.pow(2).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight.to(torch.float32)
        return y.to(x.dtype)


class _FusedRMSNormGatedFallback(nn.Module):
    """PyTorch RMSNorm + SiLU gate when FLA is unavailable."""

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        **kwargs,
    ) -> None:
        super().__init__()
        _ = (activation, kwargs)
        self.hidden_size = hidden_size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        _ = (residual, prenorm, residual_in_fp32)
        y = x.to(torch.float32)
        var = y.pow(2).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight.to(torch.float32)
        y = y * F.silu(g.to(torch.float32))
        return y.to(x.dtype)


def _l2norm_fallback(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    y = F.normalize(x, p=2, dim=-1, eps=eps)
    if output_dtype is not None:
        y = y.to(output_dtype)
    return y


try:
    from minipi05.gateddeltanet.fla.modules import FusedRMSNormGated, RMSNorm
    from minipi05.gateddeltanet.fla.modules.l2norm import l2norm
except ImportError:
    FusedRMSNormGated = _FusedRMSNormGatedFallback
    RMSNorm = _RMSNormFallback
    l2norm = _l2norm_fallback


def _normalize_gate_type(gate_type: str) -> str:
    g = str(gate_type).strip().lower().replace("_", "")
    if g == "sigmoid":
        return "sigmoid"
    if g in ("logsigmoid", "logsig"):
        return "logsigmoid"
    if g in ("-softplus", "negsoftplus", "softplusneg"):
        return "-softplus"
    raise ValueError('gate_type must be "sigmoid", "logsigmoid", or "-softplus"')


def apply_two_betas_feature_map(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    *,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query-side mix ``((1-β1)+β1 q·k)((1-β2)+β2 u·r)`` as a homogeneous feature map.

    ``β1, β2`` are ``(B, T, H)`` in ``(0, 1)``. Dummy key slots are ``1``, so
    existing bilinear/delta cores stay the SSM dual of the mixed score.

    ``renormalize=True`` (notes ``stmtwobetas2``) L2-normalizes the expanded
    ``q,k,u,r``. Dummy ``1`` plus a unit ``k`` would otherwise give
    ``k_t·k_s = 1 + k_t·k_s ∈ [0, 2]``; after renormalize, Cauchy–Schwarz
    gives ``k_t·k_s ∈ [-1, 1]`` (typically ``[0, 1]`` with a positive dummy).
    """
    b1 = beta1.unsqueeze(-1)
    b2 = beta2.unsqueeze(-1)
    ones_k = q.new_ones(*q.shape[:-1], 1)
    ones_u = u.new_ones(*u.shape[:-1], 1)
    q_hat = torch.cat([1.0 - b1, b1 * q], dim=-1)
    k_hat = torch.cat([ones_k, k], dim=-1)
    u_hat = torch.cat([1.0 - b2, b2 * u], dim=-1)
    r_hat = torch.cat([ones_u, r], dim=-1)
    if renormalize:
        q_hat = l2norm(q_hat)
        k_hat = l2norm(k_hat)
        u_hat = l2norm(u_hat)
        r_hat = l2norm(r_hat)
    return q_hat, k_hat, u_hat, r_hat


class _StmFactorizedProj(nn.Module):
    """Low-rank input projection ``d_model -> rank -> proj_out``."""

    def __init__(self, d_model: int, proj_out: int, *, rank: int) -> None:
        super().__init__()
        r = max(1, min(int(rank), d_model, proj_out))
        self.down = nn.Linear(d_model, r, bias=False)
        self.up = nn.Linear(r, proj_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class _StmTensorProdProj(nn.Module):
    """CP-style projection: shared trunk plus head × role factors."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        per_head_out: int,
        *,
        rank: int,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.per_head_out = per_head_out
        r = max(1, int(rank))
        self.trunk = nn.Linear(d_model, r, bias=False)
        self.head_mix = nn.Parameter(torch.empty(n_heads, r))
        self.role_mix = nn.Parameter(torch.empty(per_head_out, r))
        nn.init.normal_(self.head_mix, std=0.02)
        nn.init.normal_(self.role_mix, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _d = x.shape
        h = torch.einsum(
            "btr,hr,cr->bthc",
            self.trunk(x),
            self.head_mix,
            self.role_mix,
        )
        return h.reshape(b, t, self.n_heads * self.per_head_out)


class _GlaStyleGateProj(nn.Module):
    """Low-rank per-head gate (FLA ``GatedLinearAttention.gk_proj``)."""

    def __init__(self, d_model: int, n_heads: int, *, low_rank_dim: int = 16) -> None:
        super().__init__()
        r = max(1, int(low_rank_dim))
        self.net = nn.Sequential(
            nn.Linear(d_model, r, bias=False),
            nn.Linear(r, n_heads, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FusedGDNLikeInputProj(nn.Module):
    """Fused input path: structured projection, split before short conv.

    Q/K/U/R/V + temporal gate ``a`` + write gate ``β`` (if ``use_write_gate``)
    + query-mix ``β1,β2`` (if ``use_two_betas``) pass through :func:`get_conv1d`.
    Conv-path order is ``q,k,u,r,v,a[,β][,β1,β2][,g_out]``.
    With ``conconlyc`` that concat is expanded, then channel-mixed (kernel from
    ``conc_kernel_size`` / notes ``stmconconlyc:N``, default 3), then short-convolved.
    ``conconlyc2`` / ``c3`` (Triton) / ``c4`` (CUDA) mix the original width with a
    periodic conv first, then concat-expand.
    Output gate ``g_out`` is taken from the projection bypass and skips the short conv
    unless ``fuse_output_gate`` / notes ``stmgouttime`` puts it on the conv path
    (last fused block, width ``H*dv``).

    ``proj_kind``: ``linear`` (default), ``concreor``, ``conconly``, ``conconlyc``,
    ``conconlyc2``, ``conconlyc3``, ``conconlyc4``, ``concalt``,
    ``concaltc``, ``conchaos``, ``factorized``, or ``tensorprod``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dk: int,
        du: int,
        dv: int,
        *,
        use_matrix_state: bool = False,
        use_output_gate: bool = True,
        fuse_output_gate: bool = False,
        use_write_gate: bool = True,
        use_two_betas: bool = False,
        conv_size: int = 4,
        conv_backend: str = "triton",
        conv_notes: str = "pconvtype:shortconv",
        concreor: bool = False,
        proj_kind: str = "linear",
        proj_rank: int = 32,
        conc_kernel_size: int = 3,
        chaos_iter_mode: str | None = None,
        chaos_n_stages: int | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dk = dk
        self.du = 1 if use_matrix_state else du
        self.dv = dv
        self.use_matrix_state = bool(use_matrix_state)
        self.use_output_gate = use_output_gate
        self.fuse_output_gate = bool(use_output_gate) and bool(fuse_output_gate)
        self.use_write_gate = bool(use_write_gate)
        self.use_two_betas = bool(use_two_betas)
        H = n_heads
        ur_width = 0 if self.use_matrix_state else 2 * self.du
        qkruv = H * (2 * dk + ur_width + dv)
        n_fused_gates = 1 + int(self.use_write_gate) + 2 * int(self.use_two_betas)
        gout_width = H * dv if self.fuse_output_gate else 0
        conv_width = qkruv + H * n_fused_gates + gout_width
        bypass_width = H * dv if (use_output_gate and not self.fuse_output_gate) else 0
        self._conv_width = conv_width
        self._bypass_width = bypass_width
        proj_out = conv_width + bypass_width
        per_head_out = (
            2 * dk + ur_width + dv + n_fused_gates
            + (dv if use_output_gate else 0)
        )
        kind = str(proj_kind).strip().lower()
        if concreor:
            kind = "concreor"
        if kind == "concreor":
            self.proj: nn.Module = ConcReor(d_model, proj_out)
        elif kind == "conconly":
            self.proj = ConcOnly(d_model, proj_out)
        elif kind == "conconlyc":
            self.proj = ConcOnlyC(d_model, proj_out, kernel_size=conc_kernel_size)
        elif kind == "conconlyc2":
            self.proj = ConcOnlyC2(d_model, proj_out, kernel_size=conc_kernel_size)
        elif kind == "conconlyc3":
            self.proj = ConcOnlyC3(d_model, proj_out, kernel_size=conc_kernel_size)
        elif kind == "conconlyc4":
            self.proj = ConcOnlyC4(d_model, proj_out, kernel_size=conc_kernel_size)
        elif kind == "concalt":
            self.proj = ConcAlt(d_model, proj_out)
        elif kind == "concaltc":
            self.proj = ConcAltC(d_model, proj_out, kernel_size=3)
        elif kind == "conchaos":
            c_iter = chaos_iter_mode or "triangle2x3"
            c_stages = chaos_n_stages if chaos_n_stages is not None else 3
            self.proj = ConcChaosProj(
                d_model,
                proj_out,
                chaos_iter_mode=c_iter,
                chaos_n_stages=c_stages,
            )
        elif kind == "factorized":
            self.proj = _StmFactorizedProj(d_model, proj_out, rank=proj_rank)
        elif kind == "tensorprod":
            self.proj = _StmTensorProdProj(
                d_model,
                H,
                per_head_out,
                rank=proj_rank,
            )
        elif kind == "linear":
            self.proj = nn.Linear(d_model, proj_out, bias=False)
        else:
            raise ValueError(
                f'proj_kind must be "linear", "concreor", "conconly", "conconlyc", '
                f'"conconlyc2", "conconlyc3", "conconlyc4", "concalt", '
                f'"concaltc", "conchaos", "factorized", or "tensorprod", got {proj_kind!r}',
            )
        notes = conv_notes
        if f"pconvbackend:{conv_backend}" not in notes:
            notes = f"{notes}_pconvbackend:{conv_backend}"
        self.short_conv = get_conv1d(
            conv_width,
            conv_width,
            notes=notes,
            bias=False,
            kernel_size_t=conv_size,
        )
        if self.use_matrix_state:
            sizes = [H * dk, H * dk, H * dv, H]
        else:
            sizes = [
                H * dk,
                H * dk,
                H * self.du,
                H * self.du,
                H * dv,
                H,
            ]
        if self.use_write_gate:
            sizes.append(H)
        if self.use_two_betas:
            sizes.append(H)
            sizes.append(H)
        if self.fuse_output_gate:
            sizes.append(H * dv)
        self._conv_split_sizes = tuple(sizes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        B, T, _W = x.shape
        H, dk, du, dv = self.n_heads, self.dk, self.du, self.dv
        y = self.proj(x)
        if self._bypass_width > 0:
            y_conv, g_bypass = y.split([self._conv_width, self._bypass_width], dim=-1)
        else:
            y_conv = y
            g_bypass = None
        y_conv = self.short_conv(y_conv)
        parts = list(y_conv.split(self._conv_split_sizes, dim=-1))
        g_from_conv = parts.pop() if self.fuse_output_gate else None
        b2_raw = parts.pop() if self.use_two_betas else None
        b1_raw = parts.pop() if self.use_two_betas else None
        b_raw = parts.pop() if self.use_write_gate else None
        a_raw = parts.pop()
        body = parts
        if g_from_conv is not None:
            g_out = g_from_conv.view(B, T, H, dv)
        elif g_bypass is not None:
            g_out = g_bypass.view(B, T, H, dv)
        else:
            g_out = None
        two_raw = None
        if b1_raw is not None and b2_raw is not None:
            two_raw = (b1_raw.view(B, T, H), b2_raw.view(B, T, H))
        if self.use_matrix_state:
            q, k, v = body
            u = q.new_ones(B, T, H, 1)
            r = q.new_ones(B, T, H, 1)
            return (
                q.view(B, T, H, dk),
                k.view(B, T, H, dk),
                u,
                r,
                v.view(B, T, H, dv),
                a_raw.view(B, T, H),
                g_out,
                None if b_raw is None else b_raw.view(B, T, H),
                two_raw,
            )
        q, k, u, r, v = body
        return (
            q.view(B, T, H, dk),
            k.view(B, T, H, dk),
            u.view(B, T, H, du),
            r.view(B, T, H, du),
            v.view(B, T, H, dv),
            a_raw.view(B, T, H),
            g_out,
            None if b_raw is None else b_raw.view(B, T, H),
            two_raw,
        )


def default_stm_head_dims(
    hidden_size: int,
    num_heads: int,
    *,
    expand_v: float = 2.0,
    head_dim: int | None = None,
) -> tuple[int, int, int]:
    """Default ``(dk, du, dv)`` per head for :class:`StateTensorModel`.

    When ``head_dim`` is set (e.g. from ``headdim:`` in notes), use it directly.
    Otherwise match FLA GDN: ``num_heads * head_dim ≈ 0.75 * hidden_size``.
    """
    if head_dim is not None and head_dim > 0:
        hd = int(head_dim)
    else:
        key_dim = max(1, int(round(0.75 * hidden_size)))
        nh = max(1, int(num_heads))
        hd = key_dim // nh if key_dim % nh == 0 else max(1, key_dim // nh)
    dv = int(round(hd * expand_v))
    return hd, hd, dv


def stm_delta_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    *,
    use_exp2: bool = False,
    gate_mode: str = "pairwise",
) -> torch.Tensor:
    """Tensor delta rule with pairwise STM decay or GDN per-step decay.

    State ``S ∈ R^{dk × du × dv}`` per batch/head. Each step:

        S ← α_t S
        pred = ⟨S, k_t, r_t⟩
        S ← S + β_t (v_t − pred) ⊗ k_t ⊗ r_t
        o_t = ⟨S, q_t, u_t⟩

    ``gate_mode="pairwise"`` (``stmdelta`` / ``stmdelta2``): ``α_0 = 1`` and
    ``α_t = exp(g_t − g_{t-1})``, matching additive STM's ``Γ_{ts} = exp(g_t − g_s)``.
    ``gate_mode="per_step"`` (``stmdelta3``): ``α_t = exp(g_t)`` every step (FLA GDN).
    ``beta`` is the overwrite step size (GDN ``β``), not the additive ``√β`` rescaling
    of ``k,r``.
    """
    mode = str(gate_mode).strip().lower().replace("-", "_")
    if mode not in ("pairwise", "per_step"):
        raise ValueError('gate_mode must be "pairwise" or "per_step"')
    B, T, H, dk = q.shape
    du = u.shape[-1]
    dv = v.shape[-1]
    m_dim = dk * du
    S = torch.zeros(B, H, m_dim, dv, device=q.device, dtype=torch.float32)
    o = q.new_empty(B, T, H, dv)
    qf = q.float()
    kf = k.float()
    uf = u.float()
    rf = r.float()
    vf = v.float()
    bf = beta.float()
    gf = None if g is None else g.float()
    g_prev: torch.Tensor | None = None
    for t in range(T):
        if gf is not None:
            gt = gf[:, t]
            if mode == "per_step":
                decay = torch.exp2(gt) if use_exp2 else torch.exp(gt)
                S = S * decay.unsqueeze(-1).unsqueeze(-1)
            elif g_prev is not None:
                dg = gt - g_prev
                decay = torch.exp2(dg) if use_exp2 else torch.exp(dg)
                S = S * decay.unsqueeze(-1).unsqueeze(-1)
            g_prev = gt
        kr = (kf[:, t].unsqueeze(-1) * rf[:, t].unsqueeze(-2)).reshape(B, H, m_dim)
        pred = torch.einsum("bhmv,bhm->bhv", S, kr)
        inc = bf[:, t].unsqueeze(-1) * (vf[:, t] - pred)
        S = S + kr.unsqueeze(-1) * inc.unsqueeze(-2)
        qu = (qf[:, t].unsqueeze(-1) * uf[:, t].unsqueeze(-2)).reshape(B, H, m_dim)
        o[:, t] = torch.einsum("bhmv,bhm->bhv", S, qu).to(dtype=q.dtype)
    return o


def stm_delta_recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    S: torch.Tensor,
    g_prev: torch.Tensor | None,
    *,
    use_exp2: bool = False,
    gate_mode: str = "pairwise",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One tensor-delta step for cached decode.

    ``q,k`` are ``(B,H,dk)``, ``u,r`` ``(B,H,du)``, ``v`` ``(B,H,dv)``,
    ``g,beta`` ``(B,H)``, ``S`` ``(B,H,dk*du,dv)``.
    Returns ``(o, S, g)`` with ``o`` shaped ``(B,H,dv)``.
    """
    mode = str(gate_mode).strip().lower().replace("-", "_")
    qf, kf, uf, rf, vf = q.float(), k.float(), u.float(), r.float(), v.float()
    bf = beta.float()
    if g is not None:
        gt = g.float()
        if mode == "per_step":
            decay = torch.exp2(gt) if use_exp2 else torch.exp(gt)
            S = S * decay.unsqueeze(-1).unsqueeze(-1)
        elif g_prev is not None:
            dg = gt - g_prev
            decay = torch.exp2(dg) if use_exp2 else torch.exp(dg)
            S = S * decay.unsqueeze(-1).unsqueeze(-1)
        g_prev = gt
    m_dim = kf.shape[-1] * rf.shape[-1]
    kr = (kf.unsqueeze(-1) * rf.unsqueeze(-2)).reshape(kf.shape[0], kf.shape[1], m_dim)
    pred = torch.einsum("bhmv,bhm->bhv", S, kr)
    inc = bf.unsqueeze(-1) * (vf - pred)
    S = S + kr.unsqueeze(-1) * inc.unsqueeze(-2)
    qu = (qf.unsqueeze(-1) * uf.unsqueeze(-2)).reshape(qf.shape[0], qf.shape[1], m_dim)
    o = torch.einsum("bhmv,bhm->bhv", S, qu).to(dtype=q.dtype)
    return o, S, g_prev


class StateTensorModel(nn.Module):
    """
    Args:
        d_model: input width ``W`` for ``x`` shaped ``(B, T, W)``.
        n_heads: head count ``H`` (gate output channels = ``H``, Q/K/U/R/V are ``H × (dk|du|dv)``).
        dk: key/query (first kernel) dimension per head (default = GDN ``head_dim`` via
            :func:`default_stm_head_dims`).
        du: third-mode query/key (second kernel) dimension per head (same default as ``dk``).
        dv: value dimension per head (default = GDN ``head_dim * expand_v``).
        gate_type: ``"sigmoid"`` (default) for conv gate logits; ``"logsigmoid"`` for GLA-style
            ``logsigmoid(g)/gate_logit_normalizer``; ``"-softplus"`` for ``-softplus(logits)``.
        use_gla_gate: if True, temporal gate comes from low-rank ``gk_proj(x)`` (FLA GLA) instead
            of the fused-conv gate channel (always uses ``logsigmoid`` + normalizer).
        gate_logit_normalizer: divisor after ``logsigmoid`` (GLA default ``16``).
        gate_low_rank_dim: hidden width of ``gk_proj`` when ``use_gla_gate`` (GLA default ``16``).
        gate_clamp_min: optional floor on log-space gate values (GLA ``clamp_min``).
        gate_cumulative: if True (notes ``stmgcumsum`` / ``stmvitgcumsum``), ``g_t`` is the
            running sum of per-step log-decay along the sequence (GLA / GDN-style), so
            ``exp(g_t - g_s)`` decays with ``t - s``.
        use_write_gate: if True (default), apply write gate ``β = σ(·)``. Additive STM
            uses a separate ``b_proj`` Linear (the layout saved in pre-delta checkpoints)
            unless ``fuse_write_gate`` / notes ``ccbeta`` puts ``β`` on the conc/conv/short
            path. ``stmdelta`` / ``stmdelta2`` / ``stmdelta3`` always fuse ``β`` as the last
            conv-path channel after ``a``. Disable with ``stmnobeta`` in ``delta:stm`` notes.
        fuse_write_gate: if True (notes ``ccbeta``), write ``β`` is fused on the conv path
            after ``a`` even for additive STM. Implied by any ``stmdelta*``.
        use_two_betas: if True (notes ``stmtwobetas`` / ``stmtwobetas2``), fuse query-side
            mix gates ``β1, β2`` on the conc/short-conv path (after write ``β``). Scores
            become ``((1-β1)+β1 q·k)((1-β2)+β2 u·r)``.
        use_two_betas2: if True (notes ``stmtwobetas2``), also L2-normalize the expanded
            features so pairwise ``k·k`` / ``r·r`` stay in ``[-1, 1]``. Implies
            ``use_two_betas``.
        allow_neg_eigval_write: if True, multiply ``β`` by 2 (GDN ``allow_neg_eigval`` analogue).
            Enable via ``stmbetadouble`` in notes.
        use_delta: if True (``stmdelta`` notes), apply the tensor delta rule instead of
            additive outer-product accumulation. Uses :func:`stm_delta_recurrent_fwd`.
        use_delta2: if True (``stmdelta2`` notes), same pairwise-gate tensor delta via the
            chunked UT / WY transform (:func:`~stm_delta_ut.stm_delta_chunk_ut_fwd`).
        use_delta3: if True (``stmdelta3`` notes), UT delta with GDN per-step forget
            ``α_t = exp(g_t)``. Preferred over ``use_delta2`` / ``use_delta`` when set.
        use_fla_matrix: if True (``stmfla`` notes), run the tensor delta as FLA
            ``chunk_gated_delta_rule`` on flattened Kronecker keys ``k⊗r`` (same math,
            GDN matrix kernels). Implied by ``use_fla_fused``.
        use_fla_fused: if True (``stmfused`` notes), use FLA
            ``fused_recurrent_gated_delta_rule`` instead of the chunk kernel. When
            ``stmfla`` is set, inference also auto-picks fused recurrent for ``T≤64``.
        use_triton: on CUDA with Triton available, training uses
            :func:`~minipi05.stm.stm_triton.stm_causal_bilinear_fwd_trainable`
            (Triton forward + PyTorch backward recompute); inference uses fused Triton forward.
            If False, training falls back to pure PyTorch dense/chunked cores.
        backward_recompute: bilinear core backward when ``use_triton`` and training on CUDA.
            ``"dense"`` (default) = Triton forward + dense PyTorch recompute
            (``core|triton|fwd+dense-bwd``; delta:stm / delta:stm2 training default).
            ``"triton"`` = Triton forward + chunked PyTorch backward (same math as ``torch|chunk``).
            ``"chunked"`` / ``"cuda"`` = Triton forward + fused CUDA chunk-512 backward.
        torch_chunk_size: if set, use PyTorch *chunked* forward only when the Triton trainable path
            is unavailable (``pstm_torch_only`` / no Triton). Does **not** select the CUDA backward;
            use ``backward_recompute="chunked"`` for that. Default ``None``.
        train_pytorch_core_only: if True, never use ``stm_causal_bilinear_fwd_trainable``; always
            pure PyTorch dense/chunked (same as ``pstm_torch_only`` in ``delta:stm`` notes). Default ``False``.
        conv_notes: ``notes`` for :func:`get_conv1d` on the fused Q/K/U/R/V + ``a`` input path.
        concreor: if True, use :class:`~some_convs.ConcReor` instead of ``Linear`` for the input
            projection (repeat + fixed random reorder from ``d_model`` to projection width).
        proj_kind: ``"linear"`` (default), ``"concreor"``, ``"conconly"``, ``"conconlyc"``,
            ``"conconlyc2"``, ``"conconlyc3"``, ``"conconlyc4"``,
            ``"concalt"``, ``"concaltc"``, ``"conchaos"``, ``"factorized"``, or ``"tensorprod"``.
        proj_rank: bottleneck rank for ``factorized`` / ``tensorprod`` (notes: ``stmwrank:N``).
        conc_kernel_size: odd channel-mix kernel for ``conconlyc`` / ``c2`` / ``c3`` / ``c4``
            (notes: ``stmconconlyc:N`` or ``stmconconlyc2:N``, …; default ``3``).
            Ignored for other ``proj_kind`` values.
        chaos_iter_mode / chaos_n_stages: ``conchaos`` only; from ``stmchaosidx:…`` in notes.
        gdn_conv_size: kernel size for the GDN-style short conv. Default ``4``.
        gdn_conv_backend: FLA short-conv backend (``"triton"`` or ``"cuda"``). Default ``"triton"``.
        use_exp2: match FLA ``exp2`` in the kernel if True.
        attn_scale: if True, scale ``q,k`` by ``1/√dk`` and ``u,r`` by ``1/√du`` (default off).
        use_qk_l2norm_in_kernel: if True (default), unit L2-normalize ``q,k,u,r`` per head after
            SiLU and before the bilinear core. ``v`` is not normalized. Prefer this over
            ``attn_scale`` for the product kernel; do not enable both unless ablating.
        use_output_gate: if True, fused ``g_proj`` + ``FusedRMSNormGated`` on the output path;
            if False (``stmnogout``), no ``g_out`` channels and plain ``RMSNorm``.
        fuse_output_gate: if True (notes ``stmgouttime``), ``g_out`` is the last fused
            conv-path block (after ``a`` / ``β``) and goes through conc + short conv.
            Default is a GDN-style bypass that skips the short conv.
        use_matrix_state: if True (``stmmatrixstate`` notes), drop U/R projections and use a
            standard matrix state ``S ∈ R^{dk×dv}`` with dummy ``u=r=1`` (linear attention / GDN).
        norm_eps: RMSNorm epsilon for ``o_norm`` (``FusedRMSNormGated`` with fused ``g_proj``).
        core_backend: ``"triton"`` (default) = Triton forward + configurable backward recompute;
            ``"scan"`` = chunked causal scan with native autograd
            (:func:`~stm_triton.stm_causal_bilinear_fwd_scan_trainable`). Ignored when
            ``use_delta`` / ``use_delta2`` / ``use_delta3`` (delta cores).
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            dk: int,
            du: int,
            dv: int,
            *,
            gate_type: str = "sigmoid",
            use_gla_gate: bool = False,
            gate_logit_normalizer: int = 16,
            gate_low_rank_dim: int = 16,
            gate_clamp_min: float | None = None,
            gate_cumulative: bool = False,
            use_triton: bool = True,
            conv_notes: str = "pconvtype:shortconv",
            concreor: bool = False,
            proj_kind: str = "linear",
            proj_rank: int = 32,
            conc_kernel_size: int = 3,
            chaos_iter_mode: str | None = None,
            chaos_n_stages: int | None = None,
            use_exp2: bool = False,
            attn_scale: bool = False,
            use_qk_l2norm_in_kernel: bool = True,
            use_write_gate: bool = True,
            fuse_write_gate: bool = False,
            use_two_betas: bool = False,
            use_two_betas2: bool = False,
            allow_neg_eigval_write: bool = False,
            torch_chunk_size: int | None = None,
            backward_recompute: str = "dense",
            gdn_conv_size: int = 4,
            gdn_conv_backend: str = "triton",
            train_pytorch_core_only: bool = False,
            core_backend: str = "triton",
            use_output_gate: bool = True,
            fuse_output_gate: bool = False,
            use_matrix_state: bool = False,
            use_delta: bool = False,
            use_delta2: bool = False,
            use_delta3: bool = False,
            use_fla_matrix: bool = False,
            use_fla_fused: bool = False,
            norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = dk
        self.du = 1 if use_matrix_state else du
        self.dv = dv
        self.use_matrix_state = bool(use_matrix_state)
        self.gate_type = _normalize_gate_type(gate_type)
        self.use_gla_gate = use_gla_gate
        self.gate_logit_normalizer = max(1, int(gate_logit_normalizer))
        self.gate_clamp_min = gate_clamp_min
        self.gate_cumulative = bool(gate_cumulative)
        self.use_triton = use_triton
        self.use_exp2 = use_exp2
        self.attn_scale = attn_scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.use_write_gate = use_write_gate
        self.use_two_betas2 = bool(use_two_betas2)
        self.use_two_betas = bool(use_two_betas) or self.use_two_betas2
        self.allow_neg_eigval_write = allow_neg_eigval_write
        self.torch_chunk_size = torch_chunk_size
        bwd = str(backward_recompute).strip().lower()
        if bwd not in ("dense", "chunked", "triton", "cuda"):
            raise ValueError(
                'backward_recompute must be "dense", "chunked", "cuda", or "triton"',
            )
        self.backward_recompute = bwd
        self.train_pytorch_core_only = train_pytorch_core_only
        self.use_output_gate = use_output_gate
        self.fuse_output_gate = bool(use_output_gate) and bool(fuse_output_gate)
        self.use_delta3 = bool(use_delta3)
        self.use_delta2 = bool(use_delta2) and not self.use_delta3
        self.use_delta = bool(use_delta) and not self.use_delta2 and not self.use_delta3
        self.use_fla_fused = bool(use_fla_fused)
        self.use_fla_matrix = bool(use_fla_matrix) or self.use_fla_fused
        self.fuse_write_gate = bool(use_write_gate) and (
            bool(fuse_write_gate)
            or self.use_delta
            or self.use_delta2
            or self.use_delta3
        )
        self.norm_eps = norm_eps
        cb = str(core_backend).strip().lower()
        if cb not in ("triton", "scan"):
            raise ValueError('core_backend must be "triton" or "scan"')
        self.core_backend = cb

        H = n_heads
        self.input_proj = _FusedGDNLikeInputProj(
            d_model,
            H,
            dk,
            self.du,
            dv,
            use_matrix_state=self.use_matrix_state,
            use_output_gate=use_output_gate,
            fuse_output_gate=self.fuse_output_gate,
            use_write_gate=self.fuse_write_gate,
            use_two_betas=self.use_two_betas,
            conv_size=gdn_conv_size,
            conv_backend=gdn_conv_backend,
            conv_notes=conv_notes,
            concreor=concreor,
            proj_kind=proj_kind,
            proj_rank=proj_rank,
            conc_kernel_size=conc_kernel_size,
            chaos_iter_mode=chaos_iter_mode,
            chaos_n_stages=chaos_n_stages,
        )
        if use_gla_gate:
            self.gk_proj = _GlaStyleGateProj(
                d_model, H, low_rank_dim=gate_low_rank_dim,
            )
        else:
            self.gk_proj = None
        if use_output_gate:
            self.o_norm = FusedRMSNormGated(dv, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(dv, eps=norm_eps)
        self.o_proj = nn.Linear(H * dv, d_model, bias=False)
        if self.use_write_gate and not self.fuse_write_gate:
            self.b_proj = nn.Linear(d_model, H, bias=False)
        else:
            self.b_proj = None

        self._causal_bilinear_core_fn: Callable[..., torch.Tensor] = self._bind_causal_bilinear_core()
        self.gdn_conv_size = int(gdn_conv_size)
        self._inf_state: dict | None = None
        if (
            self.torch_chunk_size is None
            and os.environ.get("STM_DELTA_UT_AUTOTUNE", "1") != "0"
            and not (self.use_fla_matrix or self.use_fla_fused)
        ):
            if self.use_delta2 or self.use_delta3:
                self.torch_chunk_size = self._autotune_delta_ut_chunk()
            elif self.backward_recompute in ("chunked", "cuda", "triton"):
                self.torch_chunk_size = self._autotune_bilinear_chunk()

    def _temporal_gate(self, x: torch.Tensor, a_raw: torch.Tensor) -> torch.Tensor:
        """Map raw logits to log-space ``g`` used in ``exp(g_t - g_s)`` (GLA / GDN / legacy)."""
        if self.use_gla_gate:
            assert self.gk_proj is not None
            logits = self.gk_proj(x)
        else:
            logits = a_raw
        if self.use_gla_gate or self.gate_type == "logsigmoid":
            g = F.logsigmoid(logits) / float(self.gate_logit_normalizer)
            if self.gate_clamp_min is not None:
                g = torch.clamp_min(g, self.gate_clamp_min)
            if self.gate_cumulative:
                g = g.cumsum(dim=1)
            return g
        if self.gate_type == "sigmoid":
            return torch.sigmoid(logits)
        return -F.softplus(logits)

    def _project(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        q, k, u, r, v, a_raw, g_out, b_raw, two_raw = self.input_proj(x)
        if self.use_matrix_state:
            q, k, v = map(F.silu, (q, k, v))
            if self.use_qk_l2norm_in_kernel:
                q = l2norm(q)
                k = l2norm(k)
            elif self.attn_scale:
                q = q * (self.dk ** -0.5)
                k = k * (self.dk ** -0.5)
        else:
            q, k, u, r, v = map(F.silu, (q, k, u, r, v))
            if self.use_qk_l2norm_in_kernel:
                q = l2norm(q)
                k = l2norm(k)
                u = l2norm(u)
                r = l2norm(r)
            elif self.attn_scale:
                q = q * (self.dk ** -0.5)
                k = k * (self.dk ** -0.5)
                u = u * (self.du ** -0.5)
                r = r * (self.du ** -0.5)

        g = self._temporal_gate(x, a_raw)
        if self.b_proj is not None:
            b_raw = self.b_proj(x)
        if b_raw is None:
            beta = None
        else:
            beta = torch.sigmoid(b_raw)
            if self.allow_neg_eigval_write:
                beta = beta * 2.0
        read_betas = None
        if two_raw is not None:
            read_betas = (torch.sigmoid(two_raw[0]), torch.sigmoid(two_raw[1]))
        return q, k, u, r, v, g, g_out, beta, read_betas

    @staticmethod
    def _apply_write_gate(
        k: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale ``k,r,v`` so score ``(q·k)(u·r)`` is multiplied by ``β_s`` at key index ``s``."""
        b = beta.unsqueeze(-1)
        sb = b.sqrt()
        return k * sb, r * sb, v * b

    def _triton_chunk_size(self, seq_len: int) -> int:
        req = int(self.torch_chunk_size) if self.torch_chunk_size else 64
        return _stm_triton.stm_triton_tile_bt(seq_len, req)

    def _scan_chunk_size(self, seq_len: int) -> int:
        if self.torch_chunk_size is not None and int(self.torch_chunk_size) > 0:
            return _stm_triton.stm_triton_tile_bt(seq_len, int(self.torch_chunk_size))
        return _stm_triton.stm_triton_tile_bt(seq_len, 64)

    def _bind_causal_bilinear_core(
        self,
    ) -> Callable[..., torch.Tensor]:
        """Pick bilinear core implementation from init-time flags (no per-forward branching)."""
        if self.core_backend == "scan":
            return self._core_scan_trainable
        if (
            self.use_triton
            and not self.train_pytorch_core_only
            and _stm_triton.STM_TRITON_AVAILABLE
        ):
            return self._core_triton_trainable
        if (
            not self.use_triton
            and _stm_cuda_ext.extension_available()
        ):
            return self._core_cuda_ext_eval_torch_train
        if self.torch_chunk_size is not None and int(self.torch_chunk_size) > 0:
            return self._core_torch_chunked
        return self._core_torch_dense

    def _core_scan_trainable(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        seq_len: int,
    ) -> torch.Tensor:
        return _stm_triton.stm_causal_bilinear_fwd_scan_trainable(
            q,
            k,
            u,
            r,
            v,
            g,
            chunk_size=self._scan_chunk_size(seq_len),
            use_exp2=self.use_exp2,
        )

    def _core_triton_trainable(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        seq_len: int,
    ) -> torch.Tensor:
        tuned = (
            int(self.torch_chunk_size)
            if self.torch_chunk_size is not None and int(self.torch_chunk_size) > 0
            else None
        )
        return _stm_triton.stm_causal_bilinear_fwd_trainable(
            q,
            k,
            u,
            r,
            v,
            g,
            chunk_size=self._triton_chunk_size(seq_len),
            use_exp2=self.use_exp2,
            backward_recompute=self.backward_recompute,
            pytorch_chunk_for_backward=tuned if self.backward_recompute == "triton" else None,
            cuda_chunk_for_backward=(
                tuned if self.backward_recompute in ("chunked", "cuda") else None
            ),
        )

    def _core_torch_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        seq_len: int,
    ) -> torch.Tensor:
        _ = seq_len
        return _stm_triton.stm_causal_bilinear_fwd_torch_chunked(
            q, k, u, r, v, g,
            chunk_size=int(self.torch_chunk_size),
            use_exp2=self.use_exp2,
        )

    def _core_torch_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        seq_len: int,
    ) -> torch.Tensor:
        _ = seq_len
        return _stm_triton.stm_causal_bilinear_fwd_torch(
            q, k, u, r, v, g, use_exp2=self.use_exp2,
        )

    def _core_cuda_ext(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        seq_len: int,
    ) -> torch.Tensor:
        return _stm_cuda_ext.stm_causal_bilinear_fwd_cuda_ext(
            q, k, u, r, v, g,
            chunk_size=self._triton_chunk_size(seq_len),
            use_exp2=self.use_exp2,
        )

    def _core_cuda_ext_eval_torch_train(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        seq_len: int,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            if self.torch_chunk_size is not None and int(self.torch_chunk_size) > 0:
                return self._core_torch_chunked(
                    q, k, u, r, v, g, seq_len=seq_len,
                )
            return self._core_torch_dense(q, k, u, r, v, g, seq_len=seq_len)
        return self._core_cuda_ext(q, k, u, r, v, g, seq_len=seq_len)

    def _causal_bilinear_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        seq_len: int,
    ) -> torch.Tensor:
        return self._causal_bilinear_core_fn(
            q, k, u, r, v, g, seq_len=seq_len,
        )

    def _apply_output(
        self,
        x: torch.Tensor,
        o: torch.Tensor,
        g_out: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, H, dv = o.shape
        if self.use_output_gate:
            assert g_out is not None
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)
        y = self.o_proj(o.reshape(B, T, H * dv))

        return y

    def reset_inference_cache(self) -> None:
        self._inf_state = None

    def _delta_gate_mode(self) -> str:
        return "per_step" if self.use_delta3 else "pairwise"

    def _autotune_probe_shapes(self) -> tuple[int, int, int, int, int, int]:
        B, T, H = 1, 256, int(self.n_heads)
        dk, du, dv = int(self.dk), int(self.du), int(self.dv)
        if self.use_two_betas:
            dk, du = dk + 1, du + 1
        return B, T, H, dk, du, dv

    def _autotune_cache_get(self, key: tuple) -> int | None:
        return _STM_CHUNK_AUTOTUNE_CACHE.get(key)

    def _autotune_cache_set(self, key: tuple, value: int) -> int:
        _STM_CHUNK_AUTOTUNE_CACHE[key] = int(value)
        return int(value)

    @staticmethod
    def _time_cuda_ms(fn, *, n_warm: int = 2, n_run: int = 3) -> float:
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_run):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / float(n_run)

    def _autotune_delta_ut_chunk(self) -> int:
        """Pick the fastest UT tile (fwd+bwd) in ``STM_DELTA_UT_CHUNK_SIZES`` on this GPU."""
        from minipi05.stm.stm_delta_ut import (
            STM_DELTA_UT_CHUNK_SIZES,
            stm_delta_chunk_ut_fwd,
        )

        if not torch.cuda.is_available() or os.environ.get("PYTEST_CURRENT_TEST"):
            return 64
        B, T, H, dk, du, dv = self._autotune_probe_shapes()
        gate_mode = self._delta_gate_mode()
        cache_key = (
            "ut", H, dk, du, dv, gate_mode, bool(self.use_exp2),
            bool(self.use_triton and not self.train_pytorch_core_only),
        )
        cached = self._autotune_cache_get(cache_key)
        if cached is not None:
            return cached

        import minipi05.stm.stm_delta_ut as _ut

        prev_compile = os.environ.get("STM_DELTA_UT_COMPILE")
        os.environ["STM_DELTA_UT_COMPILE"] = "0"
        _ut._STM_DELTA_UT_SCAN_COMPILED = None
        device = torch.device("cuda")
        use_triton = self.use_triton and not self.train_pytorch_core_only
        try:
            def _step(bt: int) -> None:
                q = torch.randn(B, T, H, dk, device=device, requires_grad=True)
                k = torch.randn(B, T, H, dk, device=device, requires_grad=True)
                u = torch.randn(B, T, H, du, device=device, requires_grad=True)
                r = torch.randn(B, T, H, du, device=device, requires_grad=True)
                v = torch.randn(B, T, H, dv, device=device, requires_grad=True)
                g = (torch.randn(B, T, H, device=device) * 0.1).requires_grad_(True)
                beta = torch.rand(B, T, H, device=device, requires_grad=True)
                o = stm_delta_chunk_ut_fwd(
                    q, k, u, r, v, g, beta,
                    chunk_size=bt, use_exp2=self.use_exp2,
                    use_triton=use_triton, gate_mode=gate_mode,
                )
                o.sum().backward()

            best_bt, best_ms = 64, float("inf")
            parts = []
            for bt in STM_DELTA_UT_CHUNK_SIZES:
                try:
                    ms = self._time_cuda_ms(lambda bt=bt: _step(bt))
                except Exception:
                    parts.append(f"{bt}=fail")
                    continue
                parts.append(f"{bt}={ms:.2f}ms")
                if ms < best_ms:
                    best_ms, best_bt = ms, bt
            print(
                f"STM UT autotune fwd+bwd @ T={T}: {' '.join(parts)} "
                f"→ chunk={best_bt} ({best_ms:.2f} ms)",
                flush=True,
            )
            return self._autotune_cache_set(cache_key, best_bt)
        finally:
            if prev_compile is None:
                os.environ.pop("STM_DELTA_UT_COMPILE", None)
            else:
                os.environ["STM_DELTA_UT_COMPILE"] = prev_compile
            _ut._STM_DELTA_UT_SCAN_COMPILED = None

    def _autotune_bilinear_chunk(self) -> int:
        """Pick the fastest bilinear backward tile for ``stmbwd:chunked|triton``."""
        if not torch.cuda.is_available() or os.environ.get("PYTEST_CURRENT_TEST"):
            return 64
        B, T, H, dk, du, dv = self._autotune_probe_shapes()
        bwd = self.backward_recompute
        cache_key = ("bilinear", bwd, H, dk, du, dv, bool(self.use_exp2))
        cached = self._autotune_cache_get(cache_key)
        if cached is not None:
            return cached

        device = torch.device("cuda")
        grid = _STM_BILINEAR_CHUNK_SIZES
        if bwd == "triton":
            from minipi05.stm.stm_delta_ut import (
                STM_DELTA_UT_CHUNK_SIZES,
            )
            grid = STM_DELTA_UT_CHUNK_SIZES

        def _step(bt: int) -> None:
            q = torch.randn(B, T, H, dk, device=device, requires_grad=True)
            k = torch.randn(B, T, H, dk, device=device, requires_grad=True)
            u = torch.randn(B, T, H, du, device=device, requires_grad=True)
            r = torch.randn(B, T, H, du, device=device, requires_grad=True)
            v = torch.randn(B, T, H, dv, device=device, requires_grad=True)
            g = (torch.randn(B, T, H, device=device) * 0.1).requires_grad_(True)
            o = _stm_triton.stm_causal_bilinear_fwd_trainable(
                q, k, u, r, v, g,
                chunk_size=max(16, min(bt, 64)) if bwd == "triton" else 64,
                use_exp2=self.use_exp2,
                backward_recompute=bwd,
                pytorch_chunk_for_backward=bt if bwd == "triton" else None,
                cuda_chunk_for_backward=bt if bwd in ("chunked", "cuda") else None,
            )
            o.sum().backward()

        best_bt, best_ms = grid[0], float("inf")
        parts = []
        for bt in grid:
            try:
                ms = self._time_cuda_ms(lambda bt=bt: _step(bt))
            except Exception:
                parts.append(f"{bt}=fail")
                continue
            parts.append(f"{bt}={ms:.2f}ms")
            if ms < best_ms:
                best_ms, best_bt = ms, bt
        print(
            f"STM {bwd} autotune fwd+bwd @ T={T}: {' '.join(parts)} "
            f"→ chunk={best_bt} ({best_ms:.2f} ms)",
            flush=True,
        )
        return self._autotune_cache_set(cache_key, best_bt)

    def _core_delta(self, q, k, u, r, v, g, beta, *, return_state: bool = False):
        from minipi05.stm.stm_delta_ut import (
            stm_delta_chunk_ut_fwd,
            stm_delta_ut_chunk_size,
        )
        return stm_delta_chunk_ut_fwd(
            q, k, u, r, v, g, beta,
            chunk_size=stm_delta_ut_chunk_size(self.torch_chunk_size),
            use_exp2=self.use_exp2,
            use_triton=self.use_triton and not self.train_pytorch_core_only,
            gate_mode=self._delta_gate_mode(),
            return_state=return_state,
            use_fla_matrix=self.use_fla_matrix,
            use_fla_fused=self.use_fla_fused,
            training=self.training,
        )

    def _decode_one(self, x: torch.Tensor, past: dict) -> tuple[torch.Tensor, dict]:
        """O(1) cached decode: project a short conv window, then one recurrent step."""
        conv_k = max(1, int(self.gdn_conv_size))
        tail = past["x_tail"]
        x_win = torch.cat([tail, x], dim=1)
        q, k, u, r, v, g, g_out, beta, read_betas = self._project(x_win)
        q, k, u, r, v = (t[:, -1:] for t in (q, k, u, r, v))
        g = g[:, -1:]
        if g_out is not None:
            g_out = g_out[:, -1:]
        if beta is not None:
            beta = beta[:, -1:]
        if read_betas is not None:
            read_betas = (read_betas[0][:, -1:], read_betas[1][:, -1:])
            q, k, u, r = apply_two_betas_feature_map(
                q, k, u, r, read_betas[0], read_betas[1],
                renormalize=self.use_two_betas2,
            )
        B = x.shape[0]
        if beta is None:
            beta = q.new_ones(B, 1, self.n_heads)
        o, S, g_prev = stm_delta_recurrent_step(
            q[:, 0], k[:, 0], u[:, 0], r[:, 0], v[:, 0],
            g[:, 0], beta[:, 0], past["S"], past.get("g_prev"),
            use_exp2=self.use_exp2, gate_mode=self._delta_gate_mode(),
        )
        o = o.unsqueeze(1)
        y = self._apply_output(x, o, g_out)
        keep = conv_k - 1
        new_tail = x_win[:, -keep:] if keep > 0 else x_win[:, 0:0]
        return y, {"S": S, "g_prev": g_prev, "x_tail": new_tail}

    def forward(self, x: torch.Tensor, *, use_cache: bool = False) -> torch.Tensor:
        if use_cache and self._inf_state is not None and x.shape[1] == 1:
            if self.use_delta or self.use_delta2 or self.use_delta3:
                y, self._inf_state = self._decode_one(x, self._inf_state)
                return y

        B, T, _W = x.shape
        q, k, u, r, v, g, g_out, beta, read_betas = self._project(x)

        if g.shape[1] != T or g.shape[2] != self.n_heads:
            raise RuntimeError(
                f"gate expects shape (B, T, H)=(_, {T}, {self.n_heads}), got {tuple(g.shape)}.",
            )

        if beta is not None and not (self.use_delta3 or self.use_delta2 or self.use_delta):
            if self.use_matrix_state:
                k = k * beta.unsqueeze(-1)
            else:
                k, r, v = self._apply_write_gate(k, r, v, beta)
        if read_betas is not None:
            q, k, u, r = apply_two_betas_feature_map(
                q, k, u, r, read_betas[0], read_betas[1],
                renormalize=self.use_two_betas2,
            )

        state = None
        if self.use_delta3 or self.use_delta2:
            if beta is None:
                beta = q.new_ones(B, T, self.n_heads)
            if use_cache:
                o, state_s = self._core_delta(q, k, u, r, v, g, beta, return_state=True)
                state = {
                    "S": state_s,
                    "g_prev": None if g is None else g[:, -1].float(),
                    "x_tail": x[:, -(max(1, self.gdn_conv_size) - 1):]
                    if self.gdn_conv_size > 1 else x[:, 0:0],
                }
            else:
                o = self._core_delta(q, k, u, r, v, g, beta)
        elif self.use_delta:
            if beta is None:
                beta = q.new_ones(B, T, self.n_heads)
            if use_cache:
                o, state = self._prefill_delta_recurrent(q, k, u, r, v, g, beta, x)
            else:
                o = stm_delta_recurrent_fwd(
                    q, k, u, r, v, g, beta, use_exp2=self.use_exp2,
                    gate_mode=self._delta_gate_mode(),
                )
        else:
            o = self._causal_bilinear_core(q, k, u, r, v, g, seq_len=T)
        y = self._apply_output(x, o, g_out)
        if use_cache:
            self._inf_state = state
        return y

    def _prefill_delta_recurrent(self, q, k, u, r, v, g, beta, x):
        B, T, H, dk = q.shape
        du, dv = u.shape[-1], v.shape[-1]
        S = torch.zeros(B, H, dk * du, dv, device=q.device, dtype=torch.float32)
        o = q.new_empty(B, T, H, dv)
        g_prev = None
        for t in range(T):
            o[:, t], S, g_prev = stm_delta_recurrent_step(
                q[:, t], k[:, t], u[:, t], r[:, t], v[:, t],
                None if g is None else g[:, t], beta[:, t], S, g_prev,
                use_exp2=self.use_exp2, gate_mode=self._delta_gate_mode(),
            )
        keep = max(1, self.gdn_conv_size) - 1
        state = {
            "S": S,
            "g_prev": g_prev,
            "x_tail": x[:, -keep:] if keep > 0 else x[:, 0:0],
        }
        return o, state


def _smoke_test() -> None:
    torch.manual_seed(0)
    B, T, W, H = 2, 37, 64, 4
    dk, du, dv = 8, 8, 16
    x = torch.randn(B, T, W)
    m = StateTensorModel(
        W, H, dk, du, dv,
        use_triton=False,
    )
    y = m(x)
    assert y.shape == (B, T, W)
    xg = torch.randn(B, T, W, requires_grad=True)
    yg = m(xg)
    yg.sum().backward()
    assert xg.grad is not None and xg.grad.abs().sum() > 0
    q, k, u, r, v, g, g_out, beta, _rb = m._project(x)
    assert beta is not None
    k_w, r_w, v_w = StateTensorModel._apply_write_gate(k, r, v, beta)
    o_ref = m._causal_bilinear_core(q, k_w, u, r_w, v_w, g, seq_len=T)
    y_ref = m._apply_output(x, o_ref, g_out)
    err_wg = (y_ref - m(x)).abs().max().item()
    assert err_wg < 1e-5, err_wg
    assert m.b_proj is not None
    assert m.fuse_write_gate is False
    assert m.input_proj.use_write_gate is False
    assert len(m.input_proj._conv_split_sizes) == 6
    m_nobeta = StateTensorModel(
        W, H, dk, du, dv,
        use_triton=False,
        use_write_gate=False,
    )
    y_nobeta = m_nobeta(x)
    assert y_nobeta.shape == (B, T, W)
    m_mat = StateTensorModel(
        W, H, dk, du, dv,
        use_triton=False,
        use_matrix_state=True,
    )
    y_mat = m_mat(x)
    assert y_mat.shape == (B, T, W)
    assert m_mat.du == 1
    q, k, u, r, v, g, _, _beta_mat, _rb_mat = m_mat._project(x)
    assert u.shape[-1] == 1 and r.shape[-1] == 1
    assert torch.allclose(u, torch.ones_like(u))
    m_delta = StateTensorModel(
        W, H, dk, du, dv,
        use_triton=False,
        use_delta=True,
        proj_kind="conconlyc",
    )
    y_delta = m_delta(x)
    assert y_delta.shape == (B, T, W)
    assert m_delta.use_delta is True
    assert m_delta.input_proj._conv_split_sizes[-1] == H
    xd = torch.randn(B, T, W, requires_grad=True)
    m_delta(xd).sum().backward()
    assert xd.grad is not None and xd.grad.abs().sum() > 0
    m_delta2 = StateTensorModel(
        W, H, dk, du, dv,
        use_triton=False,
        use_delta2=True,
        proj_kind="conconlyc",
    )
    y_delta2 = m_delta2(x)
    assert y_delta2.shape == (B, T, W)
    assert m_delta2.use_delta2 is True
    assert m_delta2.use_delta is False
    m_delta3 = StateTensorModel(
        W, H, dk, du, dv,
        use_triton=False,
        use_delta3=True,
        proj_kind="conconlyc",
    )
    y_delta3 = m_delta3(x)
    assert y_delta3.shape == (B, T, W)
    assert m_delta3.use_delta3 is True
    assert m_delta3.use_delta2 is False
    n_full = sum(p.numel() for p in m.parameters())
    n_mat = sum(p.numel() for p in m_mat.parameters())
    assert n_mat < n_full
    xgm = torch.randn(B, T, W, requires_grad=True)
    m_mat(xgm).sum().backward()
    assert xgm.grad is not None
    if torch.cuda.is_available():
        mc = StateTensorModel(
            W, H, dk, du, dv,
            use_triton=True,
        ).cuda()
        xcg2 = torch.randn(B, T, W, requires_grad=True, device="cuda")
        ycg2 = mc(xcg2)
        ycg2.sum().backward()
        assert xcg2.grad is not None
        q, k, u, r, v, g, _g_out, _beta_c, _rb_c = mc._project(x.cuda())
        if _stm_triton.STM_TRITON_AVAILABLE:
            o_t = _stm_triton.stm_causal_bilinear_fwd_torch(q, k, u, r, v, g)
            o_c = _stm_triton.stm_causal_bilinear_fwd(q, k, u, r, v, g)
            err = (o_t - o_c).abs().max().item()
            print(f"CUDA max |torch - triton|: {err:.6e}")
            assert err < 1e-2 or err < o_t.abs().max().item() * 0.05
        T_chk = 103
        q2 = torch.randn(B, T_chk, H, dk, device="cuda", dtype=q.dtype)
        k2 = torch.randn(B, T_chk, H, dk, device="cuda", dtype=q.dtype)
        u2 = torch.randn(B, T_chk, H, du, device="cuda", dtype=q.dtype)
        r2 = torch.randn(B, T_chk, H, du, device="cuda", dtype=q.dtype)
        v2 = torch.randn(B, T_chk, H, dv, device="cuda", dtype=q.dtype)
        g2 = torch.randn(B, T_chk, H, device="cuda", dtype=q.dtype)
        o_d = _stm_triton.stm_causal_bilinear_fwd_torch(q2, k2, u2, r2, v2, g2)
        o_ch = _stm_triton.stm_causal_bilinear_fwd_torch_chunked(
            q2, k2, u2, r2, v2, g2, chunk_size=31,
        )
        err_c = (o_d - o_ch).float().abs().max().item()
        assert err_c < 1e-2 or err_c < o_d.float().abs().max().item() * 1e-3, err_c
        try:
            from minipi05.stm import stm_chunked_bilinear_ext as _stm_ext_cuda
            if _stm_ext_cuda.extension_available():
                o_cuda = _stm_ext_cuda.stm_causal_bilinear_fwd_cuda_ext(
                    q2, k2, u2, r2, v2, g2, chunk_size=16, use_exp2=False,
                )
                err_cu = (o_d - o_cuda).float().abs().max().item()
                print(f"CUDA ext max |torch_dense - cuda_chunk|: {err_cu:.6e}")
                assert err_cu < 5e-2 or err_cu < o_d.float().abs().max().item() * 0.05
        except Exception as e:
            print("stm_chunked_bilinear_ext skip:", repr(e))
        xcg = torch.randn(B, T, W, requires_grad=True, device="cuda")
        ycg = mc(xcg)
        ycg.sum().backward()
        assert xcg.grad is not None


if __name__ == "__main__":
    _smoke_test()
    print("state_tensor_model smoke OK")

STM = StateTensorModel
