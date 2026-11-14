# SPDX-License-Identifier: MIT
"""
Triton forward for State Tensor Model (STM) causal bilinear attention.

Computes (per batch and head) ::
    A[t,s] = (q_tÂ·k_s) * (u_tÂ·r_s) * exp(g_t - g_s)   (if USE_G)
    o_t = sum_{s<=t} A[t,s] v_s

Layout matches FLA ``chunk_fwd_kernel_o`` conventions: ``q,k`` are ``(B,T,H,K)``,
``u,r`` are ``(B,T,H,DU)``, ``v,o`` are ``(B,T,H,V)``, optional ``g`` is ``(B,T,H)``.

The pure-PyTorch :func:`stm_causal_bilinear_fwd_torch_chunked` uses the same math with autograd-friendly tiles.
:class:`stm_causal_bilinear_fwd_trainable` runs **Triton forward on CUDA during training** (optional
:func:`stm_causal_bilinear_fwd_outer_chunk` Python loop over **512**-step segments, inner tiles 64).
Gradients via
PyTorch recompute (**dense**, default), chunked PyTorch (**triton**), or JIT CUDA (**chunked**).
``backward_recompute="chunked"`` (alias ``cuda``) uses the JIT CUDA extension with outer chunk **512**
(fused, low-atomic layout). ``backward_recompute="triton"`` uses chunked PyTorch backward tiles.
Inference can use either Triton (when available) or a **JIT CUDA** extension (:mod:`~minipi05.stm.stm_chunked_bilinear_ext`, fp32 forward)
via :class:`~minipi05.stm.state_tensor_model.StateTensorModel`.
The Triton launcher is loaded when ``import triton`` and FLA helpers succeed.
"""

from __future__ import annotations

import math

import torch

# Triton 3.x: ``tl.zeros([BT, …])`` needs power-of-two block dims; ``tl.dot`` needs K >= 16.
# The fwd kernel dots ``(BT, BT) @ (BT, BV)``, so ``BT`` must be at least this (masks handle ``T < BT``).
STM_TRITON_MIN_BT = 16
STM_TRITON_MAX_BT = 64


def stm_triton_tile_bt(
        seq_len: int,
        requested_bt: int = 64,
        *,
        max_tile: int = STM_TRITON_MAX_BT,
        min_tile: int = STM_TRITON_MIN_BT,
) -> int:
    """
    Power-of-2 time tile for STM Triton kernels, in ``[min_tile, max_tile]``.

    Picks the largest power of two ``<= min(requested_bt, max_tile)``, then bumps up to
    ``min_tile`` so ``tl.dot`` inner dims are valid. Sequence masking (``o_t < T``) still
    applies when ``T < BT`` (short ``onlytesting`` batches).
    """
    _ = seq_len  # tile size is fixed; masking handles shorter sequences
    cap = min(max(1, int(requested_bt)), max(1, int(max_tile)))
    n = 1
    while (n << 1) <= cap:
        n <<= 1
    floor = max(1, int(min_tile))
    if n < floor:
        n = min(floor, max(1, int(max_tile)))
    return n


def _gate_to_bht(g: torch.Tensor, *, time: int, n_heads: int) -> torch.Tensor:
    """
    FLA-style factors need ``g`` as ``(B, H, T)`` so ``exp(g[...,t] - g[...,s])`` is ``(B, H, T, T)``.

    Accepts either ``(B, T, H)`` (matches conv nets on ``(B, T, W)``) or ``(B, H, T)``.
    """
    if g.ndim != 3:
        raise ValueError(f"gate must be rank-3, got {tuple(g.shape)}")
    b0, d1, d2 = g.shape
    if d1 == time and d2 == n_heads:
        return g.transpose(1, 2)
    if d1 == n_heads and d2 == time:
        return g
    raise ValueError(
        f"gate shape {tuple(g.shape)} not compatible with T={time}, H={n_heads} "
        "(expected (B, T, H) or (B, H, T))"
    )


def stm_causal_bilinear_fwd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    use_exp2: bool = False,
) -> torch.Tensor:
    """Reference ``O = (((QK^T)*(UR^T))*mask*Î“) @ V`` with optional FLA-style gate.

    Tensors use layout **(B, T, H, D)** (time before head), matching :class:`StateTensorModel` and the Triton kernel.
    """
    B, T, H, _K = q.shape
    device = q.device
    dtype = q.dtype

    mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    a = torch.einsum("bthd,bshd->bhts", q, k)
    b = torch.einsum("bthd,bshd->bhts", u, r)
    w = a * b
    w = w.masked_fill(~mask, 0)
    if g is not None:
        gh = _gate_to_bht(g, time=T, n_heads=H)
        if use_exp2:
            gamma = torch.exp2(
                gh[:, :, :, None].to(torch.float32) - gh[:, :, None, :].to(torch.float32)
            ).to(dtype=dtype)
        else:
            gamma = torch.exp(
                gh[:, :, :, None].to(torch.float32) - gh[:, :, None, :].to(torch.float32)
            ).to(dtype=dtype)
        w = w * gamma
    return torch.einsum("bhts,bshv->bthv", w, v)


def stm_causal_bilinear_fwd_torch_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    use_exp2: bool = False,
) -> torch.Tensor:
    """Same recurrence as :func:`stm_causal_bilinear_fwd_torch`, but never materializes a full `(T,T)` slab.

    For each output timestep block `[t_lo, t_hi)` sums key blocks ``[s_lo, s_hi)`` with ``s_hi <= t_hi``
    and applies the causal mask exactly on the intra-block square when blocks align.
    Complexity is still quadratic in sequence length; peak **activation** footprint is ``O(CHUNKÂ²)``
    per einsum coupling instead of ``O(TÂ²)``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    B, T, H, _K = q.shape
    _Vdim = v.shape[-1]
    dtype = q.dtype

    o = torch.zeros(B, T, H, _Vdim, device=q.device, dtype=dtype)

    if g is not None:
        gh = _gate_to_bht(g, time=T, n_heads=H)
    else:
        gh = None

    nt = math.ceil(T / chunk_size)

    def _pair_mask_row_col(
        row_lo: int, n_row: int, col_lo: int, n_col: int,
    ) -> torch.Tensor:
        """(n_row, n_col) causal mask in global timestep indices."""
        ri = torch.arange(row_lo, row_lo + n_row, device=q.device)
        sj = torch.arange(col_lo, col_lo + n_col, device=q.device)
        return ri[:, None] >= sj[None, :]

    for i_tr in range(nt):
        t_lo = i_tr * chunk_size
        t_hi = min(t_lo + chunk_size, T)
        n_t = t_hi - t_lo
        if n_t <= 0:
            break

        q_blk = q[:, t_lo:t_hi, :, :]
        u_blk = u[:, t_lo:t_hi, :, :]

        acc = torch.zeros(B, n_t, H, _Vdim, device=q.device, dtype=dtype)

        for i_tc in range(i_tr + 1):
            s_lo = i_tc * chunk_size
            s_hi = min(s_lo + chunk_size, T)
            n_s = s_hi - s_lo
            if n_s <= 0:
                break

            a = torch.einsum(
                "bthd,bshd->bhts", q_blk, k[:, s_lo:s_hi, :, :]
            ).to(dtype=dtype)
            b_dot = torch.einsum(
                "bthd,bshd->bhts", u_blk, r[:, s_lo:s_hi, :, :]
            ).to(dtype=dtype)

            score = a * b_dot

            mask_bt: torch.Tensor | None = None
            if i_tc == i_tr:
                mask_bt = _pair_mask_row_col(t_lo, n_t, s_lo, n_s)
                mb = mask_bt.view(1, 1, n_t, n_s)
                score = torch.where(mb, score, torch.zeros_like(score))

            if gh is not None:
                g_row = gh[:, :, t_lo:t_hi]
                g_col = gh[:, :, s_lo:s_hi]
                if use_exp2:
                    gx = torch.exp2(
                        g_row[:, :, :, None].to(dtype=torch.float32)
                        - g_col[:, :, None, :].to(dtype=torch.float32)
                    ).to(dtype=dtype)
                else:
                    gx = torch.exp(
                        g_row[:, :, :, None].to(dtype=torch.float32)
                        - g_col[:, :, None, :].to(dtype=torch.float32)
                    ).to(dtype=dtype)
                if mask_bt is None:
                    score = score * gx
                else:
                    mb = mask_bt.view(1, 1, n_t, n_s)
                    score = torch.where(mb, score * gx, torch.zeros_like(score))

            v_blk = v[:, s_lo:s_hi, :, :]
            contrib = torch.einsum("bhts,bshv->bthv", score.to(dtype), v_blk)
            acc += contrib

        o[:, t_lo:t_hi, :, :] += acc

    return o


def stm_causal_bilinear_fwd_scan_trainable(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    use_exp2: bool = False,
    prefer_cuda: bool = True,
) -> torch.Tensor:
    """Chunked causal scan with native autograd (CUDA fused when built, else PyTorch tiles).

    Avoids Triton-forward + dense-recompute backward: one autograd graph through chunked tiles
    or the JIT ``stm_fwd`` / ``stm_bwd`` pair.
    """
    q = q.contiguous()
    k = k.contiguous()
    u = u.contiguous()
    r = r.contiguous()
    v = v.contiguous()
    g = g.contiguous() if g is not None else None
    ch = max(1, int(chunk_size))
    if prefer_cuda and q.is_cuda and q.dtype == torch.float32:
        from minipi05.stm import (
            stm_chunked_bilinear_ext as _cext,
        )
        if _cext.extension_available():
            cuda_ch = _cext.effective_cuda_chunk_size(q.shape[1], ch)
            return _cext.stm_causal_bilinear_cuda_autograd(
                q, k, u, r, v, g, chunk_size=cuda_ch, use_exp2=use_exp2,
            )
    return stm_causal_bilinear_fwd_torch_chunked(
        q, k, u, r, v, g, chunk_size=ch, use_exp2=use_exp2,
    )


def _stm_accumulate_grad_v_chunked_torch(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    g: torch.Tensor | None,
    *,
    chunk_size: int,
    use_exp2: bool,
) -> torch.Tensor:
    """Recompute chunked masked scores and accumulate ``Î”v``: ``Î£_t S[t,s] Â· âˆ‚L/âˆ‚o[t]`` per head/block.

    Matches ``stm_chunk_bwd_grad_v_kernel`` tiling and ``stm_causal_bilinear_fwd_torch_chunked`` score math,
    keeps ``backward_recompute="triton"`` compilation/runtime sane (Î”v avoids fragile Triton subscripts on tiled dots).
    """
    B, T, H, _K = q.shape
    Vdim = grad_o.shape[-1]
    device = grad_o.device
    dtype = grad_o.dtype

    dv = torch.zeros(B, T, H, Vdim, device=device, dtype=dtype)

    if g is not None:
        gh = _gate_to_bht(g, time=T, n_heads=H)
    else:
        gh = None

    nt = math.ceil(T / chunk_size)

    def _pair_mask_row_col(
        row_lo: int, n_row: int, col_lo: int, n_col: int,
    ) -> torch.Tensor:
        ri = torch.arange(row_lo, row_lo + n_row, device=device)
        sj = torch.arange(col_lo, col_lo + n_col, device=device)
        return ri[:, None] >= sj[None, :]

    for i_tr in range(nt):
        t_lo = i_tr * chunk_size
        t_hi = min(t_lo + chunk_size, T)
        n_t = t_hi - t_lo
        if n_t <= 0:
            break

        q_blk = q[:, t_lo:t_hi, :, :]
        u_blk = u[:, t_lo:t_hi, :, :]
        go_blk = grad_o[:, t_lo:t_hi, :, :]

        for i_tc in range(i_tr + 1):
            s_lo = i_tc * chunk_size
            s_hi = min(s_lo + chunk_size, T)
            n_s = s_hi - s_lo
            if n_s <= 0:
                break

            a = torch.einsum(
                "bthd,bshd->bhts", q_blk, k[:, s_lo:s_hi, :, :]
            ).to(dtype=dtype)
            b_dot = torch.einsum(
                "bthd,bshd->bhts", u_blk, r[:, s_lo:s_hi, :, :]
            ).to(dtype=dtype)

            score = a * b_dot

            mask_bt: torch.Tensor | None = None
            if i_tc == i_tr:
                mask_bt = _pair_mask_row_col(t_lo, n_t, s_lo, n_s)
                mb = mask_bt.view(1, 1, n_t, n_s)
                score = torch.where(mb, score, torch.zeros_like(score))

            if gh is not None:
                g_row = gh[:, :, t_lo:t_hi]
                g_col = gh[:, :, s_lo:s_hi]
                if use_exp2:
                    gx = torch.exp2(
                        g_row[:, :, :, None].to(dtype=torch.float32)
                        - g_col[:, :, None, :].to(dtype=torch.float32)
                    ).to(dtype=dtype)
                else:
                    gx = torch.exp(
                        g_row[:, :, :, None].to(dtype=torch.float32)
                        - g_col[:, :, None, :].to(dtype=torch.float32)
                    ).to(dtype=dtype)
                if mask_bt is None:
                    score = score * gx
                else:
                    mb = mask_bt.view(1, 1, n_t, n_s)
                    score = torch.where(mb, score * gx, torch.zeros_like(score))

            contrib = torch.einsum("bhts,bthv->bshv", score.to(dtype), go_blk)
            dv[:, s_lo:s_hi, :, :] += contrib

    return dv


def _stm_accumulate_grad_qkur_chunked_torch(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    *,
    chunk_size: int,
    use_exp2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Chunked ``dq,dk,du,dr,dg`` via score recomputation (matches ``stm_chunk_bwd_qkur_kernel``)."""
    B, T, H, _K = q.shape
    device = q.device
    dtype = q.dtype

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    du = torch.zeros_like(u)
    dr = torch.zeros_like(r)
    dg = torch.zeros(B, T, H, device=device, dtype=dtype) if g is not None else None

    if g is not None:
        gh = _gate_to_bht(g, time=T, n_heads=H)
    else:
        gh = None

    nt = math.ceil(T / chunk_size)
    dg_scale = math.log(2.0) if (g is not None and use_exp2) else 1.0

    def _pair_mask_row_col(
        row_lo: int, n_row: int, col_lo: int, n_col: int,
    ) -> torch.Tensor:
        ri = torch.arange(row_lo, row_lo + n_row, device=device)
        sj = torch.arange(col_lo, col_lo + n_col, device=device)
        return ri[:, None] >= sj[None, :]

    for i_tr in range(nt):
        t_lo = i_tr * chunk_size
        t_hi = min(t_lo + chunk_size, T)
        n_t = t_hi - t_lo
        if n_t <= 0:
            break

        q_blk = q[:, t_lo:t_hi, :, :]
        u_blk = u[:, t_lo:t_hi, :, :]
        go_blk = grad_o[:, t_lo:t_hi, :, :]

        for i_tc in range(i_tr + 1):
            s_lo = i_tc * chunk_size
            s_hi = min(s_lo + chunk_size, T)
            n_s = s_hi - s_lo
            if n_s <= 0:
                break

            k_blk = k[:, s_lo:s_hi, :, :]
            r_blk = r[:, s_lo:s_hi, :, :]
            v_blk = v[:, s_lo:s_hi, :, :]

            a = torch.einsum("bthd,bshd->bhts", q_blk, k_blk).to(dtype=dtype)
            b_dot = torch.einsum("bthd,bshd->bhts", u_blk, r_blk).to(dtype=dtype)
            score = a * b_dot

            mask_bt: torch.Tensor | None = None
            if i_tc == i_tr:
                mask_bt = _pair_mask_row_col(t_lo, n_t, s_lo, n_s)
                mb = mask_bt.view(1, 1, n_t, n_s)
                score = torch.where(mb, score, torch.zeros_like(score))

            gx: torch.Tensor | None = None
            if gh is not None:
                g_row = gh[:, :, t_lo:t_hi]
                g_col = gh[:, :, s_lo:s_hi]
                if use_exp2:
                    gx = torch.exp2(
                        g_row[:, :, :, None].to(dtype=torch.float32)
                        - g_col[:, :, None, :].to(dtype=torch.float32)
                    ).to(dtype=dtype)
                else:
                    gx = torch.exp(
                        g_row[:, :, :, None].to(dtype=torch.float32)
                        - g_col[:, :, None, :].to(dtype=torch.float32)
                    ).to(dtype=dtype)
                if mask_bt is None:
                    score = score * gx
                else:
                    mb = mask_bt.view(1, 1, n_t, n_s)
                    score = torch.where(mb, score * gx, torch.zeros_like(score))

            dscore = torch.einsum("bthv,bshv->bhts", go_blk, v_blk).to(dtype=dtype)
            if mask_bt is not None:
                mb = mask_bt.view(1, 1, n_t, n_s)
                dscore = torch.where(mb, dscore, torch.zeros_like(dscore))

            fac_q = dscore * b_dot
            fac_u = dscore * a
            if gx is not None:
                fac_q = fac_q * gx
                fac_u = fac_u * gx

            dq[:, t_lo:t_hi, :, :] += torch.einsum(
                "bhts,bshd->bthd", fac_q, k_blk,
            ).to(dtype=dtype)
            dk[:, s_lo:s_hi, :, :] += torch.einsum(
                "bhts,bthd->bshd", fac_q, q_blk,
            ).to(dtype=dtype)
            du[:, t_lo:t_hi, :, :] += torch.einsum(
                "bhts,bshd->bthd", fac_u, r_blk,
            ).to(dtype=dtype)
            dr[:, s_lo:s_hi, :, :] += torch.einsum(
                "bhts,bthd->bshd", fac_u, u_blk,
            ).to(dtype=dtype)

            if dg is not None:
                inc = (dscore * score).to(dtype=torch.float32) * dg_scale
                dg[:, t_lo:t_hi, :] += inc.sum(dim=-1).permute(0, 2, 1).to(dtype=dtype)
                dg[:, s_lo:s_hi, :] -= inc.sum(dim=-2).permute(0, 2, 1).to(dtype=dtype)

    return dq, dk, du, dr, dg


def _stm_causal_bilinear_bwd_chunked_torch(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    *,
    chunk_size: int,
    use_exp2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked PyTorch backward matching CUDA / dense autograd (used when Triton bwd tiles disagree)."""
    dv = _stm_accumulate_grad_v_chunked_torch(
        grad_o, q, k, u, r, g, chunk_size=chunk_size, use_exp2=use_exp2,
    )
    dq, dk, du, dr, dg = _stm_accumulate_grad_qkur_chunked_torch(
        grad_o, q, k, u, r, v, g, chunk_size=chunk_size, use_exp2=use_exp2,
    )
    if dg is None:
        dg = torch.zeros(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float32)
    return dq, dk, du, dr, dv, dg


def _stm_bwd_cuda_chunked(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    *,
    chunk_size: int,
    use_exp2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused chunked CUDA backward (outer chunk up to 512; falls back to PyTorch chunked)."""
    from minipi05.stm import stm_chunked_bilinear_ext as _cext

    ch = _cext.effective_cuda_chunk_size(q.shape[1], chunk_size)
    if _cext.extension_available():
        return _cext.stm_causal_bilinear_bwd_cuda_ext(
            grad_o, q, k, u, r, v, g, chunk_size=ch, use_exp2=use_exp2,
        )
    return _stm_causal_bilinear_bwd_chunked_torch(
        grad_o, q, k, u, r, v, g, chunk_size=ch, use_exp2=use_exp2,
    )


def _stm_bwd_dense_recomp_torch(
    grad_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    *,
    use_exp2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward via ``stm_causal_bilinear_fwd_torch`` + ``autograd.grad`` (matches dense reference)."""
    has_gate = g is not None and bool(g.numel() > 0)
    with torch.enable_grad():
        cq = q.detach().clone().requires_grad_(True)
        ck = k.detach().clone().requires_grad_(True)
        cu = u.detach().clone().requires_grad_(True)
        cr = r.detach().clone().requires_grad_(True)
        cv = v.detach().clone().requires_grad_(True)
        gm = g.detach().clone().requires_grad_(True) if has_gate else None

        o_rc = stm_causal_bilinear_fwd_torch(
            cq, ck, cu, cr, cv, gm, use_exp2=use_exp2,
        )
        want = (cq, ck, cu, cr, cv) + ((gm,) if has_gate else ())
        grads = torch.autograd.grad(
            outputs=o_rc,
            inputs=want,
            grad_outputs=grad_o,
            retain_graph=False,
            only_inputs=True,
            allow_unused=False,
        )

    dq_, dk_, du_, dr_, dv_ = grads[:5]
    if has_gate:
        dg_out = grads[5]
        assert dg_out is not None
    else:
        dg_out = torch.zeros(
            q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=q.dtype,
        )
    return dq_, dk_, du_, dr_, dv_, dg_out


STM_TRITON_AVAILABLE: bool = False

try:
    import triton
    import triton.language as tl

    from minipi05.gateddeltanet.fla.ops.utils.op import exp, exp2
    from minipi05.gateddeltanet.fla.utils import autotune_cache_kwargs

    @triton.autotune(
        configs=[
            triton.Config({"BK": 32, "BDU": 32, "BV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BK": 64, "BDU": 64, "BV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BK": 32, "BDU": 32, "BV": 128}, num_warps=8, num_stages=2),
        ],
        key=["K", "DU", "V", "BT", "USE_G", "USE_EXP2"],
        **autotune_cache_kwargs,
    )
    @triton.jit
    def _stm_causal_bilinear_fwd_kernel(
        q,
        k,
        u,
        r,
        v,
        g,
        o,
        T,
        tile_tr_offset,
        H: tl.constexpr,
        K: tl.constexpr,
        DU: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BDU: tl.constexpr,
        BV: tl.constexpr,
        USE_G: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_tr = tile_tr_offset + tl.program_id(1)
        i_bh = tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H

        bos = i_b * T
        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        u_ptr = u + (bos * H + i_h) * DU
        r_ptr = r + (bos * H + i_h) * DU
        v_ptr = v + (bos * H + i_h) * V
        o_ptr = o + (bos * H + i_h) * V

        if USE_G:
            g_ptr = g + bos * H + i_h

        b_o = tl.zeros([BT, BV], dtype=tl.float32)

        o_t = i_tr * BT + tl.arange(0, BT)
        m_t = o_t < T

        for i_tc in range(i_tr + 1):
            p_v = tl.make_block_ptr(
                v_ptr, (T, V), (H * V, 1), (i_tc * BT, i_v * BV), (BT, BV), (1, 0)
            )
            b_vc = tl.load(p_v, boundary_check=(0, 1))

            b_a = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_q = tl.make_block_ptr(
                    q_ptr, (T, K), (H * K, 1), (i_tr * BT, i_k * BK), (BT, BK), (1, 0)
                )
                p_kc = tl.make_block_ptr(
                    k_ptr, (T, K), (H * K, 1), (i_tc * BT, i_k * BK), (BT, BK), (1, 0)
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_kc = tl.load(p_kc, boundary_check=(0, 1))
                b_a += tl.dot(b_q, tl.trans(b_kc))

            b_b = tl.zeros([BT, BT], dtype=tl.float32)
            for i_d in range(tl.cdiv(DU, BDU)):
                p_u = tl.make_block_ptr(
                    u_ptr, (T, DU), (H * DU, 1), (i_tr * BT, i_d * BDU), (BT, BDU), (1, 0)
                )
                p_rc = tl.make_block_ptr(
                    r_ptr, (T, DU), (H * DU, 1), (i_tc * BT, i_d * BDU), (BT, BDU), (1, 0)
                )
                b_u = tl.load(p_u, boundary_check=(0, 1))
                b_rc = tl.load(p_rc, boundary_check=(0, 1))
                b_b += tl.dot(b_u, tl.trans(b_rc))

            b_score = b_a * b_b

            if USE_G:
                p_gr = tl.make_block_ptr(
                    g_ptr, (T,), (H,), (i_tr * BT,), (BT,), (0,)
                )
                p_gc = tl.make_block_ptr(
                    g_ptr, (T,), (H,), (i_tc * BT,), (BT,), (0,)
                )
                b_gr = tl.load(p_gr, boundary_check=(0,))
                b_gc = tl.load(p_gc, boundary_check=(0,))
                if USE_EXP2:
                    b_score = b_score * exp2(b_gr[:, None] - b_gc[None, :])
                else:
                    b_score = b_score * exp(b_gr[:, None] - b_gc[None, :])

            o_c = i_tc * BT + tl.arange(0, BT)
            m_c = o_c < T
            if i_tc < i_tr:
                m_block = m_t[:, None] & m_c[None, :]
            else:
                m_block = (o_t[:, None] >= o_c[None, :]) & (m_t[:, None] & m_c[None, :])
            b_score = tl.where(m_block, b_score, 0.0)

            b_o += tl.dot(b_score.to(b_vc.dtype), b_vc)

        p_o = tl.make_block_ptr(
            o_ptr, (T, V), (H * V, 1), (i_tr * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    _STM_BWD_LN2: tl.constexpr = 0.693147180559945309417232121458176568075500134360255254120680009493409621898

    @triton.autotune(
        configs=[
            triton.Config({"BK": 32, "BDU": 32, "BV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BK": 64, "BDU": 64, "BV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BK": 32, "BDU": 32, "BV": 128}, num_warps=8, num_stages=2),
        ],
        key=["K", "DU", "V", "BT", "USE_G", "USE_EXP2"],
        **autotune_cache_kwargs,
    )
    @triton.jit
    def _stm_bwd_dv_kernel(
        go,
        q,
        k,
        u,
        r,
        v,
        g,
        dv,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        DU: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BDU: tl.constexpr,
        BV: tl.constexpr,
        USE_G: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        """``dv_s += sum_{t>=s} score[t,s] * grad_o[t]`` â€” tile loop mirrors forward."""
        i_v = tl.program_id(0)
        i_tc = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        bos = i_b * T

        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        u_ptr = u + (bos * H + i_h) * DU
        r_ptr = r + (bos * H + i_h) * DU
        go_ptr = go + (bos * H + i_h) * V
        v_ptr = v + (bos * H + i_h) * V
        dv_ptr = dv + (bos * H + i_h) * V

        if USE_G:
            g_ptr = g + bos * H + i_h

        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        NT = tl.cdiv(T, BT)

        for i_tr in range(i_tc, NT):
            o_t = i_tr * BT + tl.arange(0, BT)
            m_t = o_t < T

            p_go = tl.make_block_ptr(
                go_ptr, (T, V), (H * V, 1), (i_tr * BT, i_v * BV), (BT, BV), (1, 0)
            )
            b_go = tl.load(p_go, boundary_check=(0, 1))

            b_a = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_q = tl.make_block_ptr(
                    q_ptr, (T, K), (H * K, 1), (i_tr * BT, i_k * BK), (BT, BK), (1, 0)
                )
                p_kc = tl.make_block_ptr(
                    k_ptr, (T, K), (H * K, 1), (i_tc * BT, i_k * BK), (BT, BK), (1, 0)
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_kc = tl.load(p_kc, boundary_check=(0, 1))
                b_a += tl.dot(b_q, tl.trans(b_kc))

            b_b = tl.zeros([BT, BT], dtype=tl.float32)
            for i_d in range(tl.cdiv(DU, BDU)):
                p_u = tl.make_block_ptr(
                    u_ptr, (T, DU), (H * DU, 1), (i_tr * BT, i_d * BDU), (BT, BDU), (1, 0)
                )
                p_rc = tl.make_block_ptr(
                    r_ptr, (T, DU), (H * DU, 1), (i_tc * BT, i_d * BDU), (BT, BDU), (1, 0)
                )
                b_u = tl.load(p_u, boundary_check=(0, 1))
                b_rc = tl.load(p_rc, boundary_check=(0, 1))
                b_b += tl.dot(b_u, tl.trans(b_rc))

            b_score = b_a * b_b

            if USE_G:
                p_gr = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tr * BT,), (BT,), (0,))
                p_gc = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tc * BT,), (BT,), (0,))
                b_gr = tl.load(p_gr, boundary_check=(0,))
                b_gc = tl.load(p_gc, boundary_check=(0,))
                if USE_EXP2:
                    b_score = b_score * exp2(b_gr[:, None] - b_gc[None, :])
                else:
                    b_score = b_score * exp(b_gr[:, None] - b_gc[None, :])

            o_c = i_tc * BT + tl.arange(0, BT)
            m_c = o_c < T
            if i_tc < i_tr:
                m_block = m_t[:, None] & m_c[None, :]
            else:
                m_block = (o_t[:, None] >= o_c[None, :]) & (m_t[:, None] & m_c[None, :])
            b_score = tl.where(m_block, b_score, 0.0)

            b_dv += tl.dot(tl.trans(b_score), b_go)

        p_dv = tl.make_block_ptr(
            dv_ptr, (T, V), (H * V, 1), (i_tc * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    @triton.autotune(
        configs=[
            triton.Config({"BK": 32, "BDU": 32, "BV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BK": 64, "BDU": 64, "BV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BK": 32, "BDU": 32, "BV": 128}, num_warps=8, num_stages=2),
        ],
        key=["K", "DU", "V", "BT", "USE_G", "USE_EXP2"],
        **autotune_cache_kwargs,
    )
    @triton.jit
    def _stm_bwd_dq_du_kernel(
        go,
        q,
        k,
        u,
        r,
        v,
        g,
        dq,
        du,
        dg,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        DU: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BDU: tl.constexpr,
        BV: tl.constexpr,
        USE_G: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        """Accumulate ``dq``, ``du``, and row-wise ``dg`` for output tile ``i_tr``."""
        i_tr = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        bos = i_b * T

        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        u_ptr = u + (bos * H + i_h) * DU
        r_ptr = r + (bos * H + i_h) * DU
        go_ptr = go + (bos * H + i_h) * V
        v_ptr = v + (bos * H + i_h) * V
        dq_ptr = dq + (bos * H + i_h) * K
        du_ptr = du + (bos * H + i_h) * DU
        dg_ptr = dg + bos * H + i_h

        if USE_G:
            g_ptr = g + bos * H + i_h

        b_dg_row = tl.zeros([BT], dtype=tl.float32)
        o_t = i_tr * BT + tl.arange(0, BT)
        m_t = o_t < T

        for i_k in range(tl.cdiv(K, BK)):
            b_dq = tl.zeros([BT, BK], dtype=tl.float32)
            for i_tc in range(i_tr + 1):
                b_a = tl.zeros([BT, BT], dtype=tl.float32)
                for i_ka in range(tl.cdiv(K, BK)):
                    p_q = tl.make_block_ptr(
                        q_ptr, (T, K), (H * K, 1), (i_tr * BT, i_ka * BK), (BT, BK), (1, 0)
                    )
                    p_kc = tl.make_block_ptr(
                        k_ptr, (T, K), (H * K, 1), (i_tc * BT, i_ka * BK), (BT, BK), (1, 0)
                    )
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_kc = tl.load(p_kc, boundary_check=(0, 1))
                    b_a += tl.dot(b_q, tl.trans(b_kc))

                b_b = tl.zeros([BT, BT], dtype=tl.float32)
                for i_da in range(tl.cdiv(DU, BDU)):
                    p_u = tl.make_block_ptr(
                        u_ptr, (T, DU), (H * DU, 1), (i_tr * BT, i_da * BDU), (BT, BDU), (1, 0)
                    )
                    p_rc = tl.make_block_ptr(
                        r_ptr, (T, DU), (H * DU, 1), (i_tc * BT, i_da * BDU), (BT, BDU), (1, 0)
                    )
                    b_u = tl.load(p_u, boundary_check=(0, 1))
                    b_rc = tl.load(p_rc, boundary_check=(0, 1))
                    b_b += tl.dot(b_u, tl.trans(b_rc))

                b_gamma = tl.full([BT, BT], 1.0, dtype=tl.float32)
                if USE_G:
                    p_gr = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tr * BT,), (BT,), (0,))
                    p_gc = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tc * BT,), (BT,), (0,))
                    b_gr = tl.load(p_gr, boundary_check=(0,))
                    b_gc = tl.load(p_gc, boundary_check=(0,))
                    if USE_EXP2:
                        b_gamma = exp2(b_gr[:, None] - b_gc[None, :])
                    else:
                        b_gamma = exp(b_gr[:, None] - b_gc[None, :])

                b_score = b_a * b_b * b_gamma
                o_c = i_tc * BT + tl.arange(0, BT)
                m_c = o_c < T
                if i_tc < i_tr:
                    m_block = m_t[:, None] & m_c[None, :]
                else:
                    m_block = (o_t[:, None] >= o_c[None, :]) & (m_t[:, None] & m_c[None, :])
                b_score = tl.where(m_block, b_score, 0.0)

                b_dscore = tl.zeros([BT, BT], dtype=tl.float32)
                for i_v in range(tl.cdiv(V, BV)):
                    p_go = tl.make_block_ptr(
                        go_ptr, (T, V), (H * V, 1), (i_tr * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    p_vc = tl.make_block_ptr(
                        v_ptr, (T, V), (H * V, 1), (i_tc * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    b_go = tl.load(p_go, boundary_check=(0, 1))
                    b_vc = tl.load(p_vc, boundary_check=(0, 1))
                    b_dscore += tl.dot(b_go, tl.trans(b_vc))
                b_dscore = tl.where(m_block, b_dscore, 0.0)

                b_da = b_dscore * b_b * b_gamma
                p_kc = tl.make_block_ptr(
                    k_ptr, (T, K), (H * K, 1), (i_tc * BT, i_k * BK), (BT, BK), (1, 0)
                )
                b_kc = tl.load(p_kc, boundary_check=(0, 1))
                b_dq += tl.dot(b_da, b_kc)

                if USE_G:
                    b_dg_part = tl.sum(b_dscore * b_score, axis=1)
                    if USE_EXP2:
                        b_dg_part = b_dg_part * _STM_BWD_LN2
                    b_dg_row += b_dg_part

            p_dq = tl.make_block_ptr(
                dq_ptr, (T, K), (H * K, 1), (i_tr * BT, i_k * BK), (BT, BK), (1, 0)
            )
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

        for i_d in range(tl.cdiv(DU, BDU)):
            b_du = tl.zeros([BT, BDU], dtype=tl.float32)
            for i_tc in range(i_tr + 1):
                b_a = tl.zeros([BT, BT], dtype=tl.float32)
                for i_ka in range(tl.cdiv(K, BK)):
                    p_q = tl.make_block_ptr(
                        q_ptr, (T, K), (H * K, 1), (i_tr * BT, i_ka * BK), (BT, BK), (1, 0)
                    )
                    p_kc = tl.make_block_ptr(
                        k_ptr, (T, K), (H * K, 1), (i_tc * BT, i_ka * BK), (BT, BK), (1, 0)
                    )
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_kc = tl.load(p_kc, boundary_check=(0, 1))
                    b_a += tl.dot(b_q, tl.trans(b_kc))

                b_b = tl.zeros([BT, BT], dtype=tl.float32)
                for i_da in range(tl.cdiv(DU, BDU)):
                    p_u = tl.make_block_ptr(
                        u_ptr, (T, DU), (H * DU, 1), (i_tr * BT, i_da * BDU), (BT, BDU), (1, 0)
                    )
                    p_rc = tl.make_block_ptr(
                        r_ptr, (T, DU), (H * DU, 1), (i_tc * BT, i_da * BDU), (BT, BDU), (1, 0)
                    )
                    b_u = tl.load(p_u, boundary_check=(0, 1))
                    b_rc = tl.load(p_rc, boundary_check=(0, 1))
                    b_b += tl.dot(b_u, tl.trans(b_rc))

                b_gamma = tl.full([BT, BT], 1.0, dtype=tl.float32)
                if USE_G:
                    p_gr = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tr * BT,), (BT,), (0,))
                    p_gc = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tc * BT,), (BT,), (0,))
                    b_gr = tl.load(p_gr, boundary_check=(0,))
                    b_gc = tl.load(p_gc, boundary_check=(0,))
                    if USE_EXP2:
                        b_gamma = exp2(b_gr[:, None] - b_gc[None, :])
                    else:
                        b_gamma = exp(b_gr[:, None] - b_gc[None, :])

                b_score = b_a * b_b * b_gamma
                o_c = i_tc * BT + tl.arange(0, BT)
                m_c = o_c < T
                if i_tc < i_tr:
                    m_block = m_t[:, None] & m_c[None, :]
                else:
                    m_block = (o_t[:, None] >= o_c[None, :]) & (m_t[:, None] & m_c[None, :])
                b_score = tl.where(m_block, b_score, 0.0)

                b_dscore = tl.zeros([BT, BT], dtype=tl.float32)
                for i_v in range(tl.cdiv(V, BV)):
                    p_go = tl.make_block_ptr(
                        go_ptr, (T, V), (H * V, 1), (i_tr * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    p_vc = tl.make_block_ptr(
                        v_ptr, (T, V), (H * V, 1), (i_tc * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    b_go = tl.load(p_go, boundary_check=(0, 1))
                    b_vc = tl.load(p_vc, boundary_check=(0, 1))
                    b_dscore += tl.dot(b_go, tl.trans(b_vc))
                b_dscore = tl.where(m_block, b_dscore, 0.0)

                b_db = b_dscore * b_a * b_gamma
                p_rc = tl.make_block_ptr(
                    r_ptr, (T, DU), (H * DU, 1), (i_tc * BT, i_d * BDU), (BT, BDU), (1, 0)
                )
                b_rc = tl.load(p_rc, boundary_check=(0, 1))
                b_du += tl.dot(b_db, b_rc)

            p_du = tl.make_block_ptr(
                du_ptr, (T, DU), (H * DU, 1), (i_tr * BT, i_d * BDU), (BT, BDU), (1, 0)
            )
            tl.store(p_du, b_du.to(p_du.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            tl.atomic_add(dg_ptr + o_t * H, b_dg_row, mask=m_t)

    @triton.autotune(
        configs=[
            triton.Config({"BK": 32, "BDU": 32, "BV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BK": 64, "BDU": 64, "BV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BK": 32, "BDU": 32, "BV": 128}, num_warps=8, num_stages=2),
        ],
        key=["K", "DU", "V", "BT", "USE_G", "USE_EXP2"],
        **autotune_cache_kwargs,
    )
    @triton.jit
    def _stm_bwd_dk_dr_kernel(
        go,
        q,
        k,
        u,
        r,
        v,
        g,
        dk,
        dr,
        dg,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        DU: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BDU: tl.constexpr,
        BV: tl.constexpr,
        USE_G: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        """Accumulate ``dk``, ``dr``, and column-wise ``dg`` for key tile ``i_tc``."""
        i_tc = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H
        bos = i_b * T

        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        u_ptr = u + (bos * H + i_h) * DU
        r_ptr = r + (bos * H + i_h) * DU
        go_ptr = go + (bos * H + i_h) * V
        v_ptr = v + (bos * H + i_h) * V
        dk_ptr = dk + (bos * H + i_h) * K
        dr_ptr = dr + (bos * H + i_h) * DU
        dg_ptr = dg + bos * H + i_h

        if USE_G:
            g_ptr = g + bos * H + i_h

        b_dg_col = tl.zeros([BT], dtype=tl.float32)
        o_c = i_tc * BT + tl.arange(0, BT)
        m_c = o_c < T
        NT = tl.cdiv(T, BT)

        for i_k in range(tl.cdiv(K, BK)):
            b_dk = tl.zeros([BT, BK], dtype=tl.float32)
            for i_tr in range(i_tc, NT):
                o_t = i_tr * BT + tl.arange(0, BT)
                m_t = o_t < T

                b_a = tl.zeros([BT, BT], dtype=tl.float32)
                for i_ka in range(tl.cdiv(K, BK)):
                    p_q = tl.make_block_ptr(
                        q_ptr, (T, K), (H * K, 1), (i_tr * BT, i_ka * BK), (BT, BK), (1, 0)
                    )
                    p_kc = tl.make_block_ptr(
                        k_ptr, (T, K), (H * K, 1), (i_tc * BT, i_ka * BK), (BT, BK), (1, 0)
                    )
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_kc = tl.load(p_kc, boundary_check=(0, 1))
                    b_a += tl.dot(b_q, tl.trans(b_kc))

                b_b = tl.zeros([BT, BT], dtype=tl.float32)
                for i_da in range(tl.cdiv(DU, BDU)):
                    p_u = tl.make_block_ptr(
                        u_ptr, (T, DU), (H * DU, 1), (i_tr * BT, i_da * BDU), (BT, BDU), (1, 0)
                    )
                    p_rc = tl.make_block_ptr(
                        r_ptr, (T, DU), (H * DU, 1), (i_tc * BT, i_da * BDU), (BT, BDU), (1, 0)
                    )
                    b_u = tl.load(p_u, boundary_check=(0, 1))
                    b_rc = tl.load(p_rc, boundary_check=(0, 1))
                    b_b += tl.dot(b_u, tl.trans(b_rc))

                b_gamma = tl.full([BT, BT], 1.0, dtype=tl.float32)
                if USE_G:
                    p_gr = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tr * BT,), (BT,), (0,))
                    p_gc = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tc * BT,), (BT,), (0,))
                    b_gr = tl.load(p_gr, boundary_check=(0,))
                    b_gc = tl.load(p_gc, boundary_check=(0,))
                    if USE_EXP2:
                        b_gamma = exp2(b_gr[:, None] - b_gc[None, :])
                    else:
                        b_gamma = exp(b_gr[:, None] - b_gc[None, :])

                b_score = b_a * b_b * b_gamma
                if i_tc < i_tr:
                    m_block = m_t[:, None] & m_c[None, :]
                else:
                    m_block = (o_t[:, None] >= o_c[None, :]) & (m_t[:, None] & m_c[None, :])
                b_score = tl.where(m_block, b_score, 0.0)

                b_dscore = tl.zeros([BT, BT], dtype=tl.float32)
                for i_v in range(tl.cdiv(V, BV)):
                    p_go = tl.make_block_ptr(
                        go_ptr, (T, V), (H * V, 1), (i_tr * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    p_vc = tl.make_block_ptr(
                        v_ptr, (T, V), (H * V, 1), (i_tc * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    b_go = tl.load(p_go, boundary_check=(0, 1))
                    b_vc = tl.load(p_vc, boundary_check=(0, 1))
                    b_dscore += tl.dot(b_go, tl.trans(b_vc))
                b_dscore = tl.where(m_block, b_dscore, 0.0)

                b_da = b_dscore * b_b * b_gamma
                p_q = tl.make_block_ptr(
                    q_ptr, (T, K), (H * K, 1), (i_tr * BT, i_k * BK), (BT, BK), (1, 0)
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_dk += tl.dot(tl.trans(b_da), b_q)

                if USE_G:
                    b_dg_part = tl.sum(b_dscore * b_score, axis=0)
                    if USE_EXP2:
                        b_dg_part = b_dg_part * _STM_BWD_LN2
                    b_dg_col += b_dg_part

            p_dk = tl.make_block_ptr(
                dk_ptr, (T, K), (H * K, 1), (i_tc * BT, i_k * BK), (BT, BK), (1, 0)
            )
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        for i_d in range(tl.cdiv(DU, BDU)):
            b_dr = tl.zeros([BT, BDU], dtype=tl.float32)
            for i_tr in range(i_tc, NT):
                o_t = i_tr * BT + tl.arange(0, BT)
                m_t = o_t < T

                b_a = tl.zeros([BT, BT], dtype=tl.float32)
                for i_ka in range(tl.cdiv(K, BK)):
                    p_q = tl.make_block_ptr(
                        q_ptr, (T, K), (H * K, 1), (i_tr * BT, i_ka * BK), (BT, BK), (1, 0)
                    )
                    p_kc = tl.make_block_ptr(
                        k_ptr, (T, K), (H * K, 1), (i_tc * BT, i_ka * BK), (BT, BK), (1, 0)
                    )
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_kc = tl.load(p_kc, boundary_check=(0, 1))
                    b_a += tl.dot(b_q, tl.trans(b_kc))

                b_b = tl.zeros([BT, BT], dtype=tl.float32)
                for i_da in range(tl.cdiv(DU, BDU)):
                    p_u = tl.make_block_ptr(
                        u_ptr, (T, DU), (H * DU, 1), (i_tr * BT, i_da * BDU), (BT, BDU), (1, 0)
                    )
                    p_rc = tl.make_block_ptr(
                        r_ptr, (T, DU), (H * DU, 1), (i_tc * BT, i_da * BDU), (BT, BDU), (1, 0)
                    )
                    b_u = tl.load(p_u, boundary_check=(0, 1))
                    b_rc = tl.load(p_rc, boundary_check=(0, 1))
                    b_b += tl.dot(b_u, tl.trans(b_rc))

                b_gamma = tl.full([BT, BT], 1.0, dtype=tl.float32)
                if USE_G:
                    p_gr = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tr * BT,), (BT,), (0,))
                    p_gc = tl.make_block_ptr(g_ptr, (T,), (H,), (i_tc * BT,), (BT,), (0,))
                    b_gr = tl.load(p_gr, boundary_check=(0,))
                    b_gc = tl.load(p_gc, boundary_check=(0,))
                    if USE_EXP2:
                        b_gamma = exp2(b_gr[:, None] - b_gc[None, :])
                    else:
                        b_gamma = exp(b_gr[:, None] - b_gc[None, :])

                b_score = b_a * b_b * b_gamma
                if i_tc < i_tr:
                    m_block = m_t[:, None] & m_c[None, :]
                else:
                    m_block = (o_t[:, None] >= o_c[None, :]) & (m_t[:, None] & m_c[None, :])
                b_score = tl.where(m_block, b_score, 0.0)

                b_dscore = tl.zeros([BT, BT], dtype=tl.float32)
                for i_v in range(tl.cdiv(V, BV)):
                    p_go = tl.make_block_ptr(
                        go_ptr, (T, V), (H * V, 1), (i_tr * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    p_vc = tl.make_block_ptr(
                        v_ptr, (T, V), (H * V, 1), (i_tc * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    b_go = tl.load(p_go, boundary_check=(0, 1))
                    b_vc = tl.load(p_vc, boundary_check=(0, 1))
                    b_dscore += tl.dot(b_go, tl.trans(b_vc))
                b_dscore = tl.where(m_block, b_dscore, 0.0)

                b_db = b_dscore * b_a * b_gamma
                p_u = tl.make_block_ptr(
                    u_ptr, (T, DU), (H * DU, 1), (i_tr * BT, i_d * BDU), (BT, BDU), (1, 0)
                )
                b_u = tl.load(p_u, boundary_check=(0, 1))
                b_dr += tl.dot(tl.trans(b_db), b_u)

            p_dr = tl.make_block_ptr(
                dr_ptr, (T, DU), (H * DU, 1), (i_tc * BT, i_d * BDU), (BT, BDU), (1, 0)
            )
            tl.store(p_dr, b_dr.to(p_dr.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            tl.atomic_add(dg_ptr + o_c * H, -b_dg_col, mask=m_c)

    STM_OUTER_TIME_CHUNK: int = 512

    def _launch_stm_causal_bilinear_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        o: torch.Tensor,
        *,
        chunk_size: int,
        tile_tr_offset: int,
        n_tr_tiles: int,
        use_exp2: bool,
    ) -> None:
        B, T, H, K = q.shape
        _B2, _T2, H2, DU = u.shape
        _B3, _T3, H3, V = v.shape
        assert B == _B2 == _B3 and T == _T2 == _T3 and H == H2 == H3
        BT = stm_triton_tile_bt(T, chunk_size)
        use_g = g is not None
        g_ptr = g if use_g else q
        grid = lambda meta: (triton.cdiv(V, meta["BV"]), n_tr_tiles, B * H)
        _stm_causal_bilinear_fwd_kernel[grid](
            q,
            k,
            u,
            r,
            v,
            g_ptr,
            o,
            T,
            tile_tr_offset,
            H=H,
            K=K,
            DU=DU,
            V=V,
            BT=BT,
            USE_G=use_g,
            USE_EXP2=use_exp2,
        )

    def stm_causal_bilinear_fwd_outer_chunk(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        inner_chunk_size: int = 64,
        outer_chunk_size: int = STM_OUTER_TIME_CHUNK,
        use_exp2: bool = False,
    ) -> torch.Tensor:
        """Triton forward with a Python loop over outer time segments (default 512).

        Each segment launches the fused kernel on output tiles ``[t_lo, t_hi)`` while
        keys ``s <= t`` still use the full prefix (causal). Inner tiles are
        ``inner_chunk_size`` (default 64). ``T=1024`` → 4 outer passes.
        """
        if q.device.type != "cuda":
            raise RuntimeError("stm_causal_bilinear_fwd_outer_chunk requires CUDA tensors")
        q = q.contiguous()
        k = k.contiguous()
        u = u.contiguous()
        r = r.contiguous()
        v = v.contiguous()
        if g is not None:
            g = g.contiguous()

        B, T, H, K = q.shape
        o = torch.empty_like(v)
        BT = stm_triton_tile_bt(T, inner_chunk_size)
        outer = max(BT, int(outer_chunk_size))

        for t0 in range(0, T, outer):
            t1 = min(t0 + outer, T)
            nt_start = t0 // BT
            nt_end = (t1 + BT - 1) // BT
            n_tr = nt_end - nt_start
            if n_tr <= 0:
                continue
            _launch_stm_causal_bilinear_fwd(
                q, k, u, r, v, g, o,
                chunk_size=BT,
                tile_tr_offset=nt_start,
                n_tr_tiles=n_tr,
                use_exp2=use_exp2,
            )
        return o

    def stm_causal_bilinear_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        chunk_size: int = 64,
        use_exp2: bool = False,
    ) -> torch.Tensor:
        if q.device.type != "cuda":
            raise RuntimeError("stm_causal_bilinear_fwd requires CUDA tensors")
        q = q.contiguous()
        k = k.contiguous()
        u = u.contiguous()
        r = r.contiguous()
        v = v.contiguous()
        if g is not None:
            g = g.contiguous()

        B, T, H, K = q.shape
        _B2, _T2, H2, DU = u.shape
        _B3, _T3, H3, V = v.shape
        assert B == _B2 == _B3 and T == _T2 == _T3 and H == H2 == H3

        o = torch.empty_like(v)
        BT = stm_triton_tile_bt(T, chunk_size)
        NT = triton.cdiv(T, BT)
        _launch_stm_causal_bilinear_fwd(
            q, k, u, r, v, g, o,
            chunk_size=BT,
            tile_tr_offset=0,
            n_tr_tiles=NT,
            use_exp2=use_exp2,
        )
        return o

    def _stm_bwd_fused_triton(
        grad_o: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        *,
        chunk_size: int,
        use_exp2: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chunked PyTorch backward (validated); Triton tile kernels reserved for later."""
        return _stm_causal_bilinear_bwd_chunked_torch(
            grad_o, q, k, u, r, v, g, chunk_size=chunk_size, use_exp2=use_exp2,
        )

    def stm_causal_bilinear_bwd_triton(
        grad_o: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        *,
        chunk_size: int,
        use_exp2: bool,
        use_cuda_chunked: bool = False,
        use_fused_triton: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward: fused Triton tiles (default), dense PyTorch recompute, or CUDA chunked."""
        if grad_o.device.type != "cuda":
            raise RuntimeError("stm_causal_bilinear_bwd_triton requires CUDA tensors")

        def _req_fp32(name: str, t: torch.Tensor) -> None:
            if t.dtype != torch.float32 or not t.is_contiguous():
                raise RuntimeError(f"stm_causal_bilinear_bwd_triton: {name} must be float32 contiguous, got {t.dtype}")

        _req_fp32("grad_o", grad_o)
        for nm, t in (("q", q), ("k", k), ("u", u), ("r", r), ("v", v)):
            _req_fp32(nm, t)
        use_g = g is not None and bool(g.numel() > 0)
        if use_g:
            _req_fp32("g", g)

        q = q.contiguous()
        k = k.contiguous()
        u = u.contiguous()
        r = r.contiguous()
        v = v.contiguous()
        grad_o = grad_o.contiguous()
        if tuple(grad_o.shape) != tuple(v.shape):
            raise RuntimeError("stm_causal_bilinear_bwd_triton: grad_o.shape must equal v.shape")

        g_arg = g.contiguous() if use_g else None
        if use_cuda_chunked:
            return _stm_bwd_cuda_chunked(
                grad_o, q, k, u, r, v, g_arg,
                chunk_size=chunk_size, use_exp2=use_exp2,
            )
        if use_fused_triton:
            return _stm_bwd_fused_triton(
                grad_o, q, k, u, r, v, g_arg,
                chunk_size=chunk_size, use_exp2=use_exp2,
            )
        return _stm_bwd_dense_recomp_torch(
            grad_o, q, k, u, r, v, g_arg, use_exp2=use_exp2,
        )

    class _StmCausalBiLinearTritonFwdTorchRecomputeBwd(torch.autograd.Function):
        """Fused Triton forward; backward via Triton tiles, dense PyTorch recompute, or CUDA chunked."""

        @staticmethod
        def forward(  # type: ignore[override]
            ctx,
            cfg: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            u: torch.Tensor,
            r: torch.Tensor,
            v: torch.Tensor,
            gate_or_none: torch.Tensor,
        ) -> torch.Tensor:
            chunk_triton = int(cfg[0].item())
            use_exp2 = bool(int(cfg[1].item()) != 0)
            use_dense_bwd_recomp = bool(int(cfg[2].item()) != 0)
            chunk_torch = max(1, int(cfg[3].item()))
            use_cuda_bwd = bool(int(cfg[4].item()) != 0)
            cuda_chunk = max(1, int(cfg[5].item()))
            use_triton_bwd = bool(int(cfg[6].item()) != 0)
            outer_fwd = int(cfg[7].item()) if cfg.numel() > 7 else 0

            has_gate = bool(gate_or_none.numel() > 0)
            ctx.save_for_backward(q, k, u, r, v, gate_or_none, cfg)
            ctx.has_gate = has_gate
            ctx.use_dense_bwd_recomp = use_dense_bwd_recomp
            ctx.use_cuda_bwd = use_cuda_bwd
            ctx.use_triton_bwd = use_triton_bwd
            ctx.chunk_triton = chunk_triton
            ctx.chunk_torch = chunk_torch
            ctx.cuda_chunk = cuda_chunk
            ctx.use_exp2 = use_exp2

            g_arg = gate_or_none if has_gate else None
            if outer_fwd > 0:
                return stm_causal_bilinear_fwd_outer_chunk(
                    q, k, u, r, v, g_arg,
                    inner_chunk_size=chunk_triton,
                    outer_chunk_size=outer_fwd,
                    use_exp2=use_exp2,
                )
            return stm_causal_bilinear_fwd(
                q, k, u, r, v, g_arg,
                chunk_size=chunk_triton,
                use_exp2=use_exp2,
            )

        @staticmethod
        def backward(ctx, grad_o: torch.Tensor):  # type: ignore[override]
            q, k, u, r, v, gst, _cfg_unused = ctx.saved_tensors
            has_gate = ctx.has_gate
            use_exp2 = ctx.use_exp2
            chunk_torch = ctx.chunk_torch

            if ctx.use_dense_bwd_recomp:
                dq_, dk_, du_, dr_, dv_, dg_out = _stm_bwd_dense_recomp_torch(
                    grad_o, q, k, u, r, v,
                    gst if has_gate else None,
                    use_exp2=use_exp2,
                )
            elif ctx.use_cuda_bwd:
                dq_, dk_, du_, dr_, dv_, dg_out = _stm_bwd_cuda_chunked(
                    grad_o, q, k, u, r, v,
                    gst if has_gate else None,
                    chunk_size=ctx.cuda_chunk,
                    use_exp2=use_exp2,
                )
            elif ctx.use_triton_bwd:
                dq_, dk_, du_, dr_, dv_, dg_out = _stm_bwd_fused_triton(
                    grad_o, q, k, u, r, v,
                    gst if has_gate else None,
                    chunk_size=ctx.chunk_triton,
                    use_exp2=use_exp2,
                )
            else:
                with torch.enable_grad():
                    cq = q.detach().clone().requires_grad_(True)
                    ck = k.detach().clone().requires_grad_(True)
                    cu = u.detach().clone().requires_grad_(True)
                    cr = r.detach().clone().requires_grad_(True)
                    cv = v.detach().clone().requires_grad_(True)
                    gm = gst.detach().clone().requires_grad_(True) if has_gate else None
                    o_rc = stm_causal_bilinear_fwd_torch_chunked(
                        cq, ck, cu, cr, cv, gm,
                        chunk_size=chunk_torch,
                        use_exp2=use_exp2,
                    )
                    want = (cq, ck, cu, cr, cv) + ((gm,) if has_gate else ())
                    grads = torch.autograd.grad(
                        outputs=o_rc,
                        inputs=want,
                        grad_outputs=grad_o,
                        retain_graph=False,
                        only_inputs=True,
                        allow_unused=False,
                    )
                dq_, dk_, du_, dr_, dv_ = grads[:5]
                if has_gate:
                    dg_out = grads[5]
                    assert dg_out is not None
                else:
                    dg_out = torch.zeros_like(gst)

            dg_ret = dg_out if has_gate else torch.zeros_like(gst)
            return (
                None,
                dq_,
                dk_,
                du_,
                dr_,
                dv_,
                dg_ret,
            )

    def stm_causal_bilinear_fwd_trainable(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        chunk_size: int = 64,
        use_exp2: bool = False,
        backward_recompute: str = "dense",
        pytorch_chunk_for_backward: int | None = None,
        cuda_chunk_for_backward: int | None = None,
        outer_fwd_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Fused **Triton** forward during CUDA training.

        Gradients: ``dense`` (default) = full dense PyTorch recompute autograd;
        ``triton`` = chunked PyTorch backward (same math as ``torch|chunk``);
        ``chunked`` / ``cuda`` = fused CUDA backward (outer chunk 512 by default).
        """
        q = q.contiguous()
        k = k.contiguous()
        u = u.contiguous()
        r = r.contiguous()
        v = v.contiguous()
        g = g.contiguous() if g is not None else None

        if q.device.type != "cuda":
            return stm_causal_bilinear_fwd_torch_chunked(
                q, k, u, r, v, g,
                chunk_size=chunk_size,
                use_exp2=use_exp2,
            )
        if not torch.is_grad_enabled():
            return stm_causal_bilinear_fwd(
                q, k, u, r, v, g,
                chunk_size=chunk_size,
                use_exp2=use_exp2,
            )

        leaves_need = (
            q.requires_grad
            or k.requires_grad
            or u.requires_grad
            or r.requires_grad
            or v.requires_grad
            or (g.requires_grad if g is not None else False)
        )
        if not leaves_need:
            return stm_causal_bilinear_fwd(
                q, k, u, r, v, g,
                chunk_size=chunk_size,
                use_exp2=use_exp2,
            )

        if backward_recompute not in ("dense", "chunked", "triton", "cuda"):
            raise ValueError(
                'backward_recompute must be "dense", "chunked", "cuda", or "triton" '
                "(CUDA + Triton + fp32 contiguous only)",
            )

        pch = int(pytorch_chunk_for_backward if pytorch_chunk_for_backward is not None else chunk_size)
        pch = max(1, pch)

        from minipi05.stm.stm_chunked_bilinear_ext import (
            STM_CUDA_BWD_CHUNK,
            effective_cuda_chunk_size,
        )

        cuda_ch = effective_cuda_chunk_size(
            q.shape[1],
            cuda_chunk_for_backward if cuda_chunk_for_backward is not None else STM_CUDA_BWD_CHUNK,
        )

        gst = g if g is not None else q.new_zeros((0,))

        use_dense = backward_recompute == "dense"
        use_triton_bwd = backward_recompute == "triton"
        use_cuda_bwd = backward_recompute in ("chunked", "cuda")
        outer_fwd = 0
        if outer_fwd_chunk_size is not None:
            outer_fwd = max(int(outer_fwd_chunk_size), 0)

        cfg = torch.tensor(
            [
                chunk_size,
                int(use_exp2),
                int(use_dense),
                pch,
                int(use_cuda_bwd),
                cuda_ch,
                int(use_triton_bwd),
                outer_fwd,
            ],
            device=q.device,
            dtype=torch.int64,
        )

        return _StmCausalBiLinearTritonFwdTorchRecomputeBwd.apply(cfg, q, k, u, r, v, gst)

    def stm_causal_bilinear_fwd_outer_chunk_trainable(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        chunk_size: int = 64,
        outer_chunk_size: int = STM_OUTER_TIME_CHUNK,
        use_exp2: bool = False,
        backward_recompute: str = "dense",
        pytorch_chunk_for_backward: int | None = None,
        cuda_chunk_for_backward: int | None = None,
    ) -> torch.Tensor:
        """Triton forward in outer time segments (default 512); same backward modes as trainable."""
        return stm_causal_bilinear_fwd_trainable(
            q, k, u, r, v, g,
            chunk_size=chunk_size,
            use_exp2=use_exp2,
            backward_recompute=backward_recompute,
            pytorch_chunk_for_backward=pytorch_chunk_for_backward,
            cuda_chunk_for_backward=cuda_chunk_for_backward,
            outer_fwd_chunk_size=outer_chunk_size,
        )

    STM_TRITON_AVAILABLE = True

except Exception:  # pragma: no cover â€” Triton/FLA unavailable or broken import
    STM_TRITON_AVAILABLE = False

    def stm_causal_bilinear_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        chunk_size: int = 64,
        use_exp2: bool = False,
    ) -> torch.Tensor:
        from minipi05.stm import stm_chunked_bilinear_ext as _cext

        if (
            q.is_cuda
            and _cext.extension_available()
            and chunk_size >= 1
            and q.dtype == torch.float32
        ):
            ch = _cext.effective_cuda_chunk_size(q.shape[1], chunk_size)
            return _cext.stm_causal_bilinear_fwd_cuda_ext(
                q, k, u, r, v, g,
                chunk_size=ch,
                use_exp2=use_exp2,
            )
        raise RuntimeError(
            "STM CUDA inference needs the stm_chunked_bilinear JIT extension (CUDA, fp32) or "
            "Triton + FLA; use stm_causal_bilinear_fwd_torch on CPU / no_CUDA."
        )

    def stm_causal_bilinear_fwd_trainable(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        chunk_size: int = 64,
        use_exp2: bool = False,
        backward_recompute: str = "dense",
        pytorch_chunk_for_backward: int | None = None,
    ) -> torch.Tensor:
        _ = backward_recompute, pytorch_chunk_for_backward
        return stm_causal_bilinear_fwd_torch_chunked(
            q, k, u, r, v, g, chunk_size=chunk_size, use_exp2=use_exp2,
        )
