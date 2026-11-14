# SPDX-License-Identifier: MIT
"""
Chunked tensor delta rule via the WY / UT transform.

Matches :func:`stm_delta_recurrent_fwd` (pairwise STM gate ``Γ_ts = exp(g_t - g_s)``)
but parallelizes each length-``BT`` chunk like Yang et al., *Parallelizing Linear
Transformers with the Delta Rule over Sequence Length* (2024):

1. Form the strictly-lower Householder Gram
   ``L_{ts} = β_t (k_t·k_s)(r_t·r_s) Γ_{ts}`` (Kronecker key ``k⊗r``, no flatten).
2. **UT transform:** ``A = (I + L)^{-1}`` by triangular solve (all chunks in parallel).
3. Apply ``A`` to the scaled values / keys, then a short inter-chunk state scan
   (``T / BT`` steps).

On CUDA the intra-chunk UT (steps 1–2) is a Triton kernel. Backward recomputes
the PyTorch chunked form (same math, batched GEMMs). ``du == 1`` on CUDA uses
FLA ``chunk_gated_delta_rule`` (the paper's matrix kernels). Notes ``stmfla``
flatten ``k⊗r`` / ``q⊗u`` into those same matrix kernels when ``du > 1``.
Notes ``stmfused`` uses FLA ``fused_recurrent_gated_delta_rule`` instead of the
chunk kernel (also auto-selected at inference when ``T≤64``, like GDN).
``gate_mode="pairwise"`` (``stmdelta2``) converts STM pairwise ``g`` via
``g_t − g_{t-1}`` before the FLA/chunk cumsum. ``gate_mode="per_step"``
(``stmdelta3``) passes ``g`` through as GDN per-step log-decay.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

STM_DELTA_UT_CHUNK_SIZES = (16, 32, 64, 128)
STM_DELTA_UT_AVAILABLE: bool = False
_STM_DELTA_UT_SCAN_COMPILED = None
# Match FLA GDN: fused recurrent kernel when the sequence is short enough.
_FLA_FUSED_MAX_T = 64


def stm_delta_ut_chunk_size(requested: int | None = 64) -> int:
    """Largest supported UT tile in ``{16, 32, 64, 128}`` at most ``requested``."""
    req = 64 if requested is None else max(1, int(requested))
    for bt in STM_DELTA_UT_CHUNK_SIZES[::-1]:
        if req >= bt:
            return bt
    return 16


def _stm_delta_ut_scan_fn(*, return_state: bool = False):
    """``torch.compile`` the inter-chunk scan when enabled (``STM_DELTA_UT_COMPILE!=0``).

    ``return_state=True`` (cache prefill) uses the eager scan so the compiled
    graph is not traced with a different return arity.
    """
    if return_state:
        return _stm_delta_ut_scan
    global _STM_DELTA_UT_SCAN_COMPILED
    if _STM_DELTA_UT_SCAN_COMPILED is not None:
        return _STM_DELTA_UT_SCAN_COMPILED
    fn = _stm_delta_ut_scan
    if os.environ.get("STM_DELTA_UT_COMPILE", "1") != "0":
        try:
            fn = torch.compile(_stm_delta_ut_scan, fullgraph=False, dynamic=True)
        except Exception:
            fn = _stm_delta_ut_scan
    _STM_DELTA_UT_SCAN_COMPILED = fn
    return fn


def stm_g_to_per_step(g: torch.Tensor | None) -> torch.Tensor | None:
    """STM pairwise ``g`` → per-step log-decay ``δ_0 = 0``, ``δ_t = g_t - g_{t-1}``."""
    if g is None:
        return None
    step = torch.zeros_like(g)
    step[:, 1:] = g[:, 1:] - g[:, :-1]
    return step


def _normalize_gate_mode(gate_mode: str) -> str:
    mode = str(gate_mode).strip().lower().replace("-", "_")
    if mode not in ("pairwise", "per_step"):
        raise ValueError('gate_mode must be "pairwise" or "per_step"')
    return mode


def _stm_delta_local_gate(
    g: torch.Tensor | None,
    *,
    B: int,
    H: int,
    n_chunks: int,
    chunk: int,
    device: torch.device,
    gate_mode: str,
) -> torch.Tensor:
    """Chunk-local cumulative log-decay ``lg`` shaped ``(B, H, n, C)``."""
    if g is None:
        return torch.zeros(B, H, n_chunks, chunk, device=device, dtype=torch.float32)
    gf = g.float()
    if gate_mode == "pairwise":
        step = stm_g_to_per_step(gf)
        assert step is not None
        gf = step
    return gf.view(B, n_chunks, chunk, H).cumsum(dim=2).permute(0, 3, 1, 2).contiguous()


def _exp_gate(x: torch.Tensor, *, use_exp2: bool) -> torch.Tensor:
    return torch.exp2(x) if use_exp2 else torch.exp(x)


def _pad_time(x: torch.Tensor, pad: int, *, repeat_last: bool = False) -> torch.Tensor:
    if pad <= 0:
        return x
    if repeat_last:
        return torch.cat([x, x[:, -1:].expand(x.size(0), pad, *x.shape[2:])], dim=1)
    return torch.cat([x, x.new_zeros(x.size(0), pad, *x.shape[2:])], dim=1)


def _bthnc(x: torch.Tensor, n_chunks: int, chunk: int, n_heads: int):
    """``(B, n*C, H, ...)`` → ``(B, H, n, C, ...)``."""
    B = x.shape[0]
    tail = x.shape[3:]
    return x.view(B, n_chunks, chunk, n_heads, *tail).permute(0, 3, 1, 2, *range(4, 4 + len(tail)))


def _solve_ut(L: torch.Tensor) -> torch.Tensor:
    """``A = (I + L)^{-1}`` for strictly-lower ``L`` shaped ``(..., C, C)``."""
    chunk = L.shape[-1]
    eye = torch.eye(chunk, device=L.device, dtype=L.dtype)
    rhs = eye.expand_as(L).contiguous()
    return torch.linalg.solve_triangular(L + eye, rhs, upper=False)


def _kronecker_time_keys(k: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``k`` ``(B,H,C,dk)``, ``r`` ``(B,H,C,du)`` → ``(B*H, C, dk*du)``."""
    B, H, C, dk = k.shape
    du = r.shape[-1]
    return (k.unsqueeze(-1) * r.unsqueeze(-2)).reshape(B * H, C, dk * du)


def _stm_delta_ut_scan(
    q4: torch.Tensor,
    k4: torch.Tensor,
    u4: torch.Tensor,
    r4: torch.Tensor,
    A: torch.Tensor,
    u_wy: torch.Tensor,
    lg: torch.Tensor,
    scale: torch.Tensor,
    *,
    use_exp2: bool,
    out_dtype: torch.dtype,
    return_state: bool = False,
):
    """Inter-chunk tensor-state scan after the intra-chunk UT solve.

    Same math as the 5D ``S∈R^{dk×du×dv}`` recurrence, but the state is stored
    flat as ``(B*H, dk*du, dv)`` so each chunk is batched GEMMs (``k⊗r`` rows)
    instead of ``einsum`` over a 5-tensor. Intra-chunk Gram stays factored
    (``(k·k)(r·r)``) in the UT solve; only this scan flattens.

    ``*4`` / ``A`` / ``u_wy`` / ``lg`` / ``scale`` are ``(B, H, n, C, ...)``.
    Returns ``o`` as ``(B, n*C, H, dv)`` (padded time).
    """
    B, H, n_chunks, chunk, dk = k4.shape
    du = r4.shape[-1]
    dv = u_wy.shape[-1]
    bh = B * H
    state_m = dk * du
    S = q4.new_zeros(bh, state_m, dv)
    o4 = q4.new_empty(B, H, n_chunks, chunk, dv)
    causal = torch.ones(chunk, chunk, device=q4.device, dtype=torch.bool).tril()

    for i in range(n_chunks):
        lg_i = lg[:, :, i]
        exp_lg = _exp_gate(lg_i, use_exp2=use_exp2)
        k_i = k4[:, :, i]
        r_i = r4[:, :, i]
        q_i = q4[:, :, i]
        u_i = u4[:, :, i]
        kr = _kronecker_time_keys(k_i, r_i)
        qu = _kronecker_time_keys(q_i, u_i)

        pred_S = torch.bmm(kr, S).view(B, H, chunk, dv)
        v_prime = torch.matmul(A[:, :, i], pred_S * scale[:, :, i].unsqueeze(-1))
        v_new = u_wy[:, :, i] - v_prime

        o_inter = torch.bmm(qu, S).view(B, H, chunk, dv) * exp_lg.unsqueeze(-1)
        qk = torch.matmul(q_i, k_i.transpose(-1, -2))
        ur = torch.matmul(u_i, r_i.transpose(-1, -2))
        scores = qk * ur * _exp_gate(lg_i.unsqueeze(-1) - lg_i.unsqueeze(-2), use_exp2=use_exp2)
        scores = scores.masked_fill(~causal, 0)
        o4[:, :, i] = o_inter + torch.matmul(scores, v_new)

        decay = _exp_gate(lg_i[:, :, -1:] - lg_i, use_exp2=use_exp2)
        S = S * _exp_gate(lg_i[:, :, -1:], use_exp2=use_exp2).reshape(bh, 1, 1)
        inc = (decay.unsqueeze(-1) * v_new).reshape(bh, chunk, dv)
        S = S + torch.bmm(kr.transpose(1, 2), inc)

    o = o4.permute(0, 2, 3, 1, 4).reshape(B, n_chunks * chunk, H, dv).to(dtype=out_dtype)
    if return_state:
        return o, S.view(B, H, state_m, dv)
    return o


def stm_delta_chunk_ut_fwd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    use_exp2: bool = False,
    gate_mode: str = "pairwise",
    return_state: bool = False,
):
    """Chunked tensor delta rule (Yang et al. UT / WY), PyTorch reference.

    Shapes match :func:`stm_delta_recurrent_fwd`: ``q,k`` ``(B,T,H,dk)``,
    ``u,r`` ``(B,T,H,du)``, ``v`` ``(B,T,H,dv)``, ``g,beta`` ``(B,T,H)``.
    ``gate_mode`` is ``"pairwise"`` (``stmdelta2``) or ``"per_step"`` (``stmdelta3``).
    """
    gate_mode = _normalize_gate_mode(gate_mode)
    chunk = stm_delta_ut_chunk_size(chunk_size)
    B, T, H, _dk = q.shape
    pad = (chunk - (T % chunk)) % chunk
    T_pad = T + pad

    q = _pad_time(q, pad)
    k = _pad_time(k, pad)
    u = _pad_time(u, pad)
    r = _pad_time(r, pad)
    v = _pad_time(v, pad)
    beta = _pad_time(beta, pad)
    g_pad = None if g is None else _pad_time(
        g, pad, repeat_last=(gate_mode == "pairwise"),
    )

    qf = q.float()
    kf = k.float()
    uf = u.float()
    rf = r.float()
    vf = v.float()
    bf = beta.float()

    n_chunks = T_pad // chunk
    q4 = _bthnc(qf, n_chunks, chunk, H)
    k4 = _bthnc(kf, n_chunks, chunk, H)
    u4 = _bthnc(uf, n_chunks, chunk, H)
    r4 = _bthnc(rf, n_chunks, chunk, H)
    v4 = _bthnc(vf, n_chunks, chunk, H)
    b4 = _bthnc(bf, n_chunks, chunk, H)

    lg = _stm_delta_local_gate(
        g_pad, B=B, H=H, n_chunks=n_chunks, chunk=chunk,
        device=q.device, gate_mode=gate_mode,
    )

    kk = torch.einsum("bhnck,bhnsk->bhncs", k4, k4)
    rr = torch.einsum("bhncd,bhnsd->bhncs", r4, r4)
    gamma = _exp_gate(lg.unsqueeze(-1) - lg.unsqueeze(-2), use_exp2=use_exp2)
    L = (b4.unsqueeze(-1) * kk * rr * gamma).tril(-1)
    A = _solve_ut(L)

    u_wy = torch.matmul(A, (b4.unsqueeze(-1) * v4))
    scale = b4 * _exp_gate(lg, use_exp2=use_exp2)
    scanned = _stm_delta_ut_scan_fn(return_state=return_state)(
        q4, k4, u4, r4, A, u_wy, lg, scale,
        use_exp2=use_exp2, out_dtype=q.dtype, return_state=return_state,
    )
    if return_state:
        o, state = scanned
        return o[:, :T], state
    return scanned[:, :T]


def _stm_kronecker_matrix_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``q⊗u`` / ``k⊗r`` into FLA matrix keys of last dim ``dk*du``."""
    B, T, H, dk = q.shape
    du = u.shape[-1]
    if du == 1:
        scale_q = u.squeeze(-1).unsqueeze(-1)
        scale_k = r.squeeze(-1).unsqueeze(-1)
        return q * scale_q, k * scale_k
    q_m = (q.unsqueeze(-1) * u.unsqueeze(-2)).reshape(B, T, H, dk * du)
    k_m = (k.unsqueeze(-1) * r.unsqueeze(-2)).reshape(B, T, H, dk * du)
    return q_m, k_m


def _stm_delta_fla_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    *,
    use_exp2: bool,
    gate_mode: str = "pairwise",
    flatten: bool = False,
    use_fused: bool = False,
    training: bool = True,
) -> torch.Tensor | None:
    """Matrix-state shortcut: FLA chunked or fused-recurrent gated delta."""
    if not flatten and (u.shape[-1] != 1 or r.shape[-1] != 1):
        return None
    if q.device.type != "cuda":
        return None
    gate_mode = _normalize_gate_mode(gate_mode)
    T = q.shape[1]
    want_fused = bool(use_fused) or (not training and T <= _FLA_FUSED_MAX_T)
    try:
        from sleepy_orangutan.gateddeltanet.fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except Exception:
        return None
    q_m, k_m = _stm_kronecker_matrix_qk(q, k, u, r)
    if g is None:
        g_step = q.new_zeros(q.shape[0], q.shape[1], q.shape[2])
    elif gate_mode == "per_step":
        g_step = g
    else:
        g_step = stm_g_to_per_step(g)
        assert g_step is not None
    fla_kw = dict(
        scale=1.0,
        use_qk_l2norm_in_kernel=False,
        use_exp2=use_exp2,
    )
    try:
        if want_fused:
            try:
                o, _ht = fused_recurrent_gated_delta_rule(
                    q_m, k_m, v, g=g_step, beta=beta, **fla_kw,
                )
            except Exception:
                o, _ht = chunk_gated_delta_rule(
                    q_m, k_m, v, g_step, beta, **fla_kw,
                )
        else:
            o, _ht = chunk_gated_delta_rule(
                q_m, k_m, v, g_step, beta, **fla_kw,
            )
    except Exception:
        return None
    return o


class _StmDeltaUTFn(torch.autograd.Function):
    """Triton (or PyTorch) UT forward; backward via PyTorch chunked recompute."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: torch.Tensor,
        flags: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, u, r, v, g, beta, chunk_size, flags)
        has_gate = bool(int(flags[0].item()))
        use_exp2 = bool(int(flags[1].item()))
        use_triton = bool(int(flags[2].item()))
        gate_mode = "per_step" if int(flags[3].item()) else "pairwise"
        bt = int(chunk_size.item())
        g_arg = g if has_gate else None
        if use_triton and STM_DELTA_UT_AVAILABLE and q.device.type == "cuda":
            return stm_delta_chunk_ut_fwd_triton(
                q, k, u, r, v, g_arg, beta,
                chunk_size=bt, use_exp2=use_exp2, gate_mode=gate_mode,
            )
        return stm_delta_chunk_ut_fwd_torch(
            q, k, u, r, v, g_arg, beta,
            chunk_size=bt, use_exp2=use_exp2, gate_mode=gate_mode,
        )

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):  # type: ignore[override]
        q, k, u, r, v, g, beta, chunk_size, flags = ctx.saved_tensors
        has_gate = bool(int(flags[0].item()))
        use_exp2 = bool(int(flags[1].item()))
        gate_mode = "per_step" if int(flags[3].item()) else "pairwise"
        bt = int(chunk_size.item())
        with torch.enable_grad():
            cq = q.detach().clone().requires_grad_(True)
            ck = k.detach().clone().requires_grad_(True)
            cu = u.detach().clone().requires_grad_(True)
            cr = r.detach().clone().requires_grad_(True)
            cv = v.detach().clone().requires_grad_(True)
            cb = beta.detach().clone().requires_grad_(True)
            cg = g.detach().clone().requires_grad_(True) if has_gate else None
            o = stm_delta_chunk_ut_fwd_torch(
                cq, ck, cu, cr, cv, cg, cb,
                chunk_size=bt, use_exp2=use_exp2, gate_mode=gate_mode,
            )
            want = (cq, ck, cu, cr, cv, cb) + ((cg,) if has_gate else ())
            grads = torch.autograd.grad(
                outputs=o,
                inputs=want,
                grad_outputs=grad_o,
                retain_graph=False,
                only_inputs=True,
                allow_unused=False,
            )
        dq, dk, du, dr, dv, db = grads[:6]
        dg = grads[6] if has_gate else torch.zeros_like(g)
        return dq, dk, du, dr, dv, dg, db, None, None


def stm_delta_chunk_ut_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    use_exp2: bool = False,
    use_triton: bool = True,
    gate_mode: str = "pairwise",
    return_state: bool = False,
    use_fla_matrix: bool = False,
    use_fla_fused: bool = False,
    training: bool = True,
):
    """Public UT-transform tensor delta. Triton intra-chunk solve on CUDA when enabled."""
    gate_mode = _normalize_gate_mode(gate_mode)
    flatten = bool(use_fla_matrix or use_fla_fused)
    try_fla = (not return_state) and (
        flatten or (use_triton and u.shape[-1] == 1 and r.shape[-1] == 1)
    )
    if try_fla:
        fla_o = _stm_delta_fla_matrix(
            q, k, u, r, v, g, beta, use_exp2=use_exp2, gate_mode=gate_mode,
            flatten=flatten or (u.shape[-1] == 1),
            use_fused=use_fla_fused,
            training=training,
        )
        if fla_o is not None:
            return fla_o

    q = q.contiguous()
    k = k.contiguous()
    u = u.contiguous()
    r = r.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    g = g.contiguous() if g is not None else None
    bt = stm_delta_ut_chunk_size(chunk_size)

    need_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, u, r, v, beta, g) if t is not None
    )
    if not need_grad:
        if (
            use_triton
            and STM_DELTA_UT_AVAILABLE
            and q.device.type == "cuda"
        ):
            return stm_delta_chunk_ut_fwd_triton(
                q, k, u, r, v, g, beta,
                chunk_size=bt, use_exp2=use_exp2, gate_mode=gate_mode,
                return_state=return_state,
            )
        return stm_delta_chunk_ut_fwd_torch(
            q, k, u, r, v, g, beta,
            chunk_size=bt, use_exp2=use_exp2, gate_mode=gate_mode,
            return_state=return_state,
        )

    gst = g if g is not None else q.new_zeros((0,))
    flags = torch.tensor(
        [int(g is not None), int(use_exp2), int(use_triton), int(gate_mode == "per_step")],
        device=q.device,
        dtype=torch.int64,
    )
    bt_t = torch.tensor(bt, device=q.device, dtype=torch.int64)
    return _StmDeltaUTFn.apply(q, k, u, r, v, gst, beta, bt_t, flags)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _stm_delta_ut_solve_kernel(
        k_ptr, r_ptr, g_ptr, beta_ptr, A_ptr,
        T, H, DK, DU,
        stride_kb, stride_kt, stride_kh, stride_kd,
        stride_rb, stride_rt, stride_rh, stride_rd,
        stride_gb, stride_gt, stride_gh,
        stride_bb, stride_bt, stride_bh,
        stride_ab, stride_at, stride_ah, stride_ac,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BU: tl.constexpr,
        USE_G: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        """Intra-chunk Kronecker Gram + UT solve: ``A = (I + L)^{-1}``.

        ``L_ts = β_t (k_t·k_s)(r_t·r_s) exp(g_t - g_s)`` for ``s < t`` (chunk-local ``g``).
        Output ``A`` is ``[B, T, H, BT]`` (FLA ``solve_tril`` layout).
        """
        i_t = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H

        t0 = i_t * BT
        o_i = tl.arange(0, BT)
        offs_t = t0 + o_i
        mask_t = offs_t < T

        b_kk = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(DK, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)
            mask_k = offs_k < DK
            k_ptrs = (
                k_ptr
                + i_b * stride_kb
                + i_h * stride_kh
                + offs_t[:, None] * stride_kt
                + offs_k[None, :] * stride_kd
            )
            b_k = tl.load(k_ptrs, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
            b_kk += tl.dot(b_k, tl.trans(b_k))

        b_rr = tl.zeros([BT, BT], dtype=tl.float32)
        for i_u in range(tl.cdiv(DU, BU)):
            offs_u = i_u * BU + tl.arange(0, BU)
            mask_u = offs_u < DU
            r_ptrs = (
                r_ptr
                + i_b * stride_rb
                + i_h * stride_rh
                + offs_t[:, None] * stride_rt
                + offs_u[None, :] * stride_rd
            )
            b_r = tl.load(r_ptrs, mask=mask_t[:, None] & mask_u[None, :], other=0.0)
            b_rr += tl.dot(b_r, tl.trans(b_r))

        b_A = b_kk * b_rr
        b_b = tl.load(
            beta_ptr + i_b * stride_bb + i_h * stride_bh + offs_t * stride_bt,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)

        m_d = (o_i[:, None] > o_i[None, :]) & mask_t[:, None] & mask_t[None, :]
        m_I = o_i[:, None] == o_i[None, :]
        if USE_G:
            b_g = tl.load(
                g_ptr + i_b * stride_gb + i_h * stride_gh + offs_t * stride_gt,
                mask=mask_t,
                other=0.0,
            ).to(tl.float32)
            dg = b_g[:, None] - b_g[None, :]
            if USE_EXP2:
                b_A = b_A * tl.where(m_d, tl.exp2(dg), 0.0)
            else:
                b_A = b_A * tl.where(m_d, tl.exp(dg), 0.0)
        else:
            b_A = tl.where(m_d, b_A, 0.0)

        b_A = b_A * b_b[:, None]

        b_Ai = -b_A
        for i in range(2, BT):
            b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A, 0.0), 0)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai = b_Ai + m_I.to(tl.float32)

        A_ptrs = (
            A_ptr
            + i_b * stride_ab
            + i_h * stride_ah
            + offs_t[:, None] * stride_at
            + o_i[None, :] * stride_ac
        )
        tl.store(A_ptrs, b_Ai, mask=mask_t[:, None])

    def _stm_delta_ut_solve_triton(
        k: torch.Tensor,
        r: torch.Tensor,
        g_local: torch.Tensor | None,
        beta: torch.Tensor,
        *,
        chunk_size: int,
        use_exp2: bool,
    ) -> torch.Tensor:
        B, T, H, dk = k.shape
        du = r.shape[-1]
        BT = chunk_size
        A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
        g_arg = g_local if g_local is not None else k
        NT = triton.cdiv(T, BT)
        _stm_delta_ut_solve_kernel[(NT, B * H)](
            k, r, g_arg, beta, A,
            T, H, dk, du,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            r.stride(0), r.stride(1), r.stride(2), r.stride(3),
            g_arg.stride(0), g_arg.stride(1), g_arg.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2),
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            BT=BT,
            BK=16,
            BU=16,
            USE_G=g_local is not None,
            USE_EXP2=use_exp2,
        )
        return A

    def stm_delta_chunk_ut_fwd_triton(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        beta: torch.Tensor,
        *,
        chunk_size: int = 64,
        use_exp2: bool = False,
        gate_mode: str = "pairwise",
        return_state: bool = False,
    ):
        """CUDA forward: Triton UT solve, then the same inter-chunk scan as the PyTorch path."""
        gate_mode = _normalize_gate_mode(gate_mode)
        chunk = stm_delta_ut_chunk_size(chunk_size)
        B, T, H, _dk = q.shape
        pad = (chunk - (T % chunk)) % chunk
        T_pad = T + pad

        q = _pad_time(q, pad)
        k = _pad_time(k, pad)
        u = _pad_time(u, pad)
        r = _pad_time(r, pad)
        v = _pad_time(v, pad)
        beta = _pad_time(beta, pad)
        g_pad = None if g is None else _pad_time(
            g, pad, repeat_last=(gate_mode == "pairwise"),
        )

        qf = q.float().contiguous()
        kf = k.float().contiguous()
        uf = u.float().contiguous()
        rf = r.float().contiguous()
        vf = v.float().contiguous()
        bf = beta.float().contiguous()

        n_chunks = T_pad // chunk
        lg = _stm_delta_local_gate(
            g_pad, B=B, H=H, n_chunks=n_chunks, chunk=chunk,
            device=q.device, gate_mode=gate_mode,
        )
        lg_bth = None if g_pad is None else lg.permute(0, 2, 3, 1).reshape(B, T_pad, H).contiguous()

        A_flat = _stm_delta_ut_solve_triton(
            kf, rf, lg_bth, bf,
            chunk_size=chunk, use_exp2=use_exp2,
        )
        A = A_flat.view(B, n_chunks, chunk, H, chunk).permute(0, 3, 1, 2, 4).contiguous()

        q4 = _bthnc(qf, n_chunks, chunk, H)
        k4 = _bthnc(kf, n_chunks, chunk, H)
        u4 = _bthnc(uf, n_chunks, chunk, H)
        r4 = _bthnc(rf, n_chunks, chunk, H)
        v4 = _bthnc(vf, n_chunks, chunk, H)
        b4 = _bthnc(bf, n_chunks, chunk, H)

        u_wy = torch.matmul(A, (b4.unsqueeze(-1) * v4))
        scale = b4 * _exp_gate(lg, use_exp2=use_exp2)
        scanned = _stm_delta_ut_scan_fn(return_state=return_state)(
            q4, k4, u4, r4, A, u_wy, lg, scale,
            use_exp2=use_exp2, out_dtype=q.dtype, return_state=return_state,
        )
        if return_state:
            o, state = scanned
            return o[:, :T], state
        return scanned[:, :T]

    STM_DELTA_UT_AVAILABLE = True

except Exception:  # pragma: no cover — Triton unavailable
    STM_DELTA_UT_AVAILABLE = False

    def stm_delta_chunk_ut_fwd_triton(
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        beta: torch.Tensor,
        *,
        chunk_size: int = 64,
        use_exp2: bool = False,
        gate_mode: str = "pairwise",
        return_state: bool = False,
    ):
        return stm_delta_chunk_ut_fwd_torch(
            q, k, u, r, v, g, beta,
            chunk_size=chunk_size, use_exp2=use_exp2, gate_mode=gate_mode,
            return_state=return_state,
        )


def _smoke() -> None:
    from sleepy_orangutan.stm.state_tensor_model import (
        stm_delta_recurrent_fwd,
    )

    torch.manual_seed(0)
    B, T, H, dk, du, dv = 2, 37, 2, 4, 4, 6
    q = F.normalize(torch.randn(B, T, H, dk), dim=-1)
    k = F.normalize(torch.randn(B, T, H, dk), dim=-1)
    u = F.normalize(torch.randn(B, T, H, du), dim=-1)
    r = F.normalize(torch.randn(B, T, H, du), dim=-1)
    v = torch.randn(B, T, H, dv)
    g = torch.randn(B, T, H) * 0.1
    beta = torch.rand(B, T, H)
    o_ref = stm_delta_recurrent_fwd(q, k, u, r, v, g, beta)
    o_ut = stm_delta_chunk_ut_fwd_torch(q, k, u, r, v, g, beta, chunk_size=16)
    err = (o_ref - o_ut).abs().max().item()
    assert err < 2e-4, err
    o_ref3 = stm_delta_recurrent_fwd(q, k, u, r, v, g, beta, gate_mode="per_step")
    o_ut3 = stm_delta_chunk_ut_fwd_torch(
        q, k, u, r, v, g, beta, chunk_size=16, gate_mode="per_step",
    )
    err3 = (o_ref3 - o_ut3).abs().max().item()
    assert err3 < 2e-4, err3
    print(f"stm_delta_ut smoke max |recurrent - chunked UT|: {err:.3e} (per_step {err3:.3e})")


if __name__ == "__main__":
    _smoke()
