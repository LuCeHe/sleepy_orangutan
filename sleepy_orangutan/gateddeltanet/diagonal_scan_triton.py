"""
Triton forward + backward for the diagonal linear scan ``h_t = λ_t h_{t-1} + c_t`` (elementwise).

Forward uses the log-prefix closed form (see module history). Backward is the **reverse-time** adjoint
of the **equivalent recurrence** (same ``h`` as closed form):

``∂L/∂c_t = \bar h_t``, ``∂L/∂λ_t = h_{t-1} \bar h_t``, ``\bar h_{t-1} += λ_t \bar h_t``, with ``h_{-1}=h_0``.

CUDA + Triton only; forward/bwd accumulation in **float32** (``diagonal_linear_scan`` PyTorch path uses **float64**
for fp16/bf16/fp32 I/O — disable Triton via ``use_triton_scan=False`` or unset ``THEPEBBLETRAIL_DIAG_SCAN_TRITON`` if you need that match).

Use :func:`diagonal_linear_scan_triton` for autograd; :func:`diagonal_linear_scan_triton_fwd` for
inference-only (no graph).
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[misc, assignment]
    tl = None  # type: ignore[misc, assignment]
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def diagonal_scan_fwd_kernel(
        lam_ptr,
        c_ptr,
        h0_ptr,
        out_ptr,
        T,
        H: tl.constexpr,
        STRIDE_L_B: tl.constexpr,
        STRIDE_L_T: tl.constexpr,
        STRIDE_L_H: tl.constexpr,
        H0_SB: tl.constexpr,
        H0_SH: tl.constexpr,
    ) -> None:
        """One program per ``(batch, hidden)``; sequential scan over ``t`` in float32."""
        pid = tl.program_id(0)
        b = pid // H
        ih = pid - b * H

        lp = 0.0
        acc = 0.0
        h0v = tl.load(h0_ptr + b * H0_SB + ih * H0_SH).to(tl.float32)
        LP_FLOOR: tl.constexpr = -85.0

        for tt in tl.range(0, T):
            idx = b * STRIDE_L_B + tt * STRIDE_L_T + ih * STRIDE_L_H
            lam_t = tl.load(lam_ptr + idx).to(tl.float32)
            c_t = tl.load(c_ptr + idx).to(tl.float32)
            lam_t = tl.maximum(lam_t, 1e-30)
            lp = lp + tl.log(lam_t)
            lp_eff = tl.maximum(lp, LP_FLOOR)
            inv_p = tl.exp(-lp_eff)
            acc = acc + c_t * inv_p
            p = tl.exp(lp)
            h_t = p * h0v + p * acc
            tl.store(out_ptr + idx, h_t)

    @triton.jit
    def diagonal_scan_bwd_kernel(
        lam_ptr,
        h_saved_ptr,
        h0_ptr,
        grad_out_ptr,
        d_lam_ptr,
        d_c_ptr,
        dh0_ptr,
        T,
        H: tl.constexpr,
        STRIDE_L_B: tl.constexpr,
        STRIDE_L_T: tl.constexpr,
        STRIDE_L_H: tl.constexpr,
        H0_SB: tl.constexpr,
        H0_SH: tl.constexpr,
    ) -> None:
        """Reverse-time adjoint for ``h_t = λ_t h_{t-1} + c_t``; one program per ``(batch, hidden)``."""
        pid = tl.program_id(0)
        b = pid // H
        ih = pid - b * H

        h0v = tl.load(h0_ptr + b * H0_SB + ih * H0_SH).to(tl.float32)
        g = 0.0

        for k in tl.range(0, T):
            t = T - 1 - k
            idx = b * STRIDE_L_B + t * STRIDE_L_T + ih * STRIDE_L_H
            go = tl.load(grad_out_ptr + idx).to(tl.float32)
            lam_t = tl.load(lam_ptr + idx).to(tl.float32)
            g = g + go
            idx_pm1 = b * STRIDE_L_B + (t - 1) * STRIDE_L_T + ih * STRIDE_L_H
            h_prev = tl.load(h_saved_ptr + idx_pm1, mask=t > 0, other=h0v).to(tl.float32)
            tl.store(d_c_ptr + idx, g)
            tl.store(d_lam_ptr + idx, h_prev * g)
            g = lam_t * g

        tl.store(dh0_ptr + b * H0_SB + ih * H0_SH, g)


class DiagonalLinearScanTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, lam: torch.Tensor, c: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        if lam.device.type != "cuda":
            raise RuntimeError("diagonal_linear_scan_triton requires CUDA")
        lam_c = lam.contiguous()
        c_c = c.contiguous()
        h0_c = h0.contiguous()
        b, t, hdim = lam_c.shape
        sb, st, sh = lam_c.stride()
        h0_sb, h0_sh = h0_c.stride()
        if sh != 1 or h0_sh != 1:
            raise ValueError("lam and h0 must be contiguous in the last (H) dimension")

        out_f32 = torch.empty((b, t, hdim), device=lam.device, dtype=torch.float32)
        if b * hdim > 0:
            grid = (b * hdim,)
            diagonal_scan_fwd_kernel[grid](
                lam_c, c_c, h0_c, out_f32, t, hdim, sb, st, sh, h0_sb, h0_sh,
            )
        out = out_f32.to(lam.dtype)
        ctx.save_for_backward(lam_c, h0_c, out_f32)
        ctx.sb, ctx.st, ctx.sh = sb, st, sh
        ctx.h0_sb, ctx.h0_sh = h0_sb, h0_sh
        ctx.t_len = int(t)
        ctx.bsz = int(b)
        ctx.hdim = int(hdim)
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lam_c, h0_c, out_f32 = ctx.saved_tensors
        sb, st, sh = ctx.sb, ctx.st, ctx.sh
        h0_sb, h0_sh = ctx.h0_sb, ctx.h0_sh
        b, t, hdim = ctx.bsz, ctx.t_len, ctx.hdim
        grad_out_c = grad_out.contiguous().to(torch.float32)

        d_lam = torch.zeros((b, t, hdim), device=lam_c.device, dtype=torch.float32)
        d_c = torch.zeros((b, t, hdim), device=lam_c.device, dtype=torch.float32)
        dh0 = torch.zeros_like(h0_c, dtype=torch.float32)

        if b * hdim == 0:
            return (
                d_lam.to(lam_c.dtype),
                d_c.to(lam_c.dtype),
                dh0.to(h0_c.dtype),
            )
        grid = (b * hdim,)
        diagonal_scan_bwd_kernel[grid](
            lam_c,
            out_f32,
            h0_c,
            grad_out_c,
            d_lam,
            d_c,
            dh0,
            t,
            hdim,
            sb,
            st,
            sh,
            h0_sb,
            h0_sh,
        )

        return (
            d_lam.to(lam_c.dtype),
            d_c.to(lam_c.dtype),
            dh0.to(h0_c.dtype),
        )


def diagonal_linear_scan_triton_fwd(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """
    ``lam``, ``c``: ``(B, T, H)`` row-major contiguous in the last dim; ``h0``: ``(B, H)`` contiguous.
    Returns ``h`` with same dtype/device as ``lam`` (no autograd).
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed; cannot run diagonal_linear_scan_triton_fwd")
    if lam.device.type != "cuda":
        raise RuntimeError("diagonal_linear_scan_triton_fwd requires CUDA tensors")
    if lam.shape != c.shape or lam.dim() != 3:
        raise ValueError("lam and c must be (B, T, H)")
    b, t, hdim = lam.shape
    if h0.shape != (b, hdim):
        raise ValueError(f"h0 shape {h0.shape} expected ({b}, {hdim})")

    lam_c = lam.contiguous()
    c_c = c.contiguous()
    h0_c = h0.contiguous()

    sb, st, sh = lam_c.stride()
    if sh != 1:
        raise ValueError("lam must be contiguous in H (last dimension stride 1)")
    h0_sb, h0_sh = h0_c.stride()
    if h0_sh != 1:
        raise ValueError("h0 must be contiguous in H")

    out_f32 = torch.empty((b, t, hdim), device=lam.device, dtype=torch.float32)

    if b * hdim > 0:
        grid = (b * hdim,)
        diagonal_scan_fwd_kernel[grid](
            lam_c,
            c_c,
            h0_c,
            out_f32,
            t,
            hdim,
            sb,
            st,
            sh,
            h0_sb,
            h0_sh,
        )
    return out_f32.to(lam.dtype)


def diagonal_linear_scan_triton(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Same as forward kernel but supports autograd (CUDA + Triton only)."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    return DiagonalLinearScanTritonFn.apply(lam, c, h0)


def is_diagonal_triton_fwd_available() -> bool:
    """True if Triton is installed (CUDA kernel can be launched; does not require env flag)."""
    return _TRITON_AVAILABLE


def triton_scan_enabled() -> bool:
    return _TRITON_AVAILABLE and os.environ.get("THEPEBBLETRAIL_DIAG_SCAN_TRITON", "").strip() == "1"


def maybe_diagonal_linear_scan_triton(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
    *,
    force: bool = False,
) -> torch.Tensor | None:
    """If CUDA + Triton: scan kernel; else ``None``.

    By default requires env ``THEPEBBLETRAIL_DIAG_SCAN_TRITON=1``. Pass ``force=True`` to skip the env
    gate (e.g. :class:`DiagonalCausalConvRNN` with ``use_triton_scan=True``).
    """
    if not _TRITON_AVAILABLE:
        return None
    if not force and not triton_scan_enabled():
        return None
    if lam.device.type != "cuda":
        return None
    if lam.requires_grad or c.requires_grad or h0.requires_grad:
        return diagonal_linear_scan_triton(lam, c, h0)
    return diagonal_linear_scan_triton_fwd(lam, c, h0)


__all__ = [
    "DiagonalLinearScanTritonFn",
    "diagonal_linear_scan_triton",
    "diagonal_linear_scan_triton_fwd",
    "is_diagonal_triton_fwd_available",
    "maybe_diagonal_linear_scan_triton",
    "triton_scan_enabled",
]
