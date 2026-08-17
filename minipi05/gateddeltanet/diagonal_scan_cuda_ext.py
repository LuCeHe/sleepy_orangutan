"""
Raw CUDA (non-Triton) forward + backward for the diagonal linear scan ``h_t = λ_t h_{t-1} + c_t``.

FP32 entry points match :mod:`diagonal_scan_triton` (``LP_FLOOR=-85``, ``λ`` floored at ``1e-30``).
FP64 kernels use ``finfo``-style ``tiny`` and ``log(tiny)`` like :func:`diagonal_linear_scan` / ``_diagonal_linear_scan_segment``.
Built JIT via :func:`torch.utils.cpp_extension.load` on first use; if compilation fails, callers should fall back.

Use :func:`diagonal_linear_scan_cuda` (fp32) or :func:`diagonal_linear_scan_cuda_fp64` for autograd;
:func:`diagonal_linear_scan_cuda_fwd` / :func:`diagonal_linear_scan_cuda_fp64_fwd` for inference only.
:func:`diagonal_linear_scan_cuda_fp64_chunked` runs FP64 CUDA segments of ``chunk_size`` with carried state.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXT = None
_EXT_BUILD_FAILED = False
_EXT_ERR: Optional[Exception] = None
# Indices for which :func:`warmup_diagonal_linear_scan_cuda` already ran (per process).
_CUDA_SCAN_WARMED_DEVICES: set[int] = set()


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _src_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "csrc", "diagonal_linear_scan")


def _load_ext():
    global _EXT, _EXT_BUILD_FAILED, _EXT_ERR
    if _EXT is not None or _EXT_BUILD_FAILED:
        return _EXT
    if not torch.cuda.is_available():
        _EXT_BUILD_FAILED = True
        return None
    src = _src_dir()
    sources = [
        os.path.join(src, "diagonal_linear_scan.cpp"),
        os.path.join(src, "diagonal_linear_scan_cuda.cu"),
    ]
    try:
        _EXT = load(
            name="diagonal_linear_scan_cuda_ext",
            sources=sources,
            with_cuda=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.environ.get("THEPEBBLETRAIL_DIAG_CUDA_VERBOSE", "").strip() == "1",
        )
        return _EXT
    except Exception as e:
        _EXT_BUILD_FAILED = True
        _EXT_ERR = e
        return None


def extension_available() -> bool:
    return _load_ext() is not None


def extension_error() -> Optional[Exception]:
    _load_ext()
    return _EXT_ERR


class DiagonalLinearScanCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        lam: torch.Tensor,
        c: torch.Tensor,
        h0: torch.Tensor,
    ) -> torch.Tensor:
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(f"diagonal_linear_scan_cuda_ext not available: {_EXT_ERR!r}")
        if lam.device.type != "cuda":
            raise RuntimeError("diagonal_linear_scan_cuda requires CUDA")
        lam_c = lam.contiguous()
        c_c = c.contiguous()
        h0_c = h0.contiguous()
        lam_f = lam_c.float()
        c_f = c_c.float()
        h0_f = h0_c.float()
        out_f32 = ext.diag_scan_fwd(lam_f, c_f, h0_f)
        ctx.save_for_backward(lam_f, h0_f, out_f32)
        ctx.lam_dtype = lam.dtype
        ctx.c_dtype = c.dtype
        ctx.h0_dtype = h0.dtype
        return out_f32.to(lam.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ext = _load_ext()
        lam_f, h0_f, out_f32 = ctx.saved_tensors
        go = grad_out.contiguous().float()
        d_lam, d_c, dh0 = ext.diag_scan_bwd(lam_f, out_f32, h0_f, go)
        return (
            d_lam.to(ctx.lam_dtype),
            d_c.to(ctx.c_dtype),
            dh0.to(ctx.h0_dtype),
        )


class DiagonalLinearScanCudaV2Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        lam: torch.Tensor,
        c: torch.Tensor,
        h0: torch.Tensor,
    ) -> torch.Tensor:
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(f"diagonal_linear_scan_cuda_ext not available: {_EXT_ERR!r}")
        if lam.device.type != "cuda":
            raise RuntimeError("diagonal_linear_scan_cuda_v2 requires CUDA")
        lam_c = lam.contiguous()
        c_c = c.contiguous()
        h0_c = h0.contiguous()
        lam_f = lam_c.float()
        c_f = c_c.float()
        h0_f = h0_c.float()
        out_f32 = ext.diag_scan_fwd_v2(lam_f, c_f, h0_f)
        ctx.save_for_backward(lam_f, h0_f, out_f32)
        ctx.lam_dtype = lam.dtype
        ctx.c_dtype = c.dtype
        ctx.h0_dtype = h0.dtype
        return out_f32.to(lam.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ext = _load_ext()
        lam_f, h0_f, out_f32 = ctx.saved_tensors
        go = grad_out.contiguous().float()
        d_lam, d_c, dh0 = ext.diag_scan_bwd_v2(lam_f, out_f32, h0_f, go)
        return (
            d_lam.to(ctx.lam_dtype),
            d_c.to(ctx.c_dtype),
            dh0.to(ctx.h0_dtype),
        )


def diagonal_linear_scan_cuda_fwd(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """``(B,T,H)`` scan; fp16/bf16/fp32 I/O promoted to fp32 inside; returns same dtype as ``lam``."""
    ext = _load_ext()
    if ext is None:
        raise RuntimeError(f"diagonal_linear_scan_cuda_fwd: extension not built: {_EXT_ERR!r}")
    if lam.device.type != "cuda":
        raise RuntimeError("diagonal_linear_scan_cuda_fwd requires CUDA")
    if lam.shape != c.shape or lam.dim() != 3:
        raise ValueError("lam and c must be (B, T, H)")
    b, t, hdim = lam.shape
    if h0.shape != (b, hdim):
        raise ValueError(f"h0 shape {h0.shape} expected ({b}, {hdim})")
    lam_f = lam.contiguous().float()
    c_f = c.contiguous().float()
    h0_f = h0.contiguous().float()
    out_f32 = ext.diag_scan_fwd(lam_f, c_f, h0_f)
    return out_f32.to(lam.dtype)


def diagonal_linear_scan_cuda(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Autograd wrapper (CUDA + compiled extension only)."""
    if _load_ext() is None:
        raise RuntimeError(f"diagonal_linear_scan_cuda: extension not built: {_EXT_ERR!r}")
    return DiagonalLinearScanCudaFn.apply(lam, c, h0)


def diagonal_linear_scan_cuda_v2(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Autograd wrapper using the v2 fp32 CUDA kernels (T<=512 uses time-parallel scan)."""
    if _load_ext() is None:
        raise RuntimeError(f"diagonal_linear_scan_cuda_v2: extension not built: {_EXT_ERR!r}")
    return DiagonalLinearScanCudaV2Fn.apply(lam, c, h0)


class DiagonalLinearScanCudaFp64Fn(torch.autograd.Function):
    """FP64 CUDA scan; numerics aligned with :func:`diagonal_linear_scan` fp64 segment (``tiny`` / ``log(tiny)``)."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        lam: torch.Tensor,
        c: torch.Tensor,
        h0: torch.Tensor,
    ) -> torch.Tensor:
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(f"diagonal_linear_scan_cuda_fp64: extension not built: {_EXT_ERR!r}")
        if lam.device.type != "cuda":
            raise RuntimeError("diagonal_linear_scan_cuda_fp64 requires CUDA")
        lam64 = lam.contiguous().double()
        c64 = c.contiguous().double()
        h64 = h0.contiguous().double()
        out64 = ext.diag_scan_fwd_fp64(lam64, c64, h64)
        ctx.save_for_backward(lam64, h64, out64)
        ctx.lam_dtype = lam.dtype
        ctx.c_dtype = c.dtype
        ctx.h0_dtype = h0.dtype
        return out64.to(lam.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ext = _load_ext()
        lam64, h64, out64 = ctx.saved_tensors
        go = grad_out.contiguous().double()
        d_lam, d_c, dh0 = ext.diag_scan_bwd_fp64(lam64, out64, h64, go)
        return (
            d_lam.to(ctx.lam_dtype),
            d_c.to(ctx.c_dtype),
            dh0.to(ctx.h0_dtype),
        )


def diagonal_linear_scan_cuda_fp64_fwd(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Inference-only FP64 scan; I/O cast to/from ``lam`` dtype."""
    ext = _load_ext()
    if ext is None:
        raise RuntimeError(f"diagonal_linear_scan_cuda_fp64_fwd: extension not built: {_EXT_ERR!r}")
    if lam.device.type != "cuda":
        raise RuntimeError("diagonal_linear_scan_cuda_fp64_fwd requires CUDA")
    lam64 = lam.contiguous().double()
    c64 = c.contiguous().double()
    h64 = h0.contiguous().double()
    out64 = ext.diag_scan_fwd_fp64(lam64, c64, h64)
    return out64.to(lam.dtype)


def diagonal_linear_scan_cuda_fp64(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Autograd FP64 CUDA scan (full ``T`` in one kernel)."""
    if _load_ext() is None:
        raise RuntimeError(f"diagonal_linear_scan_cuda_fp64: extension not built: {_EXT_ERR!r}")
    return DiagonalLinearScanCudaFp64Fn.apply(lam, c, h0)


def diagonal_linear_scan_cuda_fp64_chunked(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """FP64 CUDA scan with time chunking; autograd chains through per-segment :class:`DiagonalLinearScanCudaFp64Fn`."""
    if _load_ext() is None:
        raise RuntimeError(f"diagonal_linear_scan_cuda_fp64_chunked: extension not built: {_EXT_ERR!r}")
    if lam.device.type != "cuda":
        raise RuntimeError("diagonal_linear_scan_cuda_fp64_chunked requires CUDA")
    lam_o = lam.contiguous()
    c_o = c.contiguous()
    h0_o = h0.contiguous()
    lam64 = lam_o.double()
    c64 = c_o.double()
    h0_64 = h0_o.double()
    _b, t, _h = lam64.shape
    cs = int(chunk_size)
    if cs <= 0:
        raise ValueError("chunk_size must be positive")
    state = h0_64
    outs: list[torch.Tensor] = []
    for s in range(0, t, cs):
        e = min(s + cs, t)
        h_seg = DiagonalLinearScanCudaFp64Fn.apply(lam64[:, s:e], c64[:, s:e], state)
        outs.append(h_seg)
        state = h_seg[:, -1, :]
    return torch.cat(outs, dim=1).to(lam_o.dtype)


def is_diagonal_cuda_scan_available() -> bool:
    """True if CUDA is available and the extension compiled successfully."""
    if not torch.cuda.is_available():
        return False
    return _load_ext() is not None


def warmup_diagonal_linear_scan_cuda(
    device: torch.device,
    *,
    repeats: int = 4,
    force: bool = False,
) -> None:
    """
    Pay one-time JIT link cost (if any) and prime CUDA kernels with a few tiny forward+backward passes.

    By default runs once per CUDA device index per process unless ``force=True``.
    """
    if device.type != "cuda" or _load_ext() is None:
        return
    dev_i = _cuda_device_index(device)
    if not force and dev_i in _CUDA_SCAN_WARMED_DEVICES:
        return
    b, t, hdim = 1, 48, 32
    lam = torch.sigmoid(torch.randn(b, t, hdim, device=device, dtype=torch.float32))
    c = torch.randn(b, t, hdim, device=device, dtype=torch.float32)
    h0 = torch.zeros(b, hdim, device=device, dtype=torch.float32)
    for _ in range(max(1, int(repeats))):
        la = lam.clone().requires_grad_(True)
        cc = c.clone().requires_grad_(True)
        hh = h0.clone().requires_grad_(True)
        diagonal_linear_scan_cuda(la, cc, hh).sum().backward()
    lam64 = torch.sigmoid(torch.randn(b, t, hdim, device=device, dtype=torch.float64))
    c64 = torch.randn(b, t, hdim, device=device, dtype=torch.float64)
    h064 = torch.zeros(b, hdim, device=device, dtype=torch.float64)
    for _ in range(max(1, int(repeats) // 2)):
        la = lam64.clone().requires_grad_(True)
        cc = c64.clone().requires_grad_(True)
        hh = h064.clone().requires_grad_(True)
        diagonal_linear_scan_cuda_fp64(la, cc, hh).sum().backward()
        la2 = lam64.clone().requires_grad_(True)
        cc2 = c64.clone().requires_grad_(True)
        hh2 = h064.clone().requires_grad_(True)
        diagonal_linear_scan_cuda_fp64_chunked(la2, cc2, hh2, 16).sum().backward()
    torch.cuda.synchronize()
    _CUDA_SCAN_WARMED_DEVICES.add(dev_i)


__all__ = [
    "DiagonalLinearScanCudaFn",
    "DiagonalLinearScanCudaFp64Fn",
    "diagonal_linear_scan_cuda",
    "diagonal_linear_scan_cuda_fp64",
    "diagonal_linear_scan_cuda_fp64_chunked",
    "diagonal_linear_scan_cuda_fp64_fwd",
    "diagonal_linear_scan_cuda_fwd",
    "extension_available",
    "extension_error",
    "is_diagonal_cuda_scan_available",
    "warmup_diagonal_linear_scan_cuda",
]
