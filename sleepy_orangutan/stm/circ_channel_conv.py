"""Circular (periodic) 1D conv along the last dimension.

Used by ``ConcOnlyC2`` (PyTorch), ``ConcOnlyC3`` (Triton), and ``ConcOnlyC4`` (CUDA).
``y[..., c] = sum_k w[k] * x[..., (c - pad + k) mod C]`` with ``pad = (KW-1)//2``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

_triton_failed = False
_triton_fail_printed = False


def circular_conv1d_last(x: Tensor, weight: Tensor) -> Tensor:
    """PyTorch circular conv along the last dim.

    Args:
        x: ``(..., C)``
        weight: ``(KW,)`` or ``(1, 1, KW)``, odd ``KW``
    """
    w = weight.reshape(1, 1, -1).to(dtype=x.dtype)
    k = int(w.shape[-1])
    pad = k // 2
    leading = x.shape[:-1]
    c = x.shape[-1]
    y = x.reshape(-1, 1, c)
    y = F.pad(y, (pad, pad), mode="circular")
    y = F.conv1d(y, w)
    return y.reshape(*leading, c)


if TRITON_AVAILABLE:

    @triton.jit
    def _circ_conv1d_fwd_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        C,
        stride_xn,
        stride_xc,
        stride_yn,
        stride_yc,
        KW: tl.constexpr,
        PAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = c_offs < C
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(KW):
            src = c_offs + (k - PAD)
            src = tl.where(src < 0, src + C, src)
            src = tl.where(src >= C, src - C, src)
            wv = tl.load(w_ptr + k)
            xv = tl.load(x_ptr + n * stride_xn + src * stride_xc, mask=mask, other=0.0)
            acc += xv.to(tl.float32) * wv.to(tl.float32)
        tl.store(y_ptr + n * stride_yn + c_offs * stride_yc, acc, mask=mask)

    @triton.jit
    def _circ_conv1d_bwd_dx_kernel(
        dy_ptr,
        w_ptr,
        dx_ptr,
        C,
        stride_dyn,
        stride_dyc,
        stride_dxn,
        stride_dxc,
        KW: tl.constexpr,
        PAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = c_offs < C
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(KW):
            src = c_offs + (PAD - k)
            src = tl.where(src < 0, src + C, src)
            src = tl.where(src >= C, src - C, src)
            wv = tl.load(w_ptr + k)
            dyv = tl.load(dy_ptr + n * stride_dyn + src * stride_dyc, mask=mask, other=0.0)
            acc += dyv.to(tl.float32) * wv.to(tl.float32)
        tl.store(dx_ptr + n * stride_dxn + c_offs * stride_dxc, acc, mask=mask)

    @triton.jit
    def _circ_conv1d_bwd_dw_kernel(
        x_ptr,
        dy_ptr,
        dw_ptr,
        C,
        stride_xn,
        stride_xc,
        stride_dyn,
        stride_dyc,
        KW: tl.constexpr,
        PAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = c_offs < C
        dyv = tl.load(dy_ptr + n * stride_dyn + c_offs * stride_dyc, mask=mask, other=0.0)
        dyf = dyv.to(tl.float32)
        for k in range(KW):
            src = c_offs + (k - PAD)
            src = tl.where(src < 0, src + C, src)
            src = tl.where(src >= C, src - C, src)
            xv = tl.load(x_ptr + n * stride_xn + src * stride_xc, mask=mask, other=0.0)
            contrib = tl.sum(dyf * xv.to(tl.float32), axis=0)
            tl.atomic_add(dw_ptr + k, contrib)


def _triton_usable(x: Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and not _triton_failed
        and x.is_cuda
        and x.is_contiguous()
    )


def _mark_triton_failed(exc: BaseException) -> None:
    global _triton_failed, _triton_fail_printed
    _triton_failed = True
    if not _triton_fail_printed:
        _triton_fail_printed = True
        print(f"ConcOnlyC3 Triton circular conv disabled, falling back to PyTorch: {exc!r}")


class _CircConv1dTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor) -> Tensor:
        n, c = x.shape
        kw = int(w.numel())
        pad = kw // 2
        y = torch.empty_like(x, dtype=torch.float32)
        x_f = x.contiguous()
        w_f = w.contiguous().reshape(-1)
        block = 128
        grid = (n, triton.cdiv(c, block))
        _circ_conv1d_fwd_kernel[grid](
            x_f,
            w_f,
            y,
            c,
            x_f.stride(0),
            x_f.stride(1),
            y.stride(0),
            y.stride(1),
            KW=kw,
            PAD=pad,
            BLOCK=block,
        )
        ctx.save_for_backward(x_f, w_f)
        ctx.c = c
        ctx.kw = kw
        ctx.pad = pad
        ctx.x_dtype = x.dtype
        ctx.w_dtype = w.dtype
        return y if x.dtype == torch.float32 else y.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, dy: Tensor):
        x_f, w_f = ctx.saved_tensors
        n, c = x_f.shape
        dy_f = dy.contiguous()
        if dy_f.dtype != torch.float32:
            dy_f = dy_f.float()
        dx = torch.empty_like(x_f, dtype=torch.float32)
        dw = torch.zeros((ctx.kw,), device=x_f.device, dtype=torch.float32)
        block = 128
        grid = (n, triton.cdiv(c, block))
        _circ_conv1d_bwd_dx_kernel[grid](
            dy_f,
            w_f,
            dx,
            c,
            dy_f.stride(0),
            dy_f.stride(1),
            dx.stride(0),
            dx.stride(1),
            KW=ctx.kw,
            PAD=ctx.pad,
            BLOCK=block,
        )
        _circ_conv1d_bwd_dw_kernel[grid](
            x_f,
            dy_f,
            dw,
            c,
            x_f.stride(0),
            x_f.stride(1),
            dy_f.stride(0),
            dy_f.stride(1),
            KW=ctx.kw,
            PAD=ctx.pad,
            BLOCK=block,
        )
        if ctx.x_dtype != torch.float32:
            dx = dx.to(dtype=ctx.x_dtype)
        if ctx.w_dtype != torch.float32:
            dw = dw.to(dtype=ctx.w_dtype)
        return dx, dw


def circular_conv1d_triton(x: Tensor, weight: Tensor) -> Tensor:
    """Triton circular conv along last dim. ``x`` is ``(N, C)``, ``weight`` is ``(KW,)``."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    return _CircConv1dTritonFn.apply(x, weight.reshape(-1))


def triton_circ_conv_available(x: Optional[Tensor] = None) -> bool:
    if x is None:
        return TRITON_AVAILABLE and not _triton_failed
    return _triton_usable(x)


def try_circular_conv1d_triton(x: Tensor, weight: Tensor) -> Optional[Tensor]:
    """Return Triton result or ``None`` if Triton cannot run."""
    x_c = x.contiguous()
    if not _triton_usable(x_c):
        return None
    try:
        leading = x_c.shape[:-1]
        c = x_c.shape[-1]
        y = circular_conv1d_triton(x_c.reshape(-1, c), weight)
        return y.reshape(*leading, c)
    except Exception as exc:
        _mark_triton_failed(exc)
        return None


def try_circular_conv1d_cuda(x: Tensor, weight: Tensor) -> Optional[Tensor]:
    """Return CUDA-extension result or ``None`` if the JIT kernel cannot run."""
    if not x.is_cuda:
        return None
    try:
        from sleepy_orangutan.stm.circ_conv1d_ext import (
            circ_conv1d_cuda,
            extension_available,
        )
    except Exception:
        return None
    if not extension_available():
        return None
    x_c = x.contiguous()
    leading = x_c.shape[:-1]
    c = x_c.shape[-1]
    try:
        y = circ_conv1d_cuda(x_c.reshape(-1, c), weight.reshape(-1).contiguous())
        return y.reshape(*leading, c)
    except Exception:
        return None
