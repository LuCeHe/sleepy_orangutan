"""
CUDA-optimized LightCausalConv2D via Triton kernels.
Inspired by https://github.com/Dao-AILab/causal-conv1d (causal depthwise conv with fused kernel).

Provides:
- light_causal_conv2d_fn: functional API (like causal_conv1d_fn)
- LightCausalConv2DCUDA: nn.Module that uses the fused kernel on CUDA, falls back to LightCausalConv2D otherwise.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv1d_input, conv1d_weight

try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice as tl_libdevice
        _two_layer_libdevice_available = True
    except ImportError:
        tl_libdevice = None
        _two_layer_libdevice_available = False
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    tl_libdevice = None
    _two_layer_libdevice_available = False
    TRITON_AVAILABLE = False

# Set to True if Triton fails at runtime (e.g. Python.h missing, CUDA driver build failure)
_triton_cuda_failed = False
_triton_failure_message_printed = False


def _can_use_cuda_kernel(x: torch.Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and not _triton_cuda_failed
        and x.is_cuda
        and x.is_contiguous()
    )


# -----------------------------------------------------------------------------
# Fused forward kernel (depthwise causal 2D + pointwise + bias + optional activation)
# -----------------------------------------------------------------------------

if TRITON_AVAILABLE:
    def _light_causal_conv2d_autotune_configs():
        configs = []
        for block_w in [64, 128, 256]:
            for block_t in [1, 4, 8]:
                for num_warps in [4, 8]:
                    configs.append(triton.Config(
                        kwargs={"BLOCK_W": block_w, "BLOCK_T": block_t},
                        num_warps=num_warps,
                    ))
        return configs

    @triton.autotune(
        configs=_light_causal_conv2d_autotune_configs(),
        key=["T", "W"],
    )
    @triton.jit
    def _light_causal_conv2d_fwd_kernel(
        x_ptr,
        out_ptr,
        stride_x_t,
        stride_x_w,
        stride_out_t,
        stride_out_w,
        T: tl.constexpr,
        W: tl.constexpr,
        kw: tl.constexpr,
        kt: tl.constexpr,
        dilation: tl.constexpr,
        pad_w_left: tl.constexpr,
        HAS_PW: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_ACTIVATION: tl.constexpr,  # 0 = none, 1 = silu/swish
        BLOCK_W: tl.constexpr,
        BLOCK_T: tl.constexpr,
        dw_weight: tl.pointer_type,
        pw_weight: tl.pointer_type,
        bias: tl.pointer_type,
    ):
        """Fused causal depthwise (kw x kt, dilated in time) + pointwise over (B, T, W).
        BLOCK_T: process multiple t values per program to reduce launch overhead."""
        n_t_blocks = (T + BLOCK_T - 1) // BLOCK_T
        pid_bt = tl.program_id(0)
        pid_w = tl.program_id(1)
        b = pid_bt // n_t_blocks
        t_block = pid_bt % n_t_blocks
        t_start = t_block * BLOCK_T
        w_start = pid_w * BLOCK_W
        w_offs = w_start + tl.arange(0, BLOCK_W)
        mask_w = w_offs < W
        for t_local in range(BLOCK_T):
            t = t_start + t_local
            if t < T:
                out_row = out_ptr + b * T * stride_out_t + t * stride_out_t
                acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
                for kt_idx in range(kt):
                    in_t = t - (kt - 1) * dilation + kt_idx * dilation
                    if in_t >= 0:
                        x_row = x_ptr + b * T * stride_x_t + in_t * stride_x_t
                        for kw_idx in range(kw):
                            in_w = w_offs - pad_w_left + kw_idx
                            dw_val = tl.load(dw_weight + kw_idx * kt + kt_idx)
                            x_val = tl.load(
                                x_row + in_w * stride_x_w,
                                mask=(in_w >= 0) & (in_w < W) & mask_w,
                                other=0.0,
                            )
                            acc += x_val * dw_val
                if HAS_PW:
                    pw = tl.load(pw_weight)
                    acc = acc * pw
                if HAS_BIAS:
                    b_val = tl.load(bias)
                    acc = acc + b_val
                if HAS_ACTIVATION:
                    acc = acc * tl.sigmoid(acc)
                tl.store(
                    out_row + w_offs * stride_out_w,
                    acc.to(tl.float32),
                    mask=mask_w,
                )


def _light_causal_conv2d_cuda_fwd(
    x: torch.Tensor,
    dw_weight: torch.Tensor,
    pw_weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    kw: int,
    kt: int,
    dilation: int,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Fused causal depthwise + pointwise forward on (B, T, W). Returns (B, T, W)."""
    global _triton_cuda_failed, _triton_failure_message_printed
    if not _can_use_cuda_kernel(x) or activation == "mish":
        return _light_causal_conv2d_ref_fwd(
            x, dw_weight, pw_weight, bias, kw, kt, dilation, activation
        )
    try:
        B, T, W = x.shape
        out = torch.empty_like(x)
        pad_w_left = (kw - 1) // 2
        has_act = activation in ("silu", "swish")  # mish uses reference path
        grid = lambda meta: (
            B * ((T + meta["BLOCK_T"] - 1) // meta["BLOCK_T"]),
            (W + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
        )
        _light_causal_conv2d_fwd_kernel[grid](
            x, out,
            dw_weight=dw_weight,
            pw_weight=pw_weight,
            bias=bias,
            stride_x_t=x.stride(1),
            stride_x_w=x.stride(2),
            stride_out_t=out.stride(1),
            stride_out_w=out.stride(2),
            T=T,
            W=W,
            kw=kw,
            kt=kt,
            dilation=dilation,
            pad_w_left=pad_w_left,
            HAS_PW=(pw_weight is not None),
            HAS_BIAS=(bias is not None),
            HAS_ACTIVATION=has_act,
        )
        return out
    except Exception as e:
        # Triton failed (e.g. Python.h missing, CUDA driver build failure on HPC)
        _triton_cuda_failed = True
        if not _triton_failure_message_printed:
            _triton_failure_message_printed = True
            import warnings
            warnings.warn(
                f"Triton CUDA kernel failed (e.g. Python.h missing on login node); "
                f"using reference Conv2d path. Error: {e!r}",
                UserWarning,
                stacklevel=2,
            )
        return _light_causal_conv2d_ref_fwd(
            x, dw_weight, pw_weight, bias, kw, kt, dilation, activation
        )


# -----------------------------------------------------------------------------
# Autograd: backward uses reference conv2d so gradients are correct
# -----------------------------------------------------------------------------

if TRITON_AVAILABLE:
    @triton.jit
    def _width_conv1d_fwd_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        stride_x_n,
        stride_x_w,
        stride_out_n,
        stride_out_w,
        N: tl.constexpr,
        W: tl.constexpr,
        KW: tl.constexpr,
        PAD_L: tl.constexpr,
        HAS_PW: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_ACTIVATION: tl.constexpr,  # 0 none, 1 silu/swish
        pw_ptr: tl.pointer_type,
        b_ptr: tl.pointer_type,
        BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_w = tl.program_id(1)
        n = pid_n
        w0 = pid_w * BLOCK_W
        offs_w = w0 + tl.arange(0, BLOCK_W)
        mask_w = offs_w < W
        x_row = x_ptr + n * stride_x_n
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for k in range(KW):
            in_w = offs_w - PAD_L + k
            x_val = tl.load(x_row + in_w * stride_x_w, mask=mask_w & (in_w >= 0) & (in_w < W), other=0.0)
            w_val = tl.load(w_ptr + k)
            acc += x_val * w_val
        if HAS_PW:
            pw = tl.load(pw_ptr)
            acc = acc * pw
        if HAS_BIAS:
            bb = tl.load(b_ptr)
            acc = acc + bb
        if HAS_ACTIVATION:
            acc = acc * tl.sigmoid(acc)
        out_row = out_ptr + n * stride_out_n
        tl.store(out_row + offs_w * stride_out_w, acc.to(tl.float32), mask=mask_w)


if TRITON_AVAILABLE:
    @triton.jit
    def _width_conv1d_bwd_dx_kernel(
        dout_ptr,
        w_ptr,
        dx_ptr,
        stride_dout_n,
        stride_dout_w,
        stride_dx_n,
        stride_dx_w,
        N: tl.constexpr,
        W: tl.constexpr,
        KW: tl.constexpr,
        PAD_L: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_w = tl.program_id(1)
        n = pid_n
        w0 = pid_w * BLOCK_W
        offs_w = w0 + tl.arange(0, BLOCK_W)
        mask_w = offs_w < W
        dout_row = dout_ptr + n * stride_dout_n
        # dx[w] = sum_k dout[w + PAD_L - k] * w[k]
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for k in range(KW):
            src_w = offs_w + PAD_L - k
            d_val = tl.load(dout_row + src_w * stride_dout_w, mask=mask_w & (src_w >= 0) & (src_w < W), other=0.0)
            w_val = tl.load(w_ptr + k)
            acc += d_val * w_val
        dx_row = dx_ptr + n * stride_dx_n
        tl.store(dx_row + offs_w * stride_dx_w, acc.to(tl.float32), mask=mask_w)


if TRITON_AVAILABLE:
    @triton.jit
    def _width_conv1d_bwd_dw_partial_kernel(
        x_ptr,
        dout_ptr,
        dw_partial_ptr,
        stride_x_n,
        stride_x_w,
        stride_dout_n,
        stride_dout_w,
        stride_dw_n,
        stride_dw_k,
        N: tl.constexpr,
        W: tl.constexpr,
        KW: tl.constexpr,
        PAD_L: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """
        Compute partial sums for dw over a block of N rows and W columns.
        Writes dw_partial[pid, k] where pid = program_id(0) indexes N-blocks.
        """
        # NOTE: Triton requires power-of-2 block shapes; KW can be non-power-of-2 (e.g. 12).
        # We therefore map k to program_id(1) and compute one scalar per (pid, k).
        pid = tl.program_id(0)  # N-block
        k = tl.program_id(1)    # kernel index in [0, KW)
        n0 = pid * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((), dtype=tl.float32)
        for w0 in tl.static_range(0, W, BLOCK_W):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            mask_w = offs_w < W
            dout = tl.load(
                dout_ptr + offs_n[:, None] * stride_dout_n + offs_w[None, :] * stride_dout_w,
                mask=mask_n[:, None] & mask_w[None, :],
                other=0.0,
            )
            in_w = offs_w[None, :] - PAD_L + k
            x = tl.load(
                x_ptr + offs_n[:, None] * stride_x_n + in_w * stride_x_w,
                mask=mask_n[:, None] & mask_w[None, :] & (in_w >= 0) & (in_w < W),
                other=0.0,
            )
            s_w = tl.sum(x * dout, axis=0)  # (BLOCK_W,)
            acc += tl.sum(s_w, axis=0)

        tl.store(dw_partial_ptr + pid * stride_dw_n + k * stride_dw_k, acc.to(tl.float32))

def _light_causal_conv2d_ref_fwd_kt1(
    x: torch.Tensor,
    dw_weight: torch.Tensor,
    pw_weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    kw: int,
    activation: Optional[str],
) -> torch.Tensor:
    """Reference forward when kt==1: use conv1d over width to avoid conv2d stride bug."""
    B, T, W = x.shape
    pad_w_left = (kw - 1) // 2
    pad_w_right = kw // 2
    # (B, T, W) -> (B*T, 1, W), conv1d over width only
    x1d = x.reshape(B * T, 1, W)
    x1d = F.pad(x1d, (pad_w_left, pad_w_right))
    # dw_weight can be (kw,) when squeezed from (1,1,kw,1), or (kw,1)
    w1d = dw_weight.reshape(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, kw)
    out = F.conv1d(x1d, w1d).reshape(B, T, W)
    return out


def _light_causal_conv2d_ref_fwd(
    x: torch.Tensor,
    dw_weight: torch.Tensor,
    pw_weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    kw: int,
    kt: int,
    dilation: int,
    activation: Optional[str],
) -> torch.Tensor:
    """Reference forward using F.conv2d + pad (for backward and CPU)."""
    if kt == 1:
        out = _light_causal_conv2d_ref_fwd_kt1(x, dw_weight, pw_weight, bias, kw, activation)
        if pw_weight is not None:
            out = out * pw_weight.view(1, 1, 1)
        if bias is not None:
            out = out + bias.view(1, 1, 1)
        if activation in ("silu", "swish"):
            out = F.silu(out)
        elif activation == "mish":
            out = F.mish(out)
        return out

    B, T, W = x.shape
    pad_t_left = (kt - 1) * dilation
    pad_w_left = (kw - 1) // 2
    pad_w_right = kw // 2
    # (B, T, W) -> (B, 1, W, T) for Conv2d
    x2d = x.unsqueeze(1).transpose(2, 3)  # (B, 1, W, T)
    x2d = F.pad(x2d, (pad_t_left, 0, pad_w_left, pad_w_right))
    # dw_weight: (kw, kt) -> (1, 1, kw, kt)
    w2d = dw_weight.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(x2d, w2d, dilation=(1, dilation))
    out = out[:, :, :W, :T]
    if pw_weight is not None:
        out = out * pw_weight.view(1, 1, 1, 1)
    if bias is not None:
        out = out + bias.view(1, 1, 1, 1)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation == "mish":
        out = F.mish(out)
    out = out.transpose(2, 3).squeeze(1)
    return out


class _LightCausalConv2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dw_weight, pw_weight, bias, kw, kt, dilation, activation):
        if _can_use_cuda_kernel(x):
            out = _light_causal_conv2d_cuda_fwd(
                x, dw_weight, pw_weight, bias, kw, kt, dilation, activation
            )
        else:
            out = _light_causal_conv2d_ref_fwd(
                x, dw_weight, pw_weight, bias, kw, kt, dilation, activation
            )
        has_pw = pw_weight is not None
        has_bias = bias is not None
        pw_save = pw_weight if has_pw else torch.tensor([], device=x.device, dtype=x.dtype)
        b_save = bias if has_bias else torch.tensor([], device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x, dw_weight, pw_save, b_save)
        ctx.kw, ctx.kt, ctx.dilation, ctx.activation = kw, kt, dilation, activation
        ctx.has_pw = has_pw
        ctx.has_bias = has_bias
        ctx.needs_input_grad = (x.requires_grad, dw_weight.requires_grad,
            (pw_weight.requires_grad if has_pw else False),
            (bias.requires_grad if has_bias else False))
        return out

    @staticmethod
    def backward(ctx, dout):
        x, dw_weight, pw_saved, b_saved = ctx.saved_tensors
        pw_weight = pw_saved if getattr(ctx, "has_pw", False) else None
        bias = b_saved if getattr(ctx, "has_bias", False) else None
        kw, kt, dilation, activation = ctx.kw, ctx.kt, ctx.dilation, ctx.activation
        B, T, W = x.shape
        pad_w_left = (kw - 1) // 2
        pad_w_right = kw // 2
        need_dx, need_dw, need_pw, need_bias = ctx.needs_input_grad
        if not (need_dx or need_dw or need_pw or need_bias):
            return (None,) * 8

        if kt == 1:
            # kt==1: width-only conv per timestep. Prefer Triton backward to avoid cuDNN wgrad at long W*T.
            x1d = x.detach().reshape(B * T, W).contiguous()
            w1d = dw_weight.detach().reshape(-1).contiguous()  # (kw,)
            conv_out = _light_causal_conv2d_ref_fwd_kt1(
                x, dw_weight, pw_weight, bias, kw, activation
            )  # (B,T,W) conv only; pw/bias/act handled below
            # Backprop through activation, bias, pw to get dout_conv
            dout_conv = dout
            d_pw = d_bias = None
            if activation in ("silu", "swish"):
                out_before_act = conv_out
                if pw_weight is not None:
                    out_before_act = out_before_act * pw_weight.view(1, 1, 1)
                if bias is not None:
                    out_before_act = out_before_act + bias.view(1, 1, 1)
                silu_x = torch.sigmoid(out_before_act)
                silu_grad = silu_x + out_before_act * silu_x * (1 - silu_x)
                dout_conv = dout_conv * silu_grad
            elif activation == "mish":
                out_before_act = conv_out
                if pw_weight is not None:
                    out_before_act = out_before_act * pw_weight.view(1, 1, 1)
                if bias is not None:
                    out_before_act = out_before_act + bias.view(1, 1, 1)
                # mish'(x) = tanh(softplus(x)) + x * (1 - tanh²(softplus(x))) * sigmoid(x)
                sp = F.softplus(out_before_act)
                t = torch.tanh(sp)
                mish_grad = t + out_before_act * (1 - t * t) * torch.sigmoid(out_before_act)
                dout_conv = dout_conv * mish_grad
            if bias is not None and need_bias:
                d_bias = dout_conv.sum().view(1)
            if pw_weight is not None:
                if need_pw:
                    d_pw = (dout_conv * conv_out).sum().view(1)
                dout_conv = dout_conv * pw_weight.view(1, 1, 1)
            if bias is not None:
                pass  # dout_conv unchanged through addition
            dout2d = dout_conv.reshape(B * T, W).contiguous()

            if _can_use_cuda_kernel(dout2d) and TRITON_AVAILABLE and not _triton_cuda_failed:
                N = B * T
                # dx
                if need_dx:
                    dx1d = torch.empty_like(x1d)
                    grid = (N, triton.cdiv(W, 256))
                    _width_conv1d_bwd_dx_kernel[grid](
                        dout2d, w1d, dx1d,
                        stride_dout_n=dout2d.stride(0),
                        stride_dout_w=dout2d.stride(1),
                        stride_dx_n=dx1d.stride(0),
                        stride_dx_w=dx1d.stride(1),
                        N=N, W=W, KW=kw, PAD_L=pad_w_left,
                        BLOCK_W=256,
                    )
                    dx = dx1d.view(B, T, W)
                else:
                    dx = None

                # dw: compute partial sums over N blocks then sum on GPU
                if need_dw:
                    BLOCK_N = 64
                    n_blocks = triton.cdiv(N, BLOCK_N)
                    dw_partial = torch.empty((n_blocks, kw), device=dout2d.device, dtype=torch.float32)
                    grid = (n_blocks, kw)
                    _width_conv1d_bwd_dw_partial_kernel[grid](
                        x1d, dout2d, dw_partial,
                        stride_x_n=x1d.stride(0),
                        stride_x_w=x1d.stride(1),
                        stride_dout_n=dout2d.stride(0),
                        stride_dout_w=dout2d.stride(1),
                        stride_dw_n=dw_partial.stride(0),
                        stride_dw_k=dw_partial.stride(1),
                        N=N, W=W, KW=kw, PAD_L=pad_w_left,
                        BLOCK_N=BLOCK_N,
                        BLOCK_W=256,
                    )
                    d_dw = dw_partial.sum(dim=0).to(dw_weight.dtype).reshape_as(dw_weight)
                else:
                    d_dw = None
            else:
                # Fallback: ATen conv1d grads (may use cuDNN)
                x_pad = F.pad(x1d.unsqueeze(1), (pad_w_left, pad_w_right))
                dout1d = dout2d.unsqueeze(1)
                w_at = w1d.view(1, 1, kw)
                if need_dx:
                    grad_x_pad = conv1d_input(
                        x_pad.shape, w_at, dout1d,
                        stride=1, padding=0, dilation=1, groups=1,
                    )
                    dx = grad_x_pad[:, :, pad_w_left:pad_w_left + W].reshape(B, T, W)
                else:
                    dx = None
                if need_dw:
                    d_dw = conv1d_weight(
                        x_pad, w_at.shape, dout1d,
                        stride=1, padding=0, dilation=1, groups=1,
                    ).reshape_as(dw_weight)
                else:
                    d_dw = None
        else:
            # Custom backward runs with grad disabled; we need a grad-enabled block for the reference graph.
            with torch.enable_grad():
                pad_t_left = (kt - 1) * dilation
                need_conv_grad = need_dx or need_dw or need_pw or need_bias
                x2d = x.detach().unsqueeze(1).transpose(2, 3).clone().requires_grad_(need_dx or need_conv_grad)
                dw2d = dw_weight.detach().unsqueeze(0).unsqueeze(0).clone().requires_grad_(need_dw or need_conv_grad)
                x_pad = F.pad(x2d, (pad_t_left, 0, pad_w_left, pad_w_right))
                out_ref = F.conv2d(x_pad, dw2d, dilation=(1, dilation))[:, :, :W, :T]
            w1d, x1d = None, None

        if kt == 1:
            # dx, d_dw, d_pw, d_bias already computed above
            pass
        else:
            with torch.enable_grad():
                pw_bias_shape = (1, 1, 1, 1)
                if pw_weight is not None:
                    pw = pw_weight.detach().view(pw_bias_shape).requires_grad_(need_pw)
                    out_ref = out_ref * pw
                else:
                    pw = None
                if bias is not None:
                    b = bias.detach().view(pw_bias_shape).requires_grad_(need_bias)
                    out_ref = out_ref + b
                else:
                    b = None
                if activation in ("silu", "swish"):
                    out_ref = F.silu(out_ref)
                elif activation == "mish":
                    out_ref = F.mish(out_ref)
                dout2d = dout.unsqueeze(1).transpose(2, 3)
                out_ref.backward(dout2d)
                dx = x2d.grad.transpose(2, 3).squeeze(1) if need_dx else None
                d_dw = dw2d.grad.squeeze(0).squeeze(0) if need_dw else None
                d_pw = (pw.grad.view_as(pw_weight) if (need_pw and pw is not None and pw_weight is not None) else None)
                d_bias = (b.grad.view_as(bias) if (need_bias and b is not None and bias is not None) else None)
        return dx, d_dw, d_pw, d_bias, None, None, None, None


# -----------------------------------------------------------------------------
# LightCausalConv2D_v2: Triton backward with block reductions (no extension build)
# -----------------------------------------------------------------------------

if TRITON_AVAILABLE:
    def _lc2dv2_bwd_autotune_dx_configs():
        # Larger BLOCK_W can help if W is large, but may reduce occupancy.
        # Keep this small to avoid long compile times on HPC.
        return [
            triton.Config(kwargs={"BLOCK_W": 128}, num_warps=4),
            triton.Config(kwargs={"BLOCK_W": 256}, num_warps=4),
            triton.Config(kwargs={"BLOCK_W": 256}, num_warps=8),
            triton.Config(kwargs={"BLOCK_W": 512}, num_warps=8),
        ]

    @triton.autotune(configs=_lc2dv2_bwd_autotune_dx_configs(), key=["W", "KT", "KW"])
    @triton.jit
    def _lc2dv2_bwd_dx_kernel(
        dy_ptr,
        w_ptr,  # (kt, kw) original orientation
        dx_ptr,
        stride_dy_b,
        stride_dy_t,
        stride_dy_w,
        stride_w_kt,
        stride_w_kw,
        stride_dx_b,
        stride_dx_t,
        stride_dx_w,
        B: tl.constexpr,
        T: tl.constexpr,
        W: tl.constexpr,
        KT: tl.constexpr,
        KW: tl.constexpr,
        PAD: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid_bt = tl.program_id(0)  # over B*T
        pid_w = tl.program_id(1)   # over W blocks
        b = pid_bt // T
        t = pid_bt % T
        w0 = pid_w * BLOCK_W
        offs_w = w0 + tl.arange(0, BLOCK_W)
        mask_w = offs_w < W
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        # dx[b,t,w] = sum_{kt_i,kw_i} dy[b,t+kt_i, w+PAD-kw_i] * w_causal[kt_i,kw_i]
        for kt_i in range(KT):
            out_t = t + kt_i
            if out_t < T:
                # Forward uses original (unflipped) w_ktkw, and causality comes from the index math.
                # So the gradient uses w[kt_i, :] directly here.
                w_row = w_ptr + kt_i * stride_w_kt
                dy_row = dy_ptr + b * stride_dy_b + out_t * stride_dy_t
                for kw_i in range(KW):
                    out_w = offs_w + PAD - kw_i
                    d = tl.load(dy_row + out_w * stride_dy_w, mask=mask_w & (out_w >= 0) & (out_w < W), other=0.0)
                    wv = tl.load(w_row + kw_i * stride_w_kw)
                    acc += d * wv
        tl.store(
            dx_ptr + b * stride_dx_b + t * stride_dx_t + offs_w * stride_dx_w,
            acc.to(tl.float32),
            mask=mask_w,
        )


if TRITON_AVAILABLE:
    def _lc2dv2_bwd_autotune_dw_configs():
        # BT_BLOCK trades reduction work per program vs parallelism across programs.
        # BLOCK_W trades memory coalescing / vectorization vs occupancy.
        return [
            triton.Config(kwargs={"BT_BLOCK": 16, "BLOCK_W": 128}, num_warps=4),
            triton.Config(kwargs={"BT_BLOCK": 32, "BLOCK_W": 128}, num_warps=4),
            triton.Config(kwargs={"BT_BLOCK": 32, "BLOCK_W": 256}, num_warps=4),
            triton.Config(kwargs={"BT_BLOCK": 32, "BLOCK_W": 256}, num_warps=8),
            triton.Config(kwargs={"BT_BLOCK": 64, "BLOCK_W": 256}, num_warps=8),
            triton.Config(kwargs={"BT_BLOCK": 64, "BLOCK_W": 512}, num_warps=8),
        ]

    @triton.autotune(configs=_lc2dv2_bwd_autotune_dw_configs(), key=["W", "KT", "KW"])
    @triton.jit
    def _lc2dv2_bwd_dw_partial_kernel(
        x_ptr,
        dy_ptr,
        dw_partial_ptr,  # (tiles, KT*KW) fp32
        stride_x_b,
        stride_x_t,
        stride_x_w,
        stride_dy_b,
        stride_dy_t,
        stride_dy_w,
        stride_p_tile,
        stride_p_elem,
        B: tl.constexpr,
        T: tl.constexpr,
        W: tl.constexpr,
        KT: tl.constexpr,
        KW: tl.constexpr,
        PAD: tl.constexpr,
        BT_BLOCK: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        # Grid: (tile over B*T, elem over KT*KW)
        pid_tile = tl.program_id(0)
        pid_elem = tl.program_id(1)
        kw_i = pid_elem % KW
        w_row_orig = pid_elem // KW
        kt_i = w_row_orig
        bt0 = pid_tile * BT_BLOCK
        offs_bt = bt0 + tl.arange(0, BT_BLOCK)
        mask_bt = offs_bt < (B * T)
        b = offs_bt // T
        t_out = offs_bt - b * T
        in_t = t_out - kt_i
        valid_t = in_t >= 0

        acc = tl.zeros((), dtype=tl.float32)
        # Static range so Triton can unroll/compile; note: W is constexpr
        for w0 in tl.static_range(0, W, BLOCK_W):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            mask_w = offs_w < W
            dy = tl.load(
                dy_ptr
                + b[:, None] * stride_dy_b
                + t_out[:, None] * stride_dy_t
                + offs_w[None, :] * stride_dy_w,
                mask=mask_bt[:, None] & mask_w[None, :],
                other=0.0,
            )
            in_w = offs_w[None, :] - PAD + kw_i
            x = tl.load(
                x_ptr
                + b[:, None] * stride_x_b
                + in_t[:, None] * stride_x_t
                + in_w * stride_x_w,
                mask=mask_bt[:, None] & valid_t[:, None] & mask_w[None, :] & (in_w >= 0) & (in_w < W),
                other=0.0,
            )
            s_w = tl.sum(x * dy, axis=0)  # (BLOCK_W,)
            acc += tl.sum(s_w, axis=0)   # scalar

        tl.store(dw_partial_ptr + pid_tile * stride_p_tile + pid_elem * stride_p_elem, acc)


# -----------------------------------------------------------------------------
# LightCausalConv2D_v2 patch-matrix forward: y[b,t,w] = sum_{i,j} k[i*KW+j] * x[b,t-(KT-1)+i, w-PAD+j]
# with k = vec(flip(w_ktkw,0)) in row-major (time, width) — same math as unfold @ k / lc2dv2.
# -----------------------------------------------------------------------------

if TRITON_AVAILABLE:
    def _lc2dv2_patch_mm_fwd_configs():
        return [
            triton.Config(kwargs={"BLOCK_W": 128}, num_warps=4),
            triton.Config(kwargs={"BLOCK_W": 256}, num_warps=4),
            triton.Config(kwargs={"BLOCK_W": 256}, num_warps=8),
            triton.Config(kwargs={"BLOCK_W": 512}, num_warps=8),
        ]

    @triton.autotune(configs=_lc2dv2_patch_mm_fwd_configs(), key=["W", "KT", "KW"])
    @triton.jit
    def _lc2dv2_patch_mm_fwd_kernel(
        x_ptr,
        k_ptr,
        out_ptr,
        bias_ptr,
        stride_x_b,
        stride_x_t,
        stride_x_w,
        stride_o_b,
        stride_o_t,
        stride_o_w,
        B,
        T,
        W,
        KT: tl.constexpr,
        KW: tl.constexpr,
        PAD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid_bt = tl.program_id(0)
        pid_w = tl.program_id(1)
        b = pid_bt // T
        t = pid_bt % T
        w0 = pid_w * BLOCK_W
        offs_w = w0 + tl.arange(0, BLOCK_W)
        mask_w = offs_w < W
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for kt_idx in range(KT):
            in_t = t - (KT - 1) + kt_idx
            mask_t = in_t >= 0
            x_row = x_ptr + b * stride_x_b + in_t * stride_x_t
            for kw_idx in range(KW):
                k_val = tl.load(k_ptr + kt_idx * KW + kw_idx)
                in_w = offs_w - PAD + kw_idx
                x_val = tl.load(
                    x_row + in_w * stride_x_w,
                    mask=mask_w & mask_t & (in_w >= 0) & (in_w < W),
                    other=0.0,
                )
                acc += x_val * k_val
        if HAS_BIAS:
            acc += tl.load(bias_ptr)
        tl.store(
            out_ptr + b * stride_o_b + t * stride_o_t + offs_w * stride_o_w,
            acc.to(tl.float32),
            mask=mask_w,
        )

    @triton.autotune(configs=_lc2dv2_patch_mm_fwd_configs(), key=["W", "KT", "KW"])
    @triton.jit
    def _lc2dv2_patch_mm_fwd_bf16_kernel(
        x_ptr,
        k_ptr,
        out_ptr,
        bias_ptr,
        stride_x_b,
        stride_x_t,
        stride_x_w,
        stride_o_b,
        stride_o_t,
        stride_o_w,
        B,
        T,
        W,
        KT: tl.constexpr,
        KW: tl.constexpr,
        PAD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """bf16 x / out; fp32 k and bias; accum in fp32 (AMP-style)."""
        pid_bt = tl.program_id(0)
        pid_w = tl.program_id(1)
        b = pid_bt // T
        t = pid_bt % T
        w0 = pid_w * BLOCK_W
        offs_w = w0 + tl.arange(0, BLOCK_W)
        mask_w = offs_w < W
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for kt_idx in range(KT):
            in_t = t - (KT - 1) + kt_idx
            mask_t = in_t >= 0
            x_row = x_ptr + b * stride_x_b + in_t * stride_x_t
            for kw_idx in range(KW):
                k_val = tl.load(k_ptr + kt_idx * KW + kw_idx).to(tl.float32)
                in_w = offs_w - PAD + kw_idx
                x_val = tl.load(
                    x_row + in_w * stride_x_w,
                    mask=mask_w & mask_t & (in_w >= 0) & (in_w < W),
                    other=0.0,
                ).to(tl.float32)
                acc += x_val * k_val
        if HAS_BIAS:
            acc += tl.load(bias_ptr).to(tl.float32)
        tl.store(
            out_ptr + b * stride_o_b + t * stride_o_t + offs_w * stride_o_w,
            acc.to(tl.bfloat16),
            mask=mask_w,
        )

    def _lc2dv2_patch_mm_chunk_fwd_configs():
        return [
            triton.Config(kwargs={"BLOCK_W": 128}, num_warps=4),
            triton.Config(kwargs={"BLOCK_W": 256}, num_warps=4),
            triton.Config(kwargs={"BLOCK_W": 256}, num_warps=8),
        ]

    @triton.autotune(configs=_lc2dv2_patch_mm_chunk_fwd_configs(), key=["W", "KT", "KW"])
    @triton.jit
    def _lc2dv2_patch_mm_fwd_chunk_kernel(
        x_ptr,
        k_ptr,
        out_ptr,
        bias_ptr,
        stride_x_b,
        stride_x_h,
        stride_x_w,
        stride_o_b,
        stride_o_t,
        stride_o_w,
        B,
        L_OUT,
        W,
        Wp,
        KT: tl.constexpr,
        KW: tl.constexpr,
        PAD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """x: (B, L_OUT+KT-1, Wp) padded height/width slice of x_pad; outputs (B, L_OUT, W)."""
        pid_bt = tl.program_id(0)
        pid_w = tl.program_id(1)
        b = pid_bt // L_OUT
        t = pid_bt % L_OUT
        w0 = pid_w * BLOCK_W
        offs_w = w0 + tl.arange(0, BLOCK_W)
        mask_w = offs_w < W
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for kt_idx in range(KT):
            in_t = t + kt_idx
            x_row = x_ptr + b * stride_x_b + in_t * stride_x_h
            for kw_idx in range(KW):
                k_val = tl.load(k_ptr + kt_idx * KW + kw_idx)
                in_w = PAD + offs_w + kw_idx
                x_val = tl.load(
                    x_row + in_w * stride_x_w,
                    mask=mask_w & (in_w >= 0) & (in_w < Wp),
                    other=0.0,
                )
                acc += x_val * k_val
        if HAS_BIAS:
            acc += tl.load(bias_ptr)
        tl.store(
            out_ptr + b * stride_o_b + t * stride_o_t + offs_w * stride_o_w,
            acc.to(tl.float32),
            mask=mask_w,
        )


def _lc2dv2_patch_mm_prepare_k(w_ktkw: torch.Tensor) -> torch.Tensor:
    """k = vec(flip(w_ktkw,0)) contiguous (KT*KW,) — matches unfold / patch order."""
    if w_ktkw.dim() == 1:
        w_ktkw = w_ktkw.unsqueeze(0)
    w_causal = torch.flip(w_ktkw, dims=(0,))
    return w_causal.reshape(-1).contiguous()


def _lc2dv2_patch_mm_cuda_fwd_full(
    x: torch.Tensor,
    k: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    kt: int,
    kw: int,
) -> torch.Tensor:
    if not TRITON_AVAILABLE or _triton_cuda_failed:
        raise RuntimeError("lc2dv2 patch_mm Triton forward requires Triton CUDA")
    B, T, W = x.shape
    out = torch.empty_like(x)
    pad_w_left = (kw - 1) // 2
    has_b = bias is not None
    grid = lambda meta: (B * T, triton.cdiv(W, meta["BLOCK_W"]))
    _lc2dv2_patch_mm_fwd_kernel[grid](
        x,
        k,
        out,
        bias if has_b else x,  # dummy if no bias
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        T,
        W,
        KT=kt,
        KW=kw,
        PAD=pad_w_left,
        HAS_BIAS=has_b,
    )
    return out


def _lc2dv2_patch_mm_cuda_fwd_full_bf16(
    x: torch.Tensor,
    k: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    kt: int,
    kw: int,
) -> torch.Tensor:
    if not TRITON_AVAILABLE or _triton_cuda_failed:
        raise RuntimeError("lc2dv2 patch_mm bf16 Triton forward requires Triton CUDA")
    if x.dtype != torch.bfloat16:
        raise TypeError("bf16 patch_mm forward expects x.dtype == torch.bfloat16")
    B, T, W = x.shape
    out = torch.empty((B, T, W), device=x.device, dtype=torch.bfloat16)
    pad_w_left = (kw - 1) // 2
    has_b = bias is not None
    grid = lambda meta: (B * T, triton.cdiv(W, meta["BLOCK_W"]))
    _lc2dv2_patch_mm_fwd_bf16_kernel[grid](
        x,
        k,
        out,
        bias if has_b else x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        T,
        W,
        KT=kt,
        KW=kw,
        PAD=pad_w_left,
        HAS_BIAS=has_b,
    )
    return out


def _lc2dv2_patch_mm_cuda_fwd_chunk(
    xh: torch.Tensor,
    k: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    l_out: int,
    kt: int,
    kw: int,
    w_out: int,
    pad_w_left: int,
) -> torch.Tensor:
    """xh: (B, l_out+kt-1, Wp) slice of x_pad; writes (B, l_out, w_out)."""
    if not TRITON_AVAILABLE or _triton_cuda_failed:
        raise RuntimeError("lc2dv2 patch_mm chunk Triton forward requires Triton CUDA")
    B, H, Wp = xh.shape
    out = xh.new_empty((B, l_out, w_out))
    has_b = bias is not None
    grid = lambda meta: (B * l_out, triton.cdiv(w_out, meta["BLOCK_W"]))
    _lc2dv2_patch_mm_fwd_chunk_kernel[grid](
        xh,
        k,
        out,
        bias if has_b else xh,
        xh.stride(0),
        xh.stride(1),
        xh.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        l_out,
        w_out,
        Wp,
        KT=kt,
        KW=kw,
        PAD=pad_w_left,
        HAS_BIAS=has_b,
    )
    return out


def _lc2dv2_patch_mm_backward_impl(ctx, dy: torch.Tensor):
    x, w_ktkw, b_saved = ctx.saved_tensors
    bias = b_saved if ctx.has_bias else None
    if dy is None:
        return None, None, None
    if not x.is_contiguous():
        x = x.contiguous()
    if not dy.is_contiguous():
        dy = dy.contiguous()
    if w_ktkw.dim() == 1:
        w_ktkw = w_ktkw.unsqueeze(0)
    if not w_ktkw.is_contiguous():
        w_ktkw = w_ktkw.contiguous()

    B, T, W = x.shape
    KT, KW = int(w_ktkw.shape[0]), int(w_ktkw.shape[1])
    PAD = (KW - 1) // 2

    dx = torch.empty_like(x) if x.requires_grad else None
    dw = torch.zeros_like(w_ktkw) if w_ktkw.requires_grad else None
    db = None
    if bias is not None and bias.requires_grad:
        db = dy.sum().view_as(bias)

    if TRITON_AVAILABLE and not _triton_cuda_failed and x.is_cuda:
        if dx is not None:
            grid = lambda meta: (B * T, triton.cdiv(W, meta["BLOCK_W"]))
            _lc2dv2_bwd_dx_kernel[grid](
                dy, w_ktkw, dx,
                stride_dy_b=dy.stride(0),
                stride_dy_t=dy.stride(1),
                stride_dy_w=dy.stride(2),
                stride_w_kt=w_ktkw.stride(0),
                stride_w_kw=w_ktkw.stride(1),
                stride_dx_b=dx.stride(0),
                stride_dx_t=dx.stride(1),
                stride_dx_w=dx.stride(2),
                B=B, T=T, W=W, KT=KT, KW=KW, PAD=PAD,
            )
        if dw is not None:
            BT_BLOCK_MIN = 16
            tiles = triton.cdiv(B * T, BT_BLOCK_MIN)
            dw_partial = torch.empty((tiles, KT * KW), device=x.device, dtype=torch.float32)
            grid = (tiles, KT * KW)
            _lc2dv2_bwd_dw_partial_kernel[grid](
                x, dy, dw_partial,
                stride_x_b=x.stride(0),
                stride_x_t=x.stride(1),
                stride_x_w=x.stride(2),
                stride_dy_b=dy.stride(0),
                stride_dy_t=dy.stride(1),
                stride_dy_w=dy.stride(2),
                stride_p_tile=dw_partial.stride(0),
                stride_p_elem=dw_partial.stride(1),
                B=B, T=T, W=W, KT=KT, KW=KW, PAD=PAD,
            )
            dw.copy_(dw_partial.sum(dim=0).to(dw.dtype).view(KT, KW))
    else:
        x_ = x.detach().requires_grad_(x.requires_grad)
        w_ = w_ktkw.detach().requires_grad_(w_ktkw.requires_grad)
        b_ = bias.detach().requires_grad_(bias is not None and bias.requires_grad) if bias is not None else None
        from minipi05.stm.some_convs import LightCausalConv2D_v2
        m = LightCausalConv2D_v2(kt=KT, kw=KW, bias=(b_ is not None)).to(x.device)
        with torch.no_grad():
            m.conv.weight.copy_(w_.view(1, 1, KT, KW))
            if b_ is not None and m.conv.bias is not None:
                m.conv.bias.copy_(b_.view(-1))
        with torch.enable_grad():
            y = m(x_)
        y.backward(dy)
        if dx is not None:
            dx.copy_(x_.grad)
        if dw is not None:
            dw.copy_(m.conv.weight.grad.view(KT, KW))
        if db is not None:
            db.copy_(m.conv.bias.grad.view_as(bias))

    return dx, dw, db


class _LightCausalConv2Dv2PatchMmTritonFn(torch.autograd.Function):
    """Forward: patch @ vec(k) Triton kernel; backward: shared lc2dv2 Triton dx / dw."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w_ktkw: torch.Tensor, bias: Optional[torch.Tensor]):
        global _triton_cuda_failed, _triton_failure_message_printed
        if w_ktkw.dim() == 1:
            w_ktkw = w_ktkw.unsqueeze(0)
        kt, kw = int(w_ktkw.shape[0]), int(w_ktkw.shape[1])
        if not _can_use_cuda_kernel(x) or not TRITON_AVAILABLE or _triton_cuda_failed:
            from minipi05.stm.some_convs import LightCausalConv2D_v2PatchMatmul
            m = LightCausalConv2D_v2PatchMatmul(kt=kt, kw=kw, bias=(bias is not None)).to(x.device, dtype=x.dtype)
            with torch.no_grad():
                m.conv.weight.copy_(w_ktkw.view(1, 1, kt, kw))
                if bias is not None and m.conv.bias is not None:
                    m.conv.bias.copy_(bias.view(-1))
            out = m(x)
        else:
            k = _lc2dv2_patch_mm_prepare_k(w_ktkw).to(dtype=x.dtype)
            try:
                out = _lc2dv2_patch_mm_cuda_fwd_full(x, k, bias, kt=kt, kw=kw)
            except Exception:
                _triton_cuda_failed = True
                if not _triton_failure_message_printed:
                    _triton_failure_message_printed = True
                    import warnings
                    warnings.warn(
                        "lc2dv2 patch_mm Triton forward failed; using unfold fallback.",
                        UserWarning,
                        stacklevel=2,
                    )
                from minipi05.stm.some_convs import LightCausalConv2D_v2PatchMatmul
                m = LightCausalConv2D_v2PatchMatmul(kt=kt, kw=kw, bias=(bias is not None)).to(x.device, dtype=x.dtype)
                with torch.no_grad():
                    m.conv.weight.copy_(w_ktkw.view(1, 1, kt, kw))
                    if bias is not None and m.conv.bias is not None:
                        m.conv.bias.copy_(bias.view(-1))
                out = m(x)

        ctx.save_for_backward(x, w_ktkw, bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype))
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        class _Shim:
            saved_tensors = ctx.saved_tensors
            has_bias = ctx.has_bias

        return _lc2dv2_patch_mm_backward_impl(_Shim(), dy)


def _lc2dv2_patch_mm_ref_fwd_bf16(
    x: torch.Tensor,
    w_ktkw: torch.Tensor,
    bias: Optional[torch.Tensor],
    kt: int,
    kw: int,
) -> torch.Tensor:
    """Reference patch-mm forward with bf16 activations, fp32 kernel vector (AMP-style)."""
    b, t, w = x.shape
    x4 = x.unsqueeze(1)
    pad_w_left = (kw - 1) // 2
    pad_w_right = kw // 2
    pad_t_left = kt - 1
    x_pad = F.pad(x4, (pad_w_left, pad_w_right, pad_t_left, 0))
    w_causal = torch.flip(w_ktkw, dims=(0,))
    k = w_causal.reshape(-1).to(torch.float32)
    cols = F.unfold(x_pad, kernel_size=(kt, kw))
    y_flat = cols.transpose(1, 2).to(torch.float32).matmul(k)
    y = y_flat.view(b, t, w).to(torch.bfloat16)
    if bias is not None:
        y = y + bias.view(1, 1, 1).to(torch.bfloat16)
    return y


class _LightCausalConv2Dv2PatchMmTritonBF16Fn(torch.autograd.Function):
    """bf16 activations, fp32 weights; Triton forward + fp32 backward (then dx cast to bf16)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w_ktkw: torch.Tensor, bias: Optional[torch.Tensor]):
        global _triton_cuda_failed, _triton_failure_message_printed
        if x.dtype != torch.bfloat16:
            raise TypeError("lc2dv2_patch_mm_bf16 expects bfloat16 input x")
        if w_ktkw.dim() == 1:
            w_ktkw = w_ktkw.unsqueeze(0)
        kt, kw = int(w_ktkw.shape[0]), int(w_ktkw.shape[1])
        if not _can_use_cuda_kernel(x) or not TRITON_AVAILABLE or _triton_cuda_failed:
            out = _lc2dv2_patch_mm_ref_fwd_bf16(x, w_ktkw, bias, kt, kw)
        else:
            k = _lc2dv2_patch_mm_prepare_k(w_ktkw).to(dtype=torch.float32)
            b32 = bias.to(torch.float32) if bias is not None else None
            try:
                out = _lc2dv2_patch_mm_cuda_fwd_full_bf16(x, k, b32, kt=kt, kw=kw)
            except Exception:
                _triton_cuda_failed = True
                if not _triton_failure_message_printed:
                    _triton_failure_message_printed = True
                    import warnings
                    warnings.warn(
                        "lc2dv2 patch_mm bf16 Triton forward failed; using unfold reference.",
                        UserWarning,
                        stacklevel=2,
                    )
                out = _lc2dv2_patch_mm_ref_fwd_bf16(x, w_ktkw, bias, kt, kw)

        ctx.save_for_backward(x, w_ktkw, bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype))
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w_ktkw, b_saved = ctx.saved_tensors
        x32 = x.detach().float().contiguous()
        dy32 = dy.float().contiguous() if dy is not None else None
        if dy32 is None:
            return None, None, None

        class _Shim:
            saved_tensors = (x32, w_ktkw, b_saved)
            has_bias = ctx.has_bias

        dx32, dw, db = _lc2dv2_patch_mm_backward_impl(_Shim(), dy32)
        dx = dx32.to(torch.bfloat16) if dx32 is not None else None
        return dx, dw, db


class _LightCausalConv2Dv2PatchMmChunkedTritonFn(torch.autograd.Function):
    """Forward: Triton patch_mm over time chunks (lower peak temp memory); backward: full-sequence lc2dv2 backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w_ktkw: torch.Tensor, bias: Optional[torch.Tensor], chunk_t: int):
        global _triton_cuda_failed, _triton_failure_message_printed
        if w_ktkw.dim() == 1:
            w_ktkw = w_ktkw.unsqueeze(0)
        kt, kw = int(w_ktkw.shape[0]), int(w_ktkw.shape[1])
        B, T, W = x.shape
        pad_w_left = (kw - 1) // 2
        pad_w_right = kw // 2
        pad_t_left = kt - 1
        x4 = x.unsqueeze(1)
        x_pad = F.pad(x4, (pad_w_left, pad_w_right, pad_t_left, 0))
        k = _lc2dv2_patch_mm_prepare_k(w_ktkw).to(dtype=x.dtype)

        outs = []
        ct = max(1, int(chunk_t))
        if _can_use_cuda_kernel(x) and TRITON_AVAILABLE and not _triton_cuda_failed:
            try:
                for t0 in range(0, T, ct):
                    L = min(ct, T - t0)
                    sl = x_pad[:, :, t0 : t0 + L + kt - 1, :].contiguous().squeeze(1)
                    outs.append(
                        _lc2dv2_patch_mm_cuda_fwd_chunk(
                            sl, k, None, l_out=L, kt=kt, kw=kw, w_out=W, pad_w_left=pad_w_left,
                        )
                    )
            except Exception:
                _triton_cuda_failed = True
                if not _triton_failure_message_printed:
                    _triton_failure_message_printed = True
                    import warnings
                    warnings.warn(
                        "lc2dv2 patch_mm chunked Triton failed; using unfold chunks.",
                        UserWarning,
                        stacklevel=2,
                    )
                outs = []
        if not outs:
            for t0 in range(0, T, ct):
                L = min(ct, T - t0)
                sl = x_pad[:, :, t0 : t0 + L + kt - 1, :]
                cols = F.unfold(sl, kernel_size=(kt, kw))
                y_flat = cols.transpose(1, 2).matmul(k)
                outs.append(y_flat.view(B, L, W))
        y = torch.cat(outs, dim=1)
        if bias is not None:
            y = y + bias.view(1, 1, 1)

        ctx.save_for_backward(x, w_ktkw, bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype))
        ctx.has_bias = bias is not None
        ctx.chunk_t = ct
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        class _Shim:
            saved_tensors = ctx.saved_tensors
            has_bias = ctx.has_bias

        dx, dw, db = _lc2dv2_patch_mm_backward_impl(_Shim(), dy)
        return dx, dw, db, None


def lc2dv2_patch_mm_triton_fn(
    x: torch.Tensor,
    w_ktkw: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional lc2dv2 patch-matrix forward (Triton on CUDA) + lc2dv2 backward."""
    return _LightCausalConv2Dv2PatchMmTritonFn.apply(x, w_ktkw, bias)


def lc2dv2_patch_mm_triton_bf16_fn(
    x: torch.Tensor,
    w_ktkw: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """bf16 activations, fp32 weights; Triton forward (fp32 accum) + backward via fp32 lc2dv2 kernels."""
    return _LightCausalConv2Dv2PatchMmTritonBF16Fn.apply(x, w_ktkw, bias)


def lc2dv2_patch_mm_triton_chunked_fn(
    x: torch.Tensor,
    w_ktkw: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    chunk_t: int = 512,
) -> torch.Tensor:
    """Like :func:`lc2dv2_patch_mm_triton_fn` but forward loops over time chunks (smaller unfold footprint)."""
    return _LightCausalConv2Dv2PatchMmChunkedTritonFn.apply(x, w_ktkw, bias, chunk_t)


def lc2dv2_triton_blockred_fn(x: torch.Tensor, w_ktkw: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    v2 forward via existing fused triton forward, backward via triton dx + block-reduced dw.

    Args:
        x: (B,T,W) float32 contiguous CUDA
        w_ktkw: (kt,kw) float32 contiguous CUDA in original orientation (as stored by nn.Conv2d)
        bias: optional (1,) float32 CUDA
    """
    return _LightCausalConv2Dv2TritonBlockRedFn.apply(x, w_ktkw, bias)


class _LightCausalConv2Dv2TritonBlockRedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w_ktkw: torch.Tensor, bias: Optional[torch.Tensor]):
        if not _can_use_cuda_kernel(x):
            # CPU / non-triton: fall back to reference conv2d path (same as LightCausalConv2D_v2)
            from minipi05.stm.some_convs import LightCausalConv2D_v2
            kt, kw = int(w_ktkw.shape[0]), int(w_ktkw.shape[1])
            m = LightCausalConv2D_v2(kt=kt, kw=kw, bias=(bias is not None)).to(x.device)
            with torch.no_grad():
                m.conv.weight.copy_(w_ktkw.view(1, 1, kt, kw))
                if bias is not None and m.conv.bias is not None:
                    m.conv.bias.copy_(bias.view(-1))
            out = m(x)
            ctx.save_for_backward(x, w_ktkw, bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype))
            ctx.has_bias = bias is not None
            return out

        if w_ktkw.dim() == 1:
            w_ktkw = w_ktkw.unsqueeze(0)
        kt, kw = int(w_ktkw.shape[0]), int(w_ktkw.shape[1])
        # Convert to (kw,kt) with kt_idx=0 oldest for the fused forward kernel
        dw = w_ktkw.flip(0).mT.contiguous()
        out = _light_causal_conv2d_cuda_fwd(
            x, dw_weight=dw, pw_weight=None, bias=bias,
            kw=kw, kt=kt, dilation=1, activation=None,
        )
        ctx.save_for_backward(x, w_ktkw, bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype))
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w_ktkw, b_saved = ctx.saved_tensors
        bias = b_saved if ctx.has_bias else None
        if dy is None:
            return None, None, None
        # Ensure contiguity
        if not x.is_contiguous():
            x = x.contiguous()
        if not dy.is_contiguous():
            dy = dy.contiguous()
        if w_ktkw.dim() == 1:
            w_ktkw = w_ktkw.unsqueeze(0)
        if not w_ktkw.is_contiguous():
            w_ktkw = w_ktkw.contiguous()

        B, T, W = x.shape
        KT, KW = int(w_ktkw.shape[0]), int(w_ktkw.shape[1])
        PAD = (KW - 1) // 2

        dx = torch.empty_like(x) if x.requires_grad else None
        dw = torch.zeros_like(w_ktkw) if w_ktkw.requires_grad else None
        db = None
        if bias is not None and bias.requires_grad:
            db = dy.sum().view_as(bias)

        if TRITON_AVAILABLE and not _triton_cuda_failed and x.is_cuda:
            # dx
            if dx is not None:
                # 2D launch: (B*T) x (W blocks). BLOCK_W is autotuned, so grid depends on meta.
                grid = lambda meta: (B * T, triton.cdiv(W, meta["BLOCK_W"]))
                _lc2dv2_bwd_dx_kernel[grid](
                    dy, w_ktkw, dx,
                    stride_dy_b=dy.stride(0),
                    stride_dy_t=dy.stride(1),
                    stride_dy_w=dy.stride(2),
                    stride_w_kt=w_ktkw.stride(0),
                    stride_w_kw=w_ktkw.stride(1),
                    stride_dx_b=dx.stride(0),
                    stride_dx_t=dx.stride(1),
                    stride_dx_w=dx.stride(2),
                    B=B, T=T, W=W, KT=KT, KW=KW, PAD=PAD,
                )
            # dw: two-pass partial reduction, then sum on GPU (no atomics)
            if dw is not None:
                # BT_BLOCK is autotuned inside the kernel. We must size the grid/partial buffer
                # for the *smallest* BT_BLOCK we might select, otherwise we'd miss tiles and
                # silently under-accumulate dw.
                BT_BLOCK_MIN = 16
                tiles = triton.cdiv(B * T, BT_BLOCK_MIN)
                dw_partial = torch.empty((tiles, KT * KW), device=x.device, dtype=torch.float32)
                grid = (tiles, KT * KW)
                _lc2dv2_bwd_dw_partial_kernel[grid](
                    x, dy, dw_partial,
                    stride_x_b=x.stride(0),
                    stride_x_t=x.stride(1),
                    stride_x_w=x.stride(2),
                    stride_dy_b=dy.stride(0),
                    stride_dy_t=dy.stride(1),
                    stride_dy_w=dy.stride(2),
                    stride_p_tile=dw_partial.stride(0),
                    stride_p_elem=dw_partial.stride(1),
                    B=B, T=T, W=W, KT=KT, KW=KW, PAD=PAD,
                )
                dw.copy_(dw_partial.sum(dim=0).to(dw.dtype).view(KT, KW))
        else:
            # Fallback: let PyTorch handle
            x_ = x.detach().requires_grad_(x.requires_grad)
            w_ = w_ktkw.detach().requires_grad_(w_ktkw.requires_grad)
            b_ = bias.detach().requires_grad_(bias is not None and bias.requires_grad) if bias is not None else None
            # use reference conv2d path
            from minipi05.stm.some_convs import LightCausalConv2D_v2
            m = LightCausalConv2D_v2(kt=KT, kw=KW, bias=(b_ is not None)).to(x.device)
            with torch.no_grad():
                m.conv.weight.copy_(w_.view(1, 1, KT, KW))
                if b_ is not None:
                    m.conv.bias.copy_(b_.view(-1))
            out = m(x_)
            out.backward(dy)
            dx = x_.grad if dx is not None else None
            dw = m.conv.weight.grad.squeeze(0).squeeze(0) if dw is not None else None
            db = m.conv.bias.grad.view_as(bias) if (bias is not None and bias.requires_grad) else None

        return dx, dw, db


# -----------------------------------------------------------------------------
# Two-layer fused kernel: lc2dv2 + GudermanLU + lc2dv2
# -----------------------------------------------------------------------------

if TRITON_AVAILABLE and _two_layer_libdevice_available:
    @triton.jit
    def _two_layer_lc2dv2_gudermanlu_fwd_kernel(
        x_ptr,
        out_ptr,
        stride_x_t,
        stride_x_w,
        stride_out_t,
        stride_out_w,
        T: tl.constexpr,
        W: tl.constexpr,
        kw: tl.constexpr,
        kt: tl.constexpr,
        pad_w: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_MID: tl.constexpr,  # BLOCK_W + 2*pad_w for conv2 receptive field
        w1_ptr: tl.pointer_type,
        b1_ptr: tl.pointer_type,
        w2_ptr: tl.pointer_type,
        b2_ptr: tl.pointer_type,
    ):
        """Fused: conv1 (lc2dv2) -> GudermanLU -> conv2 (lc2dv2). GudermanLU: 0.5 + (2/pi)*atan(tanh(x))*x."""
        TWO_PI = 0.63661977236758134308  # 2/pi
        pid = tl.program_id(0)
        b = pid // T
        t = pid % T
        pid_w = tl.program_id(1)
        w_start = pid_w * BLOCK_W
        w_offs = w_start + tl.arange(0, BLOCK_W)
        mask_w = w_offs < W

        out_acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

        for kt2_idx in range(kt):
            in_t = t - kt + 1 + kt2_idx
            if in_t >= 0:
                # Compute conv1 at (b, in_t) for w in [w_start - pad_w, w_start + BLOCK_W + pad_w - 1]
                mid_offs = tl.arange(0, BLOCK_MID)
                mid_w = w_start - pad_w + mid_offs
                mask_mid = (mid_w >= 0) & (mid_w < W)
                mid_acc = tl.zeros((BLOCK_MID,), dtype=tl.float32)
                for kw_idx in range(kw):
                    in_w = mid_w - pad_w + kw_idx
                    for kt1_idx in range(kt):
                        in_t1 = in_t - kt + 1 + kt1_idx
                        if in_t1 >= 0:
                            x_row = x_ptr + b * T * stride_x_t + in_t1 * stride_x_t
                            w1_val = tl.load(w1_ptr + kw_idx * kt + kt1_idx)
                            x_val = tl.load(
                                x_row + in_w * stride_x_w,
                                mask=(in_w >= 0) & (in_w < W) & mask_mid,
                                other=0.0,
                            )
                            mid_acc += x_val * w1_val
                b1_val = tl.load(b1_ptr)
                mid_acc = mid_acc + b1_val
                mid_acc = 0.5 + TWO_PI * tl_libdevice.atan(tl.math.tanh(mid_acc)) * mid_acc

                # Conv2: out_acc[i] += sum_kw2 w2[kw2,kt2_idx] * mid[i + kw2 - pad_w]
                # Use explicit loop over i to avoid negative indexing in mid_acc
                for i in range(BLOCK_W):
                    acc = 0.0
                    for kw2_idx in range(kw):
                        mid_idx = i + kw2_idx - pad_w
                        if mid_idx >= 0 and mid_idx < BLOCK_MID:
                            acc += tl.load(w2_ptr + kw2_idx * kt + kt2_idx) * mid_acc[mid_idx]
                    out_acc = tl.where(
                        tl.arange(0, BLOCK_W) == i,
                        out_acc + acc,
                        out_acc,
                    )

        b2_val = tl.load(b2_ptr)
        out_acc = out_acc + b2_val
        tl.store(
            out_ptr + b * T * stride_out_t + t * stride_out_t + w_offs * stride_out_w,
            out_acc.to(tl.float32),
            mask=mask_w,
        )


def _two_layer_lc2dv2_gudermanlu_fwd(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    kw: int,
    kt: int,
) -> torch.Tensor:
    """Fused two-layer lc2dv2 + GudermanLU + lc2dv2. Uses Triton when available, else reference."""
    if not _can_use_cuda_kernel(x) or not _two_layer_libdevice_available:
        return _two_layer_lc2dv2_gudermanlu_ref_fwd(x, w1, b1, w2, b2, kw, kt)
    B, T, W = x.shape
    pad_w = (kw - 1) // 2
    BLOCK_W = 64  # power of 2 required by Triton
    BLOCK_MID = 128  # power of 2, must be >= BLOCK_W + 2*pad_w
    out = torch.empty_like(x)
    grid = (B * T, triton.cdiv(W, BLOCK_W))
    try:
        _two_layer_lc2dv2_gudermanlu_fwd_kernel[grid](
            x, out,
            stride_x_t=x.stride(1),
            stride_x_w=x.stride(2),
            stride_out_t=out.stride(1),
            stride_out_w=out.stride(2),
            T=T, W=W, kw=kw, kt=kt, pad_w=pad_w,
            BLOCK_W=BLOCK_W,
            BLOCK_MID=BLOCK_MID,
            w1_ptr=w1, b1_ptr=b1, w2_ptr=w2, b2_ptr=b2,
        )
        return out
    except Exception:
        return _two_layer_lc2dv2_gudermanlu_ref_fwd(x, w1, b1, w2, b2, kw, kt)


def _two_layer_lc2dv2_gudermanlu_ref_fwd(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    kw: int,
    kt: int,
) -> torch.Tensor:
    """Reference: conv1 -> GudermanLU -> conv2 using F.conv2d. w1,w2 are (kw,kt), kt_idx=0=oldest."""
    from minipi05.stm.some_convs import (
        LightCausalConv2D_v2,
        GudermanLUActivation,
    )
    conv1 = LightCausalConv2D_v2(kt=kt, kw=kw, bias=True).to(x.device)
    conv2 = LightCausalConv2D_v2(kt=kt, kw=kw, bias=True).to(x.device)
    with torch.no_grad():
        # v2 uses flip(-2) on weight; kernel format (kw,kt) has kt_idx=0=oldest
        # Use torch.flip + .contiguous() to avoid "step must be greater than zero" with copy_
        w1_v2 = torch.flip(w1, dims=[1]).T.unsqueeze(0).unsqueeze(0).contiguous()
        w2_v2 = torch.flip(w2, dims=[1]).T.unsqueeze(0).unsqueeze(0).contiguous()
        conv1.conv.weight.copy_(w1_v2)
        conv1.conv.bias.copy_(b1)
        conv2.conv.weight.copy_(w2_v2)
        conv2.conv.bias.copy_(b2)
    act = GudermanLUActivation()
    return conv2(act(conv1(x)))


def light_causal_conv2d_fn(
    x: torch.Tensor,
    weight_dw: torch.Tensor,
    weight_pw: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    kw: Optional[int] = None,
    kt: Optional[int] = None,
    dilation: int = 1,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Causal light conv2d (depthwise over width x time + optional pointwise), CUDA-optimized when available.

    Args:
        x: (batch, time, width)
        weight_dw: (kw, kt) depthwise kernel
        weight_pw: optional (1,) pointwise scale
        bias: optional (1,) bias
        kw: width kernel size (inferred from weight_dw if None)
        kt: time kernel size (inferred from weight_dw if None)
        dilation: dilation in time dimension
        activation: None, "silu", "swish", or "mish"

    Returns:
        out: (batch, time, width)
    """
    if kw is None:
        kw = weight_dw.shape[0]
    if kt is None:
        kt = weight_dw.shape[1]
    if activation not in (None, "silu", "swish", "mish"):
        raise NotImplementedError("activation must be None, 'silu', 'swish', or 'mish'")
    return _LightCausalConv2DFunction.apply(
        x, weight_dw, weight_pw, bias, kw, kt, dilation, activation
    )
