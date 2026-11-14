"""Vendored from PebbleTrail (inference-focused stubs for optional CUDA ext)."""
from typing import Any, Optional
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import ACT2CLS

from minipi05.stm.str2val import str2val

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

possible_output_collapse_methods = [
    'convs', 'meanpool', 'stdpool', 'maxpool', 'none', 'meanstdpool', 'avgproportionpool'
]

def light_causal_conv2d_fn(*args, **kwargs):
    raise NotImplementedError("lc2dv2 CUDA path not vendored for minipi05 inference")

def _can_use_cuda_kernel(*args, **kwargs):
    return False

def _two_layer_lc2dv2_gudermanlu_fwd(*args, **kwargs):
    raise NotImplementedError

def lc2dv2_triton_blockred_fn(*args, **kwargs):
    raise NotImplementedError

def lc2dv2_patch_mm_triton_fn(*args, **kwargs):
    raise NotImplementedError

def lc2dv2_patch_mm_triton_bf16_fn(*args, **kwargs):
    raise NotImplementedError

def lc2dv2_patch_mm_triton_chunked_fn(*args, **kwargs):
    raise NotImplementedError

_symm_conv1d_ext = None
SYMM_CONV1D_EXT_AVAILABLE = False
_symm_conv1d_ext_available = lambda: False

def _lc2dv2_ext(*args, **kwargs):
    raise NotImplementedError

_lc2dv2_ext_vec4 = _lc2dv2_ext
_lc2dv2_ext_vec4_2stage_warp = _lc2dv2_ext
_lc2dv2_ext_blockred = _lc2dv2_ext
_lc2dv2_ext_2layer_fwd = _lc2dv2_ext
_lc2dv2_ext_2stage = _lc2dv2_ext
_lc2dv2_ext_2stage_warp = _lc2dv2_ext
_lc2dv2_ext_2stage_warp_blockred = _lc2dv2_ext
_lc2dv2_ext_prod512 = _lc2dv2_ext
_lc2dv2_ext_amp = _lc2dv2_ext
_lc2dv2_ext_amp_blockred = _lc2dv2_ext
_lc2dv2_amp_blockred_available = lambda: False
_lc2dv2_ext_available = lambda: False

LC2DV2_EXT_AVAILABLE = False

_HALF = (torch.float16, torch.bfloat16)

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
except ImportError:
    _causal_conv1d_fn = None

def concalt_perm(out_channels: int) -> torch.Tensor:
    """Deterministic channel order: 1, n, 2, n-1, 3, n-2, ... (0-based indices)."""
    perm: list[int] = []
    lo, hi = 0, out_channels - 1
    while lo <= hi:
        perm.append(lo)
        if lo < hi:
            perm.append(hi)
        lo += 1
        hi -= 1
    return torch.tensor(perm, dtype=torch.long)


class ConcReor(nn.Module):
    """Expand ``in_channels`` → ``out_channels`` by repeating then permuting the last axis.

    Requires ``out_channels > in_channels``. The input is tiled on ``dim=-1`` until at least
    ``out_channels`` wide, truncated to ``out_channels``, then reordered by a random permutation
    fixed at module construction (stored as a buffer, not trained).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError(
                f"ConcReor requires out_channels > in_channels, got "
                f"{out_channels} <= {in_channels}",
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        perm = torch.randperm(out_channels, generator=gen)
        self.register_buffer("perm", perm.to(torch.long))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, out_channels={self.out_channels})"
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"ConcReor expected last dim {self.in_channels}, got {x.shape[-1]}",
            )
        n_rep = (self.out_channels + self.in_channels - 1) // self.in_channels
        x = x.repeat(*([1] * (x.ndim - 1)), n_rep)
        x = x[..., : self.out_channels]
        return x[..., self.perm]


class ConcOnly(nn.Module):
    """Expand ``in_channels`` → ``out_channels`` by repeat+truncate only (no reorder, no conv)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError(
                f"ConcOnly requires out_channels > in_channels, got "
                f"{out_channels} <= {in_channels}",
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, out_channels={self.out_channels})"
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"ConcOnly expected last dim {self.in_channels}, got {x.shape[-1]}",
            )
        n_rep = (self.out_channels + self.in_channels - 1) // self.in_channels
        x = x.repeat(*([1] * (x.ndim - 1)), n_rep)
        return x[..., : self.out_channels]


class ConcAlt(nn.Module):
    """Expand ``in_channels`` → ``out_channels`` by repeat+truncate, then end-interleave reorder.

    Channel order is deterministic: index 0, n-1, 1, n-2, 2, n-3, ... (1-based: 1, n, 2, n-1, ...).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError(
                f"ConcAlt requires out_channels > in_channels, got "
                f"{out_channels} <= {in_channels}",
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer("perm", concalt_perm(out_channels))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, out_channels={self.out_channels})"
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"ConcAlt expected last dim {self.in_channels}, got {x.shape[-1]}",
            )
        n_rep = (self.out_channels + self.in_channels - 1) // self.in_channels
        x = x.repeat(*([1] * (x.ndim - 1)), n_rep)
        x = x[..., : self.out_channels]
        return x[..., self.perm]


class ConcOnlyC(nn.Module):
    """ConcOnly followed by one shared 1D conv sliding along the expanded channel axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"ConcOnlyC kernel_size must be odd and >= 1, got {kernel_size}")
        self.conconly = ConcOnly(in_channels, out_channels)
        pad = kernel_size // 2
        self.channel_mix = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def __repr__(self) -> str:
        ks = self.channel_mix.kernel_size[0]
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.conconly.in_channels}, "
            f"out_channels={self.conconly.out_channels}, kernel_size={ks})"
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conconly(x)
        b, t, c = y.shape
        y = y.reshape(b * t, 1, c)
        y = self.channel_mix(y)
        return y.reshape(b, t, c)


class ConcStrideC(nn.Module):
    """Reduce last dim with a depth-1 conv, ``stride = in_channels // out_channels``.

    Inverse of :class:`ConcOnly` tiling: each output channel is a learned mix of one
    contiguous group of ``stride`` inputs. Kernel defaults to ``stride`` so padding is 0
    and ``L_out = in_channels / stride`` exactly (e.g. FFN ``3d → d`` uses stride 3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        if out_channels >= in_channels:
            raise ValueError(
                f"ConcStrideC requires out_channels < in_channels, got "
                f"{out_channels} >= {in_channels}",
            )
        if in_channels % out_channels != 0:
            raise ValueError(
                f"ConcStrideC requires in_channels divisible by out_channels, got "
                f"{in_channels} % {out_channels}",
            )
        stride = in_channels // out_channels
        ks = int(stride if kernel_size is None else kernel_size)
        if ks < 1:
            raise ValueError(f"ConcStrideC kernel_size must be >= 1, got {ks}")
        # ``L_out = floor((L + 2p - k) / s) + 1 = L/s`` ⇒ ``2p = k - s``.
        pad_num = ks - stride
        if pad_num < 0 or pad_num % 2 != 0:
            raise ValueError(
                f"ConcStrideC kernel {ks} with stride {stride} cannot hit length "
                f"{out_channels} with symmetric padding (need k>=stride and k-stride even)",
            )
        pad = pad_num // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.channel_mix = nn.Conv1d(
            1, 1, kernel_size=ks, stride=stride, padding=pad, bias=False,
        )

    def __repr__(self) -> str:
        ks = self.channel_mix.kernel_size[0]
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={ks}, stride={self.stride})"
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"ConcStrideC expected last dim {self.in_channels}, got {x.shape[-1]}",
            )
        b, t, c = x.shape
        y = x.reshape(b * t, 1, c)
        y = self.channel_mix(y)
        if y.shape[-1] != self.out_channels:
            raise RuntimeError(
                f"ConcStrideC produced last dim {y.shape[-1]}, expected {self.out_channels}",
            )
        return y.reshape(b, t, self.out_channels)


class ChannelMixC(nn.Module):
    """Shared 1D conv along the last dim, same width (square ``o_proj``)."""

    def __init__(self, channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"ChannelMixC kernel_size must be odd and >= 1, got {kernel_size}")
        self.channels = channels
        pad = kernel_size // 2
        self.channel_mix = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"ChannelMixC expected last dim {self.channels}, got {x.shape[-1]}",
            )
        b, t, c = x.shape
        y = x.reshape(b * t, 1, c)
        y = self.channel_mix(y)
        return y.reshape(b, t, c)


def parse_allconconlyc_kernel(notes: str) -> int | None:
    """Return expand kernel if ``allconconlyc`` / ``allconconlyc:N`` is in notes, else ``None``."""
    notes = notes or ""
    if "allconconlyc" not in notes:
        return None
    if "allconconlyc:" in notes:
        return str2val(notes, "allconconlyc", default=3, output_type=int)
    return 3


def parse_headdim(notes: str) -> int | None:
    """``headdim:N`` in notes, else ``None``."""
    notes = notes or ""
    if "headdim:" not in notes:
        return None
    v = str2val(notes, "headdim", default=0, output_type=int)
    return int(v) if int(v) > 0 else None


def parse_attn_expand_v(notes: str) -> float:
    """``attnexpandv:X`` multiplies the per-head value width. Default 1."""
    notes = notes or ""
    if "attnexpandv:" not in notes:
        return 1.0
    value = float(str2val(notes, "attnexpandv", default=1.0, output_type=float))
    return value if value > 0 else 1.0


def conconlyc_io(in_channels: int, out_channels: int, *, kernel_size: int = 3) -> nn.Module:
    """Tile+conv expand, channel conv if square, strided conv if reducing."""
    if out_channels > in_channels:
        return ConcOnlyC(in_channels, out_channels, kernel_size=kernel_size)
    if out_channels == in_channels:
        return ChannelMixC(in_channels, kernel_size=kernel_size)
    return ConcStrideC(in_channels, out_channels)


class _ConcOnlyCConvThenTile(nn.Module):
    """Circular conv on ``in_channels``, then :class:`ConcOnly` expand.

    Mixes the original channel neighborhood once, then tiles. Concat-then-conv
    (``ConcOnlyC``) repeats the same mix on every copy of the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"{self.__class__.__name__} kernel_size must be odd and >= 1, got {kernel_size}",
            )
        self.conconly = ConcOnly(in_channels, out_channels)
        self.channel_mix = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=False)

    def __repr__(self) -> str:
        ks = self.channel_mix.kernel_size[0]
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.conconly.in_channels}, "
            f"out_channels={self.conconly.out_channels}, kernel_size={ks})"
        )

    def _mix(self, x: Tensor) -> Tensor:
        from minipi05.stm.circ_channel_conv import (
            circular_conv1d_last,
        )
        return circular_conv1d_last(x, self.channel_mix.weight)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.conconly.in_channels:
            raise ValueError(
                f"{self.__class__.__name__} expected last dim {self.conconly.in_channels}, "
                f"got {x.shape[-1]}",
            )
        return self.conconly(self._mix(x))


class ConcOnlyC2(_ConcOnlyCConvThenTile):
    """``stmconconlyc2``: circular PyTorch conv on the input width, then concat-expand."""


class ConcOnlyC3(_ConcOnlyCConvThenTile):
    """``stmconconlyc3``: same as :class:`ConcOnlyC2` with a Triton circular conv."""

    def _mix(self, x: Tensor) -> Tensor:
        from minipi05.stm.circ_channel_conv import (
            circular_conv1d_last,
            try_circular_conv1d_triton,
        )
        y = try_circular_conv1d_triton(x, self.channel_mix.weight)
        if y is None:
            return circular_conv1d_last(x, self.channel_mix.weight)
        return y


class ConcOnlyC4(_ConcOnlyCConvThenTile):
    """``stmconconlyc4``: same as :class:`ConcOnlyC2` with a CUDA circular conv."""

    def _mix(self, x: Tensor) -> Tensor:
        from minipi05.stm.circ_channel_conv import (
            circular_conv1d_last,
            try_circular_conv1d_cuda,
        )
        y = try_circular_conv1d_cuda(x, self.channel_mix.weight)
        if y is None:
            return circular_conv1d_last(x, self.channel_mix.weight)
        return y


class ConcAltC(nn.Module):
    """ConcAlt followed by one shared 1D conv sliding along the expanded channel axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"ConcAltC kernel_size must be odd and >= 1, got {kernel_size}")
        self.concalt = ConcAlt(in_channels, out_channels)
        pad = kernel_size // 2
        self.channel_mix = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def __repr__(self) -> str:
        ks = self.channel_mix.kernel_size[0]
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.concalt.in_channels}, "
            f"out_channels={self.concalt.out_channels}, kernel_size={ks})"
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.concalt(x)
        b, t, c = y.shape
        y = y.reshape(b * t, 1, c)
        y = self.channel_mix(y)
        return y.reshape(b, t, c)


class MultiplyByFreq(nn.Module):
    def __init__(self, freq):
        super().__init__()
        self.freq = freq

    def forward(self, x):
        return x * self.freq


class AlgebraicLUActivation(nn.Module):
    """
    \frac{1}{2} \left(\frac{u}{ \sqrt{\left(u\right)^2+1}}+1\right)u
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * (input / torch.sqrt(input ** 2 + 1) + 1) * input


class GudermanLUActivation(nn.Module):
    """
    \frac{1}{2}+ \frac{2}{\pi} \tan ^{-1}\left(\tanh \left(u\right) \right)u
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 + (2 / torch.pi) * torch.atan(torch.tanh(input)) * input


class GudermanLUActivation_2(nn.Module):
    """
    \frac{1}{2}+ \frac{2}{\pi} \tan ^{-1}\left(\tanh \left(u\right) \right)u
    """

    def forward(self, input: Tensor) -> Tensor:
        return (0.5 + (2 / torch.pi) * torch.atan(torch.tanh(input))) * input


moreACT2FN = {
    **ACT2CLS,
    "algebraiclu": AlgebraicLUActivation,
    "gudermanlu": GudermanLUActivation,
    "gudermanlu2": GudermanLUActivation_2,
    "none": lambda x: x,
}


class StdPoolBT2(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        assert C % self.output_size == 0, f"C must be divisible by {self.output_size}"

        # x = x.view(B, T, 2, C // 2)  # (B, T, 2, C//2)
        x = x.view(B, T, self.output_size, C // self.output_size)
        std = x.std(dim=-1, unbiased=False)  # (B, T, 2)

        return std



class AvgProportionPool(nn.Module):
    def __init__(self, proportion=1):
        super().__init__()
        assert 0 < proportion < 1, f"proportion must be between 0 and 1, got {proportion}"
        self.proportion = proportion
        self.inv_proportion = int(1/proportion)

    def forward(self, x):
        B, T, C = x.shape
        
        x = x.view(B, T, int(C * self.proportion), self.inv_proportion)
        mean = x.mean(dim=-1)  # (B, T, C*proportion)

        return mean


class MeanStdPoolBT2(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        assert 2 * C % self.output_size == 0, f"C must be divisible by {self.output_size}/2"

        x = x.view(B, T, self.output_size // 2, 2 * C // self.output_size)
        std = x.std(dim=-1, unbiased=False)  # (B, T, 2)
        mean = x.mean(dim=-1)
        out = torch.cat([mean, std], dim=-1)  # (B, T, 2 * output_size)
        return out


class Sin(nn.Module):
    """Maps ``sin(x)`` to ``[0, 1]`` via ``sin(x)/2 + 0.5``."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x) / 2 + 0.5


class PeriodicTriangle(nn.Module):
    """
    Continuous triangle wave, period ``2π``, range ``[0, 1]`` (same period as :class:`Sin`).

    Phase-shifted by ``π/2`` so the maximum (``y = 1``) aligns with :class:`Sin` (``sin(x)`` peaks at
    ``x = π/2`` mod ``2π``). With ``phi = remainder(x + π/2, 2π)`` and ``y = 1 - |phi/π - 1|``, the
    peak is at ``x = π/2``; minima at ``x = 3π/2`` (mod ``2π``). No jumps; piecewise linear with
    Lipschitz constant ``1/π`` (derivative jumps only at peaks, not in value).
    """

    def forward(self, x: Tensor) -> Tensor:
        tau = 2 * math.pi
        phi = torch.remainder(x + 0.5 * math.pi, tau)
        return 1.0 - torch.abs(phi / math.pi - 1.0)


class SigTri(nn.Module):
    """Apply sigmoid to half the channels and triangle to the other half (last dim split)."""

    def __init__(self, split_idx: Optional[int] = None):
        super().__init__()
        self.split_idx = split_idx
        self._sigmoid = nn.Sigmoid()
        self._triangle = PeriodicTriangle()

    def forward(self, x: Tensor) -> Tensor:
        c = x.shape[-1]
        k = self.split_idx if self.split_idx is not None else (c // 2)
        k = max(0, min(int(k), c))
        if k == 0:
            return self._triangle(x)
        if k == c:
            return self._sigmoid(x)

        x0 = self._sigmoid(x[..., :k])
        x1 = self._triangle(x[..., k:])
        return torch.cat((x0, x1), dim=-1)


def pack_hyper_bundle(hyper: Optional[dict], suffix: str) -> Optional[dict]:
    """Bundle for :func:`get_conv1d` DynaMat wrapping of ``lc2dv2dyna`` cores."""
    if hyper is None:
        return None
    return {
        "dynamat": hyper["dynamat"],
        "ctx": hyper["ctx"],
        "delta_scale": float(hyper["delta_scale"]),
        "only_dym": bool(hyper.get("only_dym", False)),
        "param_tag": f"{hyper['name_prefix']}_{suffix}",
    }


class _SmallNetNonSequential(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int = 2,
            output_activation: str = "sigmoid",
            headed: bool = True,
            output_collapse_method: str = "convs",
            doprob: float = -1,
            notes: str = "",
            use_cuda: bool = False,
            hyper: Optional[dict] = None,
    ):
        super().__init__()
        assert output_activation in ["sigmoid", "sin", "triangle", "sigtri", "none"], (
            "Only 'sigmoid', 'sin', 'triangle' and 'sigtri' output activations are supported."
        )
        assert output_collapse_method in possible_output_collapse_methods, "Unsupported out_method"

        self.out_channels = out_channels
        self.headed = headed
        self.output_collapse_method = output_collapse_method
        self.doprob = doprob
        self.notes = notes
        self.use_cuda = use_cuda

        dilation_type = str2val(notes, "pdilations", default="1", output_type=str)
        assert dilation_type in ["1", "2", "linear", "exp"]
        self.dilation_type = dilation_type

        intermediate_activation = str2val(notes, "pinteract", default="relu", output_type=str)
        intermediate_activation = moreACT2FN.get(intermediate_activation, nn.ReLU)

        if "pah2lbg" in notes:
            default_hidden_dim = in_channels
        else:
            default_hidden_dim = out_channels
        hidden_dim = str2val(notes, "pahidden", default=default_hidden_dim, output_type=int)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Hidden blocks: conv -> act -> (dropout)
        self.hidden_convs = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()
        self.hidden_drops = nn.ModuleList()

        current_dim = in_channels
        for i in range(1, num_layers):
            if dilation_type == "linear":
                dilation = i
            elif dilation_type == "exp":
                dilation = 2 ** (i - 1)
            else:
                dilation = int(dilation_type)

            self.hidden_convs.append(
                get_conv1d(
                    in_channels=current_dim,
                    out_channels=hidden_dim,
                    dilation=dilation,
                    notes=notes,
                    hyper_bundle=pack_hyper_bundle(hyper, f"hidden{i}"),
                )
            )
            self.hidden_acts.append(intermediate_activation())
            # ModuleList can only hold nn.Module instances (not lambdas).
            
            self.hidden_drops.append(nn.Dropout(doprob) if doprob > 0 else nn.Identity())
            current_dim = hidden_dim

        self.pre_head_conv = None
        if "convbefore01act" in notes:
            i = num_layers
            if dilation_type == "linear":
                dilation = i
            elif dilation_type == "exp":
                dilation = 2 ** (i - 1)
            else:
                dilation = int(dilation_type)

            if output_collapse_method == "none":
                hidden_dim = out_channels

            self.pre_head_conv = get_conv1d(
                in_channels=current_dim,
                out_channels=hidden_dim,
                dilation=dilation,
                notes=notes,
                hyper_bundle=pack_hyper_bundle(hyper, "prehead"),
            )
            current_dim = hidden_dim

        # Collapse / output projection
        if output_collapse_method == "convs":
            self.collapse = CausalConv1d(
                in_channels=current_dim,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            )
        elif output_collapse_method == "meanpool":
            self.collapse = nn.AdaptiveAvgPool1d(out_channels)
        elif output_collapse_method == "stdpool":
            self.collapse = StdPoolBT2(out_channels)
        elif output_collapse_method == "maxpool":
            self.collapse = nn.AdaptiveMaxPool1d(out_channels)
        elif output_collapse_method == "meanstdpool":
            self.collapse = MeanStdPoolBT2(out_channels)
        elif output_collapse_method == "avgproportionpool":
            self.collapse = AvgProportionPool(out_channels/current_dim)
        else:
            self.collapse = lambda x: x

        # Optional frequency scaling
        freq = str2val(notes, "lbgfreq", default=1, output_type=float)
        self.mul_freq = MultiplyByFreq(freq) if freq != 1 else lambda x: x

        # Head activation
        if headed:
            if output_activation == "sigtri":
                self.head_act = SigTri()
            elif output_activation == "sin":
                self.head_act = Sin()
            elif output_activation == "triangle":
                self.head_act = PeriodicTriangle()
            elif output_activation == "none":
                self.head_act = lambda x: x
            else:
                self.head_act = nn.Sigmoid()
        else:
            self.head_act = lambda x: x

        self.post_drop = (
            nn.Dropout(doprob)
            if (doprob > 0 and output_collapse_method != "none" and "no01drop" not in notes)
            else lambda x: x
        )

        self.do_res = self.do_residual if 'doconvres' in notes else (lambda h, y: y)

    def do_residual(self, h, y):
        if y.shape == h.shape:
            return h + y
        else:
            return y

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, C)
        """
        h = x
        for conv, act, drop in zip(self.hidden_convs, self.hidden_acts, self.hidden_drops):
            y = drop(act(conv(h)))
            h = self.do_res(h, y)

        if self.pre_head_conv is not None:
            y = self.pre_head_conv(h)
            h = self.do_res(h, y)

        h = self.collapse(h)
        h = self.mul_freq(h)
        h = self.head_act(h)
        h = self.post_drop(h)
        return h


def build_small_net(
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        output_activation: str = "sigmoid",
        headed: bool = True,
        output_collapse_method: str = "convs",
        doprob=-1,
        notes="",
        use_cuda: bool = False,
        hyper: Optional[dict] = None,
) -> nn.Module:
    """Non-Sequential variant of :func:`build_small_net` with residual connections."""
    return _SmallNetNonSequential(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        output_activation=output_activation,
        headed=headed,
        output_collapse_method=output_collapse_method,
        doprob=doprob,
        notes=notes,
        use_cuda=use_cuda,
        hyper=hyper,
    )


class FoldableProj(nn.Module):
    """
    Causal conv stack that operates on ``(B, T, C)``. For ``ndim > 3``, folds trailing
    dimensions into ``C`` (e.g. ``(B, T, n_h, d_h) -> (B, T, n_h * d_h)``), runs the net,
    then restores the original shape.
    """

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2, output_collapse_method="none", hyper: Optional[dict] = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._hyper = hyper is not None
        pconv = "lc2dv2dyna" if hyper is not None else "lc2dv2"
        self.net = build_small_net(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=depth,
            output_activation="none",
            headed=False,
            output_collapse_method=output_collapse_method,
            notes=f"pah2lbg_pconvtype:{pconv}_packernelt:4_packernel:16_convbefore01act",
            hyper=hyper,
        )
        self.proportion = out_channels / in_channels

    def __repr__(self) -> str:
        # Avoid huge recursive dumps of ``self.net`` (``_SmallNetNonSequential``) on ``print(model)``.
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.net.num_layers}, "
            f"collapse={self.net.output_collapse_method!r}, "
            f"hyper={self._hyper})"
        )

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        b, t = x.shape[0], x.shape[1]
        tail = math.prod(x.shape[2:])
        y = self.net(x.reshape(b, t, tail))

        if self.proportion != 1:
            orig_shape = (*orig_shape[:-1], int(self.proportion * orig_shape[-1]))
        return y.reshape(orig_shape)


def _short_conv1d_pytorch_fallback(
    conv: nn.Conv1d,
    x: Tensor,
    *,
    activation: str = "silu",
) -> Tensor:
    """Depthwise causal short conv on ``x`` shaped ``(B, T, D)`` (CPU / non-FLA fallback)."""
    T = x.shape[1]
    y = x.transpose(1, 2)
    y = conv(y)[..., :T]
    y = y.transpose(1, 2)
    if activation == "silu":
        return F.silu(y)
    if activation in ("swish",):
        return F.silu(y)
    if activation in ("none", None):
        return y
    return y


class ShortConv1d(nn.Module):
    """FLA depthwise short conv; I/O ``(B, T, C)`` like :func:`get_conv1d`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 4,
        bias: bool = False,
        activation: str | None = None,
        backend: str = "triton",
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        if out_channels is not None and out_channels != in_channels:
            raise ValueError("ShortConv1d requires out_channels == in_channels (depthwise)")
        self.activation_name = activation if activation is not None else "none"
        self._pytorch_only = True
        self.short_conv: nn.Module = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            bias=bias,
            padding=kernel_size - 1,
        )
        if backend == "disabled":
            return
        try:
            import triton  # noqa: F401

            from minipi05.gateddeltanet.fla.modules.conv.short_conv import (
                ShortConvolution,
            )

            self.short_conv = ShortConvolution(
                hidden_size=in_channels,
                kernel_size=kernel_size,
                bias=bias,
                activation=activation,
                backend=backend,
            )
            self._pytorch_only = False
        except ImportError:
            pass

    def forward(self, x: Tensor) -> Tensor:
        if not self._pytorch_only and x.is_cuda:
            out, _ = self.short_conv(x)
            return out
        act = self.activation_name
        if not self._pytorch_only and hasattr(self.short_conv, "activation"):
            act = self.short_conv.activation if self.short_conv.activation is not None else "none"
        return _short_conv1d_pytorch_fallback(self.short_conv, x, activation=act)


_LC2DV2_BLOCKRED_CONV_TYPES = frozenset({"cudaext2dv2br"})


def _mark_lc2dv2_blockred_spec(module: nn.Module, *, kt: int, kw: int, width: int) -> None:
    module._lc2dv2_blockred_spec = (int(kt), int(kw), int(width))


def ensure_lc2dv2_blockred_autotune_for_model(
    model: nn.Module,
    device: str | torch.device,
    *,
    seq_len: int = 1024,
    batch_size: int = 1,
    autotune: bool = False,
) -> None:
    """DDP-safe lc2dv2 block-red backward tuning once after the model is on CUDA."""
    from minipi05.stm.light_causal_conv2d_v2_ext import (
        ensure_blockred_autotuned,
    )

    if not torch.cuda.is_available():
        return
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type != "cuda":
        return

    specs: set[tuple[int, int, int]] = set()
    for mod in model.modules():
        spec = getattr(mod, "_lc2dv2_blockred_spec", None)
        if spec is not None:
            specs.add(tuple(spec))
    if not specs:
        return

    kt, kw, width = max(specs, key=lambda s: (s[2], s[0] * s[1]))
    ensure_blockred_autotuned(
        dev,
        kt=kt,
        kw=kw,
        width=width,
        seq_len=seq_len,
        batch_size=batch_size,
        autotune=autotune,
    )


def get_conv1d(
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        notes: str = '',
        bias: bool = True,
        conv_type: str = 'shortconv',
        *,
        kernel_size_w: Optional[int] = 3,
        kernel_size_t: Optional[int] = 1,
        hyper_bundle: Optional[dict] = None,
) -> nn.Module:
    """Return a causal conv mapping ``(B, T, C_in)`` -> ``(B, T, C_out)``.

    ``conv_type`` selects the implementation and overrides ``pconvtype`` from ``notes`` when set.
    If omitted, ``pconvtype`` is parsed from ``notes`` (default ``shortconv``). If ``notes`` contains
    ``mambaconv`` but not ``pconvtype``, type ``mambaconv`` is used.

    ``pconvtype:shortconv`` uses FLA :class:`ShortConv1d` (depthwise causal conv). Backend via
    ``pconvbackend:triton`` or ``pconvbackend:cuda`` in ``notes`` (default ``triton``).

    Use ``pconvtype:lc2dv2dyna`` for :class:`LightCausalConv2D_v2` plus :class:`LightCausalConv2D_v2DynaMat`
    (requires ``hyper_bundle``). Applies to both a single v2 core and :class:`ConcatLightCausalConv2D`.

    Kernel sizes: ``kernel_size_w`` / ``kernel_size_t`` override ``packernel`` / ``packernelt`` in
    ``notes``. If only ``kernel_size`` is provided, it fills any axis still unset from ``notes``.

    CUDA-extension lc2dv2 backends include ``cudaext2dv2``, ``cudaext2dv2br`` (block-reduction dw),
    ``cudaext2dv2amp`` (fp16/bf16 I/O, fp32 accum), and ``cudaext2dv2comboamp`` (combo fp32 paths +
    AMP for half tensors).
    """

    conv_type = str2val(notes, 'pconvtype', default=conv_type, output_type=str)
    kernel_size_w = str2val(notes, 'packernel', default=kernel_size_w, output_type=int)
    kernel_size_t = str2val(notes, 'packernelt', default=kernel_size_t, output_type=int)

    kwprop = str2val(notes, 'kwprop', default=-1, output_type=float)
    if kwprop > 0:
        kernel_size_w = int(in_channels * kwprop)

    pointwise = 'nopointwise' not in notes
    depthwise = 'nodepthwise' not in notes


    conv = None
    conv_layer = None
    conv_kwargs = dict(kw=kernel_size_w, kt=kernel_size_t, bias=bias)
    if conv_type == "linear":
        conv = nn.Linear(in_channels, out_channels, bias=bias)

    elif conv_type == "none":
        conv = lambda x: x

    elif conv_type == "standard":
        conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_t,
            bias=bias,
            dilation=dilation,
        )

    elif conv_type == 'mambaconv':
        kernel_size_t = max(2, min(kernel_size_t, 4))
        conv_layer = MambaGroupedCausalConv1d
        conv_kwargs = dict(
            out_channels=out_channels,
            in_channels=in_channels,
            kernel_size=kernel_size_t,
            bias=bias,
            dilation=dilation,
        )

    elif conv_type == "cudaext1d":
        # CUDA-extension style backend (like causal-conv1d): only accelerates kt==1 path.
        # For kt>1, falls back to baseline LightCausalConv2D_v1.
        conv_layer = LightCausalConv2D_v1CUDAExt
        conv_kwargs = dict(
            kw=kernel_size_w,
            kt=kernel_size_t,
            bias=bias,
            dilation=dilation,
            pointwise=pointwise,
            depthwise=depthwise,
            activation=None,
        )

    elif conv_type == "lc2dv1":
        conv_layer = LightCausalConv2D_v1
        conv_kwargs = dict(
            kw=kernel_size_w,
            kt=kernel_size_t,
            bias=bias,
            dilation=dilation,
            pointwise=pointwise,
            depthwise=depthwise,
        )

    elif conv_type == "lc2dv2":
        conv_layer = LightCausalConv2D_v2

    elif conv_type == "lc2dv2dyna":
        conv_layer = LightCausalConv2D_v2

    elif conv_type == "triton2d":
        # activation=None: build_small_net (and other callers) add activation after the conv.
        # LightCausalConv2DCUDA must not apply activation here to avoid double activation.
        conv_layer = LightCausalConv2DCUDA
        conv_kwargs = dict(
            kw=kernel_size_w,
            kt=kernel_size_t,
            bias=bias,
            dilation=dilation,
            pointwise=pointwise,
            depthwise=depthwise,
            activation=None,
        )

    elif conv_type == "triton2dv2":
        # CUDA-optimized LightCausalConv2D_v2 (single kernel, no dilation)
        conv_layer = LightCausalConv2D_v2CUDA

    elif conv_type == "triton2dv2br":
        # Triton v2 forward + Triton backward (dx + block-reduced dw).
        conv_layer = LightCausalConv2D_v2TritonBlockRed

    elif conv_type == "cudaext2dv2":
        # CUDA-extension backend for lc2dv2 (C++/CUDA like causal-conv1d). Dilation not supported (v2 has none).
        conv_layer = LightCausalConv2D_v2CUDAExt

    elif conv_type == "cudaext2dv2br":
        # CUDA-extension backend for lc2dv2 with block-reduction backward.
        conv_layer = LightCausalConv2D_v2CUDAExtBlockRed

    elif conv_type == "cudaext2dv2amp":
        # CUDA-extension lc2dv2 with fp16/bf16 I/O and fp32 accum (lc2dv2_fwd_amp / lc2dv2_bwd_amp).
        conv_layer = LightCausalConv2D_v2CUDAExtAMP

    elif conv_type == "cudaext2dv2comboamp":
        # Like cudaext2dv2 (combo dispatch for fp32) but uses AMP extension for fp16/bf16 tensors.
        conv_layer = LightCausalConv2D_v2CUDAExtComboAMP

    elif conv_type == "shortconv":
        backend = str2val(notes, "pconvbackend", default="triton", output_type=str)
        conv_layer = ShortConv1d
        conv_kwargs = dict(
            in_channels=in_channels,
            kernel_size=max(2, kernel_size_t),
            bias=bias,
            activation=None,
            backend=backend,
        )
        conv_kwargs['out_channels'] = out_channels

    else:
        raise ValueError(f"Unsupported conv1d type: {conv_type}")

    if conv_type in [
        'lc2dv1', 'lc2dv2', 'lc2dv2dyna', 'triton2d', 'triton2dv2', 'triton2dv2br', 'cudaext2dv2', 'cudaext2dv2br',
        'cudaext2dv2amp', 'cudaext2dv2comboamp', 'cudaext1d', 'mambaconv', 'shortconv',
    ]:


        if out_channels == in_channels:
            conv = conv_layer(**conv_kwargs)
        else:
            print('Using ConcatLightCausalConv2D: out_channels:', out_channels, 'in_channels:', in_channels)
            conv = ConcatLightCausalConv2D(
                out_channels=out_channels,
                in_channels=in_channels,
                layer=conv_layer,
                layer_kwargs=conv_kwargs,
            )

        if conv_type == "lc2dv2dyna":
            raise ValueError(
                "pconvtype lc2dv2dyna moved to foldable_and_dynamat; "
                "use that repo for DynaMat hyper conv experiments."
            )

        if conv_type in _LC2DV2_BLOCKRED_CONV_TYPES:
            _mark_lc2dv2_blockred_spec(
                conv,
                kt=kernel_size_t,
                kw=kernel_size_w,
                width=in_channels,
            )

    return conv


class LightCausalConv2D_v2(nn.Module):
    """
    Causal in time, symmetric in width. One filter (K_T x K_W).
    Y[t,w] = sum_{k_T=0}^{K_T-1} sum_{k_W} W[k_T,k_W] * X[t-k_T, w-k_W]

    Params: K_T * K_W (+ 1 bias if used)

    Uses nn.Conv2d submodule to hold params so torchinfo/summary shows param count.
    """

    def __init__(self, kt: int, kw: int, bias: bool = True):
        super().__init__()
        self.kt = kt
        self.kw = kw
        self.conv = nn.Conv2d(1, 1, (kt, kw), bias=bias, padding=0)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, W)
        """
        w_dtype = self.conv.weight.dtype
        if x.dtype != w_dtype:
            x = x.to(dtype=w_dtype)
        x = x.unsqueeze(1)
        pad_w_left = (self.kw - 1) // 2
        pad_w_right = self.kw // 2
        pad_t_left = self.kt - 1
        pad_t_right = 0
        x = F.pad(x, (pad_w_left, pad_w_right, pad_t_left, pad_t_right))
        w_causal = torch.flip(self.conv.weight, dims=(-2,))
        out = F.conv2d(x, w_causal, self.conv.bias, padding=0)
        return out.squeeze(1)


class LightCausalConv2D_v2PatchMatmul(LightCausalConv2D_v2):
    """
    Same parameters and I/O as :class:`LightCausalConv2D_v2`, but forward uses the patch-matrix form
    ``\\mathrm{vec}(\\mathbf{Y}) = \\mathbf{X}_{\\text{patch}} \\, \\mathbf{k}`` with
    ``\\mathbf{k} = \\mathrm{vec}(\\mathbf{K}) \\in \\mathbb{R}^{K_T K_W}`` and patches from the
    padded input via :func:`torch.nn.functional.unfold` (im2col).

    Per batch, ``cols`` from ``F.unfold`` has shape ``(B, K_T K_W, T W)``; the forward computes
    ``y_flat = cols.transpose(1, 2) @ k`` with ``k`` of shape ``(K_T K_W,)``, then reshapes to ``(B, T, W)``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected x (B, T, W), got shape {tuple(x.shape)}")
        w_dtype = self.conv.weight.dtype
        if x.dtype != w_dtype:
            x = x.to(dtype=w_dtype)
        b, t, w = x.shape
        x4 = x.unsqueeze(1)
        pad_w_left = (self.kw - 1) // 2
        pad_w_right = self.kw // 2
        pad_t_left = self.kt - 1
        pad_t_right = 0
        x_pad = F.pad(x4, (pad_w_left, pad_w_right, pad_t_left, pad_t_right))
        w_causal = torch.flip(self.conv.weight, dims=(-2,))
        k = w_causal.reshape(-1)
        cols = F.unfold(x_pad, kernel_size=(self.kt, self.kw))
        y_flat = cols.transpose(1, 2).matmul(k)
        y = y_flat.view(b, t, w)
        if self.conv.bias is not None:
            y = y + self.conv.bias.view(1, 1, 1)
        return y


class LightCausalConv2D_v2PatchMatmulCUDA(LightCausalConv2D_v2):
    """
    Same as :class:`LightCausalConv2D_v2` / :class:`LightCausalConv2D_v2PatchMatmul` numerically, but forward uses a
    Triton (CUDA) kernel that implements the patch-vector product ``vec(y) = X_patch @ k`` without materializing the full
    unfold. Backward matches lc2dv2 (Triton ``dx`` + block-reduced ``dw`` when available).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight.view(self.kt, self.kw)
        b = self.conv.bias
        return lc2dv2_patch_mm_triton_fn(x, w, b)


class LightCausalConv2D_v2PatchMatmulCUDABF16(LightCausalConv2D_v2):
    """
    Like :class:`LightCausalConv2D_v2PatchMatmulCUDA` but expects **bfloat16** activations ``x``; Triton forward uses
    fp32 accumulation (weights/bias remain fp32 on the module). Backward runs lc2dv2-style kernels in fp32, then casts
    ``dx`` to bf16.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"{type(self).__name__} expects x.dtype torch.bfloat16, got {x.dtype}")
        w = self.conv.weight.view(self.kt, self.kw)
        b = self.conv.bias
        return lc2dv2_patch_mm_triton_bf16_fn(x, w, b)


class LightCausalConv2D_v2PatchMatmulChunked(LightCausalConv2D_v2):
    """
    Patch-matrix lc2dv2 with a forward that steps over time chunks (Triton chunk kernel or ``unfold`` per chunk).
    Reduces peak temporary memory vs a full-sequence unfold. Backward is computed on the full sequence (same as v2).
    """

    def __init__(self, kt: int, kw: int, bias: bool = True, chunk_t: int = 512):
        super().__init__(kt=kt, kw=kw, bias=bias)
        self.chunk_t = int(chunk_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight.view(self.kt, self.kw)
        b = self.conv.bias
        return lc2dv2_patch_mm_triton_chunked_fn(x, w, b, self.chunk_t)


class LightCausalConv2D_v1(nn.Module):
    """
    Causal width-separable convolution over (batch, time, width) dimensions.
    
    Args:
        kw: Kernel size in width dimension
        kt: Kernel size in time dimension  
        dilation: Dilation factor (only applied in time dimension for causality)
        bias: Whether to use bias
    """

    def __init__(
            self,
            kw, kt,
            dilation=1, bias=True,
            pointwise=True,
            depthwise=True
    ):
        super().__init__()
        self.kw = kw
        self.kt = kt
        self.dilation = dilation

        # Depthwise 2D conv over width x time
        # dilation: (dilation_h, dilation_w) where h=width dimension, w=time dimension
        # For causal dilation, we only dilate in time (w), so use (1, dilation)
        self.depthwise = lambda x: x  # identity by default
        self.padding = lambda x: x
        if depthwise:
            self.depthwise = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(kw, kt),
                padding=0,  # manual padding for causality
                dilation=(1, dilation),  # dilate only in time dimension for causality
                groups=1,  # 1 input channel → depthwise trivial
                bias=False,
            )
            pad_t_left = (self.kt - 1) * self.dilation
            pad_t_right = 0
            pad_w_left = (self.kw - 1) // 2
            pad_w_right = self.kw // 2
            self.padding = lambda x: F.pad(
                x,
                (pad_t_left, pad_t_right, pad_w_left, pad_w_right)
            )

        # Pointwise conv mixes channels across hidden_dim later if needed
        self.pointwise = lambda x: x
        if pointwise:
            self.pointwise = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                bias=bias,
            )

    def forward(self, x):
        """
        Original Conv2d-based implementation (sliding window).
        """
        # Make it Conv2d-friendly: (B, 1, W, T)
        x = x.unsqueeze(1).transpose(2, 3)

        # Causal pad in time with dilation, preserve width dimension
        x = self.padding(x)

        # Depthwise conv
        x = self.depthwise(x)

        # Pointwise conv
        x = self.pointwise(x)

        # Back to (B, T, W)
        x = x.transpose(2, 3).squeeze(1)
        return x


class LightCausalConv2D_v1Matmul(nn.Module):
    """
    Same math as :class:`LightCausalConv2D_v1` with explicit ``nn.Parameter`` tensors so tests can
    copy weights from the Conv2d implementation (``depthwise.weight`` -> ``depthwise_weight``).
    """

    def __init__(
            self,
            kw: int,
            kt: int,
            dilation: int = 1,
            bias: bool = True,
            pointwise: bool = True,
            depthwise: bool = True,
    ):
        super().__init__()
        self.kw = kw
        self.kt = kt
        self.dilation = dilation
        self.has_depthwise = depthwise
        self.has_pointwise = pointwise
        if depthwise:
            self.depthwise_weight = nn.Parameter(torch.empty(1, 1, kw, kt))
            nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
            pad_t_left = (self.kt - 1) * self.dilation
            self._pad_wt = (pad_t_left, 0, (kw - 1) // 2, kw // 2)
        else:
            self.register_parameter("depthwise_weight", None)
            self._pad_wt = (0, 0, 0, 0)
        if pointwise:
            self.pointwise_weight = nn.Parameter(torch.empty(1, 1, 1, 1))
            nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
            self.bias = nn.Parameter(torch.empty(1, 1, 1, 1)) if bias else None
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            self.register_parameter("pointwise_weight", None)
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1).transpose(2, 3)
        if self.has_depthwise:
            x = F.pad(x, self._pad_wt)
            x = F.conv2d(
                x,
                self.depthwise_weight,
                None,
                dilation=(1, self.dilation),
            )
        if self.has_pointwise:
            pb = self.bias.view(-1) if self.bias is not None else None
            x = F.conv2d(x, self.pointwise_weight, pb)
        return x.transpose(2, 3).squeeze(1)


class LightCausalConv2D(nn.Module):
    """
    Time–width separable causal conv (same as :class:`LightCausalConv2D_v1`), with an optional
    parameter layout used in benchmarks/tests (``use_matmul=True``).
    """

    def __init__(
            self,
            kw: int,
            kt: int,
            dilation: int = 1,
            bias: bool = True,
            pointwise: bool = True,
            depthwise: bool = True,
            use_matmul: bool = False,
    ):
        super().__init__()
        self.use_matmul = use_matmul
        if use_matmul:
            self.impl = LightCausalConv2D_v1Matmul(
                kw=kw,
                kt=kt,
                dilation=dilation,
                bias=bias,
                pointwise=pointwise,
                depthwise=depthwise,
            )
        else:
            self.impl = LightCausalConv2D_v1(
                kw=kw,
                kt=kt,
                dilation=dilation,
                bias=bias,
                pointwise=pointwise,
                depthwise=depthwise,
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.impl(x)

    @property
    def depthwise(self) -> nn.Module:
        if self.use_matmul:
            raise AttributeError("use depthwise_weight when use_matmul=True")
        return self.impl.depthwise

    @property
    def pointwise(self) -> nn.Module:
        if self.use_matmul:
            raise AttributeError("use pointwise_weight when use_matmul=True")
        return self.impl.pointwise

    @property
    def depthwise_weight(self) -> Tensor:
        if not self.use_matmul:
            raise AttributeError("depthwise_weight exists only when use_matmul=True")
        return self.impl.depthwise_weight

    @property
    def pointwise_weight(self) -> Tensor:
        if not self.use_matmul:
            raise AttributeError("pointwise_weight exists only when use_matmul=True")
        return self.impl.pointwise_weight

    @property
    def bias(self) -> Optional[Tensor]:
        if self.use_matmul:
            return self.impl.bias
        if hasattr(self.impl.pointwise, "bias"):
            return self.impl.pointwise.bias
        return None


class LightCausalConv2D_v1CUDAExt(LightCausalConv2D_v1):
    """
    CUDA-extension style accelerator for the kt==1 case (width-only symmetric conv).

    This mirrors the design of Dao-AILab/causal-conv1d (custom CUDA extension) but for our
    symmetric-in-width conv. For kt>1 it falls back to the base Conv2d implementation.

    Forward path:
    - kt==1 & CUDA ext available: use extension for conv over width (per (B,T) row)
    - else: fallback to base class forward.

    Backward is handled by PyTorch autograd through the extension's autograd.Function.
    """

    def __init__(
            self,
            kw: int,
            kt: int,
            dilation: int = 1,
            bias: bool = True,
            pointwise: bool = True,
            depthwise: bool = True,
            activation: Optional[str] = None,
    ):
        super().__init__(
            kw=kw, kt=kt, dilation=dilation, bias=bias, pointwise=pointwise, depthwise=depthwise
        )
        if activation not in (None, "silu", "swish"):
            raise NotImplementedError("activation must be None, 'silu', or 'swish'")
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
                self.kt == 1
                and x.is_cuda
                and x.dtype == torch.float32
                and x.is_contiguous()
                and _symm_conv1d_ext_available()
                and hasattr(self.depthwise, "weight")
        ):
            # Use extension for conv-only, then apply pointwise/bias/activation in PyTorch.
            B, T, W = x.shape
            # Depthwise weights: (1,1,kw,1) when kt=1 in base module
            w = self.depthwise.weight.reshape(-1).contiguous()  # (kw,)
            x1d = x.reshape(B * T, W)
            conv_out = _symm_conv1d_ext(x1d, w, None, None, self.kw, activation=None).reshape(B, T, W)
            if hasattr(self.pointwise, "weight"):
                pw = self.pointwise.weight.squeeze()
                conv_out = conv_out * pw
                b = getattr(self.pointwise, "bias", None)
                if b is not None:
                    conv_out = conv_out + b
            if self.activation in ("silu", "swish"):
                conv_out = torch.nn.functional.silu(conv_out)
            return conv_out
        return super().forward(x)


class ConcatLightCausalConv2D(nn.Module):
    def __init__(
            self,
            out_channels: int,
            in_channels: int,
            layer=None,
            *,
            layer_kwargs: Optional[dict] = None,
            convs_wrapped: Optional[list] = None,
            **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        if out_channels < 1:
            raise ValueError("out_channels must be positive")

        if convs_wrapped is not None:
            self.convs = nn.ModuleList(convs_wrapped)
            return

        if layer is None:
            raise ValueError("ConcatLightCausalConv2D requires layer when convs_wrapped is None")
        if in_channels < 1:
            raise ValueError("in_channels must be positive")

        n_excess = int(math.ceil(out_channels / in_channels))

        self.convs = nn.ModuleList(
            [layer(**(layer_kwargs or {})) for _ in range(n_excess)]
        )

    def forward(self, x):
        """
        x: (B, T, W)
        """
        conv_outputs = self._forward_convs_parallel(x)
        x = torch.cat(conv_outputs, dim=-1)  # concatenate on width dimension
        x = x[:, :, : self.out_channels]  # trim to desired out_channels
        return x

    def _forward_convs_parallel(self, x):
        """Run convs in parallel on CUDA when beneficial (2+ convs, same input)."""
        if len(self.convs) < 2 or not x.is_cuda:
            return [conv(x) for conv in self.convs]
        streams = [torch.cuda.Stream() for _ in self.convs]
        outputs = [None] * len(self.convs)
        for i, conv in enumerate(self.convs):
            with torch.cuda.stream(streams[i]):
                outputs[i] = conv(x)
        torch.cuda.current_stream().wait_stream(streams[-1])
        for s in streams[:-1]:
            torch.cuda.current_stream().wait_stream(s)
        return outputs


class CausalConv1d(nn.Module):
    """
    Causal convolution that follows HuggingFace convention.
    Expects input in (B, T, W) format and returns output in (B, T, out_channels) format.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, W) tensor following HuggingFace convention
        Returns:
            (B, T, out_channels) tensor following HuggingFace convention
        """
        # Convert from (B, T, W) to (B, W, T) for Conv1d
        x = x.transpose(1, 2)  # (B, T, W) -> (B, W, T)
        w_dtype = self.conv.weight.dtype
        if x.dtype != w_dtype:
            x = x.to(dtype=w_dtype)

        # Apply convolution
        x = self.conv(x)  # (B, W, T) -> (B, out_channels, T)

        # Convert back from (B, out_channels, T) to (B, T, out_channels)
        x = x.transpose(1, 2)  # (B, out_channels, T) -> (B, T, out_channels)
        return x


class MambaGroupedCausalConv1d(nn.Module):
    """
    Mamba-style depthwise grouped causal Conv1d: ``groups == channels``.

    Forward I/O follows HuggingFace-style hidden states: ``x`` is ``(batch, seq_len, channels)``
    (same as :class:`CausalConv1d` / :func:`get_conv1d`). Internally we transpose to
    ``(batch, channels, seq_len)`` for ``nn.Conv1d`` / ``causal_conv1d_fn``.

    Weights live on ``self.conv1d``; ``.weight`` / ``.bias`` alias that module so call sites that
    expect ``nn.Conv1d``-style attributes keep working.

    On CUDA with ``dilation == 1`` and ``kernel_size`` in ``{2, 3, 4}``, uses ``causal_conv1d_fn``
    when installed (the package CUDA path only supports those kernel sizes). Otherwise uses the
    padded Conv1d + trim path (same as the reference Mamba implementation).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            kernel_size: int = 3,
            bias: bool = True,
            dilation: int = 1,
    ):
        super().__init__()

        if not out_channels in (None, in_channels):
            raise ValueError("out_channels must be None or equal to in_channels for MambaGroupedCausalConv1d")

        if in_channels < 1:
            raise ValueError("channels must be positive")
        self.channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        pad = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=pad,
            dilation=dilation,
        )

    @property
    def weight(self) -> Tensor:
        return self.conv1d.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self.conv1d.bias

    def forward(self, x: Tensor) -> Tensor:
        """``x``: (batch, seq_len, channels). Returns the same layout."""
        xc = x.transpose(1, 2).contiguous()
        w_dtype = self.conv1d.weight.dtype
        if xc.dtype != w_dtype:
            xc = xc.to(dtype=w_dtype)
        seq_len = xc.shape[-1]
        if (
                _causal_conv1d_fn is not None
                and self.dilation == 1
                and xc.is_cuda
                and 2 <= self.kernel_size <= 4
        ):
            y = _causal_conv1d_fn(
                x=xc,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=None,
            )
        else:
            y = self.conv1d(xc)[..., :seq_len]
        return y.transpose(1, 2)


# -----------------------------------------------------------------------------
# Module: CUDA optimized versions
# -----------------------------------------------------------------------------

class LightCausalConv2D_v1CUDA(LightCausalConv2D_v1):
    """
    Same interface as LightCausalConv2D but uses a fused Triton kernel on CUDA when available,
    with optional activation (silu/swish/mish). Falls back to Conv2d-based implementation otherwise.
    """

    def __init__(
            self,
            kw: int,
            kt: int,
            dilation: int = 1,
            bias: bool = True,
            pointwise: bool = True,
            depthwise: bool = True,
            activation: Optional[str] = None,
    ):
        super().__init__(
            kw=kw, kt=kt, dilation=dilation, bias=bias,
            pointwise=pointwise, depthwise=depthwise,
        )
        if activation not in (None, "silu", "swish", "mish"):
            raise NotImplementedError("activation must be None, 'silu', 'swish', or 'mish'")
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_cuda = _can_use_cuda_kernel(x) and hasattr(self.depthwise, "weight")
        if not use_cuda:
            return super().forward(x)
        dw = self.depthwise.weight.squeeze()  # (kw, kt)
        pw = self.pointwise.weight.squeeze() if hasattr(self.pointwise, "weight") else None
        b = getattr(self.pointwise, "bias", None)
        return light_causal_conv2d_fn(
            x, dw, weight_pw=pw, bias=b,
            kw=self.kw, kt=self.kt, dilation=self.dilation,
            activation=self.activation,
        )


class LightCausalConv2D_v2CUDA(LightCausalConv2D_v2):
    """
    CUDA-optimized LightCausalConv2D_v2 using the fused Triton kernel.
    Same interface as LightCausalConv2D_v2 (single kt x kw kernel, no dilation).
    Falls back to Conv2d-based implementation when CUDA kernel is unavailable.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _can_use_cuda_kernel(x):
            return super().forward(x)
        # v2: conv.weight is (1, 1, kt, kw) -> squeeze to (kt, kw) or (kw,) when kt=1
        # Kernel expects (kw, kt) with kt_idx=0 = oldest (matches v2's flip(dim=-2))
        w = self.conv.weight.squeeze()  # (kt, kw) or (kw,) when kt=1
        dw = w.unsqueeze(1).contiguous() if w.dim() == 1 else w.flip(0).mT.contiguous()  # (kw, kt)
        b = self.conv.bias
        return light_causal_conv2d_fn(
            x, dw, weight_pw=None, bias=b,
            kw=self.kw, kt=self.kt, dilation=1,
            activation=None,
        )


class LightCausalConv2D_v2TritonBlockRed(LightCausalConv2D_v2):
    """v2: fused Triton forward + Triton backward (dx + block-reduced dw)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight stored as (1,1,kt,kw). Keep as a view of the Parameter (no copy), otherwise grads won't flow.
        # For kt==1, .squeeze() would produce (kw,), so we reshape to (kt, kw) explicitly.
        w = self.conv.weight.view(self.kt, self.kw)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        return lc2dv2_triton_blockred_fn(x, w, b)


class LightCausalConv2D_v2CUDAExt(LightCausalConv2D_v2):
    """
    CUDA-extension (C++/CUDA) implementation of LightCausalConv2D_v2, in the style of causal-conv1d.
    Falls back to the PyTorch reference implementation when the extension is unavailable.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and _lc2dv2_ext_available()):
            return super().forward(x)
        w = self.conv.weight.squeeze().contiguous()  # (kt,kw) or (kw,) if kt==1 but conv2d keeps 2D
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        x, w, b = _lc2dv2_cuda_io_fp32(x, w, b)
        return _lc2dv2_ext(x, w, b, kt=self.kt, kw=self.kw)


class LightCausalConv2D_v2CUDAExtVec4(LightCausalConv2D_v2):
    """
    CUDA-extension v2 with vectorized float4 loads/stores over width (forward only).
    Backward uses the standard CUDA-ext backward kernels (via autograd Function).
    Requires W divisible by 4.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (
                _lc2dv2_ext_vec4 is not None
                and x.is_cuda
                and _lc2dv2_ext_available()
                and (x.shape[-1] % 4 == 0)
        ):
            return super().forward(x)
        w = self.conv.weight.squeeze().contiguous()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        x, w, b = _lc2dv2_cuda_io_fp32(x, w, b)
        return _lc2dv2_ext_vec4(x, w, b, kt=self.kt, kw=self.kw)


class LightCausalConv2D_v2CUDAExtCombo(LightCausalConv2D_v2):
    """
    "Best-of" combo (CUDA fp32 kernels; x and params are cast to float32 at the boundary):
    - prod512 shape (B=1,W=512,kt=1,kw=12): prod512 kernel
    - else if W%4==0: vec4 forward + two-stage-warp backward
    - else: two-stage-warp forward/backward
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (_lc2dv2_ext_available() and x.is_cuda):
            return super().forward(x)
        # prod512 fastpath
        if (
                _lc2dv2_ext_prod512 is not None
                and self.kt == 1
                and self.kw == 12
                and x.shape[0] == 1
                and x.shape[-1] == 512
        ):
            w = self.conv.weight.view(1, 12).contiguous()
            b = self.conv.bias
            b = b.contiguous() if b is not None else None
            x32, w32, b32 = _lc2dv2_cuda_io_fp32(x, w, b)
            return _lc2dv2_ext_prod512(x32, w32, b32)

        w = self.conv.weight.squeeze()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None

        if (
                _lc2dv2_ext_vec4_2stage_warp is not None
                and (x.shape[-1] % 4 == 0)
        ):
            wc = w.contiguous()
            x32, w32, b32 = _lc2dv2_cuda_io_fp32(x, wc, b)
            return _lc2dv2_ext_vec4_2stage_warp(x32, w32, b32, kt=self.kt, kw=self.kw)

        if _lc2dv2_ext_2stage_warp is not None:
            wc = w.contiguous()
            x32, w32, b32 = _lc2dv2_cuda_io_fp32(x, wc, b)
            return _lc2dv2_ext_2stage_warp(x32, w32, b32, kt=self.kt, kw=self.kw)

        # fallback
        return super().forward(x)


class LightCausalConv2D_v2CUDAExtComboAMP(LightCausalConv2D_v2CUDAExtCombo):
    """
    Same dispatch policy as `LightCausalConv2D_v2CUDAExtCombo` for float32 (prod512 / vec4+two-stage-warp / …).

    For fp16/bf16 activations the extension only exposes `lc2dv2_*_amp` (half I/O, fp32 accum); there is no
    separate half-precision prod512 or vec4 forward paired with the combo backward path, so half tensors
    use `LightCausalConv2D_v2CUDAExtAMP` (same CUDA entry points as `v2_cudaext_amp_*`).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in _HALF or self.conv.weight.dtype in _HALF:
            return LightCausalConv2D_v2CUDAExtAMP.forward(self, x)
        return super().forward(x)


class LightCausalConv2D_v2CUDAExtBlockRed(LightCausalConv2D_v2):
    """Same as LightCausalConv2D_v2CUDAExt, but uses block-reduction backward kernel for dw."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in _HALF or self.conv.weight.dtype in _HALF:
            return LightCausalConv2D_v2CUDAExtBlockRedAMP.forward(self, x)
        if not (x.is_cuda and _lc2dv2_ext_available()):
            return super().forward(x)
        if _lc2dv2_ext_blockred is None:
            return super().forward(x)
        w = self.conv.weight.squeeze().contiguous()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        in_dtype = x.dtype
        x, w, b = _lc2dv2_cuda_io_fp32(x, w, b)
        y = _lc2dv2_ext_blockred(x, w, b, kt=self.kt, kw=self.kw)
        return y.to(in_dtype)


class LightCausalConv2D_v2CUDAExtTwoStage(LightCausalConv2D_v2):
    """CUDA-extension v2 where backward uses two-stage dw reduction (no atomics)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (
                _lc2dv2_ext_2stage is not None
                and x.is_cuda
                and _lc2dv2_ext_available()
        ):
            return super().forward(x)
        w = self.conv.weight.squeeze().contiguous()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        x, w, b = _lc2dv2_cuda_io_fp32(x, w, b)
        return _lc2dv2_ext_2stage(x, w, b, kt=self.kt, kw=self.kw)


class LightCausalConv2D_v2CUDAExtTwoStageWarp(LightCausalConv2D_v2):
    """CUDA-extension v2 where backward uses two-stage dw reduction with warp shuffles."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (
                _lc2dv2_ext_2stage_warp is not None
                and x.is_cuda
                and _lc2dv2_ext_available()
        ):
            return super().forward(x)
        w = self.conv.weight.squeeze().contiguous()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        x, w, b = _lc2dv2_cuda_io_fp32(x, w, b)
        return _lc2dv2_ext_2stage_warp(x, w, b, kt=self.kt, kw=self.kw)


class LightCausalConv2D_v2CUDAExtTwoStageWarpBlockRed(LightCausalConv2D_v2):
    """CUDA-extension v2: forward like 2stage_warp, backward dw uses block-reduction kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (
                _lc2dv2_ext_2stage_warp_blockred is not None
                and x.is_cuda
                and _lc2dv2_ext_available()
        ):
            return super().forward(x)
        w = self.conv.weight.squeeze().contiguous()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        x, w, b = _lc2dv2_cuda_io_fp32(x, w, b)
        return _lc2dv2_ext_2stage_warp_blockred(x, w, b, kt=self.kt, kw=self.kw)


class LightCausalConv2D_v2CUDAExtProd512(nn.Module):
    """Specialized production case: B=1, W=512, kt=1, kw=12."""

    def __init__(self, bias: bool = True):
        super().__init__()
        self.kt = 1
        self.kw = 12
        self.conv = nn.Conv2d(1, 1, (self.kt, self.kw), bias=bias, padding=0)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _reference_v2() -> torch.Tensor:
            ref = LightCausalConv2D_v2(kt=1, kw=12, bias=(self.conv.bias is not None)).to(
                device=x.device, dtype=self.conv.weight.dtype
            )
            with torch.no_grad():
                ref.conv.weight.copy_(self.conv.weight)
                if self.conv.bias is not None:
                    ref.conv.bias.copy_(self.conv.bias)
            return ref(x)

        if not (
                _lc2dv2_ext_prod512 is not None
                and _lc2dv2_ext_available()
                and x.is_cuda
                and x.shape[0] == 1
                and x.shape[-1] == 512
        ):
            return _reference_v2()
        w = self.conv.weight.view(1, 12).contiguous()
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        x32, w32, b32 = _lc2dv2_cuda_io_fp32(x, w, b)
        return _lc2dv2_ext_prod512(x32, w32, b32)


class LightCausalConv2D_v2CUDAExtAMP(LightCausalConv2D_v2):
    """Mixed precision (fp16/bf16) CUDA-extension path with fp32 accum."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (_lc2dv2_ext_amp is not None and _lc2dv2_ext_available() and x.is_cuda):
            return super().forward(x)
        w = self.conv.weight.squeeze()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        x, w, b = _lc2dv2_cuda_io_half_amp(x, w, b)
        if x.dtype not in _HALF:
            return super().forward(x)
        return _lc2dv2_ext_amp(x, w, b, kt=self.kt, kw=self.kw)


class LightCausalConv2D_v2CUDAExtBlockRedAMP(LightCausalConv2D_v2):
    """
    Mixed precision (fp16/bf16) CUDA extension: same forward as AMP; backward uses block-reduction dw
    (fewer atomics than default AMP dw). Falls back to AMP if the extension predates lc2dv2_bwd_amp_blockred.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and _lc2dv2_ext_available()):
            return super().forward(x)
        w = self.conv.weight.squeeze()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        b = self.conv.bias
        b = b.contiguous() if b is not None else None
        x, w, b = _lc2dv2_cuda_io_half_amp(x, w, b)
        if x.dtype not in _HALF:
            return super().forward(x)
        if _lc2dv2_amp_blockred_available() and _lc2dv2_ext_amp_blockred is not None:
            return _lc2dv2_ext_amp_blockred(x, w, b, kt=self.kt, kw=self.kw)
        if _lc2dv2_ext_amp is not None:
            return _lc2dv2_ext_amp(x, w, b, kt=self.kt, kw=self.kw)
        return super().forward(x)


# Aliases for backward compatibility
LightCausalConv2D = LightCausalConv2D_v1
LightCausalConv2DCUDA = LightCausalConv2D_v1CUDA


class TwoLayerFusedLC2D(nn.Module):
    """
    Two-layer stack: lc2dv2 -> GudermanLU -> lc2dv2.
    Uses fused Triton kernel on CUDA when available; otherwise falls back to chained modules.
    """

    def __init__(self, kw: int, kt: int, bias: bool = True, enable_fused: bool = False):
        super().__init__()
        self.kw = kw
        self.kt = kt
        # Empirically this fused path is not consistently faster across shapes/hardware.
        # Keep it opt-in so benchmarks can enable it explicitly.
        self.enable_fused = enable_fused
        self.conv1 = LightCausalConv2D_v2(kt=kt, kw=kw, bias=bias)
        self.conv2 = LightCausalConv2D_v2(kt=kt, kw=kw, bias=bias)
        self.act = GudermanLUActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_fused and _can_use_cuda_kernel(x):
            w1 = self.conv1.conv.weight.squeeze()
            w1 = w1.unsqueeze(1).contiguous() if w1.dim() == 1 else w1.flip(0).mT.contiguous()
            w2 = self.conv2.conv.weight.squeeze()
            w2 = w2.unsqueeze(1).contiguous() if w2.dim() == 1 else w2.flip(0).mT.contiguous()
            return _two_layer_lc2dv2_gudermanlu_fwd(
                x, w1, self.conv1.conv.bias, w2, self.conv2.conv.bias,
                kw=self.kw, kt=self.kt,
            )
        return self.conv2(self.act(self.conv1(x)))


class TwoLayerFusedLC2D_CUDAExt(nn.Module):
    """
    Two-layer stack: lc2dv2 -> GudermanLU2 -> lc2dv2.
    Uses CUDA-extension fused forward when available; otherwise falls back to chained modules.
    Forward-only fused path (backward uses fallback chain).
    """

    def __init__(self, kw: int, kt: int, bias: bool = True, enable_fused: bool = False):
        super().__init__()
        self.kw = kw
        self.kt = kt
        self.enable_fused = enable_fused
        self.conv1 = LightCausalConv2D_v2(kt=kt, kw=kw, bias=bias)
        self.conv2 = LightCausalConv2D_v2(kt=kt, kw=kw, bias=bias)
        self.act = GudermanLUActivation_2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
                self.enable_fused
                and
                _lc2dv2_ext_2layer_fwd is not None
                and _lc2dv2_ext_available()
                and x.is_cuda
        ):
            w1 = self.conv1.conv.weight.squeeze().contiguous()
            if w1.dim() == 1:
                w1 = w1.unsqueeze(0)
            b1 = self.conv1.conv.bias
            b1 = b1.contiguous() if b1 is not None else None
            w2 = self.conv2.conv.weight.squeeze().contiguous()
            if w2.dim() == 1:
                w2 = w2.unsqueeze(0)
            b2 = self.conv2.conv.bias
            b2 = b2.contiguous() if b2 is not None else None
            x32, w1, b1 = _lc2dv2_cuda_io_fp32(x, w1, b1)
            w2 = w2.to(torch.float32)
            if b2 is not None:
                b2 = b2.to(torch.float32)
            return _lc2dv2_ext_2layer_fwd(x32, w1, b1, w2, b2, kt=self.kt, kw=self.kw)
        return self.conv2(self.act(self.conv1(x)))
