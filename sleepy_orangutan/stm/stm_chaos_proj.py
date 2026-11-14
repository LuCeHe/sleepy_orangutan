"""
Concatenate-expand + L2-normalize + chaos-index gather for STM input projection.

Uses the foldable/DynaMat ``chaosidx-tri-x3`` tent map (triangle iteration, 3 stages,
max n=2 per stage). Logic mirrors
:mod:`foldable_and_dynamat.neuron_utils._more_layers.modeling_dynamat`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_CHAOS_TRIANGLE_MAX_N = 4.0
_CHAOS_MULTI_STAGE_MAX_N = 2.0
STM_CONCHAOS_ITER_MODE = "triangle2x3"
STM_CONCHAOS_N_STAGES = 3
STM_CONCHAOS_MAP_TYPE = "tent"


def parse_stm_chaos_config(notes: str) -> tuple[str, int]:
    """Parse foldable-style ``stmchaosidx:sig-x3`` / ``tri-x3`` from ``delta:stm`` notes."""
    from sleepy_orangutan.stm.str2val import str2val

    raw = str2val(str(notes or ""), "stmchaosidx", default="", output_type=str).strip().lower()
    if not raw and "stmconchaossig" in str(notes or ""):
        raw = "sig-x3"
    if not raw:
        raw = "tri-x3"
    if raw.startswith("chaosidx-"):
        raw = raw[len("chaosidx-"):]
    if "-x" in raw:
        map_tag, stage_s = raw.rsplit("-x", 1)
        n_stages = max(1, int(stage_s))
    else:
        map_tag, n_stages = raw, STM_CONCHAOS_N_STAGES
    if map_tag in ("sig", "sigmoid", "chaosidx5"):
        iter_mode = f"sigmoid2x{n_stages}"
    elif map_tag in ("tri", "triangle", "chaosidx6"):
        iter_mode = f"triangle2x{n_stages}"
    else:
        raise ValueError(
            f'Unsupported stmchaosidx:{raw!r} (use foldable-style sig-xN or tri-xN, e.g. sig-x3)',
        )
    return iter_mode, n_stages


def _frac01(y: torch.Tensor) -> torch.Tensor:
    return y - torch.floor(y)


def _chaos_triangle_osc_n(
    raw: torch.Tensor,
    *,
    max_n: float = _CHAOS_TRIANGLE_MAX_N,
) -> torch.Tensor:
    max_n_f = float(max_n)
    period = 2.0 * max_n_f
    half = period / 2.0
    phase = torch.fmod(raw, period)
    phase = torch.where(phase < 0, phase + period, phase)
    return torch.where(phase < half, phase, period - phase) * (max_n_f / half)


def _chaos_iter_n_from_raw(raw: torch.Tensor, *, iter_mode: str | None) -> torch.Tensor:
    max_n = _CHAOS_MULTI_STAGE_MAX_N if (iter_mode and "2x" in iter_mode) else _CHAOS_TRIANGLE_MAX_N
    if iter_mode is not None and iter_mode.startswith("triangle"):
        return _chaos_triangle_osc_n(raw, max_n=max_n)
    if iter_mode is not None and iter_mode.startswith("sigmoid"):
        return max_n * torch.sigmoid(raw)
    if iter_mode == "sin4":
        return _CHAOS_TRIANGLE_MAX_N * (torch.sin(raw) + 1.0) / 2.0
    return raw.abs()


def chaos_index(
    i: torch.Tensor,
    I: int | float,
    shift: torch.Tensor | None = None,
    n_iter: torch.Tensor | None = None,
    *,
    map_type: str = STM_CONCHAOS_MAP_TYPE,
    n_stages: int = STM_CONCHAOS_N_STAGES,
) -> torch.Tensor:
    """Chaos map on index coordinate (foldable ``chaos_index`` forward, no custom VJP)."""
    i_f = i.to(dtype=torch.float32 if not i.is_floating_point() else i.dtype)
    i_f = i_f.to(dtype=torch.float32)
    I_f = float(I)
    x = i_f / I_f
    if shift is not None:
        delta = shift.to(dtype=x.dtype)
        while delta.dim() < x.dim():
            delta = delta.unsqueeze(-1)
        x = _frac01(x + delta)
    if n_iter is not None:
        x = x.clamp(0.0, 1.0)
        n = n_iter.to(dtype=x.dtype)
        while n.dim() < x.dim():
            n = n.unsqueeze(-1)
        if map_type == "tent":
            stages = max(1, int(n_stages))
            for _stage in range(stages):
                x = x.clamp(0.0, 1.0)
                x = 1.0 - torch.abs(1.0 - torch.fmod(2.0 ** n * x, 2.0))
            out = x
        else:
            sqrt_x = torch.sqrt(x.clamp(min=0.0, max=1.0))
            inner = torch.asin(sqrt_x.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            out = torch.sin((2.0 ** n) * inner) ** 2
    elif map_type == "tent":
        out = 1.0 - torch.abs(2.0 * x - 1.0)
    else:
        out = 4.0 * x * (1.0 - x)
    return out.to(dtype=i.dtype if i.is_floating_point() else torch.float32) * I_f


class ConcChaosProj(nn.Module):
    """
    Expand ``in_channels`` → ``out_channels`` by concatenation (repeat+truncate),
    L2-normalize, then gather along the last axis with ``chaosidx-tri-x3`` indices.

    Learnable shift / iteration heads come from a tiny linear map on the time-mean input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        chaos_iter_mode: str = STM_CONCHAOS_ITER_MODE,
        chaos_n_stages: int = STM_CONCHAOS_N_STAGES,
    ) -> None:
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError(
                f"ConcChaosProj requires out_channels > in_channels, got "
                f"{out_channels} <= {in_channels}",
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chaos_iter_mode = chaos_iter_mode
        self.chaos_n_stages = max(1, int(chaos_n_stages))
        self.chaos_head = nn.Linear(in_channels, 2, bias=True)
        nn.init.zeros_(self.chaos_head.weight)
        nn.init.zeros_(self.chaos_head.bias)

    def __repr__(self) -> str:
        tag = "sig" if self.chaos_iter_mode.startswith("sigmoid") else "tri"
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"chaos=chaosidx-{tag}-x{self.chaos_n_stages})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"ConcChaosProj expected last dim {self.in_channels}, got {x.shape[-1]}",
            )
        b, t, _d = x.shape
        out_c = self.out_channels
        n_rep = (out_c + self.in_channels - 1) // self.in_channels
        x_cat = x.repeat(*([1] * (x.ndim - 1)), n_rep)[..., :out_c]
        x_norm = F.normalize(x_cat, p=2, dim=-1, eps=1e-6)

        pooled = x.mean(dim=1)
        raw = self.chaos_head(pooled)
        shift = raw[:, 0:1]
        n_iter = _chaos_iter_n_from_raw(raw[:, 1:2], iter_mode=self.chaos_iter_mode)

        j = torch.arange(1, out_c + 1, device=x.device, dtype=x.dtype)
        j_b = j.unsqueeze(0).expand(b, -1)
        mapped = chaos_index(
            j_b,
            out_c,
            shift,
            n_iter,
            map_type=STM_CONCHAOS_MAP_TYPE,
            n_stages=self.chaos_n_stages,
        )
        denom = max(out_c - 1, 1)
        gather_idx = (
            ((mapped - 1.0) / float(denom) * float(out_c - 1))
            .round()
            .long()
            .clamp(0, out_c - 1)
        )
        gather_idx = gather_idx.unsqueeze(1).expand(-1, t, -1)
        return x_norm.gather(-1, gather_idx)
