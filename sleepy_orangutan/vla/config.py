"""VLA student config presets (GDN-only sleepy_orangutan)."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any


def _str2val(notes: str, key: str, default: str = "", output_type=str):
    m = re.search(rf"{re.escape(key)}:([^_:]+)", notes or "")
    if not m:
        return output_type(default) if output_type is not str else default
    return output_type(m.group(1))


def _gdn_heads(hidden_size: int) -> tuple[int, int]:
    target = int(round(0.75 * hidden_size))
    for num_heads in (2, 4, 6, 8):
        if target % num_heads == 0:
            return num_heads, target // num_heads
    return 4, max(1, target // 4)


def _deltanet_heads(hidden_size: int) -> tuple[int, int]:
    for num_heads in (8, 4, 16, 2):
        if hidden_size % num_heads == 0:
            return num_heads, hidden_size // num_heads
    return 4, max(1, hidden_size // 4)


@dataclass
class VLAStudentConfig:
    action_dim: int = 8
    state_dim: int = 8
    action_horizon: int = 10
    num_cameras: int = 2
    image_size: int = 128
    subgoal_dim: int = 32
    hidden_size: int = 512
    num_gdn_layers: int = 5
    num_gdn_heads: int = 0
    gdn_head_dim: int = 0
    gdn_mode: str = "chunk"  # legacy PebbleTrail field (ignored at inference)
    use_gate: bool = True
    notes: str = "delta:gateddelta"
    distill_action_weight: float = 1.0
    distill_feature_weight: float = 0.1
    inference_dtype: str = "bfloat16"
    prefer_int8: bool = True
    vision_width_mult: float = 1.0
    vision_out_dim: int | None = None
    robot_id: str = "droid_panda"
    preset: str = "custom"

    def __post_init__(self) -> None:
        if self.vision_out_dim is None:
            self.vision_out_dim = self.hidden_size
        if self.use_gate:
            self.num_gdn_heads, self.gdn_head_dim = _gdn_heads(self.hidden_size)
        else:
            self.num_gdn_heads, self.gdn_head_dim = _deltanet_heads(self.hidden_size)

    @property
    def trunk_notes(self) -> str:
        return self.notes or "delta:gateddelta"

    @property
    def delta_type(self) -> str:
        return _str2val(self.trunk_notes, "delta", default="gateddelta", output_type=str)

    def to_delta_config(self, *, layer_idx: int = 0) -> SimpleNamespace:
        return SimpleNamespace(
            hidden_size=self.hidden_size,
            notes=self.trunk_notes,
            num_attention_heads=max(1, self.hidden_size // 64),
            layer_idx=layer_idx,
        )


_PRESETS = {
    "s100_droid_50m": dict(
        hidden_size=512,
        num_gdn_layers=5,
        vision_width_mult=1.0,
        action_horizon=10,
        action_dim=8,
        state_dim=8,
        num_cameras=2,
        robot_id="droid_panda",
        notes="delta:gateddelta",
    ),
    "s100_droid_10m": dict(
        hidden_size=320,
        num_gdn_layers=3,
        vision_width_mult=0.75,
        action_horizon=10,
        action_dim=8,
        state_dim=8,
        num_cameras=2,
        robot_id="droid_panda",
        notes="delta:gateddelta",
    ),
    "s100_libero_10m": dict(
        hidden_size=320,
        num_gdn_layers=3,
        vision_width_mult=0.75,
        action_horizon=10,
        action_dim=7,
        state_dim=7,
        num_cameras=2,
        robot_id="libero_franka",
        notes="delta:gateddelta",
    ),
    "s100_libero_50m": dict(
        hidden_size=512,
        num_gdn_layers=5,
        vision_width_mult=1.0,
        action_horizon=10,
        action_dim=7,
        state_dim=7,
        num_cameras=2,
        robot_id="libero_franka",
        notes="delta:gateddelta",
    ),
}


def vla_config_from_saved_dict(raw: dict[str, Any]) -> VLAStudentConfig:
    """Load config from ``student.pt`` (drops unknown PebbleTrail-only keys)."""
    known = {f.name for f in fields(VLAStudentConfig)}
    kwargs = {k: v for k, v in dict(raw).items() if k in known}
    return VLAStudentConfig(**kwargs)


def get_vla_config(name: str, **overrides) -> VLAStudentConfig:
    if name not in _PRESETS:
        raise KeyError(f"Unknown preset {name!r}; known: {sorted(_PRESETS)}")
    fields = dict(_PRESETS[name])
    fields.update(overrides)
    return VLAStudentConfig(preset=name, **fields)
