"""VLA student config presets (GDN-only minipi05)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace


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
    use_gate: bool = True
    notes: str = "delta:gateddelta"
    vision_width_mult: float = 1.0
    vision_out_dim: int | None = None
    robot_id: str = "droid_panda"
    preset: str = "custom"

    def __post_init__(self) -> None:
        if self.vision_out_dim is None:
            self.vision_out_dim = self.hidden_size
        self.num_gdn_heads, self.gdn_head_dim = _gdn_heads(self.hidden_size)

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
}


def get_vla_config(name: str, **overrides) -> VLAStudentConfig:
    if name not in _PRESETS:
        raise KeyError(f"Unknown preset {name!r}; known: {sorted(_PRESETS)}")
    fields = dict(_PRESETS[name])
    fields.update(overrides)
    return VLAStudentConfig(preset=name, **fields)
