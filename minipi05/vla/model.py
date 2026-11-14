"""Full VLA student: vision + proprio + subgoal → action chunk."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn

from minipi05.vla.config import VLAStudentConfig
from minipi05.vla.policy import GDNActionTrunk, count_parameters
from minipi05.vla.subgoals import SubgoalSpec
from minipi05.vla.vision import SmallMultiCamEncoder


class VLAStudentPolicy(nn.Module):
    """
    Edge VLA student distilled from a heavy teacher (e.g. openpi π₀.₅).

    Inputs:
        images (B, num_cameras, 3, H, W)
        state (B, state_dim) proprioception
        subgoal (B, subgoal_dim) optional structured goal from planner

    Output:
        actions (B, action_horizon, action_dim)
    """

    def __init__(self, config: VLAStudentConfig) -> None:
        super().__init__()
        self.config = config
        fuse_in = config.state_dim + config.subgoal_dim + config.vision_out_dim

        self.vision = SmallMultiCamEncoder(
            num_cameras=config.num_cameras,
            image_size=config.image_size,
            out_dim=config.vision_out_dim,
            width_mult=config.vision_width_mult,
        )
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.trunk = GDNActionTrunk(config)
        self.action_head = nn.Linear(config.hidden_size, config.action_dim)

    def encode_observation(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        subgoal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vis = self.vision(images)
        if subgoal is None:
            subgoal = torch.zeros(
                state.shape[0], self.config.subgoal_dim, device=state.device, dtype=state.dtype
            )
        fused = torch.cat([vis, state, subgoal], dim=-1)
        return self.fuse(fused)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        subgoal: torch.Tensor | None = None,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        obs = self.encode_observation(images, state, subgoal)
        seq = self.trunk(obs)
        actions = self.action_head(seq)
        if return_features:
            return actions, obs
        return actions

    @torch.inference_mode()
    def predict_chunk(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        subgoal: SubgoalSpec | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper for deployment."""
        if isinstance(subgoal, SubgoalSpec):
            sg = subgoal.to_tensor(
                batch_size=images.shape[0],
                device=images.device,
                dim=self.config.subgoal_dim,
            )
        else:
            sg = subgoal
        return self.forward(images, state, sg)

    def num_parameters(self) -> int:
        return count_parameters(self)

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    @classmethod
    def from_config(cls, config: VLAStudentConfig) -> "VLAStudentPolicy":
        return cls(config)

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "VLAStudentPolicy":
        from minipi05.vla.config import get_vla_config

        return cls(get_vla_config(preset, **overrides))

    def save_checkpoint(
        self,
        path: str,
        *,
        args_dict: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "config": self.config_dict(),
            "state_dict": self.state_dict(),
        }
        if args_dict is not None:
            payload["args"] = args_dict
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, *, map_location: str | torch.device = "cpu") -> "VLAStudentPolicy":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        from minipi05.vla.config import vla_config_from_saved_dict

        config = vla_config_from_saved_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        if "args" in payload:
            model.checkpoint_args = payload["args"]
        return model
