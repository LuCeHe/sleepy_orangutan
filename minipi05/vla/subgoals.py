"""Structured subgoal interface between planner and low-level controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SubgoalSpec:
    """
    Low-dimensional conditioning for the control policy.

    The planner (heavier model or FSM) emits this; the GDN student maps
    subgoal + proprio + cameras → action chunk.
    """

    skill_id: int = 0
    target_pose: torch.Tensor | None = None  # (7,) or (action_dim,)
    object_id: int = -1
    embedding: torch.Tensor | None = None  # optional learned (subgoal_dim,)

    def to_tensor(self, *, batch_size: int, device: torch.device, dim: int) -> torch.Tensor:
        """Pack subgoal into a fixed-size vector (B, dim)."""
        if self.embedding is not None:
            emb = self.embedding
            if emb.dim() == 1:
                emb = emb.unsqueeze(0).expand(batch_size, -1)
            if emb.shape[-1] != dim:
                raise ValueError(f"subgoal embedding dim {emb.shape[-1]} != expected {dim}")
            return emb.to(device)

        out = torch.zeros(batch_size, dim, device=device)
        out[:, 0] = float(self.skill_id)
        out[:, 1] = float(self.object_id)
        if self.target_pose is not None:
            pose = self.target_pose.to(device)
            if pose.dim() == 1:
                pose = pose.unsqueeze(0).expand(batch_size, -1)
            n = min(pose.shape[-1], dim - 2)
            out[:, 2 : 2 + n] = pose[:, :n]
        return out
