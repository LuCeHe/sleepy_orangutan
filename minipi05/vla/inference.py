"""Deployment inference loop for edge VLA student."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from minipi05.vla.model import VLAStudentPolicy
from minipi05.vla.subgoals import SubgoalSpec


@dataclass
class InferenceConfig:
    control_hz: float = 10.0
    chunk_interpolation: bool = True
    device: str = "cpu"
    dtype: torch.dtype = torch.bfloat16


class VLAController:
    """
    Runs the student at ``control_hz``, emitting action chunks for MCU interpolation.

    The policy forward pass produces ``action_horizon`` steps; the MCU (or a simple
    linear interpolator here) fills between chunk boundaries at higher rate.
    """

    def __init__(
        self,
        policy: VLAStudentPolicy,
        *,
        cfg: InferenceConfig | None = None,
    ) -> None:
        self.policy = policy
        self.cfg = cfg or InferenceConfig()
        self._last_chunk: torch.Tensor | None = None
        self._chunk_idx = 0

    def reset(self) -> None:
        self._last_chunk = None
        self._chunk_idx = 0

    @torch.inference_mode()
    def step(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        subgoal: SubgoalSpec | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return a single action vector (action_dim,) for the current control tick.

        Refreshes the chunk every ``action_horizon`` calls at ``control_hz``.
        """
        if self._last_chunk is None or self._chunk_idx >= self.policy.config.action_horizon:
            chunk = self.policy.predict_chunk(
                images.to(self.cfg.device),
                state.to(self.cfg.device),
                subgoal=subgoal,
            )
            self._last_chunk = chunk[0].detach().cpu()
            self._chunk_idx = 0

        action = self._last_chunk[self._chunk_idx]
        self._chunk_idx += 1
        return action

    def benchmark_chunk_latency(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        *,
        warmup: int = 3,
        repeats: int = 20,
    ) -> dict[str, float]:
        """Wall-clock ms per chunk (vision + GDN fused_recurrent path in eval mode)."""
        self.policy.eval()
        device = torch.device(self.cfg.device)
        images = images.to(device)
        state = state.to(device)

        for _ in range(warmup):
            self.policy.predict_chunk(images, state)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            self.policy.predict_chunk(images, state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0 / repeats
        return {"chunk_latency_ms": dt_ms, "repeats": float(repeats)}
