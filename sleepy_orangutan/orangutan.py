"""Public API: download LIBERO students from Hugging Face and run inference."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from sleepy_orangutan.experts import ExpertEnsemble, load_expert_policies, parse_expert_combo


class SleepyOrangutan:
    """
    Edge VLA runner: downloads ``student.pt`` from the shipped catalog on first use.

    Hugging Face repos in the catalog stay named ``LucaPeroxide/minipi05-libero-*``
    (legacy publish ids); no token needed when repos are public.

    Examples::

        from sleepy_orangutan import SleepyOrangutan

        bot = SleepyOrangutan.libero_rt(seed=0)
        chunk = bot.predict_chunk(images, state)
    """

    def __init__(
        self,
        backend: Any,
        *,
        device: str,
        expert_ids: Sequence[str] | None = None,
    ) -> None:
        self.device = device
        self.backend = backend
        self.expert_ids = list(expert_ids or [])
        self._chunk: torch.Tensor | None = None
        self._chunk_idx = 0
        self._horizon = _action_horizon(backend)

    @staticmethod
    def _default_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_catalog(cls, expert_id: str, *, device: str | None = None) -> SleepyOrangutan:
        device = device or cls._default_device()
        policies = load_expert_policies([expert_id], device=device)
        return cls(policies[0], device=device, expert_ids=[expert_id])

    @classmethod
    def from_combo(
        cls,
        spec: str,
        *,
        device: str | None = None,
        preset: str | None = None,
        weights: Sequence[float] | None = None,
    ) -> SleepyOrangutan:
        device = device or cls._default_device()
        tokens = parse_expert_combo(spec)
        policies = load_expert_policies(tokens, device=device, preset=preset)
        backend = policies[0] if len(policies) == 1 else ExpertEnsemble(policies, weights=weights)
        return cls(backend, device=device, expert_ids=tokens)

    @classmethod
    def libero_rt(
        cls,
        *,
        device: str | None = None,
        seed: int | None = None,
        seeds: Sequence[int] = (0, 1, 2, 3),
        weights: Sequence[float] | None = None,
    ) -> SleepyOrangutan:
        device = device or cls._default_device()
        if seed is not None:
            return cls.from_catalog(f"rt_libero_s{int(seed)}", device=device)
        tokens = [f"rt_libero_s{int(s)}" for s in seeds]
        policies = load_expert_policies(tokens, device=device)
        backend = ExpertEnsemble(policies, weights=weights) if len(policies) > 1 else policies[0]
        return cls(backend, device=device, expert_ids=tokens)

    @classmethod
    def libero_gdn(
        cls,
        *,
        device: str | None = None,
        seed: int | None = None,
        seeds: Sequence[int] = (0, 1, 2, 3),
        weights: Sequence[float] | None = None,
    ) -> SleepyOrangutan:
        device = device or cls._default_device()
        if seed is not None:
            return cls.from_catalog(f"gdn_libero_s{int(seed)}", device=device)
        tokens = [f"gdn_libero_s{int(s)}" for s in seeds]
        policies = load_expert_policies(tokens, device=device)
        backend = ExpertEnsemble(policies, weights=weights) if len(policies) > 1 else policies[0]
        return cls(backend, device=device, expert_ids=tokens)

    def reset(self) -> None:
        self._chunk = None
        self._chunk_idx = 0
        if hasattr(self.backend, "reset_inference_cache"):
            self.backend.reset_inference_cache()

    def describe(self) -> str:
        if isinstance(self.backend, ExpertEnsemble):
            return self.backend.describe()
        cfg = self.backend.config
        notes = getattr(self.backend, "checkpoint_args", {}) or {}
        return (
            f"preset={cfg.preset} delta={cfg.delta_type} "
            f"experts={self.expert_ids or ['single']} exp={notes.get('exp_id', '?')}"
        )

    @torch.inference_mode()
    def predict_chunk(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        state = state.to(self.device)
        if isinstance(self.backend, ExpertEnsemble):
            return self.backend.predict_chunk(images, state)
        return self.backend.predict_chunk(images, state)

    @torch.inference_mode()
    def step(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self._chunk is None or self._chunk_idx >= self._horizon:
            chunk = self.predict_chunk(images, state)
            self._chunk = chunk[0].detach()
            self._chunk_idx = 0
        action = self._chunk[self._chunk_idx]
        self._chunk_idx += 1
        return action


# Deprecated alias (pre-rename)
HairyOrangutan = SleepyOrangutan


def _action_horizon(backend: Any) -> int:
    if isinstance(backend, ExpertEnsemble):
        return int(backend.policies[0].config.action_horizon)
    return int(backend.config.action_horizon)