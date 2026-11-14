"""Load distilled GDN VLA students (self-contained sleepy_orangutan)."""

from __future__ import annotations

import os
from typing import Any

from sleepy_orangutan.weights import ensure_checkpoint


def load_policy(
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    device: str = "cpu",
    hf_repo: str | None = None,
    hf_filename: str = "student.pt",
    hf_revision: str | None = None,
    gdrive_id: str | None = None,
    url: str | None = None,
    checksum_sha256: str | None = None,
) -> Any:
    """Download (if needed) and load ``VLAStudentPolicy``."""
    from sleepy_orangutan.vla.model import VLAStudentPolicy

    path = ensure_checkpoint(
        checkpoint=checkpoint,
        hf_repo=hf_repo,
        hf_filename=hf_filename,
        hf_revision=hf_revision,
        gdrive_id=gdrive_id,
        url=url,
        checksum_sha256=checksum_sha256,
    )
    policy = VLAStudentPolicy.load_checkpoint(str(path), map_location=device)
    delta = policy.config.delta_type
    if delta not in ("gateddelta", "gateddeltanet", "stm"):
        raise ValueError(
            f"Checkpoint trunk delta:{delta!r} is not supported. "
            "Use GDN or RT/STM students from the sleepy_orangutan checkpoint catalog."
        )
    policy.eval()
    policy.to(device)
    return policy


def load_controller(
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    device: str = "cpu",
    control_hz: float = 10.0,
    **download_kw: Any,
):
    """Load policy + ``VLAController`` for chunk-at-a-time deployment."""
    from sleepy_orangutan.vla.inference import InferenceConfig, VLAController

    policy = load_policy(checkpoint=checkpoint, device=device, **download_kw)
    cfg = InferenceConfig(device=device, control_hz=control_hz)
    return VLAController(policy, cfg=cfg)
