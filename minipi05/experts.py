"""Named VLA experts (GDN + RT) and multi-checkpoint ensembles."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

_FAMILY_ALIASES = {
    "gdn": "gdn",
    "gateddelta": "gdn",
    "gateddeltanet": "gdn",
    "rt": "rt",
    "stm": "rt",
    "runningtensor": "rt",
    "rtfla4": "rt",
    "rt4": "rt",
}


@dataclass(frozen=True)
class ExpertSpec:
    """One row in ``checkpoints_catalog.json``."""

    id: str
    family: str
    seed: int
    label: str = ""
    preset: str = ""
    hf_repo: str = ""
    hf_file: str = "student.pt"
    path_env: str = ""
    local_path: str = ""
    hf_revision: str | None = None

    def resolve_download_kwargs(self) -> dict[str, Any]:
        """Arguments for ``ensure_checkpoint`` / ``load_policy`` (HF → ``~/.cache/minipi05``)."""
        if self.local_path:
            return {"checkpoint": self.local_path}
        if self.hf_repo:
            kw: dict[str, Any] = {
                "hf_repo": self.hf_repo,
                "hf_filename": self.hf_file or "student.pt",
            }
            if self.hf_revision:
                kw["hf_revision"] = self.hf_revision
            return kw
        env_name = (self.path_env or "").strip()
        if env_name:
            p = os.environ.get(env_name, "").strip()
            if p:
                return {"checkpoint": p}
        raise FileNotFoundError(
            f"Expert {self.id!r} has no ``hf_repo`` in the catalog. "
            "Upload ``student.pt`` to Hugging Face and set ``hf_repo`` in "
            "checkpoints_catalog.json."
        )


def _catalog_path() -> Path:
    return Path(__file__).with_name("checkpoints_catalog.json")


def load_catalog() -> list[ExpertSpec]:
    path = _catalog_path()
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[ExpertSpec] = []
    for row in raw.get("experts", []):
        out.append(
            ExpertSpec(
                id=str(row["id"]),
                family=_normalize_family(str(row.get("family", ""))),
                seed=int(row.get("seed", 0)),
                label=str(row.get("label", "")),
                preset=str(row.get("preset", "") or ""),
                hf_repo=str(row.get("hf_repo", "") or ""),
                hf_file=str(row.get("hf_file", "student.pt") or "student.pt"),
                path_env=str(row.get("path_env", "") or ""),
                local_path=str(row.get("local_path", "") or ""),
                hf_revision=row.get("hf_revision"),
            )
        )
    return out


def _normalize_family(name: str) -> str:
    key = (name or "").strip().lower()
    return _FAMILY_ALIASES.get(key, key)


def list_expert_ids() -> list[str]:
    return [e.id for e in load_catalog()]


def format_catalog_table() -> str:
    lines = ["id           family  seed  preset              label"]
    for e in load_catalog():
        lines.append(
            f"{e.id:<14} {e.family:<6} {e.seed:<5} {(e.preset or '-'):<19} {e.label}"
        )
    return "\n".join(lines)


def get_expert_by_id(expert_id: str) -> ExpertSpec:
    key = (expert_id or "").strip()
    for e in load_catalog():
        if e.id == key:
            return e
    raise KeyError(f"Unknown expert id {key!r}. Known: {list_expert_ids()}")


def _parse_combo_token(token: str) -> tuple[str, int] | None:
    """``gdn:0`` → family+seed; bare id → None."""
    token = token.strip()
    if not token:
        raise ValueError("empty expert token")
    if ":" in token:
        fam, seed_s = token.split(":", 1)
        return _normalize_family(fam), int(seed_s)
    return None


def resolve_expert_tokens(
    tokens: Sequence[str],
    *,
    preset: str | None = None,
) -> list[ExpertSpec]:
    """
    Map CLI tokens to catalog rows.

    * ``gdn:0`` — family + seed (optional ``preset`` disambiguates LIBERO sizes).
    * ``rt_libero_10m_s0`` — exact catalog ``id``.
    """
    catalog = load_catalog()
    resolved: list[ExpertSpec] = []
    for raw in tokens:
        parsed = _parse_combo_token(raw)
        if parsed is None:
            resolved.append(get_expert_by_id(raw))
            continue
        fam, seed = parsed
        matches = [
            e
            for e in catalog
            if e.family == fam
            and e.seed == seed
            and (not preset or not e.preset or e.preset == preset)
        ]
        if not matches:
            raise KeyError(
                f"No catalog expert for {raw!r}"
                + (f" preset={preset!r}" if preset else "")
                + ". Try --list-experts."
            )
        if len(matches) > 1 and not preset:
            ids = ", ".join(m.id for m in matches)
            raise ValueError(
                f"Ambiguous {raw!r}: matches {ids}. Pass --preset s100_libero_10m "
                "or use a full id (e.g. rt_libero_10m_s0)."
            )
        resolved.append(matches[0])
    return resolved


def parse_expert_combo(spec: str) -> list[str]:
    """``gdn:0,rt:0,gdn:1`` → token list."""
    spec = (spec or "").strip()
    if not spec:
        return []
    return [t.strip() for t in re.split(r"[,;\s]+", spec) if t.strip()]


def load_expert_policies(
    tokens: Sequence[str],
    *,
    device: str = "cpu",
    preset: str | None = None,
) -> list[Any]:
    from minipi05.load import load_policy

    policies = []
    for entry in resolve_expert_tokens(list(tokens), preset=preset):
        kw = entry.resolve_download_kwargs()
        policies.append(load_policy(device=device, **kw))
    return policies


class ExpertEnsemble:
    """Weighted mean over action-chunk predictions from several students."""

    def __init__(
        self,
        policies: Sequence[Any],
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        if not policies:
            raise ValueError("ExpertEnsemble needs at least one policy")
        self.policies = list(policies)
        if weights is None:
            w = [1.0] * len(self.policies)
        else:
            w = [float(x) for x in weights]
            if len(w) != len(self.policies):
                raise ValueError("weights length must match number of experts")
        total = sum(w)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = [x / total for x in w]

    @torch.inference_mode()
    def predict_chunk(self, images, state, subgoal=None):
        chunks = [p.predict_chunk(images, state, subgoal=subgoal) for p in self.policies]
        stacked = torch.stack(chunks, dim=0)
        w = torch.tensor(self.weights, device=stacked.device, dtype=stacked.dtype)
        w = w.view(-1, *([1] * (stacked.ndim - 1)))
        return (stacked * w).sum(dim=0)

    def describe(self) -> str:
        parts = []
        for p, wt in zip(self.policies, self.weights):
            cfg = p.config
            notes = getattr(p, "checkpoint_args", {}) or {}
            exp = notes.get("exp_id", "?")
            parts.append(
                f"{wt:.3g}× preset={cfg.preset} delta={cfg.delta_type} exp={exp}"
            )
        return " | ".join(parts)