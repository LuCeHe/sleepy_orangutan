"""GDN-only action trunk builder (minipi05 ships ``delta:gateddelta`` only)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import torch.nn as nn


def _gdn_num_heads_head_dim(hidden_size: int) -> tuple[int, int]:
    key_dim = max(1, int(round(0.75 * hidden_size)))
    for num_heads in (6, 8, 4, 2, 1):
        if key_dim % num_heads == 0:
            return num_heads, key_dim // num_heads
    return 1, key_dim


def _delta_type(notes: str) -> str:
    m = re.search(r"delta:([^_:]+)", notes or "")
    return m.group(1).lower() if m else "gateddelta"


def build_gdn_layer(config: SimpleNamespace, *, layer_idx: int = 0) -> nn.Module:
    """Build one Gated DeltaNet layer matching PebbleTrail ``delta:gateddelta``."""
    notes = getattr(config, "notes", "") or ""
    delta_type = _delta_type(notes)
    if delta_type not in ("gateddelta", "gateddeltanet"):
        raise ValueError(
            f"minipi05 supports GDN students only (delta:gateddelta); got delta:{delta_type!r}. "
            "Retrain with vla_distill_delta:gateddelta or use a GDN checkpoint."
        )

    gdn_heads, gdn_head_dim = _gdn_num_heads_head_dim(int(config.hidden_size))
    gdn_kw = dict(
        hidden_size=int(config.hidden_size),
        expand_v=2,
        use_gate=True,
        num_heads=gdn_heads,
        head_dim=gdn_head_dim,
        layer_idx=layer_idx,
    )
    try:
        import triton  # noqa: F401

        from minipi05.gateddeltanet.fla.layers.gated_deltanet import GatedDeltaNet

        return GatedDeltaNet(**gdn_kw)
    except ImportError:
        import warnings

        from minipi05.gateddeltanet.gated_deltanet_naive import GatedDeltaNetNaive

        warnings.warn(
            "triton unavailable; using GatedDeltaNetNaive (PyTorch GDN on CPU).",
            stacklevel=2,
        )
        return GatedDeltaNetNaive(**gdn_kw)


def get_analyzers(config: SimpleNamespace, layer_idx: int = 0):
    """PebbleTrail-compatible hook: returns (passthrough_fn, rnn_module)."""
    rnn = build_gdn_layer(config, layer_idx=layer_idx)
    return (lambda x: x), rnn
