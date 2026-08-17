# Vendored flash-linear-attention ``fla/`` (Triton on GPU) plus CPU-safe modules at package root.
# Triton-backed symbols are lazy so ``import ...naive_linear_attn_ssm`` works on CPU-only hosts.

from __future__ import annotations

from typing import TYPE_CHECKING

from minipi05.gateddeltanet.diagonal_conv_rnn import (
    DiagonalCausalConvRNN,
    LamGate,
)
from minipi05.gateddeltanet.naive_linear_attn_ssm import (
    NaiveLinearAttentionSSM,
    NaiveLinearAttentionSSMMult,
)

if TYPE_CHECKING:
    from minipi05.gateddeltanet.fla.layers.gated_deltanet import (
        GatedDeltaNet,
    )
    from minipi05.gateddeltanet.gated_deltanet_modified import (
        GatedDeltaNetModified,
    )

__all__ = [
    "GatedDeltaNet",
    "GatedDeltaNetModified",
    "NaiveLinearAttentionSSM",
    "NaiveLinearAttentionSSMMult",
    "DiagonalCausalConvRNN",
    "LamGate",
]


def __getattr__(name: str):
    if name == "GatedDeltaNet":
        from minipi05.gateddeltanet.fla.layers.gated_deltanet import (
            GatedDeltaNet,
        )

        return GatedDeltaNet
    if name == "GatedDeltaNetModified":
        from minipi05.gateddeltanet.gated_deltanet_modified import (
            GatedDeltaNetModified,
        )

        return GatedDeltaNetModified
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
