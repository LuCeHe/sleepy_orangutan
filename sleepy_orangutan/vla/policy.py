"""Action trunk for action-chunk prediction (via PebbleTrail ``get_analyzers``)."""

from __future__ import annotations

import torch
import torch.nn as nn

from sleepy_orangutan.trunk import get_analyzers
from sleepy_orangutan.vla.config import VLAStudentConfig


def _ssm_layer_forward(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run one SSM layer; supports GDN tuples, FLA modules, and plain STM tensors."""
    try:
        out = layer(x, use_cache=False)
    except TypeError:
        out = layer(x)
    if isinstance(out, tuple):
        return out[0]
    return out


class GDNActionTrunk(nn.Module):
    """
    Sequence model over the action horizon.

    Observation context is broadcast onto learned query tokens, then processed
    by a stack of SSM layers from ``get_analyzers`` (``config.notes`` / ``delta:…``,
    same as MQAR ``train.py``). At inference (q_len <= 64, eval mode) GDN layers
    automatically use fused_recurrent kernels.
    """

    def __init__(self, config: VLAStudentConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.action_queries = nn.Parameter(
            torch.randn(config.action_horizon, h) * 0.02
        )
        self.obs_to_seq = nn.Linear(h, h)
        self.in_norm = nn.LayerNorm(h)
        layers: list[nn.Module] = []
        trunk_notes = config.trunk_notes
        for i in range(config.num_gdn_layers):
            delta_cfg = config.to_delta_config(layer_idx=i)
            _analyzer_fn, rnn = get_analyzers(delta_cfg, layer_idx=i)
            if rnn is None:
                raise ValueError(
                    f"VLA trunk notes {trunk_notes!r} did not build an SSM module "
                    f"(layer {i}; delta={config.delta_type!r}). "
                    "Set config.notes with delta:gateddelta or delta:stm (RT needs PebbleTrail)."
                )
            layers.append(rnn)
        self.layers = nn.ModuleList(layers)
        self.out_norm = nn.LayerNorm(h)

    def forward(self, obs_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_embed: (B, hidden_size) fused observation embedding.

        Returns:
            (B, action_horizon, hidden_size) sequence states.
        """
        b = obs_embed.shape[0]
        seq = self.action_queries.unsqueeze(0).expand(b, -1, -1)
        ctx = self.obs_to_seq(obs_embed).unsqueeze(1)
        x = self.in_norm(seq + ctx)
        for layer in self.layers:
            x = _ssm_layer_forward(layer, x)
        return self.out_norm(x)


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
