#!/usr/bin/env python
"""Colab / GPU smoke test: install check + HF experts + one forward pass."""

from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser(description="sleepy_orangutan Colab smoke test")
    p.add_argument("--experts", default="rt_libero_s0")
    p.add_argument("--device", default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--ensemble", action="store_true")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device} cuda={torch.cuda.is_available()}")

    try:
        import sleepy_orangutan

        print(f"sleepy_orangutan={sleepy_orangutan.__version__}")
    except ImportError as exc:
        print("sleepy_orangutan not installed:", exc, file=sys.stderr)
        return 1

    from sleepy_orangutan.experts import (
        ExpertEnsemble,
        format_catalog_table,
        load_expert_policies,
        parse_expert_combo,
    )
    from sleepy_orangutan.vla.inference import InferenceConfig, VLAController

    print("\n--- catalog ---")
    print(format_catalog_table())

    tokens = parse_expert_combo(args.experts) if args.ensemble or "," in args.experts else [args.experts]
    policies = load_expert_policies(tokens, device=device)
    cfg = policies[0].config
    images = torch.rand(1, cfg.num_cameras, 3, cfg.image_size, cfg.image_size, device=device)
    state = torch.randn(1, cfg.state_dim, device=device)

    for i, pol in enumerate(policies):
        pol.eval()
        with torch.inference_mode():
            chunk = pol.predict_chunk(images, state)
        print(f"[{i}] delta={pol.config.delta_type} chunk={tuple(chunk.shape)}")

    if args.benchmark and len(policies) == 1:
        ctrl = VLAController(policies[0], cfg=InferenceConfig(device=device))
        stats = ctrl.benchmark_chunk_latency(images, state)
        print(f"benchmark: {stats['chunk_latency_ms']:.2f} ms/chunk")

    cache = os.environ.get(
        "SLEEPY_ORANGUTAN_CACHE",
        os.path.expanduser("~/.cache/sleepy_orangutan"),
    )
    print(f"\nOK — cache: {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())