#!/usr/bin/env python
"""Colab / GPU smoke test: install check + HF experts + one forward pass."""

from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser(description="minipi05 Colab smoke test")
    p.add_argument(
        "--experts",
        default="rt_libero_s0",
        help="Catalog id or combo (e.g. rt_libero_s0 or gdn:0,rt:0)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--ensemble", action="store_true", help="Treat --experts as combo")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device} cuda={torch.cuda.is_available()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import minipi05

        print(f"minipi05={minipi05.__version__}")
    except ImportError as exc:
        print("minipi05 not installed:", exc, file=sys.stderr)
        return 1

    from minipi05.experts import (
        ExpertEnsemble,
        format_catalog_table,
        load_expert_policies,
        parse_expert_combo,
    )
    from minipi05.vla.inference import InferenceConfig, VLAController

    print("\n--- catalog ---")
    print(format_catalog_table())

    tokens = parse_expert_combo(args.experts) if args.ensemble or "," in args.experts else [args.experts]
    print(f"\nLoading experts: {tokens}")
    try:
        policies = load_expert_policies(tokens, device=device)
    except Exception as exc:
        print(f"LOAD FAILED: {exc}", file=sys.stderr)
        print(
            "Check HF_TOKEN for private repos and that hf_repo exists in checkpoints_catalog.json",
            file=sys.stderr,
        )
        return 2

    cfg = policies[0].config
    images = torch.rand(1, cfg.num_cameras, 3, cfg.image_size, cfg.image_size, device=device)
    state = torch.randn(1, cfg.state_dim, device=device)

    for i, pol in enumerate(policies):
        delta = pol.config.delta_type
        args_ckpt = getattr(pol, "checkpoint_args", {}) or {}
        print(
            f"\n[{i}] preset={pol.config.preset} delta={delta} "
            f"params={pol.num_parameters() / 1e6:.2f}M exp_id={args_ckpt.get('exp_id', '?')}"
        )
        pol.eval()
        with torch.inference_mode():
            chunk = pol.predict_chunk(images, state)
        print(f"    predict_chunk OK shape={tuple(chunk.shape)} mean={chunk.float().mean():.4f}")

    if len(policies) > 1 and args.ensemble:
        blend = ExpertEnsemble(policies)
        with torch.inference_mode():
            out = blend.predict_chunk(images, state)
        print(f"\nensemble OK shape={tuple(out.shape)}")

    if args.benchmark and len(policies) == 1:
        ctrl = VLAController(policies[0], cfg=InferenceConfig(device=device))
        stats = ctrl.benchmark_chunk_latency(images, state)
        print(f"\nbenchmark: {stats['chunk_latency_ms']:.2f} ms/chunk")

    cache = os.environ.get("MINIPI05_CACHE", os.path.expanduser("~/.cache/minipi05"))
    print(f"\nOK — weights cache: {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())