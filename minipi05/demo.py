#!/usr/bin/env python
"""Smoke demo: distilled student vs random-init baseline on synthetic observations."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch


def _action_summary(action: torch.Tensor) -> str:
    a = action.detach().float().cpu()
    return (
        f"shape={tuple(a.shape)} "
        f"mean={a.mean().item():+.4f} std={a.std().item():.4f} "
        f"min={a.min().item():+.4f} max={a.max().item():+.4f} "
        f"first4={a[:4].tolist()}"
    )


def _run_forward(policy, images: torch.Tensor, state: torch.Tensor, device: str) -> torch.Tensor:
    from minipi05.vla.inference import InferenceConfig, VLAController

    ctrl = VLAController(policy, cfg=InferenceConfig(device=device))
    ctrl.reset()
    return ctrl.step(images, state)


def _benchmark_policy(policy, images: torch.Tensor, state: torch.Tensor, device: str) -> dict:
    from minipi05.vla.inference import InferenceConfig, VLAController

    ctrl = VLAController(policy, cfg=InferenceConfig(device=device))
    return ctrl.benchmark_chunk_latency(images, state)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="minipi05 demo: distilled checkpoint vs random-init on random obs"
    )
    p.add_argument("--checkpoint", type=str, default=None, help="Local student.pt")
    p.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HF model repo (set MINIPI05_HF_REPO or pass HF_TOKEN on headless nodes)",
    )
    p.add_argument("--hf-file", type=str, default="student.pt")
    p.add_argument("--gdrive-id", type=str, default=None)
    p.add_argument("--url", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure wall-clock ms per action-chunk forward (vision+GDN), not task success",
    )
    p.add_argument(
        "--skip-random",
        action="store_true",
        help="Only run the distilled checkpoint (skip untrained baseline)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG for synthetic images/state")
    p.add_argument(
        "--list-experts",
        action="store_true",
        help="Print shipped checkpoint catalog (GDN + RT seeds) and exit",
    )
    p.add_argument(
        "--experts",
        type=str,
        default=None,
        help="Expert combo, e.g. gdn:0,rt:0 or rt_libero_10m_s0,gdn:1 "
        "(see checkpoints_catalog.json)",
    )
    p.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Disambiguate family:seed when multiple LIBERO presets exist",
    )
    p.add_argument(
        "--expert-weights",
        type=str,
        default=None,
        help="Comma-separated blend weights for --experts (default: uniform)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_experts:
        from minipi05.experts import format_catalog_table

        print(format_catalog_table())
        print(
            "\nWeights download from each row's hf_repo into "
            f"{os.environ.get('MINIPI05_CACHE', '~/.cache/minipi05')} on first use."
        )
        return

    torch.manual_seed(args.seed)

    from minipi05.load import load_policy
    from minipi05.vla.config import VLAStudentConfig
    from minipi05.vla.model import VLAStudentPolicy

    print("=== Synthetic observation (same for both policies) ===")
    print(
        "Random uint8-like images in [0,1] and Gaussian proprio — NOT a real DROID frame. "
        "Actions are sanity checks only, not meaningful robot commands."
    )

    use_ensemble = bool(args.experts)
    if use_ensemble:
        from minipi05.experts import ExpertEnsemble, load_expert_policies, parse_expert_combo

        tokens = parse_expert_combo(args.experts)
        weights = None
        if args.expert_weights:
            weights = [float(x) for x in args.expert_weights.split(",")]
        policies = load_expert_policies(
            tokens, device=args.device, preset=args.preset
        )
        ensemble = ExpertEnsemble(policies, weights=weights)
        cfg = policies[0].config
        print(f"\n=== Expert ensemble ({len(policies)} checkpoints) ===")
        print(ensemble.describe())
    else:
        policy = load_policy(
            checkpoint=args.checkpoint,
            device=args.device,
            hf_repo=args.hf_repo,
            gdrive_id=args.gdrive_id,
            url=args.url,
            hf_filename=args.hf_file,
        )
        cfg = policy.config
    b = 1
    images = torch.rand(b, cfg.num_cameras, 3, cfg.image_size, cfg.image_size)
    state = torch.randn(b, cfg.state_dim)

    if not use_ensemble:
        ckpt_args = getattr(policy, "checkpoint_args", {}) or {}
        exp_id = ckpt_args.get("exp_id", "?")
        notes = ckpt_args.get("notes") or cfg.notes

        print(f"\n=== Distilled checkpoint ===")
        print(f"preset={cfg.preset} params={policy.num_parameters() / 1e6:.2f}M")
        print(f"trunk={cfg.trunk_notes} exp_id={exp_id}")
        if notes:
            print(f"train_notes={notes}")

        distilled_action = _run_forward(policy, images, state, args.device)
    else:
        distilled_action = ensemble.predict_chunk(images, state)[0]
    print(f"action: {_action_summary(distilled_action)}")

    if args.benchmark and not use_ensemble:
        stats = _benchmark_policy(policy, images, state, args.device)
        print(
            f"benchmark: chunk_latency_ms={stats['chunk_latency_ms']:.2f} "
            f"(repeats={int(stats['repeats'])}, one predict_chunk = {cfg.action_horizon} steps)"
        )

    if not args.skip_random and not use_ensemble:
        print(f"\n=== Random init (same architecture, untrained weights) ===")
        print(
            "Same tensor shapes and FLOPs as distilled; actions are arbitrary. "
            "Use this to see that loading weights changes outputs but not necessarily latency."
        )
        rand_cfg = VLAStudentConfig(**asdict(cfg))
        random_policy = VLAStudentPolicy(rand_cfg).to(args.device).eval()
        random_action = _run_forward(random_policy, images, state, args.device)
        print(f"action: {_action_summary(random_action)}")

        if args.benchmark:
            rstats = _benchmark_policy(random_policy, images, state, args.device)
            print(
                f"benchmark: chunk_latency_ms={rstats['chunk_latency_ms']:.2f} "
                f"(repeats={int(rstats['repeats'])})"
            )

        delta = (distilled_action - random_action).float().cpu()
        print(
            f"distilled vs random |delta| mean={delta.abs().mean().item():.4f} "
            f"max={delta.abs().max().item():.4f}"
        )

    print(
        "\nNote: benchmark measures inference speed on fake data. "
        "For quality, use offline Huber / closed-loop metrics from training, or robot/sim eval."
    )


if __name__ == "__main__":
    main()
