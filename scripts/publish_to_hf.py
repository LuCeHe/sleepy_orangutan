#!/usr/bin/env python
"""Upload a distilled student.pt to Hugging Face Hub (one-time publish helper)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Publish student.pt to Hugging Face Hub")
    p.add_argument("repo_id", help="e.g. your-username/minipi05-droid-stm-10m")
    p.add_argument("checkpoint", type=Path, help="Path to student.pt")
    p.add_argument("--filename", default="student.pt")
    p.add_argument(
        "--private",
        action="store_true",
        help="Default is public (no HF token needed for Colab download).",
    )
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Not found: {args.checkpoint}")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    api.upload_file(
        path_or_fileobj=str(args.checkpoint),
        path_in_repo=args.filename,
        repo_id=args.repo_id,
        commit_message=f"Add {args.filename}",
    )
    print(f"Uploaded {args.checkpoint} -> hf://{args.repo_id}/{args.filename}")
    print(f"Friends run: export MINIPI05_HF_REPO={args.repo_id} && minipi05-demo")


if __name__ == "__main__":
    main()
