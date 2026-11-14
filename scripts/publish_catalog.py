#!/usr/bin/env python
"""Upload all ``student.pt`` files referenced in checkpoints_catalog.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Publish catalog experts to Hugging Face Hub")
    p.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sleepy_orangutan" / "checkpoints_catalog.json",
    )
    p.add_argument(
        "--student-root",
        type=Path,
        required=True,
        help="Directory with local files named {expert_id}.pt or student.pt per subfolder",
    )
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from huggingface_hub import HfApi

    raw = json.loads(args.catalog.read_text(encoding="utf-8"))
    api = HfApi()

    for row in raw.get("experts", []):
        repo = (row.get("hf_repo") or "").strip()
        expert_id = row["id"]
        if not repo:
            print(f"skip {expert_id}: no hf_repo")
            continue
        candidates = [
            args.student_root / f"{expert_id}.pt",
            args.student_root / expert_id / "student.pt",
            args.student_root / f"student_{expert_id}.pt",
        ]
        ckpt = next((c for c in candidates if c.is_file()), None)
        if ckpt is None:
            print(f"skip {expert_id}: no local file in {args.student_root}")
            continue
        hf_file = row.get("hf_file") or "student.pt"
        print(f"{ckpt} -> hf://{repo}/{hf_file}")
        if args.dry_run:
            continue
        api.create_repo(repo, exist_ok=True, private=args.private)
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=hf_file,
            repo_id=repo,
            commit_message=f"Add {hf_file} ({expert_id})",
        )


if __name__ == "__main__":
    main()