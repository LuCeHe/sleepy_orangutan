"""Download distilled student checkpoints on first use."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

DEFAULT_HF_REPO = os.environ.get("MINIPI05_HF_REPO", "")
DEFAULT_HF_FILENAME = os.environ.get("MINIPI05_HF_FILE", "student.pt")
DEFAULT_GDRIVE_FILE_ID = os.environ.get("MINIPI05_GDRIVE_ID", "")
DEFAULT_CHECKSUM = os.environ.get("MINIPI05_CHECKSUM_SHA256", "")

_CACHE_DIR = Path(os.environ.get("MINIPI05_CACHE", Path.home() / ".cache" / "minipi05"))


def cache_dir() -> Path:
    path = _CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_checkpoint_path(name: str = "student.pt") -> Path:
    return cache_dir() / name


def hf_cache_path(repo_id: str, filename: str = "student.pt") -> Path:
    """One cached file per Hub repo (so GDN + RT seeds do not overwrite ``student.pt``)."""
    safe = repo_id.strip().replace("/", "__")
    return cache_dir() / f"{safe}__{filename}"


def _verify_sha256(path: Path, expected: str) -> None:
    if not expected:
        return
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected sha256={expected}, got {got}"
        )


def download_from_huggingface(
    *,
    repo_id: str,
    filename: str = DEFAULT_HF_FILENAME,
    dest: Path | None = None,
    revision: str | None = None,
) -> Path:
    import shutil
    from huggingface_hub import hf_hub_download

    dest = dest or hf_cache_path(repo_id, filename)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    return dest


def download_from_google_drive(
    *,
    file_id: str,
    dest: Path | None = None,
) -> Path:
    """
    Download a public Google Drive file by ID.

    Prefer Hugging Face Hub when possible — Drive links are brittle (quota, redirects).
    """
    dest = dest or default_checkpoint_path("student.pt")
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Google Drive download requires gdown: pip install gdown"
        ) from exc

    url = f"https://drive.google.com/uc?id={file_id}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(dest), quiet=False)
    if not dest.is_file():
        raise RuntimeError(f"Google Drive download failed for file_id={file_id}")
    return dest


def download_from_url(url: str, *, dest: Path | None = None) -> Path:
    import requests

    dest = dest or default_checkpoint_path("student.pt")
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return dest


def ensure_checkpoint(
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    hf_repo: str | None = None,
    hf_filename: str = DEFAULT_HF_FILENAME,
    hf_revision: str | None = None,
    gdrive_id: str | None = None,
    url: str | None = None,
    checksum_sha256: str | None = None,
) -> Path:
    """
    Resolve a local ``student.pt`` path.

    Priority:
      1. ``checkpoint`` if it exists on disk
      2. cached file in ``~/.cache/minipi05/``
      3. ``hf_repo`` / env ``MINIPI05_HF_REPO``
      4. ``gdrive_id`` / env ``MINIPI05_GDRIVE_ID``
      5. direct ``url``
    """
    checksum = checksum_sha256 or DEFAULT_CHECKSUM

    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if path.is_file():
            _verify_sha256(path, checksum)
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    repo = hf_repo or DEFAULT_HF_REPO
    if repo:
        cached = hf_cache_path(repo, hf_filename)
        if cached.is_file() and cached.stat().st_size > 0:
            _verify_sha256(cached, checksum)
            return cached
        path = download_from_huggingface(
            repo_id=repo,
            filename=hf_filename,
            dest=cached,
            revision=hf_revision,
        )
        _verify_sha256(path, checksum)
        return path

    cached = default_checkpoint_path(hf_filename)
    if cached.is_file() and cached.stat().st_size > 0:
        _verify_sha256(cached, checksum)
        return cached

    drive_id = gdrive_id or DEFAULT_GDRIVE_FILE_ID
    if drive_id:
        path = download_from_google_drive(file_id=drive_id, dest=cached)
        _verify_sha256(path, checksum)
        return path

    if url:
        path = download_from_url(url, dest=cached)
        _verify_sha256(path, checksum)
        return path

    raise FileNotFoundError(
        "No checkpoint available. Pass --checkpoint PATH, or set one of:\n"
        "  MINIPI05_HF_REPO + MINIPI05_HF_FILE (recommended)\n"
        "  MINIPI05_GDRIVE_ID (needs pip install gdown)\n"
        "  --url https://...\n"
        f"Cache directory: {cache_dir()}"
    )
