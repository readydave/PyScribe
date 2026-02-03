"""Model download helpers with optional in-app progress callbacks."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Callable

import requests
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from tqdm.auto import tqdm

from .hf_auth_service import get_hf_token

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]

# Shorthand model names used by faster-whisper -> public CT2 repos.
MODEL_REPO_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def resolve_repo_id(model_name: str) -> str | None:
    model_name = normalize_model_name(model_name)
    if os.path.isdir(model_name):
        return None
    if "/" in model_name:
        return model_name
    return MODEL_REPO_MAP.get(model_name)


def normalize_model_name(model_name: str) -> str:
    """
    Accept either repo id (owner/repo) or full Hugging Face URL and normalize to repo id.
    """
    raw = (model_name or "").strip()
    if not raw:
        return raw
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        host = (parsed.netloc or "").lower()
        if host in {"huggingface.co", "www.huggingface.co"}:
            parts = [p for p in parsed.path.split("/") if p]
            # handle /owner/repo, /models/owner/repo, /spaces/owner/repo
            if parts and parts[0] in {"models", "spaces", "datasets"}:
                parts = parts[1:]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    return raw


def is_model_cached(model_name: str) -> bool:
    """Returns True when a Hugging Face model repo already has cached snapshots."""
    repo_id = resolve_repo_id(model_name)
    if not repo_id:
        return True

    return _find_cached_snapshot_path(repo_id) is not None


def estimate_model_download_size_bytes(model_name: str) -> int | None:
    """Best-effort remote model size estimate in bytes."""
    repo_id = resolve_repo_id(model_name)
    if not repo_id:
        return None
    token = get_hf_token()
    try:
        api = HfApi(token=token)
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        total = 0
        found = False
        for sibling in info.siblings or []:
            size = getattr(sibling, "size", None)
            if isinstance(size, int):
                total += size
                found = True
        if found and total > 0:
            return total
        # Fallback: estimate via HEAD Content-Length for each file.
        return _estimate_size_via_head(repo_id=repo_id, token=token, siblings=info.siblings or [])
    except Exception:
        return None


def format_bytes(num_bytes: int | None) -> str:
    if not isinstance(num_bytes, int) or num_bytes <= 0:
        return "unknown size"
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024


def _estimate_size_via_head(repo_id: str, token: str | None, siblings) -> int | None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    total = 0
    found = False
    for sibling in siblings:
        filename = getattr(sibling, "rfilename", None)
        if not filename:
            continue
        # Skip bookkeeping files; they don't materially impact user-visible download size.
        if filename.endswith((".md", ".txt")) or filename in {".gitattributes"}:
            continue
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        try:
            resp = requests.head(url, headers=headers, allow_redirects=True, timeout=8)
            if resp.status_code >= 400:
                continue
            size = resp.headers.get("Content-Length")
            if size and size.isdigit():
                total += int(size)
                found = True
        except Exception:
            continue
    return total if found else None


def ensure_model_cached(
    model_name: str,
    on_status: StatusCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> str:
    """
    Ensures model files are present locally and returns a loadable model path/id.
    """
    repo_id = resolve_repo_id(model_name)
    if not repo_id:
        return model_name

    cached_path = _find_cached_snapshot_path(repo_id)
    if cached_path:
        if on_status:
            on_status(f"Model cache ready: '{repo_id}'.")
        return cached_path

    token = get_hf_token()

    class _ProgressTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            if on_status and getattr(self, "desc", None):
                on_status(f"Downloading model: {self.desc}")

        def update(self, n=1):
            result = super().update(n)
            if on_progress and self.total:
                on_progress((self.n / self.total) * 100.0)
            return result

    if on_status:
        on_status(f"Checking model cache for '{repo_id}'...")

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            token=token,
            tqdm_class=_ProgressTqdm,
            local_files_only=False,
            resume_download=True,
        )
    except RepositoryNotFoundError:
        # If mapping is stale or repo moved, fall back to faster-whisper native loader.
        if on_status:
            on_status(f"Repository '{repo_id}' not found. Falling back to built-in model resolver.")
        return model_name
    if on_progress:
        on_progress(100.0)
    return local_dir or model_name


def _candidate_hf_cache_roots() -> list[str]:
    roots: list[str] = []
    explicit_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit_hub:
        roots.append(explicit_hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
    # Also check canonical default to reuse existing caches across env changes.
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))

    seen: set[str] = set()
    unique: list[str] = []
    for root in roots:
        norm = os.path.abspath(os.path.expanduser(root))
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def _find_cached_snapshot_path(repo_id: str) -> str | None:
    model_dir_name = f"models--{repo_id.replace('/', '--')}"
    candidates: list[Path] = []
    for cache_root in _candidate_hf_cache_roots():
        snapshots = Path(cache_root) / model_dir_name / "snapshots"
        if snapshots.is_dir():
            for item in snapshots.iterdir():
                if item.is_dir():
                    candidates.append(item)
    if not candidates:
        return None
    # Prefer most recently updated snapshot if multiple roots/snapshots exist.
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(best)
