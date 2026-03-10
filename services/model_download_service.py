"""Model download helpers with optional in-app progress callbacks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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


class ModelVerificationError(RuntimeError):
    """Raised when a downloaded Hugging Face snapshot cannot be verified."""


@dataclass(frozen=True)
class ModelVerificationEntry:
    relative_path: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class ModelVerificationManifest:
    repo_id: str
    revision: str
    files: tuple[ModelVerificationEntry, ...]


@dataclass(frozen=True)
class ModelVerificationResult:
    repo_id: str
    revision: str
    verified_files: int
    verified_bytes: int


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


def _estimate_size_via_head(repo_id: str, token: str | None, siblings: list[object]) -> int | None:
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

    token = get_hf_token()
    if on_status:
        on_status(f"Checking model cache for '{repo_id}'...")

    class _ProgressTqdm(tqdm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            if on_status and getattr(self, "desc", None):
                on_status(f"Downloading model: {self.desc}")

        def update(self, n: int = 1) -> bool | None:
            result = super().update(n)
            if on_progress and self.total:
                on_progress((self.n / self.total) * 100.0)
            return result

    cached_path = _find_cached_snapshot_path(repo_id)
    if cached_path:
        cached_revision = _snapshot_revision_from_path(cached_path)
        if cached_revision:
            try:
                manifest = _fetch_verification_manifest(
                    repo_id=repo_id,
                    token=token,
                    revision=cached_revision,
                    on_status=on_status,
                )
            except RepositoryNotFoundError:
                if on_status:
                    on_status(f"Repository '{repo_id}' not found. Falling back to built-in model resolver.")
                return model_name
            if on_status:
                on_status(f"Verifying downloaded files for '{repo_id}'...")
            try:
                _verify_model_snapshot(cached_path, manifest)
            except ModelVerificationError as exc:
                if on_status:
                    on_status(
                        f"Cached model verification failed for '{repo_id}': {exc}. "
                        "Re-downloading verified files..."
                    )
                local_dir = _download_snapshot(
                    repo_id=repo_id,
                    token=token,
                    revision=manifest.revision,
                    force_download=True,
                    on_status=on_status,
                    tqdm_class=_ProgressTqdm,
                )
                if on_status:
                    on_status(f"Verifying downloaded files for '{repo_id}'...")
                _verify_model_snapshot(local_dir, manifest)
                if on_progress:
                    on_progress(100.0)
                return local_dir
            if on_status:
                on_status(f"Model cache ready: '{repo_id}'.")
            return cached_path
        if on_status:
            on_status(
                f"Cached model path for '{repo_id}' could not be tied to a revision. "
                "Downloading a verified snapshot..."
            )

    try:
        manifest = _fetch_verification_manifest(
            repo_id=repo_id,
            token=token,
            revision=None,
            on_status=on_status,
        )
    except RepositoryNotFoundError:
        # If mapping is stale or repo moved, fall back to faster-whisper native loader.
        if on_status:
            on_status(f"Repository '{repo_id}' not found. Falling back to built-in model resolver.")
        return model_name

    local_dir = _download_snapshot(
        repo_id=repo_id,
        token=token,
        revision=manifest.revision,
        force_download=False,
        on_status=on_status,
        tqdm_class=_ProgressTqdm,
    )
    if on_status:
        on_status(f"Verifying downloaded files for '{repo_id}'...")
    _verify_model_snapshot(local_dir, manifest)
    if on_progress:
        on_progress(100.0)
    return local_dir or model_name


def ensure_hf_repo_local_dir_verified(
    repo_id: str,
    local_dir: str | os.PathLike[str],
    *,
    on_status: StatusCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> str:
    """
    Ensures a Hugging Face repo is present in ``local_dir`` and verified.

    This is used for PaddleX/PaddleOCR model caches that are downloaded into a
    plain directory instead of the standard Hugging Face snapshot cache.
    """
    token = get_hf_token()
    local_dir_path = Path(local_dir)
    if on_status:
        on_status(f"Checking model cache for '{repo_id}'...")

    class _ProgressTqdm(tqdm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            if on_status and getattr(self, "desc", None):
                on_status(f"Downloading model: {self.desc}")

        def update(self, n: int = 1) -> bool | None:
            result = super().update(n)
            if on_progress and self.total:
                on_progress((self.n / self.total) * 100.0)
            return result

    local_revision = _local_dir_revision_from_metadata(local_dir_path)
    if local_revision:
        manifest = _fetch_verification_manifest(
            repo_id=repo_id,
            token=token,
            revision=local_revision,
            on_status=on_status,
        )
        if on_status:
            on_status(f"Verifying downloaded files for '{repo_id}'...")
        try:
            _verify_model_snapshot(local_dir_path, manifest)
        except ModelVerificationError as exc:
            if on_status:
                on_status(
                    f"Cached model verification failed for '{repo_id}': {exc}. "
                    "Re-downloading verified files..."
                )
            _download_snapshot_to_local_dir(
                repo_id=repo_id,
                local_dir=local_dir_path,
                token=token,
                revision=manifest.revision,
                force_download=True,
                on_status=on_status,
                tqdm_class=_ProgressTqdm,
            )
            if on_status:
                on_status(f"Verifying downloaded files for '{repo_id}'...")
            _verify_model_snapshot(local_dir_path, manifest)
            if on_progress:
                on_progress(100.0)
            return str(local_dir_path)
        if on_status:
            on_status(f"Model cache ready: '{repo_id}'.")
        return str(local_dir_path)

    force_download = local_dir_path.exists()
    if force_download and on_status:
        on_status(
            f"Cached model path for '{repo_id}' could not be tied to a revision. "
            "Re-downloading verified files..."
        )

    manifest = _fetch_verification_manifest(
        repo_id=repo_id,
        token=token,
        revision=None,
        on_status=on_status,
    )
    _download_snapshot_to_local_dir(
        repo_id=repo_id,
        local_dir=local_dir_path,
        token=token,
        revision=manifest.revision,
        force_download=force_download,
        on_status=on_status,
        tqdm_class=_ProgressTqdm,
    )
    if on_status:
        on_status(f"Verifying downloaded files for '{repo_id}'...")
    _verify_model_snapshot(local_dir_path, manifest)
    if on_progress:
        on_progress(100.0)
    return str(local_dir_path)


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


def _fetch_verification_manifest(
    *,
    repo_id: str,
    token: str | None,
    revision: str | None,
    on_status: StatusCallback | None = None,
) -> ModelVerificationManifest:
    if on_status:
        on_status(f"Fetching model metadata for '{repo_id}'...")
    api = HfApi(token=token)
    info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
    resolved_revision = str(getattr(info, "sha", "") or "").strip()
    if not resolved_revision:
        raise ModelVerificationError(f"Unable to determine a pinned revision for '{repo_id}'.")

    entries: list[ModelVerificationEntry] = []
    for sibling in info.siblings or []:
        relative_path = getattr(sibling, "rfilename", None)
        if not relative_path:
            continue
        lfs = getattr(sibling, "lfs", None)
        if lfs is None:
            continue
        sha256 = str(getattr(lfs, "sha256", "") or "").strip().lower()
        if not sha256:
            raise ModelVerificationError(
                f"Hugging Face did not publish a SHA-256 for required model file '{relative_path}'."
            )
        size_bytes = getattr(sibling, "size", None)
        if not isinstance(size_bytes, int):
            lfs_size = getattr(lfs, "size", None)
            size_bytes = lfs_size if isinstance(lfs_size, int) else None
        entries.append(
            ModelVerificationEntry(
                relative_path=str(relative_path),
                sha256=sha256,
                size_bytes=size_bytes,
            )
        )

    if not entries:
        raise ModelVerificationError(
            f"'{repo_id}' does not publish any LFS-backed files that PyScribe can verify."
        )

    return ModelVerificationManifest(
        repo_id=repo_id,
        revision=resolved_revision,
        files=tuple(entries),
    )


def _download_snapshot(
    *,
    repo_id: str,
    token: str | None,
    revision: str,
    force_download: bool,
    on_status: StatusCallback | None,
    tqdm_class: type[tqdm],
) -> str:
    if on_status:
        on_status(f"Downloading model files for '{repo_id}'...")
    try:
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            tqdm_class=tqdm_class,
            local_files_only=False,
            force_download=force_download,
            resume_download=not force_download,
        )
    except RepositoryNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Model download failed for '{repo_id}': {exc}") from exc


def _download_snapshot_to_local_dir(
    *,
    repo_id: str,
    local_dir: str | os.PathLike[str],
    token: str | None,
    revision: str,
    force_download: bool,
    on_status: StatusCallback | None,
    tqdm_class: type[tqdm],
) -> str:
    if on_status:
        on_status(f"Downloading model files for '{repo_id}'...")
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            tqdm_class=tqdm_class,
            local_dir=str(local_dir_path),
            local_files_only=False,
            force_download=force_download,
            resume_download=not force_download,
        )
    except RepositoryNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Model download failed for '{repo_id}': {exc}") from exc
    return str(local_dir_path)


def _verify_model_snapshot(snapshot_path: str | os.PathLike[str], manifest: ModelVerificationManifest) -> ModelVerificationResult:
    base_path = Path(snapshot_path)
    if not base_path.is_dir():
        raise ModelVerificationError(f"Downloaded model path does not exist: {base_path}")

    verified_bytes = 0
    for entry in manifest.files:
        target_path = base_path / entry.relative_path
        if not target_path.is_file():
            raise ModelVerificationError(f"Required model file is missing: {entry.relative_path}")
        actual_sha256 = _sha256_file(target_path)
        if actual_sha256 != entry.sha256:
            raise ModelVerificationError(
                f"Checksum mismatch for '{entry.relative_path}' "
                f"(expected {entry.sha256}, got {actual_sha256})."
            )
        if isinstance(entry.size_bytes, int) and entry.size_bytes > 0:
            verified_bytes += entry.size_bytes
        else:
            verified_bytes += target_path.stat().st_size

    return ModelVerificationResult(
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        verified_files=len(manifest.files),
        verified_bytes=verified_bytes,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_dir_revision_from_metadata(path: str | os.PathLike[str]) -> str | None:
    metadata_dir = Path(path) / ".cache" / "huggingface" / "download"
    if not metadata_dir.is_dir():
        return None

    revisions: set[str] = set()
    found_metadata = False
    for metadata_file in metadata_dir.glob("*.metadata"):
        found_metadata = True
        try:
            lines = metadata_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            return None
        if not lines:
            return None
        revision = str(lines[0] or "").strip()
        if not revision:
            return None
        revisions.add(revision)

    if not found_metadata or len(revisions) != 1:
        return None
    return next(iter(revisions))


def _snapshot_revision_from_path(path: str | os.PathLike[str]) -> str | None:
    revision = Path(path).name.strip()
    return revision or None
