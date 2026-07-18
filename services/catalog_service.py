"""Shared catalog helpers for models and diarization options."""

from __future__ import annotations

import os
from pathlib import Path

from diar_backends import BACKENDS, backend_availability

# The single curated model list for UI dropdowns. models.TIERS keys must all
# appear here (enforced by tests/test_model_catalog.py) so tier metadata and
# selectable choices cannot drift apart again.
BASE_MODEL_CHOICES = [
    # Short faster-whisper aliases
    "tiny",
    "base",
    "small",
    "small.en",
    "medium",
    "large-v2",
    "large-v3",
    # Curated Hugging Face repos
    "Systran/faster-whisper-tiny.en",
    "Systran/faster-whisper-base.en",
    "Systran/faster-whisper-small.en",
    "Systran/faster-whisper-medium.en",
    "Systran/faster-whisper-large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "distil-whisper/distil-large-v3",
    "guillaumekln/whisper-large-v2-ct2",
    "guillaumekln/whisper-large-v3-ct2",
    # Experimental Granite Speech backend
    "ibm-granite/granite-4.0-1b-speech",
]


def _hf_hub_cache_dir() -> Path:
    """Resolves the Hugging Face hub cache dir, honoring the env overrides
    that runtime_env_service itself sets (HUGGINGFACE_HUB_CACHE, HF_HOME)."""
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _locally_cached_whisper_models() -> list[str]:
    """Returns faster-whisper repo IDs already present in the local HF cache."""
    cache_dir = _hf_hub_cache_dir()
    if not cache_dir.is_dir():
        return []
    cached: list[str] = []
    try:
        entries = list(cache_dir.iterdir())
    except OSError:
        return []
    for item in entries:
        if item.name.startswith("models--Systran--faster-whisper"):
            cached.append(item.name.removeprefix("models--").replace("--", "/"))
    return cached


def get_model_choices() -> list[str]:
    """Returns all selectable model IDs for UI dropdowns."""
    return sorted(set(BASE_MODEL_CHOICES + _locally_cached_whisper_models()))


def get_available_diarization_backends(include_off: bool = False) -> list[str]:
    """Returns installed diarization backend ids."""
    keys: list[str] = []
    for key, status in backend_availability().items():
        if not status.available:
            continue
        if key == "off" and not include_off:
            continue
        keys.append(key)
    return keys


def get_diarization_backend_availability(include_off: bool = False) -> dict[str, tuple[bool, str | None]]:
    """Returns backend availability details for UI diagnostics."""
    statuses: dict[str, tuple[bool, str | None]] = {}
    for key, status in backend_availability().items():
        if key == "off" and not include_off:
            continue
        statuses[key] = (status.available, status.reason)
    return statuses


def get_unavailable_diarization_backend_reasons(include_off: bool = False) -> dict[str, str]:
    """Returns unavailable backend ids with user-facing diagnostic reasons."""
    reasons: dict[str, str] = {}
    for key, (available, reason) in get_diarization_backend_availability(include_off=include_off).items():
        if not available:
            reasons[key] = reason or "Backend unavailable."
    return reasons


def get_backend_label(backend_id: str) -> str:
    """Returns user-facing label for a diarization backend id."""
    return BACKENDS.get(backend_id, {}).get("label", backend_id)
