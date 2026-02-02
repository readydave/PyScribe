"""Shared catalog helpers for models and diarization options."""

from __future__ import annotations

from diar_backends import BACKENDS, available_backends
from utils import get_available_hf_models

BASE_MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def get_model_choices() -> list[str]:
    """Returns all selectable model IDs for UI dropdowns."""
    return sorted(set(BASE_MODEL_CHOICES + get_available_hf_models()))


def get_available_diarization_backends(include_off: bool = False) -> list[str]:
    """Returns installed diarization backend ids."""
    keys: list[str] = []
    for key, is_available in available_backends().items():
        if not is_available:
            continue
        if key == "off" and not include_off:
            continue
        keys.append(key)
    return keys


def get_backend_label(backend_id: str) -> str:
    """Returns user-facing label for a diarization backend id."""
    return BACKENDS.get(backend_id, {}).get("label", backend_id)
