"""Shared catalog helpers for models and diarization options."""

from __future__ import annotations

from diar_backends import BACKENDS, backend_availability
from utils import get_available_hf_models

BASE_MODEL_CHOICES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "ibm-granite/granite-4.0-1b-speech",
]


def get_model_choices() -> list[str]:
    """Returns all selectable model IDs for UI dropdowns."""
    return sorted(set(BASE_MODEL_CHOICES + get_available_hf_models()))


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
