"""Shared backend services for PyScribe frontends.

This module intentionally uses lazy exports so importing :mod:`services`
does not eagerly pull in heavy runtime dependencies (torch/pyannote/etc.).
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BASE_MODEL_CHOICES": (".catalog_service", "BASE_MODEL_CHOICES"),
    "get_available_diarization_backends": (".catalog_service", "get_available_diarization_backends"),
    "get_backend_label": (".catalog_service", "get_backend_label"),
    "get_model_choices": (".catalog_service", "get_model_choices"),
    "AppConfig": (".config_service", "AppConfig"),
    "load_config": (".config_service", "load_config"),
    "save_config": (".config_service", "save_config"),
    "get_hf_token": (".hf_auth_service", "get_hf_token"),
    "save_hf_token": (".hf_auth_service", "save_hf_token"),
    "ensure_model_cached": (".model_download_service", "ensure_model_cached"),
    "estimate_model_download_size_bytes": (".model_download_service", "estimate_model_download_size_bytes"),
    "format_bytes": (".model_download_service", "format_bytes"),
    "is_model_cached": (".model_download_service", "is_model_cached"),
    "normalize_model_name": (".model_download_service", "normalize_model_name"),
    "resolve_repo_id": (".model_download_service", "resolve_repo_id"),
    "RuntimeInfo": (".model_service", "RuntimeInfo"),
    "detect_language": (".model_service", "detect_language"),
    "detect_runtime": (".model_service", "detect_runtime"),
    "load_model": (".model_service", "load_model"),
    "check_ocr_backend_ready": (".multimodal_service", "check_ocr_backend_ready"),
    "recommend_model": (".model_service", "recommend_model"),
    "open_folder": (".platform_service", "open_folder"),
    "TranscriptionResult": (".transcription_service", "TranscriptionResult"),
    "transcribe_media_file": (".transcription_service", "transcribe_media_file"),
    "transcribe_prepared_audio": (".transcription_service", "transcribe_prepared_audio"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
