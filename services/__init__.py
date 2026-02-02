"""Shared backend services for PyScribe frontends."""

from .catalog_service import (
    BASE_MODEL_CHOICES,
    get_available_diarization_backends,
    get_backend_label,
    get_model_choices,
)
from .config_service import AppConfig, load_config, save_config
from .hf_auth_service import get_hf_token, save_hf_token
from .model_download_service import (
    ensure_model_cached,
    estimate_model_download_size_bytes,
    format_bytes,
    is_model_cached,
    normalize_model_name,
    resolve_repo_id,
)
from .model_service import RuntimeInfo, detect_language, detect_runtime, load_model, recommend_model
from .platform_service import open_folder
from .transcription_service import TranscriptionResult, transcribe_media_file, transcribe_prepared_audio

__all__ = [
    "AppConfig",
    "BASE_MODEL_CHOICES",
    "RuntimeInfo",
    "detect_language",
    "detect_runtime",
    "get_available_diarization_backends",
    "get_backend_label",
    "get_hf_token",
    "get_model_choices",
    "estimate_model_download_size_bytes",
    "format_bytes",
    "is_model_cached",
    "normalize_model_name",
    "load_config",
    "ensure_model_cached",
    "load_model",
    "open_folder",
    "recommend_model",
    "save_hf_token",
    "resolve_repo_id",
    "save_config",
    "TranscriptionResult",
    "transcribe_media_file",
    "transcribe_prepared_audio",
]
