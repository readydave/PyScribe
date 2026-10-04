"""Model/runtime utilities shared across frontends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .granite_speech_service import load_granite_model
from .model_download_service import normalize_model_name, resolve_repo_id

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


@dataclass(frozen=True)
class RuntimeInfo:
    device: str
    compute_type: str
    gpu_name: str
    vram_gb: float
    cpu_count: int


GRANITE_SPEECH_REPO_IDS = {"ibm-granite/granite-4.0-1b-speech"}


@dataclass(frozen=True)
class TranscriptionModelSpec:
    requested_name: str
    normalized_name: str
    backend_kind: str
    repo_id: str | None
    display_name: str
    supports_diarization: bool
    supports_timestamps: bool
    is_experimental: bool = False


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def detect_runtime() -> RuntimeInfo:
    """Detects hardware/runtime info used for model defaults."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = "N/A"
    vram_gb = 0.0
    if device == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 1)
        except Exception:
            device = "cpu"
            gpu_name = "N/A"
            vram_gb = 0.0

    return RuntimeInfo(
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        cpu_count=os.cpu_count() or 1,
    )


def recommend_model(runtime: RuntimeInfo) -> str:
    """Returns a Whisper model recommendation for the detected runtime."""
    if runtime.device == "cuda":
        if runtime.vram_gb >= 10:
            return "large-v3"
        if runtime.vram_gb >= 8:
            return "large-v2"
        if runtime.vram_gb >= 5:
            return "medium"
        if runtime.vram_gb >= 3:
            return "small"
        return "base"

    if runtime.cpu_count >= 12:
        return "small"
    if runtime.cpu_count >= 8:
        return "base"
    return "tiny"


def resolve_transcription_model(model_name: str) -> TranscriptionModelSpec:
    """Resolve a model string to a backend/capability descriptor."""
    normalized = normalize_model_name(model_name)
    repo_id = resolve_repo_id(normalized)
    if repo_id in GRANITE_SPEECH_REPO_IDS:
        return TranscriptionModelSpec(
            requested_name=model_name,
            normalized_name=normalized,
            backend_kind="granite_transformers",
            repo_id=repo_id,
            display_name=repo_id,
            supports_diarization=False,
            supports_timestamps=False,
            is_experimental=True,
        )

    display_name = repo_id or normalized
    return TranscriptionModelSpec(
        requested_name=model_name,
        normalized_name=normalized,
        backend_kind="faster_whisper",
        repo_id=repo_id,
        display_name=display_name,
        supports_diarization=True,
        supports_timestamps=True,
        is_experimental=False,
    )


def model_supports_diarization(model_name: str) -> bool:
    return resolve_transcription_model(model_name).supports_diarization


def is_experimental_model(model_name: str) -> bool:
    return resolve_transcription_model(model_name).is_experimental


def load_model(
    model_name: str,
    *,
    device: str,
    compute_type: str,
    use_cache: bool = True,
    model_spec: TranscriptionModelSpec | None = None,
) -> Any:
    """Load and optionally cache a transcription model for the resolved backend."""
    spec = model_spec or resolve_transcription_model(model_name)
    cache_name = spec.normalized_name or model_name
    key = (cache_name, device, compute_type)
    if use_cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if spec.backend_kind == "granite_transformers":
        model = load_granite_model(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    else:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    if use_cache:
        _MODEL_CACHE[key] = model
    return model


def detect_language(audio_np: object, *, device: str) -> tuple[str, float]:
    """Detects language using a small detector model."""
    detector = load_model(
        "tiny",
        device=device,
        compute_type="int8",
        use_cache=True,
    )
    lang_code, lang_prob, *_ = detector.detect_language(audio_np)
    return lang_code, float(lang_prob)
