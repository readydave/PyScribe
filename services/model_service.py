"""Model/runtime utilities shared across frontends."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class RuntimeInfo:
    device: str
    compute_type: str
    gpu_name: str
    vram_gb: float
    cpu_count: int


_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}


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


def load_model(
    model_name: str,
    *,
    device: str,
    compute_type: str,
    use_cache: bool = True,
) -> WhisperModel:
    """Loads (and optionally caches) a Whisper model instance."""
    key = (model_name, device, compute_type)
    if use_cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

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
