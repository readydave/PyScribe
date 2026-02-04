"""Shared persisted config for PyScribe frontends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    last_model: str | None = None
    run_mode: str = "full"
    theme_mode: str = "system"
    use_diarization: bool = False
    max_speakers: int | None = None
    diar_backend: str = "accurate"
    use_visual_analysis: bool = False
    visual_profile: str = "balanced"
    visual_ocr_backend: str = "paddleocr"
    visual_sample_seconds: float = 1.0
    confirmed_visual_backends: list[str] = field(default_factory=list)
    last_open_dir: str | None = None
    last_save_dir: str | None = None


DEFAULT_CONFIG_PATH = Path.home() / ".pyscribe_config.json"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Loads config from disk with safe defaults."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AppConfig()

    return AppConfig(
        last_model=data.get("last_model"),
        run_mode=_as_run_mode(data.get("run_mode")),
        theme_mode=_as_theme_mode(data.get("theme_mode")),
        use_diarization=bool(data.get("use_diarization", False)),
        max_speakers=_as_optional_int(data.get("max_speakers")),
        diar_backend=str(data.get("diar_backend", "accurate")),
        use_visual_analysis=bool(data.get("use_visual_analysis", False)),
        visual_profile=_as_visual_profile(data.get("visual_profile")),
        visual_ocr_backend=_as_ocr_backend(data.get("visual_ocr_backend")),
        visual_sample_seconds=_as_optional_float(data.get("visual_sample_seconds"), 1.0),
        confirmed_visual_backends=_as_backend_list(data.get("confirmed_visual_backends")),
        last_open_dir=_as_optional_str(data.get("last_open_dir")),
        last_save_dir=_as_optional_str(data.get("last_save_dir")),
    )


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Saves config to disk."""
    payload = asdict(config)
    # Preserve path preferences if caller did not explicitly set them.
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
    if not payload.get("last_open_dir"):
        payload["last_open_dir"] = existing.get("last_open_dir")
    if not payload.get("last_save_dir"):
        payload["last_save_dir"] = existing.get("last_save_dir")
    path.write_text(json.dumps(payload), encoding="utf-8")


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _as_optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _as_optional_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
        if parsed <= 0:
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _as_ocr_backend(value: object) -> str:
    allowed = {"auto", "paddleocr", "surya", "pytesseract"}
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return "paddleocr"


def _as_visual_profile(value: object) -> str:
    allowed = {"fast", "balanced", "accurate"}
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return "balanced"


def _as_run_mode(value: object) -> str:
    allowed = {"full", "transcribe_only", "visual_only"}
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return "full"


def _as_theme_mode(value: object) -> str:
    allowed = {"system", "light", "dark"}
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return "system"


def _as_backend_list(value: object) -> list[str]:
    allowed = {"paddleocr", "surya", "pytesseract", "auto"}
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        backend = str(item or "").strip().lower()
        if backend in allowed and backend not in normalized:
            normalized.append(backend)
    return normalized
