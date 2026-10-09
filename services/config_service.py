"""Shared persisted config for PyScribe frontends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from services.ui_themes import DEFAULT_THEME_ID, PRESET_BY_ID, sanitize_custom_themes


@dataclass
class AppConfig:
    last_model: str | None = None
    run_mode: str = "full"
    theme_mode: str = "system"
    theme_id: str = DEFAULT_THEME_ID
    custom_themes: list[dict[str, object]] = field(default_factory=list)
    use_diarization: bool = False
    max_speakers: int | None = None
    diar_backend: str = "accurate"
    use_visual_analysis: bool = False
    visual_profile: str = "balanced"
    visual_ocr_backend: str = "auto"
    visual_ocr_fallback: str = "auto"
    visual_scope: str = "slides_only"
    visual_sample_seconds: float = 1.0
    confirmed_visual_backends: list[str] = field(default_factory=list)
    last_open_dir: str | None = None
    last_save_dir: str | None = None
    llm_profiles: list[dict[str, object]] = field(default_factory=list)
    llm_default_profile: str | None = None
    llm_default_template_id: str = "meeting-summary"
    llm_include_user_notes_default: bool = True
    llm_include_images_default: bool = True
    llm_ocr_fallback_for_images_default: bool = True
    llm_payload_preview_required: bool = False
    llm_allow_remote_lan: bool = False
    llm_allow_cloud_in_listener: bool = False
    live_source_mode: str = "microphone"
    live_input_device_id: str | None = None
    live_output_dir: str | None = None
    live_keep_audio_on_success: bool = True
    live_device_mode: str = "auto"
    live_compute_type: str = "auto"
    dock_layout: str | None = None
    dock_locked: bool = False
    setup_advanced_expanded: bool = False
    default_model: str | None = None
    default_input_mode: str = "file"
    default_hotwords: str = ""
    default_batched: bool = False
    sidebar_collapsed: bool = False
    window_geometry: str | None = None


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
        theme_id=_as_theme_id(data.get("theme_id"), data.get("custom_themes")),
        custom_themes=sanitize_custom_themes(data.get("custom_themes")),
        use_diarization=bool(data.get("use_diarization", False)),
        max_speakers=_as_optional_int(data.get("max_speakers")),
        diar_backend=str(data.get("diar_backend", "accurate")),
        use_visual_analysis=bool(data.get("use_visual_analysis", False)),
        visual_profile=_as_visual_profile(data.get("visual_profile")),
        visual_ocr_backend=_as_ocr_backend(data.get("visual_ocr_backend")),
        visual_ocr_fallback=_as_ocr_fallback(data.get("visual_ocr_fallback")),
        visual_scope=_as_visual_scope(data.get("visual_scope")),
        visual_sample_seconds=_as_optional_float(data.get("visual_sample_seconds"), 1.0),
        confirmed_visual_backends=_as_backend_list(data.get("confirmed_visual_backends")),
        last_open_dir=_as_optional_str(data.get("last_open_dir")),
        last_save_dir=_as_optional_str(data.get("last_save_dir")),
        llm_profiles=_as_profile_list(data.get("llm_profiles")),
        llm_default_profile=_as_optional_str(data.get("llm_default_profile")),
        llm_default_template_id=_as_prompt_template_id(data.get("llm_default_template_id")),
        llm_include_user_notes_default=_as_bool(data.get("llm_include_user_notes_default"), default=True),
        llm_include_images_default=_as_bool(data.get("llm_include_images_default"), default=True),
        llm_ocr_fallback_for_images_default=_as_bool(
            data.get("llm_ocr_fallback_for_images_default"),
            default=True,
        ),
        llm_payload_preview_required=_as_bool(data.get("llm_payload_preview_required"), default=False),
        llm_allow_remote_lan=_as_bool(data.get("llm_allow_remote_lan"), default=False),
        llm_allow_cloud_in_listener=_as_bool(data.get("llm_allow_cloud_in_listener"), default=False),
        live_source_mode=_as_live_source_mode(data.get("live_source_mode")),
        live_input_device_id=_as_optional_str(data.get("live_input_device_id")),
        live_output_dir=_as_optional_str(data.get("live_output_dir")),
        live_keep_audio_on_success=_as_bool(data.get("live_keep_audio_on_success"), default=True),
        live_device_mode=_as_choice(data.get("live_device_mode"), ("auto", "cpu", "gpu")),
        live_compute_type=_as_choice(data.get("live_compute_type"), ("auto", "float16", "int8")),
        dock_layout=_as_optional_str(data.get("dock_layout")),
        dock_locked=_as_bool(data.get("dock_locked"), default=False),
        setup_advanced_expanded=_as_bool(data.get("setup_advanced_expanded"), default=False),
        default_model=_as_optional_str(data.get("default_model")),
        default_input_mode="live" if data.get("default_input_mode") == "live" else "file",
        default_hotwords=_as_short_text(data.get("default_hotwords"), 500),
        default_batched=_as_bool(data.get("default_batched"), default=False),
        sidebar_collapsed=_as_bool(data.get("sidebar_collapsed"), default=False),
        window_geometry=_as_optional_str(data.get("window_geometry")),
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
    payload["llm_profiles"] = _sanitize_llm_profiles_for_storage(payload.get("llm_profiles"))
    payload["custom_themes"] = sanitize_custom_themes(payload.get("custom_themes"))
    path.write_text(json.dumps(payload), encoding="utf-8")


def _as_theme_id(value: object, custom_themes: object) -> str:
    """A known preset id or the id of a valid custom theme; anything else falls back to the default."""
    theme_id = str(value or DEFAULT_THEME_ID)
    if theme_id in PRESET_BY_ID:
        return theme_id
    if any(item.get("id") == theme_id for item in sanitize_custom_themes(custom_themes)):
        return theme_id
    return DEFAULT_THEME_ID


def _as_short_text(value: object, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


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
    allowed = {"auto", "paddleocr", "rapidocr", "surya", "pytesseract"}
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return "auto"


def _as_ocr_fallback(value: object) -> str:
    """Backend to try first when the main OCR backend cannot be used; "auto" keeps the built-in order."""
    allowed = {"auto", "rapidocr", "surya", "pytesseract"}
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "auto"


def _as_visual_scope(value: object) -> str:
    allowed = {"slides_only", "slides_chat"}
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in allowed:
        return normalized
    return "slides_only"


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


def _as_live_source_mode(value: object) -> str:
    allowed = {"microphone", "loopback"}
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return "microphone"


def _as_choice(value: object, allowed: tuple[str, ...]) -> str:
    """Normalized member of ``allowed``; the first entry (``auto``) for anything else."""
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else allowed[0]


def _as_backend_list(value: object) -> list[str]:
    allowed = {"paddleocr", "rapidocr", "surya", "pytesseract", "auto"}
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        backend = str(item or "").strip().lower()
        if backend in allowed and backend not in normalized:
            normalized.append(backend)
    return normalized


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _as_prompt_template_id(value: object) -> str:
    text = _as_optional_str(value)
    if not text:
        return "meeting-summary"
    return text.lower()


def _as_profile_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _sanitize_llm_profiles_for_storage(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    sanitized: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        profile = dict(item)
        api_key = str(profile.get("api_key") or "").strip()
        if api_key and not api_key.lower().startswith("env:"):
            profile["api_key"] = ""
        profile.pop("api_key_runtime", None)
        sanitized.append(profile)
    return sanitized
