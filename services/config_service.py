"""Shared persisted config for PyScribe frontends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AppConfig:
    last_model: str | None = None
    use_diarization: bool = False
    max_speakers: int | None = None
    diar_backend: str = "accurate"
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
        use_diarization=bool(data.get("use_diarization", False)),
        max_speakers=_as_optional_int(data.get("max_speakers")),
        diar_backend=str(data.get("diar_backend", "accurate")),
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


def _as_optional_int(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _as_optional_str(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None
