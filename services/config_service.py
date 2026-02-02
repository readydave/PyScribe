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
    )


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Saves config to disk."""
    payload = asdict(config)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _as_optional_int(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
