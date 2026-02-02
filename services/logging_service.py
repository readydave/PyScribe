"""Central logging setup for PyScribe."""

from __future__ import annotations

import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_LOG_PATH: Path | None = None


def configure_logging() -> Path:
    """
    Configures app-wide logging once and returns the log file path.
    """
    global _CONFIGURED, _LOG_PATH

    configured = _find_writable_log_path()
    if _CONFIGURED:
        return _LOG_PATH or configured

    level_name = os.environ.get("PYSCRIBE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Keep handlers idempotent in case modules call configure_logging repeatedly.
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(process)d:%(threadName)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = _make_file_handler(configured)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if os.environ.get("PYSCRIBE_LOG_STDOUT", "0") == "1":
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    actual_path = Path(file_handler.baseFilename)
    logging.getLogger(__name__).info("Logging initialized at %s", actual_path)
    _LOG_PATH = actual_path
    _CONFIGURED = True
    return actual_path


def get_log_path() -> Path:
    """Returns the active log file path (or best-effort default before init)."""
    return _LOG_PATH or _find_writable_log_path()


def _find_writable_log_path() -> Path:
    custom = os.environ.get("PYSCRIBE_LOG_DIR")
    candidates = []
    if custom:
        candidates.append(Path(custom))
    candidates.extend(
        [
            Path.home() / ".pyscribe" / "logs",
            Path.cwd() / ".pyscribe_logs",
            Path(tempfile.gettempdir()) / "pyscribe-logs",
        ]
    )
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return directory / "pyscribe.log"
        except Exception:
            continue
    # Last resort: return a path in current dir; handler creation will surface clear errors.
    return Path("pyscribe.log")


def _make_file_handler(primary_path: Path) -> RotatingFileHandler:
    for path in (
        primary_path,
        Path.cwd() / ".pyscribe_logs" / "pyscribe.log",
        Path(tempfile.gettempdir()) / "pyscribe-logs" / "pyscribe.log",
    ):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return RotatingFileHandler(
                path,
                maxBytes=2 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        except Exception:
            continue
    # Final fallback: if even this fails, let logging raise a clear error.
    return RotatingFileHandler(
        "pyscribe.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
