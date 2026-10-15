"""Central logging setup for PyScribe."""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_LOG_PATH: Path | None = None


def configure_logging() -> Path:
    """
    Configures app-wide logging once and returns the log file path.
    All processes in a single session will share the same 'pyscribe.log'.
    """
    global _CONFIGURED, _LOG_PATH

    log_dir = _find_log_directory()
    log_file = log_dir / "pyscribe.log"

    if _CONFIGURED:
        return _LOG_PATH or log_file

    # Only the primary process rotates the log at the start of a session.
    # Child processes (e.g. spawned transcription workers) inherit this env var.
    if os.environ.get("PYSCRIBE_SESSION_STARTED") is None:
        if log_file.exists():
            try:
                mtime = log_file.stat().st_mtime
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(mtime))
                archive_name = log_dir / f"pyscribe_{timestamp}.log"
                # Avoid collision if restarted in the same second
                if archive_name.exists():
                    timestamp += f"_{os.getpid()}"
                    archive_name = log_dir / f"pyscribe_{timestamp}.log"
                log_file.rename(archive_name)
            except Exception:
                # If we can't rename (e.g. file locked), we just append.
                pass
        
        # Cleanup old archives
        _cleanup_old_logs(log_dir, keep_count=21)
        os.environ["PYSCRIBE_SESSION_STARTED"] = "1"

    level_name = os.environ.get("PYSCRIBE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(process)d:%(threadName)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Use RotatingFileHandler to also prevent indefinite growth within a long session.
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10 * 1024 * 1024, 
        backupCount=5, 
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if os.environ.get("PYSCRIBE_LOG_STDOUT", "0") == "1":
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    logging.getLogger(__name__).info("Logging initialized at %s", log_file)
    _LOG_PATH = log_file
    _CONFIGURED = True
    return log_file


def get_log_path() -> Path:
    """Returns the active log file path."""
    if _LOG_PATH:
        return _LOG_PATH
    return _find_log_directory() / "pyscribe.log"


def _find_log_directory() -> Path:
    custom = os.environ.get("PYSCRIBE_LOG_DIR")
    candidates = []
    if custom:
        candidates.append(Path(custom))
    candidates.append(Path.home() / ".pyscribe" / "logs")
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            _tighten_directory_permissions(directory)
            return directory
        except Exception:
            continue
    return Path(".")


def _cleanup_old_logs(log_dir: Path, keep_count: int) -> None:
    """Keep only the N most recent timestamped log files."""
    try:
        logs = sorted(
            log_dir.glob("pyscribe_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if len(logs) > keep_count:
            for old_log in logs[keep_count:]:
                try:
                    old_log.unlink()
                except Exception:
                    continue
    except Exception:
        pass


def _tighten_directory_permissions(directory: Path) -> None:
    try:
        directory.chmod(0o700)
    except Exception:
        return
