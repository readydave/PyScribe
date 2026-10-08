"""Logic behind PyScribe's MCP server (no MCP imports, so it is easy to test).

Covers: which media files may be transcribed, where finished transcripts are kept, and a one-at-a-time
job queue for transcription. Everything the server returns to an AI client is treated as data, and every
path or id coming from a client is validated here before it touches the file system.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

MEDIA_EXTENSIONS = frozenset(
    {
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".amr",
        ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp",
    }
)
ROOTS_ENV = "PYSCRIBE_MCP_ROOTS"
STORE_DIR = Path.home() / ".pyscribe" / "mcp_transcripts"
DEFAULT_PAGE_CHARS = 20_000
MAX_PAGE_CHARS = 100_000
MAX_QUEUED_JOBS = 10
MAX_REMEMBERED_JOBS = 50
MAX_STORED_TRANSCRIPTS = 500
MAX_NAMES_TERMS_CHARS = 500

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{3,90}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
LIVE_PREFIX = "live-"


class McpToolError(Exception):
    """A problem the AI client should be told about in plain words."""


# --- file access ------------------------------------------------------------------------------


def allowed_roots(environ: dict[str, str] | None = None) -> list[Path]:
    """Folders that media files may be read from: ``PYSCRIBE_MCP_ROOTS`` (path-separated) or the home folder."""
    env = os.environ if environ is None else environ
    raw = (env.get(ROOTS_ENV) or "").strip()
    candidates = [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()] if raw else [Path.home()]
    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


def validate_media_path(raw_path: object, roots: list[Path]) -> Path:
    """Return the real path of an audio/video file inside an allowed folder, or raise ``McpToolError``."""
    if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
        raise McpToolError("Give the full path of an audio or video file.")
    if not roots:
        raise McpToolError(f"No readable folders are configured. Set {ROOTS_ENV} to the folders PyScribe may read.")
    try:
        resolved = Path(raw_path.strip()).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise McpToolError("That file was not found.") from None
    if not resolved.is_file():
        raise McpToolError("That path is not a file.")
    if resolved.suffix.lower() not in MEDIA_EXTENSIONS:
        raise McpToolError("Only audio and video files can be transcribed (for example .mp3, .wav, .m4a, .mp4, .mkv).")
    if not any(resolved.is_relative_to(root) for root in roots):
        raise McpToolError(
            f"That file is outside the folders PyScribe is allowed to read. Allowed: "
            f"{', '.join(str(root) for root in roots)}. Change them with {ROOTS_ENV}."
        )
    return resolved


# --- saved transcripts ------------------------------------------------------------------------


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:30] or "transcript"


@dataclass(frozen=True)
class TranscriptSummary:
    id: str
    title: str
    source: str  # "saved" or "live"
    created: str
    model: str
    chars: int
    has_speakers: bool


class TranscriptStore:
    """Finished MCP transcripts as private JSON files, plus read access to Qt live-session transcripts."""

    def __init__(self, directory: Path = STORE_DIR, live_root: Path | None = None) -> None:
        self.directory = directory
        self.live_root = live_root

    # saving
    def save(self, *, source_name: str, model: str, text: str, plain_text: str, duration_seconds: float) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass
        transcript_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug(Path(source_name).stem)}-{secrets.token_hex(2)}"
        payload = {
            "id": transcript_id,
            "title": Path(source_name).stem,
            "source_file": Path(source_name).name,
            "model": model,
            "created": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round(float(duration_seconds), 1),
            "text": text,
            "plain_text": plain_text,
        }
        path = self.directory / f"{transcript_id}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        self._prune()
        return transcript_id

    def _prune(self) -> None:
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for stale in files[: max(0, len(files) - MAX_STORED_TRANSCRIPTS)]:
            try:
                stale.unlink()
            except OSError:
                pass

    # listing
    def list_summaries(self, limit: int = 20, source: str = "all") -> list[TranscriptSummary]:
        limit = max(1, min(int(limit), 100))
        items: list[tuple[float, TranscriptSummary]] = []
        if source in {"all", "saved"}:
            items.extend(self._list_saved())
        if source in {"all", "live"}:
            items.extend(self._list_live())
        items.sort(key=lambda pair: pair[0], reverse=True)
        return [summary for _mtime, summary in items[:limit]]

    def _list_saved(self) -> list[tuple[float, TranscriptSummary]]:
        found = []
        if not self.directory.is_dir():
            return found
        for path in self.directory.glob("*.json"):
            data = self._read_json(path)
            if not isinstance(data, dict) or not _ID_RE.match(str(data.get("id", ""))):
                continue
            text = str(data.get("text") or "")
            found.append(
                (
                    path.stat().st_mtime,
                    TranscriptSummary(
                        id=str(data["id"]),
                        title=str(data.get("title") or ""),
                        source="saved",
                        created=str(data.get("created") or ""),
                        model=str(data.get("model") or ""),
                        chars=len(text),
                        has_speakers=text.strip() != str(data.get("plain_text") or "").strip(),
                    ),
                )
            )
        return found

    def _list_live(self) -> list[tuple[float, TranscriptSummary]]:
        found = []
        for folder in self._live_folders():
            info = self._live_info(folder)
            if info is None:
                continue
            path, meta = info
            found.append(
                (
                    path.stat().st_mtime,
                    TranscriptSummary(
                        id=f"{LIVE_PREFIX}{folder.name}",
                        title=str(meta.get("session_title") or folder.name),
                        source="live",
                        created=str(meta.get("started_at") or ""),
                        model=str(meta.get("selected_model") or ""),
                        chars=path.stat().st_size,
                        has_speakers=False,
                    ),
                )
            )
        return found

    # reading
    def read(self, transcript_id: str, *, speaker_labels: bool = True) -> tuple[str, dict[str, Any]]:
        """Return (text, metadata) for an id from ``list_summaries``; unknown or malformed ids raise ``McpToolError``."""
        text_id = str(transcript_id or "").strip()
        if text_id.startswith(LIVE_PREFIX):
            return self._read_live(text_id)
        if not _ID_RE.match(text_id):
            raise McpToolError("Unknown transcript id. Use list_transcripts to see valid ids.")
        data = self._read_json(self.directory / f"{text_id}.json")
        if not isinstance(data, dict):
            raise McpToolError("Unknown transcript id. Use list_transcripts to see valid ids.")
        text = str(data.get("text") if speaker_labels else (data.get("plain_text") or data.get("text")) or "")
        meta = {key: data.get(key) for key in ("id", "title", "source_file", "model", "created", "duration_seconds")}
        meta["source"] = "saved"
        return text, meta

    def _read_live(self, transcript_id: str) -> tuple[str, dict[str, Any]]:
        name = transcript_id[len(LIVE_PREFIX):]
        for folder in self._live_folders():
            if folder.name != name:  # only ids that appear in the listing resolve
                continue
            info = self._live_info(folder)
            if info is None:
                break
            path, meta = info
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                break
            return text, {
                "id": transcript_id,
                "title": meta.get("session_title") or folder.name,
                "model": meta.get("selected_model"),
                "created": meta.get("started_at"),
                "source": "live",
            }
        raise McpToolError("Unknown transcript id. Use list_transcripts to see valid ids.")

    # helpers
    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            if path.stat().st_size > 50_000_000:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _live_folders(self) -> list[Path]:
        if self.live_root is None or not self.live_root.is_dir():
            return []
        try:
            return [p for p in self.live_root.iterdir() if p.is_dir() and not p.is_symlink()]
        except OSError:
            return []

    def _live_info(self, folder: Path) -> tuple[Path, dict[str, Any]] | None:
        """(final transcript path, session metadata) for a finished live session, or None."""
        meta = self._read_json(folder / "session.json")
        if not isinstance(meta, dict) or meta.get("status") != "completed":
            return None
        raw = meta.get("final_transcript_path")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            path = Path(raw).resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not path.is_file() or not path.is_relative_to(folder.resolve()):
            return None  # a session file must not point outside its own folder
        return path, meta


def page_text(text: str, offset: int, max_chars: int) -> dict[str, Any]:
    """One page of ``text`` with the information needed to ask for the next one."""
    offset = max(0, int(offset))
    max_chars = max(1000, min(int(max_chars), MAX_PAGE_CHARS))
    chunk = text[offset : offset + max_chars]
    end = offset + len(chunk)
    return {
        "text": chunk,
        "offset": offset,
        "returned_chars": len(chunk),
        "total_chars": len(text),
        "next_offset": end if end < len(text) else None,
    }


# --- jobs -------------------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    params: dict[str, Any]
    status: str = "queued"  # queued, running, completed, failed, cancelled
    stage: str = "waiting"
    percent: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    transcript_id: str | None = None
    error: str | None = None
    message: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def done(self) -> bool:
        return self.status in {"completed", "failed", "cancelled"}

    def summary(self, queue_position: int | None = None) -> dict[str, Any]:
        elapsed = None
        if self.started_at:
            elapsed = round((self.finished_at or time.time()) - self.started_at, 1)
        out: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "percent": round(self.percent, 1),
            "message": self.message,
            "elapsed_seconds": elapsed,
            "transcript_id": self.transcript_id,
            "error": self.error,
        }
        if queue_position is not None:
            out["queue_position"] = queue_position
        return out


@dataclass(frozen=True)
class RunnerOutput:
    text: str
    plain_text: str
    model: str
    duration_seconds: float


class JobHooks:
    """What a runner uses to report progress and notice cancellation."""

    def __init__(self, job: Job) -> None:
        self._job = job

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_event.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        return self._job.cancel_event

    def stage(self, stage: str, message: str = "") -> None:
        self._job.stage = stage
        if message:
            self._job.message = message

    def progress(self, percent: float) -> None:
        self._job.percent = max(0.0, min(100.0, float(percent)))


Runner = Callable[[dict[str, Any], JobHooks], RunnerOutput]


class JobManager:
    """Runs transcription jobs one at a time on a background thread (the GPU handles one job well)."""

    def __init__(self, runner: Runner, store: TranscriptStore) -> None:
        self._runner = runner
        self._store = store
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: list[str] = []
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None

    def submit(self, params: dict[str, Any]) -> Job:
        with self._cond:
            if len(self._queue) >= MAX_QUEUED_JOBS:
                raise McpToolError(f"Too many transcriptions are waiting ({MAX_QUEUED_JOBS}). Wait for one to finish.")
            job = Job(id=secrets.token_hex(4), params=params)
            self._jobs[job.id] = job
            self._queue.append(job.id)
            while len(self._jobs) > MAX_REMEMBERED_JOBS:
                oldest = next((k for k, j in self._jobs.items() if j.done), None)
                if oldest is None:
                    break
                del self._jobs[oldest]
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="pyscribe-mcp-jobs", daemon=True)
                self._worker.start()
            self._cond.notify_all()
            return job

    def get(self, job_id: str) -> Job:
        with self._cond:
            job = self._jobs.get(str(job_id or "").strip())
        if job is None:
            raise McpToolError("Unknown job id. Jobs are forgotten when the server restarts.")
        return job

    def queue_position(self, job: Job) -> int | None:
        with self._cond:
            return self._queue.index(job.id) + 1 if job.id in self._queue else None

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self._cond:
            if job.done:
                return job
            job.cancel_event.set()
            if job.id in self._queue:  # not started yet: cancel immediately
                self._queue.remove(job.id)
                job.status = "cancelled"
                job.stage = "cancelled"
                job.finished_at = time.time()
                self._cond.notify_all()
        return job

    def wait(self, job_id: str, timeout_seconds: float, on_tick: Callable[[Job], None] | None = None) -> Job:
        """Block until the job finishes or ``timeout_seconds`` pass, calling ``on_tick`` about once a second."""
        job = self.get(job_id)
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while not job.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            with self._cond:
                self._cond.wait(timeout=min(1.0, remaining))
            if on_tick is not None:
                on_tick(job)
        return job

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    if not self._cond.wait(timeout=60.0) and not self._queue:
                        return  # idle: let the thread end; submit() starts a new one
                job = self._jobs[self._queue.pop(0)]
                if job.cancel_event.is_set():
                    job.status, job.stage, job.finished_at = "cancelled", "cancelled", time.time()
                    self._cond.notify_all()
                    continue
                job.status, job.stage, job.started_at = "running", "starting", time.time()
            self._run_one(job)
            with self._cond:
                self._cond.notify_all()

    def _run_one(self, job: Job) -> None:
        hooks = JobHooks(job)
        try:
            output = self._runner(job.params, hooks)
            if job.cancel_event.is_set():
                job.status, job.stage = "cancelled", "cancelled"
            elif not output.text.strip():
                job.status, job.stage, job.error = "failed", "failed", "No speech was found in the file."
            else:
                job.transcript_id = self._store.save(
                    source_name=str(job.params.get("file_name") or "transcript"),
                    model=output.model,
                    text=output.text,
                    plain_text=output.plain_text,
                    duration_seconds=output.duration_seconds,
                )
                job.status, job.stage, job.percent = "completed", "done", 100.0
        except McpToolError as exc:
            job.status, job.stage, job.error = "failed", "failed", str(exc)
        except Exception as exc:  # the worker must survive any single job failing
            LOGGER.warning("MCP transcription job %s failed", job.id, exc_info=True)
            job.status, job.stage, job.error = "failed", "failed", f"Transcription failed: {type(exc).__name__}: {exc}"[:300]
        finally:
            job.finished_at = time.time()


def clean_names_terms(value: object) -> str:
    """Names/terms hint: plain text only, trimmed and length-limited."""
    text = " ".join(str(value or "").split())
    return text[:MAX_NAMES_TERMS_CHARS]
