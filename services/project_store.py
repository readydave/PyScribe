"""Local project history: every finished job is saved as a project under ``~/.pyscribe/projects``.

A project holds the transcript, the timed speaker segments, optional visual (OCR) text and any AI
outputs, so it can be reopened in the Transcript Studio. Qt-free so the Listener and MCP can share it.

Security: project ids are generated here and validated on every access, so a caller-supplied id can
never point outside the store. Directories are 0700 and files 0600; writes are atomic.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1
ENV_PROJECTS_DIR = "PYSCRIBE_PROJECTS_DIR"
DEFAULT_PROJECTS_ROOT = Path.home() / ".pyscribe" / "projects"
PROJECT_FILE = "project.json"
PEAKS_FILE = "peaks.json"
SUMMARY_FILE = "summary.json"  # small card data so listing never reads whole transcripts

KINDS = ("file", "batch", "live")
STATUSES = ("transcribing", "completed", "failed", "cancelled")

MAX_TITLE = 120
MAX_SEGMENTS = 200_000
MAX_SPEAKER_LABEL = 16
MAX_SPEAKER_NAME = 40
MAX_PEAKS = 128
MAX_FULL_PEAKS = 4000
MAX_AI_OUTPUTS = 50

_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")
_UNKNOWN_SPEAKERS = {"", "S?", "?"}


def resolve_projects_root(configured: str | None = None) -> Path:
    """Projects folder: the env override, else an absolute configured path, else ``~/.pyscribe/projects``."""
    override = os.environ.get(ENV_PROJECTS_DIR, "").strip()
    for candidate in (override, (configured or "").strip()):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_absolute():
            return path
        LOGGER.warning("Ignoring a relative projects folder setting.")
    return DEFAULT_PROJECTS_ROOT


def is_valid_project_id(value: object) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value))


def new_project_id(now: datetime.datetime | None = None) -> str:
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(4)}"


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="microseconds")


def _clean_text(value: object, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if number == number and abs(number) != float("inf") else default


def sanitize_segments(raw: object) -> list[dict[str, Any]]:
    """Keep only well-formed ``{start, end, speaker, text}`` segments (words and extras are dropped)."""
    if not isinstance(raw, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in raw[:MAX_SEGMENTS]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, _number(item.get("start")))
        end = max(start, _number(item.get("end"), start))
        speaker = _clean_text(item.get("speaker"), MAX_SPEAKER_LABEL) or None
        segments.append({"start": round(start, 3), "end": round(end, 3), "speaker": speaker, "text": text})
    return segments


def _format_line(segment: dict[str, Any]) -> str:
    speaker = segment.get("speaker")
    return f"[{speaker}] {segment['text']}" if speaker else str(segment["text"])


def speaker_labels(segments: list[dict[str, Any]]) -> list[str]:
    """Distinct known speaker labels in order of first appearance."""
    seen: list[str] = []
    for segment in segments:
        label = segment.get("speaker")
        if label and label not in _UNKNOWN_SPEAKERS and label not in seen:
            seen.append(label)
    return seen


def _sanitize_peaks(raw: object, limit: int = MAX_PEAKS) -> list[float]:
    if not isinstance(raw, list):
        return []
    return [round(max(0.0, min(1.0, _number(v))), 3) for v in raw[:limit]]


def _sanitize_names(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    names: dict[str, str] = {}
    for key, value in list(raw.items())[:64]:
        label, name = _clean_text(key, MAX_SPEAKER_LABEL), _clean_text(value, MAX_SPEAKER_NAME)
        if label and name:
            names[label] = name
    return names


def _sanitize_ai_outputs(raw: object) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        return {}
    outputs: dict[str, dict[str, str]] = {}
    for key, value in list(raw.items())[:MAX_AI_OUTPUTS]:
        if not isinstance(value, dict):
            continue
        outputs[_clean_text(key, 64)] = {
            "text": str(value.get("text") or ""),
            "created": _clean_text(value.get("created"), 40),
            "profile": _clean_text(value.get("profile"), 80),
        }
    return outputs


@dataclass
class Project:
    id: str
    title: str
    kind: str = "file"
    status: str = "transcribing"
    created: str = field(default_factory=_now_iso)
    updated: str = field(default_factory=_now_iso)
    media_path: str = ""
    model: str = ""
    duration_s: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)
    transcript_text: str = ""
    visual_report: str = ""
    speaker_names: dict[str, str] = field(default_factory=dict)
    ai_outputs: dict[str, dict[str, str]] = field(default_factory=dict)
    preview_peaks: list[float] = field(default_factory=list)

    @property
    def speaker_count(self) -> int:
        return len(speaker_labels(self.segments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "created": self.created,
            "updated": self.updated,
            "media_path": self.media_path,
            "model": self.model,
            "duration_s": self.duration_s,
            "segments": self.segments,
            "transcript_text": self.transcript_text,
            "visual_report": self.visual_report,
            "speaker_names": self.speaker_names,
            "ai_outputs": self.ai_outputs,
            "preview_peaks": self.preview_peaks,
        }

    @classmethod
    def from_dict(cls, data: object) -> Project | None:
        """Rebuild a project from stored JSON, sanitising every field. ``None`` for unusable data."""
        if not isinstance(data, dict) or not is_valid_project_id(data.get("id")):
            return None
        segments = sanitize_segments(data.get("segments"))
        kind, status = str(data.get("kind")), str(data.get("status"))
        return cls(
            id=data["id"],
            title=_clean_text(data.get("title"), MAX_TITLE) or "Untitled",
            kind=kind if kind in KINDS else "file",
            status=status if status in STATUSES else "completed",
            created=_clean_text(data.get("created"), 40) or _now_iso(),
            updated=_clean_text(data.get("updated"), 40) or _now_iso(),
            media_path=_clean_text(data.get("media_path"), 1024),
            model=_clean_text(data.get("model"), 120),
            duration_s=max(0.0, _number(data.get("duration_s"))),
            segments=segments,
            transcript_text=str(data.get("transcript_text") or ""),
            visual_report=str(data.get("visual_report") or ""),
            speaker_names=_sanitize_names(data.get("speaker_names")),
            ai_outputs=_sanitize_ai_outputs(data.get("ai_outputs")),
            preview_peaks=_sanitize_peaks(data.get("preview_peaks")),
        )


_UPDATABLE_FIELDS = frozenset(f.name for f in fields(Project)) - {"id", "created", "updated"}


@dataclass(frozen=True)
class ProjectSummary:
    """What Projects Home needs to draw a card."""

    id: str
    title: str
    kind: str
    status: str
    created: str
    updated: str
    duration_s: float
    speaker_count: int
    model: str
    preview_peaks: tuple[float, ...] = ()

    @classmethod
    def of(cls, project: Project) -> ProjectSummary:
        return cls(
            id=project.id,
            title=project.title,
            kind=project.kind,
            status=project.status,
            created=project.created,
            updated=project.updated,
            duration_s=project.duration_s,
            speaker_count=project.speaker_count,
            model=project.model,
            preview_peaks=tuple(project.preview_peaks),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["preview_peaks"] = list(self.preview_peaks)
        return data

    @classmethod
    def from_dict(cls, data: object) -> ProjectSummary | None:
        if not isinstance(data, dict) or not is_valid_project_id(data.get("id")):
            return None
        kind, status = str(data.get("kind")), str(data.get("status"))
        return cls(
            id=data["id"],
            title=_clean_text(data.get("title"), MAX_TITLE) or "Untitled",
            kind=kind if kind in KINDS else "file",
            status=status if status in STATUSES else "completed",
            created=_clean_text(data.get("created"), 40),
            updated=_clean_text(data.get("updated"), 40),
            duration_s=max(0.0, _number(data.get("duration_s"))),
            speaker_count=int(max(0.0, min(_number(data.get("speaker_count")), 999))),
            model=_clean_text(data.get("model"), 120),
            preview_peaks=tuple(_sanitize_peaks(data.get("preview_peaks"))),
        )


class ProjectStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else resolve_projects_root()

    # --- paths ---------------------------------------------------------------------------------

    def project_dir(self, project_id: str) -> Path:
        """Folder for ``project_id``; raises ``ValueError`` for anything that is not a generated id."""
        if not is_valid_project_id(project_id):
            raise ValueError("Invalid project id.")
        return self.root / project_id

    def _file(self, project_id: str) -> Path:
        return self.project_dir(project_id) / PROJECT_FILE

    # --- writes --------------------------------------------------------------------------------

    def create(
        self,
        *,
        title: str,
        kind: str = "file",
        media_path: str = "",
        model: str = "",
        status: str = "transcribing",
    ) -> Project:
        project = Project(
            id=new_project_id(),
            title=_clean_text(title, MAX_TITLE) or "Untitled",
            kind=kind if kind in KINDS else "file",
            status=status if status in STATUSES else "transcribing",
            media_path=_clean_text(media_path, 1024),
            model=_clean_text(model, 120),
        )
        self.save(project)
        return project

    def save(self, project: Project) -> None:
        """Write ``project`` atomically (sanitised), creating the folder with private permissions."""
        clean = Project.from_dict(project.to_dict())
        if clean is None:
            raise ValueError("Invalid project.")
        clean.updated = _now_iso()
        directory = self.project_dir(clean.id)
        self._make_private_dir(self.root)
        self._make_private_dir(directory)
        self._atomic_write(directory, SUMMARY_FILE, ProjectSummary.of(clean).to_dict())
        self._atomic_write(directory, PROJECT_FILE, clean.to_dict())
        project.updated = clean.updated

    @staticmethod
    def _atomic_write(directory: Path, name: str, data: dict[str, Any]) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".write-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(data, ensure_ascii=False))
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, directory / name)
        except OSError:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _make_private_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path, 0o700)
        except OSError:
            LOGGER.debug("Could not restrict permissions on %s", path)

    def update(self, project_id: str, **changes: Any) -> Project | None:
        """Load, apply ``changes`` to known fields and save. ``None`` when the project does not exist."""
        project = self.load(project_id)
        if project is None:
            return None
        for name, value in changes.items():
            if name not in _UPDATABLE_FIELDS:
                raise ValueError(f"Cannot update field: {name}")
            setattr(project, name, value)
        self.save(project)
        return self.load(project_id)

    def update_segments(self, project_id: str, segments: list[dict[str, Any]]) -> Project | None:
        clean = sanitize_segments(segments)
        text = "\n".join(_format_line(s) for s in clean)
        return self.update(project_id, segments=clean, transcript_text=text)

    def set_speaker_segments(self, project_id: str, segments: list[dict[str, Any]]) -> Project | None:
        """Replace segments after re-identifying speakers; old speaker names no longer apply, so they reset."""
        clean = sanitize_segments(segments)
        text = "\n".join(_format_line(s) for s in clean)
        return self.update(project_id, segments=clean, transcript_text=text, speaker_names={})

    def rename(self, project_id: str, title: str) -> Project | None:
        title = _clean_text(title, MAX_TITLE)
        if not title:
            raise ValueError("A project needs a title.")
        return self.update(project_id, title=title)

    def rename_speaker(self, project_id: str, label: str, name: str) -> Project | None:
        project = self.load(project_id)
        if project is None:
            return None
        names = dict(project.speaker_names)
        name = _clean_text(name, MAX_SPEAKER_NAME)
        if name:
            names[_clean_text(label, MAX_SPEAKER_LABEL)] = name
        else:
            names.pop(label, None)
        return self.update(project_id, speaker_names=names)

    def set_ai_output(self, project_id: str, template_id: str, text: str, profile: str = "") -> Project | None:
        project = self.load(project_id)
        if project is None:
            return None
        outputs = dict(project.ai_outputs)
        outputs[_clean_text(template_id, 64)] = {"text": text, "created": _now_iso(), "profile": profile}
        return self.update(project_id, ai_outputs=outputs)

    def set_peaks(self, project_id: str, peaks: list[float], preview: list[float]) -> bool:
        """Cache the full waveform for the Studio and store the small preview used on cards."""
        if self.load(project_id) is None:
            return False
        self._atomic_write(self.project_dir(project_id), PEAKS_FILE, {"peaks": _sanitize_peaks(peaks, MAX_FULL_PEAKS)})
        self.update(project_id, preview_peaks=preview)
        return True

    def load_peaks(self, project_id: str) -> list[float]:
        try:
            data = json.loads((self.project_dir(project_id) / PEAKS_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return _sanitize_peaks(data.get("peaks") if isinstance(data, dict) else None, MAX_FULL_PEAKS)

    def delete(self, project_id: str) -> bool:
        """Remove a project folder (project data only; the source media is never touched)."""
        directory = self.project_dir(project_id)
        if not directory.is_dir() or directory.is_symlink():
            return False
        shutil.rmtree(directory)
        return True

    # --- reads ---------------------------------------------------------------------------------

    def load(self, project_id: str) -> Project | None:
        try:
            path = self._file(project_id)
            return Project.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    def _read_summary(self, project_id: str) -> ProjectSummary | None:
        """The small summary file, rebuilt from the full project when it is missing or damaged."""
        try:
            text = (self.project_dir(project_id) / SUMMARY_FILE).read_text(encoding="utf-8")
            summary = ProjectSummary.from_dict(json.loads(text))
            if summary is not None and summary.id == project_id:
                return summary
        except (OSError, ValueError):
            pass
        project = self.load(project_id)
        return ProjectSummary.of(project) if project is not None else None

    def list_summaries(self, limit: int = 200) -> list[ProjectSummary]:
        """Newest-first summaries; unreadable or malformed projects are skipped."""
        if not self.root.is_dir():
            return []
        limit = max(1, min(int(limit), 1000))
        summaries: list[ProjectSummary] = []
        for directory in sorted(self.root.iterdir(), reverse=True):
            if not is_valid_project_id(directory.name) or directory.is_symlink():
                continue
            summary = self._read_summary(directory.name)
            if summary is None:
                LOGGER.warning("Skipping unreadable project folder %s", directory.name)
                continue
            summaries.append(summary)
        summaries.sort(key=lambda s: s.updated, reverse=True)
        return summaries[:limit]
