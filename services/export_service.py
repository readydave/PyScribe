"""Transcript export: plain text, Markdown, SRT, WebVTT and JSON from a project's timed segments.

Pure functions (no Qt, no file dialogs). Speaker labels such as ``S1`` are shown with the names the user
gave them. Exports never include the local media path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from services.project_store import Project

MAX_CUE_CHARS = 2000


@dataclass(frozen=True)
class ExportFormat:
    key: str
    label: str
    extension: str


FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat("txt", "Plain text (.txt)", ".txt"),
    ExportFormat("md", "Markdown (.md)", ".md"),
    ExportFormat("srt", "Subtitles (.srt)", ".srt"),
    ExportFormat("vtt", "WebVTT (.vtt)", ".vtt"),
    ExportFormat("json", "JSON (.json)", ".json"),
)
FORMAT_BY_KEY = {fmt.key: fmt for fmt in FORMATS}


def speaker_display(project: Project, label: str | None) -> str:
    """The user's name for ``label`` when set, else the label; empty for an unknown speaker."""
    if not label or label in ("S?", "?"):
        return ""
    return project.speaker_names.get(label, label)


def timestamp(seconds: float, *, comma: bool = False, hours: bool = True) -> str:
    """``HH:MM:SS.mmm`` (or ``,mmm`` for SRT); ``MM:SS`` without hours for transcript lines."""
    millis = int(round(max(0.0, seconds) * 1000))
    total, ms = divmod(millis, 1000)
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    if not hours:
        return f"{h * 60 + m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"


def _clock(seconds: float) -> str:
    total = int(max(0.0, seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:MAX_CUE_CHARS]


def _line(project: Project, segment: dict, *, with_time: bool) -> str:
    who = speaker_display(project, segment.get("speaker"))
    parts = []
    if with_time:
        parts.append(f"[{timestamp(segment['start'], hours=False)}]")
    if who:
        parts.append(f"{who}:")
    parts.append(_clean(segment["text"]))
    return " ".join(parts)


def export_txt(project: Project) -> str:
    return "\n".join(_line(project, s, with_time=True) for s in project.segments) + ("\n" if project.segments else "")


def transcript_for_llm(project: Project) -> str:
    """Speaker-labelled text without timestamps, for prompts (uses the user's speaker names)."""
    return "\n".join(_line(project, s, with_time=False) for s in project.segments)


def export_md(project: Project) -> str:
    lines = [f"# {project.title}", ""]
    facts = []
    if project.duration_s:
        facts.append(_clock(project.duration_s))
    if project.speaker_count:
        facts.append(f"{project.speaker_count} speaker{'s' if project.speaker_count != 1 else ''}")
    if facts:
        lines += [" · ".join(facts), ""]
    for segment in project.segments:
        who = speaker_display(project, segment.get("speaker"))
        stamp = timestamp(segment["start"], hours=False)
        head = f"**{who}** `{stamp}`" if who else f"`{stamp}`"
        lines += [f"{head}  ", _clean(segment["text"]), ""]
    return "\n".join(lines).rstrip() + "\n"


def _cue_text(project: Project, segment: dict) -> str:
    who = speaker_display(project, segment.get("speaker"))
    text = _clean(segment["text"])
    return f"{who}: {text}" if who else text


def export_srt(project: Project) -> str:
    blocks = []
    for index, segment in enumerate(project.segments, start=1):
        end = max(segment["end"], segment["start"] + 0.5)
        blocks.append(
            f"{index}\n{timestamp(segment['start'], comma=True)} --> {timestamp(end, comma=True)}\n{_cue_text(project, segment)}\n"
        )
    return "\n".join(blocks)


def export_vtt(project: Project) -> str:
    blocks = ["WEBVTT\n"]
    for segment in project.segments:
        end = max(segment["end"], segment["start"] + 0.5)
        text = _cue_text(project, segment).replace("-->", "->")
        blocks.append(f"{timestamp(segment['start'])} --> {timestamp(end)}\n{text}\n")
    return "\n".join(blocks)


def export_json(project: Project) -> str:
    payload = {
        "title": project.title,
        "duration_seconds": project.duration_s,
        "speakers": {label: speaker_display(project, label) for label in _labels(project)},
        "segments": [
            {
                "start": s["start"],
                "end": s["end"],
                "speaker": speaker_display(project, s.get("speaker")) or None,
                "text": s["text"],
            }
            for s in project.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _labels(project: Project) -> list[str]:
    seen: list[str] = []
    for segment in project.segments:
        label = segment.get("speaker")
        if label and label not in ("S?", "?") and label not in seen:
            seen.append(label)
    return seen


_EXPORTERS = {"txt": export_txt, "md": export_md, "srt": export_srt, "vtt": export_vtt, "json": export_json}


def render(project: Project, fmt: str) -> str:
    """Render ``project`` in ``fmt`` (one of ``FORMAT_BY_KEY``)."""
    exporter = _EXPORTERS.get(fmt)
    if exporter is None:
        raise ValueError(f"Unsupported export format: {fmt}")
    return exporter(project)


def suggested_filename(project: Project, fmt: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "", project.title, flags=re.UNICODE).strip().replace(" ", "_")[:80] or "transcript"
    return f"{stem}{FORMAT_BY_KEY[fmt].extension}"


def export_to_path(project: Project, fmt: str, path: Path) -> None:
    """Write the export to ``path`` (UTF-8). Raises ``OSError`` if the file cannot be written."""
    Path(path).write_text(render(project, fmt), encoding="utf-8")
