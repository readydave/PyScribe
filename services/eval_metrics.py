"""Evaluation metrics for ASR and diarization runs (WER/CER, DER, resource use)."""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNKNOWN_SPEAKER = "S?"
_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, unify apostrophes, drop punctuation and collapse whitespace.

    Accents are kept so Spanish references compare correctly.
    """
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).lower()
    text = text.replace("-", " ").replace("—", " ").replace("–", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = text.replace("_", " ")
    return _SPACE_RE.sub(" ", text).strip()


def compute_wer_cer(reference: str, hypothesis: str) -> dict[str, float]:
    """Return WER and CER (0..1+) after consistent normalization."""
    import jiwer

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        raise ValueError("Reference text is empty after normalization.")
    return {
        "wer": float(jiwer.wer(ref, hyp)),
        "cer": float(jiwer.cer(ref, hyp)),
        "ref_words": float(len(ref.split())),
    }


def parse_rttm(path: str | Path) -> list[dict[str, Any]]:
    """Parse SPEAKER lines of an RTTM file into start/end/speaker dicts."""
    segments: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            continue
        start = float(fields[3])
        segments.append({"start": start, "end": start + float(fields[4]), "speaker": fields[7]})
    return segments


def _to_annotation(segments: list[dict[str, Any]]) -> Any:
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for seg in segments:
        if seg["end"] > seg["start"]:
            annotation[Segment(seg["start"], seg["end"])] = seg["speaker"]
    return annotation


def compute_der(
    reference: list[dict[str, Any]],
    hypothesis: list[dict[str, Any]],
    *,
    collar: float = 0.25,
) -> float:
    """Diarization error rate (0..1+) via pyannote.metrics."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    return float(metric(_to_annotation(reference), _to_annotation(hypothesis)))


def count_unknown_speakers(segments: list[dict[str, Any]]) -> int:
    """Count segments left unlabelled (`S?`) after speaker assignment."""
    return sum(1 for seg in segments if seg.get("speaker", UNKNOWN_SPEAKER) == UNKNOWN_SPEAKER)


def count_mislabelled_segments(
    asr_segments: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> int:
    """Count ASR segments whose label disagrees with the reference speaker.

    Hypothesis labels are mapped to reference speakers by total overlap
    (greedy, one-to-one); each segment is judged against the reference speaker
    with the largest overlap inside it. Segments with no reference overlap are
    ignored.
    """

    def overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
        return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))

    totals: dict[tuple[str, str], float] = {}
    for seg in asr_segments:
        label = seg.get("speaker", UNKNOWN_SPEAKER)
        for ref in reference:
            ov = overlap(seg, ref)
            if ov > 0:
                key = (label, ref["speaker"])
                totals[key] = totals.get(key, 0.0) + ov

    mapping: dict[str, str] = {}
    used_refs: set[str] = set()
    for (label, ref_spk), _ in sorted(totals.items(), key=lambda item: -item[1]):
        if label in mapping or ref_spk in used_refs:
            continue
        mapping[label] = ref_spk
        used_refs.add(ref_spk)

    wrong = 0
    for seg in asr_segments:
        per_speaker: dict[str, float] = {}
        for ref in reference:
            ov = overlap(seg, ref)
            if ov > 0:
                per_speaker[ref["speaker"]] = per_speaker.get(ref["speaker"], 0.0) + ov
        if not per_speaker:
            continue
        truth = max(per_speaker, key=per_speaker.get)
        if mapping.get(seg.get("speaker", UNKNOWN_SPEAKER)) != truth:
            wrong += 1
    return wrong


def real_time_factor(processing_seconds: float, audio_seconds: float) -> float:
    """Processing time divided by audio duration (lower is faster)."""
    if audio_seconds <= 0:
        return 0.0
    return processing_seconds / audio_seconds


@dataclass
class ResourcePeaks:
    ram_mb: float = 0.0
    vram_mb: float = 0.0


class ResourceMonitor:
    """Samples process RSS and GPU memory in a background thread."""

    def __init__(self, interval: float = 0.25) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peaks = ResourcePeaks()

    def __enter__(self) -> "ResourceMonitor":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        nvml_handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pynvml = None  # type: ignore[assignment]
        while not self._stop.is_set():
            try:
                family = [proc, *proc.children(recursive=True)]
                self.peaks.ram_mb = max(
                    self.peaks.ram_mb, sum(p.memory_info().rss for p in family) / 1e6
                )
                pids = {p.pid for p in family}
            except psutil.Error:
                pids = {os.getpid()}
            if nvml_handle is not None:
                try:
                    used = sum(
                        (p.usedGpuMemory or 0)
                        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(nvml_handle)
                        if p.pid in pids
                    )
                    self.peaks.vram_mb = max(self.peaks.vram_mb, used / 1e6)
                except Exception:
                    pass
            self._stop.wait(self._interval)
