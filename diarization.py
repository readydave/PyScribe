# diarization.py
# Speaker diarization helpers for PyScribe.

from __future__ import annotations

import logging
from typing import Callable
from huggingface_hub import HfFolder
import torchaudio
import numpy as np
import torch
from services.runtime_compat import ensure_platform_sys_version_compat
LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[float], None]
StatusCallback = Callable[[str], None]
Segment = dict[str, object]


def _lazy_import_pyannote() -> object:
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for diarization. Install with: pip install pyannote.audio"
        ) from e
    return Pipeline


def run_diarization(
    audio_path: str,
    device: str = "cpu",
    max_speakers: int | None = None,
    progress_cb: ProgressCallback | None = None,
    status_cb: StatusCallback | None = None,
) -> list[Segment]:
    """
    Runs diarization on the provided audio file.
    Returns a list of segments: [{"start": float, "end": float, "speaker": "S1"}, ...]
    """
    ensure_platform_sys_version_compat()
    Pipeline = _lazy_import_pyannote()
    # Some torchaudio builds dropped set_audio_backend; provide a no-op to avoid pipeline errors.
    if not hasattr(torchaudio, "set_audio_backend"):
        def _noop_backend(name: str | None = None) -> None:
            return None
        torchaudio.set_audio_backend = _noop_backend  # type: ignore
    # Numpy 2.x removed np.NaN alias; guard for older code paths in dependencies.
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore

    token = HfFolder.get_token()
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    except Exception:
        # fallback to 3.0 if 3.1 still gated or unavailable
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=token)
        except Exception as e2:
            raise RuntimeError(f"Failed to load pyannote pipeline (3.1 then 3.0): {e2}")

    requested_device = device
    effective_device = "cpu"
    try:
        pipeline.to(torch.device(device))
        effective_device = device
        if status_cb:
            status_cb(f"Diarization backend: accurate | Device: {str(effective_device).upper()}")
    except Exception as exc:
        effective_device = "cpu"
        if status_cb:
            status_cb(f"Diarization backend fallback to CPU (requested {requested_device.upper()})")
        LOGGER.warning(
            "Diarization pipeline.to(%s) failed; using CPU fallback. reason=%s",
            requested_device,
            exc,
        )

    diarization = pipeline(audio_path, num_speakers=max_speakers)
    if progress_cb:
        try:
            progress_cb(95)
        except Exception:
            pass
    segments: list[Segment] = []
    speaker_map: dict[object, str] = {}
    speaker_idx = 1
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"S{speaker_idx}"
            speaker_idx += 1
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker_map[speaker],
            }
        )
    LOGGER.info(
        "Diarization complete backend=accurate requested_device=%s effective_device=%s segments=%s",
        requested_device,
        effective_device,
        len(segments),
    )
    return segments


def assign_speakers(asr_segments: list[Segment], spk_segments: list[Segment]) -> list[Segment]:
    """
    Assigns a speaker label to each ASR segment based on maximum overlap.
    Returns updated ASR segments with 'speaker' key.
    """
    def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    for seg in asr_segments:
        best_spk = None
        best_ov = 0.0
        for spk in spk_segments:
            ov = overlap(seg["start"], seg["end"], spk["start"], spk["end"])
            if ov > best_ov:
                best_ov = ov
                best_spk = spk["speaker"]
        seg["speaker"] = best_spk or "S?"
    return asr_segments
