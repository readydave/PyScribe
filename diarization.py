# diarization.py
# Speaker diarization helpers for PyScribe.

from __future__ import annotations

from typing import List, Dict, Optional
from huggingface_hub import HfFolder
import torchaudio
import numpy as np


def _lazy_import_pyannote():
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
    max_speakers: Optional[int] = None,
    progress_cb=None,
) -> List[Dict]:
    """
    Runs diarization on the provided audio file.
    Returns a list of segments: [{"start": float, "end": float, "speaker": "S1"}, ...]
    """
    Pipeline = _lazy_import_pyannote()
    # Some torchaudio builds dropped set_audio_backend; provide a no-op to avoid pipeline errors.
    if not hasattr(torchaudio, "set_audio_backend"):
        def _noop_backend(name=None):
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

    try:
        pipeline.to(device)
    except Exception:
        pass

    diarization = pipeline(audio_path, num_speakers=max_speakers)
    if progress_cb:
        try:
            progress_cb(95)
        except Exception:
            pass
    segments = []
    speaker_map = {}
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
    return segments


def assign_speakers(asr_segments: List[Dict], spk_segments: List[Dict]) -> List[Dict]:
    """
    Assigns a speaker label to each ASR segment based on maximum overlap.
    Returns updated ASR segments with 'speaker' key.
    """
    def overlap(a_start, a_end, b_start, b_end):
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
