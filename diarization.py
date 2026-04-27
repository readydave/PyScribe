# diarization.py
# Speaker diarization helpers for PyScribe.

from __future__ import annotations

import logging
from typing import Callable
import torchaudio
import numpy as np
import torch
from services.hf_auth_service import get_hf_token
from services.runtime_compat import ensure_platform_sys_version_compat
LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[float], None]
StatusCallback = Callable[[str], None]
Segment = dict[str, object]
_TORCHAUDIO_SOUNDFILE_PATCHED = False


def _torch_cuda_snapshot() -> str:
    """Returns concise torch/CUDA runtime diagnostics for logs."""
    parts = [
        f"torch={getattr(torch, '__version__', '<unknown>')}",
        f"torch.version.cuda={getattr(getattr(torch, 'version', object()), 'cuda', None)}",
    ]
    try:
        parts.append(f"cuda_available={torch.cuda.is_available()}")
    except Exception as exc:
        parts.append(f"cuda_available_error={exc}")
    try:
        parts.append(f"cuda_device_count={torch.cuda.device_count()}")
    except Exception as exc:
        parts.append(f"cuda_device_count_error={exc}")
    return " | ".join(parts)


def _prefer_torchaudio_soundfile_backend() -> str | None:
    """
    Prefer torchaudio's soundfile backend for pyannote IO.

    pyannote may still call ``torchaudio.set_audio_backend("soundfile")``, but on
    modern torchaudio (2.9+) that API and ``list_audio_backends`` are removed.
    The dispatcher now picks the best available backend automatically.
    """

    global _TORCHAUDIO_SOUNDFILE_PATCHED
    if _TORCHAUDIO_SOUNDFILE_PATCHED:
        return "soundfile"

    # For torchaudio < 2.9, we check available backends.
    # For torchaudio >= 2.9, list_audio_backends is removed.
    available = []
    if hasattr(torchaudio, "list_audio_backends"):
        try:
            available = list(torchaudio.list_audio_backends())
        except Exception as exc:
            LOGGER.warning("Unable to inspect torchaudio backends for diarization. reason=%s", exc, exc_info=True)
            return None
    else:
        # Modern torchaudio uses a dispatcher. We'll assume soundfile is available
        # or that the dispatcher will handle it. We still want to force soundfile
        # for pyannote if possible to avoid SoX segfaults.
        available = ["soundfile", "ffmpeg"]

    if "soundfile" not in available:
        LOGGER.info("TorchAudio soundfile backend unavailable for diarization. available=%s", available)
        return None

    # Some experimental torchaudio versions (e.g. 2.11.0) might lack 'info' attribute
    # but still have 'load'.
    original_info = getattr(torchaudio, "info", None)
    original_load = getattr(torchaudio, "load", None)

    def _wrap_with_soundfile_default(func):
        def _wrapped(*args, **kwargs):
            # If a backend is explicitly requested by the caller, honor it.
            # Otherwise, default to soundfile.
            if "backend" not in kwargs:
                kwargs["backend"] = "soundfile"
            return func(*args, **kwargs)

        setattr(_wrapped, "__pyscribe_soundfile_default__", True)
        return _wrapped

    if original_info and not getattr(original_info, "__pyscribe_soundfile_default__", False):
        torchaudio.info = _wrap_with_soundfile_default(original_info)
    if original_load and not getattr(original_load, "__pyscribe_soundfile_default__", False):
        torchaudio.load = _wrap_with_soundfile_default(original_load)

    _TORCHAUDIO_SOUNDFILE_PATCHED = True
    LOGGER.info(
        "Configured torchaudio to prefer backend=soundfile for diarization. available=%s",
        available,
    )
    return "soundfile"


def _lazy_import_pyannote() -> object:
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for diarization. Install with: pip install pyannote.audio"
        ) from e
    return Pipeline


def _load_pyannote_pipeline(Pipeline: object, token: str | None, requested_device: str) -> tuple[object, str]:
    LOGGER.info(
        "Loading pyannote diarization pipeline preferred=3.1 fallback=3.0 token_present=%s requested_device=%s",
        bool(token),
        requested_device,
    )
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        return pipeline, "3.1"
    except Exception as e1:
        LOGGER.warning("Failed to load pyannote pipeline 3.1; trying 3.0. reason=%s", e1, exc_info=True)
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=token)
            return pipeline, "3.0"
        except Exception as e2:
            LOGGER.error(
                "Failed to load pyannote pipeline versions 3.1 and 3.0 requested_device=%s torch_diag=%s",
                requested_device,
                _torch_cuda_snapshot(),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to load pyannote pipeline (3.1 then 3.0): {e2}") from e2


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
    backend_override = _prefer_torchaudio_soundfile_backend()
    Pipeline = _lazy_import_pyannote()
    # Some torchaudio builds dropped set_audio_backend; provide a no-op to avoid pipeline errors.
    if not hasattr(torchaudio, "set_audio_backend"):
        def _noop_backend(name: str | None = None) -> None:
            return None
        torchaudio.set_audio_backend = _noop_backend  # type: ignore
    # Numpy 2.x removed np.NaN alias; guard for older code paths in dependencies.
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore

    token = get_hf_token()
    LOGGER.info(
        "Preparing pyannote diarization requested_device=%s audio_backend=%s",
        device,
        backend_override or "default",
    )
    pipeline, pipeline_version = _load_pyannote_pipeline(Pipeline, token, device)
    LOGGER.info(
        "Loaded pyannote diarization pipeline version=%s token_present=%s",
        pipeline_version,
        bool(token),
    )

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
            "Diarization pipeline.to(%s) failed; using CPU fallback. reason=%s torch_diag=%s",
            requested_device,
            exc,
            _torch_cuda_snapshot(),
            exc_info=True,
        )
        try:
            pipeline, pipeline_version = _load_pyannote_pipeline(Pipeline, token, effective_device)
            LOGGER.info(
                "Reloaded pyannote diarization pipeline on CPU after %s device move failure.",
                requested_device,
            )
        except Exception:
            LOGGER.error(
                "Failed to reload pyannote diarization pipeline on CPU after %s device move failure.",
                requested_device,
                exc_info=True,
            )
            raise

    LOGGER.info(
        "Running diarization inference backend=accurate model=%s requested_device=%s effective_device=%s max_speakers=%s",
        pipeline_version,
        requested_device,
        effective_device,
        max_speakers,
    )
    try:
        diarization = pipeline(audio_path, num_speakers=max_speakers)
    except Exception:
        LOGGER.error(
            "Diarization inference failed backend=accurate model=%s requested_device=%s effective_device=%s max_speakers=%s torch_diag=%s",
            pipeline_version,
            requested_device,
            effective_device,
            max_speakers,
            _torch_cuda_snapshot(),
            exc_info=True,
        )
        raise
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
