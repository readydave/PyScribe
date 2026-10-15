# diarization.py
# Speaker diarization helpers for PyScribe.

from __future__ import annotations

import logging
import os
import warnings
from typing import Callable

import torch

from services.hf_auth_service import get_hf_token
from services.runtime_compat import ensure_platform_sys_version_compat
from services.speaker_assignment import assign_speakers  # noqa: F401  (re-exported for callers)

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[float], None]
StatusCallback = Callable[[str], None]
Segment = dict[str, object]


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


def _direct_soundfile_load(
    uri: str | os.PathLike,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    start = frame_offset if frame_offset > 0 else 0
    stop = (start + num_frames) if num_frames > 0 else None
    data, samplerate = sf.read(uri, start=start, stop=stop, dtype="float32")
    tensor = torch.from_numpy(data)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif channels_first:
        tensor = tensor.transpose(0, 1)

    return tensor, int(samplerate)


def _lazy_import_pyannote() -> object:
    try:
        # pyannote warns (with a long install guide) when torchcodec cannot load, e.g. when the system
        # FFmpeg is newer than torchcodec supports. PyScribe hands pyannote already-decoded audio
        # (see _load_audio_for_pyannote), so the warning is irrelevant; log one short line instead.
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings("always", message=r".*torchcodec.*", category=UserWarning)
            from pyannote.audio import Pipeline  # type: ignore
        for item in caught:
            if "torchcodec" in str(item.message):
                LOGGER.info("torchcodec is unavailable; pyannote will use PyScribe's in-memory audio loading.")
            else:
                warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for diarization. Install with: pip install pyannote.audio"
        ) from e
    return Pipeline


def _from_pretrained(Pipeline: object, name: str, token: str | None) -> object:
    """pyannote.audio 4.x renamed `use_auth_token` to `token`; support both."""
    try:
        return Pipeline.from_pretrained(name, token=token)
    except TypeError:
        return Pipeline.from_pretrained(name, use_auth_token=token)


COMMUNITY_PIPELINE = "pyannote/speaker-diarization-community-1"
_LEGACY_PIPELINES = (("3.1", "pyannote/speaker-diarization-3.1"), ("3.0", "pyannote/speaker-diarization-3.0"))


def _pyannote_major_version() -> int:
    try:
        from importlib.metadata import version

        return int(version("pyannote.audio").split(".")[0])
    except Exception:
        return 3


def _pipeline_candidates() -> list[tuple[str, str]]:
    """Ordered (label, repo id) pairs to try; community-1 needs pyannote.audio 4.x."""
    candidates = list(_LEGACY_PIPELINES)
    if _pyannote_major_version() >= 4:
        candidates.insert(0, ("community-1", COMMUNITY_PIPELINE))
    return candidates


def _load_pyannote_pipeline(Pipeline: object, token: str | None, requested_device: str) -> tuple[object, str]:
    candidates = _pipeline_candidates()
    LOGGER.info(
        "Loading pyannote diarization pipeline candidates=%s token_present=%s requested_device=%s",
        [label for label, _ in candidates],
        bool(token),
        requested_device,
    )

    last_error: Exception | None = None
    for label, repo_id in candidates:
        try:
            return _from_pretrained(Pipeline, repo_id, token), label
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Failed to load pyannote pipeline %s; trying the next candidate. reason=%s",
                label,
                exc,
                exc_info=True,
            )
    LOGGER.error(
        "Failed to load any pyannote pipeline candidates=%s requested_device=%s torch_diag=%s",
        [label for label, _ in candidates],
        requested_device,
        _torch_cuda_snapshot(),
    )
    tried = " then ".join(label for label, _ in candidates)
    hint = (
        " Accept the model terms for pyannote/speaker-diarization-community-1 on Hugging Face "
        "and make sure your Hugging Face token is configured."
        if candidates[0][0] == "community-1"
        else ""
    )
    raise RuntimeError(f"Failed to load pyannote pipeline ({tried}): {last_error}.{hint}") from last_error


def _load_audio_for_pyannote(audio_path: str) -> dict[str, object]:
    """Decode audio with soundfile so pyannote never needs torchcodec/ffmpeg for file IO."""
    waveform, sample_rate = _direct_soundfile_load(audio_path)
    return {"waveform": waveform, "sample_rate": sample_rate}


class _ProgressHook:
    """pyannote pipeline hook that reports overall progress (0-100) from step progress."""

    # Share of the overall run attributed to each pyannote step.
    _STEP_RANGES = {"segmentation": (0.0, 40.0), "embeddings": (40.0, 95.0)}

    def __init__(self, progress_cb: ProgressCallback | None) -> None:
        self._progress_cb = progress_cb

    def __call__(self, step_name: str, step_artifact: object, file: object = None, total=None, completed=None) -> None:
        if self._progress_cb is None or step_name not in self._STEP_RANGES or not total:
            return
        low, high = self._STEP_RANGES[step_name]
        fraction = max(0.0, min(1.0, float(completed or 0) / float(total)))
        try:
            self._progress_cb(low + (high - low) * fraction)
        except Exception:
            LOGGER.debug("Diarization progress callback failed", exc_info=True)


def _speaker_annotation(output: object) -> object:
    """Return the annotation to merge with ASR from a pipeline result.

    pyannote 4.x returns an object; `exclusive_speaker_diarization` (no overlapped speech)
    is built for ASR merging. 3.x returns the Annotation directly.
    """
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(output, attr, None)
        if annotation is not None:
            return annotation
    return output


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
    token = get_hf_token()
    LOGGER.info("Preparing pyannote diarization requested_device=%s", device)
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
        audio_input = _load_audio_for_pyannote(audio_path)
        diarization = _speaker_annotation(
            pipeline(audio_input, num_speakers=max_speakers, hook=_ProgressHook(progress_cb))
        )
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
