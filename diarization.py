# diarization.py
# Speaker diarization helpers for PyScribe.

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Callable

import numpy as np
import torch
import torchaudio

from services.hf_auth_service import get_hf_token
from services.runtime_compat import ensure_platform_sys_version_compat

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[float], None]
StatusCallback = Callable[[str], None]
Segment = dict[str, object]

_TORCHAUDIO_SOUNDFILE_PATCHED = False

# PyTorch 2.6+ changed default of torch.load to weights_only=True.
# We must allowlist classes used by pyannote/speechbrain pipelines.
if hasattr(torch.serialization, "add_safe_globals"):
    try:
        # Basic torch globals
        _trusted = [torch.torch_version.TorchVersion]
        # Add numpy globals if available
        if hasattr(np, "dtype"):
            _trusted.append(np.dtype)
        try:
            import numpy.core.multiarray
            _trusted.append(numpy.core.multiarray.scalar)
        except (ImportError, AttributeError):
            pass
        torch.serialization.add_safe_globals(_trusted)
    except Exception as exc:
        LOGGER.warning("Failed to update PyTorch safe globals: %s", exc)

# Robust monkeypatch for legacy torchaudio backend APIs removed in 2.9+
if not hasattr(torchaudio, "set_audio_backend"):
    def _noop_set_backend(backend: str | None) -> None:
        pass
    torchaudio.set_audio_backend = _noop_set_backend  # type: ignore

if not hasattr(torchaudio, "list_audio_backends"):
    def _noop_list_backends() -> list[str]:
        return ["soundfile", "ffmpeg"]
    torchaudio.list_audio_backends = _noop_list_backends  # type: ignore

if not hasattr(torchaudio, "get_audio_backend"):
    def _noop_get_backend() -> str:
        return "soundfile"
    torchaudio.get_audio_backend = _noop_get_backend  # type: ignore

# Guard against 'No module named torchaudio.backend' in modern torchaudio
if "torchaudio.backend" not in sys.modules:
    _dummy_backend = types.ModuleType("torchaudio.backend")
    # Some older torchaudio-dependent code might look for 'common' or 'utils' inside backend
    _dummy_backend_common = types.ModuleType("torchaudio.backend.common")
    
    # Modern torchaudio (e.g. 2.11+) may have removed AudioMetaData entirely from public API.
    # Provide the expected dataclass structure as a stub.
    from dataclasses import dataclass
    
    @dataclass(frozen=True)
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    if hasattr(torchaudio, "AudioMetaData"):
        _dummy_backend_common.AudioMetaData = torchaudio.AudioMetaData  # type: ignore
    else:
        _dummy_backend_common.AudioMetaData = AudioMetaData  # type: ignore
    
    _dummy_backend.common = _dummy_backend_common  # type: ignore
    sys.modules["torchaudio.backend"] = _dummy_backend
    sys.modules["torchaudio.backend.common"] = _dummy_backend_common


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


def _soundfile_info(uri: str | os.PathLike, *args, **kwargs):
    import soundfile as sf

    info = sf.info(uri)
    metadata_cls = getattr(torchaudio, "AudioMetaData", None)
    if metadata_cls is None:
        metadata_cls = sys.modules["torchaudio.backend.common"].AudioMetaData
    return metadata_cls(
        sample_rate=int(info.samplerate),
        num_frames=int(info.frames),
        num_channels=int(info.channels),
        bits_per_sample=0,
        encoding=str(info.subtype or info.format or "UNKNOWN"),
    )


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
    if hasattr(torchaudio, "list_audio_backends") and getattr(torchaudio.list_audio_backends, "__name__", None) != "_noop_list_backends":
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

    if original_info is None:
        torchaudio.info = _soundfile_info  # type: ignore[attr-defined]
    elif not getattr(original_info, "__pyscribe_soundfile_default__", False):
        torchaudio.info = _wrap_with_soundfile_default(original_info)

    if original_load and not getattr(original_load, "__pyscribe_soundfile_default__", False):
        torchaudio.load = _wrap_with_soundfile_default(original_load)

    if hasattr(torchaudio, "load_with_torchcodec") and not getattr(
        torchaudio.load_with_torchcodec,
        "__pyscribe_soundfile_direct__",
        False,
    ):
        setattr(_direct_soundfile_load, "__pyscribe_soundfile_direct__", True)
        torchaudio.load_with_torchcodec = _direct_soundfile_load  # type: ignore[attr-defined]

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

    # PyTorch 2.6+ defaults to weights_only=True, which breaks pyannote's complex pipeline load.
    # We temporarily monkeypatch torch.load to be permissive strictly during this trusted load.
    original_load = torch.load
    def permissive_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = permissive_load
    try:
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
    finally:
        torch.load = original_load


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
