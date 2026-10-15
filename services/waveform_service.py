"""Waveform peaks for the Transcript Studio: decode media to a few thousand normalised amplitude buckets.

Audio that libsndfile can open is read directly; everything else (mp3, video containers, ...) is decoded
by ``ffmpeg`` to 8 kHz mono PCM. Qt-free; the UI runs this on a worker thread.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_BUCKETS = 2000
PREVIEW_BUCKETS = 64
FFMPEG_RATE = 8000
FRAME_SAMPLES = FFMPEG_RATE // 50  # 20 ms frames
READ_CHUNK = 1 << 20
FFMPEG_TIMEOUT_S = 600


class WaveformError(RuntimeError):
    """The media could not be decoded."""


def _bucketize(frame_peaks: np.ndarray, buckets: int) -> list[float]:
    """Max-pool ``frame_peaks`` into ``buckets`` values and normalise to 0..1 (silence stays flat)."""
    if frame_peaks.size == 0:
        return []
    buckets = max(1, min(buckets, frame_peaks.size))
    edges = np.linspace(0, frame_peaks.size, buckets + 1, dtype=np.int64)[:-1]
    pooled = np.maximum.reduceat(frame_peaks, edges).astype(np.float64)
    top = float(pooled.max())
    if top <= 0:
        return [0.0] * buckets
    return [round(float(v), 3) for v in pooled / top]


def _peaks_from_soundfile(path: str, buckets: int) -> list[float] | None:
    """Peaks via libsndfile, or ``None`` when it cannot open the file."""
    try:
        import soundfile as sf

        with sf.SoundFile(path) as handle:
            frame = max(1, handle.samplerate // 50)
            carry = np.zeros(0, dtype=np.float32)
            parts: list[np.ndarray] = []
            for block in handle.blocks(blocksize=frame * 500, dtype="float32", always_2d=True):
                mono = np.abs(block).max(axis=1)
                data = np.concatenate([carry, mono])
                usable = (data.size // frame) * frame
                if usable:
                    parts.append(data[:usable].reshape(-1, frame).max(axis=1))
                carry = data[usable:]
            if carry.size:
                parts.append(np.array([carry.max()], dtype=np.float32))
    except Exception:
        return None
    return _bucketize(np.concatenate(parts), buckets) if parts else []


def _peaks_from_ffmpeg(path: str, buckets: int) -> list[float]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise WaveformError("ffmpeg is not available.")
    command = [ffmpeg, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(FFMPEG_RATE), "-f", "s16le", "-"]
    parts: list[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:  # noqa: S603
            if process.stdout is None:
                raise WaveformError("Could not read ffmpeg output.")
            leftover = b""
            while True:
                chunk = process.stdout.read(READ_CHUNK)
                if not chunk:
                    break
                chunk = leftover + chunk
                usable_bytes = len(chunk) - (len(chunk) % 2)
                leftover = chunk[usable_bytes:]
                samples = np.abs(np.frombuffer(chunk[:usable_bytes], dtype="<i2").astype(np.float32))
                data = np.concatenate([carry, samples])
                usable = (data.size // FRAME_SAMPLES) * FRAME_SAMPLES
                if usable:
                    parts.append(data[:usable].reshape(-1, FRAME_SAMPLES).max(axis=1))
                carry = data[usable:]
            process.wait(timeout=FFMPEG_TIMEOUT_S)
            if process.returncode != 0 and not parts:
                raise WaveformError("ffmpeg could not decode this file.")
    except (OSError, subprocess.SubprocessError) as exc:
        raise WaveformError(f"Could not run ffmpeg: {exc}") from exc
    if carry.size:
        parts.append(np.array([carry.max()], dtype=np.float32))
    if not parts:
        raise WaveformError("The file has no audio.")
    return _bucketize(np.concatenate(parts), buckets)


def compute_peaks(media_path: str, buckets: int = DEFAULT_BUCKETS) -> list[float]:
    """Normalised peaks (0..1) for ``media_path``. Raises ``WaveformError`` when it cannot be decoded."""
    path = str(media_path)
    if not Path(path).is_file():
        raise WaveformError("The media file is not available.")
    peaks = _peaks_from_soundfile(path, buckets)
    if peaks is not None:
        if not peaks:
            raise WaveformError("The file has no audio.")
        return peaks
    LOGGER.info("libsndfile cannot read this file; decoding with ffmpeg.")
    return _peaks_from_ffmpeg(path, buckets)


def preview_from(peaks: list[float], count: int = PREVIEW_BUCKETS) -> list[float]:
    """Downsample peaks for a project card."""
    if not peaks:
        return []
    return _bucketize(np.asarray(peaks, dtype=np.float32), count)
