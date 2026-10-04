"""faster-whisper decoding helpers: VRAM-aware batching, OOM fallback, word timestamps."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from services.live_vram_service import get_gpu_memory_info

LOGGER = logging.getLogger(__name__)

# (minimum free VRAM in GB, batch size). Batched large-v3-turbo needs roughly
# 0.25 GB per extra batch item on top of the model; leave headroom for OCR/diarization.
_BATCH_TIERS = ((8.0, 16), (5.0, 8), (3.5, 4))


def resolve_batch_size(device: str, requested: int | None = None) -> int:
    """Return the decode batch size (1 means sequential).

    Batching is GPU-only: on CPU it was faster but measurably less accurate in the
    Phase 2 evaluation, so CPU stays sequential unless a size is requested explicitly.
    """
    if requested is not None:
        return max(1, int(requested))
    if device.lower() != "cuda":
        return 1
    info = get_gpu_memory_info()
    if info is None:
        return 1
    for min_free_gb, size in _BATCH_TIERS:
        if info.free_gb >= min_free_gb:
            return size
    return 1


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda failed with error out of memory" in message


def _transcribe(
    model: Any,
    audio_np: Any,
    language: str | None,
    *,
    batch_size: int,
    hotwords: str | None,
    word_timestamps: bool,
    start_seconds: float = 0.0,
) -> Iterator[Any]:
    kwargs: dict[str, Any] = {
        "task": "transcribe",
        "language": language,
        "beam_size": 5,
        "hotwords": (hotwords or "").strip() or None,
        "word_timestamps": word_timestamps,
    }
    if start_seconds > 0:
        kwargs["clip_timestamps"] = [start_seconds]
    if batch_size > 1:
        from faster_whisper import BatchedInferencePipeline

        segments, _ = BatchedInferencePipeline(model).transcribe(audio_np, batch_size=batch_size, **kwargs)
    else:
        segments, _ = model.transcribe(audio_np, vad_filter=True, **kwargs)
    return segments


def open_segment_stream(
    model: Any,
    audio_np: Any,
    language: str | None,
    *,
    batch_size: int = 1,
    hotwords: str | None = None,
    word_timestamps: bool = False,
) -> Iterator[Any]:
    """Return an iterator of decoded segments.

    If batched decoding runs out of CUDA memory (on the first batch or mid-stream),
    decoding resumes sequentially from the last yielded segment.
    """
    options = {"hotwords": hotwords, "word_timestamps": word_timestamps}
    last_end = 0.0
    try:
        for segment in _transcribe(model, audio_np, language, batch_size=batch_size, **options):
            last_end = float(getattr(segment, "end", last_end))
            yield segment
        return
    except RuntimeError as exc:
        if batch_size <= 1 or not is_oom_error(exc):
            raise
    LOGGER.warning(
        "Batched decode (batch_size=%d) ran out of memory at %.1fs; resuming sequentially.", batch_size, last_end
    )
    _free_cuda_cache()
    # The first CUDA call after an OOM fails once with "invalid device ordinal"; the next one works.
    for attempt in (1, 2):
        try:
            for segment in _transcribe(
                model, audio_np, language, batch_size=1, start_seconds=last_end, **options
            ):
                last_end = float(getattr(segment, "end", last_end))
                yield segment
            return
        except RuntimeError as exc:
            if attempt == 2 or "invalid device ordinal" not in str(exc).lower():
                raise
            LOGGER.warning("Sequential retry hit a stale CUDA context after OOM; retrying once.")


def segment_words(segment: Any) -> list[dict[str, Any]]:
    """Extract word-level timings from a faster-whisper segment (empty if unavailable)."""
    return [
        {
            "start": float(word.start),
            "end": float(word.end),
            "word": str(word.word),
            "probability": round(float(getattr(word, "probability", 0.0)), 4),
        }
        for word in (getattr(segment, "words", None) or [])
    ]


def _free_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # best effort only
        LOGGER.debug("Could not clear CUDA cache", exc_info=True)
