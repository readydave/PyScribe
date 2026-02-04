"""Shared transcription pipeline used by desktop and listener frontends."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from threading import Event
from typing import Callable

import ffmpeg
from faster_whisper import WhisperModel

from diar_backends import run_diarization_backend
from diarization import assign_speakers
from services.multimodal_service import analyze_video_stream
from services.model_download_service import ensure_model_cached
from services.model_service import load_model
from utils import convert_to_16k_mono, get_ffmpeg_cmd, load_audio_waveform


StatusCallback = Callable[[str], None]
TextCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]
LOGGER = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    transcript: str
    transcript_only: str
    visual_report: str
    segments: list[dict]
    cancelled: bool
    duration_seconds: float
    transcription_seconds: float
    diarization_seconds: float
    visual_analysis_seconds: float


def _should_retry_diarization_on_cpu(exc: Exception, *, backend: str, device: str) -> bool:
    """Returns True when diarization failed due to likely CUDA runtime/linker issues."""
    if device.lower() != "cuda":
        return False
    if backend not in {"accurate", "fast"}:
        # sortformer is CUDA-only; don't auto-switch backend implicitly.
        return False
    message = str(exc).lower()
    cuda_linker_markers = (
        "libnvrtc",
        "libcudart",
        "cuda failed",
        "cuda error",
        "cuda initialization",
        "failed call to cuinit",
        "no cuda gpus are available",
        "torch.cuda",
    )
    return any(marker in message for marker in cuda_linker_markers)


def _collect_cuda_diagnostics() -> str:
    """Collect lightweight CUDA runtime diagnostics for fallback logging."""
    parts: list[str] = [f"env.LD_LIBRARY_PATH={_short_env('LD_LIBRARY_PATH', max_len=280)}"]
    try:
        import torch  # local import to keep startup costs unchanged

        parts.append(f"torch={getattr(torch, '__version__', '<unknown>')}")
        parts.append(f"torch.version.cuda={getattr(getattr(torch, 'version', object()), 'cuda', None)}")
        try:
            parts.append(f"torch.cuda.is_available={torch.cuda.is_available()}")
        except Exception as exc:
            parts.append(f"torch.cuda.is_available_error={exc}")
        try:
            parts.append(f"torch.cuda.device_count={torch.cuda.device_count()}")
        except Exception as exc:
            parts.append(f"torch.cuda.device_count_error={exc}")
    except Exception as exc:
        parts.append(f"torch_import_error={exc}")

    try:
        nvrtc_candidates = sorted(
            str(path)
            for path in Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*")
        )
        if nvrtc_candidates:
            parts.append(f"nvrtc_candidates={';'.join(nvrtc_candidates[:4])}")
        else:
            parts.append("nvrtc_candidates=<none-under-sys.prefix>")
    except Exception as exc:
        parts.append(f"nvrtc_scan_error={exc}")
    return " | ".join(parts)


def _probe_duration_seconds(wav_path: str) -> float:
    try:
        probe = ffmpeg.probe(wav_path)
        return float(probe["format"]["duration"])
    except (ffmpeg.Error, KeyError, ValueError, TypeError):
        return 0.0


def _format_speaker_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "S?")
        lines.append(f"[{speaker}] {seg['text']}")
    return "\n".join(lines)


def transcribe_prepared_audio(
    wav_path: str,
    model: WhisperModel,
    language: str | None,
    *,
    cancel_event: Event | None = None,
    use_diarization: bool = False,
    diar_backend: str = "accurate",
    device: str = "cpu",
    max_speakers: int | None = None,
    on_status: StatusCallback | None = None,
    on_text: TextCallback | None = None,
    on_progress: ProgressCallback | None = None,
    on_diar_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """
    Transcribes a prepared 16k mono wav file and optionally runs diarization.
    """
    duration = _probe_duration_seconds(wav_path)
    task = "transcribe"
    segments_generator, _ = model.transcribe(audio_np := load_audio_waveform(wav_path), task=task, language=language, beam_size=5)

    all_text_segments: list[str] = []
    all_segments_struct: list[dict] = []
    transcription_started = time.perf_counter()
    diarization_seconds = 0.0

    if on_diar_progress:
        on_diar_progress(0)

    for segment in segments_generator:
        if cancel_event and cancel_event.is_set():
            transcription_seconds = time.perf_counter() - transcription_started
            LOGGER.info(
                "Transcription cancelled during ASR audio_seconds=%.2f transcribe_seconds=%.2f diar_seconds=%.2f",
                duration,
                transcription_seconds,
                diarization_seconds,
            )
            return TranscriptionResult(
                transcript=" ".join(all_text_segments).strip(),
                transcript_only=" ".join(all_text_segments).strip(),
                visual_report="",
                segments=all_segments_struct,
                cancelled=True,
                duration_seconds=duration,
                transcription_seconds=transcription_seconds,
                diarization_seconds=diarization_seconds,
                visual_analysis_seconds=0.0,
            )

        segment_text = segment.text.strip()
        all_text_segments.append(segment_text)
        all_segments_struct.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment_text,
            }
        )

        partial_text = " ".join(all_text_segments).strip()
        if on_text:
            on_text(partial_text)
        if on_progress and duration > 0:
            on_progress((segment.end / duration) * 100)

    transcript = " ".join(all_text_segments).strip()
    final_segments = all_segments_struct
    transcription_seconds = time.perf_counter() - transcription_started

    if cancel_event and cancel_event.is_set():
        LOGGER.info(
            "Transcription cancelled after ASR audio_seconds=%.2f transcribe_seconds=%.2f diar_seconds=%.2f",
            duration,
            transcription_seconds,
            diarization_seconds,
        )
        return TranscriptionResult(
            transcript=transcript,
            transcript_only=transcript,
            visual_report="",
            segments=final_segments,
            cancelled=True,
            duration_seconds=duration,
            transcription_seconds=transcription_seconds,
            diarization_seconds=diarization_seconds,
            visual_analysis_seconds=0.0,
        )

    if use_diarization:
        diarization_started = time.perf_counter()
        try:
            if on_status:
                on_status(f"Running diarization ({diar_backend}) on {device.upper()} (detecting speakers)...")
            if on_diar_progress:
                on_diar_progress(25)

            def _diar_progress(value: float) -> None:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Cancelled during diarization.")
                if on_diar_progress:
                    on_diar_progress(value)

            try:
                diar_segments = run_diarization_backend(
                    audio_path=wav_path,
                    backend=diar_backend,
                    device=device,
                    max_speakers=max_speakers,
                    progress_cb=_diar_progress,
                    status_cb=on_status,
                )
            except Exception as diar_exc:
                if not _should_retry_diarization_on_cpu(diar_exc, backend=diar_backend, device=device):
                    raise
                LOGGER.warning(
                    "Diarization on CUDA failed for backend=%s; retrying on CPU. reason=%s diagnostics=%s",
                    diar_backend,
                    diar_exc,
                    _collect_cuda_diagnostics(),
                    exc_info=True,
                )
                if on_status:
                    on_status(
                        "Diarization CUDA runtime unavailable. Retrying diarization on CPU (slower, keeps speakers)..."
                    )
                try:
                    diar_segments = run_diarization_backend(
                        audio_path=wav_path,
                        backend=diar_backend,
                        device="cpu",
                        max_speakers=max_speakers,
                        progress_cb=_diar_progress,
                        status_cb=on_status,
                    )
                    LOGGER.info(
                        "Diarization CPU retry succeeded backend=%s requested_device=%s",
                        diar_backend,
                        device,
                    )
                except Exception:
                    LOGGER.error(
                        "Diarization CPU retry failed backend=%s requested_device=%s diagnostics=%s",
                        diar_backend,
                        device,
                        _collect_cuda_diagnostics(),
                        exc_info=True,
                    )
                    raise

            if on_status:
                on_status("Assigning speakers to transcript...")
            if on_diar_progress:
                on_diar_progress(65)

            final_segments = assign_speakers(all_segments_struct, diar_segments)
            transcript = _format_speaker_transcript(final_segments)

            if on_diar_progress:
                on_diar_progress(100)
            diarization_seconds = time.perf_counter() - diarization_started
        except InterruptedError:
            diarization_seconds = time.perf_counter() - diarization_started
            LOGGER.info(
                "Transcription cancelled during diarization audio_seconds=%.2f transcribe_seconds=%.2f diar_seconds=%.2f backend=%s",
                duration,
                transcription_seconds,
                diarization_seconds,
                diar_backend,
            )
            return TranscriptionResult(
                transcript=transcript,
                transcript_only=transcript,
                visual_report="",
                segments=final_segments,
                cancelled=True,
                duration_seconds=duration,
                transcription_seconds=transcription_seconds,
                diarization_seconds=diarization_seconds,
                visual_analysis_seconds=0.0,
            )
        except Exception as exc:
            diarization_seconds = time.perf_counter() - diarization_started
            # Preserve transcript output even when diarization backend fails.
            diag_msg = str(exc)
            if "failed to parse CPython sys.version" in diag_msg:
                diag_msg += (
                    "\n\nDetected interpreter string parsing issue. If this persists, run PyScribe from a clean shell "
                    "without Conda/Pinokio LD_LIBRARY_PATH overrides."
                )
            if on_status:
                on_status(f"Diarization unavailable, continuing without speakers: {diag_msg}")
            if on_diar_progress:
                on_diar_progress(0)

    LOGGER.info(
        "Transcription pipeline completed cancelled=%s audio_seconds=%.2f transcribe_seconds=%.2f diar_seconds=%.2f diar_enabled=%s backend=%s",
        False,
        duration,
        transcription_seconds,
        diarization_seconds,
        use_diarization,
        diar_backend,
    )
    return TranscriptionResult(
        transcript=transcript,
        transcript_only=transcript,
        visual_report="",
        segments=final_segments,
        cancelled=False,
        duration_seconds=duration,
        transcription_seconds=transcription_seconds,
        diarization_seconds=diarization_seconds,
        visual_analysis_seconds=0.0,
    )


def transcribe_media_file(
    media_path: str,
    model_name: str,
    *,
    run_mode: str = "full",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    cancel_event: Event | None = None,
    use_diarization: bool = False,
    diar_backend: str = "accurate",
    max_speakers: int | None = None,
    use_visual_analysis: bool = False,
    visual_profile: str = "balanced",
    visual_ocr_backend: str = "paddleocr",
    visual_sample_seconds: float = 1.0,
    on_status: StatusCallback | None = None,
    on_text: TextCallback | None = None,
    on_progress: ProgressCallback | None = None,
    on_diar_progress: ProgressCallback | None = None,
    on_visual_progress: ProgressCallback | None = None,
    on_model_download_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """
    End-to-end transcription for media input files used by listener mode.
    """
    job_id = uuid.uuid4().hex[:8]
    _log_job_technical_header(
        job_id=job_id,
        media_path=media_path,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        use_diarization=use_diarization,
        diar_backend=diar_backend,
        max_speakers=max_speakers,
        use_visual_analysis=use_visual_analysis,
        visual_profile=visual_profile,
        visual_ocr_backend=visual_ocr_backend,
        visual_sample_seconds=visual_sample_seconds,
        run_mode=run_mode,
    )

    run_mode = str(run_mode or "full").strip().lower()
    if run_mode not in {"full", "transcribe_only", "visual_only"}:
        run_mode = "full"

    if run_mode == "visual_only":
        visual = analyze_video_stream(
            media_path,
            ocr_backend=visual_ocr_backend,
            visual_profile=visual_profile,
            sample_seconds=visual_sample_seconds,
            cancel_event=cancel_event,
            on_status=on_status,
            on_progress=on_visual_progress,
        )
        text = visual.report if visual.report else ""
        LOGGER.info(
            "Job[%s] visual-only completed cancelled=%s visual_available=%s visual_seconds=%.2f",
            job_id,
            visual.cancelled,
            visual.available,
            visual.elapsed_seconds,
        )
        return TranscriptionResult(
            transcript=text,
            transcript_only="",
            visual_report=visual.report,
            segments=[],
            cancelled=visual.cancelled,
            duration_seconds=0.0,
            transcription_seconds=0.0,
            diarization_seconds=0.0,
            visual_analysis_seconds=visual.elapsed_seconds,
        )

    ffmpeg_cmd = get_ffmpeg_cmd(tool="ffmpeg")
    if not ffmpeg_cmd:
        raise FileNotFoundError("ffmpeg not found.")

    with tempfile.TemporaryDirectory() as temp_dir:
        if on_status:
            on_status("Preparing audio...")
        wav_path = convert_to_16k_mono(media_path, temp_dir, ffmpeg_cmd)

        if on_status:
            on_status(f"Loading model '{model_name}' on {device.upper()}...")
        model_ref = ensure_model_cached(
            model_name,
            on_status=on_status,
            on_progress=on_model_download_progress,
        )
        model = load_model(
            model_ref,
            device=device,
            compute_type=compute_type,
            use_cache=True,
        )

        if on_status:
            on_status(f"Transcribing with '{model_name}'...")
        result = transcribe_prepared_audio(
            wav_path=wav_path,
            model=model,
            language=language,
            cancel_event=cancel_event,
            use_diarization=use_diarization,
            diar_backend=diar_backend,
            device=device,
            max_speakers=max_speakers,
            on_status=on_status,
            on_text=on_text,
            on_progress=on_progress,
            on_diar_progress=on_diar_progress,
        )

        if run_mode == "transcribe_only" or not use_visual_analysis or result.cancelled:
            LOGGER.info("Job[%s] completed without visual analysis cancelled=%s", job_id, result.cancelled)
            return result

        visual = analyze_video_stream(
            media_path,
            ocr_backend=visual_ocr_backend,
            visual_profile=visual_profile,
            sample_seconds=visual_sample_seconds,
            cancel_event=cancel_event,
            on_status=on_status,
            on_progress=on_visual_progress,
        )
        if on_status and visual.available and not visual.cancelled:
            on_status("Visual analysis complete.")
        transcript_with_visual = result.transcript
        if visual.report:
            transcript_with_visual = f"{result.transcript}\n\n{visual.report}".strip()

        LOGGER.info(
            "Job[%s] completed cancelled=%s visual_available=%s visual_seconds=%.2f",
            job_id,
            result.cancelled or visual.cancelled,
            visual.available,
            visual.elapsed_seconds,
        )

        return TranscriptionResult(
            transcript=transcript_with_visual,
            transcript_only=result.transcript,
            visual_report=visual.report,
            segments=result.segments,
            cancelled=result.cancelled or visual.cancelled,
            duration_seconds=result.duration_seconds,
            transcription_seconds=result.transcription_seconds,
            diarization_seconds=result.diarization_seconds,
            visual_analysis_seconds=visual.elapsed_seconds,
        )


def _log_job_technical_header(
    *,
    job_id: str,
    media_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    use_diarization: bool,
    diar_backend: str,
    max_speakers: int | None,
    use_visual_analysis: bool,
    visual_profile: str,
    visual_ocr_backend: str,
    visual_sample_seconds: float,
    run_mode: str,
) -> None:
    lines = [
        f"=== Job Technical [{job_id}] ===",
        f"media_path={media_path}",
        f"model={model_name} device={device} compute_type={compute_type}",
        f"diarization={use_diarization} backend={diar_backend} max_speakers={max_speakers}",
        (
            f"visual={use_visual_analysis} profile={visual_profile} backend={visual_ocr_backend} "
            f"sample_seconds={visual_sample_seconds}"
        ),
        f"run_mode={run_mode}",
        f"python={sys.version.splitlines()[0]} executable={sys.executable}",
        f"platform={platform.platform()}",
        f"cwd={os.getcwd()}",
        f"env.HF_HOME={_short_env('HF_HOME')}",
        f"env.HUGGINGFACE_HUB_CACHE={_short_env('HUGGINGFACE_HUB_CACHE')}",
        f"env.PADDLE_HOME={_short_env('PADDLE_HOME')}",
        f"env.PADDLE_PDX_CACHE_HOME={_short_env('PADDLE_PDX_CACHE_HOME')}",
        f"env.PADDLE_PDX_MODEL_SOURCE={_short_env('PADDLE_PDX_MODEL_SOURCE')}",
        f"env.PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK={_short_env('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK')}",
        f"env.PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT={_short_env('PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT')}",
        f"env.FLAGS_enable_pir_in_executor={_short_env('FLAGS_enable_pir_in_executor')}",
        f"env.LD_LIBRARY_PATH={_short_env('LD_LIBRARY_PATH', max_len=280)}",
        "=== End Job Technical ===",
    ]
    LOGGER.info("\n%s", "\n".join(lines))


def _short_env(name: str, *, max_len: int = 180) -> str:
    value = os.environ.get(name)
    if value is None:
        return "<unset>"
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."
