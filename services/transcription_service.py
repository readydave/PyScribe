"""Shared transcription pipeline used by desktop and listener frontends."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import platform
import queue
from pathlib import Path
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from threading import Event
from typing import Callable

from services.granite_speech_service import GraniteSpeechModelBundle, transcribe_granite_audio
from services.model_download_service import ensure_model_cached
from services.model_service import TranscriptionModelSpec, load_model, resolve_transcription_model
from utils import convert_to_16k_mono, get_ffmpeg_cmd, load_audio_waveform


StatusCallback = Callable[[str], None]
TextCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]
LOGGER = logging.getLogger(__name__)
STREAM_TEXT_UPDATE_INTERVAL_SECONDS = 0.30
_PYANNOTE_BACKENDS = {"accurate", "fast"}


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
    if backend not in _PYANNOTE_BACKENDS:
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
        "cudnn",
        "subprocess exited unexpectedly",
        "exit code -",
        "expected all tensors to be on the same device",
    )
    return any(marker in message for marker in cuda_linker_markers)


def _should_isolate_diarization_backend(backend: str) -> bool:
    """Run pyannote diarization in a fresh process to avoid cuDNN state conflicts."""
    return backend in _PYANNOTE_BACKENDS


def _can_spawn_isolated_diarization() -> bool:
    """Returns True when the current __main__ module can be re-imported by spawn."""
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", "")
    return bool(main_file and not str(main_file).startswith("<"))


def _isolated_diarization_entry(
    audio_path: str,
    backend: str,
    device: str,
    max_speakers: int | None,
    event_queue: mp.Queue,
) -> None:
    try:
        from services.runtime_env_service import configure_runtime_environment, prepare_linux_dynamic_loader_environment

        configure_runtime_environment()
        prepare_linux_dynamic_loader_environment()

        from diar_backends import run_diarization_backend

        def _status_cb(text: str) -> None:
            event_queue.put({"type": "status", "value": str(text)})

        def _progress_cb(value: float) -> None:
            event_queue.put({"type": "progress", "value": float(value)})

        segments = run_diarization_backend(
            audio_path=audio_path,
            backend=backend,
            device=device,
            max_speakers=max_speakers,
            progress_cb=_progress_cb,
            status_cb=_status_cb,
        )
        event_queue.put({"type": "result", "value": segments})
    except Exception as exc:
        event_queue.put({"type": "error", "value": str(exc)})


def _stop_process(proc: mp.Process, *, reason: str) -> None:
    try:
        if not proc.is_alive():
            return
    except Exception:
        return

    LOGGER.warning("Stopping diarization subprocess pid=%s reason=%s", getattr(proc, "pid", None), reason)
    try:
        proc.terminate()
    except Exception:
        pass

    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        try:
            if not proc.is_alive():
                break
        except Exception:
            break
        time.sleep(0.05)

    try:
        if proc.is_alive():
            proc.kill()
    except Exception:
        pass

    try:
        proc.join(timeout=1.0)
    except Exception:
        pass


def _run_diarization_backend_in_subprocess(
    *,
    audio_path: str,
    backend: str,
    device: str,
    max_speakers: int | None,
    cancel_event: Event | None,
    progress_cb: ProgressCallback | None,
    status_cb: StatusCallback | None,
) -> list[dict]:
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    proc = ctx.Process(
        target=_isolated_diarization_entry,
        args=(audio_path, backend, device, max_speakers, event_queue),
    )
    result: list[dict] | None = None
    error_text: str | None = None

    proc.start()
    try:
        while True:
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Cancelled during diarization.")

            drained = False
            while True:
                try:
                    evt = event_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                etype = evt.get("type")
                value = evt.get("value")
                if etype == "status" and status_cb:
                    status_cb(str(value))
                elif etype == "progress" and progress_cb:
                    progress_cb(float(value))
                elif etype == "result":
                    result = list(value)
                elif etype == "error":
                    error_text = str(value)

            if result is not None or error_text is not None:
                break
            if not proc.is_alive() and not drained:
                break
            time.sleep(0.05)

        proc.join(timeout=2.0)

        while True:
            try:
                evt = event_queue.get_nowait()
            except queue.Empty:
                break
            etype = evt.get("type")
            value = evt.get("value")
            if etype == "status" and status_cb:
                status_cb(str(value))
            elif etype == "progress" and progress_cb:
                progress_cb(float(value))
            elif etype == "result":
                result = list(value)
            elif etype == "error":
                error_text = str(value)

        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Cancelled during diarization.")
        if result is not None:
            return result
        if error_text is not None:
            raise RuntimeError(error_text)
        raise RuntimeError(
            f"Diarization subprocess exited unexpectedly (exit code {getattr(proc, 'exitcode', None)})."
        )
    finally:
        try:
            if proc.is_alive():
                _stop_process(proc, reason="diarization cleanup")
        except Exception:
            pass
        try:
            event_queue.close()
            event_queue.join_thread()
        except Exception:
            pass


def _run_diarization_backend(
    *,
    audio_path: str,
    backend: str,
    device: str,
    max_speakers: int | None,
    cancel_event: Event | None,
    progress_cb: ProgressCallback | None,
    status_cb: StatusCallback | None,
) -> list[dict]:
    if _should_isolate_diarization_backend(backend) and _can_spawn_isolated_diarization():
        return _run_diarization_backend_in_subprocess(
            audio_path=audio_path,
            backend=backend,
            device=device,
            max_speakers=max_speakers,
            cancel_event=cancel_event,
            progress_cb=progress_cb,
            status_cb=status_cb,
        )
    if _should_isolate_diarization_backend(backend):
        LOGGER.info("Skipping isolated diarization process because __main__ is not spawn-importable.")

    from diar_backends import run_diarization_backend

    return run_diarization_backend(
        audio_path=audio_path,
        backend=backend,
        device=device,
        max_speakers=max_speakers,
        progress_cb=progress_cb,
        status_cb=status_cb,
    )


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
    import ffmpeg

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
    model: object,
    language: str | None,
    *,
    model_spec: TranscriptionModelSpec | None = None,
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
    spec = model_spec or resolve_transcription_model("")
    audio_np = load_audio_waveform(wav_path)

    if spec.backend_kind == "granite_transformers":
        if cancel_event and cancel_event.is_set():
            return TranscriptionResult(
                transcript="",
                transcript_only="",
                visual_report="",
                segments=[],
                cancelled=True,
                duration_seconds=duration,
                transcription_seconds=0.0,
                diarization_seconds=0.0,
                visual_analysis_seconds=0.0,
            )
        transcription_started = time.perf_counter()
        transcript = transcribe_granite_audio(
            model if isinstance(model, GraniteSpeechModelBundle) else model,
            audio_np,
            progress_cb=on_progress,
        )
        transcription_seconds = time.perf_counter() - transcription_started
        if on_text and transcript:
            on_text(transcript)
        was_cancelled = bool(cancel_event and cancel_event.is_set())
        return TranscriptionResult(
            transcript="" if was_cancelled else transcript,
            transcript_only="" if was_cancelled else transcript,
            visual_report="",
            segments=[],
            cancelled=was_cancelled,
            duration_seconds=duration,
            transcription_seconds=transcription_seconds,
            diarization_seconds=0.0,
            visual_analysis_seconds=0.0,
        )

    task = "transcribe"
    segments_generator, _ = model.transcribe(audio_np, task=task, language=language, beam_size=5)

    all_text_segments: list[str] = []
    streamed_text = ""
    all_segments_struct: list[dict] = []
    transcription_started = time.perf_counter()
    last_text_emit = transcription_started
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
                transcript=streamed_text,
                transcript_only=streamed_text,
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
        if segment_text:
            streamed_text = f"{streamed_text} {segment_text}".strip() if streamed_text else segment_text
        all_segments_struct.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment_text,
            }
        )

        now = time.perf_counter()
        if on_text and (now - last_text_emit >= STREAM_TEXT_UPDATE_INTERVAL_SECONDS):
            on_text(streamed_text)
            last_text_emit = now
        if on_progress and duration > 0:
            on_progress((segment.end / duration) * 100)

    if on_text and streamed_text:
        on_text(streamed_text)

    transcript = streamed_text
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
                diar_segments = _run_diarization_backend(
                    audio_path=wav_path,
                    backend=diar_backend,
                    device=device,
                    max_speakers=max_speakers,
                    cancel_event=cancel_event,
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
                    diar_segments = _run_diarization_backend(
                        audio_path=wav_path,
                        backend=diar_backend,
                        device="cpu",
                        max_speakers=max_speakers,
                        cancel_event=cancel_event,
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

            from diarization import assign_speakers

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
    visual_ocr_backend: str = "auto",
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
        from services.multimodal_service import analyze_video_stream

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

        model_spec = resolve_transcription_model(model_name)
        if use_diarization and not model_spec.supports_diarization:
            if on_status:
                on_status(
                    f"Diarization unavailable for '{model_spec.display_name}'. "
                    "Continuing without speaker identification."
                )
            use_diarization = False

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
            model_spec=model_spec,
        )

        if on_status:
            on_status(f"Transcribing with '{model_name}'...")
        result = transcribe_prepared_audio(
            wav_path=wav_path,
            model=model,
            language=language,
            model_spec=model_spec,
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

        from services.multimodal_service import analyze_video_stream

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
