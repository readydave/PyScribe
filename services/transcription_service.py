"""Shared transcription pipeline used by desktop and listener frontends."""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from threading import Event
from typing import Callable

import ffmpeg
from faster_whisper import WhisperModel

from diar_backends import run_diarization_backend
from diarization import assign_speakers
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
    segments: list[dict]
    cancelled: bool
    duration_seconds: float
    transcription_seconds: float
    diarization_seconds: float


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
                segments=all_segments_struct,
                cancelled=True,
                duration_seconds=duration,
                transcription_seconds=transcription_seconds,
                diarization_seconds=diarization_seconds,
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
            segments=final_segments,
            cancelled=True,
            duration_seconds=duration,
            transcription_seconds=transcription_seconds,
            diarization_seconds=diarization_seconds,
        )

    if use_diarization:
        diarization_started = time.perf_counter()
        try:
            if on_status:
                on_status(f"Running diarization ({diar_backend}) on {device.upper()} (detecting speakers)...")
            if on_diar_progress:
                on_diar_progress(25)

            def _diar_progress(value: float):
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Cancelled during diarization.")
                if on_diar_progress:
                    on_diar_progress(value)

            diar_segments = run_diarization_backend(
                audio_path=wav_path,
                backend=diar_backend,
                device=device,
                max_speakers=max_speakers,
                progress_cb=_diar_progress,
                status_cb=on_status,
            )

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
                segments=final_segments,
                cancelled=True,
                duration_seconds=duration,
                transcription_seconds=transcription_seconds,
                diarization_seconds=diarization_seconds,
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
        segments=final_segments,
        cancelled=False,
        duration_seconds=duration,
        transcription_seconds=transcription_seconds,
        diarization_seconds=diarization_seconds,
    )


def transcribe_media_file(
    media_path: str,
    model_name: str,
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    cancel_event: Event | None = None,
    use_diarization: bool = False,
    diar_backend: str = "accurate",
    max_speakers: int | None = None,
    on_status: StatusCallback | None = None,
    on_text: TextCallback | None = None,
    on_progress: ProgressCallback | None = None,
    on_diar_progress: ProgressCallback | None = None,
    on_model_download_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """
    End-to-end transcription for media input files used by listener mode.
    """
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
        return transcribe_prepared_audio(
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
