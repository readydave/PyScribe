"""Qt live-transcription support helpers and session orchestration."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import multiprocessing as mp
import os
import queue
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.functional as F
from PySide6.QtCore import QCoreApplication
from PySide6.QtMultimedia import QAudioFormat, QMediaDevices

from services.model_service import load_model, resolve_transcription_model


LOGGER = logging.getLogger(__name__)
LIVE_SAMPLE_RATE = 16_000
LIVE_SAMPLE_WIDTH_BYTES = 2
LIVE_CHANNELS = 1
LIVE_DECODE_HOP_SECONDS = 3.0
LIVE_DECODE_WINDOW_SECONDS = 12.0
LIVE_STABILIZATION_TAIL_SECONDS = 2.0
LIVE_SESSION_ROOT = Path.home() / "PyScribe Live Sessions"
_LOOPBACK_MARKERS = ("monitor", "loopback", "stereo mix", "what u hear", "monitor of")


@dataclass(frozen=True)
class LiveAudioDevice:
    id: str
    name: str
    kind: str
    available: bool


@dataclass(frozen=True)
class LiveSessionOptions:
    model_name: str
    device: str
    compute_type: str
    language: str | None
    source_mode: str
    input_device_id: str | None
    input_device_name: str
    output_root: str
    keep_audio_on_success: bool
    use_diarization: bool
    diar_backend: str
    max_speakers: int | None
    session_title: str | None = None


@dataclass
class LiveSessionMetadata:
    session_id: str
    started_at: str
    updated_at: str
    selected_model: str
    selected_device: str
    selected_device_name: str
    source_mode: str
    status: str
    saved_audio_path: str
    session_title: str | None = None
    final_transcript_path: str | None = None
    error_text: str | None = None


@dataclass
class LiveSessionSnapshot:
    transcript: str
    transcript_only: str
    segments: list[dict[str, Any]]
    recording_seconds: float
    session_dir: str
    capture_path: str
    metadata_path: str
    final_transcript_path: str
    session_title: str | None = None


@dataclass
class _DecodeRequest:
    request_id: int
    window_start_seconds: float
    window_end_seconds: float
    audio_np: np.ndarray
    stabilization_tail_seconds: float
    final: bool = False


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _device_id_text(raw: object) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", errors="ignore")
    return str(raw or "")


def classify_live_audio_device(name: str) -> str:
    lowered = str(name or "").strip().lower()
    if any(marker in lowered for marker in _LOOPBACK_MARKERS):
        return "loopback"
    return "microphone"


def normalize_session_title(title: str | None) -> str | None:
    """Normalizes a title into a filesystem-safe format."""
    if not title:
        return None
    import re
    # Trim and replace spaces with hyphens
    s = str(title).strip().replace(" ", "-")
    # Remove filesystem-invalid characters
    s = re.sub(r"[^a-zA-Z0-9\-_]", "", s)
    # Collapse repeated separators
    s = re.sub(r"-+", "-", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("-").strip("_") or None


def list_live_audio_inputs() -> list[LiveAudioDevice]:
    """Enumerates Qt audio input devices for the live-transcription UI."""
    if QCoreApplication.instance() is None:
        raise RuntimeError("Qt audio device enumeration requires an active QCoreApplication.")
    devices: list[LiveAudioDevice] = []
    for device in QMediaDevices.audioInputs():
        device_id = _device_id_text(bytes(device.id()))
        name = str(device.description() or device_id or "Audio Input")
        devices.append(
            LiveAudioDevice(
                id=device_id,
                name=name,
                kind=classify_live_audio_device(name),
                available=True,
            )
        )
    return devices


def live_model_supported(model_name: str) -> bool:
    return resolve_transcription_model(model_name).supports_timestamps


def default_live_output_dir() -> str:
    return str(LIVE_SESSION_ROOT)


def choose_live_audio_devices(
    devices: list[LiveAudioDevice],
    *,
    source_mode: str,
) -> list[LiveAudioDevice]:
    wanted = "loopback" if str(source_mode).strip().lower() == "loopback" else "microphone"
    return [device for device in devices if device.available and device.kind == wanted]


def audio_format_to_dict(fmt: QAudioFormat) -> dict[str, Any]:
    return {
        "sample_rate": int(fmt.sampleRate() or 0),
        "channel_count": int(fmt.channelCount() or 0),
        "sample_format": str(fmt.sampleFormat().name if hasattr(fmt.sampleFormat(), "name") else fmt.sampleFormat()),
    }


def build_live_capture_format(device: object) -> QAudioFormat:
    """Requests the normalized live format, then falls back to the device preferred format."""
    preferred = device.preferredFormat()
    target = QAudioFormat()
    target.setSampleRate(LIVE_SAMPLE_RATE)
    target.setChannelCount(LIVE_CHANNELS)
    target.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    try:
        if device.isFormatSupported(target):
            return target
    except Exception:
        pass
    return preferred


def _sample_format_name(sample_format: object) -> str:
    return getattr(sample_format, "name", str(sample_format))


def normalize_live_pcm_chunk(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channel_count: int,
    sample_format: object,
) -> tuple[np.ndarray, bytes]:
    """Converts a captured chunk to 16k mono float32 + PCM16 bytes."""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32), b""
    if sample_rate <= 0 or channel_count <= 0:
        raise ValueError("Audio format is missing sample rate or channel count.")

    fmt_name = _sample_format_name(sample_format)
    if fmt_name == "UInt8":
        raw = np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32)
        audio_np = (raw - 128.0) / 128.0
    elif fmt_name == "Int16":
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif fmt_name == "Int32":
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif fmt_name == "Float":
        audio_np = np.frombuffer(pcm_bytes, dtype=np.float32).astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio sample format: {fmt_name}")

    if channel_count > 1:
        usable = (audio_np.size // channel_count) * channel_count
        audio_np = audio_np[:usable].reshape(-1, channel_count).mean(axis=1)

    if sample_rate != LIVE_SAMPLE_RATE and audio_np.size:
        tensor = torch.from_numpy(audio_np)
        tensor = F.resample(tensor, sample_rate, LIVE_SAMPLE_RATE)
        audio_np = tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    clipped = np.clip(audio_np, -1.0, 1.0).astype(np.float32, copy=False)
    pcm16 = (clipped * 32767.0).astype(np.int16).tobytes()
    return clipped, pcm16


def reconcile_live_transcript(
    committed_segments: list[dict[str, Any]],
    new_segments: list[dict[str, Any]],
    *,
    window_start_seconds: float,
    window_end_seconds: float,
    stabilization_tail_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replaces the rolling live window while keeping stable older text."""
    commit_before = max(window_start_seconds, window_end_seconds - max(stabilization_tail_seconds, 0.0))
    stable: list[dict[str, Any]] = []
    draft: list[dict[str, Any]] = []
    for segment in new_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        item = {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": text,
        }
        if item["end"] <= commit_before:
            stable.append(item)
        else:
            draft.append(item)

    preserved = [seg for seg in committed_segments if float(seg.get("end", 0.0)) <= window_start_seconds]
    merged = preserved + stable
    return merged, draft


def render_live_transcript(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip()).strip()


class LiveSessionController:
    """Owns session storage, rolling ASR, and transcript reconciliation."""

    def __init__(self, options: LiveSessionOptions) -> None:
        if not live_model_supported(options.model_name):
            raise ValueError("Selected model does not support live transcription.")

        output_root = Path(options.output_root or default_live_output_dir()).expanduser()
        timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe_title = normalize_session_title(options.session_title)
        
        # Use timestamp + title for session directory name if title exists
        dir_name = timestamp
        if safe_title:
            dir_name = f"{timestamp}-{safe_title}"
        
        session_id = dir_name + "_" + uuid.uuid4().hex[:8]
        self.options = options
        self.session_dir = output_root / session_id
        self.capture_path = self.session_dir / "capture.wav"
        self.metadata_path = self.session_dir / "session.json"
        
        # Final transcript path will be calculated during finalization
        self.final_transcript_path = self.session_dir / "final_transcript.txt"
        
        self.metadata = LiveSessionMetadata(
            session_id=session_id,
            started_at=_now_iso(),
            updated_at=_now_iso(),
            selected_model=options.model_name,
            selected_device=options.input_device_id or "",
            selected_device_name=options.input_device_name,
            source_mode=options.source_mode,
            status="capturing",
            saved_audio_path=str(self.capture_path),
            session_title=options.session_title,
        )
        self._wave_file: wave.Wave_write | None = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_sample = 0
        self._total_samples = 0
        self._last_decode_target_sample = 0
        self._next_request_id = 1
        self._decode_in_flight = False
        self._awaiting_final_decode = False
        self._process: mp.Process | None = None
        self._request_queue: mp.Queue | None = None
        self._event_queue: mp.Queue | None = None
        self.committed_segments: list[dict[str, Any]] = []
        self.draft_segments: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self._last_emitted_transcript = ""
        self._started = False

    def update_title(self, title: str | None) -> None:
        """Updates the session title in metadata. Does not rename directories or files while active."""
        self.metadata.session_title = title
        self._write_metadata()

    @property
    def recording_seconds(self) -> float:
        return self._total_samples / float(LIVE_SAMPLE_RATE)

    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._wave_file = wave.open(str(self.capture_path), "wb")
        self._wave_file.setnchannels(LIVE_CHANNELS)
        self._wave_file.setsampwidth(LIVE_SAMPLE_WIDTH_BYTES)
        self._wave_file.setframerate(LIVE_SAMPLE_RATE)
        self._write_metadata()
        self._start_asr_process()
        self._started = True

    def snapshot(self) -> LiveSessionSnapshot:
        segments = [dict(seg) for seg in (self.committed_segments + self.draft_segments)]
        return LiveSessionSnapshot(
            transcript=render_live_transcript(segments),
            transcript_only=render_live_transcript(segments),
            segments=segments,
            recording_seconds=self.recording_seconds,
            session_dir=str(self.session_dir),
            capture_path=str(self.capture_path),
            metadata_path=str(self.metadata_path),
            final_transcript_path=str(self.final_transcript_path),
            session_title=self.metadata.session_title,
        )

    def append_audio_chunk(self, audio_np: np.ndarray, pcm16_bytes: bytes) -> None:
        if not self._started:
            raise RuntimeError("Live session has not started.")
        if self._wave_file is None:
            raise RuntimeError("Live session capture file is unavailable.")
        if audio_np.size == 0:
            return

        self._wave_file.writeframes(pcm16_bytes)
        self._buffer = np.concatenate([self._buffer, audio_np.astype(np.float32, copy=False)])
        self._total_samples += int(audio_np.size)
        self._trim_buffer()
        self._maybe_queue_decode()

    def request_final_decode(self) -> None:
        if not self._started:
            return
        self._awaiting_final_decode = True
        self._maybe_queue_decode(force=True, final=True)

    def poll_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._event_queue is None:
            return events
        while True:
            try:
                evt = self._event_queue.get_nowait()
            except queue.Empty:
                break
            etype = str(evt.get("type", ""))
            if etype == "ready":
                events.append({"type": "status", "value": str(evt.get("value", ""))})
                continue
            if etype == "status":
                events.append({"type": "status", "value": str(evt.get("value", ""))})
                continue
            if etype == "error":
                self.last_error = str(evt.get("value", "Live ASR failed."))
                self.metadata.status = "failed"
                self.metadata.error_text = self.last_error
                self._write_metadata()
                self._decode_in_flight = False
                events.append({"type": "error", "value": self.last_error})
                continue
            if etype != "result":
                continue

            request_id = int(evt.get("request_id", 0))
            if request_id <= 0:
                continue
            self._decode_in_flight = False
            segments = list(evt.get("segments") or [])
            window_start = float(evt.get("window_start_seconds", 0.0))
            window_end = float(evt.get("window_end_seconds", 0.0))
            stabilization_tail = float(evt.get("stabilization_tail_seconds", LIVE_STABILIZATION_TAIL_SECONDS))
            final = bool(evt.get("final"))
            self.committed_segments, self.draft_segments = reconcile_live_transcript(
                self.committed_segments,
                segments,
                window_start_seconds=window_start,
                window_end_seconds=window_end,
                stabilization_tail_seconds=stabilization_tail,
            )
            snapshot = self.snapshot()
            transcript = snapshot.transcript
            self._last_emitted_transcript = transcript
            if final:
                self._awaiting_final_decode = False
            events.append(
                {
                    "type": "transcript",
                    "value": transcript,
                    "snapshot": snapshot,
                }
            )
            if final:
                events.append({"type": "status", "value": "Live capture finalized. Starting final post-pass..."})
            elif self._awaiting_final_decode:
                # If stop was requested while a rolling decode was already in flight,
                # force the terminal decode now instead of waiting for another hop.
                self._maybe_queue_decode(force=True, final=True)
            else:
                self._maybe_queue_decode()
        return events

    def is_idle(self) -> bool:
        if self._decode_in_flight:
            return False
        pending_samples = self._total_samples - self._last_decode_target_sample
        if self._awaiting_final_decode and self._total_samples > 0:
            return pending_samples <= 0
        return True

    def close_capture(self) -> None:
        if self._wave_file is not None:
            try:
                self._wave_file.close()
            finally:
                self._wave_file = None

    def mark_finalizing(self) -> None:
        self.metadata.status = "finalizing"
        self.metadata.error_text = None
        self._write_metadata()

    def finalize_success(self, transcript: str) -> None:
        safe_title = normalize_session_title(self.metadata.session_title)
        timestamp = self.metadata.session_id.split("_")[0]
        
        final_audio_path = self.capture_path
        final_txt_path = self.final_transcript_path
        
        if safe_title:
            base_name = f"{timestamp}-{safe_title}"
            candidate_audio = self.session_dir / f"{base_name}.wav"
            candidate_txt = self.session_dir / f"{base_name}.txt"
            
            # Simple collision avoidance if needed
            counter = 1
            while candidate_audio.exists() or candidate_txt.exists():
                counter += 1
                candidate_audio = self.session_dir / f"{base_name}-{counter}.wav"
                candidate_txt = self.session_dir / f"{base_name}-{counter}.txt"
            
            final_audio_path = candidate_audio
            final_txt_path = candidate_txt
            
            if self.capture_path.exists():
                try:
                    self.capture_path.rename(final_audio_path)
                    self.metadata.saved_audio_path = str(final_audio_path)
                except OSError:
                    LOGGER.warning("Failed to rename capture.wav to %s", final_audio_path, exc_info=True)
        
        final_txt_path.write_text(transcript or "", encoding="utf-8")
        self.metadata.status = "completed"
        self.metadata.final_transcript_path = str(final_txt_path)
        self.metadata.error_text = None
        self._write_metadata()
        
        if not self.options.keep_audio_on_success and final_audio_path.exists():
            try:
                final_audio_path.unlink()
            except OSError:
                LOGGER.warning("Failed to delete live capture after success: %s", final_audio_path, exc_info=True)

    def finalize_cancelled(self) -> None:
        self.metadata.status = "cancelled"
        self._write_metadata()

    def finalize_failed(self, error_text: str) -> None:
        self.metadata.status = "failed"
        self.metadata.error_text = error_text
        self._write_metadata()

    def shutdown(self, *, preserve_error: bool = False) -> None:
        self.close_capture()
        if self._request_queue is not None:
            try:
                self._request_queue.put({"type": "shutdown"})
            except Exception:
                pass
        self._stop_process()
        if self._request_queue is not None:
            try:
                self._request_queue.close()
                self._request_queue.join_thread()
            except Exception:
                pass
        if self._event_queue is not None:
            try:
                self._event_queue.close()
                self._event_queue.join_thread()
            except Exception:
                pass
        self._request_queue = None
        self._event_queue = None
        self._process = None
        if preserve_error and self.last_error and self.metadata.status != "failed":
            self.finalize_failed(self.last_error)

    def _maybe_queue_decode(self, *, force: bool = False, final: bool = False) -> None:
        if self._request_queue is None or self._decode_in_flight:
            return
        total_samples = self._total_samples
        if total_samples <= 0:
            return
        hop_samples = int(LIVE_DECODE_HOP_SECONDS * LIVE_SAMPLE_RATE)
        pending_samples = total_samples - self._last_decode_target_sample
        if not force and pending_samples < hop_samples:
            return

        window_samples = min(total_samples, int(LIVE_DECODE_WINDOW_SECONDS * LIVE_SAMPLE_RATE))
        window_end_sample = total_samples
        window_start_sample = max(0, window_end_sample - window_samples)
        local_start = max(0, window_start_sample - self._buffer_start_sample)
        local_end = local_start + window_samples
        audio_np = np.array(self._buffer[local_start:local_end], dtype=np.float32, copy=True)
        if audio_np.size == 0:
            return
        request = _DecodeRequest(
            request_id=self._next_request_id,
            window_start_seconds=window_start_sample / float(LIVE_SAMPLE_RATE),
            window_end_seconds=window_end_sample / float(LIVE_SAMPLE_RATE),
            audio_np=audio_np,
            stabilization_tail_seconds=0.0 if final else LIVE_STABILIZATION_TAIL_SECONDS,
            final=final,
        )
        self._next_request_id += 1
        self._last_decode_target_sample = window_end_sample
        self._decode_in_flight = True
        self._request_queue.put(
            {
                "type": "decode",
                "request_id": request.request_id,
                "window_start_seconds": request.window_start_seconds,
                "window_end_seconds": request.window_end_seconds,
                "audio_np": request.audio_np,
                "language": self.options.language,
                "stabilization_tail_seconds": request.stabilization_tail_seconds,
                "final": request.final,
            }
        )

    def _trim_buffer(self) -> None:
        max_keep_samples = int((LIVE_DECODE_WINDOW_SECONDS + LIVE_DECODE_HOP_SECONDS + 2.0) * LIVE_SAMPLE_RATE)
        if self._buffer.size <= max_keep_samples:
            return
        drop = int(self._buffer.size - max_keep_samples)
        self._buffer = self._buffer[drop:]
        self._buffer_start_sample += drop

    def _write_metadata(self) -> None:
        self.metadata.updated_at = _now_iso()
        payload = asdict(self.metadata)
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _start_asr_process(self) -> None:
        ctx = mp.get_context("spawn")
        self._request_queue = ctx.Queue()
        self._event_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_live_asr_process_entry,
            args=(self.options.model_name, self.options.device, self.options.compute_type, self._request_queue, self._event_queue),
        )
        self._process.start()

    def _stop_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            if not proc.is_alive():
                proc.join(timeout=1.0)
                return
        except Exception:
            return
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
        deadline = time.perf_counter() + 0.75
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


def _live_asr_process_entry(
    model_name: str,
    device: str,
    compute_type: str,
    request_queue: mp.Queue,
    event_queue: mp.Queue,
) -> None:
    try:
        spec = resolve_transcription_model(model_name)
        if spec.backend_kind != "faster_whisper" or not spec.supports_timestamps:
            raise RuntimeError(f"Model '{spec.display_name}' does not support live transcription.")
        model = load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            use_cache=False,
            model_spec=spec,
        )
        event_queue.put({"type": "ready", "value": f"Live ASR worker ready on {device.upper()}."})
        while True:
            request = request_queue.get()
            if not isinstance(request, dict):
                continue
            rtype = str(request.get("type", "")).strip().lower()
            if rtype == "shutdown":
                break
            if rtype != "decode":
                continue
            audio_np = np.array(request.get("audio_np"), dtype=np.float32, copy=False)
            window_start_seconds = float(request.get("window_start_seconds", 0.0))
            window_end_seconds = float(request.get("window_end_seconds", 0.0))
            language = request.get("language")
            request_id = int(request.get("request_id", 0))
            stabilization_tail_seconds = float(
                request.get("stabilization_tail_seconds", LIVE_STABILIZATION_TAIL_SECONDS)
            )
            final = bool(request.get("final"))

            segments_gen, _ = model.transcribe(audio_np, task="transcribe", language=language, beam_size=5)
            segments = []
            for segment in segments_gen:
                text = str(getattr(segment, "text", "") or "").strip()
                if not text:
                    continue
                segments.append(
                    {
                        "start": float(getattr(segment, "start", 0.0)) + window_start_seconds,
                        "end": float(getattr(segment, "end", 0.0)) + window_start_seconds,
                        "text": text,
                    }
                )
            event_queue.put(
                {
                    "type": "result",
                    "request_id": request_id,
                    "segments": segments,
                    "window_start_seconds": window_start_seconds,
                    "window_end_seconds": window_end_seconds,
                    "stabilization_tail_seconds": stabilization_tail_seconds,
                    "final": final,
                }
            )
    except Exception as exc:
        event_queue.put({"type": "error", "value": str(exc)})
