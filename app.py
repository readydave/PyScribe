# app.py
# Gradio-based listener UI for PyScribe.

import argparse
from collections.abc import Iterator
import datetime
import logging
import os
import socket
import sys
import tempfile
import threading
from typing import Any, Callable

import gradio as gr
import pyperclip
from services.listener_security_service import (
    reject_legacy_auth_pass_flag,
    resolve_listener_auth,
    validate_listener_security,
)
import services as pyscribe_services
from services.runtime_compat import ensure_platform_sys_version_compat
from services.runtime_env_service import (
    configure_runtime_environment,
    reexec_if_loader_env_changed,
)
LOGGER = logging.getLogger(__name__)

_LISTENER_RUNTIME_READY = False
RUNTIME = None
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
RECOMMENDED_MODEL = "small"
ALL_MODELS: list[str] = []
AVAILABLE_DIAR_BACKENDS: list[str] = []
APP_CONFIG = pyscribe_services.AppConfig()


def _ensure_listener_runtime() -> None:
    global _LISTENER_RUNTIME_READY
    global RUNTIME, DEVICE, COMPUTE_TYPE, RECOMMENDED_MODEL
    global ALL_MODELS, AVAILABLE_DIAR_BACKENDS, APP_CONFIG
    if _LISTENER_RUNTIME_READY:
        return
    ensure_platform_sys_version_compat()
    configure_runtime_environment()
    reexec_if_loader_env_changed()
    RUNTIME = pyscribe_services.detect_runtime()
    DEVICE = RUNTIME.device
    COMPUTE_TYPE = RUNTIME.compute_type
    RECOMMENDED_MODEL = pyscribe_services.recommend_model(RUNTIME)
    ALL_MODELS = pyscribe_services.get_model_choices()
    AVAILABLE_DIAR_BACKENDS = pyscribe_services.get_available_diarization_backends(include_off=False)
    if not AVAILABLE_DIAR_BACKENDS:
        AVAILABLE_DIAR_BACKENDS = ["accurate"]
    APP_CONFIG = pyscribe_services.load_config()
    _LISTENER_RUNTIME_READY = True

CUSTOM_CSS = """
html.pyscribe-prog-red progress,
html.pyscribe-prog-red [role="progressbar"],
html.pyscribe-prog-red .progress-bar,
html.pyscribe-prog-red .progress-bar-wrap > div {
  accent-color: #dc2626 !important;
  background-color: #dc2626 !important;
}
html.pyscribe-prog-orange progress,
html.pyscribe-prog-orange [role="progressbar"],
html.pyscribe-prog-orange .progress-bar,
html.pyscribe-prog-orange .progress-bar-wrap > div {
  accent-color: #f97316 !important;
  background-color: #f97316 !important;
}
html.pyscribe-prog-yellow progress,
html.pyscribe-prog-yellow [role="progressbar"],
html.pyscribe-prog-yellow .progress-bar,
html.pyscribe-prog-yellow .progress-bar-wrap > div {
  accent-color: #facc15 !important;
  background-color: #facc15 !important;
}
html.pyscribe-prog-blue progress,
html.pyscribe-prog-blue [role="progressbar"],
html.pyscribe-prog-blue .progress-bar,
html.pyscribe-prog-blue .progress-bar-wrap > div {
  accent-color: #2563eb !important;
  background-color: #2563eb !important;
}
html.pyscribe-prog-green progress,
html.pyscribe-prog-green [role="progressbar"],
html.pyscribe-prog-green .progress-bar,
html.pyscribe-prog-green .progress-bar-wrap > div {
  accent-color: #16a34a !important;
  background-color: #16a34a !important;
}
"""

CUSTOM_HEAD = """
<script>
(function () {
  const classes = [
    "pyscribe-prog-red",
    "pyscribe-prog-orange",
    "pyscribe-prog-yellow",
    "pyscribe-prog-blue",
    "pyscribe-prog-green",
  ];

  function pickClass(pct) {
    if (pct >= 100) return "pyscribe-prog-green";
    if (pct >= 76) return "pyscribe-prog-blue";
    if (pct >= 51) return "pyscribe-prog-yellow";
    if (pct >= 26) return "pyscribe-prog-orange";
    return "pyscribe-prog-red";
  }

  function extractPct() {
    // Preferred: aria value from rendered progress bar.
    const pb = document.querySelector('[role="progressbar"][aria-valuenow]');
    if (pb) {
      const v = Number(pb.getAttribute("aria-valuenow"));
      if (!Number.isNaN(v)) return v;
    }
    // Fallback: parse "Transcribing xx%" text.
    const candidates = document.querySelectorAll("div,span,p");
    for (const el of candidates) {
      const txt = (el.textContent || "").trim();
      const m = txt.match(/Transcribing\\s+(\\d+)%/i);
      if (m) return Number(m[1]);
    }
    return null;
  }

  function tick() {
    const pct = extractPct();
    if (pct === null) return;
    const root = document.documentElement;
    for (const c of classes) root.classList.remove(c);
    root.classList.add(pickClass(pct));
  }

  setInterval(tick, 250);
})();
</script>
"""

# --- State for Cancellation ---
_cancel_event = threading.Event()
_transcription_active = threading.Event()
GradioUpdate = dict[str, object]
TranscribeYield = tuple[str, str, GradioUpdate, GradioUpdate, str]


def _progress_badge(pct: float) -> str:
    if pct >= 100:
        return "🟢"
    if pct >= 76:
        return "🔵"
    if pct >= 51:
        return "🟡"
    if pct >= 26:
        return "🟠"
    return "🔴"


def _normalize_run_mode(value: str) -> str:
    mode = str(value or "full").strip().lower()
    if mode in {"full", "transcribe_only", "visual_only"}:
        return mode
    return "full"


def _reload_listener_config() -> None:
    global APP_CONFIG
    APP_CONFIG = pyscribe_services.load_config()


def _enabled_llm_profiles() -> list[pyscribe_services.LLMConnectionProfile]:
    _reload_listener_config()
    return pyscribe_services.get_enabled_llm_profiles(APP_CONFIG.llm_profiles)


def _find_enabled_llm_profile(profile_name: str) -> pyscribe_services.LLMConnectionProfile | None:
    target = str(profile_name or "").strip()
    if not target:
        return None
    for profile in _enabled_llm_profiles():
        if profile.name == target:
            return profile
    return None


def _llm_profile_name_choices() -> list[str]:
    return [profile.name for profile in _enabled_llm_profiles()]


def _llm_template_choices() -> tuple[list[tuple[str, str]], str | None]:
    templates, fallback_default_id = pyscribe_services.load_prompt_templates()
    choices = [(template.name, template.id) for template in templates]
    default_id = str(APP_CONFIG.llm_default_template_id or "").strip().lower()
    if default_id and any(template_id == default_id for _, template_id in choices):
        return choices, default_id
    if fallback_default_id and any(template_id == fallback_default_id for _, template_id in choices):
        return choices, fallback_default_id
    if choices:
        return choices, choices[0][1]
    return choices, None


def _read_uploaded_text(file_obj: Any) -> str:
    if file_obj is None:
        return ""
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError as exc:
        raise gr.Error(f"Could not read file '{path}': {exc}")


def _normalize_uploaded_image_paths(file_obj: Any) -> tuple[str, ...]:
    if file_obj is None:
        return ()
    candidates: list[str] = []
    if isinstance(file_obj, list):
        raw_items = file_obj
    else:
        raw_items = [file_obj]
    for item in raw_items:
        path = item.name if hasattr(item, "name") else str(item)
        normalized = str(path or "").strip()
        if not normalized or not os.path.isfile(normalized):
            continue
        if normalized not in candidates:
            candidates.append(normalized)
    return tuple(candidates)


def _update_postprocess_source_fields(source_mode: str) -> tuple[GradioUpdate, GradioUpdate]:
    use_current = str(source_mode or "").strip().lower() == "current transcript"
    return gr.update(visible=not use_current), gr.update(visible=not use_current)


def test_listener_llm_connection(profile_name: str, model_override: str) -> tuple[str, GradioUpdate]:
    profile = _find_enabled_llm_profile(profile_name)
    if profile is None:
        return "Connection test: no enabled profile selected.", gr.update()

    result = pyscribe_services.test_connection(profile)
    lines: list[str] = [f"Overall: {result.status.upper()}"]
    if result.selected_model:
        lines.append(f"Selected model: {result.selected_model}")
    if result.loaded_model:
        lines.append(f"Loaded model: {result.loaded_model}")
    lines.append("Stages:")
    for stage in result.stages:
        lines.append(f"- [{stage.status.upper()}] {stage.stage}: {stage.detail}")
        for suggestion in stage.suggestions:
            lines.append(f"  * {suggestion}")
    if result.failure_code:
        lines.append(f"Failure: {result.failure_code} - {result.failure_detail}")

    model_choices = list(result.detected_models)
    preferred_model = (
        str(model_override or "").strip()
        or result.selected_model
        or profile.default_model
        or ""
    )
    if preferred_model and preferred_model not in model_choices:
        model_choices.append(preferred_model)

    return "\n".join(lines), gr.update(choices=model_choices, value=preferred_model)


def run_listener_llm_postprocess(
    current_transcript: str,
    source_mode: str,
    uploaded_transcript_file: Any,
    pasted_transcript: str,
    uploaded_ocr_file: Any,
    uploaded_images: Any,
    include_images: bool,
    image_ocr_backend: str,
    ocr_fallback_for_images: bool,
    notes_text: str,
    extra_context_text: str,
    payload_preview_text: str,
    profile_name: str,
    template_id: str,
    selected_model: str,
) -> tuple[str, str]:
    profile = _find_enabled_llm_profile(profile_name)
    if profile is None:
        return "Post-process failed: no enabled profile selected.", ""

    if _transcription_active.is_set():
        if profile.scope == "local":
            return (
                "Post-process blocked: local profile cannot run while local transcription is active.",
                "",
            )
        if not profile.allow_concurrent_with_local_transcription:
            return (
                "Post-process blocked: this profile is not marked concurrent-safe during local transcription.",
                "",
            )

    transcript_text, ocr_text = _resolve_listener_postprocess_context(
        current_transcript=current_transcript,
        source_mode=source_mode,
        uploaded_transcript_file=uploaded_transcript_file,
        pasted_transcript=pasted_transcript,
        uploaded_ocr_file=uploaded_ocr_file,
    )
    if not transcript_text:
        return "Post-process failed: transcript text is required (current or uploaded/pasted).", ""

    template = pyscribe_services.get_prompt_template(str(template_id or "").strip().lower())
    if template is None:
        return "Post-process failed: selected prompt template is unavailable.", ""
    image_paths = _normalize_uploaded_image_paths(uploaded_images)

    request = pyscribe_services.LLMPostprocessRequest(
        transcript_text=transcript_text,
        ocr_text=ocr_text,
        notes_text=str(notes_text or "").strip(),
        selected_model=str(selected_model or "").strip() or None,
        extra_context_text=str(extra_context_text or "").strip(),
        image_paths=image_paths,
        include_images=bool(include_images),
        image_ocr_backend=str(image_ocr_backend or "auto").strip().lower() or "auto",
        ocr_fallback_for_images=bool(ocr_fallback_for_images),
    )
    prepared = pyscribe_services.prepare_llm_postprocess_payload(profile, template, request)
    if prepared.status != "pass":
        return (
            f"Post-process failed: {prepared.error_code or 'error'} - {prepared.error_detail or 'Unknown error'}",
            "",
        )
    if APP_CONFIG.llm_payload_preview_required and not str(payload_preview_text or "").strip():
        return "Post-process blocked: payload preview is required. Click 'Preview Payload' first.", ""
    result = pyscribe_services.run_llm_postprocess(profile, template, request, prepared_payload=prepared)
    if result.status != "pass":
        return (
            f"Post-process failed: {result.error_code or 'error'} - {result.error_detail or 'Unknown error'}",
            "",
        )

    used_model = result.model or profile.default_model or "unknown"
    suffix = f" {result.info_note}" if result.info_note else ""
    return f"Post-process complete ({profile.name}, model: {used_model}).{suffix}", result.output_text


def preview_listener_llm_payload(
    current_transcript: str,
    source_mode: str,
    uploaded_transcript_file: Any,
    pasted_transcript: str,
    uploaded_ocr_file: Any,
    uploaded_images: Any,
    include_images: bool,
    image_ocr_backend: str,
    ocr_fallback_for_images: bool,
    notes_text: str,
    extra_context_text: str,
    template_id: str,
    profile_name: str,
    selected_model: str,
) -> tuple[str, str]:
    profile = _find_enabled_llm_profile(profile_name)
    if profile is None:
        return "Payload preview failed: no enabled profile selected.", ""
    transcript_text, ocr_text = _resolve_listener_postprocess_context(
        current_transcript=current_transcript,
        source_mode=source_mode,
        uploaded_transcript_file=uploaded_transcript_file,
        pasted_transcript=pasted_transcript,
        uploaded_ocr_file=uploaded_ocr_file,
    )
    if not transcript_text:
        return "Payload preview failed: transcript text is required.", ""
    template = pyscribe_services.get_prompt_template(str(template_id or "").strip().lower())
    if template is None:
        return "Payload preview failed: selected prompt template is unavailable.", ""
    image_paths = _normalize_uploaded_image_paths(uploaded_images)
    request = pyscribe_services.LLMPostprocessRequest(
        transcript_text=transcript_text,
        ocr_text=ocr_text,
        notes_text=str(notes_text or "").strip(),
        selected_model=str(selected_model or "").strip() or None,
        extra_context_text=str(extra_context_text or "").strip(),
        image_paths=image_paths,
        include_images=bool(include_images),
        image_ocr_backend=str(image_ocr_backend or "auto").strip().lower() or "auto",
        ocr_fallback_for_images=bool(ocr_fallback_for_images),
    )
    prepared = pyscribe_services.prepare_llm_postprocess_payload(profile, template, request)
    if prepared.status != "pass":
        return (
            f"Payload preview failed: {prepared.error_code or 'error'} - {prepared.error_detail or 'Unknown error'}",
            "",
        )
    status = "Payload preview ready."
    if prepared.info_note:
        status = f"{status} {prepared.info_note}"
    return status, prepared.payload_text


def _resolve_listener_postprocess_context(
    *,
    current_transcript: str,
    source_mode: str,
    uploaded_transcript_file: Any,
    pasted_transcript: str,
    uploaded_ocr_file: Any,
) -> tuple[str, str]:
    mode = str(source_mode or "").strip().lower()
    transcript_text = ""
    if mode == "current transcript":
        transcript_text = str(current_transcript or "").strip()
    else:
        transcript_text = _read_uploaded_text(uploaded_transcript_file) or str(pasted_transcript or "").strip()
        if not transcript_text:
            transcript_text = str(current_transcript or "").strip()
    ocr_text = _read_uploaded_text(uploaded_ocr_file)
    return transcript_text, ocr_text


def find_open_port(host: str, preferred_port: int, max_tries: int = 50) -> int:
    """
    Returns the preferred port if available, otherwise the first open port in range.
    """
    ports = [preferred_port] + list(range(preferred_port + 1, preferred_port + 1 + max_tries))
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No open port found between {preferred_port} and {preferred_port + max_tries}."
    )

def transcribe(
    audio_path: Any,
    model_name: str,
    run_mode: str,
    use_diarization: bool,
    diar_backend: str,
    max_speakers_text: str,
    use_visual_analysis: bool,
    visual_profile: str,
    visual_ocr_backend: str,
    visual_sample_seconds: float,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[TranscribeYield]:
    """
    The main transcription function for the Gradio interface.
    """
    _ensure_listener_runtime()
    _cancel_event.clear()

    if audio_path is None:
        yield "Status: No audio file provided.", "", gr.update(visible=True), gr.update(visible=False), ""
        return

    media_path = audio_path.name if hasattr(audio_path, "name") else str(audio_path)
    run_mode = _normalize_run_mode(run_mode)
    should_transcribe = run_mode in {"full", "transcribe_only"}
    should_run_visual = run_mode in {"full", "visual_only"}

    model_name = pyscribe_services.normalize_model_name(model_name)
    if should_transcribe and not model_name:
        yield "Status: No model selected.", "", gr.update(visible=True), gr.update(visible=False), ""
        return

    model_spec = pyscribe_services.resolve_transcription_model(model_name) if should_transcribe else None
    max_speakers = int(max_speakers_text) if str(max_speakers_text).strip().isdigit() else None
    use_diarization = bool(use_diarization and should_transcribe)
    if use_diarization and model_spec is not None and not model_spec.supports_diarization:
        use_diarization = False
        yield (
            f"Status: Speaker identification is unavailable for '{model_spec.display_name}'. "
            "Continuing without diarization.",
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
        )
    if not use_diarization:
        diar_backend = "off"
    use_visual_analysis = bool(use_visual_analysis and should_run_visual)
    if run_mode == "visual_only" and not use_visual_analysis:
        yield (
            "Status: Visual-only mode requires enabling 'Analyze visuals'.",
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
        )
        return
    visual_profile = str(visual_profile or "balanced").lower()
    visual_ocr_backend = str(visual_ocr_backend or "auto").lower()
    if use_visual_analysis:
        ready, reason = pyscribe_services.check_ocr_backend_ready(visual_ocr_backend)
        if not ready:
            fallback = ""
            for candidate in ("rapidocr", "paddleocr", "pytesseract", "surya"):
                fallback_ready, _ = pyscribe_services.check_ocr_backend_ready(candidate)
                if fallback_ready and candidate != visual_ocr_backend:
                    fallback = candidate
                    break
            if fallback:
                visual_ocr_backend = fallback
                yield (
                    f"Status: Visual backend unavailable ({reason}). Using '{fallback}' instead.",
                    "",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "",
                )
            else:
                use_visual_analysis = False
                yield (
                    f"Status: Visual backend '{visual_ocr_backend}' unavailable ({reason}). "
                    "Continuing transcription without visual analysis.",
                    "",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "",
                )
    try:
        global APP_CONFIG
        listener_cfg = pyscribe_services.load_config()
        listener_cfg.last_model = model_name
        listener_cfg.use_diarization = bool(use_diarization)
        listener_cfg.max_speakers = max_speakers
        listener_cfg.diar_backend = diar_backend
        listener_cfg.run_mode = run_mode
        listener_cfg.use_visual_analysis = bool(use_visual_analysis)
        listener_cfg.visual_profile = visual_profile
        listener_cfg.visual_ocr_backend = visual_ocr_backend
        listener_cfg.visual_sample_seconds = float(visual_sample_seconds or 1.0)
        pyscribe_services.save_config(listener_cfg)
        APP_CONFIG = listener_cfg
    except Exception as exc:
        LOGGER.warning("Failed to save listener config: %s", exc, exc_info=True)
        
    status_prefix = "Visual analysis" if run_mode == "visual_only" else "Transcription"
    yield f"Status: {status_prefix} starting...", "", gr.update(visible=False), gr.update(visible=True, value="Cancel"), ""
    _transcription_active.set()

    full_transcript = ""
    was_cancelled = False
    last_phase = "Transcribing"
    last_pct = 0.0
    try:
        def _on_status(msg: str) -> None:
            nonlocal last_phase
            last_phase = msg
            progress(None, desc=msg)

        def _on_progress(pct: float) -> None:
            nonlocal last_pct
            last_pct = min(max(pct, 0.0), 100.0)
            badge = _progress_badge(last_pct)
            step_label = "Analyzing visuals" if run_mode == "visual_only" else "Transcribing"
            desc = f"{badge} {step_label} {last_pct:.0f}%"
            progress(min(max(last_pct / 100.0, 0.0), 1.0), desc=desc)

        def _on_model_download_progress(pct: float) -> None:
            badge = _progress_badge(min(max(pct, 0.0), 100.0))
            progress(min(max(pct / 100.0, 0.0), 1.0), desc=f"{badge} Downloading model files...")

        def _on_text(text: str) -> None:
            nonlocal full_transcript
            full_transcript = text

        result = pyscribe_services.transcribe_media_file(
            media_path=media_path,
            model_name=model_name,
            run_mode=run_mode,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=None,
            cancel_event=_cancel_event,
            use_diarization=use_diarization,
            diar_backend=diar_backend,
            max_speakers=max_speakers,
            use_visual_analysis=use_visual_analysis,
            visual_profile=visual_profile,
            visual_ocr_backend=visual_ocr_backend,
            visual_sample_seconds=float(visual_sample_seconds or 1.0),
            on_status=_on_status,
            on_text=_on_text,
            on_progress=_on_progress,
            on_model_download_progress=_on_model_download_progress,
        )
        full_transcript = result.transcript
        was_cancelled = result.cancelled

    except Exception as e:
        # Use gr.Error to properly raise exceptions in the Gradio UI
        raise gr.Error(f"An error occurred: {e}")
    finally:
        _transcription_active.clear()

    if was_cancelled or _cancel_event.is_set():
        _cancel_event.clear()
        yield f"Status: {status_prefix} cancelled.", full_transcript.strip(), gr.update(visible=True), gr.update(visible=False), "Cancelled."
        return

    final_badge = _progress_badge(100.0)
    final_done = "Visual analysis complete!" if run_mode == "visual_only" else "Transcription complete!"
    yield f"Status: {final_badge} {last_phase}", full_transcript.strip(), gr.update(visible=True), gr.update(visible=False), final_done

def save_transcript(transcript: str, audio_path: Any, model_name: str) -> str | None:
    if not transcript:
        gr.Info("Nothing to save.")
        return None

    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        if audio_path is not None:
            source_path = audio_path.name if hasattr(audio_path, "name") else str(audio_path)
            base_name = os.path.splitext(os.path.basename(source_path))[0]
        else:
            base_name = "transcript"

        safe_model_name = model_name.replace("/", "-")
        
        # Create a temporary file to save the transcript
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, f"{ts}_{base_name}_{safe_model_name}.txt")

        with open(save_path, "w", encoding="utf-8") as transcript_file:
            transcript_file.write(transcript)
        
        gr.Info(f"Transcript ready for download.")
        return save_path

    except (OSError, AttributeError) as e:
        raise gr.Error(f"Could not save file: {e}")


def save_postprocess_output(postprocess_output: str, template_id: str) -> str | None:
    if not postprocess_output:
        gr.Info("Nothing to save.")
        return None
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe_template = str(template_id or "postprocess").strip().replace("/", "-")
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, f"postprocess_{safe_template}_{ts}.md")
        with open(save_path, "w", encoding="utf-8") as output_file:
            output_file.write(postprocess_output)
        gr.Info("Post-processed output ready for download.")
        return save_path
    except OSError as exc:
        raise gr.Error(f"Could not save file: {exc}")

def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)
    gr.Info("Copied to clipboard!")

def set_cancel_flag() -> None:
    _cancel_event.set()

# --- Gradio Interface Definition ---
def create_interface() -> gr.Blocks:
    _ensure_listener_runtime()
    initial_run_mode = _normalize_run_mode(APP_CONFIG.run_mode)
    initial_allow_transcription = initial_run_mode in {"full", "transcribe_only"}
    initial_allow_visual = initial_run_mode in {"full", "visual_only"}
    llm_profile_choices = _llm_profile_name_choices()
    if APP_CONFIG.llm_default_profile in llm_profile_choices:
        default_llm_profile = APP_CONFIG.llm_default_profile
    else:
        default_llm_profile = llm_profile_choices[0] if llm_profile_choices else None
    llm_template_choices, default_template_id = _llm_template_choices()

    def _update_mode_visibility(
        run_mode: str,
        use_diarization: bool,
        use_visual_analysis: bool,
    ) -> tuple[GradioUpdate, GradioUpdate, GradioUpdate, GradioUpdate, GradioUpdate, GradioUpdate, GradioUpdate, GradioUpdate]:
        mode = _normalize_run_mode(run_mode)
        allow_transcription = mode in {"full", "transcribe_only"}
        allow_visual = mode in {"full", "visual_only"}
        return (
            gr.update(visible=allow_transcription),  # model
            gr.update(visible=allow_transcription),  # diar checkbox
            gr.update(visible=allow_transcription and bool(use_diarization)),  # diar backend
            gr.update(visible=allow_transcription and bool(use_diarization)),  # max speakers
            gr.update(visible=allow_visual),  # visual checkbox
            gr.update(visible=allow_visual and bool(use_visual_analysis)),  # visual profile
            gr.update(visible=allow_visual and bool(use_visual_analysis)),  # visual backend
            gr.update(visible=allow_visual and bool(use_visual_analysis)),  # visual interval
        )

    def _update_diar_fields(use_diarization: bool, run_mode: str) -> tuple[GradioUpdate, GradioUpdate]:
        allow_diar = bool(use_diarization) and _normalize_run_mode(run_mode) in {"full", "transcribe_only"}
        return gr.update(visible=allow_diar), gr.update(visible=allow_diar)

    def _update_visual_fields(
        use_visual_analysis: bool,
        run_mode: str,
    ) -> tuple[GradioUpdate, GradioUpdate, GradioUpdate]:
        allow_visual_controls = bool(use_visual_analysis) and _normalize_run_mode(run_mode) in {"full", "visual_only"}
        return (
            gr.update(visible=allow_visual_controls),
            gr.update(visible=allow_visual_controls),
            gr.update(visible=allow_visual_controls),
        )

    with gr.Blocks(title="PyScribe Listener") as iface:
        gr.Markdown(
            """
            # PyScribe Listener
            Drop an audio/video file to transcribe on this host machine.
            """
        )
        gr.Markdown(
            """
            **Model tips:** use built-in choices or a custom Hugging Face repo ID (`owner/repo`).
            If not cached, PyScribe estimates size (best-effort), asks for confirmation, then downloads with progress.
            For private/gated repos, authenticate with an HF token and accept model terms on Hugging Face.
            Granite 4.0 Speech is available as an experimental backend and does not support speaker identification in PyScribe yet.
            Optional multimodal mode can OCR sampled video frames (slides/chat text) and append highlights to the output.
            """
        )
        gr.Markdown(
            """
            **LLM post-processing:** configure profiles in Qt (`Tools > LLM Connections...`) and then use
            the Listener's **LLM Post-Processing** section for connection tests and template-based summarization.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(
                    label="Upload Audio/Video File",
                    file_types=["audio", "video"]
                )

                model_dropdown = gr.Dropdown(
                    choices=ALL_MODELS,
                    value=APP_CONFIG.last_model if APP_CONFIG.last_model in ALL_MODELS else RECOMMENDED_MODEL,
                    label="Select Transcription Model",
                    info=f"Recommended for your hardware ({DEVICE.upper()}): {RECOMMENDED_MODEL}",
                    allow_custom_value=True,
                    visible=initial_allow_transcription,
                )
                run_mode_dropdown = gr.Dropdown(
                    choices=["full", "transcribe_only", "visual_only"],
                    value=initial_run_mode,
                    label="Run mode",
                    info="full = transcript + optional OCR, transcribe_only = transcript only, visual_only = OCR only",
                )
                diar_checkbox = gr.Checkbox(
                    label="Identify speakers",
                    value=APP_CONFIG.use_diarization,
                    visible=initial_allow_transcription,
                )
                default_backend = (
                    APP_CONFIG.diar_backend
                    if APP_CONFIG.diar_backend in AVAILABLE_DIAR_BACKENDS
                    else AVAILABLE_DIAR_BACKENDS[0]
                )
                diar_backend_dropdown = gr.Dropdown(
                    choices=AVAILABLE_DIAR_BACKENDS,
                    value=default_backend,
                    label="Diarization mode",
                    info=pyscribe_services.get_backend_label(default_backend),
                    visible=initial_allow_transcription and APP_CONFIG.use_diarization,
                )
                max_speakers_input = gr.Textbox(
                    label="Max speakers (optional)",
                    value="" if APP_CONFIG.max_speakers is None else str(APP_CONFIG.max_speakers),
                    placeholder="e.g. 2",
                    visible=initial_allow_transcription and APP_CONFIG.use_diarization,
                )
                visual_checkbox = gr.Checkbox(
                    label="Analyze visuals (slides/chat OCR, beta)",
                    value=APP_CONFIG.use_visual_analysis,
                    visible=initial_allow_visual,
                )
                visual_profile_dropdown = gr.Dropdown(
                    choices=["fast", "balanced", "accurate"],
                    value=str(APP_CONFIG.visual_profile or "balanced").lower(),
                    label="Visual mode",
                    info="fast = quickest, balanced = default, accurate = most thorough.",
                    visible=initial_allow_visual and APP_CONFIG.use_visual_analysis,
                )
                visual_backend_dropdown = gr.Dropdown(
                    choices=["auto", "rapidocr", "paddleocr", "surya", "pytesseract"],
                    value=str(APP_CONFIG.visual_ocr_backend or "auto").lower(),
                    label="Visual OCR backend",
                    info="auto picks the best available backend; paddleocr/surya/rapidocr may download models on first run.",
                    visible=initial_allow_visual and APP_CONFIG.use_visual_analysis,
                )
                visual_interval = gr.Slider(
                    minimum=0.5,
                    maximum=10.0,
                    step=0.5,
                    value=float(APP_CONFIG.visual_sample_seconds or 1.0),
                    label="Visual sample interval (seconds)",
                    info="Lower values capture more slide/chat changes but use more compute.",
                    visible=initial_allow_visual and APP_CONFIG.use_visual_analysis,
                )

                submit_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=2):
                status_output = gr.Textbox(label="Status", interactive=False)
                transcript_output = gr.Textbox(
                    label="Transcription",
                    interactive=True,
                    lines=15,
                    max_lines=15,
                )
                with gr.Row():
                    copy_btn = gr.Button("Copy to Clipboard")
                    save_btn = gr.Button("Save Transcript")
                download_file = gr.File(label="Download Transcript", interactive=False)
                final_status_output = gr.Textbox(label="", interactive=False)

                with gr.Accordion("LLM Post-Processing (Beta)", open=False):
                    llm_profile_dropdown = gr.Dropdown(
                        choices=llm_profile_choices,
                        value=default_llm_profile,
                        label="LLM profile",
                        info="Configure profiles in Qt Tools > LLM Connections...",
                    )
                    llm_template_dropdown = gr.Dropdown(
                        choices=llm_template_choices,
                        value=default_template_id,
                        label="Prompt template",
                    )
                    llm_model_dropdown = gr.Dropdown(
                        choices=[],
                        value="",
                        allow_custom_value=True,
                        label="LLM model override (optional)",
                    )
                    with gr.Row():
                        llm_test_btn = gr.Button("Test LLM Connection")
                        llm_preview_btn = gr.Button("Preview Payload")
                        llm_run_btn = gr.Button("Run LLM Post-Process", variant="primary")
                    llm_source_mode = gr.Radio(
                        choices=["Current transcript", "Upload/paste transcript"],
                        value="Current transcript",
                        label="Transcript source",
                    )
                    llm_transcript_file = gr.File(
                        label="Upload transcript text file (.txt/.md)",
                        file_types=[".txt", ".md"],
                        visible=False,
                    )
                    llm_transcript_paste = gr.Textbox(
                        label="Paste transcript text",
                        lines=8,
                        max_lines=12,
                        visible=False,
                    )
                    llm_include_images_checkbox = gr.Checkbox(
                        label="Include image attachments",
                        value=bool(APP_CONFIG.llm_include_images_default),
                    )
                    llm_image_files = gr.File(
                        label="Attach screenshots/images (optional)",
                        file_types=["image"],
                        file_count="multiple",
                    )
                    llm_image_ocr_backend = gr.Dropdown(
                        choices=["auto", "rapidocr", "paddleocr", "surya", "pytesseract"],
                        value="auto",
                        label="Image OCR backend (fallback)",
                    )
                    llm_image_ocr_fallback = gr.Checkbox(
                        label="Use OCR fallback when model is text-only",
                        value=bool(APP_CONFIG.llm_ocr_fallback_for_images_default),
                    )
                    llm_ocr_file = gr.File(
                        label="Upload OCR/context text file (optional)",
                        file_types=[".txt", ".md"],
                    )
                    llm_notes_input = gr.Textbox(
                        label="Additional notes/context (optional)",
                        lines=4,
                        max_lines=8,
                    )
                    llm_extra_context_input = gr.Textbox(
                        label="Pasted context (optional)",
                        lines=4,
                        max_lines=8,
                    )
                    llm_status_output = gr.Textbox(label="LLM status", interactive=False, lines=6, max_lines=12)
                    llm_payload_preview_output = gr.Textbox(label="LLM payload preview", interactive=True, lines=10, max_lines=16)
                    llm_output = gr.Textbox(label="LLM output", interactive=True, lines=12, max_lines=18)
                    with gr.Row():
                        llm_copy_btn = gr.Button("Copy LLM Output")
                        llm_save_btn = gr.Button("Save LLM Output")
                    llm_download_file = gr.File(label="Download LLM Output", interactive=False)

        completion_btn = gr.Button("Cancel", variant="stop", visible=False)

        click_event = submit_btn.click(
            fn=transcribe,
            inputs=[
                audio_input,
                model_dropdown,
                run_mode_dropdown,
                diar_checkbox,
                diar_backend_dropdown,
                max_speakers_input,
                visual_checkbox,
                visual_profile_dropdown,
                visual_backend_dropdown,
                visual_interval,
            ],
            outputs=[status_output, transcript_output, submit_btn, completion_btn, final_status_output],
        )
        run_mode_dropdown.change(
            fn=_update_mode_visibility,
            inputs=[run_mode_dropdown, diar_checkbox, visual_checkbox],
            outputs=[
                model_dropdown,
                diar_checkbox,
                diar_backend_dropdown,
                max_speakers_input,
                visual_checkbox,
                visual_profile_dropdown,
                visual_backend_dropdown,
                visual_interval,
            ],
        )
        diar_checkbox.change(
            fn=_update_diar_fields,
            inputs=[diar_checkbox, run_mode_dropdown],
            outputs=[diar_backend_dropdown, max_speakers_input],
        )
        visual_checkbox.change(
            fn=_update_visual_fields,
            inputs=[visual_checkbox, run_mode_dropdown],
            outputs=[visual_profile_dropdown, visual_backend_dropdown, visual_interval],
        )
        completion_btn.click(
            fn=set_cancel_flag,
            inputs=[],
            outputs=[],
            cancels=[click_event]
        )
        save_btn.click(
            fn=save_transcript,
            inputs=[transcript_output, audio_input, model_dropdown],
            outputs=[download_file]
        )
        copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[transcript_output],
            outputs=[]
        )
        llm_source_mode.change(
            fn=_update_postprocess_source_fields,
            inputs=[llm_source_mode],
            outputs=[llm_transcript_file, llm_transcript_paste],
        )
        llm_test_btn.click(
            fn=test_listener_llm_connection,
            inputs=[llm_profile_dropdown, llm_model_dropdown],
            outputs=[llm_status_output, llm_model_dropdown],
        )
        llm_preview_btn.click(
            fn=preview_listener_llm_payload,
            inputs=[
                transcript_output,
                llm_source_mode,
                llm_transcript_file,
                llm_transcript_paste,
                llm_ocr_file,
                llm_image_files,
                llm_include_images_checkbox,
                llm_image_ocr_backend,
                llm_image_ocr_fallback,
                llm_notes_input,
                llm_extra_context_input,
                llm_template_dropdown,
                llm_profile_dropdown,
                llm_model_dropdown,
            ],
            outputs=[llm_status_output, llm_payload_preview_output],
        )
        llm_run_btn.click(
            fn=run_listener_llm_postprocess,
            inputs=[
                transcript_output,
                llm_source_mode,
                llm_transcript_file,
                llm_transcript_paste,
                llm_ocr_file,
                llm_image_files,
                llm_include_images_checkbox,
                llm_image_ocr_backend,
                llm_image_ocr_fallback,
                llm_notes_input,
                llm_extra_context_input,
                llm_payload_preview_output,
                llm_profile_dropdown,
                llm_template_dropdown,
                llm_model_dropdown,
            ],
            outputs=[llm_status_output, llm_output],
        )
        llm_copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[llm_output],
            outputs=[],
        )
        llm_save_btn.click(
            fn=save_postprocess_output,
            inputs=[llm_output, llm_template_dropdown],
            outputs=[llm_download_file],
        )

    return iface


def launch_listener(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    max_tries: int = 50,
    queue_size: int = 16,
    auth_user: str | None = None,
    auth_pass: str | None = None,
    on_start: Callable[[int], None] | None = None,
) -> int:
    _ensure_listener_runtime()
    iface = create_interface()
    # Serialize jobs so only one transcription runs at a time on the host.
    iface.queue(default_concurrency_limit=1, max_size=queue_size)
    chosen_port = find_open_port(host=host, preferred_port=port, max_tries=max_tries)
    if on_start is not None:
        on_start(chosen_port)
    auth = None
    if auth_user and auth_pass:
        auth = (auth_user, auth_pass)
    if share and auth is None:
        raise RuntimeError("Gradio share mode requires auth_user/auth_pass.")
    iface.launch(
        server_name=host,
        server_port=chosen_port,
        share=share,
        show_error=True,
        auth=auth,
        css=CUSTOM_CSS,
        head=CUSTOM_HEAD,
    )
    return chosen_port


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyScribe as a Gradio listener.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Preferred port (default: 7860)")
    parser.add_argument("--max-port-tries", type=int, default=50, help="How many fallback ports to try")
    parser.add_argument("--queue-size", type=int, default=16, help="Max queued listener requests")
    parser.add_argument("--auth-user", default=None, help="Optional basic-auth username")
    parser.add_argument(
        "--allow-nonlocal-host",
        action="store_true",
        help="Allow binding to non-local interfaces (requires auth).",
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio public share URL")
    return parser.parse_args()


if __name__ == "__main__":
    reject_legacy_auth_pass_flag(sys.argv)
    args = _parse_args()
    auth_user, auth_pass = resolve_listener_auth(args.auth_user)
    validate_listener_security(
        args.host,
        auth_user=auth_user,
        auth_pass=auth_pass,
        allow_nonlocal_host=bool(args.allow_nonlocal_host),
        share=bool(args.share),
    )
    bound_port = launch_listener(
        host=args.host,
        port=args.port,
        share=args.share,
        max_tries=args.max_port_tries,
        queue_size=args.queue_size,
        auth_user=auth_user,
        auth_pass=auth_pass,
    )
    print(f"PyScribe listener running on http://{args.host}:{bound_port}")
