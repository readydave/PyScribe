# app.py
# Gradio-based listener UI for PyScribe.

import argparse
from collections.abc import Iterator
import datetime
import getpass
import os
import socket
import sys
import tempfile
import threading
from typing import Any, Callable

import gradio as gr
import pyperclip
from services.runtime_compat import ensure_platform_sys_version_compat
from services.runtime_env_service import (
    configure_runtime_environment,
    reexec_if_loader_env_changed,
)

ensure_platform_sys_version_compat()
configure_runtime_environment()
reexec_if_loader_env_changed()

from services import (
    AppConfig,
    check_ocr_backend_ready,
    detect_runtime,
    get_available_diarization_backends,
    get_backend_label,
    get_model_choices,
    load_config,
    normalize_model_name,
    recommend_model,
    save_config,
    transcribe_media_file,
)

RUNTIME = detect_runtime()
DEVICE = RUNTIME.device
COMPUTE_TYPE = RUNTIME.compute_type
RECOMMENDED_MODEL = recommend_model(RUNTIME)
ALL_MODELS = get_model_choices()
AVAILABLE_DIAR_BACKENDS = get_available_diarization_backends(include_off=False)
if not AVAILABLE_DIAR_BACKENDS:
    AVAILABLE_DIAR_BACKENDS = ["accurate"]

APP_CONFIG = load_config()

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
    _cancel_event.clear()

    if audio_path is None:
        yield "Status: No audio file provided.", "", gr.update(visible=True), gr.update(visible=False), ""
        return

    media_path = audio_path.name if hasattr(audio_path, "name") else str(audio_path)
    run_mode = _normalize_run_mode(run_mode)
    should_transcribe = run_mode in {"full", "transcribe_only"}
    should_run_visual = run_mode in {"full", "visual_only"}

    model_name = normalize_model_name(model_name)
    if should_transcribe and not model_name:
        yield "Status: No model selected.", "", gr.update(visible=True), gr.update(visible=False), ""
        return

    max_speakers = int(max_speakers_text) if str(max_speakers_text).strip().isdigit() else None
    use_diarization = bool(use_diarization and should_transcribe)
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
    visual_ocr_backend = str(visual_ocr_backend or "paddleocr").lower()
    if use_visual_analysis:
        ready, reason = check_ocr_backend_ready(visual_ocr_backend)
        if not ready:
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
        save_config(
            AppConfig(
                last_model=model_name,
                use_diarization=bool(use_diarization),
                max_speakers=max_speakers,
                diar_backend=diar_backend,
                run_mode=run_mode,
                use_visual_analysis=bool(use_visual_analysis),
                visual_profile=visual_profile,
                visual_ocr_backend=visual_ocr_backend,
                visual_sample_seconds=float(visual_sample_seconds or 1.0),
            )
        )
    except Exception:
        pass
        
    status_prefix = "Visual analysis" if run_mode == "visual_only" else "Transcription"
    yield f"Status: {status_prefix} starting...", "", gr.update(visible=False), gr.update(visible=True, value="Cancel"), ""
        
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

        result = transcribe_media_file(
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
        save_path = os.path.join(temp_dir, f"{base_name}_{ts}_{safe_model_name}.txt")

        with open(save_path, "w", encoding="utf-8") as transcript_file:
            transcript_file.write(transcript)
        
        gr.Info(f"Transcript ready for download.")
        return save_path

    except (OSError, AttributeError) as e:
        raise gr.Error(f"Could not save file: {e}")

def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)
    gr.Info("Copied to clipboard!")

def set_cancel_flag() -> None:
    _cancel_event.set()

# --- Gradio Interface Definition ---
def create_interface() -> gr.Blocks:
    initial_run_mode = _normalize_run_mode(APP_CONFIG.run_mode)
    initial_allow_transcription = initial_run_mode in {"full", "transcribe_only"}
    initial_allow_visual = initial_run_mode in {"full", "visual_only"}

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
            Optional multimodal mode can OCR sampled video frames (slides/chat text) and append highlights to the output.
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
                    info=get_backend_label(default_backend),
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
                    choices=["paddleocr", "surya", "pytesseract", "auto"],
                    value=str(APP_CONFIG.visual_ocr_backend or "paddleocr").lower(),
                    label="Visual OCR backend",
                    info="paddleocr/surya may download OCR models on first run.",
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
                transcript_output = gr.Textbox(label="Transcription", interactive=True, lines=15)
                with gr.Row():
                    copy_btn = gr.Button("Copy to Clipboard")
                    save_btn = gr.Button("Save Transcript")
                download_file = gr.File(label="Download Transcript", interactive=False)
                final_status_output = gr.Textbox(label="", interactive=False)

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
    iface = create_interface()
    # Serialize jobs so only one transcription runs at a time on the host.
    iface.queue(default_concurrency_limit=1, max_size=queue_size)
    chosen_port = find_open_port(host=host, preferred_port=port, max_tries=max_tries)
    if on_start is not None:
        on_start(chosen_port)
    auth = None
    if auth_user and auth_pass:
        auth = (auth_user, auth_pass)
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


def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def _validate_nonlocal_bind(
    host: str,
    *,
    auth_user: str | None,
    auth_pass: str | None,
    allow_nonlocal_host: bool,
) -> None:
    if _is_loopback_host(host):
        return
    if not allow_nonlocal_host:
        raise SystemExit(
            "Refusing non-local listener bind. Use --host 127.0.0.1, "
            "or add --allow-nonlocal-host to explicitly expose the listener."
        )
    if not (auth_user and auth_pass):
        raise SystemExit(
            "Non-local listener bind requires authentication. "
            "Provide --auth-user and set PYSCRIBE_AUTH_PASS, or set "
            "PYSCRIBE_AUTH_USER/PYSCRIBE_AUTH_PASS."
        )


def _clean_env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_listener_auth(auth_user: str | None) -> tuple[str | None, str | None]:
    resolved_user = (auth_user or "").strip() or _clean_env_value("PYSCRIBE_AUTH_USER")
    resolved_pass = _clean_env_value("PYSCRIBE_AUTH_PASS")
    if resolved_user and not resolved_pass and sys.stdin and sys.stdin.isatty():
        prompted = getpass.getpass("Listener auth password (input hidden): ").strip()
        resolved_pass = prompted or None
    if bool(resolved_user) != bool(resolved_pass):
        raise SystemExit(
            "Listener auth requires both username and password "
            "(provide --auth-user and set PYSCRIBE_AUTH_PASS, or set both "
            "PYSCRIBE_AUTH_USER/PYSCRIBE_AUTH_PASS)."
        )
    return resolved_user, resolved_pass


def _reject_legacy_auth_pass_flag(argv: list[str]) -> None:
    for arg in argv[1:]:
        if arg == "--auth-pass" or arg.startswith("--auth-pass="):
            raise SystemExit(
                "`--auth-pass` is no longer supported to avoid credential leakage. "
                "Set PYSCRIBE_AUTH_PASS instead."
            )


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
    _reject_legacy_auth_pass_flag(sys.argv)
    args = _parse_args()
    auth_user, auth_pass = _resolve_listener_auth(args.auth_user)
    _validate_nonlocal_bind(
        args.host,
        auth_user=auth_user,
        auth_pass=auth_pass,
        allow_nonlocal_host=bool(args.allow_nonlocal_host),
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
