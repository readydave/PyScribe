# app.py
# Gradio-based listener UI for PyScribe.

import argparse
import datetime
import os
import socket
import tempfile
import threading

import gradio as gr
import pyperclip

from services import (
    AppConfig,
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

# --- State for Cancellation ---
_cancel_event = threading.Event()


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

def transcribe(audio_path, model_name, use_diarization, diar_backend, max_speakers_text, progress=gr.Progress()):
    """
    The main transcription function for the Gradio interface.
    """
    _cancel_event.clear()

    if audio_path is None:
        yield "Status: No audio file provided.", "", gr.update(visible=True), gr.update(visible=False), ""
        return

    media_path = audio_path.name if hasattr(audio_path, "name") else str(audio_path)

    model_name = normalize_model_name(model_name)
    if not model_name:
        yield "Status: No model selected.", "", gr.update(visible=True), gr.update(visible=False), ""
        return

    max_speakers = int(max_speakers_text) if str(max_speakers_text).strip().isdigit() else None
    if not use_diarization:
        diar_backend = "off"
    try:
        save_config(
            AppConfig(
                last_model=model_name,
                use_diarization=bool(use_diarization),
                max_speakers=max_speakers,
                diar_backend=diar_backend,
            )
        )
    except Exception:
        pass
        
    yield "Status: Transcription starting...", "", gr.update(visible=False), gr.update(visible=True, value="Cancel"), ""
        
    full_transcript = ""
    was_cancelled = False
    last_phase = "Transcribing"
    last_pct = 0.0
    try:
        def _on_status(msg: str):
            nonlocal last_phase
            last_phase = msg
            progress(None, desc=msg)

        def _on_progress(pct: float):
            nonlocal last_pct
            last_pct = min(max(pct, 0.0), 100.0)
            badge = _progress_badge(last_pct)
            desc = f"{badge} Transcribing {last_pct:.0f}%"
            progress(min(max(last_pct / 100.0, 0.0), 1.0), desc=desc)

        def _on_model_download_progress(pct: float):
            badge = _progress_badge(min(max(pct, 0.0), 100.0))
            progress(min(max(pct / 100.0, 0.0), 1.0), desc=f"{badge} Downloading model files...")

        def _on_text(text: str):
            nonlocal full_transcript
            full_transcript = text

        result = transcribe_media_file(
            media_path=media_path,
            model_name=model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=None,
            cancel_event=_cancel_event,
            use_diarization=use_diarization,
            diar_backend=diar_backend,
            max_speakers=max_speakers,
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
        yield "Status: Transcription cancelled.", full_transcript.strip(), gr.update(visible=True), gr.update(visible=False), "Cancelled."
        return

    final_badge = _progress_badge(100.0)
    yield f"Status: {final_badge} {last_phase}", full_transcript.strip(), gr.update(visible=True), gr.update(visible=False), "Transcription complete!"

def save_transcript(transcript, audio_path, model_name):
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

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        gr.Info(f"Transcript ready for download.")
        return save_path

    except (OSError, AttributeError) as e:
        raise gr.Error(f"Could not save file: {e}")

def copy_to_clipboard(text):
    pyperclip.copy(text)
    gr.Info("Copied to clipboard!")

def set_cancel_flag():
    _cancel_event.set()

# --- Gradio Interface Definition ---
def create_interface() -> gr.Blocks:
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
                )
                diar_checkbox = gr.Checkbox(label="Identify speakers", value=APP_CONFIG.use_diarization)
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
                )
                max_speakers_input = gr.Textbox(
                    label="Max speakers (optional)",
                    value="" if APP_CONFIG.max_speakers is None else str(APP_CONFIG.max_speakers),
                    placeholder="e.g. 2",
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
            inputs=[audio_input, model_dropdown, diar_checkbox, diar_backend_dropdown, max_speakers_input],
            outputs=[status_output, transcript_output, submit_btn, completion_btn, final_status_output],
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
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    max_tries: int = 50,
    queue_size: int = 16,
    auth_user: str | None = None,
    auth_pass: str | None = None,
):
    iface = create_interface()
    # Serialize jobs so only one transcription runs at a time on the host.
    iface.queue(default_concurrency_limit=1, max_size=queue_size)
    chosen_port = find_open_port(host=host, preferred_port=port, max_tries=max_tries)
    auth = None
    if auth_user and auth_pass:
        auth = (auth_user, auth_pass)
    iface.launch(server_name=host, server_port=chosen_port, share=share, show_error=True, auth=auth)
    return chosen_port


def _parse_args():
    parser = argparse.ArgumentParser(description="Run PyScribe as a Gradio listener.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Preferred port (default: 7860)")
    parser.add_argument("--max-port-tries", type=int, default=50, help="How many fallback ports to try")
    parser.add_argument("--queue-size", type=int, default=16, help="Max queued listener requests")
    parser.add_argument("--auth-user", default=None, help="Optional basic-auth username")
    parser.add_argument("--auth-pass", default=None, help="Optional basic-auth password")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public share URL")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    bound_port = launch_listener(
        host=args.host,
        port=args.port,
        share=args.share,
        max_tries=args.max_port_tries,
        queue_size=args.queue_size,
        auth_user=args.auth_user,
        auth_pass=args.auth_pass,
    )
    print(f"PyScribe listener running on http://{args.host}:{bound_port}")
