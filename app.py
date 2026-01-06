# app.py
# New Gradio-based UI for PyScribe

import os
import tempfile
import datetime
import gradio as gr
import torch
import pyperclip
from faster_whisper import WhisperModel

from utils import get_available_hf_models, get_ffmpeg_cmd, convert_to_16k_mono

# --- Constants ---
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

# --- Hardware Detection ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
VRAM_GB = 0
if DEVICE == "cuda":
    try:
        props = torch.cuda.get_device_properties(0)
        VRAM_GB = round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        DEVICE = "cpu" # Fallback to CPU if CUDA fails

# --- Model Recommendation ---
def recommend_model():
    """Recommends a Whisper model size based on available hardware."""
    if DEVICE == "cuda":
        if VRAM_GB >= 10: return "large-v3"
        if VRAM_GB >= 8: return "large-v2"
        if VRAM_GB >= 5: return "medium"
        if VRAM_GB >= 3: return "small"
        return "base"
    cpu_count = os.cpu_count() or 1
    if cpu_count >= 12: return "small"
    if cpu_count >= 8: return "base"
    return "tiny"

RECOMMENDED_MODEL = recommend_model()
ALL_MODELS = sorted(list(set(MODEL_CHOICES + get_available_hf_models())))

# --- State for Cancellation ---
cancel_flag = gr.State(False)

def transcribe(audio_path, model_name, progress=gr.Progress()):
    """
    The main transcription function for the Gradio interface.
    """
    if audio_path is None:
        yield "Status: No audio file provided.", "", gr.update(visible=True), gr.update(visible=False), ""
        return

    if not model_name:
        yield "Status: No model selected.", "", gr.update(visible=True), gr.update(visible=False), ""
        return
        
    yield "Status: Transcription starting...", "", gr.update(visible=False), gr.update(visible=True, value="Cancel"), ""
        
    try:
        # 1. Prepare Audio
        progress(0, desc="Preparing audio...")
        ffmpeg_cmd = get_ffmpeg_cmd(tool="ffmpeg")
        if not ffmpeg_cmd:
            raise gr.Error("ffmpeg not found. Please ensure it's installed and in your system's PATH.")

        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_audio_path = convert_to_16k_mono(audio_path, temp_dir, ffmpeg_cmd)

            # 2. Load Model
            progress(0.2, desc=f"Loading model '{model_name}' on {DEVICE.upper()}...")
            model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)

            # 3. Transcribe
            progress(0.4, desc=f"Transcribing with '{model_name}'...")
            segments, info = model.transcribe(
                prepared_audio_path,
                beam_size=5,
                language=None, # Auto-detect language
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            
            total_duration = info.duration
            full_transcript = ""
            for segment in segments:
                if cancel_flag.value:
                    cancel_flag.value = False
                    break
                full_transcript += segment.text
                progress(segment.end / total_duration)
                yield "Status: Transcribing...", full_transcript.strip(), gr.update(visible=False), gr.update(visible=True), ""

    except Exception as e:
        # Use gr.Error to properly raise exceptions in the Gradio UI
        raise gr.Error(f"An error occurred: {e}")

    yield "Status: Transcription complete!", full_transcript.strip(), gr.update(visible=True), gr.update(visible=False), "Transcription complete!"

def save_transcript(transcript, audio_path, model_name):
    if not transcript:
        gr.Info("Nothing to save.")
        return None

    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        if audio_path is not None:
            base_name = os.path.splitext(os.path.basename(audio_path.name))[0]
            # Get the directory of the original file
            source_dir = os.path.dirname(audio_path.name)
        else:
            base_name = "transcript"
            source_dir = os.path.expanduser("~") # Default to home directory

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
    cancel_flag.value = True

# --- Gradio Interface Definition ---
with gr.Blocks(title="PyScribe v1.4") as iface:
    gr.Markdown(
        """
        # PyScribe: Gradio Edition
        A web interface for the PyScribe transcription tool, powered by `faster-whisper`.
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
                value=RECOMMENDED_MODEL,
                label="Select Transcription Model",
                info=f"Recommended for your hardware ({DEVICE.upper()}): {RECOMMENDED_MODEL}"
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
        inputs=[audio_input, model_dropdown],
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


if __name__ == "__main__":
    iface.launch(share=True)
