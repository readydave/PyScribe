# transcriber.py
# This module contains the core transcription logic for the PyScribe application.
# It is designed to be run in a separate thread to keep the GUI responsive.

import time
import datetime # For formatting the elapsed time
import tempfile
import ffmpeg
from faster_whisper import WhisperModel
from utils import get_ffmpeg_cmd, convert_to_16k_mono, load_audio_waveform

def run_transcription(app_instance, model_name: str):
    """
    Runs the full transcription process from start to finish. This function is intended
    to be the target of a background thread.

    It communicates with the main UI thread via callback methods on the app_instance.

    Args:
        app_instance: The instance of the PyScribeApp class. Used to update the UI.
        model_name (str): The name/ID of the Whisper model to use for transcription.
    """
    app_instance.start_hw_monitor()
    try:
        start_time = time.time()

        # Step 1: Ensure Audio is Prepared
        app_instance.ensure_audio_is_prepared()
        wav_path = app_instance.prepared_audio_path
        if not wav_path:
            raise ValueError("Audio could not be prepared for transcription.")

        audio_np = load_audio_waveform(wav_path)
        app_instance.update_progress(0)

        # Step 2: Language Detection and Model Validation
        app_instance.set_status("Detecting language...")
        
        lang_detector_model = WhisperModel("tiny", device=app_instance.device, compute_type="int8")
        detected_lang_code, lang_prob, *_ = lang_detector_model.detect_language(audio_np)
        del lang_detector_model

        is_english_only_model = ".en" in model_name.lower()
        lang_to_transcribe_in = detected_lang_code

        if is_english_only_model and detected_lang_code != 'en':
            app_instance.prompt_for_english_override(detected_lang_code, model_name)
            
            if app_instance.user_override_choice:
                lang_to_transcribe_in = 'en'
                app_instance.set_status("User override: Forcing transcription in English.")
            else:
                app_instance.set_status("Transcription cancelled due to language/model mismatch.")
                return

        # Step 3: Load the User-Selected Model
        if app_instance.current_model_name != model_name:
            app_instance.set_status(f"Loading '{model_name}' model...")
            compute_type = "float16" if app_instance.device == "cuda" else "int8"
            app_instance.model = WhisperModel(model_name, device=app_instance.device, compute_type=compute_type)
            app_instance.current_model_name = model_name
            app_instance.refresh_hf_model_list()

        # Get audio duration for progress bar calculation.
        try:
            probe = ffmpeg.probe(wav_path)
            duration = float(probe['format']['duration'])
        except (ffmpeg.Error, KeyError):
            duration = 0

        # Step 4: Transcribe
        status_msg = f"Transcribing in {lang_to_transcribe_in}..."
        app_instance.set_status(status_msg)
        
        segments_generator, _ = app_instance.model.transcribe(
            audio_np, task="transcribe", language=lang_to_transcribe_in, beam_size=5
        )
        
        all_text_segments = []
        for segment in segments_generator:
            if app_instance.cancel_event.is_set():
                app_instance.set_status("Transcription cancelled by user.")
                return

            all_text_segments.append(segment.text.strip())
            app_instance.update_text_area(" ".join(all_text_segments))
            if duration > 0:
                progress = (segment.end / duration) * 100
                app_instance.update_progress(progress)

        # --- Step 5: Finalize ---
        app_instance.transcription = " ".join(all_text_segments)
        app_instance.update_progress(100)
        
        elapsed_time = time.time() - start_time
        # --- CHANGE: Format elapsed time to HH:MM:SS ---
        formatted_time = str(datetime.timedelta(seconds=int(elapsed_time)))
        final_status = f"Complete in {formatted_time} | Language: {lang_to_transcribe_in} | Model: {model_name}"
        app_instance.set_status(final_status)
        
        app_instance.enable_save_buttons()

    except Exception as e:
        if not app_instance.cancel_event.is_set():
            app_instance.show_error(f"An error occurred during transcription:\n\n{e}")
            app_instance.set_status("An error occurred.")
    finally:
        app_instance.stop_hw_monitor()
        app_instance.finish_transcription_flow()
