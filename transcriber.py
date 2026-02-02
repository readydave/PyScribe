# transcriber.py
# This module contains the core transcription logic for the PyScribe application.
# It is designed to be run in a separate thread to keep the GUI responsive.

import time
import datetime # For formatting the elapsed time
from utils import load_audio_waveform
from services import detect_language, load_model, transcribe_prepared_audio

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
        
        detected_lang_code, lang_prob = detect_language(audio_np, device=app_instance.device)

        is_english_only_model = ".en" in model_name.lower()
        lang_to_transcribe_in = detected_lang_code

        # Ask user to confirm/override if not English
        if detected_lang_code != "en":
            lang_to_transcribe_in = app_instance.prompt_detected_language(detected_lang_code, lang_prob)

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
            app_instance.model = load_model(
                model_name,
                device=app_instance.device,
                compute_type=compute_type,
                use_cache=True,
            )
            app_instance.current_model_name = model_name
            app_instance.refresh_hf_model_list()

        # Step 4: Transcribe
        status_msg = f"Transcribing in {lang_to_transcribe_in}..."
        app_instance.set_status(status_msg)
        result = transcribe_prepared_audio(
            wav_path=wav_path,
            model=app_instance.model,
            language=lang_to_transcribe_in,
            cancel_event=app_instance.cancel_event,
            use_diarization=getattr(app_instance, "use_diarization", False),
            diar_backend=getattr(app_instance, "diar_backend", "accurate"),
            device=app_instance.device,
            max_speakers=app_instance.max_speakers_override,
            on_status=app_instance.set_status,
            on_text=app_instance.update_text_area,
            on_progress=app_instance.update_progress,
            on_diar_progress=app_instance.update_diar_progress,
        )
        if result.cancelled:
            app_instance.set_status("Transcription cancelled by user.")
            return

        app_instance.transcription = result.transcript

        # Reflect final transcript (with speakers if available) in the UI text box.
        app_instance.update_text_area(app_instance.transcription)
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
