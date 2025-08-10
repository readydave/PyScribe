# transcriber.py
# Core transcription logic for PyScribe.

import time
import tempfile
import ffmpeg
from faster_whisper import WhisperModel
from utils import get_ffmpeg_cmd, convert_to_16k_mono, load_audio_waveform

def run_transcription(app_instance, model_name: str, media_path: str):
    """
    Runs the full transcription process.
    
    Args:
        app_instance: The instance of the PyScribeApp to update the UI.
        model_name (str): The name of the model to use.
        media_path (str): The path to the media file.
    """
    temp_dir = tempfile.TemporaryDirectory()
    try:
        start_time = time.time()

        # Step 1: Load Model (if needed)
        if app_instance.current_model_name != model_name:
            app_instance.set_status(f"Loading '{model_name}' model...")
            compute_type = "float16" if app_instance.device == "cuda" else "int8"
            app_instance.model = WhisperModel(model_name, device=app_instance.device, compute_type=compute_type)
            app_instance.current_model_name = model_name
            app_instance.refresh_hf_model_list()

        # Step 2: Prepare Audio
        app_instance.set_status("Preparing audio file...")
        ffmpeg_cmd = get_ffmpeg_cmd()
        if not ffmpeg_cmd:
            raise FileNotFoundError("ffmpeg not found.")
        
        wav_path = convert_to_16k_mono(media_path, temp_dir.name, ffmpeg_cmd)
        
        try:
            probe = ffmpeg.probe(wav_path)
            duration = float(probe['format']['duration'])
        except (ffmpeg.Error, KeyError):
            duration = 0
        
        audio_np = load_audio_waveform(wav_path)
        app_instance.update_progress(0)

        # Step 3: Detect Language
        task = "transcribe"
        lang_code = 'en'
        is_english_only_model = ".en" in model_name.lower()
        
        if not is_english_only_model:
            app_instance.set_status("Detecting language...")
            detected_lang_code, lang_prob, *_ = app_instance.model.detect_language(audio_np)
            
            if detected_lang_code != 'en' and lang_prob > 0.6:
                app_instance.prompt_for_translation(detected_lang_code)
                task = app_instance.user_task_choice
                lang_code = detected_lang_code if task == "transcribe" else 'en'
            else:
                lang_code = detected_lang_code
        else:
            app_instance.set_status("English-only model selected.")

        # Step 4: Transcribe
        status_msg = "Translating..." if task == "translate" else f"Transcribing in {lang_code}..."
        app_instance.set_status(status_msg)
        
        segments_generator, _ = app_instance.model.transcribe(
            audio_np, task=task, language=lang_code if task == "transcribe" else None, beam_size=5
        )
        
        all_text_segments = []
        for segment in segments_generator:
            all_text_segments.append(segment.text.strip())
            app_instance.update_text_area(" ".join(all_text_segments))
            if duration > 0:
                progress = (segment.end / duration) * 100
                app_instance.update_progress(progress)

        # Step 5: Finalize
        app_instance.transcription = " ".join(all_text_segments)
        app_instance.update_progress(100)
        
        elapsed_time = time.time() - start_time
        final_status = f"Complete in {elapsed_time:.1f}s | Language: {lang_code} | Model: {model_name}"
        app_instance.set_status(final_status)
        
        app_instance.enable_save_buttons()

    except Exception as e:
        app_instance.show_error(f"An error occurred during transcription:\n\n{e}")
        app_instance.set_status("An error occurred.")
    finally:
        app_instance.finish_transcription_flow()
        try:
            temp_dir.cleanup()
        except Exception:
            pass
