# benchmark.py
# This module contains the logic for the benchmarking feature.

import os
import time
import threading
from faster_whisper import WhisperModel
from utils import load_audio_waveform, convert_to_16k_mono, get_ffmpeg_cmd
import tempfile

def run_benchmark(app_instance, benchmark_window, selected_models: list[str], audio_language: str):
    """
    Runs the transcription benchmark on a list of selected models.

    Args:
        app_instance: The main PyScribeApp instance for accessing hardware info.
        benchmark_window: The BenchmarkWindow instance to send results back to.
        selected_models (list[str]): A list of model names to benchmark.
        audio_language (str): The language of the audio file to use ('en' or 'es').
    """
    temp_dir = tempfile.TemporaryDirectory()
    benchmark_window.start_hw_monitor()
    try:
        # Define the paths to the benchmark audio files.
        benchmark_files = {
            "en": os.path.join("assets", "benchmark-sherlock-holmes-en.mp3"),
            "es": os.path.join("assets", "benchmark-napoleon-es.mp3")
        }
        
        # Select the audio file based on the user's choice.
        audio_path = benchmark_files.get(audio_language)

        # --- Pre-flight Checks ---
        if not audio_path or not os.path.exists(audio_path):
            benchmark_window.show_error(f"Benchmark audio file for '{audio_language}' not found in the 'assets' folder.")
            return

        # Prepare the audio file once for all tests.
        ffmpeg_cmd = get_ffmpeg_cmd(tool="ffmpeg")
        if not ffmpeg_cmd:
            raise FileNotFoundError("ffmpeg not found.")
        wav_path = convert_to_16k_mono(audio_path, temp_dir.name, ffmpeg_cmd)
        audio_np = load_audio_waveform(wav_path)

        # --- Run Benchmark Loop ---
        for i, model_name in enumerate(selected_models):
            # Check for a cancellation signal from the UI.
            if benchmark_window.cancel_event.is_set():
                benchmark_window.update_results(f"\nBenchmark cancelled by user.")
                break

            # Update the status and progress in the benchmark window.
            progress_text = f"Testing model {i + 1}/{len(selected_models)}: {model_name}"
            benchmark_window.update_status(progress_text)
            benchmark_window.update_progress((i / len(selected_models)) * 100)
            
            # --- Transcription and Measurement ---
            start_time = time.time()
            
            # Load the model.
            compute_type = "float16" if app_instance.device == "cuda" else "int8"
            model = WhisperModel(model_name, device=app_instance.device, compute_type=compute_type)
            
            # Transcribe the audio.
            segments, _ = model.transcribe(audio_np, language=audio_language, beam_size=5)
            
            # The transcription itself is done, but we iterate to ensure it's complete.
            _ = [s.text for s in segments]
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # --- Format and Display Results ---
            result_line = f"\n- {model_name}: {elapsed_time:.2f} seconds"
            benchmark_window.update_results(result_line)
            
            del model

        benchmark_window.update_progress(100) # Final progress update

        # --- Final Status ---
        # The recommendation logic has been removed.
        if not benchmark_window.cancel_event.is_set():
            benchmark_window.update_status("Benchmark complete.")
        
        benchmark_window.enable_start_button()

    except Exception as e:
        benchmark_window.show_error(f"An error occurred during benchmark:\n\n{e}")
        benchmark_window.update_status("Benchmark failed.")
    finally:
        benchmark_window.stop_hw_monitor()
        try:
            temp_dir.cleanup()
        except Exception:
            pass
