# utils.py
# Helper functions for PyScribe application.

"""
This module provides various utility functions for the PyScribe application,
including dependency checking, FFmpeg integration, and audio processing.
"""

import os
import sys
import shutil
import subprocess
import importlib.util
import tkinter as tk
from tkinter import messagebox

import ffmpeg
import numpy as np

def check_and_install_dependencies():
    """
    Checks for required Python packages and offers to install them via pip if missing.
    
    Returns:
        bool: True if all dependencies are met or successfully installed, False otherwise.
    """
    required_packages = {
        "numpy": "numpy",
        "faster_whisper": "faster-whisper",
        "torch": "torch",
        "ffmpeg": "ffmpeg-python",
        "ttkthemes": "ttkthemes"
    }
    
    missing_packages = []
    for import_name, install_name in required_packages.items():
        if not importlib.util.find_spec(import_name):
            missing_packages.append(install_name)
            
    if not missing_packages:
        return True

    root = tk.Tk()
    root.withdraw()
    
    msg = (f"The following required packages are missing:\n\n"
           f"{', '.join(missing_packages)}\n\n"
           f"Do you want to attempt to install them now?")
           
    if messagebox.askyesno("Missing Dependencies", msg, parent=root):
        try:
            args = [sys.executable, "-m", "pip", "install"] + missing_packages
            subprocess.check_call(args)
            messagebox.showinfo("Installation Complete", "Packages installed successfully. Please restart the application.", parent=root)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Installation Failed", f"Failed to install packages. Please install them manually.\n\nError: {e}", parent=root)
        finally:
            root.destroy()
            return False
    else:
        messagebox.showwarning("Dependencies Missing", "The application cannot run without the required packages. Exiting.", parent=root)
        root.destroy()
        return False

def get_available_hf_models() -> list[str]:
    """
    Retrieves a list of recommended Hugging Face models and any faster-whisper
    models found in the local Hugging Face cache.
    
    Returns:
        list[str]: A sorted list of unique model identifiers.
    """
    popular_models = [
        "Systran/faster-whisper-tiny.en",
        "Systran/faster-whisper-base.en",
        "Systran/faster-whisper-small.en",
        "Systran/faster-whisper-medium.en",
        "Systran/faster-whisper-large-v3",
        "Systran/faster-whisper-large-v2", # Added large-v2 as it's a common choice
    ]
    
    local_models = []
    # Define the cache directory for Hugging Face models
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    
    # Check if the cache directory exists
    if os.path.isdir(cache_dir):
        # Iterate through items in the cache directory
        for item in os.listdir(cache_dir):
            # Look for directories that match the faster-whisper model naming convention
            # (e.g., models--Systran--faster-whisper-large-v3)
            if item.startswith("models--Systran--faster-whisper"):
                # Extract the actual model ID from the directory name
                # e.g., "models--Systran--faster-whisper-large-v3" -> "Systran/faster-whisper-large-v3"
                model_id = item.replace("models--", "").replace("--", "/")
                local_models.append(model_id)
    
    all_models = sorted(list(set(popular_models + local_models)))
    return all_models

def get_ffmpeg_cmd() -> str | None:
    """
    Finds the path to the ffmpeg executable.
    
    It first checks an environment variable `FFMPEG_PATH`, then attempts
    to find 'ffmpeg' in the system's PATH.
    
    Returns:
        str | None: The path to the ffmpeg executable if found, otherwise None.
    """
    # Check for a custom FFMPEG_PATH environment variable
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    # Otherwise, try to find ffmpeg in the system's PATH
    return shutil.which("ffmpeg")

def convert_to_16k_mono(src_path: str, tmpdir: str, ffmpeg_cmd: str) -> str:
    """
    Uses ffmpeg to convert any media file to a temporary 16kHz mono WAV file.
    This format is required by the faster-whisper library.
    
    Args:
        src_path (str): The path to the source audio/video file.
        tmpdir (str): The directory where the temporary WAV file will be saved.
        ffmpeg_cmd (str): The path to the ffmpeg executable.
        
    Returns:
        str: The path to the newly created 16kHz mono WAV file.
        
    Raises:
        RuntimeError: If an ffmpeg error occurs during conversion.
    """
    out_path = os.path.join(tmpdir, "audio_16k_mono.wav")
    try:
        (
            ffmpeg.input(src_path)
            .output(out_path, acodec="pcm_s16le", ar=16000, ac=1)
            .run(cmd=ffmpeg_cmd, quiet=True, overwrite_output=True)
        )
        return out_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg error: {e.stderr.decode()}") from e

def load_audio_waveform(file_path: str) -> np.ndarray:
    """
    Loads an audio file using ffmpeg and converts it to a float32 NumPy array.
    This is the required input format for faster-whisper models (16kHz, mono, float32).
    
    Args:
        file_path (str): The path to the audio file (e.g., WAV, MP3, MP4).
        
    Returns:
        np.ndarray: A NumPy array representing the audio waveform,
                    sampled at 16kHz, mono, and with float32 data type.
                    The values are normalized to the range [-1.0, 1.0].
                    
    Raises:
        RuntimeError: If an ffmpeg error occurs during audio loading.
    """
    try:
        # Use ffmpeg to read the audio as raw PCM (signed 16-bit little-endian)
        out, _ = (
            ffmpeg
            .input(file_path)
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        audio_np = np.frombuffer(out, dtype=np.int16)
        # Convert to float32 and normalize to [-1.0, 1.0] by dividing by the max value for int16 (32768)
        return audio_np.astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
