# utils.py
# Helper functions for PyScribe application.

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
    """Checks for required packages and offers to install them via pip."""
    required_packages = {
        "numpy": "numpy",
        "faster_whisper": "faster-whisper",
        "torch": "torch",
        "ffmpeg": "ffmpeg-python",
        "ttkthemes": "ttkthemes",
        "psutil": "psutil",
        "pynvml": "pynvml"
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
    """Gets a list of curated HF models and any locally cached ones."""
    popular_models = [
        "Systran/faster-whisper-tiny.en",
        "Systran/faster-whisper-base.en",
        "Systran/faster-whisper-small.en",
        "Systran/faster-whisper-medium.en",
        "Systran/faster-whisper-large-v3",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        "distil-whisper/distil-large-v3",
        "guillaumekln/whisper-large-v2-ct2",
        "guillaumekln/whisper-large-v3-ct2",
    ]
    
    local_models = []
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    if os.path.isdir(cache_dir):
        for item in os.listdir(cache_dir):
            if item.startswith("models--Systran--faster-whisper"):
                model_id = item.replace("models--", "").replace("--", "/")
                local_models.append(model_id)
    
    all_models = sorted(list(set(popular_models + local_models)))
    return all_models

def get_ffmpeg_cmd(tool: str = "ffmpeg") -> str | None:
    """
    Finds the path to an ffmpeg tool (ffmpeg, ffprobe, or ffplay).
    
    Args:
        tool (str): The name of the tool to find ('ffmpeg', 'ffprobe', 'ffplay').
    """
    env_dir = os.environ.get("FFMPEG_PATH")
    if env_dir and os.path.isdir(env_dir):
        tool_path = os.path.join(env_dir, f"{tool}.exe")
        if os.path.isfile(tool_path):
            return tool_path
            
    return shutil.which(tool)

def convert_to_16k_mono(src_path: str, tmpdir: str, ffmpeg_cmd: str) -> str:
    """Uses ffmpeg to convert any media file to a temporary 16kHz mono WAV file."""
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
    Loads an audio file and converts it to a float32 NumPy array,
    which is the format expected by Whisper models.
    """
    try:
        out, _ = (
            ffmpeg
            .input(file_path)
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        audio_np = np.frombuffer(out, dtype=np.int16)
        return audio_np.astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
