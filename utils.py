# utils.py
# Helper functions for PyScribe application.

import os
import shutil
import sys

import numpy as np


def get_ffmpeg_cmd(tool: str = "ffmpeg") -> str | None:
    """
    Finds the path to an ffmpeg tool (ffmpeg, ffprobe, or ffplay).

    Args:
        tool (str): The name of the tool to find ('ffmpeg', 'ffprobe', 'ffplay').
    """
    env_dir = os.environ.get("FFMPEG_PATH")
    if env_dir and os.path.isdir(env_dir):
        candidates = [tool]
        if sys.platform.startswith("win"):
            candidates.insert(0, f"{tool}.exe")
        for candidate in candidates:
            tool_path = os.path.join(env_dir, candidate)
            if os.path.isfile(tool_path):
                return tool_path

    return shutil.which(tool)


def convert_to_16k_mono(src_path: str, tmpdir: str, ffmpeg_cmd: str) -> str:
    """Uses ffmpeg to convert any media file to a temporary 16kHz mono WAV file."""
    import ffmpeg

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
    import ffmpeg

    try:
        out, _ = (
            ffmpeg.input(file_path)
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        audio_np = np.frombuffer(out, dtype=np.int16)
        return audio_np.astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
