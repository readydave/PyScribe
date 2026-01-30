# PyScribe - Local Transcription GUI

PyScribe is a Windows-friendly GUI for fast, private audio/video transcription powered by `faster-whisper`. It now includes streamlined model selection, drag-and-drop input, dual progress bars, and optional speaker diarization backends (pyannote or NVIDIA Sortformer).

---

## What’s new (high level)
- Single model chooser with tier badges (🟢 Fast, 🟡 Balanced, 🔴 Pro) and cache status (☑ downloaded, ● cached, ☐ fetch).
- “Refresh models” merges Hugging Face faster‑whisper/distil models with anything already in your HF cache; “Filter to available” shows only cached models.
- Drag & Drop audio/video next to Browse File; toolbar consolidated at the top.
- Dual progress bars: transcription and diarization, with clear status messages.
- Diarization modes: Off, Fast (pyannote-lite), Accurate (pyannote 3.x), and optional GPU Sortformer (NeMo, CUDA/WSL/Linux).
- Language confirmations and English override prompts when model/language mismatch.
- Cache rescan button and hardware‑best model preselect.

---

## Key Features
- **High-speed transcription** with `faster-whisper`.
- **Hardware-aware recommendations** for GPU/CPU and model tier.
- **Live progress + live transcript** with dual bars (transcribe + diarize).
- **Live hardware monitoring** (CPU, RAM, GPU, VRAM).
- **Audio playback & cancel** while transcribing.
- **Auto language detection** with optional forced English override.
- **Benchmark tool** to compare models on your hardware.
- **Model selection** from curated faster‑whisper/distil models plus your cached models; badges show tier and download status.
- **Speaker diarization** with selectable backends and max-speakers limit.
- **Drag & drop input** for quick file selection.

---

## Requirements (Windows)
- **Python 3.12** (recommended).
- **FFmpeg** on PATH (install with `winget install Gyan.FFmpeg`).
- **CUDA 12.1** if using GPU Torch on Windows.

Optional (only if you want Sortformer via WSL/Linux):
- CUDA 12.1 + cuDNN 9.x, Python 3.10+, `nemo_toolkit[asr]`, and `nvidia-cudnn-cu12` wheels inside WSL/Linux.

---

## Install (Windows, external venv)
1) Install FFmpeg: `winget install Gyan.FFmpeg`  
2) Extract repo to e.g. `C:\Code\PyScribe`  
3) Create envs folder: `C:\Code\_envs`  
4) Create venv: `py -3.12 -m venv C:\Code\_envs\pyscribe`  
5) Activate: `C:\Code\_envs\pyscribe\Scripts\activate`  
6) `cd C:\Code\PyScribe`  
7) Install deps (GPU path shown; CPU path is just `-r requirements.txt`):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Launch: double‑click `launch.bat` (auto-uses the external venv).

---

## Usage guide
- **Choose Model**: single dialog with search; badges show tier; status icons show cache.  
  - **Refresh models**: fetch latest HF faster‑whisper/distil; merge with cache.  
  - **Filter to available**: show only cached models.  
  - **Rescan cache**: re-index local HF cache folders.  
- **Drag & Drop**: drop audio/video onto the red-outlined box; or click **Browse File**.  
- **Diarization**: check **Identify speakers**, set **Max speakers**.  
  - **Diar mode**: Off | Fast (pyannote-lite) | Accurate (pyannote 3.x) | Sortformer (NeMo, requires CUDA via WSL/Linux).  
  - Dual progress bars show when diarization runs.  
- **Language prompts**: if non‑English is detected with an English-only model, you can force English or cancel.  
- **Save/Copy**: buttons enable automatically when a run finishes.

---

## Optional: Sortformer (NeMo) in WSL/Linux
Only needed if you want the fastest diarization:
1) In WSL/Ubuntu: install CUDA drivers; create a Python 3.10+ venv.  
2) Install CUDA Torch wheels (cu121) and project requirements.  
3) Add cuDNN wheel and NeMo:  
```bash
pip install nvidia-cudnn-cu12==9.1.0.70
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
pip install "nemo_toolkit[asr]==1.23.0"
```
4) Run `python main.py` inside WSL and select **Diar mode: sortformer**.  
If you stay on Windows native, use **Accurate** or **Fast** modes instead.

---

## License
GPLv3

## Acknowledgments
Benchmark audio from [LibriVox](https://librivox.org/).
