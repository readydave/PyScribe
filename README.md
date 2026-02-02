# PyScribe - Local Transcription GUI

PyScribe is a cross-platform local transcription app for Windows and Linux, powered by `faster-whisper`. It runs fully on your machine (desktop UI or browser listener mode) for privacy and high performance.

## Key Features
- High-speed transcription with `faster-whisper`
- Hardware-aware model recommendations (GPU/CPU)
- Live progress, live transcript, and hardware metrics
- Optional speaker diarization with selectable backends
- Tk desktop UI, Qt desktop UI, and Gradio listener mode
- Model download confirmation + in-app progress
- Hugging Face token support for gated diarization models

## Requirements
- Python 3.12 (recommended)
- FFmpeg in PATH
  - Windows: `winget install Gyan.FFmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

## Installation (Windows)
```bash
py -3.12 -m venv C:\Code\_envs\pyscribe
C:\Code\_envs\pyscribe\Scripts\activate
cd C:\Code\PyScribe
pip install -r requirements.txt
```

## Installation (Linux)
```bash
sudo apt update
sudo apt install -y ffmpeg python3.12 python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# Tk desktop UI
python main.py

# Qt desktop UI
python main.py qt
# or
python main.py --gui qt

# Gradio listener mode
python main.py serve --host 0.0.0.0 --port 7860
```

Listener mode auto-falls to the next free port if `7860` is in use.

## Hugging Face Token (Diarization)
Some diarization pipelines are gated on Hugging Face. In Qt mode, click `HF Token` and paste your token. You may also need to accept terms on model pages (for example, `pyannote/speaker-diarization-3.1`).

## Custom Model Repos
Users can choose built-in models or custom Hugging Face repos.

- Preferred: `owner/repo`
- Full HF URLs are accepted and auto-converted to `owner/repo`
- App performs best-effort size estimation and asks before download
- Private/gated repos require authentication and accepted model terms

## Packaging and Service Mode
- Optional CLI install:
  ```bash
  pip install .
  pyscribe --help
  ```
- Linux listener helper: `scripts/run_listener.sh`
- systemd example: `deploy/systemd/pyscribe-listener.service.example`

## License
GPLv3

## Acknowledgments
Benchmark audio from [LibriVox](https://librivox.org/).
