# PyScribe - Local Transcription GUI

PyScribe is a cross-platform local transcription app for Windows and Linux, powered by `faster-whisper`. It runs fully on your machine (desktop UI or browser listener mode) for privacy and high performance.

## Key Features
- High-speed transcription with `faster-whisper`
- Hardware-aware model recommendations (GPU/CPU)
- Live progress, live transcript, and hardware metrics
- Optional speaker diarization with selectable backends
- Optional multimodal visual analysis (OCR on sampled video frames for slides/chat text)
- Qt desktop UI and Gradio listener mode
- Model download confirmation + in-app progress
- Hugging Face token support for gated diarization models
- Interactive launcher menu (`python main.py`) so users can choose Qt desktop or listener mode without remembering commands
- Qt quality-of-life controls: `Cancel`, `Force Stop`, and `Exit`
- Save dialog defaults to source media folder and remembers the last browse/save location
- Save dropdown modes: `Save All (Transcript + OCR)`, `Save Transcript Only`, `Save OCR Only`
- Qt menu bar with `Tools` (HF Token, Benchmark) and `Help` (PyScribe Help, Model Help, Open Logs Folder, About)

## Requirements
- Python 3.12 (recommended)
- FFmpeg in PATH
  - Windows: `winget install Gyan.FFmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- Optional for multimodal OCR:
  - Windows: install Tesseract OCR and add to PATH
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - PaddleOCR model files are cached in a user-writable location by default (`~/.cache/pyscribe`)

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
# Interactive launcher (choose Qt / Listener)
python main.py

# Qt desktop UI
python main.py qt

# Gradio listener mode
python main.py serve --host 0.0.0.0 --port 7860
```

Listener mode auto-falls to the next free port if `7860` is in use.

## Qt Notes
- `Open Folder` opens the selected media folder (or last-opened folder if no media is selected).
- Speaker identification toggle clearly shows on/off state.
- Deselect speaker identification to disable diarization and its progress bar.
- Enable **Analyze visuals (slides/chat OCR)** to append on-screen text highlights with timestamps.
  - The OCR pass focuses on the shared-content area and separately attempts to capture right-panel chat text.
  - Choose visual mode: `fast`, `balanced`, `accurate` (speed vs thoroughness).
  - OCR backend can be selected (`paddleocr`, `surya`, `pytesseract`, `auto`).
  - `paddleocr` may download OCR model files on first run (Qt prompts before first use).
  - `surya` is experimental and may require a separate environment with newer Torch versions.
  - Report includes both the requested backend and the backend actually used (including fallback reason when applicable).
  - Visual analysis now skips OCR on unchanged frames (dedupe) for faster processing.
- Save dropdown behavior:
  - **Save All (Transcript + OCR)**: saves spoken transcript plus the visual/OCR section in one file.
  - **Save Transcript Only**: saves only spoken transcript (including speaker labels when diarization is enabled).
  - **Save OCR Only**: saves only visual analysis/OCR findings (if available).
- See `docs/qt_help.md` for the in-app help content shown by **Help -> PyScribe Help**.

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
- AppImage/runtime note: PyScribe auto-sets OCR/model cache env defaults (`PADDLE_HOME`, `PADDLE_PDX_CACHE_HOME`, `HF_HOME`, `MODELSCOPE_CACHE`) to user-writable paths and enables direct source tries (`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`). Override root with `PYSCRIBE_CACHE_DIR` if desired.

## License
GPLv3

## Acknowledgments
Benchmark audio from [LibriVox](https://librivox.org/).
