# PyScribe

PyScribe is a local transcription app for Windows and Linux built on `faster-whisper`.
It supports both a Qt desktop UI and a Gradio listener UI, with optional speaker diarization and optional visual OCR analysis for video files.

## Documentation

- `docs/user_guide.md` - full feature guide for Qt, Listener, CLI, and environment settings.
- `docs/qt_help.md` - in-app Qt help content.
- `CONTRIBUTING.md` - development and contribution workflow.
- `SECURITY.md` - vulnerability reporting and security guidance.
- `CHANGELOG.md` - project change history.

## Highlights

- Local transcription using `faster-whisper`
- Hardware-aware model recommendations
- Qt desktop mode and Gradio listener mode
- Optional speaker diarization with selectable backend
- Optional visual analysis (OCR on sampled video frames)
- Live status, progress, and transcript updates
- Qt controls for cancel and force stop
- Save modes: combined output, transcript-only, OCR-only
- Optional Hugging Face token support for gated/private model access

## Requirements

- Python 3.10+ (3.12 recommended)
- FFmpeg available in PATH
  - Windows: `winget install Gyan.FFmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- Optional OCR runtime:
  - `pytesseract` + OS `tesseract` executable
  - or PaddleOCR runtime dependencies

## Installation

### Windows

```bash
py -3.12 -m venv C:\Code\_envs\pyscribe
C:\Code\_envs\pyscribe\Scripts\activate
cd C:\Code\PyScribe
pip install -r requirements.txt
```

### Linux

```bash
sudo apt update
sudo apt install -y ffmpeg python3.12 python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Interactive launcher menu (choose mode)
python main.py

# Qt desktop
python main.py qt

# Listener mode (localhost by default)
python main.py serve --port 7860
```

Listener mode automatically moves to the next free port if the preferred port is occupied.

## Listener Security

Binding to non-local interfaces is intentionally restricted.

- Local-only bind (safe default): `--host 127.0.0.1`
- Non-local bind requires:
  - `--allow-nonlocal-host`
  - `--auth-user <username>`
  - `PYSCRIBE_AUTH_PASS` environment variable

Example:

```bash
PYSCRIBE_AUTH_USER=admin PYSCRIBE_AUTH_PASS=change-me \
python main.py serve --host 0.0.0.0 --allow-nonlocal-host --port 7860
```

## Models

- Built-in model choices are available in both UIs.
- Custom Hugging Face repo IDs are supported (for example `owner/repo`).
- Full Hugging Face model URLs are normalized to repo IDs.
- Qt mode asks for model download confirmation with a size estimate before first download.

## Feature Notes

- **Diarization:** optional; when unavailable/failing, transcription still completes without speaker labels.
- **Visual analysis:** optional; supports `fast`, `balanced`, `accurate` profiles and OCR backend selection.
- **Qt output save modes:** `Save All`, `Save Transcript Only`, `Save OCR Only`.
- **Benchmarking:** Qt Tools menu includes benchmark runner for bundled sample media.

## CLI / Packaging

```bash
pip install .
pyscribe --help
```

- Listener helper script: `scripts/run_listener.sh`
- systemd example unit: `deploy/systemd/pyscribe-listener.service.example`

## Runtime Environment Defaults

PyScribe configures writable cache defaults for OCR/model assets when needed, including:
`PYSCRIBE_CACHE_DIR`, `PADDLE_HOME`, `PADDLE_PDX_CACHE_HOME`, `HF_HOME`, and `MODELSCOPE_CACHE`.

See `docs/user_guide.md` for all environment variables and behavior.

## License

GPL-3.0-only

## Acknowledgments

Benchmark audio source: [LibriVox](https://librivox.org/).
