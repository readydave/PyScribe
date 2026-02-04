# PyScribe User Guide

This guide explains every user-facing feature in PyScribe.

## 1) Launch Modes

PyScribe has two frontends:

- **Qt Desktop** (`python main.py qt`)
- **Gradio Listener** (`python main.py serve`)

If you run `python main.py` with no mode, you get an interactive launcher menu.

## 2) Common Concepts

### Media Input

- Audio and video files are supported.
- Video files can be transcribed from audio and optionally analyzed visually (OCR).

### Models

- You can use built-in model names or custom Hugging Face repos (`owner/repo`).
- Full Hugging Face URLs are accepted and normalized to repo IDs.

### Processing Stages

- **Transcription**: speech-to-text.
- **Diarization** (optional): speaker labeling (for example `[S1]`, `[S2]`).
- **Visual analysis** (optional): OCR over sampled video frames.

## 3) Qt Desktop Features

### File Selection

- **Browse**: open file picker for media.
- **Drag-and-drop zone**: drop a media file directly.
- **Open Folder**: open selected file directory (or last-used folder if no file selected).

### Model Selection

- **Model** dropdown is editable.
- Recommended model is shown based on detected hardware.

### Processing Toggles

- **Transcribe audio**:
  - On: speech transcription runs.
  - Off: transcription is skipped.
- **Speaker Identification is On/Off**:
  - On: diarization runs after transcription.
  - Off: diarization is skipped.
- **Analyze visuals (slides/chat OCR, beta)**:
  - On: OCR stage runs on video frames.
  - Off: OCR stage is skipped.

Qt automatically derives run mode:

- transcription + visuals -> `full`
- transcription only -> `transcribe_only`
- visuals only -> `visual_only`

If both transcription and visuals are off, PyScribe blocks run start.

### Diarization Controls

- **Mode**: backend selector for diarization engine.
- **Max Speakers**: optional speaker cap (blank = auto).
- Diarization progress bar:
  - Hidden/disabled when diarization is off.
  - Shows indeterminate state during long diarization operations.

### Visual Analysis Controls

- **Mode**:
  - `fast` (fewer OCR calls, fastest)
  - `balanced` (default)
  - `accurate` (most OCR coverage)
- **OCR Backend**:
  - `paddleocr`
  - `surya`
  - `pytesseract`
  - `auto` (best available fallback)
- **Sample every (sec)**:
  - Lower values = more frame coverage, slower runtime.
  - Clamped to `0.5` to `10.0`.

Fallback behavior:

- If selected OCR backend is unavailable, Qt offers fallback options where possible.
- For backend first-use downloads (for example PaddleOCR model files), Qt asks for confirmation.

### Job Controls

- **Process File**: start run.
- **Cancel**: cooperative cancellation.
- **Force Stop**: immediate process termination if cancellation stalls.
- **Exit**: close app.

### Output Controls

- **Transcript panel**: live transcript text output.
- **Copy**: copy transcript panel text to clipboard.
- **Save menu**:
  - **Save All (Transcript + OCR)**
  - **Save Transcript Only**
  - **Save OCR Only**
- Save dialog defaults to source media folder when available.
- Last open/save directories are remembered.

### Status + Timing

- Main status label shows current stage and results.
- Progress bars:
  - transcription progress
  - diarization progress
- Timing labels:
  - transcription time
  - diarization time
  - visual analysis time
- Hardware metrics label includes CPU/RAM and GPU/VRAM when available.

### Menus

### Tools

- **HF Token...**
  - Save Hugging Face token for gated/private model access.
- **Benchmark...**
  - Benchmark selected models using bundled sample audio.

### Help

- **PyScribe Help**
- **Model Help**
- **Open Logs Folder**
- **About PyScribe**

### Language Handling

- Language auto-detection is attempted before run.
- For `.en` models with non-English detected audio, Qt prompts for confirmation.
- If detection fails, run continues with model auto behavior.

## 4) Gradio Listener Features

### Server Launch Behavior

- Binds to `127.0.0.1` by default.
- If preferred port is unavailable, listener tries fallback ports.
- Queue is enabled with concurrency limit 1 (serialized jobs per host process).

### Listener UI Inputs

- **Upload Audio/Video File**
- **Select Transcription Model** (editable)
- **Run mode**
  - `full`
  - `transcribe_only`
  - `visual_only`
- **Identify speakers** + diarization controls
- **Analyze visuals** + visual mode/backend/sample interval controls

Visibility of controls adapts to run mode and toggles.

### Listener Actions

- **Transcribe**: starts run.
- **Cancel** (shown during active run): sets cancellation flag.
- **Copy to Clipboard**
- **Save Transcript**: prepares downloadable text file.

### Listener Outputs

- **Status**
- **Transcription**
- **Final status** (completion/cancel summary)
- **Download Transcript** file output

## 5) CLI Features

### Main Entry

```bash
python main.py
python main.py qt
python main.py serve [options]
```

### `serve` Options

- `--host` (default `127.0.0.1`)
- `--port` (default `7860`)
- `--max-port-tries` (default `50`)
- `--queue-size` (default `16`)
- `--auth-user` (optional username)
- `--allow-nonlocal-host` (required for non-local bind)
- `--share` (Gradio public share link)

### Listener Security Rules

- Non-local bind is rejected unless `--allow-nonlocal-host` is set.
- Non-local bind also requires auth credentials.
- Password must be supplied by environment variable (`PYSCRIBE_AUTH_PASS`).
- Legacy `--auth-pass` CLI argument is intentionally rejected.

## 6) Model Download + Auth Features

- Cached model reuse is automatic.
- Qt mode prompts before downloading uncached model repos.
- Listener mode downloads as needed (with progress/status updates).
- HF token sources:
  1. `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`
  2. saved token via Hugging Face cache

## 7) Logging Features

PyScribe configures rotating logs with safe writable fallbacks.

Log path priority:

1. `PYSCRIBE_LOG_DIR/pyscribe.log` (if set)
2. `~/.pyscribe/logs/pyscribe.log`
3. `./.pyscribe_logs/pyscribe.log`
4. OS temp fallback

Logging environment variables:

- `PYSCRIBE_LOG_LEVEL` (default `INFO`)
- `PYSCRIBE_LOG_STDOUT` (`1` enables stdout logging)
- `PYSCRIBE_LOG_DIR` (custom directory)

## 8) Runtime Environment Features

PyScribe sets defaults for writable caches and runtime compatibility.

Common environment variables:

- `PYSCRIBE_CACHE_DIR`
- `PADDLE_HOME`
- `PADDLE_PDX_CACHE_HOME`
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`
- `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT`
- `PADDLE_PDX_MODEL_SOURCE`
- `PADDLE_PDX_HUGGING_FACE_ENDPOINT`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `MODELSCOPE_CACHE`
- `XDG_CACHE_HOME`
- `FFMPEG_PATH`

Listener-related variables:

- `PYSCRIBE_AUTH_USER`
- `PYSCRIBE_AUTH_PASS`
- `PYSCRIBE_ALLOW_NONLOCAL_HOST`
- `PYSCRIBE_HOST`, `PYSCRIBE_PORT`, `PYSCRIBE_MAX_PORT_TRIES`, `PYSCRIBE_QUEUE_SIZE` (used by `scripts/run_listener.sh`)

## 9) Benchmark Feature

Qt benchmark dialog supports:

- selecting multiple models
- selecting benchmark language (English/Spanish bundled sample)
- progress reporting
- cancellation

## 10) Troubleshooting Quick Checks

- Ensure `ffmpeg` is installed and in PATH.
- Confirm required Python deps are installed in active environment.
- For gated/private models, configure HF token and accept model terms.
- For OCR backends, install runtime dependencies (`pytesseract` + OS package, or PaddleOCR stack).
- If Linux dynamic library issues occur, run from a clean shell and avoid conflicting injected library paths.
