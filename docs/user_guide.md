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

### Layout + Navigation

- The main Qt window uses a left navigation sidebar and a right content stack:
  - **Transcription**
  - **LLM**
  - **Settings**
- **New Project** returns to the Transcription screen.
- Left sidebar can be collapsed/expanded with the small toggle button in the sidebar header.
- Transcription screen includes a right-side status rail that can also be hidden/shown from the page header.
- The app now opens sized to the available screen area and remains fully resizable.

### File Selection

- **Browse Files** (inside drop zone): open file picker for media.
- **Clickable Drop Zone**: clicking anywhere within the dashed drop area will also open the file browser.
- **Drag-and-drop zone**: drop a media file directly.
- **Open Folder**: open selected file directory (or last-used folder if no file selected).

### Input Modes

- **Input** selector:
  - `File`: existing file-based transcription workflow.
  - `Live`: Qt-only live capture workflow (Linux-first).
- Live mode hides the drop zone and disables visual OCR controls.
- Live mode supports one source per session:
  - **Microphone**
  - **Loopback**
- Loopback requires the OS to expose a monitor/loopback input device. On Linux this is typically a PipeWire/PulseAudio monitor source.
- Each live session writes into `~/PyScribe Live Sessions` by default unless you choose another output folder.
- Granite Speech is blocked in live mode because live mode requires timestamp-capable Whisper backends.

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

In live mode:

- transcription is always on
- visual analysis is always off
- speaker identification can still be enabled, but it only runs after **Stop** during the final post-pass on the saved capture file

### Live Capture Controls

- **Source**: choose microphone or loopback capture.
- **Device**: pick the matching Qt audio input for the selected source type.
- **Output Folder**: root folder for live session subfolders.
- **Keep recorded audio after completion**:
  - On: keep `capture.wav` after a successful final pass.
  - Off: remove `capture.wav` after a successful final pass, but keep the session folder, `session.json`, and `final_transcript.txt`.
- **Timer**: elapsed recorded time for the current session. It freezes while live capture is paused.
- Each live session folder contains:
  - `capture.wav`
  - `session.json`
  - `final_transcript.txt` after a successful stop/finalize cycle

### Diarization Controls

- **Mode**: backend selector for diarization engine.
  - `Accurate` (pyannote): default high-quality engine.
  - `Fast` (pyannote): faster variant.
- **Max Speakers**: optional speaker cap (blank = auto).
- Pyannote diarization backends run in a separate worker process so GPU speaker ID can stay isolated from CUDA ASR runtime state.
- If GPU diarization is unavailable, PyScribe retries diarization on CPU before giving up on speaker labels.
- On modern Torchaudio releases, PyScribe uses `soundfile` fallbacks for metadata/loading APIs that pyannote expects.
- If diarization fails or produces no speaker segments, PyScribe keeps the plain transcript instead of filling the output with `[S?]` speaker labels.
- Diarization progress bar:
  - Disabled when transcription is off.
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
- **Start Live**: begin live microphone or loopback capture.
- **Pause / Resume** (live mode): temporarily suspend or resume live capture while keeping the same session folder and `capture.wav`.
- **Stop** (live mode): stop capture cleanly, finalize the rolling draft, and start the final post-pass on the saved recording.
- **Cancel**: cooperative cancellation.
  - In live mode, cancel asks for confirmation, then stops capture immediately, skips the final post-pass, and preserves the session folder and recorded audio.
- **Force Stop**: immediate process termination if cancellation stalls; Qt escalates from terminate to kill when needed.
  - In live mode, force stop preserves the live session folder and recorded audio.
- **Exit**: close app.

### Output Controls

- **Transcript panel**: live transcript text output.
  - In live mode, this shows a rolling draft first, then the final cleaned transcript after **Stop** completes.
- **Copy**: copy transcript panel text to clipboard.
- **Save menu**:
  - **Save All (Transcript + OCR)**
  - **Save Transcript Only**
  - **Save OCR Only**
- Save dialog defaults to source media folder when available.
- Last open/save directories are remembered.

### Status + Timing

- Main status label shows current stage and results.
- Transcription view includes a terminal-style live event log panel.
  - Live mode logs device selection, recording start/pause/resume/stop, final post-pass handoff, and preserved session paths on cancel/failure.
- HF token status label shows whether a token is configured.
- Progress bars:
  - transcription progress
  - diarization progress
- Timing labels:
  - transcription time
  - diarization time
  - visual analysis time
- Hardware metrics label includes CPU/RAM and GPU/VRAM when available.

### Responsive Behavior

- General and Advanced settings render in:
  - two columns on wider window sizes
  - one stacked column on narrower sizes
- Right status rail auto-hides on narrow windows and can be manually toggled.
- Both transcription center content and settings page content use scroll areas for smaller displays.

### Menus

Qt menu bar includes **Tools**, **View**, and **Help**.

### Tools

- **HF Token...**
  - Save Hugging Face token for gated/private model access.
  - Shortcut: `Ctrl+Shift+T`
- **Benchmark...**
  - Benchmark selected models using bundled sample audio.
  - Shortcut: `Ctrl+B`
- **LLM Connections...**
  - Configure enabled local/LAN LLM profiles.
  - Supports `ollama`, `lm_studio`, and OpenAI-compatible endpoints.
  - API key field supports `env:VAR_NAME` references for secure persisted config.
  - Direct API keys are treated as session-only and are not written to disk.
  - Includes subnet detection and LAN scan utilities to discover reachable local-network endpoints.
  - Shortcut: `Ctrl+Shift+L`
- **LLM Post-Process...**
  - Run prompt-template post-processing against the current transcript/OCR context.
  - Shortcut: `Ctrl+Shift+P`
- **Process Existing Transcript...**
  - Open post-processing workflow for previously saved transcript files.
  - This is useful for offline or delayed summarization/review.

### View

- **Theme**
  - `System`
  - `Light`
  - `Dark`
  - Theme preference is persisted across launches.

### Help

- **PyScribe Help**
  - Shortcut: Help key / `F1`
- **Model Help**
- **Open Logs Folder**
- **About PyScribe**

### Language Handling

- Language auto-detection is attempted before run.
- For `.en` models with non-English detected audio, Qt prompts to force English or cancel.
- For non-English detected audio on non-`.en` models, Qt prompts to use detected language or force English.
- If detection fails, run continues with model auto behavior.
- Live mode uses model auto language behavior during rolling ASR and then reprocesses the saved capture during the final post-pass.

### LLM Post-Processing Workflow (Qt)

- Open **Tools > LLM Connections...** and configure at least one enabled profile.
- Use **Detect Networks** and **Scan Selected Network** to find reachable endpoints on detected subnets.
- In multi-network environments (for example LAN + VPN), pick the target detected subnet before scanning.
- Optionally run **Test Connection** to validate endpoint reachability/auth/model discovery.
- Scope policy is enforced at run time (not only during connection tests).
- Open **Tools > LLM Post-Process...** for the current transcript, or
  **Tools > Process Existing Transcript...** to load a saved transcript file.
- The dialog uses a split workspace:
  - Left: configuration + attachments
  - Right: input context, payload preview, and output panes
- Select profile, template, and model, then run post-processing.
- You can create/edit/delete custom user templates in the same dialog (built-ins remain read-only).
- Use **Pasted Context**, optional image attachments, and **Payload Preview** to review exactly what will be sent before execution.
- **Cancel Generation** now prompts for confirmation and immediately requests cancellation.
- Closing the dialog during active generation prompts to cancel before the window closes.
- For image attachments:
  - Multimodal-capable models receive image content directly.
  - Text-only models can use OCR fallback to convert image context into text.
- Concurrency policy:
  - Local profiles are blocked while local transcription is in progress.
  - LAN profiles may run concurrently only if profile concurrent mode is enabled.

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
- **LLM Post-Processing (Beta)**:
  - Pick configured LLM profile + prompt template.
  - Test connection and fetch model list.
  - Choose transcript source (`Current transcript` or `Upload/paste transcript`).
  - Optionally upload OCR/context text, add extra notes, include pasted context, and attach images.
  - Enable/disable image include and OCR fallback behavior for text-only models.
  - Preview final request payload before sending to the configured model.
  - Run post-processing and save/copy generated output.

### Listener Outputs

- **Status**
- **Transcription**
- **Final status** (completion/cancel summary)
- **Download Transcript** file output
- **LLM status**
- **LLM output**
- **Download LLM Output** file output

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
- `--share` also requires auth credentials, even on localhost binds.
- Password must be supplied by environment variable (`PYSCRIBE_AUTH_PASS`).
- Interactive LAN mode uses password from `PYSCRIBE_LAN_AUTH_PASS` or secure prompt input.
- Legacy `--auth-pass` CLI argument is intentionally rejected.

## 6) Model Download + Auth Features

- Cached model reuse is automatic.
- Qt mode prompts before downloading uncached model repos.
- Listener mode downloads as needed (with progress/status updates).
- HF token sources:
  1. `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`
  2. saved token via Hugging Face cache

## 7) Logging Features

PyScribe uses a consolidated `pyscribe.log` file for the active session, with automatic timestamped archiving of previous logs on startup.

- **Consolidated Logs**: the active session always logs to a single file named `pyscribe.log`.
- **Session Archiving**: on application startup, the previous `pyscribe.log` is automatically moved to a timestamped file (for example `pyscribe_20260428_130148.log`).
- **Automatic Rotation**: the system automatically scans the log directory on startup and keeps only the **21 most recent** archived log files to manage disk space.

Log directory priority:

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
- For Qt live loopback capture on Linux, confirm your audio stack exposes a monitor/loopback input.
- For OCR backends, install runtime dependencies (`pytesseract` + OS package, or PaddleOCR stack).
- If Linux dynamic library issues occur, run from a clean shell and avoid conflicting injected library paths.
