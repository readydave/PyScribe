# PyScribe User Guide

This guide explains every user-facing feature in PyScribe.

## 1) Launch Modes

PyScribe has two frontends:

- **Qt Desktop** (`python main.py qt`)
- **Gradio Listener** (`python main.py serve`)

If you run `python main.py` with no mode, you get an interactive launcher menu. If you do not choose within 5 seconds, PyScribe starts Desktop (Qt) automatically.

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

- The main Qt window uses a left navigation sidebar and a content stack:
  - **Transcription**
  - **LLM**
  - **Settings**
- **New Project** clears the current file, transcript and progress and returns to the Transcription screen (it asks first if there is a transcript, and is blocked while a job runs). Settings and the batch queue are kept.
- Left sidebar can be collapsed/expanded with the small toggle button in the sidebar header. The choice is remembered.
- The Transcription screen is built from movable panels around the transcript:
  - **Setup** (File/Live switch, drop zone, model, **More options**), **Progress**, **Hardware**, and **Batch queue**.
  - Drag a panel by its title bar to move, tab, or float it; drag the visible divider bars between panels (they highlight on hover) to resize them, also while **Lock layout** is on; the title-bar buttons float or close a panel.
  - **View > Panels** re-opens closed panels, **Lock layout** stops accidental moves, and **Reset layout** restores the default.
  - The layout (including panel sizes), window size, and sidebar state are saved shortly after you change them and when you close the app. A narrow Setup panel scrolls sideways instead of clipping its controls. On windows narrower than 1500 px the Hardware panel starts as a tab beside Setup.
- **Progress** shows one row per stage (Transcribe, Speakers, Visuals) with state and elapsed time; **Details** opens the event log.
- **Hardware** shows 60-second CPU, memory, GPU, and VRAM traces while a job runs, coloured by the running stage.
- Speaker labels in the transcript are colour-coded per speaker. While recording live, the timer and Stop button turn red.
- The app opens sized to the available screen area and remains fully resizable.

### File Selection

- **Browse Files** (inside drop zone): open file picker for media.
- **Clickable Drop Zone**: clicking anywhere within the dashed drop area will also open the file browser.
- **Drag-and-drop zone**: drop a media file directly.
- **Open Folder**: open selected file directory (or last-used folder if no file selected).

### Batch Queue

- Add individual media files or import a folder to queue multiple files for serialized processing.
- Files with the same basename can be queued when they come from different folders.
- Exact duplicate paths are skipped.
- When duplicate basenames are present, the queue display includes parent-folder context so items remain distinguishable.

### Input Modes

- **File / Live** switch (top of the Setup panel):
  - `File`: existing file-based transcription workflow.
  - `Live`: Qt-only live capture workflow (Linux-first).
- Live mode hides the drop zone and disables visual OCR controls.
- Live mode supports one source per session:
  - **Microphone**
  - **Loopback**
- Loopback requires the OS to expose a monitor/loopback input device. On Linux this is typically a PipeWire/PulseAudio monitor source.
- Each live session writes into `~/PyScribe Live Sessions` by default unless you choose another output folder.
- Live mode accepts timestamp-capable Whisper models and `nvidia/nemotron-speech-streaming-en-0.6b` (English only, streams text as you speak; the final pass re-runs Nemotron on the capture and adds speaker labels). Granite Speech is blocked in live mode.

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
- **Session Title**: optional title used for live session file naming; it is editable before capture starts and after completion for rename/apply-title workflows, but locked while capture/finalization is active.
- **Keep recorded audio after completion**:
  - On: keep the timestamped `YYYY-MM-DD_HHMMSS-live-capture.wav` file after a successful final pass.
  - Off: remove the saved live capture audio after a successful final pass, but keep the session folder, `session.json`, and `final_transcript.txt`.
- **Timer**: elapsed recorded time for the current session. It freezes while live capture is paused.
- **Compute**: `Auto` (default, uses the GPU when CUDA is available), `CPU`, or `GPU`. It applies to the live session and its final post-pass, and is remembered between runs. If `GPU` is chosen without CUDA, PyScribe falls back to CPU and says so in the event log.
- **Precision**: `Auto`, `float16`, or `int8`. `float16` is GPU-only (CPU uses `int8`). The option is hidden for Nemotron, which always uses float16 on GPU and float32 on CPU.
- When Compute resolves to the GPU, PyScribe checks currently free GPU memory before live capture; choosing `CPU` skips this check. If free VRAM appears too low for the selected model, it warns before starting and suggests unloading LM Studio, reducing GPU layers, choosing a smaller Whisper model, or switching to CPU/int8.
- Each live session folder contains:
  - `YYYY-MM-DD_HHMMSS-live-capture.wav`
  - `session.json`
  - `final_transcript.txt` after a successful stop/finalize cycle

### Diarization Controls

- **Mode**: backend selector for diarization engine.
  - `Accurate` (pyannote): the diarization engine (the former `Fast` mode was identical and has been removed).
- **Max Speakers**: optional speaker cap (blank = auto).
- Pyannote diarization backends run in a separate worker process so GPU speaker ID can stay isolated from CUDA ASR runtime state.
- If GPU diarization is unavailable, PyScribe retries diarization on CPU before giving up on speaker labels.
- On modern Torchaudio releases, PyScribe uses `soundfile` fallbacks for metadata/loading APIs that pyannote expects, and passes decoded audio to pyannote in memory so `torchcodec`/FFmpeg are not needed for file IO.
- With pyannote.audio 4.x, diarization uses `pyannote/speaker-diarization-community-1` (CC-BY-4.0, gated): accept its terms on Hugging Face with the account that owns your token. When a transcript has word timestamps, speakers are assigned per word, so a speaker change in the middle of a sentence starts a new line.
- If diarization fails or produces no speaker segments, PyScribe keeps the plain transcript instead of filling the output with `[S?]` speaker labels.
- Diarization progress bar:
  - Disabled when transcription is off.
  - Uses staged determinate progress for backend initialization, model loading, inference, speaker assignment, and completion.

### Visual Analysis Controls

- **Mode**:
  - `fast` (fewer OCR calls, fastest)
  - `balanced` (default)
  - `accurate` (most OCR coverage)
- **OCR Backend**:
  - `rapidocr`
  - `paddleocr`
  - `surya`
  - `pytesseract`
  - `auto` (best available fallback; PaddleOCR leads only when it will run on the GPU, otherwise RapidOCR)
- **PaddleOCR on GPU**: `requirements.txt` installs the CPU `paddlepaddle` build (about 5 s per frame). Run `scripts/install_paddle_gpu.sh` to install the CUDA build instead (about 0.08 s per frame, about 1.2 GB VRAM); it shares the CUDA libraries that torch already installs. Re-run the script after any `pip install -r requirements.txt`. PaddleOCR uses the GPU only when at least 1.5 GB of VRAM is free and falls back to CPU (at initialization or on a run-time GPU error) otherwise.
- **OCR Fallback**: which backend to try first when the main one cannot run (for example PaddleOCR cannot verify its models because Hugging Face is unreachable). `auto` (default) keeps the built-in order; or pick `rapidocr`, `pytesseract`, or `surya`. A fallback that is not installed is skipped and the next available backend is used. The report note says why the main backend was not used and which fallback ran. The choice is saved as soon as you change it.
- **Scope**:
  - `Slides only` (default; avoids noisy chat/meeting side panels)
  - `Slides + chat` (captures the right-side chat/panel crop when useful)
- **Sample every (sec)**:
  - Lower values = more frame coverage, slower runtime.
  - Clamped to `0.5` to `10.0`.
- Long videos use lower frame caps and prefer faster auto OCR backends to reduce webinar OCR runtime.

Fallback behavior:

- If selected OCR backend is unavailable, Qt offers fallback options where possible.
- For backend first-use downloads (for example PaddleOCR model files), Qt asks for confirmation.

### Job Controls

- **Process File**: start run.
- **Start Live**: begin live microphone or loopback capture.
- **Pause / Resume** (live mode): temporarily suspend or resume live capture while keeping the same session folder and timestamped live capture file.
- **Stop** (live mode): stop capture cleanly, finalize the rolling draft, and start the final post-pass on the saved recording.
  - After a successful final post-pass, live mode remains ready for another live session and restores the Live Capture controls.
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
- When two or more processing parts are enabled, PyScribe automatically saves separate part files beside the source media where possible:
  - `<stem>_transcript.txt`
  - `<stem>_diarized.txt`
  - `<stem>_ocr.txt`
  - Existing files are preserved with numbered suffixes.

### Status + Timing

- Main status label shows current stage and results.
- Transcription view includes a terminal-style live event log panel.
  - Live mode logs device selection, recording start/pause/resume/stop, final post-pass handoff, and preserved session paths on cancel/failure.
- HF token status label shows whether a token is configured.
- Progress bars:
  - transcription progress
  - diarization progress
  - visual analysis progress
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
  - Configure enabled local, LAN, and hosted (`cloud`) LLM profiles.
  - Supports `ollama`, `lm_studio`, OpenAI-compatible endpoints, and Anthropic (native Claude API).
  - **Add Cloud Profile** adds a starting point for Claude, OpenAI, Gemini, or OpenRouter. **Test Connection** lists the models your key can use; any model your provider offers can be the default.
  - Cloud profiles must use `https://`, need an API key and a ticked confirmation that transcripts and images are sent to the provider, and always verify TLS.
  - API key field supports `env:VAR_NAME` references for secure persisted config.
  - Direct API keys are treated as session-only and are not written to disk, unless you tick **Store in system keyring** (shown only when your system keyring is usable). The key then lives in the operating system keyring and the config file only holds an opaque `keyring:<id>` reference; the field shows "stored in system keyring" instead of the key.
  - A stored key is removed from the keyring only when you delete the profile, replace or clear the key, or switch the profile to `env:` or a session key, and only once you press **Save and Close** (Cancel keeps the previous state). If the keyring is locked or missing, PyScribe says so and does not save the key anywhere else.
  - **Test Connection** (here and in LLM Post-Process) runs in the background, so the window stays responsive; the button is disabled while it runs, and you can close the window at any time.
  - **Context tokens**, **Max output tokens**, and **Temperature** are per profile (empty = automatic or provider default).
  - Includes subnet detection and LAN scan utilities to discover reachable local-network endpoints.
  - Shortcut: `Ctrl+Shift+L`
- **LLM Post-Process...**
  - Run prompt-template post-processing against the current transcript/OCR context.
  - Shortcut: `Ctrl+Shift+P`
- **Process Existing Transcript...**
  - Open post-processing workflow for previously saved transcript files.
  - This is useful for offline or delayed summarization/review.

### View

- **Panels**
  - Toggle each panel, **Lock layout**, and **Reset layout** (see Layout + Navigation).
- **Theme**
  - `System`
  - `Light`
  - `Dark`
  - Theme preference is persisted across launches.
- **Colour theme**
  - Four presets (Iron-gall, Verdigris, Ochre, Graphite), each with light and dark versions, plus your own.
  - **Edit themes...** opens the editor: click a swatch to change a colour with live preview; text colours that would be hard to read are adjusted automatically; presets can't be changed, so editing one saves a copy; custom themes can be duplicated, renamed, deleted, exported, and imported (`.json`).
  - The Listener uses the same colour theme when it starts.

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
  - Cloud profiles do not compete for local compute and may run at any time.
- Long transcripts:
  - Each profile has a context size (automatic by default: 16k for Ollama, 8k for LM Studio and other local servers, large for hosted models).
  - A transcript that does not fit is split into parts, each part is processed, and the results are merged; the status line shows progress.
  - A reply that stops at the output limit is flagged so you know it may be cut off.
- The Listener hides cloud profiles unless `llm_allow_cloud_in_listener` is `true` in `~/.pyscribe_config.json` (anyone who can open the Listener could otherwise use your key).

### Settings and start-up defaults

- **Settings > Start-up defaults**: default model (empty = last model used), start in File or Live, names/terms, faster GPU decoding, and the folder file dialogs open in. **Use current settings as defaults** copies the Transcription page's current choices.
- Remembered automatically: run mode, speaker and visual options (saved when a job starts), live capture choices, theme and colour theme, panel layout, window size, and sidebar state.
- **Settings > AI connections** opens the LLM Connections dialog. API keys are set per connection.

### MCP server

- `python main.py mcp` runs PyScribe as an MCP server over stdio so Claude Code, Codex, and other MCP clients can transcribe audio and read transcripts. See `docs/mcp.md`.

## 4) Gradio Listener Features

### Server Launch Behavior

- Binds to `127.0.0.1` by default.
- If preferred port is unavailable, listener tries fallback ports.
- Queue is enabled with concurrency limit 1 (serialized jobs per host process).

### Listener Text Size

- A small **A− 100% A+** widget is fixed to the top-right corner of the page.
- **A−** / **A+** change the text size from 80% to 200% in 10% steps; clicking the percentage resets to 100%.
- The setting applies to labels, inputs, transcript boxes, and descriptive text, and is stored in the browser (`localStorage`), so it is per browser and per device, not per server. If the browser blocks storage, the size resets to 100% on reload.
- It is separate from the browser's own zoom (Ctrl/Cmd `+` / `-`), which still works and stacks with it.
- Implementation note: Gradio defines several derived font-size variables as fixed pixel values, so `CUSTOM_CSS` in `app.py` scales those directly. After a Gradio upgrade, re-check that all text still scales.

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
- **Stage strip** (Load model / Transcribe / Speakers / Visuals chips under Status; shows each stage as waiting, running with a percent, done, failed, or off when that step is not used in the chosen mode; it follows the Listener colour theme in light and dark)
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
