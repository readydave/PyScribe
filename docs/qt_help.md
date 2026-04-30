# PyScribe Qt Help

## Quick Start

1. Use **Browse Files** in the drop zone (or drag and drop a media file) to select audio/video.
2. Pick a model in the **Model** dropdown (built-in or custom `owner/repo`).
3. Choose **Input = File** for the traditional workflow, or **Input = Live** for microphone/loopback capture.
4. For file mode, choose processing options (**Transcribe audio**, **Speaker Identification**, **Analyze visuals**) and click **Process File**.
5. For live mode, choose source/device/output folder and click **Start Live**. Use **Pause / Resume** when needed, then **Stop** to run the final post-pass.
6. Save or copy output when complete.

## Main Window

- **Navigation sidebar**: switch between **Transcription**, **LLM**, and **Settings** screens.
- **Sidebar toggle**: collapse/expand the left navigation area.
- **Status panel toggle**: hide/show the right status rail on the transcription page.
- **Drop zone**: choose a local media file via drag/drop or **Browse Files**.
- **Model**: supports built-in model names and custom Hugging Face repo IDs.
- **Input**:
  - `File`: normal file transcription flow.
  - `Live`: Linux-first microphone/loopback capture flow.
- **Recommended model label**: shows the hardware-based recommendation.
- **Status + metrics**:
  - Run status messages
  - HF token status (`configured` / `not configured`)
  - CPU/RAM and GPU/VRAM telemetry when available
- **Progress + timing**:
  - Transcription progress bar and elapsed time
  - Diarization progress bar and elapsed time (when enabled)
  - Visual analysis elapsed time (when enabled)
- **Live pipeline log**:
  - Terminal-style read-only event feed for real-time stage updates.
- **Responsive cards**:
  - General/Advanced settings show in two columns on wide windows and collapse to one column on narrow windows.

## Processing Options

- **Transcribe audio**:
  - Enables/disables ASR transcription stage.
  - In live mode, transcription is always on.
- **Speaker Identification**:
  - Enables/disables diarization (speaker labels).
  - Choose backend mode and optional max speakers (blank = auto).
  - Pyannote diarization runs in a separate worker process so GPU speaker ID is isolated from CUDA ASR runtime state.
  - If CUDA diarization is unavailable, PyScribe retries diarization on CPU before dropping speaker labels.
  - If diarization produces no speaker segments, PyScribe keeps the plain transcript instead of writing `[S?]` labels for every line.
  - In live mode, diarization runs only after **Stop** during the final post-pass on the saved capture.
- **Analyze visuals (slides/chat OCR, beta)**:
  - Optional video-frame OCR to capture on-screen text.
  - Choose visual mode: `fast`, `balanced`, `accurate`.
  - Choose OCR backend: `paddleocr`, `surya`, `pytesseract`, or `auto`.
  - Set sample interval in seconds (`0.5` to `10.0`; lower = more coverage, slower runtime).
  - Backend/model downloads may be prompted on first use.
  - Live mode disables visual analysis.

## Live Capture

- **Source**:
  - `Microphone`
  - `Loopback`
- **Device**: audio input filtered to the selected source type.
- **Output Folder**: root for timestamped live session folders.
- **Session Title**: optional title used for live session file naming; editable before capture and after completion, locked while capture/finalization is active.
- **Keep recorded audio after completion**:
  - when enabled, the timestamped `YYYY-MM-DD_HHMMSS-live-capture.wav` file remains after a successful final pass
  - when disabled, successful runs keep metadata and final transcript but remove the raw capture
- **Timer**: elapsed recorded time for the active session. It freezes while capture is paused.
- CUDA live capture checks free GPU memory first and warns if another local workload, such as LM Studio, appears to leave too little VRAM for the selected model.
- Live mode writes a recoverable session folder containing:
  - `YYYY-MM-DD_HHMMSS-live-capture.wav`
  - `session.json`
  - `final_transcript.txt` after a successful stop/finalize cycle
- Granite is unavailable in live mode because live capture requires timestamp-capable Whisper models.
- Loopback capture depends on the OS exposing a monitor/loopback input. On Linux that usually means a PipeWire/PulseAudio monitor source.

Run mode is inferred automatically:

- transcription + visuals -> `full`
- transcription only -> `transcribe_only`
- visuals only -> `visual_only`
- both off -> run is blocked (you must enable at least one)

## Batch Queue

- Add files or import a folder to queue multiple media files.
- Same-named files from different folders can be queued together.
- Exact duplicate paths are skipped.
- Duplicate basenames show parent-folder context in the queue list.

## Tools Menu

- **HF Token...** (`Ctrl+Shift+T`)
  - Save a Hugging Face token for gated/private model access.
  - If gated diarization still fails, accept terms on the model page in Hugging Face.
- **Benchmark...** (`Ctrl+B`)
  - Compare selected models using bundled benchmark audio.
  - Supports English and Spanish sample sets.
  - Allows multi-model selection with Start/Cancel controls.
- **LLM Connections...** (`Ctrl+Shift+L`)
  - Add/edit local or LAN LLM profiles (`ollama`, `lm_studio`, `openai_compatible`).
  - API key supports `env:VAR_NAME` for secure persisted configuration.
  - Direct API key values are session-only and are not saved to disk.
  - Run staged connection tests with pass/fail diagnostics and suggested fixes.
  - Detect local interfaces/subnets and scan selected networks for reachable LLM endpoints.
  - Supports multi-network setups (for example LAN + VPN) by selecting a detected subnet before scan.
- **LLM Post-Process...** (`Ctrl+Shift+P`)
  - Run prompt-template-based post-processing on the current transcript.
  - Optionally include current OCR context, additional notes, pasted context, and image attachments.
  - Includes payload preview so you can review exactly what will be sent.
  - If the selected model appears text-only, image context can fall back to OCR text.
- **Process Existing Transcript...**
  - Open the same post-processing workflow, starting in file-load mode for saved transcript files.
  - Useful when transcription was completed in an earlier session.

## View Menu

- **Theme**
  - **System**, **Light**, **Dark**
  - Theme choice is saved and restored on next launch.

## Help Menu

- **PyScribe Help** (`F1` / Help key)
  - Opens this guide.
- **Model Help**
  - Quick guidance for custom model IDs and download behavior.
- **Open Logs Folder**
  - Opens the folder containing `pyscribe.log`.
- **About PyScribe**
  - Basic application and repository information.

## Language Prompts

Before transcription, PyScribe may detect language and prompt:

- For `.en` models with non-English detected audio: prompt to force English or cancel.
- For non-English detected audio on non-`.en` models: prompt to use detected language or force English.
- If detection fails, run continues with model auto behavior.

## Transcription Controls

- **Process File**: starts a new job.
- **Start Live**: begins live capture.
- **Pause / Resume**: temporarily suspend or resume live capture without creating a new session folder.
- **Stop**: stops live capture and starts the final post-pass on the saved recording.
  - After a completed final post-pass, live mode returns to a ready state for the next live session.
- **Cancel**: cooperative stop (safe).
  - In live mode, cancel asks for confirmation, then stops capture immediately and keeps the saved session folder without running the final post-pass.
- **Force Stop**: immediate stop if cancel is stuck; escalates to a hard kill if the worker does not exit cleanly.
  - In live mode, force stop preserves the session folder and recorded audio.
- **Save** (dropdown, default action = Save All):
  - **Save All (Transcript + OCR)**
  - **Save Transcript Only**
  - **Save OCR Only**
- **Copy**: copy transcript to clipboard.
- **Open Folder**: open selected media folder (or last-open folder when no file selected).
  - In live mode, this opens the active live session folder or configured live output root.
- **Exit**: close app (blocked while a job is running).

## LLM Post-Process Dialog

- Split layout:
  - Left: **Configuration** and **Attachments**
  - Right: **Input Context**, **Payload Preview**, **LLM Output**
- **Cancel Generation** asks for confirmation before stopping an in-flight request.
- If you close the dialog during generation, PyScribe asks whether to cancel first.
- Partial output is preserved when cancellation occurs after streaming has started.

## Model and Download Notes

- If a model is not cached, PyScribe estimates download size and asks before downloading.
- Size estimates are best-effort and may differ from actual transfer size.
- Private/gated models may require:
  1. configured HF token, and
  2. accepted model terms on Hugging Face.

## Troubleshooting

- **No file selected**
  - Select a media file first.
- **No loopback device**
  - Confirm your Linux audio stack exposes a monitor/loopback input.
  - Many PulseAudio/PipeWire setups label these devices with `Monitor` or `Loopback`.
- **No task selected**
  - Enable at least one of **Transcribe audio** or **Analyze visuals**.
- **No output / error dialog**
  - Confirm FFmpeg is installed and available in PATH.
- **Gated diarization errors**
  - Verify HF token and model access terms.
- **GPU issues**
  - Pyannote diarization runs in a separate process to avoid CUDA runtime conflicts with `faster-whisper`.
  - If GPU diarization still cannot start, PyScribe retries speaker ID on CPU automatically.
  - Modern Torchaudio metadata/loading compatibility is handled with `soundfile` fallbacks.
  - Retry with a smaller model or disable diarization if GPU memory is tight.
- **Visual analysis unavailable**
  - Install OCR runtime (`pytesseract` + OS package `tesseract-ocr`), or configure another backend.
- **Post-process blocked while transcription is running**
  - Local profiles are blocked during active local transcription to avoid GPU/compute contention.
  - Use a LAN profile with concurrent mode enabled, or wait for transcription to finish.
- **Policy check failures at run time**
  - Scope/CIDR policy is enforced when sending post-process requests, not only in connection tests.
  - For `127.0.0.1`/`localhost` endpoints, use `local` scope.
- **Template editing**
  - Built-in templates are read-only.
  - Create/edit/delete custom templates from within the LLM Post-Process dialog.
- **Image context fallback**
  - If image attachment send is blocked for a text-only model, enable OCR fallback and choose an OCR backend.

## Logging

- Log file target order:
  1. `PYSCRIBE_LOG_DIR/pyscribe.log` (if set)
  2. `~/.pyscribe/logs/pyscribe.log`
  3. `./.pyscribe_logs/pyscribe.log`
  4. OS temp fallback
- Optional environment overrides:
  - `PYSCRIBE_LOG_LEVEL=DEBUG`
  - `PYSCRIBE_LOG_STDOUT=1`
  - `PYSCRIBE_LOG_DIR=<custom path>`
