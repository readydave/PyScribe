# PyScribe Qt Help

## Quick Start

1. Click **Browse** (or drag and drop a media file) to select audio/video.
2. Pick a model in the **Model** dropdown (built-in or custom `owner/repo`).
3. Choose processing options (**Transcribe audio**, **Speaker Identification**, **Analyze visuals**).
4. Click **Process File**.
5. Save or copy output when complete.

## Main Window

- **Browse / Drop zone**: choose a local media file.
- **Model**: supports built-in model names and custom Hugging Face repo IDs.
- **Recommended model label**: shows the hardware-based recommendation.
- **Status + metrics**:
  - Run status messages
  - HF token status (`configured` / `not configured`)
  - CPU/RAM and GPU/VRAM telemetry when available
- **Progress + timing**:
  - Transcription progress bar and elapsed time
  - Diarization progress bar and elapsed time (when enabled)
  - Visual analysis elapsed time (when enabled)

## Processing Options

- **Transcribe audio**:
  - Enables/disables ASR transcription stage.
- **Speaker Identification**:
  - Enables/disables diarization (speaker labels).
  - Choose backend mode and optional max speakers (blank = auto).
- **Analyze visuals (slides/chat OCR, beta)**:
  - Optional video-frame OCR to capture on-screen text.
  - Choose visual mode: `fast`, `balanced`, `accurate`.
  - Choose OCR backend: `paddleocr`, `surya`, `pytesseract`, or `auto`.
  - Set sample interval in seconds (`0.5` to `10.0`; lower = more coverage, slower runtime).
  - Backend/model downloads may be prompted on first use.

Run mode is inferred automatically:

- transcription + visuals -> `full`
- transcription only -> `transcribe_only`
- visuals only -> `visual_only`
- both off -> run is blocked (you must enable at least one)

## Tools Menu

- **HF Token...** (`Ctrl+Shift+T`)
  - Save a Hugging Face token for gated/private model access.
  - If gated diarization still fails, accept terms on the model page in Hugging Face.
- **Benchmark...** (`Ctrl+B`)
  - Compare selected models using bundled benchmark audio.
  - Supports English and Spanish sample sets.
  - Allows multi-model selection with Start/Cancel controls.
- **LLM Connections...** (`Ctrl+Shift+L`)
  - Add/edit local or LAN LLM profiles (Ollama or OpenAI-compatible APIs).
  - Run staged connection tests with pass/fail diagnostics and suggested fixes.
- **LLM Post-Process...** (`Ctrl+Shift+P`)
  - Run prompt-template-based post-processing on the current transcript.
  - Optionally include current OCR context, additional notes, and pasted context.
  - Includes payload preview so you can review exactly what will be sent.
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
- **Cancel**: cooperative stop (safe).
- **Force Stop**: immediate stop if cancel is stuck.
- **Save** (dropdown, default action = Save All):
  - **Save All (Transcript + OCR)**
  - **Save Transcript Only**
  - **Save OCR Only**
- **Copy**: copy transcript to clipboard.
- **Open Folder**: open selected media folder (or last-open folder when no file selected).
- **Exit**: close app (blocked while a job is running).

## Model and Download Notes

- If a model is not cached, PyScribe estimates download size and asks before downloading.
- Size estimates are best-effort and may differ from actual transfer size.
- Private/gated models may require:
  1. configured HF token, and
  2. accepted model terms on Hugging Face.

## Troubleshooting

- **No file selected**
  - Select a media file first.
- **No task selected**
  - Enable at least one of **Transcribe audio** or **Analyze visuals**.
- **No output / error dialog**
  - Confirm FFmpeg is installed and available in PATH.
- **Gated diarization errors**
  - Verify HF token and model access terms.
- **GPU issues**
  - Retry with a smaller model or disable diarization.
- **Visual analysis unavailable**
  - Install OCR runtime (`pytesseract` + OS package `tesseract-ocr`), or configure another backend.
- **Post-process blocked while transcription is running**
  - Local profiles are blocked during active local transcription to avoid GPU/compute contention.
  - Use a LAN profile with concurrent mode enabled, or wait for transcription to finish.
- **Template editing**
  - Built-in templates are read-only.
  - Create/edit/delete custom templates from within the LLM Post-Process dialog.

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
