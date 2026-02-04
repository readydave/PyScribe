# PyScribe Qt Help

## Quick Start

1. Click **Browse** (or drag and drop a media file) to select audio/video.
2. Pick a model in the **Model** dropdown.
3. Choose processing options (**Transcribe audio**, **Identify Speakers**, **Analyze visuals**).
4. Click **Process File**.
5. Save or copy the transcript when complete.

## Main Workflow

- **Browse / Drop zone**: choose a local media file.
- **Model**: supports built-in model names and custom Hugging Face repo IDs (`owner/repo`).
- **Transcribe audio**:
  - Enables/disables ASR transcription stage.
  - If disabled while visual analysis is enabled, the run becomes visual-only.
- **Speaker Identification**:
  - Turn on to enable diarization (speaker labels).
  - Choose backend mode and optional max speaker count.
- **Analyze visuals (slides/chat OCR, beta)**:
  - Optional video-frame OCR to capture on-screen text from slides/chats.
  - OCR prioritizes the shared-content area and also scans the right-side panel for chat-like text.
  - Choose visual mode: `fast`, `balanced`, `accurate`.
  - Choose OCR backend: `paddleocr`, `surya`, `pytesseract`, or `auto`.
  - `paddleocr` may download model files on first run (Qt prompts for confirmation).
  - `surya` is experimental and may require a separate environment with newer Torch versions.
  - Output reports show requested backend and actual backend used (including fallback reason).
  - PyScribe skips OCR on unchanged frames to reduce runtime.
  - Set sample interval (seconds): lower values capture more changes but increase runtime.

## Tools Menu

- **HF Token...**
  - Save a Hugging Face token for gated/private diarization models.
- **Benchmark...**
  - Compare model speed against bundled sample audio.

## Help Menu

- **PyScribe Help**
  - Opens this help guide.
- **Model Help**
  - Details on custom model repo IDs and download behavior.
- **Open Logs Folder**
  - Opens the folder that contains `pyscribe.log`.
- **About PyScribe**
  - Basic application information.

## Model and Download Notes

- If a model is not cached, PyScribe estimates download size and asks before downloading.
- Size estimates are best-effort and may differ from actual transfer size.
- Private/gated models may require:
  1. configured HF token, and
  2. accepted model terms on Hugging Face.

## Transcription Controls

- **Process File**: starts a new job.
- **Cancel**: cooperative stop (safe).
- **Force Stop**: immediate stop if cancel is stuck.
- **Save** (dropdown, default action = Save All):
  - **Save All (Transcript + OCR)**: one file containing transcript and visual/OCR report.
  - **Save Transcript Only**: transcript text only.
  - **Save OCR Only**: visual/OCR report only (shown when Analyze visuals was enabled and produced output).
- **Copy**: copy transcript to clipboard.
- **Open Folder**: open selected media folder.

## Troubleshooting

- **No output / error dialog**
  - Confirm FFmpeg is installed and available in PATH.
- **No task selected**
  - Enable at least one of **Transcribe audio** or **Analyze visuals**.
- **Gated diarization errors**
  - Verify HF token and model access terms.
- **GPU issues**
  - Retry with a smaller model or disable diarization.
- **Visual analysis unavailable**
  - Install OCR runtime (`pytesseract` + OS package `tesseract-ocr`).
  - For packaged builds (AppImage), OCR model cache defaults to `~/.cache/pyscribe`; override with `PYSCRIBE_CACHE_DIR`.

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
