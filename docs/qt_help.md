# PyScribe Qt Help

## Quick Start

1. Click **Browse** (or drag and drop a media file) to select audio/video.
2. Pick a model in the **Model** dropdown.
3. Click **Transcribe**.
4. Save or copy the transcript when complete.

## Main Workflow

- **Browse / Drop zone**: choose a local media file.
- **Model**: supports built-in model names and custom Hugging Face repo IDs (`owner/repo`).
- **Speaker Identification**:
  - Turn on to enable diarization (speaker labels).
  - Choose backend mode and optional max speaker count.

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

- **Transcribe**: starts a new job.
- **Cancel**: cooperative stop (safe).
- **Force Stop**: immediate stop if cancel is stuck.
- **Save**: write transcript to a text file.
- **Copy**: copy transcript to clipboard.
- **Open Folder**: open selected media folder.

## Troubleshooting

- **No output / error dialog**
  - Confirm FFmpeg is installed and available in PATH.
- **Gated diarization errors**
  - Verify HF token and model access terms.
- **GPU issues**
  - Retry with a smaller model or disable diarization.

## Logging

- Log file path defaults to:
  - `%USERPROFILE%\\.pyscribe\\logs\\pyscribe.log`
- Optional environment overrides:
  - `PYSCRIBE_LOG_LEVEL=DEBUG`
  - `PYSCRIBE_LOG_STDOUT=1`
  - `PYSCRIBE_LOG_DIR=<custom path>`
