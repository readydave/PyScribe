# October 2026 accuracy and speed upgrade

Record of the runtime, transcription, and diarization upgrade done in October 2026: what changed, what was
measured, what was decided, and what is left. Numbers come from `scripts/evaluate.py` and
`scripts/accuracy_sweep.py` on one RTX 4090 machine.

## What changed (on `main`)

| Area | Before | After |
|---|---|---|
| Runtime | torch 2.5.1 + CUDA 12.1, NumPy 1.26, ctranslate2 4.6.1, pyannote.audio 3.1.1 | torch 2.11 + CUDA 12.8, NumPy 2, ctranslate2 4.8, pyannote.audio 4.x, `soundfile` now an explicit requirement |
| Speaker labels | Whole ASR segments got one speaker (maximum overlap) | Words are labelled individually and segments split at speaker changes (`services/speaker_assignment.py`) |
| Diarization model | `pyannote/speaker-diarization-3.1` | `pyannote/speaker-diarization-community-1` (gated, CC-BY-4.0); 3.1/3.0 remain fallbacks on pyannote 3.x |
| Diarization audio | pyannote decoded files itself | Audio is decoded with `soundfile` and passed in memory; real progress via a pyannote hook |
| Diarization modes | `Accurate` and `Fast` (identical) | `Accurate` only; saved `fast` settings map to `accurate` |
| ASR decoding | `model.transcribe(...)` inline | `services/asr_decode.py`: sequential by default, optional VRAM-sized batching, OOM fallback, word timestamps, `no_repeat_ngram_size=4` loop guard |
| Names and jargon | none | "Names / terms" field (Qt and Listener) passed as faster-whisper `hotwords` |
| Live mode | no VAD | Silero VAD on live windows (no hallucinated text on silence/noise; short speech kept) |
| Compatibility shims | torchaudio, `torch.load`, NumPy shims in `diarization.py` | removed (not needed on the new stack) |
| Evaluation | none | `scripts/evaluate.py`, `scripts/accuracy_sweep.py`, `scripts/build_ami_refs.py`, `services/eval_metrics.py`, `evaluation/` LibriVox manifest and references |
| CI | smoke tests only | also `pytest -q tests` in a light environment (modules needing torch/Qt/etc. are skipped when missing) |

## Results

Test material: two bundled LibriVox clips (English, Spanish; read speech, ~15 min each) and three public AMI meetings
(conversational, 4 speakers, 14-25 min) with AMI's manual word annotations and speaker turns.

**Accuracy (WER, `large-v3-turbo`)**

| Setup | EN read | ES read | AMI mean | Notes |
|---|---|---|---|---|
| Original stack, GPU | 1.72% | 3.88% | - | baseline |
| New stack, sequential, GPU | 1.72% | 4.23% | 24.4% | one AMI meeting hit a repetition loop |
| Current defaults (sequential + `no_repeat_ngram_size=4`) | 1.98% | 4.18% | 23.0% | loop fixed |
| Batched (opt-in) | 1.98% | 4.23% | 26.1% | ~2.5x faster ASR, drops fillers and short replies |

AMI WER is high for every setup because of overlapping speech (about 20% even with fillers removed), so only large
relative differences are meaningful. Three meetings is a small sample.

**Diarization (end-to-end DER on labelled ASR segments, 3 AMI meetings)**

| Setup | DER | Time per 15 min of audio |
|---|---|---|
| pyannote 3.1, segment-level labels (old) | 58.8% | ~260 s (CPU fallback on the old stack) |
| pyannote 3.1, word-level labels | 34.1% | same |
| `community-1`, word-level labels (current) | 32.9% | ~15 s on GPU |

**Real interview (32 min, no reference):** previous version 513 s vs 42-47 s now; speaker labels went from 4 labels
with 44 `[S?]` lines to 2 clean speakers with `max_speakers=2`; the earlier "bye" x43 hallucination is gone.

## Decisions and findings

- **Batched decoding is opt-in.** On LibriVox it looked free. On a real 32-minute interview it kept 3,804 words vs
  4,489 sequential (no `um/uh`, some short real replies) and saved only ~15 s of ASR. Enable "Faster GPU decoding"
  only for clean, single-speaker audio.
- **Repetition loops are the main accuracy risk on long conversations.** One AMI meeting lost 66 words to "uh" x53.
  `no_repeat_ngram_size=4` fixed it at no speed cost. Precision (fp32/int8), beam size 1/5/10, and full `large-v3`
  gave no consistent gain; `large-v3` looped badly on the long read clips. Turbo stays.
- **Set Max Speakers** when you know it: auto-detect split a two-person interview into three.
- **Granite Speech 4.1** is available but not more accurate than 4.0 on our clips (EN 2.77% vs 1.01%, ES 5.12% vs
  4.44%); whisper stays the default. Appending keywords to its punctuation prompt keeps punctuation (the model card's
  own keyword prompt drops it). (Branch `phase-4-granite-41`.)
- **Parakeet TDT v3** (ONNX): deferred. English 1.47%, Spanish 5.80%; 5.5x faster than Whisper on CPU; needs a tuned
  VAD (library defaults gave 7.5% English). Possible CPU-only backend later.
- **Nemotron streaming**: conditional go for an English live backend (1.05% WER, ~1.2 s behind speech) but it needs
  `transformers>=5.13`, which the app's `huggingface-hub==0.36.0` pin blocks. The English model is under the NVIDIA
  Open Model License; the multilingual 3.5 model is OpenMDW-1.1.
- **PaddleOCR-VL**: no-go. On synthetic slide/chat frames classic PaddleOCR 3.x (word F1 99.6%) beat it (96.2%) and the
  app's RapidOCR (61.7%), with far less VRAM. The synthetic frames are a stand-in; real frames are the better test.
- **Pre-existing bug:** the app's `paddleocr` backend has been silently falling back to RapidOCR because
  `requirements.txt` resolves `paddleocr` 2.10 while the code needs 3.x. Pinning `paddleocr>=3.0` resolves with the
  current pins (not re-run end to end).

## Setup notes

- **Accept the `community-1` terms** at https://huggingface.co/pyannote/speaker-diarization-community-1 with the
  account that owns your Hugging Face token, or diarization fails with a 403.
- Create a fresh virtual environment from `requirements.txt` (torch 2.11 / CUDA 12.8, driver 570+). With `uv`, add
  `--index-strategy unsafe-best-match` (multiple package indexes); plain `pip` works as is. Keep the old environment
  for rollback (old pins are in the git history of `requirements.txt`).
- `torchcodec` needs FFmpeg 4-8 shared libraries; on newer FFmpeg it cannot load. That is harmless here (audio is
  decoded in memory) and its warning is suppressed.

## Reproducing the measurements

```bash
python scripts/evaluate.py --device cuda --label baseline            # LibriVox WER/RTF/VRAM; DER with an RTTM manifest
python scripts/build_ami_refs.py --annotations <dir with words/*.xml> --meetings ES2004a IS1009a TS3003a
python scripts/accuracy_sweep.py --manifest evaluation/manifest.json --manifest eval/manifest-ami-asr.json
```

AMI audio and RTTM live in the git-ignored `eval/ami/`; annotations are
https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip (CC BY 4.0).
Private recordings, references, and results (`eval/`, `eval_results/`) are never committed.

## Not merged / future work

Local branches on top of `main` (not pushed): `phase-4-granite-41` (Granite 4.1, experimental),
`phase-5-parakeet-spike`, `phase-6-nemotron-spike`, `phase-7-paddleocr-vl` (spike scripts under `scripts/spikes/`).

Exploratory ideas (none started): Listener text-size control and other Qt GUI changes (to be defined),
`paddleocr>=3.0` pin and re-test, Transformers 5 migration plus a Nemotron live backend, Parakeet as a CPU-only
backend, Granite 4.1 `-plus` (word timestamps and speaker attribution).

## Known gaps

- Live microphone mode, the visual/OCR path, and Windows were not exercised after the upgrade.
- Accuracy conclusions rest on LibriVox plus three AMI meetings and one unreferenced real interview.
- Differences between precision settings (fp16, fp32, int8) and beam sizes are within noise on these sets; CPU int8
  scored better than GPU fp16 on English read speech in one baseline (0.92% vs 1.72%) and this was not explained.
