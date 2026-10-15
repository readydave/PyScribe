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
| OCR | `paddleocr` backend silently fell back to RapidOCR (pin resolved to 2.10, code needs 3.x) | `paddleocr>=3.0`; PaddleOCR 3.x runs on the GPU when the CUDA Paddle build and >=1.5 GB VRAM are available (CPU retry on failure); `auto` prefers it only on GPU. `scripts/install_paddle_gpu.sh` installs the CUDA build |
| Granite Speech | 4.0 1B only | 4.1 2B added as an experimental model (punctuation prompt, keyword biasing from the Names / terms field); 4.0 unchanged |
| Listener | fixed small text | A- / A+ text-size widget (80-200%, remembered per browser) |
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
  own keyword prompt drops it). Merged as an experimental model.
- **Parakeet TDT v3** (ONNX): deferred. English 1.47%, Spanish 5.80%; 5.5x faster than Whisper on CPU; needs a tuned
  VAD (library defaults gave 7.5% English). Possible CPU-only backend later.
- **Nemotron streaming**: conditional go for an English live backend (1.05% WER, ~1.2 s behind speech) but it needs
  `transformers>=5.13`, which the app's `huggingface-hub==0.36.0` pin blocked (pins now moved, see below). The English model is under the NVIDIA
  Open Model License; the multilingual 3.5 model is OpenMDW-1.1.
- **PaddleOCR-VL**: no-go. On synthetic slide/chat frames classic PaddleOCR 3.x (word F1 99.6%) beat it (96.2%) and the
  app's RapidOCR (61.7%), with far less VRAM. The synthetic frames are a stand-in; real frames are the better test.
- **`paddleocr` backend fixed and moved to GPU.** It had been silently falling back to RapidOCR because the pin
  resolved to 2.10 while the code needs 3.x. With `paddleocr>=3.0` (3.4.1) plus explicit detection/recognition model
  names, the app's backend scores word F1 about 0.99 on the synthetic frames (RapidOCR 0.62). The default CPU
  `paddlepaddle` build takes about 5.5 s per frame; the CUDA build takes about 0.08 s (GPU changes speed, not accuracy).
  `paddlepaddle-gpu` wheels pin exact `nvidia-cudnn-cu12` / `nvidia-cublas-cu12` versions that conflict with torch's, and
  the resolver then silently picks an old 2.6.x release, so `scripts/install_paddle_gpu.sh` installs 3.3.1 (cu126 index)
  with `--no-deps` and Paddle reuses torch's CUDA 12.8 libraries (torch, ctranslate2 and Paddle ran in one process
  without conflicts). A real 1 h 1080p60 meeting recording ran end to end (ASR + diarization + OCR) in 95 s, with a
  13.2 GB whole-GPU peak (+7.8 GB over other processes) on a 24 GB card.

## Transformers 5 migration

`requirements.txt` now has `transformers>=5.13` (5.18.0 tested) and `huggingface-hub>=1.0` (1.33.0 tested). Verified in a
fresh venv against the previous environment, same machine:

- Code change: `HfFolder` no longer exists in huggingface-hub 1.x; `services/hf_auth_service.py` uses `get_token()` and
  `login(token=..., add_to_git_credential=False)` (the latter validates the token online when persisting). One test
  needed an `httpx.Response` to build `RepositoryNotFoundError`.
- Granite 4.0 1B: EN 1.05% / ES 4.40% (before 1.01% / 4.44%); Granite 4.1 2B: EN 2.65% / ES 5.12% (before 2.77% / 5.12%).
- Whisper `large-v3-turbo` defaults: EN 1.98% / ES 4.18%, identical to before. Diarization DER on the 3 AMI meetings is
  identical on both stacks (26.9% / 38.1% / 24.4%).
- 212 tests pass; Qt, Listener, live and OCR modules import. Live microphone mode was covered by the unit tests only, not
  exercised with a real microphone.

## Nemotron streaming backend

`nvidia/nemotron-speech-streaming-en-0.6b` is available as an experimental, English-only model for Qt live mode and
file transcription (`services/nemotron_streaming_service.py`; the live worker has a streaming branch that appends audio
instead of re-decoding a rolling window). Measurements on the same machine:

- English read speech through the app path: WER 0.97% (CER 0.33%), RTF 0.019, 1.8 GB VRAM.
- Token timestamps (80 ms frames) vs faster-whisper word times on 3 minutes of read speech: median start offset +130 ms,
  median absolute difference 170 ms, 92% within 640 ms.
- 32-minute two-speaker interview (private, not committed), same pyannote turns for both: Nemotron and Whisper words got the
  same speaker on 99.9% of matched words (4 of 4,085 differ, all within 1 s of a turn change). Nemotron produced 4,346 words
  vs Whisper's 4,519; Whisper keeps more uh/um/yeah/okay, Nemotron writes numbers as words ("twenty twenty six"). Accuracy
  on conversational audio has no reference yet, only this disagreement.
- Whole-file time for that interview: ASR 35 s, diarization 19 s. The per-token text streamer used for live mode makes
  decoding about 5x slower, so file mode turns it off.
- Real-time live run (spawned worker, audio fed at 1x from the interview): first text about 2 s after speech began, stop
  flush 0.2 s.

Design notes: file mode (and so the live final post-pass) feeds the whole file through the same chunked streaming path;
the final output equals a Whisper-free pipeline plus diarization. Live capture does not produce speaker labels. Names /
terms are applied as a spelling-correction pass (`services/term_correction.py`) on file output and the live final pass; the decoder is not biased. Not exercised with a real microphone.

## Granite 4.1 `-plus` evaluation (no-go)

`ibm-granite/granite-speech-4.1-2b-plus` (Apache-2.0) adds word timestamps and speaker-attributed ASR to Granite 4.1 by prompt;
it gives up punctuation and capitalization. Spike: `scripts/spikes/granite_plus_spike.py` (bf16, same machine, Transformers 5.18).
The model card says timestamps are reliable up to about 3.5 minutes of audio, so audio is chunked.

| Test | Result |
|---|---|
| LibriVox English, 15 min, 3 min chunks | WER 1.01% (Whisper 1.98%, Granite 4.0 1.05%); word starts median 60 ms from Whisper's (Nemotron: 170 ms); RTF 0.17 (160 s), 4.7 GB |
| AMI ES2004a, 3 min chunks | WER 19.4% (Whisper 25.4%), but AMI-IHM is in Granite's timestamp training data, so this is optimistic; word times drift early by about 1 s per 30 s inside a chunk (up to 17 s in one chunk), unusable for speaker assignment |
| AMI ES2004a, silence-cut chunks of at most 30 s | word times good (median 226 ms, 77% within 640 ms) but WER 33.9% |
| Speaker-attributed ASR, first 3 min of ES2004a | produced 3 speaker tags in 16 turns; not scored |

Decision: not integrated. Accurate text and good timing did not come from the same chunk length on conversational audio, the model
is about 20x slower than Whisper (about 10x slower than Nemotron), and it has no punctuation. A backend would need VAD chunking
plus a timing correction, and Whisper (with `no_repeat_ngram_size=4`) and Nemotron already give word times. Speaker-attributed ASR
numbering across chunks (the card's incremental `prefix_text` mode) was not tried; our pyannote path already labels words.

## Setup notes

- **Accept the `community-1` terms** at https://huggingface.co/pyannote/speaker-diarization-community-1 with the
  account that owns your Hugging Face token, or diarization fails with a 403.
- Create a fresh virtual environment from `requirements.txt` (torch 2.11 / CUDA 12.8, driver 570+). With `uv`, add
  `--index-strategy unsafe-best-match` (multiple package indexes), for example
  `uv venv --python 3.12 .venv` then `uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt`;
  plain `pip` works as is. Keep the old environment for rollback (old pins are in the git history of `requirements.txt`).
- For GPU OCR run `scripts/install_paddle_gpu.sh` after installing requirements; it replaces the CPU `paddlepaddle` wheel
  with `paddlepaddle-gpu` and must be re-run after any reinstall of `requirements.txt`. Skip it on CPU-only machines.
- `pytest` is not in `requirements.txt` (see `CONTRIBUTING.md`); install it separately to run the tests.
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

Local branch `phase-7-paddleocr-vl` (not pushed, spike scripts only) keeps the OCR comparison scripts under
`scripts/spikes/`, useful for re-testing OCR on real frames. The Parakeet and Nemotron spike branches were removed (their
results are recorded above; spike scripts on `main`: `scripts/spikes/nemotron_timestamps.py`, `scripts/spikes/granite_plus_spike.py`).

Done since the first write-up: Transformers 5 migration, the Nemotron streaming backend, Names / terms for Nemotron
(`services/term_correction.py`), and the Granite 4.1 `-plus` evaluation (no-go).

Open: Qt GUI changes (to be defined by the maintainer); a diarization over-segmentation check on a long real meeting (13
speakers were found); a Nemotron vs Whisper accuracy comparison on hand-corrected conversational audio; an OCR re-test on real
hand-corrected frames; Parakeet as a CPU-only backend (needs an ONNX Runtime dependency policy).

## Known gaps

- Live microphone mode (Whisper and Nemotron) and Windows were not exercised after the upgrade; Nemotron live was only run through a spawned worker fed from a recording. Nemotron term correction was unit-tested only, not run on real audio. The visual/OCR path was exercised on one real
  1-hour meeting recording and on synthetic frames, not scored against real hand-corrected frames. The GPU-to-CPU OCR retry
  was tested with fakes only, and the cu126 Paddle wheel on torch's cu128 libraries was verified on one machine (driver 615).
- Accuracy conclusions rest on LibriVox plus three AMI meetings and one unreferenced real interview.
- Differences between precision settings (fp16, fp32, int8) and beam sizes are within noise on these sets; CPU int8
  scored better than GPU fp16 on English read speech in one baseline (0.92% vs 1.72%) and this was not explained.
