# OCR backend check, October 2026 upgrade

Question: with `paddleocr 3.4.1` and `paddlepaddle-gpu 3.3.1` in `.venv`, what does the `auto` OCR backend choose,
does PaddleOCR really run (GPU and CPU) or silently fall back to RapidOCR, and does slides-only UI-noise filtering
still work with 3.x output?

Result: diagnosis only. No application code was changed. Date of run: 2026-10-09, RTX 4090 (22.2 GB free of 24 GB),
branch `llm-frontier-and-context`.

## Method

All frames are synthetic. No real user media was used and nothing was downloaded (all five PaddleX models were
already in `~/.cache/pyscribe/paddlex/official_models/`: doc_ori, UVDoc, PP-OCRv5_server_det,
textline_ori, en_PP-OCRv5_mobile_rec).

- Frames: `scripts/spikes/make_ocr_frames.py` from branch `phase-7-paddleocr-vl`, read with `git show` (no checkout):
  10 JPEG frames (6 slides, 4 chat panes) with reference `.txt` files.
- Test video: the 6 slide frames, each with a fake meeting toolbar added at the top
  ("Sysadmin meeting Copilot Take control Pop out") and bottom ("Mute mic Camera Share Leave Chat People Raise React"),
  3 s per slide, 18 s total, H.264, made with ffmpeg.
- Scripts (scratchpad only, not in the repo): `matrix.py` (selection logic with stubbed builders),
  `real_ocr.py` (the app's real OCR functions, word F1 and seconds per frame after a warm-up frame),
  `e2e.py` (the app's `analyze_video_stream` on the test video).
- Commands (from the repo root, `.venv/bin/python`):
  - `matrix.py`
  - `real_ocr.py <frames> auto|rapidocr|paddleocr`
  - `CUDA_VISIBLE_DEVICES= real_ocr.py <frames> paddleocr` (forces the CPU path)
  - `HF_HUB_OFFLINE=1 real_ocr.py <frames> auto` (Hugging Face unreachable)
  - `e2e.py auto|rapidocr short|long <video>`; "long" patches the duration to 8000 s (over the 2 h threshold) and the
    sample interval to 1 s, so the long-video branch runs on the 18 s clip.

## Results

### 1. What `auto` chooses (`_build_ocr_fn`, `_choose_paddle_device`)

Real `_choose_paddle_device`, stubbed builders, so no models were loaded.

| Free VRAM | Short video | Long video (>= 2 h) |
|---|---|---|
| NVML unavailable (`None`) | paddleocr (GPU assumed) | paddleocr |
| 22.0 GB (real) | paddleocr, `gpu:0` | paddleocr, `gpu:0` |
| 1.6 GB | paddleocr, `gpu:0` | paddleocr |
| 1.4 GB (below the 1.5 GB floor) | rapidocr, cpu | rapidocr |
| 0.5 GB | rapidocr, cpu | rapidocr |
| No CUDA (`CUDA_VISIBLE_DEVICES=`), real run | rapidocr | rapidocr |

- Short vs long video makes no difference to the first choice. `long_video` only reorders the fallbacks after the
  first (pytesseract before surya) and changes the status note text.
- GPU vs CPU decides it: Paddle leads only when it will run on the GPU.

### 2. Does PaddleOCR really run? Accuracy and speed (synthetic frames, app's own OCR functions)

| Backend / device | Slide word F1 | Chat word F1 | s/frame |
|---|---|---|---|
| paddleocr, GPU (`auto` on this machine) | 0.991 | 0.981 | 0.08 |
| paddleocr, CPU (forced) | 0.994 | 0.957 | 5.04 |
| rapidocr (`auto` with no GPU) | 0.546 | 0.724 | 0.34 to 0.52 |

- PaddleOCR 3.x output is parsed correctly (`_extract_paddle_lines` reads `rec_texts` / `rec_scores`); the status
  line says "Initializing PaddleOCR (GPU)..." and `_PADDLE_OCR_DEVICE` is `gpu:0`. The 0.08 s vs 5 s gap confirms the
  GPU is really used.
- On this machine PaddleOCR does not silently fall back to RapidOCR in normal conditions.

### 3. End to end on the 18 s test video (`analyze_video_stream`)

| Run | Engine used | Frames | Slide text quality | UI noise leaked |
|---|---|---|---|---|
| auto, short, GPU | paddleocr | 18 (6 OCR calls) | clean lines | none |
| auto, long branch, GPU | paddleocr | 18 | same | none |
| auto, short or long, no CUDA | rapidocr | 18 | words run together ("Revenuegrew12percentyear overyear") | footer "Confidential - internal use only" leaked (OCR mangled it past the filter) |
| rapidocr requested | rapidocr | 18 | as above | footer leaked |

- Toolbar noise ("Mute mic", "Share", "Leave", "Copilot", "Take control", "Pop out", "Sysadmin meeting") was removed
  from the slides-only report with PaddleOCR 3.x output, so the noise filtering still works.
- The report's "OCR engine used" line says which engine ran, but a fallback note appears only in `on_status`
  and the report note (see finding B).

### 4. Fallback causes found

| Cause | Observed | Visible to the user? |
|---|---|---|
| Free VRAM below 1.5 GB, or no CUDA Paddle build / no GPU | `auto` picks RapidOCR by design | Report says "OCR engine used: rapidocr"; no reason |
| Hugging Face unreachable (`HF_HUB_OFFLINE=1`), models fully cached | `auto` silently ends on RapidOCR | Now names the reason (see finding B) |
| Paddle import / init error | Falls back to RapidOCR, same note | Same |

## Findings that need a decision

A. **Offline use loses PaddleOCR (decision recorded, reason now surfaced).** The model-manifest check against
Hugging Face is a deliberate security check and stays strict: with the network down and models fully cached, `auto`
still falls to RapidOCR (slide F1 about 0.55 instead of 0.99). Decision (2026-10-09): keep refusing, but say why.

B. **Resolved: the fallback reason is no longer lost in `auto` mode.** `_build_ocr_fn` now builds the note from the
failed backend's stored reason through `_brief_failure_reason` (one line, URLs, paths and token-like strings removed).
With Hugging Face unreachable the note reads "PaddleOCR unavailable: model manifest check failed (Hugging Face
unreachable); using RapidOCR." The note is also produced when no status callback is passed. Regression tests:
`tests/test_multimodal_service.py` (`test_auto_fallback_note_names_the_real_failure_reason`,
`test_brief_failure_reason_is_one_clean_line`). The backend order and the manifest verification are unchanged.

Follow-up (2026-10-09): a user-selectable fallback was added (`visual_ocr_fallback`, Qt "OCR Fallback" combo; choices
auto, rapidocr, pytesseract, surya; default auto = unchanged order). It is tried right after the first-choice backend;
if it is not installed the next backend in the usual order is used, and the note names both reasons. The service reads
it from the saved config at run time, so the combo saves immediately and again at job start. The manifest check has no
bypass switch.

C. **Slide titles are dropped.** Two- or three-word capitalised titles ("Quarterly Review", "Migration Plan",
"Security Checklist") never appear in the slides-only report, with either backend. `_is_low_value_slide_line` and
`_looks_like_person_name` treat them as participant-name fragments. This is independent of the upgrade.

D. **`auto` on CPU prefers RapidOCR** (about 10x faster, but slide F1 0.55 vs 0.99; Paddle on CPU is 5 s per frame).
For a short video with a CPU-only machine, Paddle may be worth the time; the current choice is deliberate and was
documented in the code comment.

E. Small: PaddleOCR is created with `use_textline_orientation=True` and the default doc orientation and unwarping
models, so two extra models load per run. The frames here OCR fine, but `use_doc_unwarping` on screen captures is
not needed and costs VRAM and time.

## Not covered (needs a human)

- A real webinar or Teams recording with chat and participant panes (names, avatars, animated content). Everything here
  is synthetic.
- Real GPU out-of-memory mid-run (the GPU to CPU retry in `_ocr` was exercised with fakes only).
- Peak VRAM was not re-measured here; the code comment cites about 1.2 GB from phase 7.
- `scripts/install_paddle_gpu.sh` reinstall behaviour.
