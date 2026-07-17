# PyScribe Architecture Review — July 2026

Scope: full-repo review across architecture, maintainability, reliability, UX,
performance, and security lenses. Findings were verified against code on
`main` (commit `603b7f7`). Items marked **assumption** could not be verified by
running the app (review environment had no ML/Qt dependencies installed).

---

## 1. Executive assessment

PyScribe is in **much better architectural shape than the average solo-developer
ML desktop app**. The services layer is real (both frontends genuinely share
`transcription_service`, `config_service`, `llm_*`, `catalog_service`), the
security posture around listener binding and credentials is deliberate and
well-tested in design, and process isolation for CUDA-heavy work (spawned
subprocesses for diarization and Qt transcription) is the right call and
carefully implemented.

The three structural debts that will hurt most over the next year:

1. **`ui_qt/main_window.py` is a ~4,000-line god module.** `MainWindow` alone
   has ~150 methods covering batch queue, live capture, theming, hardware
   monitoring, save logic, dialogs, and worker management. Every feature added
   makes the next one harder.
2. **The real test suite never runs in CI.** ~20 test modules (~3,500 lines)
   exist, but CI runs only `smoke_cli.py` (help text + `py_compile`). Most
   tests hard-import PySide6/numpy, so they only run on a fully provisioned dev
   machine. Regressions in the most fragile areas (worker control, live mode)
   are invisible to CI.
3. **Install/packaging is GPU-monolithic.** `requirements.txt` pins CUDA 12.1
   wheels as the only path and bundles every optional backend (paddle, pyannote,
   transformers) as mandatory; `pyproject.toml` declares **no dependencies at
   all**, so `pip install .` produces a broken `pyscribe` script.

Reliability engineering inside the pipeline is strong; the weak spots are at
the edges: shared config written non-atomically by two frontends, listener
module-level state shared across web sessions, and fragile string-matching for
error classification across process boundaries.

---

## 2. Top 10 findings (ranked by impact)

| # | Finding | Severity | Confidence |
|---|---------|----------|------------|
| 1 | CI never runs the real test suite | High | Verified |
| 2 | `MainWindow` god class (~4,000 lines, ~150 methods) | High | Verified |
| 3 | Packaging: no deps in `pyproject.toml`; CUDA-only monolithic requirements | High | Verified |
| 4 | Shared config file: non-atomic writes, no locking, silent total reset on parse failure | Medium-High | Verified |
| 5 | Listener global mutable state shared across all web sessions (`_cancel_event`, `APP_CONFIG`, config persistence per job) | Medium-High | Verified |
| 6 | Error classification via substring matching on exception text | Medium | Verified |
| 7 | Process/event-pump plumbing duplicated ~4× | Medium | Verified |
| 8 | Model catalog duplicated across 3 modules + dead tkinter/pip-install code in `utils.py` | Medium | Verified |
| 9 | Listener privacy/UX edges: transcripts written to shared server temp dir, never cleaned; clipboard copies land on the *server* | Medium | Verified |
| 10 | No lint/format/type tooling configured (no ruff/mypy/pre-commit) | Medium | Verified |

---

## 3. Findings detail

### F1. CI never runs the real test suite — High
**Evidence:** `.github/workflows/ci.yml` runs `main.py --help`, `py_compile`,
`bash -n`, and `pytest tests/smoke_cli.py` only. Meanwhile
`tests/test_qt_main_window_worker.py`, `test_qt_live_mode.py`,
`test_transcription_service.py`, etc. exist and hard-import PySide6 / numpy /
services modules. Only `test_benchmark_dialog.py` uses `skipIf` for missing Qt.
**Impact:** The most fragile flows PROJECT.md itself lists (worker
cancellation, live mode transitions, config compatibility) have regression
tests that no automation executes.
**Recommendation:** Split tests into two tiers. Tier A (pure logic: config,
listener security, prompt templates, LLM profile parsing, transcript
reconciliation) must run with only `pytest + pyyaml + requests` installed —
refactor imports so these modules don't drag in numpy/Qt. Tier B (Qt/pipeline)
runs in a second CI job installing CPU torch + PySide6 with
`QT_QPA_PLATFORM=offscreen`. Even Tier A alone in CI is a large win.

### F2. `MainWindow` god class — High
**Evidence:** `ui_qt/main_window.py` is 3,989 lines; `MainWindow` spans lines
647–3,989 and mixes: batch queue model + handlers, live capture session
control (~20 methods), transcription worker lifecycle, theme engine (inline QSS
~200 lines), hardware monitor thread, save/auto-save logic, HF token dialog,
help/about, responsive layout.
**Impact:** Highest ongoing maintenance cost in the repo; every fragile flow
in PROJECT.md's risk list lives in this one class; impossible to unit test
except through the whole window.
**Recommendation:** Incremental extraction, no rewrite:
`ui_qt/batch_queue.py` (model + controller), `ui_qt/live_controller.py`
(the `_live_*` methods around `LiveSessionController`),
`ui_qt/theme.py` (QSS + theme mode logic), `ui_qt/hw_monitor.py`,
`ui_qt/save_service.py` (`_save_payload_for_mode`, `_auto_save_completed_parts`).
`TranscriptionWorker` + `_transcription_process_entry` also belong in their own
module (`ui_qt/transcription_worker.py`) — they are already self-contained.

### F3. Packaging and install path — High
**Evidence:** `pyproject.toml` has `[project.scripts] pyscribe = "main:main"`
but no `dependencies` key at all. `requirements.txt` pins
`torch==2.5.1+cu121` and makes paddleocr/paddlepaddle/pyannote/transformers
mandatory. README admits CPU-only installs "may need a custom Torch install."
**Impact:** `pip install .` yields a console script that crashes on import;
CPU-only and AMD users have no supported path; install size for a
"transcribe a file" user is enormous.
**Recommendation:** Declare core deps in `pyproject.toml` with extras:
`pyscribe[gpu]`, `pyscribe[diarization]`, `pyscribe[ocr-paddle]`, etc.
Provide `requirements-cpu.txt`. Keep `requirements.txt` as the pinned
GPU-dev lockfile.

### F4. Config persistence — Medium-High
**Evidence:** `services/config_service.py:83-96` — `save_config` does a bare
`path.write_text(...)` (no temp-file + `os.replace`), no inter-process locking.
`load_config` catches *all* exceptions and returns a default `AppConfig`; the
next save then permanently overwrites whatever was on disk — including all LLM
profiles — after a single corrupt/partial write. Both Qt and the listener write
this file (`app.py:539-554` saves on every transcription run).
**Impact:** Low-probability, high-annoyance data loss (LLM profiles, saved
prefs); last-write-wins races when Qt and listener run simultaneously — a
supported scenario.
**Recommendation:** (1) Atomic write: write to `path.with_suffix(".tmp")`,
then `os.replace`. (2) On parse failure, rename the bad file to
`.pyscribe_config.json.bad` before returning defaults. (3) Reduce listener
config writes to explicit user-facing preference changes rather than every job.

### F5. Listener session-shared state — Medium-High
**Evidence:** `app.py:150-151` — `_cancel_event` and `_transcription_active`
are module-level. With `default_concurrency_limit=1` jobs serialize, but the
Cancel button of *any* connected browser session sets the same event, killing
whichever job is running. `APP_CONFIG` is a module-level global mutated from
request handlers.
**Impact:** In LAN mode (an advertised feature), user B can cancel user A's
job; per-user preferences overwrite each other. Single-user localhost is
unaffected, which is why this hasn't bitten yet.
**Recommendation:** Move cancel state into `gr.State` per session, or key
cancel events by job id. If multi-user is a non-goal, document that the LAN
listener is single-operator and leave the code — but decide explicitly.

### F6. String-matched error classification — Medium
**Evidence:** `transcription_service._should_retry_diarization_on_cpu`
(matches `"libnvrtc"`, `"exit code -"`, `"expected all tensors..."`),
`main_window._transcription_process_entry` (matches `"initialization error"`,
`"cuda failed"`). Errors cross process boundaries as flattened `str(exc)`.
**Impact:** Retry/fallback behavior silently changes when torch/pyannote
reword messages on upgrade; overly-broad markers (`"exit code -"`) can
misclassify unrelated failures as CUDA issues.
**Recommendation:** Emit structured error events from child processes:
`{"type": "error", "code": "cuda_init" | "oom" | ..., "detail": str}`,
classifying *inside* the child where the exception type is still available
(e.g. `torch.cuda.OutOfMemoryError`). Keep string matching only as a
last-resort fallback in one shared helper.

### F7. Duplicated process/event plumbing — Medium
**Evidence:** Near-identical drain loops + terminate/kill/join sequences in:
`transcription_service._run_diarization_backend_in_subprocess` (drain twice,
`_stop_process`), `main_window.TranscriptionWorker.run` (drain twice,
`_stop_child_process`), and a third variant in
`live_transcription_service._stop_process`.
**Impact:** Bug fixes to one copy (timeout handling, late-event draining)
don't propagate; the copies have already drifted (different timeouts,
different force-kill behavior).
**Recommendation:** Extract `services/process_worker.py` with a
`SpawnedJob` helper: `start()`, `drain_events(dispatch)`, `stop(reason,
timeout)`, `run_until_done(cancel_check, dispatch)`. All three call sites
shrink dramatically.

### F8. Model catalog triplication + dead code — Medium
**Evidence:** Three independent model lists: `utils.get_available_hf_models`
(popular_models), `catalog_service.BASE_MODEL_CHOICES`, `models.TIERS`.
They disagree (e.g. `small.en` is in TIERS but not in the choice lists;
Granite appears in two of three). Also `utils.check_and_install_dependencies`
(lines 14-54) is dead code from the tkinter era — never called, imports
`tkinter` at module level (breaks headless/minimal Python installs that lack
Tk, since `utils` is imported by `transcription_service`), references
`ttkthemes` which is not in requirements, and would run `pip install` at
runtime if ever invoked.
**Impact:** Adding/removing a model requires editing 2–3 files; the
module-level `import tkinter` is a real cross-platform reliability risk.
**Recommendation:** Delete `check_and_install_dependencies` and the tkinter
imports. Fold `get_available_hf_models` into `catalog_service` (or
`models.py`) so there is exactly one curated list plus one cache-scan
function.

### F9. Listener output/clipboard privacy edges — Medium
**Evidence:** `app.py:624-668` — saved transcripts and LLM outputs are
written to `tempfile.gettempdir()` (world-shared on multi-user Linux; files
persist until OS cleanup). `copy_to_clipboard` uses `pyperclip`, which sets
the clipboard of the *server host*, not the browser — for a LAN user the
button silently does nothing for them and leaks the transcript to the host's
clipboard.
**Impact:** Contradicts the local-first/privacy positioning in a minor but
real way; confusing UX in LAN mode.
**Recommendation:** Save downloads under `~/.pyscribe/exports` (0700) and
clean up files older than N days at startup. Remove the server-side clipboard
button in listener mode (Gradio textboxes already have a native copy
control), or gate it to localhost binds.

### F10. No lint/format/type tooling — Medium
**Evidence:** No ruff/flake8/black/mypy config anywhere in the repo; CI has no
lint step. Code style is actually quite consistent (suggesting manual
discipline), but nothing enforces it.
**Recommendation:** Add `ruff` (lint + format) with a minimal rule set and a
CI step. Given how many coding agents work on this repo (per AGENTS.md /
PROJECT.md), automated style enforcement pays off doubly. `mypy` on
`services/` only, later.

### Additional observations (below top-10)

- **`_normalize_base_url_for_profile` globals() indirection**
  (`llm_connection_service.py:673-695`): a defensive `globals().get()` lookup
  with a "stale/hot-reload runtimes" comment — a debugging artifact that
  obscures the code. Safe to inline now. (Verified)
- **Busy-wait polling** (`time.sleep(0.05)` loops in worker pumps): correct
  but wasteful; a blocking `queue.get(timeout=0.25)` with periodic cancel
  checks would be cleaner. Low priority. (Verified)
- **LAN listener is plain HTTP with basic auth** — credentials and media
  transit unencrypted on the LAN. Acceptable for the stated threat model but
  worth one sentence in README's security section. (Verified)
- **Log rotation env-var guard** (`PYSCRIBE_SESSION_STARTED`): two *separate*
  concurrent app launches share `pyscribe.log` with interleaved writes from
  two `RotatingFileHandler`s — rotation can misbehave. Rare scenario; low
  priority. (Verified, low impact)
- **`multimodal_service.py` (1,278 lines) and `diar_backends.py` internals
  were not deep-read** — no adverse findings, but confidence in those areas is
  lower than for the rest of this review. (Assumption)

---

## 4. Quick wins (next 1–2 weekends)

Each item is small, low-risk, and independently shippable:

1. **Atomic config writes + `.bad` quarantine on parse failure**
   (`config_service.py`) — ~30 lines, protects LLM profiles. (F4)
2. **Delete `check_and_install_dependencies` + tkinter imports from
   `utils.py`** — removes a headless-install landmine. (F8)
3. **Add a CI job that runs the dependency-light tests** (config, listener
   security, prompt templates, LLM profile parsing) — even 5 test modules in
   CI beats zero. (F1)
4. **Add ruff + CI lint step.** (F10)
5. **Consolidate the three model lists into `catalog_service`.** (F8)
6. **Inline `_normalize_base_url`, remove the `globals()` fallback.** (misc)
7. **Listener exports directory + cleanup instead of `gettempdir()`;
   drop/gate the server-side clipboard button.** (F9)

---

## 5. Refactor roadmap

### Phase 0 — Stabilize the safety net (1–2 weekends)
Quick wins above, prioritizing #1–#4. Goal: CI meaningfully guards the
fragile areas before any structural refactoring starts.

### Phase 1 — Test tiering + CI matrix (1–2 weeks of spare time)
- Restructure test imports so pure-logic tests don't import numpy/Qt.
- Second CI job: CPU torch + PySide6 offscreen, running worker/live-mode/
  pipeline tests (allow it to be slow; run on PRs only).
- Add a `pytest.ini`/`pyproject` marker scheme: `unit`, `qt`, `pipeline`.

### Phase 2 — Decompose `main_window.py` (incremental, several weekends)
Extraction order chosen so each step is mechanical and testable:
1. `TranscriptionWorker` + `_transcription_process_entry` → own module.
2. Batch queue (model + handlers) → `ui_qt/batch_queue.py`.
3. Theme/QSS → `ui_qt/theme.py`.
4. Live-mode controller methods → `ui_qt/live_controller.py` (talks to the
   existing `LiveSessionController`).
5. Save/auto-save logic → service-layer function shared with listener.
Target: `main_window.py` under ~1,200 lines, pure composition + navigation.

### Phase 3 — Process plumbing + typed errors (1–2 weeks)
- `services/process_worker.py` (`SpawnedJob`) replacing the four
  drain/stop copies. (F7)
- Structured error codes across process boundaries; classify in-child. (F6)
- Optional: replace polling sleeps with blocking queue gets.

### Phase 4 — Packaging + listener hardening (as motivation allows)
- Dependencies + extras in `pyproject.toml`; CPU install path. (F3)
- Per-session cancel state in the listener; decide the multi-user story. (F5)
- Revisit `--share`/LAN docs to mention plain-HTTP transport explicitly.

---

## 6. Open questions needing human input

1. **Is the LAN listener ever used by more than one person at a time?**
   Determines whether F5 is a bug to fix or a constraint to document.
2. **Is CPU-only (or AMD/ROCm) a supported target?** Determines how much of
   F3 to invest in vs. simply documenting "NVIDIA required."
3. **Is `pip install .` / the `pyscribe` console script a real distribution
   path**, or is source-checkout the only supported install? If the latter,
   the script could be removed instead of fixed.
4. **How important is Windows live capture?** Live mode is Linux-first; the
   loopback path and device handling need Windows verification before Phase 2
   touches that code.
5. **Cadence of dependency upgrades** (torch/pyannote/gradio): typed error
   codes (F6) matter more if upgrades are frequent.
