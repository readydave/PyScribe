# HANDOFF.md — PyScribe Improvement Plan (Phased Implementation)

This is the working implementation plan for the July 2026 architecture review
(`docs/architecture_review_2026-07.md`). Finding references (F1–F10) point to
that document. This file is meant to be handed to a developer or coding-agent
session; each task is scoped to be completed and verified independently.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done ·
`[!]` blocked (note why)

---

## 0. Ground rules (read before every session)

1. **Cross-OS is a requirement, not a nice-to-have.** Every change must work on
   Windows 10/11 and Linux (Ubuntu/Debian). Rules that follow from this:
   - Use `pathlib` / `os.path.join`; never hard-code `/` or `\` separators.
   - Atomic file replace is `os.replace()` (works on Windows; `Path.rename()`
     over an existing file does NOT).
   - `multiprocessing` context is always explicit `spawn` (already the repo
     convention — keep it; it's also the only option on Windows).
   - No POSIX-only calls (`os.fork`, `fcntl`, signals beyond SIGTERM) in shared
     code paths. `select.select` on stdin is POSIX-only — `main.py` already
     branches on `os.name`; preserve that pattern.
   - File locking semantics differ (Windows locks opened files); never assume
     a log/config file can be renamed while another process has it open —
     always wrap in try/except and degrade gracefully (existing
     `logging_service` pattern).
2. **Incremental only.** No big-bang rewrites. Every task lands as a small,
   reviewable commit that leaves the app fully working. If a task can't be
   split from another, that's a sign to re-scope it.
3. **Tests move with code.** When extracting a module, move/adjust its tests in
   the same commit. New logic gets Tier A tests (see Phase 1) where feasible.
4. **Docs sync (PROJECT.md rule).** If a task changes user-visible behavior,
   update `README.md`, `docs/user_guide.md`, `docs/qt_help.md`, and
   `CHANGELOG.md` in the same commit. Update `PROJECT.md`'s decision log for
   architectural decisions.
5. **Don't touch:** `AGENTS.md`, `SECURITY.md` (policy text), license, and the
   listener security model (localhost default, no `--auth-pass`, auth required
   for non-local binds). Behavior-preserving refactors only, unless the task
   explicitly says otherwise.
6. **Fragile areas (extra care + manual verification):** Qt worker
   cancel/force-stop, live mode state transitions, config compatibility with
   existing `~/.pyscribe_config.json` files, CUDA/loader environment setup.

### Per-phase workflow

- One branch per phase (`improve/phase-0-safety-net`, etc.); one commit per
  task where practical.
- Before marking a task done, run its **Verify** block on Linux; run the
  cross-OS verify on Windows for any task marked **[XOS]**. If you only have
  one OS available, mark the task `[~]` with a note ("needs Windows verify")
  instead of `[x]`.
- **Stop rule:** if any Verify step fails, stop — do not merge the phase
  branch and do not start the next task. Either fix it within the task's
  commit or mark the task `[!]` with a written note of the failure (an issue
  or a note in this file) before moving on. A failed Verify that gets merged
  anyway is the one way this plan goes bad.
- Update the checkboxes in this file as part of the commits.
- **Docs DoD:** every phase merge includes a CHANGELOG entry; user-visible
  changes also update README / `docs/user_guide.md` / `docs/qt_help.md` per
  ground rule 4. Treat this as part of each phase's exit criteria.
- Time estimates are deliberately optimistic pacing guides, not commitments —
  cross-OS verification is the usual source of slip. Slipping is fine;
  skipping Verify to hold the estimate is not.

---

## Phase 0 — Safety net & quick wins (est. 1–2 weekends)

Goal: protect user data, remove landmines, and get *something* meaningful into
CI before structural refactoring starts. No behavior changes visible to a
happy-path user.

### P0.1 `[~]` Atomic config writes + corruption quarantine (F4) [XOS]
*(implemented + Linux tests green 2026-07-18; needs Windows verify — the
PermissionError retry path in `_replace_config_file` in particular)*
**Files:** `services/config_service.py`, `tests/` (new Tier A test)
- `save_config`: serialize to `path.with_name(path.name + ".tmp")`, `flush` +
  `os.fsync`, then `os.replace(tmp, path)`. Wrap in try/except; on failure log
  and leave the previous file untouched.
- `load_config`: distinguish "file missing" (return defaults silently) from
  "file exists but unparseable" (rename the bad file to
  `<name>.bad-<timestamp>` before returning defaults, log a warning). Never
  let a corrupt file be silently overwritten with defaults.
- Add tests: round-trip; corrupt-file quarantine; tmp-file cleanup; unknown
  keys in an old config are preserved-or-ignored without crash (config
  compatibility rule).
- **Windows note:** `os.replace` fails if another process holds the file open
  with exclusive access — catch `PermissionError` and retry once after 100ms,
  then give up with a logged warning (config save is not worth crashing over).
**Verify:** new tests pass; hand-corrupt `~/.pyscribe_config.json`, launch Qt,
confirm a `.bad-*` file appears and settings reset without a crash.

### P0.2 `[~]` Remove dead tkinter/pip-install code (F8) [XOS]
*(done + verified on Linux (Tk-less venv imports `utils` cleanly) 2026-07-18;
Windows verify is a formality — pure deletion)*
**Files:** `utils.py`
- Delete `check_and_install_dependencies` and the `tkinter`, `subprocess`,
  `importlib.util` imports it needed. Nothing calls it (verified in review).
- This removes a module-level `import tkinter` from a module imported by
  `transcription_service` — a real crash risk on minimal Python installs
  (Linux distros without `python3-tk`, some Windows embeddable builds).
**Verify:** `grep -rn "check_and_install_dependencies\|tkinter\|ttkthemes"`
returns nothing outside docs; `python -c "import utils"` works in a venv
without Tk; smoke tests pass.

### P0.3 `[x]` Consolidate the model catalog (F8)
*(done 2026-07-18; choice set verified identical to old union except the
intended `+small.en` drift fix; also made `diar_backends`→`diarization`
import lazy so catalog/availability checks need no torch)*
**Files:** `models.py` or `services/catalog_service.py` (pick ONE owner —
recommendation: `services/catalog_service.py`), `utils.py`
- Move `utils.get_available_hf_models`'s curated `popular_models` list and the
  local-cache scan into `catalog_service`. Merge with `BASE_MODEL_CHOICES` so
  there is exactly one curated list constant.
- `models.TIERS` stays (it's metadata, not a choice list) but add a unit test
  asserting every TIERS key appears in the catalog choices, so the lists can't
  drift silently again.
- Cache scan: keep using `HF_HOME`-aware paths, not hard-coded
  `~/.cache/huggingface/hub/` (current code hard-codes it — fix while here;
  respect `HF_HOME`/`HUGGINGFACE_HUB_CACHE` env vars, which the app itself
  sets via `runtime_env_service`).
**Verify:** Qt and listener model dropdowns show the same list as before
(manually compare); new drift test passes.

### P0.4 `[ ]` Minimal real-test CI job (F1)
**Files:** `.github/workflows/ci.yml`
- Add a `unit-tests` job (ubuntu + windows matrix) that installs only
  `pytest pyyaml requests` and runs the test modules that already work without
  heavy deps. Determine the exact list by trial: expected candidates are
  `test_prompt_templates_and_config.py`, `test_security_hardening.py`,
  `test_listener_llm_postprocess_helpers.py`, `test_llm_connection_service.py`,
  `test_launcher.py`.
- **Fallback rule (do not let this block Phase 0):** any candidate that pulls
  in numpy/Qt transitively is dropped from this job on the spot, listed in a
  short "deferred to P1.2" note in this file, and handled by the Phase 1
  import-untangling work. P0.4 is done when *at least three* modules run
  green on both OS runners — more is a bonus, not the bar.
- Keep the existing smoke job unchanged.
**Verify:** CI green on both OS runners; deliberately break a config test
locally to confirm the job actually fails.

### P0.5 `[ ]` Add ruff + CI lint step (F10)
**Files:** `pyproject.toml` (`[tool.ruff]`), `.github/workflows/ci.yml`
- Start permissive: line-length 120, rule set `E,F,W,I` (errors, pyflakes,
  warnings, import sort). Run `ruff check --fix` + `ruff format` once across
  the repo as its own commit ("mechanical only — no logic changes").
- Add `ruff check` (no format enforcement yet) to CI.
- Do NOT enable opinionated rule families yet; goal is a tripwire, not a
  style war with 4,000-line files that Phase 2 is about to carve up anyway.
**Verify:** `ruff check .` clean; CI green; `git diff` of the format commit
shows no logic changes (spot-check the worker/live-mode files carefully).

### P0.6 `[~]` Listener export directory + clipboard fix (F9) [XOS]
*(implemented 2026-07-18; helpers verified (dir/0700/cleanup/save) with real
gradio import; needs a live browser download check + Windows verify)*
**Files:** `app.py`, `docs/user_guide.md`, `CHANGELOG.md`
- Replace `tempfile.gettempdir()` in `save_transcript` /
  `save_postprocess_output` with `~/.pyscribe/exports/` (create with
  `mkdir(parents=True)`; on POSIX `chmod 0o700` like `logging_service` does —
  skip chmod errors on Windows, same pattern as `_tighten_directory_permissions`).
- On listener startup, delete export files older than 7 days (best-effort,
  swallow per-file errors).
- Remove the `pyperclip` "Copy to Clipboard" buttons from the listener
  (Gradio textboxes have built-in copy affordances; `pyperclip` copies to the
  *server's* clipboard and additionally requires `xclip`/`xsel` on Linux).
  Remove `pyperclip` from `requirements.txt` if nothing else uses it.
- Qt copy behavior is unaffected (Qt uses its own clipboard API — verify).
**Verify:** listener save produces files under `~/.pyscribe/exports`; download
buttons still work in browser on another machine (LAN test if possible); old
files pruned; `grep -rn pyperclip` only in CHANGELOG.

### P0.7 `[x]` Inline `_normalize_base_url` (review misc)
*(done 2026-07-18; `tests/test_llm_connection_service.py` green)*
**Files:** `services/llm_connection_service.py`
- Remove the `globals().get("_normalize_base_url")` indirection and its
  fallback branch in `_normalize_base_url_for_profile`; call the helper
  directly. Keep behavior identical; keep existing tests green.
**Verify:** `python -m pytest tests/test_llm_connection_service.py` (in a full
env) or at minimum CI Tier A if that module is covered there.

**Phase 0 exit criteria:** all seven tasks `[x]` (or `[~]` pending Windows
verify with notes); CI has ≥5 real test modules running on ubuntu + windows;
CHANGELOG updated once for the phase.

---

## Phase 1 — Test tiering & CI matrix (est. 1–2 weeks of spare time)

Goal: every test in the repo runs *somewhere* in CI. Two explicit tiers.

### P1.1 `[ ]` Define tiers with pytest markers
**Files:** `pyproject.toml` (`[tool.pytest.ini_options]`), all `tests/*.py`
- Markers: `unit` (no heavy deps), `qt` (needs PySide6), `pipeline` (needs
  numpy/torch/faster-whisper). Default addopts: none (running `pytest` runs
  everything, as today, on a dev machine).
- Add a conftest-level guard: `qt`/`pipeline` tests use
  `pytest.importorskip` for their heavy imports so a partial environment
  skips instead of erroring at collection. Follow the existing pattern in
  `test_benchmark_dialog.py` (`skipIf(QApplication is None, ...)`).

### P1.2 `[ ]` Break heavy imports out of pure-logic modules
**Files:** likely `services/transcription_service.py` (imports `utils` which
imports numpy at module level), `tests/*`
- Audit which Tier-A-candidate tests fail at import time in a bare venv and
  why. Typical fix: move `import numpy` in `utils.py` into the two functions
  that use it (`load_audio_waveform` already lazy-imports `ffmpeg`; numpy can
  follow that pattern) so `catalog_service`→`utils` doesn't require numpy.
- Do NOT restructure packages for this — function-local imports are the
  established repo convention for heavy deps (see PROJECT.md design decision
  on lazy imports).

### P1.3 `[ ]` CI Tier B job (Qt + pipeline, Linux) 
**Files:** `.github/workflows/ci.yml`
- New job on `ubuntu-latest`: install CPU torch
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`),
  `faster-whisper`, `PySide6`, `numpy`, `pytest`; env
  `QT_QPA_PLATFORM=offscreen`; also `apt-get install -y libegl1 libgl1` (PySide6
  runtime needs them even offscreen). Run `pytest -m "qt or pipeline"`.
- Accept that this job is slow (~5–10 min). Run it on `pull_request` only, not
  every push, if it becomes annoying.
- **Windows Tier B is a stretch goal with kill criteria:** attempt
  `windows-latest` only after the Linux job is stable, and give it at most
  two working attempts. If either attempt exceeds ~15 minutes wall time or
  fails on torch install/download, stop: keep Windows at Tier A +
  `compileall` permanently (that is D5's default, not a failure), record the
  outcome here, and move on.

### P1.4 `[ ]` Compile-all replaces the hand-maintained file lists
**Files:** `.github/workflows/ci.yml`, `tests/smoke_cli.py`
- Replace both hard-coded `py_compile` lists with
  `python -m compileall -q main.py app.py utils.py models.py diarization.py diar_backends.py services ui_qt`
  so new modules can't be silently omitted.

**Phase 1 exit criteria:** `pytest -m unit` green on ubuntu+windows CI;
`pytest -m "qt or pipeline"` green on ubuntu CI; zero test modules that run in
no CI job (document any deliberate exceptions here).

---

## Phase 2 — Decompose `ui_qt/main_window.py` (est. several weekends, incremental)

Goal: `main_window.py` under ~1,200 lines; each extracted module has focused
tests. **Order matters** — each step is chosen to be mechanical and
independently shippable. After each step: full manual smoke on Linux
(transcribe a file, cancel mid-run, force-stop, batch of 2 files, live capture
start/pause/resume/stop) and at minimum a launch + single transcription on
Windows [XOS].

### P2.1 `[ ]` Extract `ui_qt/transcription_worker.py`
- Move `TranscriptionWorker`, `_transcription_process_entry`,
  `DiarBackendProbeWorker` verbatim. Update imports in `main_window.py` and in
  `tests/test_qt_main_window_worker.py`.
- **Spawn-safety note [XOS]:** `_transcription_process_entry` is the target of
  a `spawn` Process — its module must be importable without side effects on
  both OSes. Moving it OUT of the giant main_window module actually *improves*
  spawn startup time (child no longer imports the whole UI). Verify a
  transcription still runs end-to-end on Windows after the move (spawn
  re-imports by module path; a bad move breaks Windows first).

### P2.2 `[ ]` Extract `ui_qt/batch_queue.py`
- Move `BatchQueueItem`, `BatchQueueModel`, and the `_on_*queue*` /
  `_process_next_batch_item` handlers into a `BatchQueueController(QObject)`
  owned by MainWindow. MainWindow keeps only signal wiring.
- Move/extend `tests/test_batch_queue.py` accordingly.

### P2.3 `[ ]` Extract `ui_qt/theme.py`
- Move `_apply_theme` QSS blocks, `_sanitize_theme_mode`,
  `_effective_theme_mode`, `_set_theme_mode`, `_progress_color`,
  `_set_bar_color`. Pure functions where possible (pass the palette in, return
  stylesheet strings) so they're unit-testable without a QApplication.

### P2.4 `[ ]` Extract `ui_qt/live_controller.py`
- Move the ~25 `_live_*` / live-session methods into `LiveModeController`
  that owns the `LiveSessionController` (service layer) plus the Qt audio
  source. MainWindow keeps the widgets; controller exposes signals
  (`transcript_changed`, `status_changed`, `session_error`, `elapsed_tick`).
- **This is the riskiest step** (PROJECT.md fragile area). Do it last among
  the extractions if momentum is uncertain. Full live-mode manual pass on
  Linux required: start/pause/resume/stop, cancel-with-confirm, force-stop,
  rename-with-title, VRAM preflight dialog. On Windows, verify graceful
  behavior when loopback is unavailable (live mode is Linux-first — the UI
  must degrade, not crash).

### P2.5 `[ ]` Extract save logic into `services/output_service.py`
- Move `_save_payload_for_mode`, `_auto_save_completed_parts`,
  `_default_output_dir` logic into service-layer functions taking explicit
  arguments (no widget access). MainWindow becomes a thin caller.
- This creates the seam for the listener to reuse identical output naming
  later (`<stem>_transcript.txt` etc.) — don't wire the listener in this
  phase, just make the seam.

### P2.6 `[ ]` Extract `ui_qt/hw_monitor.py`
- Move `start_hw_monitor` / `stop_hw_monitor` / `_hw_monitor_worker` into a
  `HardwareMonitor(QObject)` with a `sample` signal.
- **Prescribed pattern:** `HardwareMonitor` lives on a `QThread` (worker-object
  pattern, same as `TranscriptionWorker`), emits a `sample(dict)` signal, and
  never touches widgets; MainWindow connects the signal to a slot that updates
  labels on the GUI thread. If the current code mutates widgets directly from
  its raw `threading.Thread`, that is a latent cross-thread bug — fixing it is
  in scope for this task, using exactly this pattern (do not invent a third
  threading style).

**Phase 2 exit criteria:** `main_window.py` ≤ ~1,200 lines; all extracted
modules importable without instantiating MainWindow; Tier B CI green; manual
smoke matrix (below) passed on both OSes.

---

## Phase 3 — Process plumbing & typed errors (est. 1–2 weeks)

### P3.1 `[ ]` `services/process_worker.py` — shared `SpawnedJob` (F7)
- API sketch:
  ```python
  job = SpawnedJob(target=..., args=(...))          # always spawn ctx
  job.start()
  job.pump(dispatch: Callable[[dict], None],        # drains queue, calls back
           cancel_check: Callable[[], bool],
           poll_interval=0.05) -> None              # returns when child exits
  job.stop(reason: str, wait_timeout: float) -> bool  # terminate→kill→join
  job.close()                                        # queue close/join_thread
  ```
- Port call sites one per commit, in this order (increasing risk):
  1. `transcription_service._run_diarization_backend_in_subprocess`
  2. `ui_qt/transcription_worker.TranscriptionWorker.run`
  3. `services/live_transcription_service` (`_start_asr_process`/`_stop_process`)
- Behavior parity is the requirement: preserve the late-event drain after
  child exit and the force-stop semantics. The existing
  `tests/test_qt_main_window_worker.py` fakes (queue/event/process) should
  port to `SpawnedJob` tests.
- **[XOS]** Windows: `terminate()` on Windows is `TerminateProcess` (no
  SIGTERM grace) — the existing code already tolerates this; keep timeouts
  identical.

### P3.2 `[ ]` Typed error codes across process boundaries (F6)
- Child processes classify exceptions where the exception *type* is available
  and emit `{"type": "error", "code": <code>, "value": str(exc)}`.
  Codes: `cuda_init`, `cuda_oom`, `cuda_runtime`, `ffmpeg`, `model_download`,
  `cancelled`, `unknown`.
- One shared classifier in `services/errors.py`:
  `classify_exception(exc) -> str` — checks types first
  (`torch.cuda.OutOfMemoryError`, `ffmpeg.Error`, ...; guard imports), string
  markers second (move the existing marker tuples here so they live in exactly
  one place).
- Parents switch retry decisions to codes:
  `_should_retry_diarization_on_cpu` → `code in {"cuda_init","cuda_runtime","cuda_oom"}`;
  same for the CPU fallback in `_transcription_process_entry`. Keep the
  string-marker fallback for `code == "unknown"` so behavior never regresses.
- Unit tests: each code path via synthetic exceptions.

### P3.3 `[ ]` Optional: blocking queue gets (perf polish)
- Replace `get_nowait` + `time.sleep(0.05)` pump loops with
  `queue.get(timeout=0.1)` inside `SpawnedJob.pump`, checking cancel between
  gets. Only after P3.1/P3.2 are stable. Skip if time-constrained.

**Phase 3 exit criteria:** one drain/stop implementation in the codebase
(`grep -rn "get_nowait" services ui_qt` shows only `SpawnedJob` internals and
`LiveSessionController.poll_events`); retry logic driven by codes; parity
manual test: force-stop mid-transcription on both OSes leaves no orphan
python processes (check Task Manager / `ps`).

---

## Phase 4 — Packaging, install paths & listener hardening (as motivation allows)

### P4.1 `[ ]` Declare dependencies in `pyproject.toml` (F3) [XOS]
**Decision (D2 resolved):** NVIDIA GPU is the only supported product target.
No CPU-only install path, no `requirements-cpu.txt`. CPU execution remains a
*fallback/retry path at runtime* (existing CUDA→CPU retries stay) and a *CI
test vehicle* (Tier B installs CPU torch to run tests) — but it is not a
documented install target, and README should say "NVIDIA GPU with CUDA 12+
required" plainly rather than "highly recommended".
- `[project] dependencies`: the true core (faster-whisper, ffmpeg-python,
  numpy, PySide6, gradio, requests, PyYAML, Pillow, psutil, huggingface-hub,
  ctranslate2, tqdm) — *unpinned or loosely pinned* (`>=` floors). Torch stays
  out of core: it is installed via the documented two-step cu121 index install
  (extras can't carry `--index-url`).
- `[project.optional-dependencies]`:
  - `diarization`: `pyannote.audio`
  - `ocr`: `pytesseract`, `paddleocr`, `paddlepaddle`
  - `granite`: `transformers`, `peft`
  - `dev`: `pytest`, `ruff`
- Keep `requirements.txt` as the pinned GPU-dev lockfile (add a header comment
  saying exactly that).
- Gate optional imports: transcribing with Granite when `transformers` isn't
  installed must produce a clear "install pyscribe[granite]" error, not an
  ImportError traceback. Same for diarization backends (diar availability
  probing already exists — reuse its reasons).
**Verify:** fresh venv per OS: torch two-step + `pip install .` →
`pyscribe --help` works and transcribes the bundled benchmark MP3 with model
`tiny`; `pip install .[dev]` → Tier A tests pass; and
`pip install .[diarization,ocr,granite,dev]` resolves and installs cleanly
(the combined-extras case is where dependency conflicts hide).

### P4.2 `[ ]` Listener per-session cancel state (F5)
**Decision (D1 resolved):** multi-user LAN was a loose future idea with no
confirmed use case. Plan accordingly: do the cheap correctness work now (this
task) so the door stays open, but build nothing multi-user-specific beyond it
(no per-user prefs, no user-scoped job queues) unless a real use case shows
up. Plan: keep `default_concurrency_limit=1`, but key cancellation to the
running job:
  generate a job id per `transcribe()` call, store the active job id +
  `threading.Event` in a small module-level registry, and have the Cancel
  handler receive the session's job id via `gr.State` so it only cancels its
  own job. A second session's cancel becomes a no-op with a status message.
- Stop persisting per-job option changes to the shared config file from the
  listener (`transcribe()` currently saves on every run — move to an explicit
  "save as defaults" checkbox or drop entirely). This also closes the
  Qt-vs-listener last-write-wins race in practice.

### P4.3 `[ ]` Listener transport docs (review misc)
- README + user guide: one paragraph stating LAN mode is plain HTTP (basic
  auth credentials and media are not encrypted in transit) and recommending a
  reverse proxy (Caddy/nginx) or SSH tunnel for anything beyond a trusted
  home LAN. Documentation only — no code.

### P4.4 `[ ]` Config write coordination follow-up (F4 residual)
- After P4.2 removes routine listener writes, assess whether locking is still
  needed. If yes: lightweight cross-OS lock via `msvcrt.locking` /
  `fcntl.flock` behind one helper, or adopt `filelock` (pure-python, tiny) as
  a dependency. Don't build this before measuring that it's still a problem.

### P4.5 `[ ]` Full documentation audit (closeout)
The per-task docs rule keeps docs in sync with *changes made by this plan*;
this task catches everything else — pre-existing drift and cross-document
inconsistency accumulated over Phases 0–4. Do it last in Phase 4 (or first
thing in Phase 5 if Phase 4 drags).
- Sweep every committed doc against actual shipped behavior:
  `README.md`, `docs/user_guide.md`, `docs/qt_help.md`, `PROJECT.md`
  (architecture overview + fragile-areas list — module names change in
  Phase 2), `STACK.md`, `CONTRIBUTING.md` (test commands change in Phase 1),
  `SECURITY.md` cross-references (policy text itself stays untouched per
  ground rule 5), `assets/prompts/README.md`, and the systemd example unit.
- Specific known checks:
  - README "Testing" section must show the tiered pytest commands from
    Phase 1, not the old two-command example.
  - README "CLI / Packaging" section must match P4.1 reality (extras,
    torch two-step, the note about undeclared packaged data).
  - PROJECT.md architecture overview must list the new `ui_qt/` and
    `services/` modules from Phases 2–3.
  - Screenshots: retake any that no longer match the UI; otherwise note
    their date.
  - `docs/qt_help.md` (in-app help) reflects any renamed controls.
- Method: open each doc side-by-side with the app/code and mark every claim
  as verified/fixed/removed. Commit per document or per small group.
- Update `docs/architecture_review_2026-07.md` with a short "status as of
  completion" postscript listing which findings were resolved (F1–F10) so the
  review doc doesn't read as a list of open problems forever.
- Finally, archive this file's role: mark remaining unstarted items as moved
  to PROJECT.md's roadmap (or explicitly dropped), so HANDOFF.md ends as a
  record, not a stale to-do list.

**Phase 4 exit criteria:** `pip install .` works on both OSes (after the
documented torch two-step); README states the NVIDIA GPU requirement plainly;
two browser sessions on a LAN listener can't cancel each other's jobs;
P4.5 documentation audit complete (every committed doc verified against
shipped behavior).

---

## Phase 5 — Deferred / opportunistic

Ordered roughly by value; pick from the top when there's spare time.

- **P5.1** `[ ]` Focused review of `services/multimodal_service.py` (1,278
  lines) and `diar_backends.py` — the two modules the July 2026 review
  explicitly did NOT deep-read (low-confidence areas). Do this after the
  Phase 3 process-worker refactor: check for dead code, cross-OS path/subprocess
  issues, and whether either should adopt `SpawnedJob`. Output: findings note
  appended to `docs/architecture_review_2026-07.md` or a follow-up doc.
- **P5.2** `[ ]` Windows live-capture support decision (Open Decision D4):
  either implement WASAPI loopback verification or explicitly label live mode
  "Linux only" in the UI instead of "Linux-first".
- **P5.3** `[ ]` Listener UI parity for output save modes via the
  `output_service` seam from P2.5.
- **P5.4** `[ ]` `mypy` on `services/` only (the layer with the best typing
  already).
- **P5.5** `[ ]` Ratchet ruff rules (add `B`, `UP`, `SIM` families
  module-by-module).
- **P5.6** `[ ]` Logging: per-process log filenames for concurrent app
  launches (review misc; rare scenario).

---

## Manual smoke matrix (run at each phase exit)

| Check | Linux | Windows |
|---|---|---|
| `python main.py` launcher menu + 5s timeout autostart | required | required |
| Qt: transcribe bundled `assets/benchmark-sherlock-holmes-en.mp3` (tiny model) | required | required |
| Qt: cancel mid-transcription; then force-stop a fresh run; no orphan processes | required | required |
| Qt: batch queue with 2 files incl. same-named files from different folders | required | best-effort |
| Qt: live capture start→pause→resume→stop; session folder recoverable | required | verify graceful degrade |
| Qt: diarization on a short file (pyannote subprocess path) | required (GPU+CPU-retry if possible) | best-effort |
| Listener localhost: upload→transcribe→cancel→save/download | required | required |
| Listener LAN (`--allow-nonlocal-host` + auth): reachable from second device, auth enforced | best-effort | best-effort |
| Old `~/.pyscribe_config.json` from `main` loads without loss | required | required |

---

## Open decisions (defaults an agent may assume if unanswered)

- **D1 — LAN listener concurrency: RESOLVED (2026-07-17).** Multi-user LAN
  was "eventually planned" but has no confirmed use case. Do P4.2's job-id
  cancel keying (cheap correctness + keeps the door open); build nothing
  further for multi-user unless a concrete use case emerges.
- **D2 — CPU-only support tier: RESOLVED (2026-07-17).** Not a target.
  NVIDIA GPU is the supported configuration; CPU remains only a runtime
  fallback path and a CI test vehicle. P4.1 updated accordingly
  (no `requirements-cpu.txt`).
- **D3 — `pyscribe` console script:** keep and fix (P4.1) or remove?
  *Default:* keep and fix.
- **D4 — Windows live capture:** invest or label Linux-only?
  *Default:* label accurately in UI/docs now (Phase 0/2 timeframe), decide
  investment later (Phase 5).
- **D5 — Tier B CI on Windows:** worth the runner minutes?
  *Default:* Linux-only Tier B; Windows runs Tier A + compileall.

---

## Cross-cutting risks & rollback

- Every phase branch merges to `main` only after its exit criteria pass; if a
  regression surfaces post-merge, revert the specific task commit (tasks are
  atomic by design) rather than the phase.
- Phase 2 (MainWindow) and Phase 3 (process plumbing) must NOT run
  concurrently in separate sessions — they touch the same worker code.
  Sequence them.
- Anything touching `_transcription_process_entry`, live mode, or loader
  env setup gets a mandatory Windows verification before `[x]` — spawn and
  DLL-path behavior are where Windows breaks first.
