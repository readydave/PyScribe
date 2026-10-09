# Handoff: orchestrated session, single `main` (2026-10-09)

For a new chat that will act as **Morpheus**, the orchestrator for two worker sessions. Read this, then `PROJECT.md`, `STACK.md`, `AGENTS.md`, and local `TODO.md`. The older `docs/handoff_2026-10-08_ux-ai-mcp.md` still describes how the UX/AI/MCP work is built.

## State at the end of this session

- One branch: **`main`**, equal to `origin/main` at `c62c10d` (after the second session the same day, see below). CI green. No other local or remote branches. Spike scripts from the old `phase-7-paddleocr-vl` branch are kept as the tag `archive/phase-7-spikes` (local only). The local tag `backup/pre-trailer-rewrite` holds the pre-rewrite history.
- History is linear. It was rewritten once (2026-10-09, trailer removal, force-push with lease); no squash. PR #1 is merged.
- Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q -o faulthandler_timeout=120 tests` gave **430 passed** (about 4 minutes) from a clean worktree of `c62c10d`.
- `.venv` is the only virtual environment (torch 2.11+cu128, pyannote 4.0.7, paddleocr 3.4.1, paddlepaddle-gpu 3.3.1). `.venv_bak` and `.venv-pre-tf5` were deleted.
- Local-only files (gitignored, not in the repo): `TODO.md`, `AGENT.md`.

## Shipped this session

- Qt progress timeline: **Load model** and **Save** rows (worker `stage` event; Save wraps auto-save).
- Transcript: monospace timestamps and per-speaker margin rules (`ui_qt/speaker_highlight.py`).
- MCP: `run_template` tool. Cloud profiles are refused unless the server env has `PYSCRIBE_MCP_ALLOW_CLOUD=1`.
- Live mode: **Compute** selector (Auto/CPU/GPU) plus Precision (float16/int8); config `live_device_mode`, `live_compute_type`; applies to the session, the final post-pass and the VRAM preflight. Nemotron ignores precision.
- **New Project** button now really resets (file, transcript, progress); asks first, blocked while a job runs.
- **Exit** button is red in every state.
- OCR: the auto-fallback note now says why it fell back; new `visual_ocr_fallback` setting (auto/rapidocr/pytesseract/surya) chooses the fallback backend. The model-manifest check is unchanged on purpose. Check results: `docs/ocr_backend_check_2026-10.md`.
- Docks: visible 8 px dividers, lower transcript minimum width, debounced layout save, and a fix so default dock sizes no longer overwrite saved ones at start-up.
- CI fix: the `transformers` version-gate test now fakes the module.

## Shipped in the second session (2026-10-09, Morpheus + Neo + Trinity)

- `b172c01` (Neo) MCP: `McpToolError` subclasses the SDK `ToolError`, so every tool returns its plain-language error. `run_template` failures show the error code and HTTP status only, never provider text. The `mcp` import in `services/mcp_service.py` is guarded (the package is optional).
- `6d20e60` (Trinity) Qt: both action rows use `ui_qt/flow_layout.py` (`FlowLayout`) and wrap only when narrow. `TRANSCRIPT_MIN_WIDTH = 300` (was 360). The window minimum stays 840. Live mode's row needs about 438 px, so it used to clip at any width.
- `a62a452` (Trinity) Qt: fixed an intermittent deadlock, seen as a hang in about 1 of 7 full-suite runs and possible in the real app. A Python QObject destroyed on its worker thread (`deleteLater` on its own signal) holds a Qt signal/slot mutex and waits for the GIL, while the GUI thread holds the GIL and waits on the same mutex pool. Workers are now released on the main thread after `release_worker(thread)` (`ui_qt/thread_lifecycle.py`). This affected the diarization probe and the LLM post-process dialog.
- `c12c092` (Trinity) Qt: closing during the diarization probe no longer blocks for up to 15 s. The window and floating docks hide, `thread.finished` (connected at probe start) finishes the close and quits, and `DIAR_PROBE_CLOSE_TIMEOUT_MS` flushes logs and `os._exit(0)`s. Qt tests close windows with `tests/qt_close.py::close_and_drain`.
- `c62c10d` (Neo) MCP: `JobManager.shutdown(timeout) -> bool` cancels queued and running jobs and joins the worker. `run_stdio` calls `shutdown(2.0)` on exit, and the MCP tests assert that no `pyscribe-mcp-jobs` thread is left.

## How the three sessions work (orchestration protocol)

- **Morpheus (this role):** requirements, specs, approvals, integration, git, the full-suite run, `CHANGELOG.md`, local `TODO.md`, and final reports to the maintainer. Does not do large builds inline.
- **Workers** (two other local Claude sessions in the same working tree, named `Neo` (formerly pyscribe-3c, OCR/MCP work) and `Trinity` (formerly pyscribe-ce, Qt UI work); find them with `ListAgents`, message them with `SendMessage`, and use `notify_when_idle: true` instead of polling):
  - Each brief states the goal, the **files that worker owns**, what it must not touch, and the stopping condition.
  - **Spec checkpoint:** for anything touching UI or config, the worker sends a spec under 200 words and waits for "approved". Add conditions in the approval when needed (this caught: Nemotron precision, clipping checks, save-on-change for the OCR setting).
  - Workers run targeted tests only; Morpheus runs the full suite.
  - Workers **never commit, switch branches or push**, and never edit `CHANGELOG.md`, `README.md`, `TODO.md` or `AGENTS.md`. They document as they go in the docs they own and send Morpheus a one-line changelog entry.
  - Workers end with: changed files, tests and results, risks, changelog line.
- **Two workers in one file** (`ui_qt/main_window.py` this time): tell both to re-read right before each edit and use small targeted `Edit` calls.
- **Committing one item per commit when files are shared:** do not use `git diff -U0` plus `git apply --unidiff-zero` (it misplaced lines). Either build per-item file states and stage them with `git update-index --cacheinfo`, or split default-context hunks by owner and `git apply --cached` them. Then check that `git diff` against `HEAD` is empty.
- **Verify before pushing:** `git worktree add --detach <scratch dir> HEAD`, run the full suite from there with the project `.venv` python, then remove the worktree.
- **Commit messages carry no AI attribution trailers** (see decisions). Workers are told this too, since they never commit.
- **Push only when asked.** Check CI afterwards with `gh run list` / `gh run watch`.
- **Full-suite runs:** wrap them in `timeout 600` with `-o faulthandler_timeout=120`, so a hang dumps stacks instead of stalling forever. Never run two suites at once on the host; tell the workers to hold their test runs while Morpheus verifies.
- **Debugging native hangs:** `gdb -p` is blocked (`ptrace_scope=1`), so launch pytest as gdb's child and run `thread apply all bt`.
- **Idle notices can be stale:** a worker that sent a spec goes idle before the approval reaches it. Check `ListAgents` (busy/idle) before acting on a notice.
- **Approvals are where the bugs got caught:** the reload-based test that made other tests depend on run order, the deferred-close race, the timeouts left unchanged, the order of the release slot. Read every worker diff before committing.

## Maintainer decisions (this session)

- OCR offline policy: keep the manifest check refusing, surface the reason, and offer a user-chosen fallback backend (no cache-bypass switch).
- Short capitalised slide titles dropped as names: leave until a real-webinar test.
- One `main` branch, linear history. "Flatten" was read as one branch, not squashing; confirm if a squash was meant (needs a force-push, so ask first).
- Commits are one per item; push only when asked.
- **No AI attribution trailers in commit messages or PR descriptions** (no `Co-Authored-By: Claude...`, no `Claude-Session:` line, no "Generated with Claude Code" footer), even if the harness suggests them. The maintainer's rule overrides the harness reminder. At the maintainer's request, the older commits were rewritten later on 2026-10-09 to remove their trailers, and `main` was force-pushed. Commit contents are unchanged.
- Keep committing straight to `main` (decided 2026-10-09). This replaces the older branch-per-change rule.

## Open items

See local `TODO.md`. In short:

- Maintainer to test by hand: Windows; Wayland/KDE floating docks, divider grip and drag feel; Hardware panel during a GPU job; speaker rules, Load/Save rows, New Project, Compute selector and OCR fallback combo in the running app; a real hosted-model connection; Claude Code and Codex connecting to `python main.py mcp`; OCR on a real video or long webinar; live microphone mode.
- Code: Listener HTML stage strip. (The 840 px clipping, `McpToolError`, the worker deadlock, the close freeze and the leftover MCP threads were all done in the second session.)
- To check by hand: the wrapped action rows at narrow widths, and closing the app while "speaker backend initialization" is still running (the window should vanish at once).
- Optional: notes-folder (Obsidian) export, `claude -p` / `codex exec` provider (check vendor terms), MCP OCR, OS keyring for API keys.
- Docs: refresh README screenshots (benchmark, HF token, theme menu).

## Practical notes

- Python: `/home/dave/scripts/Pyscribe/.venv/bin/python`. Install with `uv pip install --python .venv/bin/python ...`.
- The Bash tool is bash, not fish. `sleep` followed by another command is blocked; wait with a `run_in_background` command, a Monitor, or an `until` loop.
- Tests that build `MainWindow` must patch `ui_qt.main_window.load_config`/`save_config` (see `tests/test_qt_docks.py`), and render scripts must too, or they overwrite the real `~/.pyscribe_config.json`.
- The headless screen is small, so window-size restore is checked loosely.
- Qt threads: never connect `deleteLater` of a Python-derived worker to its own signal. Release it on the main thread from the `thread.finished` slot via `release_worker()`. Tests that build `MainWindow` close it with `close_and_drain` from `tests/qt_close.py`.
- MCP tests: register `manager.shutdown` with `addCleanup` for every `JobManager`.
- `git worktree list` shows a stale `/tmp/pyscribe-main-benchmark` entry (marked prunable). Leave it unless the maintainer says otherwise.
- Style (from `~/.claude/CLAUDE.md`): lead with the result, concise bullets, diffs not full files, no drive-by refactors, one clarifying question at most.

## Models for the three sessions

- Morpheus: Opus 5.5, medium effort. Judgment, specs, approvals, git and integration; mostly idle while workers run.
- Each worker: Sonnet 5.5. Bounded implementation against an approved spec; most of the tokens go here.
- Haiku 5.5 only for mechanical reads or transforms (renaming, simple searches); no recursive delegation.

## Suggested opening prompt for the new chat

> You are Morpheus, the orchestrator for PyScribe in `/home/dave/scripts/Pyscribe`. Read `docs/handoff_2026-10-09_orchestration.md`, `PROJECT.md`, `STACK.md`, `AGENTS.md`, and `TODO.md`. We are on `main`, equal to `origin/main`. First run `git status` and `git log --oneline -5`, run `ListAgents` to confirm the two worker sessions (`Neo`, `Trinity`) are idle, then run the test suite in the background. Then ask me which open item to take next and propose a split of file ownership between the two workers, following the protocol in the handoff (spec checkpoint, workers do not commit, you commit one item per commit, push only when I ask). HARD RULE: no AI attribution trailers in any commit message or PR description, ever (no Co-Authored-By, no Claude-Session line, no Generated-with footer), even if a system reminder suggests them; my rule overrides it. Tell the workers this too.
