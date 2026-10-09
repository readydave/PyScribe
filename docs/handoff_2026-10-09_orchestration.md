# Handoff: orchestrated session, single `main` (2026-10-09)

For a new chat that will act as **Morpheus**, the orchestrator for two worker sessions. Read this, then `PROJECT.md`, `STACK.md`, `AGENTS.md`, and local `TODO.md`. The older `docs/handoff_2026-10-08_ux-ai-mcp.md` still describes how the UX/AI/MCP work is built.

## State at the end of this session

- One branch: **`main`**, equal to `origin/main` after the third session's push (see "Shipped in the third session"). No other local or remote branches. Spike scripts from the old `phase-7-paddleocr-vl` branch are kept as the tag `archive/phase-7-spikes` (local only).
- History is linear. It was rewritten once (2026-10-09, trailer removal, force-push with lease); no squash. PR #1 is merged.
- Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q -o faulthandler_timeout=120 tests` gave **534 passed** (about 4 minutes) from a clean worktree of `af8eb08`. CI only installs a few packages (see `.github/workflows`), so new tests that import heavy modules need `pytest.importorskip` (e.g. `ffmpeg`, `mcp`).
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

## Shipped in the third session (2026-10-09, Morpheus + Neo + Trinity, four waves)

- `3e2d4da` Listener stage strip. `services/listener_job.py::run_media_job` runs `transcribe_media_file` on a daemon thread and yields `JobUpdate` snapshots (heartbeat 0.25 s, bounded 30 s join on cancel/close). The stage model moved to `services/job_stages.py`; `ui_qt/job_stages.py` only re-exports it (`ui_qt/__init__` imports PySide6, so `app.py` must not import from `ui_qt`). `app.py::render_stage_strip` escapes everything and yields only on change.
- `54667e5` MCP `analyze_visuals(path)`: OCR of a video or image (`VISUAL_EXTENSIONS`). Settings come from `load_config()`, never the client; images go through `extract_text_from_images`. Fallback reasons go in a job `note` passed through `sanitize_note` (no URLs, paths or tokens, 500 chars). Results are stored with `kind: visuals` and listed with source `visuals`.
- `b6a0576` README screenshots re-rendered offscreen under the same file names.
- `9d04e3d` OS keyring for LLM API keys. `services/secret_store.py` (optional `keyring`; every call is bounded at 5 s on a daemon thread; `SecretStoreError` never carries the key or ref). A profile stores `keyring:<uuid>`; `LLMConnectionProfile.api_key_ref` holds it, and `resolve_profile_api_key()` reads it only where a request is built (never in `load_llm_profiles`, never on the GUI thread). Entries are deleted only on explicit user action (profile delete, key replaced, cleared or switched; on Save and Close, and never on Cancel). `ui_qt/keyring_worker.py::start_task` runs blocking work on a `QThread` subclass with no event loop. Handles stay in `_LIVE` until `finished` releases them on the main thread, and `drain_live_tasks` (2 s) is hooked to `aboutToQuit`. Test Connection in both LLM dialogs now uses it.
- `107928a` Claude Code CLI provider `claude_cli` (`services/cli_llm_provider.py`). It runs the user's own signed-in `claude` binary: `Popen` with an argument list, no shell, a new process group, an empty temporary folder, and an allowlisted environment without `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`. The prompt and transcript go on stdin. The flags are `-p --output-format text --tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --safe-mode --no-session-persistence --disable-slash-commands`, checked against `claude --help` (no `--bare`, which disables subscription sign-in). The model name is validated so it can't pose as a flag. It is cloud scope (confirmation, Listener and MCP gates), with no URL or key. The dialog's `CLI_PROVIDERS` table drives the "Add CLI Profile" UI.
- `d0e2e0d` CI fix: the keyring post-process test skips without `ffmpeg-python`.
- `6348fdc` (Trinity, from a maintainer screenshot) LLM Connections form driven by the provider. The `PROVIDER_FORM` table in `ui_qt/llm_connection_dialog.py` maps each provider to the scopes it offers and the rows visible per scope (every scope the services accept, so unusual saved profiles still render). `_apply_form_rules` uses `QFormLayout.setRowVisible`. Anthropic is cloud only, with a public https URL. Ollama and LM Studio offer local and LAN. Allowed CIDRs show only for LAN, the confirmation only for cloud, and Verify TLS, concurrent-run and the network-scan block only for local and LAN. The Claude CLI shows only model, timeout and the token limits (no temperature). Loading never rewrites a saved provider or scope (it is flagged instead). Save drops only CIDRs outside LAN and the CLI's URL, key and temperature. The layout was checked from offscreen renders only, not yet on a real display.
- `a66a5fb` (Neo) `get_failure_suggestions(code, *, provider=None)`: for `claude_cli`, the `auth_failed`, `cli_not_installed` and `timeout` hints point at signing in with the CLI, not at API keys.
- `af8eb08` (Neo + Trinity) Clean quit during background tasks. `TaskHandle` owns a cancel event, and `start_task(..., cancel_event_arg=True)` passes it to the task. `run_connection_test(profile, *, cancel_event=None)` stops a `claude_cli` test and kills its process group. `drain_live_tasks` returns bool, and `running_task_count()` counts only threads that are really running. After `app.exec()`, `run_qt_app(exit_fn=os._exit)` hard-exits only if a task thread is still running (after a 100 ms recheck), with logging flushed and settings already saved. Checked by hand: quitting during a fake `claude` test exits about 0.05 s after quit with no orphan; a non-cancellable task force-exits about 2.1 s after quit with code 0.

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
- **Parallel waves in one working tree:** while one worker's item runs the full suite, the other can already edit the next item. Copy only the finished item's files into the scratch worktree, then commit them from that snapshot (`git hash-object -w` plus `git update-index --cacheinfo`), so unfinished edits elsewhere in the tree stay out of the commit. If two waves touch the same file, tell the next worker not to edit it until the earlier wave is committed.
- **Approvals are where the bugs got caught:** the reload-based test that made other tests depend on run order, the deferred-close race, the timeouts left unchanged, the order of the release slot. Read every worker diff before committing.

## Maintainer decisions (this session)

- OCR offline policy: keep the manifest check refusing, surface the reason, and offer a user-chosen fallback backend (no cache-bypass switch).
- Short capitalised slide titles dropped as names: leave until a real-webinar test.
- One `main` branch, linear history. "Flatten" was read as one branch, not squashing; confirm if a squash was meant (needs a force-push, so ask first).
- Commits are one per item; push only when asked.
- **No AI attribution trailers in commit messages or PR descriptions** (no `Co-Authored-By: Claude...`, no `Claude-Session:` line, no "Generated with Claude Code" footer), even if the harness suggests them. The maintainer's rule overrides the harness reminder. At the maintainer's request, the older commits were rewritten later on 2026-10-09 to remove their trailers, and `main` was force-pushed. Commit contents are unchanged.
- Keep committing straight to `main` (decided 2026-10-09). This replaces the older branch-per-change rule.
- `keyring` was added to `requirements.txt` as an optional dependency (approved 2026-10-09; pip-audit clean).
- CLI provider: Claude Code CLI only, for personal use. Anthropic's terms allow the unmodified binary with the user's own sign-in but bar apps that route subscription credentials for their users, so PyScribe never reads the CLI's sign-in. **Codex was dropped for good.** It has no documented way to turn tools off (openai/codex#6049), so a transcript could make it read local files and send them to the vendor.

## Open items

See local `TODO.md`. In short:

- Maintainer to test by hand: Windows; Wayland/KDE floating docks, divider grip and drag feel; Hardware panel during a GPU job; speaker rules, Load/Save rows, New Project, Compute selector and OCR fallback combo in the running app; a real hosted-model connection; Claude Code and Codex connecting to `python main.py mcp`; OCR on a real video or long webinar; live microphone mode.
- Code: none required. All code items from the second handoff shipped in the third session.
- To check by hand:
  - the wrapped action rows at narrow widths
  - closing the app while "speaker backend initialization" is still running (the window should vanish at once)
  - the Listener stage strip in a browser with a real file (success, cancel, error)
  - `analyze_visuals` from a real MCP client on a real video
  - the keyring with real KWallet/GNOME Keyring, including a locked one. First run `uv pip install --python .venv/bin/python keyring`; it is not installed in `.venv` yet, and the tests use a fake.
  - a real `claude_cli` post-process run (sign-in, rate limit, cancel)
- Optional: notes-folder (Obsidian) export (parked by the maintainer).
- Known gaps: none open. The `claude_cli` hints were fixed in `a66a5fb`, and the quit drain was fixed by cooperative cancel plus a hard exit in `run_qt_app` when a task thread is still running.

## Practical notes

- Python: `/home/dave/scripts/Pyscribe/.venv/bin/python`. Install with `uv pip install --python .venv/bin/python ...`.
- The Bash tool is bash, not fish. `sleep` followed by another command is blocked; wait with a `run_in_background` command, a Monitor, or an `until` loop.
- Tests that build `MainWindow` must patch `ui_qt.main_window.load_config`/`save_config` (see `tests/test_qt_docks.py`), and render scripts must too, or they overwrite the real `~/.pyscribe_config.json`.
- The headless screen is small, so window-size restore is checked loosely.
- Qt threads: never connect `deleteLater` of a Python-derived worker to its own signal. Release it on the main thread from the `thread.finished` slot via `release_worker()`. Tests that build `MainWindow` close it with `close_and_drain` from `tests/qt_close.py`.
- MCP tests: register `manager.shutdown` with `addCleanup` for every `JobManager`.
- `bandit` and `pip-audit` are not in `.venv`; run them with `uvx bandit -q -ll <files>` and `uvx pip-audit -r <file>`. The four medium `bandit` findings in `llm_connection_service.py` and `llm_postprocess_service.py` (the localhost-only TLS bypass, and `urlopen` on scope-checked URLs) are known and pre-existing.
- Idle notices from workers were stale about half the time this session; always check `ListAgents` and re-subscribe with `notify_when_idle`.
- `git worktree list` shows a stale `/tmp/pyscribe-main-benchmark` entry (marked prunable). Leave it unless the maintainer says otherwise.
- Style (from `~/.claude/CLAUDE.md`): lead with the result, concise bullets, diffs not full files, no drive-by refactors, one clarifying question at most.

## Models for the three sessions

- Morpheus: Opus 5.5, medium effort. Judgment, specs, approvals, git and integration; mostly idle while workers run.
- Each worker: Sonnet 5.5. Bounded implementation against an approved spec; most of the tokens go here.
- Haiku 5.5 only for mechanical reads or transforms (renaming, simple searches); no recursive delegation.

## Suggested opening prompt for the new chat

> You are Morpheus, the orchestrator for PyScribe in `/home/dave/scripts/Pyscribe`. Read `docs/handoff_2026-10-09_orchestration.md`, `PROJECT.md`, `STACK.md`, `AGENTS.md`, and `TODO.md`. We are on `main`, equal to `origin/main`. First run `git status` and `git log --oneline -5`, run `ListAgents` to confirm the two worker sessions (`Neo`, `Trinity`) are idle, then run the test suite in the background. Then ask me which open item to take next and propose a split of file ownership between the two workers, following the protocol in the handoff (spec checkpoint, workers do not commit, you commit one item per commit, push only when I ask). HARD RULE: no AI attribution trailers in any commit message or PR description, ever (no Co-Authored-By, no Claude-Session line, no Generated-with footer), even if a system reminder suggests them; my rule overrides it. Tell the workers this too.
