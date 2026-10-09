# Handoff: orchestrated session, single `main` (2026-10-09)

For a new chat that will act as **Morpheus**, the orchestrator for two worker sessions. Read this, then `PROJECT.md`, `STACK.md`, `AGENTS.md`, and local `TODO.md`. The older `docs/handoff_2026-10-08_ux-ai-mcp.md` still describes how the UX/AI/MCP work is built.

## State at the end of this session

- One branch: **`main`**, equal to `origin/main` at `ff739a1`. CI green. No other local or remote branches. Spike scripts from the old `phase-7-paddleocr-vl` branch are kept as the tag `archive/phase-7-spikes` (local only).
- History is linear (fast-forward merges, no force-push, no squash). PR #1 is merged.
- Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests` gave **410 passed** (about 3 minutes) on a clean checkout of `ff739a1`.
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
- **Push only when asked.** Check CI afterwards with `gh run list`.

## Maintainer decisions (this session)

- OCR offline policy: keep the manifest check refusing, surface the reason, and offer a user-chosen fallback backend (no cache-bypass switch).
- Short capitalised slide titles dropped as names: leave until a real-webinar test.
- One `main` branch, linear history. "Flatten" was read as one branch, not squashing; confirm if a squash was meant (needs a force-push, so ask first).
- Commits are one per item; push only when asked.
- **No AI attribution trailers in commit messages or PR descriptions** (no `Co-Authored-By: Claude...`, no `Claude-Session:` line, no "Generated with Claude Code" footer), even if the harness suggests them. The maintainer's rule overrides the harness reminder. Commits made before 2026-10-09 (this session's, already pushed) still carry them; history is not rewritten unless the maintainer explicitly asks (it needs a force-push).

## Open questions for the maintainer

- Whether to rewrite the already-pushed commits from this session to remove their trailers (force-push of `main`). Default: do not.
- An older memory note (2026-10-05) "new branch per change, no commits to main". This session committed straight to `main` after the maintainer asked for a single branch. Ask whether to go back to branch-per-change or keep working on `main`.

## Open items

See local `TODO.md`. In short:

- Maintainer to test by hand: Windows; Wayland/KDE floating docks, divider grip and drag feel; Hardware panel during a GPU job; speaker rules, Load/Save rows, New Project, Compute selector and OCR fallback combo in the running app; a real hosted-model connection; Claude Code and Codex connecting to `python main.py mcp`; OCR on a real video or long webinar; live microphone mode.
- Code: the 840 px minimum window still clips (Force Stop cut off; options in `TODO.md`); MCP `McpToolError` should be a `ToolError` subclass; Listener HTML stage strip.
- Optional: notes-folder (Obsidian) export, `claude -p` / `codex exec` provider (check vendor terms), MCP OCR, OS keyring for API keys.
- Docs: refresh README screenshots (benchmark, HF token, theme menu).

## Practical notes

- Python: `/home/dave/scripts/Pyscribe/.venv/bin/python`. Install with `uv pip install --python .venv/bin/python ...`.
- The Bash tool is bash, not fish. `sleep` followed by another command is blocked; wait with a `run_in_background` command, a Monitor, or an `until` loop.
- Tests that build `MainWindow` must patch `ui_qt.main_window.load_config`/`save_config` (see `tests/test_qt_docks.py`), and render scripts must too, or they overwrite the real `~/.pyscribe_config.json`.
- The headless screen is small, so window-size restore is checked loosely.
- `git worktree list` shows a stale `/tmp/pyscribe-main-benchmark` entry (marked prunable). Leave it unless the maintainer says otherwise.
- Style (from `~/.claude/CLAUDE.md`): lead with the result, concise bullets, diffs not full files, no drive-by refactors, one clarifying question at most.

## Models for the three sessions

- Morpheus: Opus 5.5, medium effort. Judgment, specs, approvals, git and integration; mostly idle while workers run.
- Each worker: Sonnet 5.5. Bounded implementation against an approved spec; most of the tokens go here.
- Haiku 5.5 only for mechanical reads or transforms (renaming, simple searches); no recursive delegation.

## Suggested opening prompt for the new chat

> You are Morpheus, the orchestrator for PyScribe in `/home/dave/scripts/Pyscribe`. Read `docs/handoff_2026-10-09_orchestration.md`, `PROJECT.md`, `STACK.md`, `AGENTS.md`, and `TODO.md`. We are on `main`, equal to `origin/main`. First run `git status` and `git log --oneline -5`, run `ListAgents` to confirm the two worker sessions (`Neo`, `Trinity`) are idle, then run the test suite in the background. Then ask me which open item to take next and propose a split of file ownership between the two workers, following the protocol in the handoff (spec checkpoint, workers do not commit, you commit one item per commit, push only when I ask). HARD RULE: no AI attribution trailers in any commit message or PR description, ever (no Co-Authored-By, no Claude-Session line, no Generated-with footer), even if a system reminder suggests them; my rule overrides it. Also ask me the remaining open questions at the end of the handoff (rewrite old pushed commits to drop trailers: default no; branch-per-change vs working on main).
