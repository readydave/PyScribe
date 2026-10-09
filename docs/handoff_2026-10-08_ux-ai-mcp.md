# Handoff: UX redesign, themes, AI providers, MCP server (2026-10-08)

For a new chat picking up PyScribe. Read this, then `PROJECT.md`, `STACK.md`, and `AGENTS.md` (the repo rules).

## Where things stand

- Repo: `/home/dave/scripts/Pyscribe` (GitHub `readydave/PyScribe`).
- Working branch: **`llm-frontier-and-context`**. History on it, oldest first, on top of `docs-handoff-terms-plus` (already pushed):
  1. `434f9b5` shared Qt theme + job progress timeline
  2. `dbf965f` movable dock layout, hardware panel, restyled Listener
  3. `bf747bd` contrast fixes, keyboard focus, tab order, docs
  4. `c1b0be2` colour themes, theme editor, speaker colours
  5. `375d275` start-up defaults, narrow-window layout fixes
  6. `15eefc5` hosted AI providers + long-transcript handling
  7. `d8ad752` MCP server
- **Nothing from this work is pushed.** `main` and `origin/*` do not have it. The maintainer has not decided between pushing the branch and opening a PR.
- The local branches `ux-redesign-phase-0-1` and `ux-redesign-phase-2` are older labels on the same line; safe to delete once the work is merged.
- Docs were updated after the last commit and are **uncommitted**: `README.md`, `docs/user_guide.md`, `PROJECT.md`, `SECURITY.md`, this file. `TODO.md` is local-only (gitignored).
- Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests` gave **365 passed** (about 90 s). Also run `python main.py --help`, `main.py mcp --help`, `main.py qt --help`.

## What was built

**Qt UI** (`ui_qt/`)
- `theme.py`: Fusion style, bundled Atkinson Hyperlegible Next font (`assets/fonts/`), one QSS built from a palette, light/dark/system (follows the OS live), themed checkboxes/scrollbars. `apply_theme(app, mode, theme_id, custom_themes)`; it skips re-applying an identical stylesheet (important for speed).
- `main_window.py`: the Transcription page is a `QMainWindow` of docks (Setup, Progress, Hardware, Batch queue) around the transcript. Layout, window size, sidebar state, and `More options` state persist in the config. Default dock layout depends on window width (`NARROW_WINDOW_WIDTH = 1500`); the window is fitted to the screen *before* the UI is built.
- `job_stages.py` + `job_timeline.py`: stage tracker (Transcribe/Speakers/Visuals) and the progress rows; old widget names (`progress_bar`, `diar_progress_bar`, `visual_progress_bar`, `*_time_label`, `terminal_log`) are kept as aliases because tests pin them.
- `hw_panel.py`: painted 60-second CPU/RAM/GPU/VRAM traces, fed by the `hw_sample` signal (only while a job runs).
- `speaker_highlight.py`: colours `[S1]`-style labels. Live timer and Stop button turn red while recording.
- `theme_dialog.py`: colour theme editor (live preview, duplicate/rename/delete, import/export, contrast notes).
- Settings page: **Start-up defaults** (model, File/Live, names/terms, faster decoding, open-files folder) are real now; the old API-key fields were removed.

**Themes** (Qt-free, shared with the Listener)
- `services/ui_themes.py`: four presets (Iron-gall, Verdigris, Ochre, Graphite), core-colour model, `derive_palette` (auto contrast fixes), strict `#RRGGBB` validation, JSON round trip. `services/ui_tokens.py` keeps the font constants and the default palettes under the old names.
- Config: `theme_mode`, `theme_id`, `custom_themes`.

**AI** (`services/llm_*.py`)
- New `cloud` scope (https only, key + `cloud_acknowledged` required, TLS always verified); providers `ollama`, `lm_studio`, `openai_compatible`, `anthropic` (native Messages API, streaming). Presets for Claude, OpenAI, Gemini, OpenRouter in the connections dialog. Model names are never hard-coded except the Claude default `claude-sonnet-5-5`.
- Per-profile `context_tokens`, `max_output_tokens`, `temperature` (None = provider default; local default 0.2). Ollama gets `num_ctx`/`num_predict`. Transcripts that exceed the budget are split, summarised per part, merged (`_run_long`), with `on_status` progress. Replies cut off at the output limit are flagged.
- `open_url` (connection service) refuses redirects when a request carries `Authorization` or `x-api-key`. Tests patch `services.llm_postprocess_service.open_url` / `services.llm_connection_service.open_url`.
- The Listener hides cloud profiles unless `llm_allow_cloud_in_listener` is true.

**MCP server** (`python main.py mcp`, stdio only; `docs/mcp.md`)
- `services/mcp_service.py`: media-path validation (extension allow-list, resolved path inside `PYSCRIBE_MCP_ROOTS` or home), transcript store (`~/.pyscribe/mcp_transcripts/`, 0600), live-session reader, one-at-a-time job manager, paging.
- `services/mcp_server.py`: tools `list_transcription_models`, `start_transcription`, `get_job`, `wait_for_job`, `cancel_job`, `list_transcripts`, `get_transcript`, `list_templates`, `get_template`. Only downloaded models; transcripts flagged as untrusted in the server instructions.
- Uses **`mcp` 2.x** (`from mcp.server.mcpserver import MCPServer, Context`, `from mcp_types import ToolAnnotations`). FastMCP no longer exists in 2.x. `requirements.txt` has `mcp>=2.3,<3`; it is installed in `.venv`.

## Not verified (needs a real machine or real accounts)

- Windows (HiDPI, dark title bar, colour dialog, bundled font), Wayland/KDE floating docks and **Reset layout**, the Hardware panel during a real GPU job.
- Any real hosted-provider call (Claude/OpenAI/Gemini). Everything was tested with mocked HTTP.
- Connecting Claude Code and Codex to the MCP server. The setup commands in `docs/mcp.md` were written from memory; confirm against current client docs.
- The headless screen is 800x800, so window-size restore is only checked loosely.

## Open items

See `TODO.md`. In short: decide the branch/PR; on-device checks above; Load/Save rows in the Qt timeline (needs worker events); Listener stage strip (needs `transcribe_media_file` on a worker thread); mono timestamp font and per-speaker margin rules; notes-folder (Obsidian) export (designed, parked on purpose: keep current save behaviour); optional CLI provider (`claude -p`/`codex exec`; check vendor terms first); MCP `run_template` and OCR; OS keyring for keys; refresh the old dialog screenshots in the README.

## Decisions the maintainer made

- Frontier models must be addable by the user; hosted use is an explicit opt-in per profile.
- Keep save behaviour as it was (no Obsidian export for now).
- Wants the PyScribe-as-MCP-server direction (Claude Code/Codex call PyScribe), not a model-driven MCP client. A human-triggered "Send to..." client feature was suggested as a later option.
- Prefers commits only when asked, never pushes unless asked, one commit per logical unit when splitting.

## Practical notes for the next session

- Python: use `/home/dave/scripts/Pyscribe/.venv/bin/python`. Install with `uv pip install --python .venv/bin/python ...` (no `pip` in the venv).
- The Bash tool is bash, not fish, even though the user's shell is fish. Avoid `pkill -f "<text that appears in the command>"`: it kills the tool's own shell.
- Tests that build `MainWindow` must patch `ui_qt.main_window.load_config` / `save_config` (see `tests/test_qt_docks.py`), or they write to the real `~/.pyscribe_config.json`. Scripts that render screenshots must do the same; one earlier script overwrote `theme_mode` and it had to be reset to `system`.
- A running full test suite can exceed the tool's 2-minute limit; run it in the background and wait.
- `/plan` is not available over Remote Control; plan in conversation instead.
- User style (from `~/.claude/CLAUDE.md`): lead with the result, concise bullets, diffs not full files, no drive-by refactors, back up risky system files first.

## Suggested opening prompt for a new chat

> Read `docs/handoff_2026-10-08_ux-ai-mcp.md`, `PROJECT.md`, `STACK.md`, and `AGENTS.md` in `/home/dave/scripts/Pyscribe`. The work is on branch `llm-frontier-and-context` (unpushed). First check `git status` and run the test suite, then ask me which open item from `TODO.md` to take next.
