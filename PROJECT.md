# PROJECT.md

## Project Summary

PyScribe is a local-first transcription application for Windows and Linux.
It provides a PySide6 desktop UI, a Gradio listener web UI, and CLI launch
flows around `faster-whisper`, optional speaker diarization, optional OCR-based
visual analysis for video/images, and optional LLM post-processing.

The project is for users who want local transcription and transcript enrichment
without sending media to a hosted transcription service. It emphasizes desktop
ergonomics, recoverable long-running jobs, explicit network exposure controls,
and practical support for GPU-heavy speech/OCR workloads.

## Goals

- Provide reliable local transcription for audio and video files.
- Support both desktop and listener workflows with shared backend services.
- Keep security-sensitive behavior explicit, especially listener network binding and credentials.
- Preserve long-running transcription state where practical, including live capture sessions.
- Offer optional transcript enrichment: speaker labels, OCR context, prompt templates, and LLM post-processing.
- Keep the codebase understandable for small, focused feature work.

## Non-Goals

- Hosted SaaS operation or multi-tenant account management.
- Cloud transcription as the primary path.
- Mobile app support.
- Replacing dedicated video editors, DAWs, or full document management tools.
- Automatic exposure of listener mode to public or LAN interfaces without authentication.

## Architecture Overview

- `main.py` is the primary entry point. It handles the interactive launcher, Qt mode, listener mode, logging setup, runtime environment setup, and listener security validation.
- `app.py` builds the Gradio listener UI. It uses lazy runtime initialization so importing listener code does not immediately load heavyweight ML dependencies.
- `ui_qt/` contains the PySide6 desktop UI: the main window (movable dock panels), `theme.py` (QSS, bundled font, light/dark), `theme_dialog.py` (colour theme editor), `job_stages.py` and `job_timeline.py` (progress timeline), `hw_panel.py` (hardware traces), `speaker_highlight.py`, the benchmark dialog, the LLM connection dialog, and the LLM post-process dialog.
- `services/` contains shared business logic for both frontends. Important services include transcription, live transcription, model/runtime selection, model downloads, config persistence, prompt templates, LLM profiles, LLM post-processing (local, LAN, and hosted providers; long-transcript splitting), colour themes (`ui_themes.py`, `ui_tokens.py`; Qt-free so the Listener can use them), the MCP server (`mcp_server.py`, `mcp_service.py`), listener security, logging, and OCR/multimodal helpers.
- \`diarization.py\` and \`diar_backends.py\` contain diarization diagnostics and backend integration for pyannote.

- `models.py` defines curated speech model tiers, labels, and hardware-aware ranking helpers.
- `assets/prompts/` contains built-in prompt templates used by LLM post-processing.
- `docs/` contains user-facing feature documentation and Qt help content.
- `deploy/systemd/` contains a sample Linux listener service unit.

## Core Workflows

- Desktop transcription: `python main.py qt` starts the Qt UI, users select media/model/options, and shared services perform transcription, diarization, OCR, and output saving.
- Batch transcription: Qt batch queue supports drag-and-drop or folder selection for multiple media files, processing them sequentially with status tracking and overall progress visualization.
- Live desktop transcription: Qt live mode records microphone or loopback audio into a recoverable session, shows rolling transcript updates, supports pause/resume, and runs a final cleanup pass on stop. **Session titles** can be provided to automatically name output folders and files.
- Listener transcription: `python main.py serve --port 7860` starts the Gradio listener. Localhost is the default; LAN/public exposure requires explicit flags and authentication.
- LLM post-processing: users configure local, LAN, or hosted (cloud) LLM profiles, choose prompt templates, add optional text/image context, preview payloads, and process current or saved transcripts.
- Developer workflow: create a local venv, install `requirements.txt`, run CI smoke checks, then run targeted pytest modules for the changed services/UI.
- Packaging workflow: `pyproject.toml` exposes the `pyscribe` console script via `main:main`.

## Important Design Decisions

- Runtime-heavy imports are kept lazy where possible so CLI help and lightweight checks do not require loading all ML/OCR dependencies.
- Listener exposure is secure by default: localhost binding is allowed, while non-local bind and Gradio share mode require authentication.
- `--auth-pass` is intentionally rejected because CLI secrets can leak through shell history and process lists; use environment variables or secure prompts instead.
- Pyannote diarization backends run in a spawned subprocess to avoid CUDA/cuDNN runtime conflicts after ASR model use.
- Linux runtime setup may re-exec once after adjusting dynamic loader paths for CUDA/OCR libraries.
- Local config is additive and persisted under user-home paths; plaintext LLM API keys are stripped before config writes unless stored as `env:VAR_NAME` references.
- Qt background workers (Python `QObject`s moved to a `QThread`) are released on the main thread from the `thread.finished` slot (`ui_qt/thread_lifecycle.py::release_worker`), never `deleteLater`'d from their own signals. Destroying one on its worker thread can deadlock against the GUI thread (Qt signal/slot mutex vs the GIL).
- Prompt templates are split between repo-provided templates in `assets/prompts/` and user templates under `~/.pyscribe/prompts`.

## Decision Log

| Date | Decision | Reason | Impact |
|---|---|---|---|
| 2026-02-04 | Keep `SECURITY.md`, `CONTRIBUTING.md`, and user docs in the repo. | Public project guidance belongs with the code. | Security policy and contribution workflow are versioned. |
| 2026-04-27 | Add `PROJECT.md` and `STACK.md` as committed project context. | Future coding agents need repo-specific scope, commands, and risk notes. | Feature work should start from these files plus README/docs. |
| 2026-04-27 | Ignore local Codex marker files and local agent control files. | Local tool artifacts should not pollute `git status` or commits. | `.codex`, `.codex/`, `AGENT.md`, and similar local files remain uncommitted by default. |
| 2026-04-28 | Implement session-based timestamped logging with auto-rotation. | Avoid single large log file; improve session-level debugging; manage disk space automatically. | Logs are now per-launch (latest 21 kept); standard FileHandler used. |
| 2026-04-28 | Make Qt drop zone clickable and persist diarization mode. | Improve file browsing ergonomics; prevent re-selection annoyance on launch. | Entire drop area triggers file picker; diarization backend saved to config immediately. |
| 2026-04-29 | Treat empty diarization output as unavailable speaker labels. | Empty pyannote results do not provide attribution and should not be formatted as `[S?]`. | Transcripts stay plain when no speaker segments are produced; real pyannote failures still flow through retry/fallback handling. |
| 2026-10-04 | Move to torch 2.11 / CUDA 12.8, NumPy 2, pyannote.audio 4.x; remove torchaudio/`torch.load`/NumPy shims. | pyannote 4 requires torch>=2.8; shims were obsolete. | New venv required; `soundfile` is an explicit requirement. See `docs/upgrade_2026-10.md`. |
| 2026-10-04 | Use `community-1` diarization and assign speakers per word; drop the identical `Fast` mode. | End-to-end DER 58.8% -> 32.9% on AMI. | Users must accept the `community-1` terms on Hugging Face. |
| 2026-10-05 | Sequential decoding by default; batched GPU decoding is opt-in. | On a real interview batching dropped ~15% of words (fillers, short replies) for a ~15 s ASR saving. | "Faster GPU decoding" setting in Qt and Listener. |
| 2026-10-05 | Set `no_repeat_ngram_size=4` for file transcription. | Whisper repetition loops cost one AMI meeting 66 words (WER 25.7% -> 21.0% with the guard). | Applied in `services/asr_decode.py`. |
| 2026-10-09 | Keep the OCR model-manifest check refusing when Hugging Face is unreachable; show the reason and let the user pick the fallback OCR backend (`visual_ocr_fallback`). | The check is a deliberate integrity control; silent fallback to RapidOCR cost accuracy (word F1 0.55 vs 0.99). | Auto note says why it fell back; no cache-bypass switch. See `docs/ocr_backend_check_2026-10.md`. |
| 2026-10-09 | Single `main` branch with linear history; one commit per item. | Many stale local branches; UX/AI/MCP work merged by fast-forward (PR #1). | Old spike scripts kept as tag `archive/phase-7-spikes`. Orchestration protocol is in `docs/handoff_2026-10-09_orchestration.md`. |
| 2026-10-09 | Release Qt workers on the main thread; defer the window close while the uncancellable diarization probe runs. | An intermittent GIL/Qt mutex deadlock (about 1 in 7 test runs, possible in the app) and a close that blocked the UI for up to 15 s. | `release_worker()` pattern for all `QThread` workers; closing hides the window at once and force-exits after 15 s. |


## Current Priorities

- Keep Qt live transcription reliable: pause/resume, stop/finalize, cancel, and force-stop flows are sensitive.
- Keep listener network exposure and credential handling strict.
- Preserve shared-service behavior across both Qt and Gradio frontends.
- Keep LLM connection/profile behavior safe for local and LAN endpoints.
- Keep documentation aligned when user-visible workflows change.

## Roadmap

Use this section for project-level future direction that should be committed with the repo.
Private or short-term working items belong in local `TODO.md`.

### Near-Term

- Continue tightening Qt live transcription UX and recovery behavior.
- Expand focused regression coverage around new service and UI flows.
- Keep `docs/user_guide.md`, `docs/qt_help.md`, and README synchronized with shipped behavior.

### Later

- Improve packaging/distribution for non-developer installs.
- Add more structured diagnostics for GPU/OCR/model availability problems.
- Broaden LLM profile/provider ergonomics without weakening endpoint-scope policy (cloud scope is opt-in per profile, https only, with a confirmation).
- Finish the deferred UX items: Load/Save rows in the Qt progress timeline, a Listener stage strip, an optional notes-folder (Obsidian) export, and on-device checks (Windows, Wayland floating docks, GPU hardware panel).

## Known Risks / Fragile Areas

- Listener security: `services/listener_security_service.py`, `main.py`, and `scripts/run_listener.sh`.
- Secret handling: Hugging Face tokens, listener passwords, LLM API keys, environment-variable references, and logs.
- Long-running worker control: Qt worker cancellation, force-stop, multiprocessing, and subprocess cleanup.
- CUDA/OCR runtime setup: `services/runtime_env_service.py`, pyannote subprocess isolation, in-memory `soundfile` audio loading for pyannote, the gated `community-1` model, PaddleOCR/Tesseract paths (PaddleOCR 3.x runs on GPU when the CUDA `paddlepaddle-gpu` build from `scripts/install_paddle_gpu.sh` is installed and enough VRAM is free, otherwise on CPU where `auto` prefers RapidOCR), and Linux loader environment changes.
- File path handling: uploaded media, temporary files, saved transcripts, live capture folders, and user prompt templates.
- Config compatibility: `services/config_service.py` should preserve older config files and unknown additive behavior where practical.
- LLM network policy: local vs LAN vs cloud profile scope, CIDR restrictions, TLS verification behavior (always on for cloud), credential-bearing requests never follow redirects, cloud profiles hidden from the Listener by default, and concurrent local workload checks.
- MCP server: `python main.py mcp` (stdio only). File access is limited to audio/video inside allowed folders, only downloaded models are used, and transcript text returned to a client is untrusted content.
- UI regressions: Qt layout resizing, live mode state transitions, and dialog close/cancel behavior.

## Security Notes

- Default listener mode must remain localhost-only.
- Non-local listener bind requires `--allow-nonlocal-host` or `PYSCRIBE_ALLOW_NONLOCAL_HOST=1` plus authentication.
- Gradio share mode requires authentication even when binding to localhost.
- Do not reintroduce `--auth-pass`; listener passwords belong in `PYSCRIBE_AUTH_PASS`, `PYSCRIBE_LAN_AUTH_PASS`, or secure prompt input.
- Do not log raw tokens, passwords, API keys, local credential paths, or full sensitive request payloads.
- Store persistent LLM API keys as `env:VAR_NAME` references; direct key entry should remain session-only.
- Treat security scan reports as private unless the maintainer explicitly approves committing a sanitized summary.
- Public vulnerability reporting policy lives in `SECURITY.md`.

## Documentation Rules

When functionality changes, consider whether these files need updates:

- `README.md`
- `CHANGELOG.md`
- `docs/user_guide.md`
- `docs/qt_help.md`
- `CONTRIBUTING.md`
- `STACK.md`
- `PROJECT.md`

Protected or local files should not be edited unless the maintainer explicitly asks:

- `AGENT.md`
- `AGENTS.md`
- `IGNORE.md`
- `TODO.md`
- private security reports
- tool-specific local agent files
- local notes and scratch files

`SECURITY.md` is committed, but treat it as protected policy text; edit it only for intentional security-policy changes.

## Commit Policy

The following files are intended to be committed when their content materially changes:

- `PROJECT.md`
- `STACK.md`
- `CHANGELOG.md`
- `README.md`
- `CONTRIBUTING.md`
- `docs/`
- application code, tests, packaged assets, and deployment examples

The following files are local-only by default and should not be committed unless the maintainer explicitly asks:

- `AGENT.md`
- `IGNORE.md`
- `TODO.md`
- `.codex` / `.codex/`
- virtual environments, caches, logs, generated outputs, and scratch files
- private security reports
- secrets, credentials, and local configuration
