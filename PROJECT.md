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
- `ui_qt/` contains the PySide6 desktop UI, including the main window, benchmark dialog, LLM connection dialog, and LLM post-process dialog.
- `services/` contains shared business logic for both frontends. Important services include transcription, live transcription, model/runtime selection, model downloads, config persistence, prompt templates, LLM profiles, LLM post-processing, listener security, logging, and OCR/multimodal helpers.
- `diarization.py` and `diar_backends.py` contain diarization diagnostics and backend integration, including pyannote and NeMo Sortformer paths.
- `models.py` defines curated speech model tiers, labels, and hardware-aware ranking helpers.
- `assets/prompts/` contains built-in prompt templates used by LLM post-processing.
- `docs/` contains user-facing feature documentation and Qt help content.
- `deploy/systemd/` contains a sample Linux listener service unit.

## Core Workflows

- Desktop transcription: `python main.py qt` starts the Qt UI, users select media/model/options, and shared services perform transcription, diarization, OCR, and output saving.
- Batch transcription: Qt batch queue supports drag-and-drop or folder selection for multiple media files, processing them sequentially with status tracking and overall progress visualization.
- Live desktop transcription: Qt live mode records microphone or loopback audio into a recoverable session, shows rolling transcript updates, supports pause/resume, and runs a final cleanup pass on stop. **Session titles** can be provided to automatically name output folders and files.
- Listener transcription: `python main.py serve --port 7860` starts the Gradio listener. Localhost is the default; LAN/public exposure requires explicit flags and authentication.
- LLM post-processing: users configure local or LAN LLM profiles, choose prompt templates, add optional text/image context, preview payloads, and process current or saved transcripts.
- Developer workflow: create a local venv, install `requirements.txt`, run CI smoke checks, then run targeted pytest modules for the changed services/UI.
- Packaging workflow: `pyproject.toml` exposes the `pyscribe` console script via `main:main`.

## Important Design Decisions

- Runtime-heavy imports are kept lazy where possible so CLI help and lightweight checks do not require loading all ML/OCR dependencies.
- Listener exposure is secure by default: localhost binding is allowed, while non-local bind and Gradio share mode require authentication.
- `--auth-pass` is intentionally rejected because CLI secrets can leak through shell history and process lists; use environment variables or secure prompts instead.
- Pyannote diarization backends run in a spawned subprocess to avoid CUDA/cuDNN runtime conflicts after ASR model use.
- Linux runtime setup may re-exec once after adjusting dynamic loader paths for CUDA/OCR libraries.
- Local config is additive and persisted under user-home paths; plaintext LLM API keys are stripped before config writes unless stored as `env:VAR_NAME` references.
- Prompt templates are split between repo-provided templates in `assets/prompts/` and user templates under `~/.pyscribe/prompts`.

## Decision Log

| Date | Decision | Reason | Impact |
|---|---|---|---|
| 2026-02-04 | Keep `SECURITY.md`, `CONTRIBUTING.md`, and user docs in the repo. | Public project guidance belongs with the code. | Security policy and contribution workflow are versioned. |
| 2026-04-27 | Add `PROJECT.md` and `STACK.md` as committed project context. | Future coding agents need repo-specific scope, commands, and risk notes. | Feature work should start from these files plus README/docs. |
| 2026-04-27 | Ignore local Codex marker files and local agent control files. | Local tool artifacts should not pollute `git status` or commits. | `.codex`, `.codex/`, `AGENT.md`, and similar local files remain uncommitted by default. |
| 2026-04-28 | Implement session-based timestamped logging with auto-rotation. | Avoid single large log file; improve session-level debugging; manage disk space automatically. | Logs are now per-launch (latest 21 kept); standard FileHandler used. |
| 2026-04-28 | Make Qt drop zone clickable and persist diarization mode. | Improve file browsing ergonomics; prevent re-selection annoyance on launch. | Entire drop area triggers file picker; diarization backend saved to config immediately. |

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
- Broaden LLM profile/provider ergonomics without weakening endpoint-scope policy.

## Known Risks / Fragile Areas

- Listener security: `services/listener_security_service.py`, `main.py`, and `scripts/run_listener.sh`.
- Secret handling: Hugging Face tokens, listener passwords, LLM API keys, environment-variable references, and logs.
- Long-running worker control: Qt worker cancellation, force-stop, multiprocessing, and subprocess cleanup.
- CUDA/OCR runtime setup: `services/runtime_env_service.py`, pyannote subprocess isolation, PaddleOCR/Tesseract paths, and Linux loader environment changes.
- File path handling: uploaded media, temporary files, saved transcripts, live capture folders, and user prompt templates.
- Config compatibility: `services/config_service.py` should preserve older config files and unknown additive behavior where practical.
- LLM network policy: local vs LAN profile scope, CIDR restrictions, TLS verification behavior, and concurrent local workload checks.
- UI regressions: Qt layout resizing, live mode state transitions, and dialog close/cancel behavior.

## Security Notes

- Default listener mode must remain localhost-only.
- Non-local listener bind requires `--allow-nonlocal-host` or `PYSCRIBE_ALLOW_NONLOCAL_HOST=1` plus authentication.
- Gradio share mode requires authentication even when binding to localhost.
- Do not reintroduce `--auth-pass`; listener passwords belong in `PYSCRIBE_AUTH_PASS`, `PYSCRIBE_LAN_AUTH_PASS`, or secure prompt input.
- Do not log raw tokens, passwords, API keys, local credential paths, or full sensitive request payloads.
- Store persistent LLM API keys as `env:VAR_NAME` references; direct key entry should remain session-only.
- Treat security scan reports as private unless Dave explicitly approves committing a sanitized summary.
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

Protected or local files should not be edited unless Dave explicitly asks:

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

The following files are local-only by default and should not be committed unless Dave explicitly asks:

- `AGENT.md`
- `IGNORE.md`
- `TODO.md`
- `.codex` / `.codex/`
- virtual environments, caches, logs, generated outputs, and scratch files
- private security reports
- secrets, credentials, and local configuration
