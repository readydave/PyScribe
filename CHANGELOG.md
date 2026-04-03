# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Added

- Qt live transcription mode with microphone/loopback capture, rolling ASR updates, recoverable session folders, and final post-pass cleanup.
- `services/live_transcription_service.py` with live session coordination, Qt audio-device filtering, rolling transcript reconciliation, and session metadata handling.
- `tests/test_live_transcription_service.py` covering live session merge logic, retention behavior, and Granite/live gating.
- `tests/test_qt_live_mode.py` covering Qt live-mode visibility, loopback gating, stop handoff, and cancel/force-stop reset behavior.
- Shared listener auth/bind validation service in `services/listener_security_service.py`.
- `tests/test_listener_and_diar_backends.py` covering listener security helpers and NeMo Sortformer compatibility paths.
- `tests/test_diarization.py` covering diarization runtime safeguards (`soundfile` backend preference and CPU pipeline reload after CUDA failure).
- `tests/test_qt_main_window_worker.py` covering Qt worker force-stop escalation and missing-terminal-event recovery.
- Prompt template scaffold in `assets/prompts/` with YAML index and built-in templates for meeting summary workflows.
- Prompt template loading/validation service in `services/prompt_template_service.py`.
- `tests/test_prompt_templates_and_config.py` covering prompt template loading and additive config behavior.
- Implementation task tracker for LLM post-processing in `docs/llm_postprocess_plan.md`.
- LLM connection diagnostics service in `services/llm_connection_service.py` with profile normalization and staged connection tests.
- `tests/test_llm_connection_service.py` covering local/LAN policy checks, provider smoke tests, and failure mappings.
- Qt LLM Connections dialog in `ui_qt/llm_connection_dialog.py` for profile editing and staged connection test results.
- LLM post-processing execution service in `services/llm_postprocess_service.py` for Ollama and OpenAI-compatible endpoints.
- Qt LLM Post-Process dialog in `ui_qt/llm_postprocess_dialog.py` with template/profile/model selection and transcript file loading.
- `tests/test_llm_postprocess_service.py` covering request validation, provider success paths, and failure-code mapping.
- Listener LLM Post-Processing panel in `app.py` with connection test, template/model selection, and current-vs-upload transcript source controls.
- `tests/test_listener_llm_postprocess_helpers.py` covering listener LLM helper behaviors and concurrency policy checks.
- User prompt template CRUD support in `services/prompt_template_service.py` (create/update/delete + user default) with templates stored under `~/.pyscribe/prompts`.
- Qt LLM Post-Process dialog now includes user template management (new/edit/delete), optional pasted context, and explicit payload preview.
- Listener LLM Post-Processing panel now includes pasted context and payload preview before run.
- Qt and Listener LLM workflows now support optional image attachments for post-processing context.
- Image-aware payload preparation now checks model multimodal capability and supports OCR fallback via configured OCR backend.
- LLM connection diagnostics now include local subnet discovery + LAN scan helpers and explicit `lm_studio` provider profile support.
- Qt main window now uses a unified dashboard shell with left navigation (Transcription, LLM, Settings) and stacked workspaces.
- Transcription workspace now includes collapsible left navigation and hide/show right status rail controls.
- Transcription workspace now includes a terminal-style live pipeline event log panel.
- Qt LLM post-process dialog now uses a splitter layout with grouped configuration/attachments and structured context/preview/output panes.
- LLM post-process flow now includes explicit cancel confirmation with close-window cancellation handling.

### Changed

- `main.py` and `app.py` refactored to use shared listener security/auth helpers.
- Listener runtime initialization in `app.py` is now lazy to avoid unnecessary startup setup until listener code is needed.
- Pyannote diarization backends now run in an isolated spawned subprocess so CUDA speaker ID does not share cuDNN runtime state with `faster-whisper` ASR.
- Diarization now prefers `torchaudio`'s `soundfile` backend for audio loading because the default SoX path can crash on some systems.
- Diarization CPU retry detection now treats cuDNN mismatch and related GPU runtime failures as retryable for pyannote backends.
- NeMo Sortformer handling now supports both modern callable diarizer APIs and legacy RTTM-output APIs.
- Sortformer availability checks now require both NeMo ASR importability and CUDA availability.
- Streaming transcript text updates are throttled in `services/transcription_service.py` to reduce UI churn while preserving final output.
- Qt transcript text area behavior in `ui_qt/main_window.py` updated for wrapping, sizing, and scroll behavior.
- `services/config_service.py` now includes additive LLM/template defaults for upcoming post-processing features.
- `services/config_service.py` now also persists live capture defaults (source mode, selected device, output root, and keep-audio preference).
- `services/__init__.py` now exports LLM prompt and connection services for both frontends.
- Qt Tools menu now includes **LLM Connections...** (`Ctrl+Shift+L`) for in-app connection configuration/testing.
- Qt Tools menu now includes **LLM Post-Process...** (`Ctrl+Shift+P`) and **Process Existing Transcript...** flows.
- LLM post-processing now blocks concurrent runs for local profiles while local transcription is active; LAN profiles must explicitly allow concurrency.
- Listener config saves now load-and-update existing config fields to preserve additive settings (including LLM profile data).
- LLM post-processing request model now supports separate `extra_context_text` and explicit payload preview generation.
- LLM payload preparation now resolves image context before send and includes OCR-derived context when fallback is used.
- LAN scope policy diagnostics now explicitly flag loopback usage and recommend `local` scope for `127.0.0.1`/`localhost`.
- Qt speaker-backend capability detection is now lazy and runs asynchronously when speaker ID is enabled, instead of blocking main-window startup.
- Qt now shows temporary speaker-backend initialization state in both status text and window title while background probing runs.
- LLM Post-Process payload confirmation is now opt-in by default (no mandatory confirm prompt unless enabled).
- LLM Post-Process status messaging now clarifies that connection tests are optional and may differ from generation latency.
- LLM Post-Process now shows explicit in-button busy states (preview/run/refresh) during long operations so UI pause periods are communicated.
- Built-in `Meeting Summary` prompt template updated with stricter pre-summary checks, expanded section structure, prioritized actions, Q&A, participant sentiment tags, and standardized naming/style rules.
- Built-in `Action Items`, `Decision Log`, `Customer Call Summary`, and `Incident Summary` templates upgraded to richer execution-focused structures with stronger evidence, prioritization, and style constraints.
- LLM Connections dialog now includes an explicit `Rename Profile` action with duplicate-name prevention and default-profile remapping on rename.
- Interactive LAN listener auth no longer relies on a built-in default password; password now comes from `PYSCRIBE_LAN_AUTH_PASS` or secure prompt input.
- LLM Connections API key handling now supports persisted `env:VAR_NAME` references and treats direct key entry as session-only.
- Config persistence now strips plaintext LLM API keys from disk writes.
- LLM post-processing now enforces endpoint scope/CIDR policy at run time (not only during connection tests).
- LLM connection/post-process HTTP calls now honor profile `verify_tls` behavior for HTTPS endpoints.
- Qt theme/QSS styling refreshed to a card-based teal-accent design for both light and dark themes.
- Qt transcription settings cards now adapt between two-column and single-column layouts based on available width.
- Qt startup sizing now fits to available screen area to avoid oversized initial windows on smaller displays.
- Qt diarization backend selector now remains usable while lazy backend probing resolves capability details.

### Fixed

- Listener and Qt config-save failures now log warnings instead of failing silently.
- Fixed Qt live-session handoff so **Stop** can finalize the rolling draft and transition into the existing file-based post-pass workflow.
- Fixed Qt live mode leaving controls disabled after the final post-pass completed.
- Fixed Qt transcription runs that could remain stuck when a worker process exited without emitting a terminal event.
- Fixed Qt **Force Stop** so Linux worker cleanup escalates beyond `terminate()` when necessary.
- Fixed diarization crashes caused by `torchaudio`'s SoX backend during pyannote audio reads.
- Fixed CUDA speaker-ID failures after `faster-whisper` ASR by isolating pyannote diarization into a fresh subprocess.
- Prevented startup stalls from eager NeMo/Sortformer availability checks during initial Qt window construction.
- LLM post-processing now retries once with an extended timeout when the first request times out (helps cold model starts).
- Public listener share mode now requires authentication credentials, including localhost share scenarios.
- Fixed Qt startup/import error for missing `QSplitter` symbol in the refactored transcription layout.
- Fixed Qt font warning spam (`QFont::setPointSize <= 0`) triggered during dynamic resize in the new GUI layout.

## [2026-02-04]

### Added

- `CONTRIBUTING.md` with setup, testing, and PR workflow guidance.
- `SECURITY.md` with vulnerability reporting policy.
- `docs/user_guide.md` with complete user-facing feature documentation.
- Application icon assets in `assets/icons/pyscribe.ico` and `assets/icons/pyscribe.png`.

### Changed

- `README.md` refreshed to match current app behavior and docs layout.
- `docs/qt_help.md` updated for current Qt labels and flow.
- Added explicit type contracts across the codebase (function signatures and class attributes).
- Improved diarization resilience and Linux CUDA loader environment setup.
- Updated repository ignore rules for local runtime/tool artifacts.
- Added ignore coverage for local launcher script artifacts.

### Security

- Removed tracked local `.gradio` certificate artifact.
- Rewrote git history metadata to replace exposed author email with noreply address.
