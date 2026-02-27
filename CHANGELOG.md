# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Added

- Shared listener auth/bind validation service in `services/listener_security_service.py`.
- `tests/test_listener_and_diar_backends.py` covering listener security helpers and NeMo Sortformer compatibility paths.
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

### Changed

- `main.py` and `app.py` refactored to use shared listener security/auth helpers.
- Listener runtime initialization in `app.py` is now lazy to avoid unnecessary startup setup until listener code is needed.
- NeMo Sortformer handling now supports both modern callable diarizer APIs and legacy RTTM-output APIs.
- Sortformer availability checks now require both NeMo ASR importability and CUDA availability.
- Streaming transcript text updates are throttled in `services/transcription_service.py` to reduce UI churn while preserving final output.
- Qt transcript text area behavior in `ui_qt/main_window.py` updated for wrapping, sizing, and scroll behavior.
- `services/config_service.py` now includes additive LLM/template defaults for upcoming post-processing features.
- `services/__init__.py` now exports LLM prompt and connection services for both frontends.
- Qt Tools menu now includes **LLM Connections...** (`Ctrl+Shift+L`) for in-app connection configuration/testing.
- Qt Tools menu now includes **LLM Post-Process...** (`Ctrl+Shift+P`) and **Process Existing Transcript...** flows.
- LLM post-processing now blocks concurrent runs for local profiles while local transcription is active; LAN profiles must explicitly allow concurrency.
- Listener config saves now load-and-update existing config fields to preserve additive settings (including LLM profile data).

### Fixed

- Listener and Qt config-save failures now log warnings instead of failing silently.

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
