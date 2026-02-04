# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Added

- `CONTRIBUTING.md` with setup, testing, and PR workflow guidance.
- `SECURITY.md` with vulnerability reporting policy.
- `docs/user_guide.md` with complete user-facing feature documentation.

### Changed

- `README.md` refreshed to match current app behavior and docs layout.
- `docs/qt_help.md` updated for current Qt labels and flow.

## [2026-02-04]

### Changed

- Added explicit type contracts across the codebase (function signatures and class attributes).
- Updated repository ignore rules for local runtime/tool artifacts.

### Security

- Removed tracked local `.gradio` certificate artifact.
- Rewrote git history metadata to replace exposed author email with noreply address.
