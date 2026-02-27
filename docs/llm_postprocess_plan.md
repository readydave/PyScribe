# LLM Post-Processing Implementation Plan

This plan tracks implementation of LLM post-processing for PyScribe in small,
reversible milestones.

## Rollback Strategy

- Ship additive changes first (no breaking config migrations).
- Keep transcription pipeline unchanged while scaffolding LLM features.
- Commit each milestone independently so it can be reverted cleanly.

## Milestone 1: Foundation (No UI Behavior Change)

- [x] Create feature branch for isolated development.
- [x] Add prompt template storage scaffold (`assets/prompts/`).
- [x] Add prompt template loading/validation service.
- [x] Extend app config schema with additive LLM/template fields.
- [x] Document foundation changes in `CHANGELOG.md`.

## Milestone 2: Connection Profiles + Testing

- [x] Add local/remote LLM profile data model and persistence.
- [x] Add endpoint test pipeline (reachability/auth/models/smoke test).
- [x] Add pass/fail diagnostics and suggested fixes structure.
- [x] Add logging events for test and run metadata (no content logging).

## Milestone 3: Qt Workflow Integration

- [x] Add LLM post-process controls to Qt.
- [x] Enforce sequencing policy for local GPU reuse.
- [x] Add profile/model selection and connection test in Qt.
- [x] Add "Process Existing Transcript" flow in Qt.

## Milestone 4: Listener Workflow Integration

- [ ] Add LLM post-process controls to Listener UI.
- [ ] Add profile/model selection and connection test in Listener UI.
- [ ] Add "Process Existing Transcript" flow in Listener UI.

## Milestone 5: Templates + Context Enrichment

- [ ] Add built-in prompt templates and user template CRUD.
- [ ] Add optional pasted text context.
- [ ] Add optional screenshot/image attachments.
- [ ] Add multimodal capability check + OCR fallback for image context.
- [ ] Add payload preview ("what will be sent").

## Milestone 6: Docs + Hardening

- [ ] Update `README.md` for LLM setup and workflow.
- [ ] Update `docs/user_guide.md` with post-processing flows.
- [ ] Update `docs/qt_help.md` for new controls.
- [ ] Add/extend tests for config, template loading, and connection diagnostics.
