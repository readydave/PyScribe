# PyScribe Security Review

Review date: 2026-03-10

## Executive Summary

PyScribe's highest-risk listener exposure path is in materially better shape than many local AI tools: non-local Gradio binds require auth, public share mode requires auth, the deprecated `--auth-pass` CLI flag is blocked, Hugging Face downloads are revision-pinned and checksum-verified, and the new branch state closes the main local hardening gaps identified in the first review pass.

I did not find any critical, high, or medium-severity direct-code issues in the current branch after the hardening changes. The main remaining items are operational: an opt-in path still exists to persist Hugging Face tokens into the shared HF cache, and the dependency audit cannot fully assess the CUDA-specific PyTorch wheel variants.

## Scope And Method

- Repo review of the Python desktop/listener application, Qt UI, config, model download, prompt-template, OCR/LLM, logging, and listener security code.
- Manual code review with file-and-line evidence.
- Live dependency audit run in `C:\Code\_envs\pyscribe` using:
  - `python -m pip_audit -r requirements.txt --format json`

## Critical

No critical findings.

## High

No high-severity findings.

## Medium

No medium-severity findings.

## Low

### PS-SEC-001: Hugging Face token persistence is now optional, but disk persistence still uses the shared HF cache

- Evidence:
  - `services/hf_auth_service.py:12-20` loads the token from in-memory session state, environment variables, or the Hugging Face cache.
  - `services/hf_auth_service.py:25-33` stores tokens in memory by default and only persists to `HfFolder.save_token(...)` when requested.
  - `ui_qt/main_window.py:2048-2075` now presents a storage choice and recommends session-only storage.
- Risk detail:
  - This is now an operator-choice issue rather than a default behavior issue.
  - Session-only storage materially reduces exposure. If the operator explicitly chooses `Save to Disk`, the token still lands in the shared Hugging Face auth cache.
- Recommended action:
  - Prefer session-only storage unless persistence is required.
  - For stronger persistence hygiene, move the opt-in disk path to Windows Credential Manager or another OS-backed secret store.

### PS-SEC-002: Dependency audit coverage is incomplete for CUDA-specific PyTorch wheels

- Evidence:
  - `requirements.txt:7-12` pins `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, and `torchaudio==2.5.1+cu121`.
  - The 2026-03-10 `pip-audit` run reported no known vulnerabilities for auditable packages, but skipped those CUDA-suffixed wheels because they are not directly auditable on PyPI.
- Risk detail:
  - This is a supply-chain visibility gap, not a confirmed vulnerability.
  - It means your automated audit does not fully cover some of the heaviest native dependencies in the environment.
- Recommended action:
  - Track PyTorch and NVIDIA security advisories separately from PyPI audits.
  - Keep the CUDA wheel versions pinned and reviewed during dependency refreshes.
  - Generate and retain an SBOM for packaged releases so skipped native dependencies are still documented.

## Positive Controls Observed

- Listener auth and exposure controls are strong for the current scope:
  - `services/listener_security_service.py:38-65` requires auth for non-local binds and public share mode.
  - `services/listener_security_service.py:67-73` rejects the insecure legacy `--auth-pass` CLI flag.
- LLM profile persistence is better than typical local tools:
  - `services/config_service.py:73-86` sanitizes config before write.
  - `services/config_service.py:188-200` strips plaintext runtime API keys from persisted LLM profiles unless the value is an `env:` reference.
- Prompt templates use safe YAML parsing:
  - `services/prompt_template_service.py:526-537` uses `yaml.safe_load(...)`.
- User prompt-template paths are now constrained to the template root:
  - `services/prompt_template_service.py:214-219` resolves update paths under the user prompt root.
  - `services/prompt_template_service.py:284-291` refuses to delete files outside the user prompt root.
  - `services/prompt_template_service.py:351-357` skips index entries whose file paths escape the template root.
- HTTPS certificate-verification bypass is now limited to localhost/loopback development endpoints:
  - `services/llm_connection_service.py:255-265` enforces TLS policy during connection tests.
  - `services/llm_connection_service.py:657-669` enforces the same policy at runtime before LLM calls.
  - `ui_qt/llm_connection_dialog.py:82-85` documents the restriction in the UI.
  - `ui_qt/llm_connection_dialog.py:423-432` blocks saving an HTTPS LAN profile with TLS verification disabled.
- Hugging Face model integrity is now materially improved in the current branch:
  - `services/model_download_service.py:163-266` pins the downloaded snapshot to a specific revision and re-verifies cached snapshots before reuse.
  - `services/model_download_service.py:306-395` fetches Hub file metadata and verifies every LFS-backed file by SHA-256 before returning the model path.
- HF token handling is tighter than the original branch state:
  - `services/hf_auth_service.py:25-33` no longer mirrors tokens into `HF_TOKEN`.
  - `diarization.py:68` now uses the shared auth service, so session-only tokens still work with gated diarization.
- File logging now prefers the user-private log directory and no longer falls back to shared temp locations:
  - `services/logging_service.py:62-77` prefers only `PYSCRIBE_LOG_DIR` or `Path.home() / ".pyscribe" / "logs"`.
  - `services/logging_service.py:80-105` no longer rotates logs in temp or `.pyscribe_logs` fallback directories.

## Prioritized Action List

### Immediate

1. Keep session-only HF token storage as the default recommendation.
2. Preserve the enforced TLS policy that blocks HTTPS verification bypass outside localhost/loopback development use.

### Short-Term

1. Move the optional persistent HF token path to Windows Credential Manager or another OS-backed secret store.
2. Consider a `stdout-only` logging mode for locked-down environments that do not want local file logs at all.

### Defense-In-Depth

1. Keep monthly dependency audits, but add vendor advisory monitoring for CUDA/PyTorch wheels that `pip-audit` cannot assess.
2. Preserve the new Hugging Face checksum verification path and extend the same pattern to other remote model/artifact downloads if PyScribe adds them later.
