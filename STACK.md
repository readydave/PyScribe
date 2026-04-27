# STACK.md

## Runtime

- Primary language: Python
- Supported versions: Python 3.10+
- Recommended version: Python 3.12
- Operating systems supported: Windows and Linux
- Primary UI frameworks: PySide6 for desktop, Gradio for listener web UI
- Core transcription engine: `faster-whisper` / CTranslate2
- Optional ML/OCR components: PyTorch CUDA wheels, pyannote.audio, PaddleOCR, RapidOCR, pytesseract/Tesseract, transformers/PEFT

## Package / Dependency Management

- Package manager: `pip`
- Project metadata: `pyproject.toml` with setuptools backend
- Runtime dependencies: `requirements.txt`
- Lock file: none currently committed
- Install command:

```bash
python -m pip install -r requirements.txt
```

- Editable/package install:

```bash
python -m pip install .
```

- Dependency update policy: keep changes deliberate and tested; GPU-related pins in `requirements.txt` are sensitive because they target CUDA 12.1 PyTorch wheels.

## Application Entry Points

```bash
# Interactive launcher menu
python main.py
```

```bash
# Qt desktop UI
python main.py qt
```

```bash
# Gradio listener, localhost by default
python main.py serve --port 7860
```

```bash
# Installed console script
pyscribe --help
```

```bash
# Listener helper script
scripts/run_listener.sh
```

## Development Commands

```bash
# Create and activate a Linux venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```bash
# Create and activate a Windows venv
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

```bash
# Run locally
python main.py
python main.py qt
python main.py serve --port 7860
```

```bash
# CLI help smoke checks
python main.py --help
python main.py serve --help
python main.py qt --help
```

```bash
# Compile smoke matching CI coverage
python -m py_compile main.py app.py services/logging_service.py services/config_service.py services/model_service.py services/transcription_service.py services/prompt_template_service.py services/llm_connection_service.py services/llm_postprocess_service.py ui_qt/main_window.py ui_qt/benchmark_dialog.py ui_qt/llm_connection_dialog.py ui_qt/llm_postprocess_dialog.py
```

```bash
# Listener shell-script syntax check
bash -n scripts/run_listener.sh
```

```bash
# Test smoke suite
python -m pytest -q tests/smoke_cli.py
```

```bash
# Full test suite
python -m pytest -q
```

```bash
# Build source/wheel distributions when build is installed
python -m build
```

No dedicated lint or type-check command is configured yet.

## Testing

- Test framework: pytest
- Test directory: `tests/`
- CI smoke command: `python -m pytest -q tests/smoke_cli.py`
- Full test command: `python -m pytest -q`
- Focused service tests are named by feature, for example `tests/test_live_transcription_service.py`, `tests/test_llm_connection_service.py`, and `tests/test_security_hardening.py`.
- Before commit, run at least the CI smoke checks from `.github/workflows/ci.yml`; run targeted tests for any changed service/UI workflow.
- UI-heavy Qt tests may require PySide6 and a suitable display/offscreen environment.
- GPU, OCR, and diarization behavior can be environment-dependent; prefer focused tests plus manual smoke when touching those paths.

## Architecture Notes

- Main modules: `main.py`, `app.py`, `services/`, `ui_qt/`, `models.py`, `diarization.py`, `diar_backends.py`, `utils.py`.
- Desktop UI: `ui_qt/main_window.py` plus dialogs for benchmarking, LLM connections, and LLM post-processing.
- Listener UI: `app.py` builds Gradio workflows and delegates backend work to `services`.
- Backend/services: shared service modules own transcription, model resolution/download, runtime detection, live audio, config, prompt templates, LLM profiles, LLM post-processing, listener security, logging, platform helpers, and OCR/multimodal processing.
- Persistence/config: user config defaults to `~/.pyscribe_config.json`; user prompt templates are stored under `~/.pyscribe/prompts`.
- Built-in assets: prompt templates live under `assets/prompts/`; benchmark media and icons live under `assets/`.
- Deployment example: `deploy/systemd/pyscribe-listener.service.example`.

## Important Constraints

- Do not weaken listener bind/auth checks or bring back plaintext password CLI flags.
- Do not commit tokens, credentials, local config, scan reports, model caches, virtual environments, logs, or generated outputs.
- Preserve lazy imports where they keep CLI help and smoke tests lightweight.
- Preserve pyannote subprocess isolation unless replacing it with an equally robust CUDA-safe design.
- Preserve config backward compatibility and additive load/save behavior.
- Preserve Qt cancellation, pause/resume, force-stop, and close-window state behavior when changing worker flows.
- Treat GPU/OCR dependency changes as higher risk than ordinary Python dependency changes.

## Known Environment Notes

- Required external tools:
  - FFmpeg available on `PATH`
- Optional external tools:
  - OS `tesseract` executable for pytesseract OCR
  - CUDA-capable NVIDIA runtime for GPU acceleration and Sortformer
  - Bash for `scripts/run_listener.sh` syntax checks on Windows via Git Bash/WSL
- Important environment variables:
  - `PYSCRIBE_CACHE_DIR`
  - `PYSCRIBE_LOG_DIR`, `PYSCRIBE_LOG_LEVEL`, `PYSCRIBE_LOG_STDOUT`
  - `PYSCRIBE_AUTH_USER`, `PYSCRIBE_AUTH_PASS`, `PYSCRIBE_ALLOW_NONLOCAL_HOST`
  - `PYSCRIBE_LAN_AUTH_USER`, `PYSCRIBE_LAN_AUTH_PASS`
  - `PYSCRIBE_HOST`, `PYSCRIBE_PORT`, `PYSCRIBE_MAX_PORT_TRIES`, `PYSCRIBE_QUEUE_SIZE`
  - `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`
  - `PADDLE_HOME`, `PADDLE_PDX_CACHE_HOME`, `PADDLE_PDX_MODEL_SOURCE`, `PADDLE_PDX_HUGGING_FACE_ENDPOINT`
  - `MODELSCOPE_CACHE`
- Local-only files/directories:
  - `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
  - `.pyscribe_logs/`, `.gradio/`, `.codex`, `.codex/`
  - local agent files, scratch files, private reports, and generated outputs

## Security Scanning

Install scan tools in the project venv or another isolated environment before running these commands.

### Python

```bash
# Static security analysis
python -m bandit -r . -x .venv,venv,env,build,dist,__pycache__ -ll
```

```bash
# Dependency audit from requirements file
python -m pip_audit -r requirements.txt
```

```bash
# Dependency audit from active environment
python -m pip_audit
```

### Secret Scanning

```bash
# Scan current working tree
gitleaks detect --source . --no-git --redact
```

```bash
# Scan repository history
gitleaks detect --source . --redact
```

### Optional Static Analysis

```bash
# Local Semgrep scan
semgrep scan --config auto
```

```bash
# Treat findings as failures where appropriate
semgrep scan --config auto --error
```

### Recommended Security Review Command Set

Use this baseline for security-focused changes:

```bash
python -m bandit -r . -x .venv,venv,env,build,dist,__pycache__ -ll
python -m pip_audit -r requirements.txt
gitleaks detect --source . --no-git --redact
python -m pytest -q tests/test_security_hardening.py tests/test_listener_and_diar_backends.py tests/test_llm_connection_service.py
```

### Security Scan Notes

- Prefer project-specific tests over generic scanner output alone.
- Do not run broad automated fix commands without approval.
- Record unresolved findings in local `TODO.md` or a private report unless Dave approves a public issue.
- Record notable completed security fixes in `CHANGELOG.md`.
- Do not commit scan reports if they contain sensitive paths, secrets, exploit details, or private environment information.

## Agent Notes

Start with `AGENT.md`, `PROJECT.md`, `STACK.md`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and `SECURITY.md` before non-trivial changes.

Use `.github/workflows/ci.yml` as the source of truth for minimum CI smoke checks. For feature work, add or run targeted tests around the changed service/UI behavior.
