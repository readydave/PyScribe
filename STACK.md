# STACK.md

## Runtime

- Primary language: Python
- Supported versions: Python 3.10+
- Recommended version: Python 3.12
- Operating systems supported: Windows and Linux
- Primary application type: Local transcription app with desktop, listener, and CLI-style launch modes

## Package / Dependency Management

- Package manager: `pip`
- Dependency file: `requirements.txt`
- Recommended environment: project-local `.venv`
- Install command:

```bash
pip install -r requirements.txt
```

- Do not install project dependencies into the global or user Python environment.
- Do not change dependency management tools unless Dave explicitly asks.

## Python Environment

Use the project-local virtual environment when present.

Windows:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `.venv` does not exist, ask Dave before creating one.

## External Runtime Requirements

- FFmpeg must be available in `PATH`.
- **NVIDIA GPU with CUDA 12+** is highly recommended for diarization and Whisper acceleration.
- **Linux Dynamic Loader**: PyScribe automatically manages `LD_LIBRARY_PATH` to resolve CUDA dependencies (like `libtorch_cuda_linalg.so`) in bleeding-edge Torch environments.

Windows:

```bash
winget install Gyan.FFmpeg
```

Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

Optional OCR runtime:

- `pytesseract` plus the OS `tesseract` executable
- or PaddleOCR runtime dependencies

Optional model/auth integrations:

- Hugging Face token for gated/private model access
- Local or LAN LLM profile for post-processing
- Ollama or OpenAI-compatible endpoint, depending on configured profile

## Application Entry Points

```bash
# Interactive launcher menu
python main.py
```

```bash
# Qt desktop mode
python main.py qt
```

```bash
# Listener / Gradio mode
python main.py serve --port 7860
```

```bash
# CLI/package entry point, if installed
pyscribe --help
```

## Development Commands

```bash
# Help commands
python main.py --help
python main.py serve --help
python main.py qt --help
```

```bash
# Run locally
python main.py
python main.py qt
python main.py serve --port 7860
```

```bash
# Smoke test
python -m pytest -q tests/smoke_cli.py
```

```bash
# Example targeted unittest
python -m unittest tests.test_listener_and_diar_backends
```

```bash
# Python compile smoke check
python -m py_compile main.py app.py services/logging_service.py services/config_service.py services/model_service.py services/transcription_service.py services/prompt_template_service.py services/llm_connection_service.py services/llm_postprocess_service.py ui_qt/main_window.py ui_qt/benchmark_dialog.py ui_qt/llm_connection_dialog.py ui_qt/llm_postprocess_dialog.py
```

```bash
# Listener shell script syntax check
bash -n scripts/run_listener.sh
```

```bash
# Package install check
pip install .
pyscribe --help
```

## Testing

- Test framework: `pytest` and `unittest`
- Test directory: `tests/`
- Smoke test command:

```bash
python -m pytest -q tests/smoke_cli.py
```

- Before-PR validation command set:

```bash
python main.py --help
python main.py serve --help
python main.py qt --help
python -m py_compile main.py app.py services/logging_service.py services/config_service.py services/model_service.py services/transcription_service.py services/prompt_template_service.py services/llm_connection_service.py services/llm_postprocess_service.py ui_qt/main_window.py ui_qt/benchmark_dialog.py ui_qt/llm_connection_dialog.py ui_qt/llm_postprocess_dialog.py
bash -n scripts/run_listener.sh
python -m pip install pytest
python -m pytest -q tests/smoke_cli.py
```

## Security Scanning

Run relevant security checks before release prep, dependency changes, listener/network changes, authentication changes, file handling changes, model download changes, or LLM/OCR integration changes.

```bash
# Static security analysis
python -m bandit -r . -x .venv,venv,env,build,dist,__pycache__ -ll
```

```bash
# Dependency audit from requirements file
python -m pip_audit -r requirements.txt
```

```bash
# Optional dependency audit from active environment
python -m pip_audit
```

```bash
# Optional JSON report
python -m pip_audit -r requirements.txt --format json
```

```bash
# Secret scan of current working tree
gitleaks detect --source . --no-git --redact
```

```bash
# Secret scan including git history
gitleaks detect --source . --redact
```

### Security Scan Notes

- Prefer project-specific commands over generic examples.
- Do not run broad automated fix commands without approval.
- Do not commit scan reports if they contain sensitive paths, exploit details, secrets, private environment information, or local endpoint details.
- Record unresolved findings in local `TODO.md` unless Dave asks to track them elsewhere.
- Record notable completed security fixes in `CHANGELOG.md`.
- CUDA-specific PyTorch wheels may require separate vendor advisory review because dependency audit tools may not fully assess them.
- Re-run relevant scans after fixing security findings.

## Architecture Notes

### Main Entry Points

- `main.py` - launcher and CLI/listener entry behavior
- `app.py` - listener/Gradio UI and listener runtime behavior

### Core Services

- `services/config_service.py` - configuration persistence and additive settings
- `services/logging_service.py` - session-based timestamped logging with automatic rotation (keeps latest 21 logs)
- `services/model_service.py` - model-related behavior
- `services/transcription_service.py` - transcription pipeline and transcript updates
- `services/live_transcription_service.py` - live transcription session coordination
- `services/listener_security_service.py` - listener bind/auth validation
- `services/prompt_template_service.py` - prompt template loading and user template CRUD
- `services/llm_connection_service.py` - LLM profile diagnostics and connection policy
- `services/llm_postprocess_service.py` - LLM post-processing execution

### UI

- `ui_qt/main_window.py` - Qt main window and transcription workspace
- `ui_qt/benchmark_dialog.py` - benchmark dialog
- `ui_qt/llm_connection_dialog.py` - LLM profile management and diagnostics
- `ui_qt/llm_postprocess_dialog.py` - LLM post-processing workflow

### Supporting Areas

- `assets/prompts/` - built-in prompt templates
- `assets/icons/` - application icons
- `assets/images/` - screenshots and README images
- `docs/user_guide.md` - full user-facing feature guide
- `docs/qt_help.md` - in-app Qt help content
- `scripts/run_listener.sh` - listener helper script
- `deploy/systemd/` - example systemd deployment unit

## Important Constraints

### Do Not

- Do not install dependencies globally.
- Do not bypass project-local virtual environment usage.
- Do not commit secrets, tokens, passwords, local certificates, private endpoints, or local config.
- Do not weaken listener bind/auth validation.
- Do not reintroduce CLI password flags that expose secrets in shell history or process lists.
- Do not disable TLS verification for non-localhost HTTPS profiles.
- Do not persist plaintext LLM API keys to disk.
- Do not make broad dependency upgrades without reviewing security and compatibility impact.
- Do not run automated force-fix dependency commands without approval.

### Preserve

- Local-first transcription behavior.
- Secure listener defaults.
- Authentication requirement for non-local listener binds and public share mode.
- Session-only token preference where applicable.
- Environment-variable references for persistent secrets.
- Spawned subprocess isolation for pyannote diarization.
- Final transcription completion even when optional diarization fails.
- Existing Qt and listener workflows unless Dave explicitly requests behavior changes.

### Security-Sensitive Areas

- Listener host binding and public share behavior
- Listener authentication
- Token handling and Hugging Face authentication
- LLM profile persistence
- TLS verification behavior
- Model downloads and cache validation
- File logging paths
- Prompt template file path handling
- OCR/image attachment handling
- Shell execution or helper scripts
- Config persistence and sanitization

### Performance-Sensitive Areas

- Faster Whisper ASR pipeline
- Live transcription rolling updates
- Qt UI responsiveness
- Diarization subprocess lifecycle
- CUDA/GPU resource handling
- Model download/cache behavior
- LLM post-processing timeouts and cancellation

## Known Environment Notes

- Listener mode defaults to local-only behavior unless configured otherwise.
- Interactive LAN listener mode requires authentication.
- Public share mode requires authentication.
- Some diarization backends require CUDA.
- Granite remains file-only.
- Qt live transcription is Linux-first.
- Windows without Bash may need Git Bash or WSL for `bash -n scripts/run_listener.sh`.

## Documentation Update Expectations

Update these files when behavior, setup, commands, architecture, or user-facing functionality changes:

- `README.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `PROJECT.md`
- `STACK.md`
- `docs/user_guide.md`
- `docs/qt_help.md`

Do not update protected/local files unless Dave explicitly asks:

- `AGENT.md`
- `AGENTS.md`
- `IGNORE.md`
- local `TODO.md`
- private security reports

## Agent Notes

Agents should use this file to determine the correct commands before running, installing, testing, linting, building, packaging, or scanning.

If this file is incomplete, inspect the repository before making assumptions.