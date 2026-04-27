# Contributing to PyScribe

Thanks for contributing.

## Development Setup

1. Clone the repository.
2. Create a project-local virtual environment.
3. Activate the virtual environment.
4. Install dependencies from `requirements.txt`.

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

Do not install project dependencies into the global or user Python environment.

## Run Locally

```bash
python main.py
python main.py qt
python main.py serve --port 7860
```

## Before Opening a PR

Run the same smoke checks CI runs:

```bash
python main.py --help
python main.py serve --help
python main.py qt --help
python -m py_compile main.py app.py services/logging_service.py services/config_service.py services/model_service.py services/transcription_service.py services/prompt_template_service.py services/llm_connection_service.py services/llm_postprocess_service.py ui_qt/main_window.py ui_qt/benchmark_dialog.py ui_qt/llm_connection_dialog.py ui_qt/llm_postprocess_dialog.py
bash -n scripts/run_listener.sh
python -m pip install pytest
python -m pytest -q tests/smoke_cli.py
```

Note: `bash -n scripts/run_listener.sh` is the same syntax check CI runs on Linux.
If you are on Windows without Bash, run this check in Git Bash/WSL or skip it locally.

## Security Checks

Run the relevant security checks before release prep, dependency changes, listener/network changes, authentication changes, file handling changes, model download changes, or LLM/OCR integration changes.

```bash
python -m bandit -r . -x .venv,venv,env,build,dist,__pycache__ -ll
python -m pip_audit -r requirements.txt
gitleaks detect --source . --no-git --redact
```

Notes:

- Do not run broad automated fix commands without reviewing the impact first.
- Do not commit security scan reports if they contain sensitive paths, exploit details, secrets, or private environment information.
- CUDA-specific PyTorch wheels may require separate vendor advisory review because dependency audit tools may not fully assess them.
- Re-run the relevant scan after fixing a security finding.

## Code Guidelines

- Keep changes focused and small when possible.
- Follow existing project conventions.
- Add or maintain type hints on function signatures.
- Prefer clear names over abbreviations.
- Add or maintain tests for changed behavior.
- Preserve secure defaults for listener, auth, token handling, TLS, model downloads, logging, and config persistence.
- Avoid checking in runtime artifacts such as `.gradio/`, cache folders, logs, local config, generated output, or temporary files.
- Do not commit secrets, tokens, passwords, local credentials, private endpoints, or local certificates.
- Do not commit local agent-control files unless Dave explicitly asks.

## Pull Request Guidance

- Explain what changed and why.
- Mention any user-visible behavior changes.
- Include test and verification notes.
- Include security scan notes when the change touches security-sensitive areas.
- Update docs such as `README.md`, `CHANGELOG.md`, `PROJECT.md`, `STACK.md`, or `docs/` when behavior, setup, commands, architecture, or security posture changes.
- Keep PRs focused and reviewable.

## Commit Messages

Use concise, descriptive commit messages in imperative mood.

Examples:

- `Fix listener host validation for non-local binding`
- `Add OCR backend fallback message in Qt UI`
- `Update user guide for visual-only mode`
