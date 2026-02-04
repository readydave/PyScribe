# Contributing to PyScribe

Thanks for contributing.

## Development Setup

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies.

Windows:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
python -m py_compile main.py app.py services/logging_service.py services/config_service.py services/model_service.py services/transcription_service.py ui_qt/main_window.py ui_qt/benchmark_dialog.py
python -m pytest -q tests/smoke_cli.py
```

## Code Guidelines

- Keep changes focused and small when possible.
- Add/maintain type hints on function signatures.
- Prefer clear names over abbreviations.
- Avoid checking in runtime artifacts (`.gradio/`, cache folders, logs).
- Do not commit secrets or local credentials.

## Pull Request Guidance

- Explain what changed and why.
- Mention any user-visible behavior changes.
- Include test/verification notes.
- Update docs (`README.md`, `docs/`) when behavior changes.

## Commit Messages

Use concise, descriptive commit messages in imperative mood.

Examples:

- `Fix listener host validation for non-local binding`
- `Add OCR backend fallback message in Qt UI`
- `Update user guide for visual-only mode`
