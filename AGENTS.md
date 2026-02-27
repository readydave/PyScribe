# Agent Instructions for PyScribe

## Python Environment Policy

- Always check for a project-local virtual environment (for example `.venv`) before running Python or pip commands.
- For this repository, prefer the project venv interpreter by default.
- Do not install packages with the global/user Python interpreter for this project.
- If a matching venv does not exist, ask the user for approval to create one before installing dependencies.
