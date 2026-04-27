# STACK.md

## Runtime

- Primary language:
- Supported versions:
- Recommended version:
- Operating systems supported:

## Package / Dependency Management

- Package manager:
- Lock file:
- Install command:
- Dependency update policy:

## Application Entry Points

```bash
# Main app
```

```bash
# Alternative modes
```

## Development Commands

```bash
# Run locally
```

```bash
# Run tests
```

```bash
# Run lint
```

```bash
# Run type checks
```

```bash
# Build/package
```

## Testing

- Test framework:
- Test directory:
- Smoke test command:
- Full test command:
- Tests that should run before commit:

## Architecture Notes

- Main modules:
- UI framework:
- Backend/services:
- Persistence/config:
- External integrations:

## Important Constraints

- Do not:
- Preserve:
- Security-sensitive areas:
- Performance-sensitive areas:

## Known Environment Notes

- Required external tools:
- Optional external tools:
- Environment variables:
- Local-only files/directories:

## Security Scanning

Define the security commands for this project.

### Python

```bash
# Static security analysis
python -m bandit -r . -x .venv,venv,env,build,dist,__pycache__ -ll

# Dependency audit from requirements file
python -m pip_audit -r requirements.txt

# Dependency audit from active environment
python -m pip_audit

# Optional JSON report
python -m pip_audit -r requirements.txt --format json
```

### JavaScript / TypeScript

```bash
# npm projects
npm audit --audit-level=moderate

# pnpm projects
pnpm audit --audit-level moderate
```

### Multi-Language Static Analysis

```bash
# Local Semgrep scan
semgrep scan --config auto

# Treat findings as failures where appropriate
semgrep scan --config auto --error
```

### Secret Scanning

```bash
# Scan current working tree
gitleaks detect --source . --no-git --redact

# Scan repository history
gitleaks detect --source . --redact
```

### Recommended Security Review Command Set

Use the commands that match the project stack:

```bash
# Python baseline
python -m bandit -r . -x .venv,venv,env,build,dist,__pycache__ -ll
python -m pip_audit -r requirements.txt
gitleaks detect --source . --no-git --redact
```

### Security Scan Notes

- Prefer project-specific commands over generic examples.
- Do not run broad automated fix commands without approval.
- Record unresolved findings in `TODO.md`.
- Record notable completed security fixes in `CHANGELOG.md`.
- Do not commit scan reports if they contain sensitive paths, secrets, exploit details, or private environment information.

## Agent Notes

Agents should use this file to determine the correct commands before running, installing, testing, linting, building, or packaging.

If this file is incomplete, inspect the repository before making assumptions.
