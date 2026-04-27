# PROJECT.md

## Project Summary

Describe what this project does, who it is for, and what problem it solves.

## Goals

-
-
-

## Non-Goals

-
-
-

## Architecture Overview

Describe the major parts of the system and how they relate.

Suggested areas to document:

- Application entry points
- Core modules
- UI layer, if applicable
- Service/business logic layer
- Data/configuration layer
- External integrations
- Background jobs, workers, or automation flows

## Core Workflows

Document the main workflows the project supports.

Examples:

- User workflow:
- Developer workflow:
- Data processing workflow:
- Deployment workflow:

## Important Design Decisions

Track important architectural or implementation choices.

-
-
-

## Decision Log

Use this section to preserve important project decisions so future agents do not re-litigate them.

| Date | Decision | Reason | Impact |
|---|---|---|---|
| YYYY-MM-DD |  |  |  |

## Current Priorities

Use this section to help coding agents understand what matters most right now.

-
-
-

## Roadmap

Use this section for project-level future direction that should be committed with the repo.

For private or short-term working items, use local `TODO.md` instead.

### Near-Term

-
-

### Later

-
-

## Known Risks / Fragile Areas

Document areas where agents should be especially careful.

Examples:

- Security-sensitive logic
- Authentication or authorization flows
- Input validation
- Secret handling
- File upload, file path, or shell command logic
- Data migration code
- External API integrations
- Performance-sensitive paths
- Complex legacy modules
- UI flows that are easy to regress

## Security Notes

Document project-specific security expectations, sensitive areas, and scan requirements.

Examples:

- Security-sensitive modules:
- Required security scan commands:
- Dependency audit expectations:
- Secret handling rules:
- Network exposure rules:
- Logging restrictions:

Detailed vulnerability reporting policy belongs in local/protected `SECURITY.md` unless Dave explicitly chooses to commit it.

## Documentation Rules

When functionality changes, consider whether these files need updates:

- `README.md`
- `CHANGELOG.md`
- `TODO.md`
- `CONTRIBUTING.md`
- `STACK.md`

Protected/local files should not be committed unless Dave explicitly asks:

- `AGENT.md`
- `AGENTS.md`
- `IGNORE.md`
- `SECURITY.md`
- tool-specific agent files
- private security reports
- local notes

## Commit Policy

The following files are intended to be committed by default:

- `PROJECT.md`
- `STACK.md`
- `CHANGELOG.md`
- `README.md`
- `CONTRIBUTING.md`

The following files are local-only by default and should not be committed unless Dave explicitly asks:

- `AGENT.md`
- `AGENTS.md`
- `IGNORE.md`
- `TODO.md`
- `SECURITY.md`
- tool-specific agent instruction files
- private security reports
- scratch files
- local notes
