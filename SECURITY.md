# Security Policy

## Supported Versions

Security fixes are applied on the `main` branch.

## Reporting a Vulnerability

Please report vulnerabilities privately.

Preferred path:

1. Open a private GitHub Security Advisory for this repository.

If private reporting is unavailable, open a GitHub issue with minimal details and request a private follow-up channel.

## What to Include

- Affected component(s)
- Reproduction steps
- Impact assessment
- Suggested mitigation (if known)

## Sensitive Data Handling

Do not include secrets in reports, issues, or logs:

- HF tokens
- passwords
- private keys
- local credential files

- API keys for hosted AI providers
- transcripts and recordings (they may contain sensitive conversations)

Redact any sensitive values before sharing.

## Notes on Features That Send Data Elsewhere

- **Cloud LLM profiles** send transcripts (and any attached images) to the provider you configure. Each cloud profile requires an explicit confirmation, an `https://` address, and a key given as `env:VAR_NAME` (saved) or typed for one session (never saved). Redirects are refused on requests that carry a key. The Listener hides cloud profiles unless `llm_allow_cloud_in_listener` is set.
- **MCP server** (`python main.py mcp`) uses stdio only and opens no network port. It reads only audio/video files inside the folders in `PYSCRIBE_MCP_ROOTS` (default: your home folder). Text returned to an MCP client goes to whatever provider sits behind that client.

## Response Expectations

- Initial triage acknowledgment target: within 7 days
- Remediation timeline depends on severity and complexity

## Disclosure

Please allow time for a fix before public disclosure.
