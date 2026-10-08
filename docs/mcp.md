# PyScribe MCP server

PyScribe can run as an [MCP](https://modelcontextprotocol.io) server, so tools like **Claude Code**, **Codex**,
and other MCP clients can transcribe audio on your computer and read your transcripts. The AI work (summaries,
action items, follow-ups) then happens inside that client, using your own subscription or settings, with no API key
in PyScribe.

```bash
python main.py mcp        # speaks MCP over stdio; normally started by the client, not by you
```

## Connecting a client

Use the full path to the Python in PyScribe's `.venv` and to `main.py`.

**Claude Code**

```bash
claude mcp add pyscribe -- /path/to/PyScribe/.venv/bin/python /path/to/PyScribe/main.py mcp
```

**Codex** (`~/.codex/config.toml`)

```toml
[mcp_servers.pyscribe]
command = "/path/to/PyScribe/.venv/bin/python"
args = ["/path/to/PyScribe/main.py", "mcp"]
```

On Windows use `.venv\Scripts\python.exe`. Client commands change between versions; if one of these is rejected,
check your client's MCP documentation for "add a stdio server".

## What the client can do

| Tool | Purpose |
|---|---|
| `list_transcription_models` | Models PyScribe has, and which are downloaded |
| `start_transcription` | Start transcribing a file (optional speaker labels, names/terms, language) |
| `wait_for_job` / `get_job` / `cancel_job` | Follow, wait for, or stop a transcription |
| `list_transcripts` | Saved transcripts and finished live-recording sessions |
| `get_transcript` | Read a transcript in pages (follow `next_offset`) |
| `list_templates` / `get_template` | Your saved summary instructions, for the client to apply |

A typical request: *"Transcribe `~/Recordings/standup.m4a`, then use my meeting-summary template on it."*
Your live meetings recorded in PyScribe appear in `list_transcripts` automatically once they finish.

## Safety

- **Local only.** The server uses stdio. It opens no network port.
- **Files.** Only audio and video files (by extension) inside your home folder can be transcribed. Set
  `PYSCRIBE_MCP_ROOTS` to a list of folders (separated by `:` on Linux/macOS, `;` on Windows) to narrow or change that.
  Paths are resolved first, so `..` and symlinks cannot escape the allowed folders.
- **Models.** Only models already downloaded in PyScribe are used, and only ones PyScribe lists. Nothing is
  downloaded behind your back; run PyScribe once to fetch a model (it asks first).
- **Transcripts are untrusted.** They contain whatever people said. The server tells the client to treat them as
  data, but a client with powerful tools should still be used with care on recordings you didn't make.
- **Where the text goes.** Reading a transcript through Claude Code or Codex sends that text to the provider behind
  that client. Use a local-model client if a recording must not leave your network.
- **Storage.** Transcripts made through MCP are saved in `~/.pyscribe/mcp_transcripts/` (owner-only permissions,
  newest 500 kept).
- **One job at a time.** Jobs queue (up to 10). If the PyScribe desktop app is transcribing at the same time, both
  compete for the GPU, so run one or the other.
