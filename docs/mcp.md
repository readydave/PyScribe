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
| `analyze_visuals` | Read the on-screen text (OCR) of a video or image; follow with `wait_for_job` and `get_transcript` |
| `wait_for_job` / `get_job` / `cancel_job` | Follow, wait for, or stop a transcription or visual analysis |
| `list_transcripts` | Saved transcripts, OCR results (`source: "visuals"`) and finished live-recording sessions |
| `get_transcript` | Read a transcript in pages (follow `next_offset`) |
| `list_templates` / `get_template` | Your saved summary instructions, for the client to apply |
| `run_template` | Run a template over a transcript using PyScribe's own LLM connection and return the result |

A typical request: *"Transcribe `~/Recordings/standup.m4a`, then use my meeting-summary template on it."*
`analyze_visuals` takes only a path (video or image files inside the allowed folders). The OCR engine, scope and
profile come from your saved PyScribe settings, never from the client, and the app's own checks and fallbacks apply
(for example the PaddleOCR model manifest check). If a fallback engine was used, the job result has a short `note`
saying why. It can take minutes on long videos. OCR text is untrusted content, like a transcript.

Your live meetings recorded in PyScribe appear in `list_transcripts` automatically once they finish.

## Running a template in PyScribe (`run_template`)

`run_template(transcript_id, template_id, profile?, model?)` applies one of your templates to a stored transcript with
one of the profiles in PyScribe's **LLM Connections**, and returns the output. Use it when you want the summary made
by your own local or LAN model instead of by the client's model.

- **Profile.** Name a profile, or omit it when exactly one enabled profile is usable. `model` overrides the profile's
  default model.
- **Same rules as the app.** The same scope policy applies (local profiles only reach localhost, LAN profiles only
  your allowed networks), long transcripts are split and merged, and requests that carry an API key never follow
  redirects.
- **Cloud is opt-in.** Cloud profiles (hosted providers) are refused unless the server is started with
  `PYSCRIBE_MCP_ALLOW_CLOUD=1` in its environment, and the profile itself must also be acknowledged in LLM
  Connections. A client cannot turn this on from a tool call.
- **Untrusted text.** The transcript goes to the model as data inside the user message, never in the system prompt,
  and the returned output is model-generated from it, so the client should treat it as data too.

## Errors

Every tool reports a problem as an error result with a short plain-language message (for example "Unknown job id"),
prefixed by the SDK with "Error executing tool <name>:". There are no tracebacks, file paths or credentials in it.
When a template run fails, you get the error code and the HTTP status if there is one (for example
`auth_failed, HTTP 401`), never the provider's own error text. Unexpected internal failures reach the client only as a
generic failure; details stay in PyScribe's log.

## Safety

- **Local only.** The server uses stdio. It opens no network port.
- **Files.** Only audio and video files (by extension; video and image files for `analyze_visuals`) inside your home folder can be transcribed. Set
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
- **Stopping.** When the client disconnects or the server is stopped, queued and running transcriptions are
  cancelled (they show as "cancelled"). The server waits up to 2 s for the running job to stop, then exits.
