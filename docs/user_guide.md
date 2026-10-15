# PyScribe User Guide

This guide explains every user-facing feature in PyScribe.

## 1) Launch Modes

PyScribe has two frontends:

- **Qt Desktop** (`python main.py qt`)
- **Gradio Listener** (`python main.py serve`)

If you run `python main.py` with no mode, you get an interactive launcher menu. If you do not choose within 5 seconds, PyScribe starts Desktop (Qt) automatically.

## 2) Common Concepts

### Media Input

- Audio and video files are supported.
- Video files can be transcribed from audio and optionally analyzed visually (OCR).

### Models

- You can use built-in model names or custom Hugging Face repos (`owner/repo`).
- Full Hugging Face URLs are accepted and normalized to repo IDs.

### Processing Stages

- **Transcription**: speech-to-text.
- **Diarization** (optional): speaker labeling (for example `[S1]`, `[S2]`).
- **Visual analysis** (optional): OCR over sampled video frames.

## 3) Qt Desktop Features

### Layout + Navigation

- The Qt window has a left navigation rail and a content area. The rail has five destinations:
  - **Projects**: start or resume work (Projects Home). Opening a project shows the Transcript Studio under this entry.
  - **Transcribe**: choose a recording, pick a model and process it.
  - **Live**: record from a microphone or system audio and read the transcript as it arrives.
  - **Templates**: browse and manage the prompt templates used for summaries and action items.
  - **Settings**: appearance, privacy, AI connections, start-up defaults and diagnostics.
- The app opens on **Projects**. The rail can be collapsed with the small button in its header; the choice is remembered. **New Project** (bottom of the rail) clears the current file, transcript and progress and returns to Transcribe (it asks first if there is a transcript, and is blocked while a job runs). Settings and the batch queue are kept.
- While a job is running, a slim **status bar** appears above the content on every page: the current stage in words (Loading model, Transcribing, Identifying speakers, Analyzing visuals, Saving, or Recording), elapsed time, a progress bar, **Details** and **Cancel**. It disappears when the work ends.
- **Details** opens the **Job details** drawer on the right: one row per stage with state and elapsed time, the pipeline event log, and **Force stop**. **Show performance** (or **View > Performance**, `Ctrl+Shift+H`) opens the **Performance** drawer with 60-second CPU, memory, GPU and VRAM traces coloured by the running stage. **View > Job details** is `Ctrl+Shift+J`. Only one drawer is open at a time; `Esc` closes the one with focus.
- The status line (for example "Selected: meeting.wav" or "Transcription complete.") sits under the file path on the Transcribe page, and messages from the Studio also appear in the status bar at the bottom of the window.
- Speaker labels in the transcript are colour-coded per speaker. While recording live, the timer and Stop button turn red.
- The app opens sized to the available screen area and remains resizable. It is designed for 1366×768 and larger; below about 900 px wide the setup column on Transcribe scrolls sideways instead of clipping its controls (collapse the rail to give it room back).
- Keyboard: every control can be reached with Tab; focus is drawn as a visible outline and selected items (rail entry, inspector tab, project card) have their own highlight. **Reduce motion** in Settings turns off sliding panels and the busy-bar animation.

### Projects Home

- A greeting, then one large **New transcription** card (drop a media file on it, or click it to choose one), and two smaller cards: **Live capture** and **Import folder** (queues every media file in a folder).
- **Recent projects** shows your latest six as cards: title, duration, speaker count, a waveform cue, status (Completed, Transcribing with a percentage, Failed, Cancelled) and last activity. **View all** shows every project. The **...** menu on a card offers Open, Rename, Show in folder and Delete (the original recording is never touched).
- With no projects yet, the page explains what you can import and offers the primary action.
- A small line at the bottom says all processing is local by default.
- This page shows no model, compute, OCR or hardware settings; those live on Transcribe, Live and Settings.

### Projects (saved automatically)

- Every finished file, batch or live job is saved as a project in `~/.pyscribe/projects` (change the folder under **Settings > Privacy & storage**; new projects go to the new folder and older ones stay where they were). A project holds the transcript, its timed speaker segments, any visual (OCR) text, speaker names you set, and AI outputs you generated. Files are private to your account (mode 0600).
- A job that fails or is cancelled with nothing to reopen is not kept. A cancelled job with partial text is kept and marked Cancelled.
- `PYSCRIBE_PROJECTS_DIR` overrides the folder (the setting is then read-only).

### Transcript Studio

- Opens when you open a project, and automatically when a single-file or live job finishes. **Projects** (top left) goes back.
- **Transcript**: one block per segment with a timestamp, a speaker badge and name, and the text at 16 px by default (change it under Settings; 12-28 px). The line being played is highlighted. Click a line, or select it and press `Enter`, to jump playback there; `Space` plays or pauses. Double-click a line (or press `F2`) to edit it; `Enter` saves, `Esc` cancels. Edits are saved shortly after, when you leave the Studio, and when you close the app.
- **Search** (`Ctrl+F`) highlights matches in the transcript and shows the count ("2 / 5"); `Enter` and the arrow buttons step through them; `Esc` clears the search.
- **Playback bar** (bottom): play/pause, back and forward 10 seconds, current time, a waveform you can click or drag to seek (Left/Right arrows move 5 seconds when it has focus), and speed (0.75x to 2x). Text, waveform and keyboard stay in sync. If the recording is gone (for example a live session saved without audio), the transcript can still be read and exported and a note explains why playback is off.
- **Inspector** (right; collapse it with the » button for more transcript width; one tab open at a time):
  - **Summary** and **Action items**: show what you generated for this project. **Generate...** opens the same LLM dialog as Tools > LLM Post-Process with the right template selected, so connection scope rules, the cloud confirmation and payload preview work exactly as before. Nothing is sent anywhere until you run it. The result is kept with the project.
  - **Speakers**: give each speaker a name (it appears in the transcript, exports and AI prompts), with turns and speaking time.
  - **Visuals**: the OCR text from slides or chat, when visual analysis was on.
- **Export** (top right): plain text, Markdown, SubRip (`.srt`), WebVTT (`.vtt`) or JSON. Exports use your speaker names and never include the path of the recording.
- The **Local only** badge means transcription ran on this device; AI actions only send text where your connection profile points.
- Long recordings are fine: a three-hour transcript (about 3,000 segments) opens in well under two seconds.

### File Selection

- **Browse Files** (inside drop zone): open file picker for media.
- **Clickable Drop Zone**: clicking anywhere within the dashed drop area will also open the file browser.
- **Drag-and-drop zone**: drop a media file directly, on Transcribe or on the New transcription card of Projects Home.
- **Open Folder**: open selected file directory (or last-used folder if no file selected).

### Batch Queue

- Add individual media files or import a folder to queue multiple files for serialized processing.
- Files with the same basename can be queued when they come from different folders.
- Exact duplicate paths are skipped.
- When duplicate basenames are present, the queue display includes parent-folder context so items remain distinguishable.

### Input Modes

- **Transcribe** and **Live** in the navigation rail choose the mode:
  - `File` (Transcribe): file-based transcription workflow.
  - `Live` (Live): Qt-only live capture workflow (Linux-first).
- The Live page shows a state chip (Ready, Recording, Paused, Finalizing), the source, device and optional session title first, and keeps the output folder, compute device, precision and "keep recorded audio" behind **More live options**.
- Live mode hides the drop zone and disables visual OCR controls.
- Live mode supports one source per session:
  - **Microphone**
  - **Loopback**
- Loopback requires the OS to expose a monitor/loopback input device. On Linux this is typically a PipeWire/PulseAudio monitor source.
- Each live session writes into `~/PyScribe Live Sessions` by default unless you choose another output folder.
- Live mode accepts timestamp-capable Whisper models and `nvidia/nemotron-speech-streaming-en-0.6b` (English only, streams text as you speak; the final pass re-runs Nemotron on the capture and adds speaker labels). Granite Speech is blocked in live mode.

### Model Selection

- **Model** dropdown is editable.
- Recommended model is shown based on detected hardware.

### Processing Toggles

- **Transcribe audio**:
  - On: speech transcription runs.
  - Off: transcription is skipped.
- **Speaker Identification is On/Off**:
  - On: diarization runs after transcription.
  - Off: diarization is skipped.
- **Analyze visuals (slides/chat OCR, beta)**:
  - On: OCR stage runs on video frames.
  - Off: OCR stage is skipped.

Qt automatically derives run mode:

- transcription + visuals -> `full`
- transcription only -> `transcribe_only`
- visuals only -> `visual_only`

If both transcription and visuals are off, PyScribe blocks run start.

In live mode:

- transcription is always on
- visual analysis is always off
- speaker identification can still be enabled, but it only runs after **Stop** during the final post-pass on the saved capture file

### Live Capture Controls

- **Source**: choose microphone or loopback capture.
- **Device**: pick the matching Qt audio input for the selected source type.
- **Output Folder**: root folder for live session subfolders.
- **Session Title**: optional title used for live session file naming; it is editable before capture starts and after completion for rename/apply-title workflows, but locked while capture/finalization is active.
- **Keep recorded audio after completion**:
  - On: keep the timestamped `YYYY-MM-DD_HHMMSS-live-capture.wav` file after a successful final pass.
  - Off: remove the saved live capture audio after a successful final pass, but keep the session folder, `session.json`, and `final_transcript.txt`.
- **Timer**: elapsed recorded time for the current session. It freezes while live capture is paused.
- **Compute**: `Auto` (default, uses the GPU when CUDA is available), `CPU`, or `GPU`. It applies to the live session and its final post-pass, and is remembered between runs. If `GPU` is chosen without CUDA, PyScribe falls back to CPU and says so in the event log.
- **Precision**: `Auto`, `float16`, or `int8`. `float16` is GPU-only (CPU uses `int8`). The option is hidden for Nemotron, which always uses float16 on GPU and float32 on CPU.
- When Compute resolves to the GPU, PyScribe checks currently free GPU memory before live capture; choosing `CPU` skips this check. If free VRAM appears too low for the selected model, it warns before starting and suggests unloading LM Studio, reducing GPU layers, choosing a smaller Whisper model, or switching to CPU/int8.
- Each live session folder contains:
  - `YYYY-MM-DD_HHMMSS-live-capture.wav`
  - `session.json`
  - `final_transcript.txt` after a successful stop/finalize cycle

### Diarization Controls

- **Mode**: backend selector for diarization engine.
  - `Accurate` (pyannote): the diarization engine (the former `Fast` mode was identical and has been removed).
- **Max Speakers**: optional speaker cap (blank = auto).
- Pyannote diarization backends run in a separate worker process so GPU speaker ID can stay isolated from CUDA ASR runtime state.
- If GPU diarization is unavailable, PyScribe retries diarization on CPU before giving up on speaker labels.
- On modern Torchaudio releases, PyScribe uses `soundfile` fallbacks for metadata/loading APIs that pyannote expects, and passes decoded audio to pyannote in memory so `torchcodec`/FFmpeg are not needed for file IO.
- With pyannote.audio 4.x, diarization uses `pyannote/speaker-diarization-community-1` (CC-BY-4.0, gated): accept its terms on Hugging Face with the account that owns your token. When a transcript has word timestamps, speakers are assigned per word, so a speaker change in the middle of a sentence starts a new line.
- If diarization fails or produces no speaker segments, PyScribe keeps the plain transcript instead of filling the output with `[S?]` speaker labels.
- Diarization progress bar:
  - Disabled when transcription is off.
  - Uses staged determinate progress for backend initialization, model loading, inference, speaker assignment, and completion.

### Visual Analysis Controls

- **Mode**:
  - `fast` (fewer OCR calls, fastest)
  - `balanced` (default)
  - `accurate` (most OCR coverage)
- **OCR Backend**:
  - `rapidocr`
  - `paddleocr`
  - `surya`
  - `pytesseract`
  - `auto` (best available fallback; PaddleOCR leads only when it will run on the GPU, otherwise RapidOCR)
- **PaddleOCR on GPU**: `requirements.txt` installs the CPU `paddlepaddle` build (about 5 s per frame). Run `scripts/install_paddle_gpu.sh` to install the CUDA build instead (about 0.08 s per frame, about 1.2 GB VRAM); it shares the CUDA libraries that torch already installs. Re-run the script after any `pip install -r requirements.txt`. PaddleOCR uses the GPU only when at least 1.5 GB of VRAM is free and falls back to CPU (at initialization or on a run-time GPU error) otherwise.
- **OCR Fallback**: which backend to try first when the main one cannot run (for example PaddleOCR cannot verify its models because Hugging Face is unreachable). `auto` (default) keeps the built-in order; or pick `rapidocr`, `pytesseract`, or `surya`. A fallback that is not installed is skipped and the next available backend is used. The report note says why the main backend was not used and which fallback ran. The choice is saved as soon as you change it.
- **Scope**:
  - `Slides only` (default; avoids noisy chat/meeting side panels)
  - `Slides + chat` (captures the right-side chat/panel crop when useful)
- **Sample every (sec)**:
  - Lower values = more frame coverage, slower runtime.
  - Clamped to `0.5` to `10.0`.
- Long videos use lower frame caps and prefer faster auto OCR backends to reduce webinar OCR runtime.

Fallback behavior:

- If selected OCR backend is unavailable, Qt offers fallback options where possible.
- For backend first-use downloads (for example PaddleOCR model files), Qt asks for confirmation.

### Job Controls

- **Process File**: start run.
- **Start Live**: begin live microphone or loopback capture.
- **Pause / Resume** (live mode): temporarily suspend or resume live capture while keeping the same session folder and timestamped live capture file.
- **Stop** (live mode): stop capture cleanly, finalize the rolling draft, and start the final post-pass on the saved recording.
  - After a successful final post-pass, live mode remains ready for another live session and restores the Live Capture controls.
- **Cancel** (job status bar, visible while work is active): cooperative cancellation.
  - In live mode, cancel asks for confirmation, then stops capture immediately, skips the final post-pass, and preserves the session folder and recorded audio.
- **Force stop** (Job details drawer): immediate process termination if cancellation stalls; Qt escalates from terminate to kill when needed.
  - In live mode, force stop preserves the live session folder and recorded audio.
- **Exit**: close app.

### Output Controls

- **Transcript panel**: live transcript text output.
  - In live mode, this shows a rolling draft first, then the final cleaned transcript after **Stop** completes.
- **Copy**: copy transcript panel text to clipboard.
- **Save menu**:
  - **Save All (Transcript + OCR)**
  - **Save Transcript Only**
  - **Save OCR Only**
- Save dialog defaults to source media folder when available.
- Last open/save directories are remembered.
- When two or more processing parts are enabled, PyScribe automatically saves separate part files beside the source media where possible:
  - `<stem>_transcript.txt`
  - `<stem>_diarized.txt`
  - `<stem>_ocr.txt`
  - Existing files are preserved with numbered suffixes.

### Status + Timing

- The status line under the file path shows the current stage and results.
- The job status bar and the Job details drawer (see Layout + Navigation) show stage, elapsed time, the per-stage timeline and a terminal-style event log.
  - Live mode logs device selection, recording start/pause/resume/stop, final post-pass handoff, and preserved session paths on cancel/failure.
- HF token status (`configured` / `not configured`) is shown in the status bar at the bottom of the window.
- The Performance drawer shows CPU/RAM and GPU/VRAM traces when available.

### Batch queue section

- On Transcribe, **Batch queue** is a collapsible section under the settings. It stays closed until you add files, then opens and shows the item count.

### Responsive Behavior

- The setup column and the Settings page scroll when the window is short; the transcript keeps at least 300 px of width.
- Panels that were movable in earlier versions are gone. Old saved dock layouts are ignored.

### Menus

Qt menu bar includes **Tools**, **View**, and **Help**.

### Tools

- **HF Token...**
  - Save Hugging Face token for gated/private model access.
  - Shortcut: `Ctrl+Shift+T`
- **Benchmark...**
  - Benchmark selected models using bundled sample audio.
  - Shortcut: `Ctrl+B`
- **LLM Connections...**
  - Configure enabled local, LAN, and hosted (`cloud`) LLM profiles.
  - Supports `ollama`, `lm_studio`, OpenAI-compatible endpoints, and Anthropic (native Claude API).
  - **Add Cloud Profile** adds a starting point for Claude, OpenAI, Gemini, or OpenRouter. **Test Connection** lists the models your key can use; any model your provider offers can be the default.
  - Cloud profiles must use `https://`, need an API key and a ticked confirmation that transcripts and images are sent to the provider, and always verify TLS.
  - **Claude CLI (`claude_cli`), personal use only.** Runs the `claude` program you already installed and signed in to with your own account. It needs no URL or key, still needs the confirmation box (transcripts leave this computer), and is treated as a cloud profile, so the Listener and MCP gates above apply. PyScribe never reads or stores the CLI's sign-in; it only starts the program, in an empty temporary folder, with tools and MCP servers switched off, sending the prompt over stdin. `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are deliberately not passed to it, so it uses your own sign-in instead of an API key. Do not use this to serve other people: Anthropic's terms allow your own subscription in the unmodified CLI, not routing requests for others. Not available for images (the OCR fallback is used). Only the Claude Code CLI is supported. In the dialog, **Add CLI Profile** creates one: choosing it (or switching a profile's provider to `claude_cli`) greys out and clears Base URL, API Key and the keyring option and fixes the scope to `cloud`. The confirmation box reads that transcripts are sent through your signed-in CLI to the vendor. If the program is missing or not signed in, **Test Connection** shows the plain-language reason. Switching a profile that held a keyring key to a CLI provider removes that key from the keyring when you press **Save and Close**.
  - The form follows the provider and scope: rows that do not apply are hidden. For example, Anthropic shows only the URL, key, model settings and the cloud confirmation (scope fixed to `cloud`, URL reset to `https://api.anthropic.com` if it was plain `http` or a private address); Ollama and LM Studio offer `local` and `lan`; Allowed CIDRs appear only for `lan`; Verify TLS and the concurrent-run option only for `local` and `lan`; the Claude CLI shows no URL, key, TLS, CIDR, concurrency or temperature rows. Saved profiles are never changed when you open them; a profile with an unusual scope is shown as saved with a short note. CIDRs are not saved for non-`lan` profiles.
  - API key field supports `env:VAR_NAME` references for secure persisted config.
  - Direct API keys are treated as session-only and are not written to disk, unless you tick **Store in system keyring** (shown only when your system keyring is usable). The key then lives in the operating system keyring and the config file only holds an opaque `keyring:<id>` reference; the field shows "stored in system keyring" instead of the key.
  - A stored key is removed from the keyring only when you delete the profile, replace or clear the key, or switch the profile to `env:` or a session key, and only once you press **Save and Close** (Cancel keeps the previous state). If the keyring is locked or missing, PyScribe says so and does not save the key anywhere else.
  - **Test Connection** (here and in LLM Post-Process) runs in the background, so the window stays responsive; the button is disabled while it runs, and you can close the window at any time.
  - **Context tokens**, **Max output tokens**, and **Temperature** are per profile (empty = automatic or provider default).
  - Includes subnet detection and LAN scan utilities to discover reachable local-network endpoints.
  - Shortcut: `Ctrl+Shift+L`
- **LLM Post-Process...**
  - Run prompt-template post-processing against the current transcript/OCR context.
  - Shortcut: `Ctrl+Shift+P`
- **Process Existing Transcript...**
  - Open post-processing workflow for previously saved transcript files.
  - This is useful for offline or delayed summarization/review.

### View

- **Job details** (`Ctrl+Shift+J`) and **Performance** (`Ctrl+Shift+H`)
  - Open or close the drawers described in Layout + Navigation.
- **Theme**
  - `System`
  - `Light`
  - `Dark`
  - Theme preference is persisted across launches.
- **Colour theme**
  - Five presets (Graphite Studio, the default, plus Iron-gall, Verdigris, Ochre, Graphite), each with light and dark versions, plus your own. Existing choices are kept.
  - **Edit themes...** opens the editor: click a swatch to change a colour with live preview; text colours that would be hard to read are adjusted automatically; presets can't be changed, so editing one saves a copy; custom themes can be duplicated, renamed, deleted, exported, and imported (`.json`).
  - The Listener uses the same colour theme when it starts.

### Help

- **PyScribe Help**
  - Shortcut: Help key / `F1`
- **Model Help**
- **Open Logs Folder**
- **About PyScribe**

### Language Handling

- Language auto-detection is attempted before run.
- For `.en` models with non-English detected audio, Qt prompts to force English or cancel.
- For non-English detected audio on non-`.en` models, Qt prompts to use detected language or force English.
- If detection fails, run continues with model auto behavior.
- Live mode uses model auto language behavior during rolling ASR and then reprocesses the saved capture during the final post-pass.

### LLM Post-Processing Workflow (Qt)

- Open **Tools > LLM Connections...** and configure at least one enabled profile.
- Use **Detect Networks** and **Scan Selected Network** to find reachable endpoints on detected subnets.
- In multi-network environments (for example LAN + VPN), pick the target detected subnet before scanning.
- Optionally run **Test Connection** to validate endpoint reachability/auth/model discovery.
- Scope policy is enforced at run time (not only during connection tests).
- Open **Tools > LLM Post-Process...** for the current transcript, or
  **Tools > Process Existing Transcript...** to load a saved transcript file.
- From the **Templates** page, pick a template and press **Use on a transcript...**; from the Transcript Studio, **Generate...** on the Summary or Action items tab does the same for the open project and stores the result with it.
- The dialog uses a split workspace:
  - Left: configuration + attachments
  - Right: input context, payload preview, and output panes
- Select profile, template, and model, then run post-processing.
- You can create/edit/delete custom user templates in the same dialog (built-ins remain read-only).
- Use **Pasted Context**, optional image attachments, and **Payload Preview** to review exactly what will be sent before execution.
- **Cancel Generation** now prompts for confirmation and immediately requests cancellation.
- Closing the dialog during active generation prompts to cancel before the window closes.
- For image attachments:
  - Multimodal-capable models receive image content directly.
  - Text-only models can use OCR fallback to convert image context into text.
- Concurrency policy:
  - Local profiles are blocked while local transcription is in progress.
  - LAN profiles may run concurrently only if profile concurrent mode is enabled.
  - Cloud profiles do not compete for local compute and may run at any time.
- Long transcripts:
  - Each profile has a context size (automatic by default: 16k for Ollama, 8k for LM Studio and other local servers, large for hosted models).
  - A transcript that does not fit is split into parts, each part is processed, and the results are merged; the status line shows progress.
  - A reply that stops at the output limit is flagged so you know it may be cut off.
- The Listener hides cloud profiles unless `llm_allow_cloud_in_listener` is `true` in `~/.pyscribe_config.json` (anyone who can open the Listener could otherwise use your key).

### Templates page

- Lists built-in and your own prompt templates with a preview of the system and user prompts, output format and tags.
- **New template...** creates one; **Edit...** and **Delete...** work on your own templates (built-ins are read-only); **Set as default** picks which of your templates is preselected; **AI connections...** opens the connections dialog.
- Templates are stored as before: built-ins in `assets/prompts/`, yours in `~/.pyscribe/prompts`.

### Settings

- **Appearance & accessibility**: theme mode (System, Light, Dark), colour theme with **Edit themes...**, transcript text size (12-28 px) and **Reduce motion** (Automatic follows the `PYSCRIBE_REDUCED_MOTION` environment variable, On, Off).
- **Privacy & storage**: where projects are saved, with **Change...**, **Reset** and **Open**.
- **AI connections** opens the LLM Connections dialog. API keys are set per connection.
- **Start-up defaults**: default model (empty = last model used), start in File or Live, names/terms, faster GPU decoding, and the folder file dialogs open in. **Use current settings as defaults** copies the Transcribe page's current choices.
- **Diagnostics**: the Performance panel, **Run benchmark...**, the Hugging Face token dialog, and the logs folder. None of it is needed for everyday transcription.
- Remembered automatically: run mode, speaker and visual options (saved when a job starts), live capture choices, theme and colour theme, text size, reduce-motion choice, window size, and whether the rail is collapsed.

### MCP server

- `python main.py mcp` runs PyScribe as an MCP server over stdio so Claude Code, Codex, and other MCP clients can transcribe audio, run OCR on a video or image (`analyze_visuals`), read transcripts, and run a saved template through your own LLM profile (`run_template`; cloud profiles need `PYSCRIBE_MCP_ALLOW_CLOUD=1`). See `docs/mcp.md`.

## 4) Gradio Listener Features

### Server Launch Behavior

- Binds to `127.0.0.1` by default.
- If preferred port is unavailable, listener tries fallback ports.
- Queue is enabled with concurrency limit 1 (serialized jobs per host process).

### Listener Text Size

- A small **A− 100% A+** widget is fixed to the top-right corner of the page.
- **A−** / **A+** change the text size from 80% to 200% in 10% steps; clicking the percentage resets to 100%.
- The setting applies to labels, inputs, transcript boxes, and descriptive text, and is stored in the browser (`localStorage`), so it is per browser and per device, not per server. If the browser blocks storage, the size resets to 100% on reload.
- It is separate from the browser's own zoom (Ctrl/Cmd `+` / `-`), which still works and stacks with it.
- Implementation note: Gradio defines several derived font-size variables as fixed pixel values, so `CUSTOM_CSS` in `app.py` scales those directly. After a Gradio upgrade, re-check that all text still scales.

### Listener UI Inputs

The page follows the same hierarchy as the desktop app: the basics first, tuning behind **Advanced**.

- **Upload Audio/Video File** and **Transcribe**
- **Select Transcription Model** (editable)
- **Identify speakers**
- **Advanced** (closed by default):
  - **Run mode**: `full`, `transcribe_only` or `visual_only`
  - diarization mode and maximum speakers
  - names / terms and faster GPU decoding
  - **Analyze visuals** + visual mode/backend/sample interval controls

Visibility of controls adapts to run mode and toggles.

### Listener Actions

- **Transcribe**: starts run.
- **Cancel** (shown during active run): sets cancellation flag.
- **Transcript** tab: **Copy to Clipboard** and **Save Transcript** (prepares a downloadable text file).
- **AI summary** tab (LLM post-processing, beta):
  - Pick configured LLM profile + prompt template.
  - Test connection and fetch model list.
  - Choose transcript source (`Current transcript` or `Upload/paste transcript`).
  - Optionally upload OCR/context text, add extra notes, include pasted context, and attach images.
  - Enable/disable image include and OCR fallback behavior for text-only models.
  - Preview final request payload before sending to the configured model.
  - Run post-processing and save/copy generated output.
- A footer notes that processing happens on the machine running the Listener. The model tips and LLM help are in the **About models and LLM post-processing** section at the bottom.

### Listener Outputs

- **Status**
- **Stage strip** (Load model / Transcribe / Speakers / Visuals chips under Status; shows each stage as waiting, running with a percent, done, failed, or off when that step is not used in the chosen mode; it follows the Listener colour theme in light and dark)
- **Transcription**
- **Final status** (completion/cancel summary)
- **Download Transcript** file output
- **LLM status**
- **LLM output**
- **Download LLM Output** file output

## 5) CLI Features

### Main Entry

```bash
python main.py
python main.py qt
python main.py serve [options]
python main.py mcp
```

`mcp` runs the MCP server over stdio (see `docs/mcp.md`).
### `serve` Options

- `--host` (default `127.0.0.1`)
- `--port` (default `7860`)
- `--max-port-tries` (default `50`)
- `--queue-size` (default `16`)
- `--auth-user` (optional username)
- `--allow-nonlocal-host` (required for non-local bind)
- `--share` (Gradio public share link)

### Listener Security Rules

- Non-local bind is rejected unless `--allow-nonlocal-host` is set.
- Non-local bind also requires auth credentials.
- `--share` also requires auth credentials, even on localhost binds.
- Password must be supplied by environment variable (`PYSCRIBE_AUTH_PASS`).
- Interactive LAN mode uses password from `PYSCRIBE_LAN_AUTH_PASS` or secure prompt input.
- Legacy `--auth-pass` CLI argument is intentionally rejected.

## 6) Model Download + Auth Features

- Cached model reuse is automatic.
- Qt mode prompts before downloading uncached model repos.
- Listener mode downloads as needed (with progress/status updates).
- HF token sources:
  1. `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`
  2. saved token via Hugging Face cache

## 7) Logging Features

PyScribe uses a consolidated `pyscribe.log` file for the active session, with automatic timestamped archiving of previous logs on startup.

- **Consolidated Logs**: the active session always logs to a single file named `pyscribe.log`.
- **Session Archiving**: on application startup, the previous `pyscribe.log` is automatically moved to a timestamped file (for example `pyscribe_20260428_130148.log`).
- **Automatic Rotation**: the system automatically scans the log directory on startup and keeps only the **21 most recent** archived log files to manage disk space.

Log directory priority:

1. `PYSCRIBE_LOG_DIR/pyscribe.log` (if set)
2. `~/.pyscribe/logs/pyscribe.log`
3. `./.pyscribe_logs/pyscribe.log`
4. OS temp fallback

Logging environment variables:

- `PYSCRIBE_LOG_LEVEL` (default `INFO`)
- `PYSCRIBE_LOG_STDOUT` (`1` enables stdout logging)
- `PYSCRIBE_LOG_DIR` (custom directory)

## 8) Runtime Environment Features

PyScribe sets defaults for writable caches and runtime compatibility.

Common environment variables:

- `PYSCRIBE_CACHE_DIR`
- `PADDLE_HOME`
- `PADDLE_PDX_CACHE_HOME`
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`
- `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT`
- `PADDLE_PDX_MODEL_SOURCE`
- `PADDLE_PDX_HUGGING_FACE_ENDPOINT`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `MODELSCOPE_CACHE`
- `XDG_CACHE_HOME`
- `FFMPEG_PATH`

Listener-related variables:

- `PYSCRIBE_AUTH_USER`
- `PYSCRIBE_AUTH_PASS`
- `PYSCRIBE_ALLOW_NONLOCAL_HOST`
- `PYSCRIBE_HOST`, `PYSCRIBE_PORT`, `PYSCRIBE_MAX_PORT_TRIES`, `PYSCRIBE_QUEUE_SIZE` (used by `scripts/run_listener.sh`)
- `PYSCRIBE_LAN_AUTH_USER`, `PYSCRIBE_LAN_AUTH_PASS` (interactive LAN launcher)

MCP server variables (see `docs/mcp.md`):

- `PYSCRIBE_MCP_ROOTS` (folders the server may read media from; default is your home folder)
- `PYSCRIBE_MCP_ALLOW_CLOUD` (`1` lets `run_template` use cloud profiles)

## 9) Benchmark Feature

Qt benchmark dialog supports:

- selecting multiple models
- selecting benchmark language (English/Spanish bundled sample)
- progress reporting
- cancellation

## 10) Troubleshooting Quick Checks

- Ensure `ffmpeg` is installed and in PATH.
- Confirm required Python deps are installed in active environment.
- For gated/private models, configure HF token and accept model terms.
- For Qt live loopback capture on Linux, confirm your audio stack exposes a monitor/loopback input.
- For OCR backends, install runtime dependencies (`pytesseract` + OS package, or PaddleOCR stack).
- If Linux dynamic library issues occur, run from a clean shell and avoid conflicting injected library paths.
