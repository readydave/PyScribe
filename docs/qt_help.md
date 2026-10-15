# PyScribe Qt Help

## Quick Start

1. On **Projects**, drop a recording on **New transcription** (or click it to choose a file). **Live capture** starts a recording and **Import folder** queues a whole folder.
2. On **Transcribe**, pick a model in the **Model** dropdown (built-in or custom `owner/repo`) and click **Process File**. Speaker identification, names/terms and visual analysis are under **More options**.
3. For a live recording, open **Live**, check the source and device (optionally add a session title) and click **Start Live**. Use **Pause / Resume** when needed, then **Stop** to run the final post-pass. Output folder, compute and precision are under **More live options**.
4. When the job finishes, the **Transcript Studio** opens with the result. Search, play, edit, name the speakers, generate a summary and export from there. The project is saved automatically and is always on **Projects**.

## Navigation

- The rail on the left has **Projects**, **Transcribe**, **Live**, **Templates** and **Settings**. The small button in its header collapses it; the choice is remembered.
- **New Project** (bottom of the rail) clears the selected file, transcript and progress (asks first if there is a transcript; blocked while a job runs). Settings and the batch queue are kept.
- While a job runs, a status bar appears at the top of every page: stage, elapsed time, progress, **Details** and **Cancel**.
  - **Details** opens the **Job details** drawer: one row per stage (Load model, Transcribe, Speakers, Visuals, Save), the event log, and **Force stop** if cancelling stalls. `Ctrl+Shift+J`.
  - **Show performance** (or `Ctrl+Shift+H`) opens the **Performance** drawer: 60-second CPU, memory, GPU and VRAM traces, coloured by the running stage (GPU rows appear only when a GPU is detected).
  - Only one drawer is open at a time; `Esc` closes it.
- HF token status (`configured` / `not configured`) is in the status bar at the bottom of the window.

## Projects Home

- Three cards at the top: **New transcription** (drag and drop or click), **Live capture**, **Import folder**.
- **Recent projects**: your latest six, with duration, speakers, a waveform cue, status and last activity. **View all** shows every project. The **...** menu on a card: Open, Rename, Show in folder, Delete (the recording itself is never deleted).
- Projects are saved in `~/.pyscribe/projects` (change it under **Settings > Privacy & storage**).

## Transcript Studio

- **Transcript**: timestamp, speaker badge and name, text. The line being played is highlighted.
  - Click a line (or select it and press `Enter`) to jump there; `Space` plays or pauses.
  - Double-click a line (or press `F2`) to edit it; `Enter` saves, `Esc` cancels. Edits are saved automatically.
- **Search** (`Ctrl+F`): highlights matches and shows the count; `Enter` or the arrows step through them; `Esc` clears.
- **Playback bar**: play/pause, back/forward 10 seconds, time, waveform (click or drag to seek; Left/Right arrows move 5 seconds), speed 0.75x to 2x. If the recording is no longer available the transcript still works and a note explains.
- **Inspector** (right, one tab at a time, collapses with the » button):
  - **Summary** and **Action items**: **Generate...** opens the LLM dialog with the right template; nothing is sent until you run it, and the result is kept with the project.
  - **Speakers**: name each speaker; turns and speaking time are shown.
  - **Visuals**: OCR text from slides or chat.
- **Export**: text, Markdown, SRT, WebVTT or JSON, using your speaker names.
- **Local only** means transcription ran on this device; AI actions only send text where your connection profile points.

## Transcribe and Live

- **Drop zone**: choose a local media file via drag/drop or **Browse Files**.
- **Model**: supports built-in model names and custom Hugging Face repo IDs; the recommended model for your hardware is shown under it.
- **More options** (Transcribe): speaker identification, names/terms, faster GPU decoding, and visual analysis. It starts collapsed and remembers whether you opened it.
- **Batch queue** (collapsible, under the settings): add files or a folder and process them one after another. It opens when you add files.
- **Live** page: a state chip (Ready, Recording, Paused, Finalizing), source, device, session title and timer first; **More live options** holds output folder, compute (Auto, CPU or GPU), precision and keeping the recording.
- **Status** line: under the file path.
- Speaker labels are colour-coded in the transcript.
- The setup column scrolls sideways on very narrow windows (below about 900 px) instead of cutting controls off; collapse the rail for more room.

## AI Connections (Tools > LLM Connections)

- **Providers**: Ollama, LM Studio, any OpenAI-compatible server, and Anthropic (native Claude API).
- **Scope** controls where a connection may point:
  - `local`: this machine only.
  - `lan`: private-network addresses only.
  - `cloud`: a hosted provider over `https://`. Transcripts and images leave your network, so each cloud profile needs the confirmation box ticked, an API key, and keeps TLS verification on.
- **Add Cloud Profile** adds a starting point for Claude, OpenAI, Gemini, or OpenRouter. Press **Test Connection** to list the models your key can use, then type or paste the one you want as the default model. Any model your provider offers works; nothing is hard-coded.
- **API keys** are never saved as plain text. Enter an environment variable name such as `env:ANTHROPIC_API_KEY` (saved), or paste a key to use for this session only. If your system keyring is usable (optional `keyring` package), **Store in system keyring** keeps a pasted key in the OS keyring; the config file then holds only a `keyring:<id>` reference.
- **Claude CLI (`claude_cli`, personal use only)**: **Add CLI Profile** creates a profile that runs your own signed-in `claude` program, with no URL or key. It counts as a cloud profile (confirmation box, hidden from the Listener and MCP unless allowed) and runs with tools and MCP servers off. **Test Connection** explains what is missing if the program is not installed or signed in. Only the Claude Code CLI is supported.
- The form shows only the settings that apply to the chosen provider and scope.
- **Context tokens** is how much the model can read at once (automatic by default: 16k for Ollama, 8k for LM Studio and other local servers, large for hosted models). Transcripts longer than this are split into parts, summarised, and merged automatically; the status line shows progress. For LM Studio, set this to the context length you loaded the model with.
- **Max output tokens** caps the length of the reply; PyScribe tells you if a reply was cut off.
- **Temperature**: empty uses the provider's default (some hosted models reject a custom value).
- The Listener does not offer cloud profiles unless `llm_allow_cloud_in_listener` is set to `true` in `~/.pyscribe_config.json`, because anyone who can open the Listener could otherwise use your key.

## Templates

- Lists the built-in prompt templates and your own, with a preview of the prompts.
- **New template...**, **Edit...**, **Delete...** and **Set as default** work on your own templates; built-ins are read-only. **Use on a transcript...** opens the LLM dialog with that template.

## Settings

- **Appearance & accessibility**: theme mode, colour theme and **Edit themes...**, transcript text size (12-28 px), and **Reduce motion** (turns off sliding panels and the busy-bar animation).
- **Privacy & storage**: the projects folder (**Change...**, **Reset**, **Open**). New projects go to the new folder; older ones stay where they were.
- **AI connections**: opens the connections dialog.
- **Start-up defaults** sets what PyScribe starts with:
  - **Model**: leave empty to start with the last model you used, or enter a model name or Hugging Face repo ID to always start with it.
  - **Start in**: File or Live.
  - **Names / terms** and **Faster GPU decoding**: pre-filled on the Transcribe page.
  - **Open files from**: the folder file dialogs open in.
  - **Use current settings as defaults** copies the Transcribe page's current model, File/Live choice, names/terms, and decoding option.
- **Diagnostics**: Performance panel, **Run benchmark...**, Hugging Face token, logs folder.
- Remembered automatically: run mode, speaker and visual options (saved when you start a job), live capture source/device/folder/keep-audio/compute choice, colour theme and mode, text size, window size, and whether the rail is collapsed.

## Colour Themes

- **Settings > Appearance & accessibility** (or **View > Theme** and **View > Colour theme**) chooses System, Light, or Dark and the colours: Graphite Studio (the default), Iron-gall, Verdigris, Ochre, Graphite, or one of your own. Every theme has a light and a dark version.
- **Edit themes...** (Settings, or **View > Colour theme**) opens the editor:
  - Pick a theme on the left; presets can't be changed, so editing one saves a copy.
  - Click any swatch to change a colour. Changes preview immediately; **Cancel** undoes them, **Save** keeps them.
  - Text colours that would be hard to read are adjusted automatically, and the editor tells you what moved. If a background makes text unreadable, it points that out.
  - **Duplicate**, **Rename...**, **Delete**, and **Import...** / **Export...** (a small `.json` file) work on custom themes.
- The Listener uses the same colour theme. It applies when the Listener starts, so restart it after changing the theme.

## Processing Options

- **Transcribe audio**:
  - Enables/disables ASR transcription stage.
  - In live mode, transcription is always on.
- **Speaker Identification**:
  - Enables/disables diarization (speaker labels).
  - Choose backend mode and optional max speakers (blank = auto).
  - Pyannote diarization runs in a separate worker process so GPU speaker ID is isolated from CUDA ASR runtime state.
  - If CUDA diarization is unavailable, PyScribe retries diarization on CPU before dropping speaker labels.
  - If diarization produces no speaker segments, PyScribe keeps the plain transcript instead of writing `[S?]` labels for every line.
  - In live mode, diarization runs only after **Stop** during the final post-pass on the saved capture.
- **Analyze visuals (slides/chat OCR, beta)**:
  - Optional video-frame OCR to capture on-screen text.
  - Choose visual mode: `fast`, `balanced`, `accurate`.
  - Choose OCR backend: `rapidocr`, `paddleocr`, `surya`, `pytesseract`, or `auto`.
  - Choose **OCR Fallback** (`auto` by default, or `rapidocr`, `pytesseract`, `surya`): the backend tried first when the main one cannot run. The visual report notes why and which fallback was used.
  - Choose scope: `Slides only` (default) or `Slides + chat`.
  - Set sample interval in seconds (`0.5` to `10.0`; lower = more coverage, slower runtime).
  - Long videos use lower frame caps and faster auto OCR backend preference.
  - Backend/model downloads may be prompted on first use.
  - Live mode disables visual analysis.

## Live Capture

- **Source**:
  - `Microphone`
  - `Loopback`
- **Device**: audio input filtered to the selected source type.
- **Output Folder**: root for timestamped live session folders.
- **Session Title**: optional title used for live session file naming; editable before capture and after completion, locked while capture/finalization is active.
- **Keep recorded audio after completion**:
  - when enabled, the timestamped `YYYY-MM-DD_HHMMSS-live-capture.wav` file remains after a successful final pass
  - when disabled, successful runs keep metadata and final transcript but remove the raw capture
- **Timer**: elapsed recorded time for the active session. It freezes while capture is paused.
- **Compute**: `Auto`, `CPU`, or `GPU` (applies to the session and its final post-pass; remembered). **Precision**: `Auto`, `float16`, or `int8`; hidden for Nemotron, which picks its own.
- CUDA live capture checks free GPU memory first and warns if another local workload, such as LM Studio, appears to leave too little VRAM for the selected model.
- Live mode writes a recoverable session folder containing:
  - `YYYY-MM-DD_HHMMSS-live-capture.wav`
  - `session.json`
  - `final_transcript.txt` after a successful stop/finalize cycle
- Live capture requires a timestamp-capable Whisper model or `nvidia/nemotron-speech-streaming-en-0.6b` (English only; Names / terms are applied as a spelling pass on the final transcript). Granite is unavailable in live mode.
- Loopback capture depends on the OS exposing a monitor/loopback input. On Linux that usually means a PipeWire/PulseAudio monitor source.

Run mode is inferred automatically:

- transcription + visuals -> `full`
- transcription only -> `transcribe_only`
- visuals only -> `visual_only`
- both off -> run is blocked (you must enable at least one)

## Batch Queue

- Add files or import a folder to queue multiple media files.
- Same-named files from different folders can be queued together.
- Exact duplicate paths are skipped.
- Duplicate basenames show parent-folder context in the queue list.

## Tools Menu

- **HF Token...** (`Ctrl+Shift+T`)
  - Save a Hugging Face token for gated/private model access.
  - If gated diarization still fails, accept terms on the model page in Hugging Face. Diarization with pyannote.audio 4.x uses `pyannote/speaker-diarization-community-1`, which needs its terms accepted with the same account as the token (it falls back to the older 3.1 pipeline only on pyannote.audio 3.x).
- **Benchmark...** (`Ctrl+B`)
  - Compare selected models using bundled benchmark audio.
  - Supports English and Spanish sample sets.
  - Allows multi-model selection with Start/Cancel controls.
- **LLM Connections...** (`Ctrl+Shift+L`)
  - Add/edit local, LAN, or hosted LLM profiles (`ollama`, `lm_studio`, `openai_compatible`, `anthropic`, `claude_cli`).
  - API key supports `env:VAR_NAME` for secure persisted configuration.
  - Direct API key values are session-only and are not saved to disk, unless you tick **Store in system keyring**.
  - Run staged connection tests with pass/fail diagnostics and suggested fixes.
  - Detect local interfaces/subnets and scan selected networks for reachable LLM endpoints.
  - Supports multi-network setups (for example LAN + VPN) by selecting a detected subnet before scan.
- **LLM Post-Process...** (`Ctrl+Shift+P`)
  - Run prompt-template-based post-processing on the current transcript.
  - Optionally include current OCR context, additional notes, pasted context, and image attachments.
  - Includes payload preview so you can review exactly what will be sent.
  - If the selected model appears text-only, image context can fall back to OCR text.
- **Process Existing Transcript...**
  - Open the same post-processing workflow, starting in file-load mode for saved transcript files.
  - Useful when transcription was completed in an earlier session.

## View Menu

- **Theme**
  - **System**, **Light**, **Dark**
  - Theme choice is saved and restored on next launch.

## Help Menu

- **PyScribe Help** (`F1` / Help key)
  - Opens this guide.
- **Model Help**
  - Quick guidance for custom model IDs and download behavior.
- **Open Logs Folder**
  - Opens the folder containing `pyscribe.log`.
- **About PyScribe**
  - Basic application and repository information.

## Language Prompts

Before transcription, PyScribe may detect language and prompt:

- For `.en` models with non-English detected audio: prompt to force English or cancel.
- For non-English detected audio on non-`.en` models: prompt to use detected language or force English.
- If detection fails, run continues with model auto behavior.

## Transcription Controls

- **Process File**: starts a new job.
- **Start Live**: begins live capture.
- **Pause / Resume**: temporarily suspend or resume live capture without creating a new session folder.
- **Stop**: stops live capture and starts the final post-pass on the saved recording.
  - After a completed final post-pass, live mode returns to a ready state for the next live session.
- **Cancel**: cooperative stop (safe).
  - In live mode, cancel asks for confirmation, then stops capture immediately and keeps the saved session folder without running the final post-pass.
- **Force Stop**: immediate stop if cancel is stuck; escalates to a hard kill if the worker does not exit cleanly.
  - In live mode, force stop preserves the session folder and recorded audio.
- **Save** (dropdown, default action = Save All):
  - **Save All (Transcript + OCR)**
  - **Save Transcript Only**
  - **Save OCR Only**
  - Multi-part runs also auto-save separate `<stem>_transcript.txt`, `<stem>_diarized.txt`, and/or `<stem>_ocr.txt` files beside the source media.
- **Copy**: copy transcript to clipboard.
- **Open Folder**: open selected media folder (or last-open folder when no file selected).
  - In live mode, this opens the active live session folder or configured live output root.
- **Exit**: close app (blocked while a job is running).

## LLM Post-Process Dialog

- Split layout:
  - Left: **Configuration** and **Attachments**
  - Right: **Input Context**, **Payload Preview**, **LLM Output**
- **Cancel Generation** asks for confirmation before stopping an in-flight request.
- If you close the dialog during generation, PyScribe asks whether to cancel first.
- Partial output is preserved when cancellation occurs after streaming has started.

## Model and Download Notes

- If a model is not cached, PyScribe estimates download size and asks before downloading.
- Size estimates are best-effort and may differ from actual transfer size.
- Private/gated models may require:
  1. configured HF token, and
  2. accepted model terms on Hugging Face.

## Troubleshooting

- **No file selected**
  - Select a media file first.
- **No loopback device**
  - Confirm your Linux audio stack exposes a monitor/loopback input.
  - Many PulseAudio/PipeWire setups label these devices with `Monitor` or `Loopback`.
- **No task selected**
  - Enable at least one of **Transcribe audio** or **Analyze visuals**.
- **No output / error dialog**
  - Confirm FFmpeg is installed and available in PATH.
- **Gated diarization errors**
  - Verify HF token and model access terms.
- **GPU issues**
  - Pyannote diarization runs in a separate process to avoid CUDA runtime conflicts with `faster-whisper`.
  - If GPU diarization still cannot start, PyScribe retries speaker ID on CPU automatically.
  - Modern Torchaudio metadata/loading compatibility is handled with `soundfile` fallbacks.
  - Retry with a smaller model or disable diarization if GPU memory is tight.
- **Visual analysis unavailable**
  - Install OCR runtime (`pytesseract` + OS package `tesseract-ocr`), or configure another backend.
- **Post-process blocked while transcription is running**
  - Local profiles are blocked during active local transcription to avoid GPU/compute contention.
  - Use a LAN profile with concurrent mode enabled, or wait for transcription to finish.
- **Policy check failures at run time**
  - Scope/CIDR policy is enforced when sending post-process requests, not only in connection tests.
  - For `127.0.0.1`/`localhost` endpoints, use `local` scope.
- **Template editing**
  - Built-in templates are read-only.
  - Create/edit/delete custom templates from within the LLM Post-Process dialog.
- **Image context fallback**
  - If image attachment send is blocked for a text-only model, enable OCR fallback and choose an OCR backend.

## Logging

- Log file target order:
  1. `PYSCRIBE_LOG_DIR/pyscribe.log` (if set)
  2. `~/.pyscribe/logs/pyscribe.log`
  3. `./.pyscribe_logs/pyscribe.log`
  4. OS temp fallback
- Optional environment overrides:
  - `PYSCRIBE_LOG_LEVEL=DEBUG`
  - `PYSCRIBE_LOG_STDOUT=1`
  - `PYSCRIBE_LOG_DIR=<custom path>`
