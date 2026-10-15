# UX/UI redesign, October 2026

This records what the redesign changed, why, how it is built, and what still needs a check on real hardware.
The source was the "PyScribe UX UI Redesign Implementation Specification" (a six-page PDF). All six phases are done.

## What the specification asked for

- Make the transcript, playback position, speaker context and output actions the visual centre of the app.
- Two main experiences: a **Projects Home** for starting and resuming work and a **Transcript Studio** for reviewing, editing, searching, exporting and asking AI about a recording.
- A small, stable navigation: Projects, Transcribe, Live, Templates, Settings.
- Progressive disclosure: model, OCR, hardware and recovery controls behind purposeful panels; local-first wording as a product feature; a calm status indicator instead of constant dashboard movement.
- Dark graphite default, indigo for primary action and selection, teal for success; transcript text 16-18 px and supporting UI 12-14 px; few control types; subtle motion that respects reduced-motion.
- Do not start with isolated palette or font changes; keep the back end and job controls.

## What was built

| Phase | Result |
|---|---|
| 1 Design foundation | `ui_qt/components/` (tokens, buttons, card, segmented control, chip, drawer, inspector tabs, list row, empty state, nav rail, mini waveform); component QSS; the **Graphite Studio** theme (new default); `reduced_motion` and `transcript_font_px` settings |
| 2 Projects Home | `services/project_store.py`; timed segments reach the GUI; every finished job is saved as a project; navigation rail; Projects Home with empty state |
| 3 Transcript Studio | Segment list with playback highlight and inline edit, waveform and transport bar, inspector (Summary, Action items, Speakers, Visuals), search, five export formats |
| 4 Contextual controls | Docks removed; job status bar; Job details and Performance drawers; collapsible batch queue; sectioned Settings; Templates page |
| 5 Live and Listener parity | Live page with a recording-state chip and tuning behind a disclosure; Listener with basics up front, Advanced section and Transcript / AI summary tabs |
| 6 Validation | `tests/test_acceptance.py` checks the specification's acceptance criteria; documentation and screenshots refreshed |

Back-end transcription, diarization, OCR, live capture and the LLM services were not changed. The only service-level additions are the project store, export, waveform and a few additive hooks on the LLM dialog.

## Decisions

- **Docks removed** (not kept as an expert mode): one fixed layout is simpler to keep correct. Old `dock_layout` / `dock_locked` config keys are preserved in the file and ignored.
- **Every finished job becomes a project**, stored locally (`~/.pyscribe/projects`, private permissions). A failed or cancelled job with nothing to reopen is dropped.
- **The Studio transcript is a styled list, not a text box**, so each segment can carry a timestamp, a speaker badge and a playing-line highlight like the specification's figure. Editing is per segment.
- **AI actions reuse the existing LLM dialog**, so connection scope, the cloud confirmation and payload preview stay in one place. Nothing is sent until the user runs it.
- **The media player is created only when a recording is loaded** and freed afterwards.
- **Listener colours were not changed.** Its primary button still uses the theme's red "primary" colour (pinned by `tests/test_listener_theme.py`), while the Qt app uses indigo; see the follow-ups below.

## Shortcuts

| Keys | Action |
|---|---|
| `Ctrl+F` | Search the transcript (Studio) |
| `Enter` / `Space` | Play from the selected line / play or pause (Studio transcript) |
| `F2` or double-click | Edit a line; `Enter` saves, `Esc` cancels |
| Left / Right (waveform focused) | Seek 5 seconds |
| `Ctrl+Shift+J` / `Ctrl+Shift+H` | Job details / Performance drawer |
| `Esc` | Close the focused drawer, or clear the search |
| `Ctrl+B`, `Ctrl+Shift+T`, `Ctrl+Shift+L`, `Ctrl+Shift+P` | Benchmark, HF token, LLM connections, LLM post-process (unchanged) |

## Verification done

- Full test suite (offscreen): 700+ tests, including `tests/test_acceptance.py`:
  - first screen shows no advanced settings; a file chosen on Home leads to setup with Advanced still closed;
  - search, seek, active speaker, AI output and export from one Studio workspace;
  - stage, elapsed time and Cancel visible immediately and on every page;
  - every page fits at 1366×768 and 1100×700 with no sideways scrolling, and at the 840×560 floor with nothing clipped;
  - every control on Home and in the Studio reachable with Tab, no empty tab stops, focus and selected states styled;
  - reduce motion removes the drawer animation and the busy bar;
  - a 3,000-segment (three-hour) project opens, searches and follows playback quickly;
  - contrast of the playing-line highlight, search matches and speaker names in every preset and mode.
- `bandit -ll` on the new modules and `app.py`: no new findings (the three `B104` notes in `main.py` pre-date this work).
- Screenshots were rendered offscreen (`scripts/capture_screenshots.py`) and the Listener with headless Chromium, and compared with the specification's figures in light and dark.

## Still needs a check on real hardware

These cannot be proven in the headless test environment:

- **Windows rendering** (fonts, native title bar, the Qt multimedia backend) and a **high-DPI** display at 125-200% scaling.
- **Real audio playback** on Linux (PipeWire and PulseAudio, X11 and Wayland) and on Windows, including seeking in long files and a video container.
- **Live capture** end to end with a real microphone and loopback source: the Live page and status bar, Stop, and the Studio opening on finish.
- The **GPU rows** in the Performance drawer on a machine with an NVIDIA GPU.
- A **first-time-user walkthrough** (specification phase 6): can someone import a file or start a live capture from the first screen without help?

## Follow-ups (not done)

- Listener primary action colour: align with the Qt app's indigo (change `build_theme` and its test together).
- Optional: live preview of the rolling transcript inside the Studio for a session that is still recording (today the Studio opens when the final pass finishes).
- Optional: re-render the three dialog screenshots (HF token, benchmark, LLM connections) under the new default theme.

## Regenerating the screenshots

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/capture_screenshots.py --out assets/images
```

The script uses a temporary projects folder and synthetic conversations, hides the login name in the greeting and writes `YYYY-MM-DD-qt-<name>.png`. For the Listener, start `python main.py serve --port 7862` with a temporary `HOME` and capture it with `chromium --headless=new --screenshot=... "http://127.0.0.1:7862/?__theme=dark"`.
