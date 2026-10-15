"""Render the README screenshots offscreen with synthetic data (no personal transcripts, names or paths).

Usage (from the project root, with the project venv):

    QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/capture_screenshots.py [--out assets/images] [--only home,studio]

Images are named ``YYYY-MM-DD-qt-<name>.png``. The Listener (Gradio) screenshot is not covered here; capture it
from a headless browser against ``python main.py serve``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
import random
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from services.config_service import AppConfig  # noqa: E402
from services.model_service import RuntimeInfo  # noqa: E402
from services.project_store import ProjectStore, sanitize_segments  # noqa: E402

LOGGER = logging.getLogger("capture_screenshots")

CONVERSATION = [
    ("S1", "Thanks everyone for joining. I'd like to use this time to review Q1 performance, align on Q2 priorities, and discuss any roadblocks."),
    ("S2", "Sounds good. I'll start with a quick recap of our results."),
    ("S2", "Overall, we finished Q1 at 12% year-over-year growth, which exceeded our target of 8%. This was driven by strong demand in the enterprise segment and continued momentum in self-serve."),
    ("S3", "That's great to see. Can you break down which segments contributed the most to that growth?"),
    ("S2", "Yes. Enterprise was up 18%, SMB was up 9%, and self-serve was up 6%. Enterprise accounted for roughly 60% of total revenue this quarter."),
    ("S4", "On the product side, we shipped the new analytics dashboard in March, which has received really positive feedback from early customers."),
    ("S1", "Nice. What are we hearing about adoption?"),
    ("S4", "Initial signals are strong. About 40% of eligible accounts have tried it, and of those, 70% are using it weekly."),
    ("S3", "That's a solid start. I'd like to double down on enablement and customer education in Q2 to increase that to at least 60% adoption."),
]
NAMES = {"S1": "Sarah", "S2": "Michael", "S3": "Emma", "S4": "David"}
SUMMARY = (
    "The team reviewed **Q1 results**, which exceeded targets with 12% year-over-year growth, driven by enterprise "
    "demand and self-serve momentum.\n\n- Analytics dashboard shipped in March\n- Adoption at 40% of eligible accounts"
)
ACTIONS = "- [ ] Increase enablement and customer education in Q2 (Emma)\n- [ ] Share detailed adoption metrics (Michael)"


def write_tone(path: Path, seconds: float = 90.0) -> None:
    rate = 8000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = (int(9000 * math.sin(i / 11.0) * abs(math.sin(i / 9000.0))) for i in range(int(rate * seconds)))
        handle.writeframes(b"".join(struct.pack("<h", f) for f in frames))


def pump(app: QApplication, seconds: float = 0.5) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()


def seed_projects(store: ProjectStore, media: Path) -> str:
    """Three recent projects; returns the id of the one with the full Studio content."""
    rng = random.Random(3)
    peaks = [round(rng.random() * 0.9 + 0.05, 2) for _ in range(64)]
    segments, clock = [], 0.0
    for speaker, text in CONVERSATION:
        length = 4 + len(text) / 14
        segments.append({"start": clock, "end": clock + length, "speaker": speaker, "text": text})
        clock += length
    studio_id = ""
    for title, status, duration in (("Quarterly Review", "completed", 3137.0), ("Vendor Call", "completed", 1683.0), ("Product Planning", "transcribing", 3984.0)):
        project = store.create(title=title, media_path=str(media) if title == "Quarterly Review" else "")
        store.update(
            project.id, status=status, duration_s=duration if title != "Quarterly Review" else 90.0,
            segments=sanitize_segments(segments)[:3] if title != "Quarterly Review" else sanitize_segments(segments),
            preview_peaks=peaks,
            **({"speaker_names": NAMES, "ai_outputs": {
                "meeting-summary": {"text": SUMMARY, "created": "", "profile": ""},
                "action-items": {"text": ACTIONS, "created": "", "profile": ""}}} if title == "Quarterly Review" else {}),
        )
        if title == "Quarterly Review":
            studio_id = project.id
    return studio_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("assets/images"))
    parser.add_argument("--only", default="", help="comma-separated names to render (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    wanted = {name for name in args.only.split(",") if name}
    stamp = datetime.date.today().isoformat()
    args.out.mkdir(parents=True, exist_ok=True)

    runtime = RuntimeInfo(device="cpu", compute_type="int8", gpu_name="N/A", vram_gb=0.0, cpu_count=8)
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PYSCRIBE_PROJECTS_DIR": tmp}):
        media = Path(tmp) / "review.wav"
        write_tone(media)
        app = QApplication.instance() or QApplication([])

        def render(name: str, mode: str, theme_id: str, show) -> None:
            if wanted and name not in wanted:
                return
            config = AppConfig(theme_mode=mode, theme_id=theme_id)
            with patch("ui_qt.main_window.detect_runtime", return_value=runtime), \
                 patch("ui_qt.main_window.load_config", return_value=config), \
                 patch("ui_qt.main_window.save_config"), \
                 patch("ui_qt.main_window.list_live_audio_inputs", return_value=[]), \
                 patch("ui_qt.views.projects_home.getpass.getuser", return_value=""):
                from ui_qt.main_window import MainWindow

                window = MainWindow()
                window.resize(1366, 768)
                window.show()
                store = window.project_store  # already rooted in the temporary folder
                project_id = seed_projects(store, media)
                window._refresh_projects()
                show(window, project_id)
                pump(app)
                target = args.out / f"{stamp}-qt-{name}.png"
                window.grab().save(str(target))
                LOGGER.info("wrote %s", target)
                window._live_capture_active = False
                window.close()
                window.deleteLater()  # a closed window is still restyled on every theme change
                QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                for leftover in store.list_summaries():
                    store.delete(leftover.id)

        def home(window, _id):
            window.nav_rail.set_current("projects")

        def studio(window, project_id):
            window._open_project(project_id)
            pump(app, 1.5)
            window.studio.transport.seek_ms(31000)

        def transcribe(window, _id):
            window.nav_rail.set_current("transcribe")

        def live(window, _id):
            window.nav_rail.set_current("live")
            window._live_capture_active = True
            window.live_title_input.setText("Weekly standup")
            window.text_area.setPlainText("[S1] Morning everyone. Quick round of updates before we start.\n[S2] I finished the export work and started on the queue.")
            window._update_live_mode_ui()

        def settings(window, _id):
            window.nav_rail.set_current("settings")
            window.projects_dir_input.setText("~/.pyscribe/projects")
            for button in (window.projects_dir_browse_btn, window.projects_dir_reset_btn):
                button.setEnabled(True)  # the capture sets PYSCRIBE_PROJECTS_DIR, which would grey these out

        def templates(window, _id):
            window.nav_rail.set_current("templates")

        render("home-dark", "dark", "graphite-studio", home)
        render("home-light", "light", "graphite-studio", home)
        render("studio-dark", "dark", "graphite-studio", studio)
        render("studio-light", "light", "graphite-studio", studio)
        render("transcribe-light", "light", "graphite-studio", transcribe)
        render("live-dark", "dark", "graphite-studio", live)
        render("settings-light", "light", "graphite-studio", settings)
        render("templates-light", "light", "graphite-studio", templates)
        render("theme-ochre-light", "light", "ochre", studio)
        render("theme-graphite-dark", "dark", "graphite", studio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
