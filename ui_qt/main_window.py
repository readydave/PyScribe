"""PySide6 desktop frontend for PyScribe."""

from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
import os
import queue
import threading
import time

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QProgressDialog,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services import (
    AppConfig,
    detect_runtime,
    detect_language,
    get_available_diarization_backends,
    get_backend_label,
    get_hf_token,
    get_model_choices,
    normalize_model_name,
    estimate_model_download_size_bytes,
    format_bytes,
    is_model_cached,
    load_config,
    open_folder,
    recommend_model,
    resolve_repo_id,
    save_hf_token,
    save_config,
    transcribe_media_file,
)
from services.logging_service import configure_logging, get_log_path
from ui_qt.benchmark_dialog import BenchmarkDialog
from utils import load_audio_waveform
AUDIO_VIDEO_FILTER = (
    "Media Files (*.m4a *.mp3 *.wav *.flac *.aac *.ogg *.wma *.mp4 *.mov *.mkv *.avi *.flv);;All Files (*.*)"
)
ALLOWED_MEDIA_EXTS = {
    ".m4a",
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".flv",
}
_UNSET = object()
LOGGER = logging.getLogger(__name__)


class DropLabel(QLabel):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__("Drop audio/video file here")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(100)
        self.setObjectName("dropZone")

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("activeDrop", True)
            self.style().polish(self)
            return
        event.ignore()

    def dragLeaveEvent(self, event):  # noqa: N802
        self.setProperty("activeDrop", False)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        self.setProperty("activeDrop", False)
        self.style().polish(self)
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.file_dropped.emit(path)
        event.acceptProposedAction()


def _transcription_process_entry(
    media_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    use_diarization: bool,
    diar_backend: str,
    max_speakers: int | None,
    event_queue,
    cancel_event,
):
    configure_logging()

    def _emit(kind: str, value):
        event_queue.put({"type": kind, "value": value})

    try:
        def _run_with(device_name: str, compute_name: str):
            return transcribe_media_file(
                media_path=media_path,
                model_name=model_name,
                device=device_name,
                compute_type=compute_name,
                language=language,
                cancel_event=cancel_event,
                use_diarization=use_diarization,
                diar_backend=diar_backend,
                max_speakers=max_speakers,
                on_status=lambda msg: _emit("status", msg),
                on_text=lambda text: _emit("transcript", text),
                on_progress=lambda p: _emit("progress", max(0, min(100, int(p)))),
                on_diar_progress=lambda p: _emit("diar_progress", max(0, min(100, int(p)))),
                on_model_download_progress=lambda p: _emit("model_download_progress", max(0, min(100, int(p)))),
            )

        try:
            result = _run_with(device, compute_type)
        except Exception as cuda_exc:
            msg = str(cuda_exc).lower()
            if device == "cuda" and ("initialization error" in msg or "cuda failed" in msg):
                _emit("status", "CUDA init failed in worker process. Retrying on CPU...")
                result = _run_with("cpu", "int8")
            else:
                raise
        _emit(
            "finished",
            {
                "cancelled": result.cancelled,
                "transcript": result.transcript,
                "transcription_seconds": result.transcription_seconds,
                "diarization_seconds": result.diarization_seconds,
            },
        )
    except Exception as exc:
        _emit("error", str(exc))


class TranscriptionWorker(QObject):
    status = Signal(str)
    transcript = Signal(str)
    progress = Signal(int)
    model_download_progress = Signal(int)
    diar_progress = Signal(int)
    finished = Signal(bool, str, float, float)
    failed = Signal(str)

    def __init__(
        self,
        media_path: str,
        model_name: str,
        use_diarization: bool,
        diar_backend: str,
        max_speakers: int | None,
        language: str | None,
    ):
        super().__init__()
        self.media_path = media_path
        self.model_name = model_name
        self.use_diarization = use_diarization
        self.diar_backend = diar_backend
        self.max_speakers = max_speakers
        self.language = language
        self.runtime = detect_runtime()
        self._lock = threading.Lock()
        self._cancel_requested = False
        self._force_stop_requested = False
        self._mp_cancel_event = None
        self._process = None

    @Slot()
    def request_cancel(self):
        LOGGER.info("Qt worker: cancel requested")
        with self._lock:
            self._cancel_requested = True
            if self._mp_cancel_event is not None:
                self._mp_cancel_event.set()

    @Slot()
    def request_force_stop(self):
        LOGGER.warning("Qt worker: force-stop requested")
        with self._lock:
            self._force_stop_requested = True
            proc = self._process
        if proc is not None and proc.is_alive():
            proc.terminate()

    @Slot()
    def run(self):
        LOGGER.info("Qt worker: run start model=%s diar=%s backend=%s", self.model_name, self.use_diarization, self.diar_backend)
        # Always use spawn to avoid CUDA re-init issues after forking.
        mp_ctx = mp.get_context("spawn")
        event_queue = mp_ctx.Queue()
        cancel_event = mp_ctx.Event()
        process = mp_ctx.Process(
            target=_transcription_process_entry,
            args=(
                self.media_path,
                self.model_name,
                self.runtime.device,
                self.runtime.compute_type,
                self.language,
                self.use_diarization,
                self.diar_backend,
                self.max_speakers,
                event_queue,
                cancel_event,
            ),
        )
        latest_transcript = ""
        terminal_emitted = False
        force_stopped = False
        try:
            with self._lock:
                self._mp_cancel_event = cancel_event
                self._process = process
            process.start()
            LOGGER.info("Qt worker: child process started pid=%s", process.pid)

            while True:
                with self._lock:
                    cancel_req = self._cancel_requested
                    force_req = self._force_stop_requested

                if cancel_req and not cancel_event.is_set():
                    cancel_event.set()
                if force_req and process.is_alive():
                    self.status.emit("Force stop requested. Terminating worker process...")
                    process.terminate()
                    force_stopped = True

                drained = False
                while True:
                    try:
                        evt = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    etype = evt.get("type")
                    value = evt.get("value")
                    if etype == "status":
                        self.status.emit(str(value))
                    elif etype == "transcript":
                        latest_transcript = str(value)
                        self.transcript.emit(latest_transcript)
                    elif etype == "progress":
                        self.progress.emit(int(value))
                    elif etype == "diar_progress":
                        self.diar_progress.emit(int(value))
                    elif etype == "model_download_progress":
                        self.model_download_progress.emit(int(value))
                    elif etype == "finished":
                        terminal_emitted = True
                        LOGGER.info("Qt worker: finished event cancelled=%s", bool(value.get("cancelled")))
                        self.finished.emit(
                            bool(value.get("cancelled")),
                            str(value.get("transcript", latest_transcript)),
                            float(value.get("transcription_seconds", 0.0)),
                            float(value.get("diarization_seconds", 0.0)),
                        )
                    elif etype == "error":
                        terminal_emitted = True
                        LOGGER.error("Qt worker: error event from child: %s", value)
                        self.failed.emit(str(value))

                if not process.is_alive() and not drained:
                    break
                time.sleep(0.05)

            process.join(timeout=2.0)

            # Drain any final queued events.
            while True:
                try:
                    evt = event_queue.get_nowait()
                except queue.Empty:
                    break
                etype = evt.get("type")
                value = evt.get("value")
                if etype == "status":
                    self.status.emit(str(value))
                elif etype == "transcript":
                    latest_transcript = str(value)
                    self.transcript.emit(latest_transcript)
                elif etype == "progress":
                    self.progress.emit(int(value))
                elif etype == "diar_progress":
                    self.diar_progress.emit(int(value))
                elif etype == "model_download_progress":
                    self.model_download_progress.emit(int(value))
                elif etype == "finished" and not terminal_emitted:
                    terminal_emitted = True
                    LOGGER.info("Qt worker: late finished event cancelled=%s", bool(value.get("cancelled")))
                    self.finished.emit(
                        bool(value.get("cancelled")),
                        str(value.get("transcript", latest_transcript)),
                        float(value.get("transcription_seconds", 0.0)),
                        float(value.get("diarization_seconds", 0.0)),
                    )
                elif etype == "error" and not terminal_emitted:
                    terminal_emitted = True
                    LOGGER.error("Qt worker: late error event from child: %s", value)
                    self.failed.emit(str(value))

            if force_stopped and not terminal_emitted:
                LOGGER.warning("Qt worker: force-stopped before terminal event")
                self.finished.emit(True, latest_transcript, 0.0, 0.0)
        finally:
            LOGGER.info("Qt worker: cleanup begin")
            with self._lock:
                self._mp_cancel_event = None
                self._process = None
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
            except Exception:
                pass
            try:
                event_queue.close()
                event_queue.join_thread()
            except Exception:
                pass
            LOGGER.info("Qt worker: cleanup end")


class MainWindow(QMainWindow):
    hw_metrics = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyScribe Qt")
        self.resize(980, 700)

        self.runtime = detect_runtime()
        self.config = load_config()
        self.media_path: str | None = None
        self.last_open_dir = self.config.last_open_dir or os.path.expanduser("~")
        self.last_save_dir = self.config.last_save_dir
        self.transcript_text = ""
        self.worker_thread: QThread | None = None
        self.worker: TranscriptionWorker | None = None
        self.monitoring_active = False
        self.metrics_thread: threading.Thread | None = None
        self.download_progress_dialog: QProgressDialog | None = None
        self.diarization_warning: str | None = None
        self._current_use_diarization = bool(self.config.use_diarization)

        self._build_ui()
        self._build_menus()
        self._apply_theme()
        self._set_bar_color(self.progress_bar, "#dc2626")
        self._set_bar_color(self.diar_progress_bar, "#dc2626")
        self.hw_metrics.connect(self.hw_metrics_label.setText)
        self._update_diar_ui_state(self.diar_checkbox.isChecked())
        LOGGER.info("Qt MainWindow initialized runtime=%s compute=%s", self.runtime.device, self.runtime.compute_type)

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        self.path_label = QLabel("No file selected")
        self.path_label.setObjectName("pathLabel")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.on_browse)
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setObjectName("exitButton")
        self.exit_btn.clicked.connect(self.close)
        top_row.addWidget(self.path_label, 1)
        top_row.addWidget(browse_btn)
        top_row.addWidget(self.exit_btn)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        all_models = get_model_choices()
        self.model_combo.addItems(all_models)
        recommended = recommend_model(self.runtime)
        initial_model = self.config.last_model if self.config.last_model in all_models else recommended
        idx = self.model_combo.findText(initial_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        model_row.addWidget(self.model_combo, 1)
        model_hint = QLabel(f"Recommended: {recommended} ({self.runtime.device.upper()})")
        model_hint.setObjectName("hint")
        model_row.addWidget(model_hint)

        diar_row = QHBoxLayout()
        self.diar_checkbox = QCheckBox("Identify Speakers")
        self.diar_checkbox.setChecked(bool(self.config.use_diarization))
        self._update_diar_toggle_label(self.diar_checkbox.isChecked())
        self.diar_checkbox.toggled.connect(self._update_diar_toggle_label)
        self.diar_checkbox.toggled.connect(self._update_diar_ui_state)
        diar_row.addWidget(self.diar_checkbox)
        diar_row.addWidget(QLabel("Mode"))
        self.diar_backend_combo = QComboBox()
        self._diar_backends = get_available_diarization_backends(include_off=False) or ["accurate"]
        for key in self._diar_backends:
            self.diar_backend_combo.addItem(get_backend_label(key), key)
        if self.config.diar_backend in self._diar_backends:
            self.diar_backend_combo.setCurrentIndex(self._diar_backends.index(self.config.diar_backend))
        diar_row.addWidget(self.diar_backend_combo)
        diar_row.addWidget(QLabel("Max Speakers"))
        self.max_speakers_input = QLineEdit()
        self.max_speakers_input.setPlaceholderText("auto")
        self.max_speakers_input.setFixedWidth(80)
        if self.config.max_speakers is not None:
            self.max_speakers_input.setText(str(self.config.max_speakers))
        diar_row.addWidget(self.max_speakers_input)
        diar_row.addStretch(1)

        self.drop_label = DropLabel()
        self.drop_label.file_dropped.connect(self.set_media_path)

        actions = QHBoxLayout()
        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_transcription)
        self.force_stop_btn = QPushButton("Force Stop")
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.clicked.connect(self.force_stop_transcription)
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_transcript)
        self.open_btn = QPushButton("Open Folder")
        self.open_btn.clicked.connect(self.open_transcriptions_folder)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self.copy_transcript)
        actions.addWidget(self.transcribe_btn)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.force_stop_btn)
        actions.addWidget(self.save_btn)
        actions.addWidget(self.open_btn)
        actions.addWidget(self.copy_btn)

        self.status_label = QLabel("Ready")
        self.hf_token_status = QLabel(self._hf_token_status_text())
        self.hf_token_status.setObjectName("tokenLabel")
        self.hw_metrics_label = QLabel("CPU: -- | RAM: -- | GPU: -- | VRAM: --")
        self.hw_metrics_label.setObjectName("metricsLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.diar_progress_bar = QProgressBar()
        self.diar_progress_bar.setRange(0, 100)
        self.diar_progress_bar.setValue(0)
        self.diar_progress_bar.setFormat("Diarization %p%")
        self._set_bar_color(self.progress_bar, "#dc2626")
        self._set_bar_color(self.diar_progress_bar, "#dc2626")
        self.transcription_time_label = QLabel("Transcription time: --")
        self.transcription_time_label.setObjectName("metricsLabel")
        self.diar_time_label = QLabel("Diarization time: --")
        self.diar_time_label.setObjectName("metricsLabel")

        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Transcript appears here...")

        layout.addLayout(top_row)
        layout.addLayout(model_row)
        layout.addLayout(diar_row)
        layout.addWidget(self.drop_label)
        layout.addLayout(actions)
        layout.addWidget(self.status_label)
        layout.addWidget(self.hf_token_status)
        layout.addWidget(self.hw_metrics_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.transcription_time_label)
        layout.addWidget(self.diar_progress_bar)
        layout.addWidget(self.diar_time_label)
        layout.addWidget(self.text_area, 1)
        self.setCentralWidget(root)

    def _build_menus(self):
        tools_menu = self.menuBar().addMenu("&Tools")
        help_menu = self.menuBar().addMenu("&Help")

        hf_action = QAction("HF Token...", self)
        hf_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        hf_action.triggered.connect(self.configure_hf_token)
        tools_menu.addAction(hf_action)

        benchmark_action = QAction("Benchmark...", self)
        benchmark_action.setShortcut(QKeySequence("Ctrl+B"))
        benchmark_action.triggered.connect(self.open_benchmark_dialog)
        tools_menu.addAction(benchmark_action)

        app_help_action = QAction("PyScribe Help", self)
        app_help_action.setShortcut(QKeySequence.HelpContents)
        app_help_action.triggered.connect(self.show_app_help)
        help_menu.addAction(app_help_action)

        model_help_action = QAction("Model Help", self)
        model_help_action.triggered.connect(self.show_model_help)
        help_menu.addAction(model_help_action)

        logs_action = QAction("Open Logs Folder", self)
        logs_action.triggered.connect(self.open_logs_folder)
        help_menu.addAction(logs_action)

        about_action = QAction("About PyScribe", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def _apply_theme(self):
        self.setFont(QFont("Noto Sans", 10))
        self.setStyleSheet(
            """
            QWidget {
                background: #f5f7fa;
                color: #0f172a;
            }
            #pathLabel {
                background: #ffffff;
                border: 1px solid #d0d7e2;
                border-radius: 8px;
                padding: 8px;
            }
            #hint {
                color: #334155;
            }
            #metricsLabel {
                color: #475569;
                font-size: 11px;
            }
            #tokenLabel {
                color: #0f766e;
                font-size: 11px;
            }
            #dropZone {
                border: 2px dashed #2563eb;
                border-radius: 12px;
                background: #eef4ff;
                color: #1d4ed8;
                font-weight: 600;
            }
            #dropZone[activeDrop="true"] {
                border-color: #ea580c;
                background: #fff7ed;
                color: #c2410c;
            }
            QPushButton {
                background: #0f766e;
                color: white;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #94a3b8;
            }
            QPushButton#exitButton {
                background: #dc2626;
            }
            QPushButton#exitButton:hover {
                background: #b91c1c;
            }
            QCheckBox {
                color: #0f172a;
                font-weight: 600;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #334155;
                border-radius: 4px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #0f766e;
                background: #0f766e;
                image: url(:/qt-project.org/styles/commonstyle/images/checkbox_checked.png);
            }
            QTextEdit {
                background: #ffffff;
                border: 1px solid #d0d7e2;
                border-radius: 8px;
                padding: 8px;
            }
            QProgressBar {
                border: 1px solid #d0d7e2;
                border-radius: 7px;
                background: #ffffff;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #ea580c;
                border-radius: 6px;
            }
            """
        )

    def set_media_path(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Missing file", "The selected file does not exist.")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_MEDIA_EXTS:
            QMessageBox.warning(
                self,
                "Unsupported file",
                "Please select an audio/video media file.\n\n"
                f"Unsupported extension: {ext or '(none)'}",
            )
            return
        self.media_path = path
        self.last_open_dir = os.path.dirname(path) or self.last_open_dir
        self.path_label.setText(path)
        self.status_label.setText(f"Selected: {os.path.basename(path)}")
        self._save_config()

    @Slot()
    def on_browse(self):
        start_dir = self.last_open_dir if os.path.isdir(self.last_open_dir) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select Media File", start_dir, AUDIO_VIDEO_FILTER)
        if path:
            self.set_media_path(path)

    @Slot()
    def start_transcription(self):
        if not self.media_path:
            QMessageBox.warning(self, "No file", "Please choose or drop a media file first.")
            return

        model_name = self.model_combo.currentText().strip()
        model_name = normalize_model_name(model_name)
        if not model_name:
            QMessageBox.warning(self, "No model", "Please select a model.")
            return
        self.model_combo.setCurrentText(model_name)
        LOGGER.info("Qt start transcription model=%s media=%s", model_name, self.media_path)

        if not self._confirm_model_download(model_name):
            return

        forced_language = self._resolve_language_choice(model_name)
        if forced_language == "__cancel__":
            self._hide_download_progress_dialog()
            return

        self.diarization_warning = None
        self.progress_bar.setValue(0)
        self.diar_progress_bar.setValue(0)
        self.transcription_time_label.setText("Transcription time: --")
        self.diar_time_label.setText("Diarization time: --")
        self.status_label.setText("Starting...")
        self.transcribe_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.force_stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.start_hw_monitor()

        max_speakers_text = self.max_speakers_input.text().strip()
        max_speakers = int(max_speakers_text) if max_speakers_text.isdigit() else None
        use_diarization = self.diar_checkbox.isChecked()
        self._current_use_diarization = use_diarization
        diar_backend = self.diar_backend_combo.currentData() if use_diarization else "off"
        self._save_config(
            last_model=model_name,
            use_diarization=use_diarization,
            max_speakers=max_speakers,
            diar_backend=diar_backend,
        )

        self.worker_thread = QThread()
        self.worker = TranscriptionWorker(
            self.media_path,
            model_name,
            use_diarization=use_diarization,
            diar_backend=diar_backend,
            max_speakers=max_speakers,
            language=forced_language,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.status.connect(self._on_status_update)
        self.worker.transcript.connect(self._on_transcript_update)
        self.worker.progress.connect(self._on_transcription_progress)
        self.worker.model_download_progress.connect(self._on_model_download_progress)
        self.worker.diar_progress.connect(self._on_diar_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    @Slot()
    def cancel_transcription(self):
        if self.worker is not None:
            self.worker.request_cancel()
        self.status_label.setText("Cancelling... waiting for current stage to yield.")
        self.cancel_btn.setEnabled(False)

    @Slot()
    def force_stop_transcription(self):
        if self.worker is not None:
            self.worker.request_force_stop()
        self.status_label.setText("Force stop requested...")
        self.cancel_btn.setEnabled(False)
        self.force_stop_btn.setEnabled(False)

    @Slot(str)
    def _on_transcript_update(self, text: str):
        self.transcript_text = text
        self.text_area.setPlainText(text)

    @Slot(str)
    def _on_status_update(self, text: str):
        self.status_label.setText(text)
        if text.startswith("Diarization unavailable"):
            self.diarization_warning = text
        # Pyannote diarization can be a long blocking stage; show busy indicator instead of a stuck 25%.
        if "Running diarization" in text and self.diar_progress_bar.maximum() != 0:
            self.diar_progress_bar.setRange(0, 0)
        elif "Assigning speakers" in text and self.diar_progress_bar.maximum() == 0:
            self.diar_progress_bar.setRange(0, 100)
            self.diar_progress_bar.setValue(65)
            self._set_bar_color(self.diar_progress_bar, self._progress_color(65))

    @Slot(bool, str)
    def _on_worker_finished(
        self,
        cancelled: bool,
        transcript: str,
        transcription_seconds: float,
        diarization_seconds: float,
    ):
        LOGGER.info(
            "Qt worker finished cancelled=%s transcript_len=%s transcribe_seconds=%.2f diar_seconds=%.2f",
            cancelled,
            len(transcript or ""),
            transcription_seconds,
            diarization_seconds,
        )
        self.transcript_text = transcript
        self.text_area.setPlainText(transcript)
        self.transcribe_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.force_stop_btn.setEnabled(False)
        self.stop_hw_monitor()
        self.progress_bar.setRange(0, 100)
        self.diar_progress_bar.setRange(0, 100)
        if not cancelled:
            self.progress_bar.setValue(100)
            self._set_bar_color(self.progress_bar, self._progress_color(100))
        done = "Cancelled." if cancelled else "Transcription complete."
        self.status_label.setText(done)
        self.transcription_time_label.setText(f"Transcription time: {self._format_seconds(transcription_seconds)}")
        if self._current_use_diarization:
            self.diar_time_label.setText(f"Diarization time: {self._format_seconds(diarization_seconds)}")
        else:
            self.diar_time_label.setText("Diarization time: n/a (disabled)")
        if transcript:
            self.save_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
        if self.diarization_warning:
            QMessageBox.warning(self, "Diarization", self.diarization_warning)
        self._hide_download_progress_dialog()
        self._update_diar_ui_state(self.diar_checkbox.isChecked())
        self._cleanup_worker()

    @Slot(str)
    def _on_worker_failed(self, error_msg: str):
        LOGGER.error("Qt worker failed: %s", error_msg)
        self.transcribe_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.force_stop_btn.setEnabled(False)
        self.stop_hw_monitor()
        self.progress_bar.setRange(0, 100)
        self.diar_progress_bar.setRange(0, 100)
        self.status_label.setText("Error")
        self.transcription_time_label.setText("Transcription time: --")
        self.diar_time_label.setText("Diarization time: --")
        QMessageBox.critical(self, "Transcription error", error_msg)
        self._hide_download_progress_dialog()
        self._update_diar_ui_state(self.diar_checkbox.isChecked())
        self._cleanup_worker()

    @Slot(int)
    def _on_transcription_progress(self, value: int):
        self.progress_bar.setValue(value)
        self._set_bar_color(self.progress_bar, self._progress_color(value))

    @Slot(int)
    def _on_diar_progress(self, value: int):
        if self.diar_progress_bar.maximum() == 0 and value < 100:
            # Keep busy state if backend does not emit granular values.
            return
        if self.diar_progress_bar.maximum() == 0:
            self.diar_progress_bar.setRange(0, 100)
        self.diar_progress_bar.setValue(value)
        self._set_bar_color(self.diar_progress_bar, self._progress_color(value))

    @Slot(bool)
    def _update_diar_ui_state(self, enabled: bool):
        self.diar_backend_combo.setEnabled(enabled)
        self.max_speakers_input.setEnabled(enabled)
        self.diar_progress_bar.setEnabled(enabled)
        if not enabled:
            self.diar_progress_bar.setRange(0, 100)
            self.diar_progress_bar.setValue(0)
            self.diar_progress_bar.setFormat("Diarization disabled")
            self._set_bar_color(self.diar_progress_bar, "#94a3b8")
        else:
            self.diar_progress_bar.setFormat("Diarization %p%")
            self._set_bar_color(self.diar_progress_bar, self._progress_color(self.diar_progress_bar.value()))

    @Slot(bool)
    def _update_diar_toggle_label(self, enabled: bool):
        self.diar_checkbox.setText("Speaker Identification is On" if enabled else "Speaker Identification is Off")

    @Slot(int)
    def _on_model_download_progress(self, value: int):
        if self.download_progress_dialog is not None:
            self.download_progress_dialog.setValue(max(0, min(100, value)))
            if value >= 100:
                self._hide_download_progress_dialog()

    def _show_download_progress_dialog(self, model_name: str):
        if self.download_progress_dialog is not None:
            self.download_progress_dialog.close()
        dlg = QProgressDialog(f"Downloading model '{model_name}'...", None, 0, 100, self)
        dlg.setWindowTitle("Model Download")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        self.download_progress_dialog = dlg

    def _hide_download_progress_dialog(self):
        if self.download_progress_dialog is None:
            return
        self.download_progress_dialog.close()
        self.download_progress_dialog = None

    def _cleanup_worker(self):
        if not self.worker_thread:
            return
        LOGGER.info(
            "Qt cleanup worker begin thread_running=%s",
            self.worker_thread.isRunning(),
        )
        if QThread.currentThread() == self.worker_thread:
            self.worker_thread.quit()
            LOGGER.info("Qt cleanup worker called on worker thread; deferred quit.")
            return
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            if not self.worker_thread.wait(3000):
                LOGGER.warning("Qt cleanup worker timed out; requesting force stop.")
                if self.worker is not None:
                    self.worker.request_force_stop()
                self.worker_thread.quit()
                self.worker_thread.wait(3000)
        self.worker_thread = None
        self.worker = None
        LOGGER.info("Qt cleanup worker end")

    def closeEvent(self, event):  # noqa: N802
        if self.worker_thread and self.worker_thread.isRunning():
            LOGGER.warning("Qt close requested while transcription is running.")
            if self.worker is not None:
                self.worker.request_cancel()
            QMessageBox.information(
                self,
                "Transcription running",
                "A job is still running. Please wait for the current stage to finish, then close.",
            )
            event.ignore()
            return
        LOGGER.info("Qt close accepted")
        self.stop_hw_monitor()
        event.accept()

    def _hf_token_status_text(self) -> str:
        return "HF token: configured" if get_hf_token() else "HF token: not configured"

    def _confirm_model_download(self, model_name: str) -> bool:
        if is_model_cached(model_name):
            return True

        repo_id = resolve_repo_id(model_name) or model_name
        est_size = estimate_model_download_size_bytes(model_name)
        size_text = format_bytes(est_size)
        msg = (
            f"Model '{repo_id}' is not cached locally.\n\n"
            f"Estimated download size: {size_text}\n\n"
            "Note: this is a best-effort estimate and may differ from the final transfer size.\n\n"
            "Download now?"
        )
        answer = QMessageBox.question(
            self,
            "Download model",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            self.status_label.setText("Download canceled by user.")
            return False
        self._show_download_progress_dialog(repo_id)
        return True

    @staticmethod
    def _progress_color(value: int) -> str:
        if value >= 100:
            return "#16a34a"  # green
        if value >= 76:
            return "#2563eb"  # blue
        if value >= 51:
            return "#facc15"  # yellow
        if value >= 26:
            return "#f97316"  # orange
        return "#dc2626"  # red

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if seconds <= 0:
            return "0.0s"
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins = int(seconds // 60)
        rem = seconds - (mins * 60)
        return f"{mins}m {rem:.1f}s"

    @staticmethod
    def _set_bar_color(bar: QProgressBar, color: str):
        bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid #d0d7e2;
                border-radius: 7px;
                background: #ffffff;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 6px;
            }}
            """
        )

    @Slot()
    def configure_hf_token(self):
        text, ok = QInputDialog.getText(
            self,
            "Hugging Face Token",
            "Paste your HF access token (used for gated diarization models):",
            QLineEdit.Password,
            "",
        )
        if not ok:
            return
        token = (text or "").strip()
        if not token:
            QMessageBox.information(self, "HF token", "No token entered.")
            return
        try:
            save_hf_token(token)
            self.hf_token_status.setText(self._hf_token_status_text())
            QMessageBox.information(
                self,
                "HF token saved",
                "Token saved. If diarization is still gated, accept terms on the model page once.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "HF token error", str(exc))

    @Slot()
    def show_model_help(self):
        QMessageBox.information(
            self,
            "Custom Model Help",
            (
                "You can select built-in models or use custom Hugging Face repos.\n\n"
                "Recommended format:\n"
                "- owner/repo (e.g. deepdml/faster-whisper-large-v3-turbo-ct2)\n"
                "- Full HF URL also works (auto-converted to owner/repo)\n\n"
                "If model is not cached, app shows a best-effort size estimate and asks before download.\n"
                "For private/gated repos, configure HF token and accept model terms on Hugging Face."
            ),
        )

    @Slot()
    def show_app_help(self):
        help_text = self._load_help_markdown()
        dlg = QDialog(self)
        dlg.setWindowTitle("PyScribe Help")
        dlg.resize(760, 560)

        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setMarkdown(help_text)
        layout.addWidget(text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec()

    @Slot()
    def show_about_dialog(self):
        repo_url = "https://github.com/readydave/PyScribe"
        msg = QMessageBox(self)
        msg.setWindowTitle("About PyScribe")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "PyScribe<br><br>"
            "Local transcription app with Qt desktop and Gradio listener modes.<br><br>"
            f'Repository: <a href="{repo_url}">{repo_url}</a>'
        )
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    @Slot()
    def open_logs_folder(self):
        try:
            log_dir = os.path.dirname(str(get_log_path()))
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            open_folder(log_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Open logs folder failed", str(exc))

    @staticmethod
    def _load_help_markdown() -> str:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "qt_help.md")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return (
                "# PyScribe Help\n\n"
                "Documentation file was not found.\n\n"
                "Expected path: docs/qt_help.md\n"
            )

    def _resolve_language_choice(self, model_name: str) -> str | None:
        """
        Detect language and let the user confirm/override before transcription.
        """
        if not self.media_path:
            return None
        self.status_label.setText("Detecting language...")
        QApplication.processEvents()
        try:
            audio_np = load_audio_waveform(self.media_path)
            lang_code, lang_prob = detect_language(audio_np, device=self.runtime.device)
        except Exception as exc:
            msg = str(exc)
            if "Output file does not contain any stream" in msg or "does not contain any stream" in msg:
                msg = (
                    "The selected file appears to have no audio stream.\n"
                    "Please choose an audio/video file that contains audio."
                )
            QMessageBox.warning(self, "Language detection failed", f"Continuing with auto-detect.\n\n{msg}")
            return None

        if ".en" in model_name.lower() and lang_code != "en":
            msg = (
                f"The detected language is '{lang_code}', but '{model_name}' is English-only.\n\n"
                "Force transcription in English anyway?"
            )
            force_en = QMessageBox.question(
                self,
                "Language mismatch",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if force_en == QMessageBox.Yes:
                return "en"
            return "__cancel__"

        if lang_code != "en":
            msg = (
                f"Detected language: '{lang_code}' (confidence {lang_prob * 100:.1f}%).\n\n"
                "Yes = use detected language.\nNo = force US English."
            )
            use_detected = QMessageBox.question(
                self,
                "Confirm language",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            return lang_code if use_detected == QMessageBox.Yes else "en"

        return lang_code

    def start_hw_monitor(self):
        self.monitoring_active = True
        if self.metrics_thread and self.metrics_thread.is_alive():
            return
        self.metrics_thread = threading.Thread(target=self._hw_monitor_worker, daemon=True)
        self.metrics_thread.start()

    def stop_hw_monitor(self):
        self.monitoring_active = False
        self.hw_metrics.emit("CPU: -- | RAM: -- | GPU: -- | VRAM: --")

    def _hw_monitor_worker(self):
        import psutil

        pynvml = None
        gpu_handle = None
        if self.runtime.device == "cuda":
            try:
                import pynvml as _pynvml  # type: ignore

                pynvml = _pynvml
                pynvml.nvmlInit()
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                gpu_handle = None

        while self.monitoring_active:
            try:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                text = f"CPU: {cpu:.1f}% | RAM: {ram:.1f}%"
                if gpu_handle and pynvml:
                    gpu = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    used = mem.used / (1024**3)
                    total = mem.total / (1024**3)
                    text += f" | GPU: {gpu}% | VRAM: {used:.1f}/{total:.1f} GB"
                self.hw_metrics.emit(text)
                time.sleep(1)
            except Exception:
                break

        if gpu_handle and pynvml:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    @Slot()
    def save_transcript(self):
        if not self.transcript_text:
            return
        stem = "transcript"
        if self.media_path:
            stem = os.path.splitext(os.path.basename(self.media_path))[0]
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        suggested = f"{stem}_{ts}.txt"
        default_dir = os.path.dirname(self.media_path) if self.media_path else self.last_save_dir
        if not default_dir or not os.path.isdir(default_dir):
            default_dir = self.last_open_dir if os.path.isdir(self.last_open_dir) else os.path.expanduser("~")
        suggested_path = os.path.join(default_dir, suggested)
        path, _ = QFileDialog.getSaveFileName(self, "Save Transcript", suggested_path, "Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.transcript_text)
            self.last_save_dir = os.path.dirname(path) or self.last_save_dir
            self._save_config()
            self.status_label.setText(f"Saved: {os.path.basename(path)}")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    @Slot()
    def copy_transcript(self):
        if not self.transcript_text:
            return
        QApplication.clipboard().setText(self.transcript_text)
        self.status_label.setText("Transcript copied.")

    @Slot()
    def open_transcriptions_folder(self):
        if self.media_path:
            folder = os.path.dirname(self.media_path)
        elif self.last_open_dir and os.path.isdir(self.last_open_dir):
            folder = self.last_open_dir
        else:
            folder = os.path.expanduser("~")
        try:
            open_folder(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Open folder failed", str(exc))

    @Slot()
    def open_benchmark_dialog(self):
        dlg = BenchmarkDialog(parent=self, runtime=self.runtime)
        dlg.exec()

    def _save_config(
        self,
        *,
        last_model: str | object = _UNSET,
        use_diarization: bool | object = _UNSET,
        max_speakers: int | None | object = _UNSET,
        diar_backend: str | object = _UNSET,
    ):
        try:
            if last_model is not _UNSET:
                self.config.last_model = last_model
            if use_diarization is not _UNSET:
                self.config.use_diarization = use_diarization
            if max_speakers is not _UNSET:
                self.config.max_speakers = max_speakers
            if diar_backend is not _UNSET:
                self.config.diar_backend = diar_backend
            self.config.last_open_dir = self.last_open_dir if os.path.isdir(self.last_open_dir) else self.config.last_open_dir
            self.config.last_save_dir = self.last_save_dir if self.last_save_dir and os.path.isdir(self.last_save_dir) else self.config.last_save_dir
            save_config(self.config)
        except Exception:
            pass


def run_qt_app():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    LOGGER.info("Qt app event loop starting")
    app.exec()
    LOGGER.info("Qt app event loop exited")
