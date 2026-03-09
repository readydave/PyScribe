"""PySide6 desktop frontend for PyScribe."""

from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from _thread import LockType

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QFont, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QSizePolicy,
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QMenu,
    QFileDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPlainTextEdit,
    QPushButton,
    QProgressDialog,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from services import (
    AppConfig,
    TranscriptionResult,
    RuntimeInfo,
    check_ocr_backend_ready,
    detect_runtime,
    detect_language,
    get_available_diarization_backends,
    get_backend_label,
    get_enabled_llm_profiles,
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
from ui_qt.llm_connection_dialog import LLMConnectionsDialog
from ui_qt.llm_postprocess_dialog import LLMPostprocessDialog
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


class DropLabel(QFrame):
    file_dropped: Signal = Signal(str)
    browse_requested: Signal = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        self.setMinimumHeight(140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)
        layout.addStretch(1)

        self.title_label = QLabel("Drop audio/video file here")
        self.title_label.setObjectName("dropTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont(self.font())
        title_font.setPointSize(17)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("or click to browse your files")
        self.subtitle_label.setObjectName("dropSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont(self.font())
        subtitle_font.setPointSize(10)
        self.subtitle_label.setFont(subtitle_font)
        layout.addWidget(self.subtitle_label)

        self.browse_btn = QPushButton("Browse Files")
        self.browse_btn.setObjectName("dropBrowseButton")
        self.browse_btn.clicked.connect(self.browse_requested.emit)
        layout.addWidget(self.browse_btn, alignment=Qt.AlignCenter)
        layout.addStretch(1)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
            self.setProperty("activeDrop", True)
            self.style().polish(self)
            return
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # noqa: N802
        self.setProperty("activeDrop", False)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
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
    run_mode: str,
    device: str,
    compute_type: str,
    language: str | None,
    use_diarization: bool,
    diar_backend: str,
    max_speakers: int | None,
    use_visual_analysis: bool,
    visual_profile: str,
    visual_ocr_backend: str,
    visual_sample_seconds: float,
    event_queue: object,
    cancel_event: object,
) -> None:
    configure_logging()

    def _emit(kind: str, value: object) -> None:
        event_queue.put({"type": kind, "value": value})

    try:
        def _run_with(device_name: str, compute_name: str) -> TranscriptionResult:
            return transcribe_media_file(
                media_path=media_path,
                model_name=model_name,
                run_mode=run_mode,
                device=device_name,
                compute_type=compute_name,
                language=language,
                cancel_event=cancel_event,
                use_diarization=use_diarization,
                diar_backend=diar_backend,
                max_speakers=max_speakers,
                use_visual_analysis=use_visual_analysis,
                visual_profile=visual_profile,
                visual_ocr_backend=visual_ocr_backend,
                visual_sample_seconds=visual_sample_seconds,
                on_status=lambda msg: _emit("status", msg),
                on_text=lambda text: _emit("transcript", text),
                on_progress=lambda p: _emit("progress", max(0, min(100, int(p)))),
                on_diar_progress=lambda p: _emit("diar_progress", max(0, min(100, int(p)))),
                on_visual_progress=(
                    (lambda p: _emit("progress", max(0, min(100, int(p)))))
                    if run_mode == "visual_only"
                    else None
                ),
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
                "transcript_only": result.transcript_only,
                "visual_report": result.visual_report,
                "transcription_seconds": result.transcription_seconds,
                "diarization_seconds": result.diarization_seconds,
                "visual_analysis_seconds": result.visual_analysis_seconds,
            },
        )
    except Exception as exc:
        _emit("error", str(exc))


class TranscriptionWorker(QObject):
    status: Signal = Signal(str)
    transcript: Signal = Signal(str)
    progress: Signal = Signal(int)
    model_download_progress: Signal = Signal(int)
    diar_progress: Signal = Signal(int)
    finished: Signal = Signal(bool, str, str, str, float, float, float)
    failed: Signal = Signal(str)

    def __init__(
        self,
        media_path: str,
        model_name: str,
        run_mode: str,
        use_diarization: bool,
        diar_backend: str,
        max_speakers: int | None,
        use_visual_analysis: bool,
        visual_profile: str,
        visual_ocr_backend: str,
        visual_sample_seconds: float,
        language: str | None,
    ) -> None:
        super().__init__()
        self.media_path: str = media_path
        self.model_name: str = model_name
        self.run_mode: str = run_mode
        self.use_diarization: bool = use_diarization
        self.diar_backend: str = diar_backend
        self.max_speakers: int | None = max_speakers
        self.use_visual_analysis: bool = use_visual_analysis
        self.visual_profile: str = visual_profile
        self.visual_ocr_backend: str = visual_ocr_backend
        self.visual_sample_seconds: float = visual_sample_seconds
        self.language: str | None = language
        self.runtime: RuntimeInfo = detect_runtime()
        self._lock: LockType = threading.Lock()
        self._cancel_requested: bool = False
        self._force_stop_requested: bool = False
        self._mp_cancel_event: object = None
        self._process: mp.Process | None = None

    @Slot()
    def request_cancel(self) -> None:
        LOGGER.info("Qt worker: cancel requested")
        with self._lock:
            self._cancel_requested = True
            if self._mp_cancel_event is not None:
                self._mp_cancel_event.set()

    @Slot()
    def request_force_stop(self) -> None:
        LOGGER.warning("Qt worker: force-stop requested")
        with self._lock:
            self._force_stop_requested = True
            proc = self._process
        if proc is not None and proc.is_alive():
            proc.terminate()

    @Slot()
    def run(self) -> None:
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
                self.run_mode,
                self.runtime.device,
                self.runtime.compute_type,
                self.language,
                self.use_diarization,
                self.diar_backend,
                self.max_speakers,
                self.use_visual_analysis,
                self.visual_profile,
                self.visual_ocr_backend,
                self.visual_sample_seconds,
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
                            str(value.get("transcript_only", latest_transcript)),
                            str(value.get("visual_report", "")),
                            float(value.get("transcription_seconds", 0.0)),
                            float(value.get("diarization_seconds", 0.0)),
                            float(value.get("visual_analysis_seconds", 0.0)),
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
                        str(value.get("transcript_only", latest_transcript)),
                        str(value.get("visual_report", "")),
                        float(value.get("transcription_seconds", 0.0)),
                        float(value.get("diarization_seconds", 0.0)),
                        float(value.get("visual_analysis_seconds", 0.0)),
                    )
                elif etype == "error" and not terminal_emitted:
                    terminal_emitted = True
                    LOGGER.error("Qt worker: late error event from child: %s", value)
                    self.failed.emit(str(value))

            if force_stopped and not terminal_emitted:
                LOGGER.warning("Qt worker: force-stopped before terminal event")
                self.finished.emit(True, latest_transcript, latest_transcript, "", 0.0, 0.0, 0.0)
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


class DiarBackendProbeWorker(QObject):
    finished: Signal = Signal(object, str)

    @Slot()
    def run(self) -> None:
        try:
            backends = get_available_diarization_backends(include_off=False) or ["accurate"]
            self.finished.emit(list(backends), "")
        except Exception as exc:
            self.finished.emit([], str(exc))


class MainWindow(QMainWindow):
    hw_metrics: Signal = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._window_title_base = "PyScribe Qt"
        self.setWindowTitle(self._window_title_base)
        self.setMinimumSize(840, 560)
        self.resize(1100, 760)

        self.runtime: RuntimeInfo = detect_runtime()
        self.config: AppConfig = load_config()
        self.media_path: str | None = None
        self.last_open_dir: str = self.config.last_open_dir or os.path.expanduser("~")
        self.last_save_dir: str | None = self.config.last_save_dir
        self.transcript_text: str = ""
        self.transcript_only_text: str = ""
        self.visual_report_text: str = ""
        self.worker_thread: QThread | None = None
        self.worker: TranscriptionWorker | None = None
        self.monitoring_active: bool = False
        self.metrics_thread: threading.Thread | None = None
        self.download_progress_dialog: QProgressDialog | None = None
        self.diarization_warning: str | None = None
        self._diar_backends: list[str] = self._default_diar_backends()
        self._diar_backends_resolved: bool = False
        self._diar_probe_status_before: str = ""
        self._diar_probe_thread: QThread | None = None
        self._diar_probe_worker: DiarBackendProbeWorker | None = None
        self.theme_mode: str = self._sanitize_theme_mode(getattr(self.config, "theme_mode", "system"))
        self._current_run_mode: str = "full"
        self._current_use_diarization: bool = bool(self.config.use_diarization)
        self._current_use_visual_analysis: bool = bool(self.config.use_visual_analysis)
        self._confirmed_visual_backend_downloads: set[str] = set(
            str(b).strip().lower() for b in (self.config.confirmed_visual_backends or [])
        )
        self._sidebar_collapsed: bool = False
        self._status_panel_hidden: bool = False

        self._build_ui()
        self._build_menus()
        self._fit_to_available_screen()
        self._apply_theme()
        self._set_bar_color(self.progress_bar, "#dc2626")
        self._set_bar_color(self.diar_progress_bar, "#dc2626")
        self.hw_metrics.connect(self.hw_metrics_label.setText)
        self._update_diar_ui_state(self.diar_checkbox.isChecked())
        self._update_visual_ui_state(self.visual_checkbox.isChecked())
        self._update_service_visibility()
        QTimer.singleShot(0, self._apply_responsive_layout_state)
        LOGGER.info("Qt MainWindow initialized runtime=%s compute=%s", self.runtime.device, self.runtime.compute_type)

    def _fit_to_available_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        target_w = min(max(840, int(available.width() * 0.84)), available.width())
        target_h = min(max(560, int(available.height() * 0.82)), available.height())
        self.resize(target_w, target_h)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("Sidebar")
        self.sidebar_frame.setFixedWidth(230)
        sidebar_layout = QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(14, 14, 14, 14)
        sidebar_layout.setSpacing(10)

        brand_row = QHBoxLayout()
        self.sidebar_brand_label = QLabel("PyScribe")
        self.sidebar_brand_label.setObjectName("SidebarBrand")
        brand_font = QFont(self.font())
        brand_font.setPointSize(14)
        brand_font.setBold(True)
        self.sidebar_brand_label.setFont(brand_font)
        self.sidebar_toggle_btn = QToolButton()
        self.sidebar_toggle_btn.setObjectName("sidebarToggleButton")
        self.sidebar_toggle_btn.setText("◀")
        self.sidebar_toggle_btn.setToolTip("Hide left panel")
        self.sidebar_toggle_btn.clicked.connect(self._toggle_sidebar_collapsed)
        brand_row.addWidget(self.sidebar_brand_label)
        brand_row.addStretch(1)
        brand_row.addWidget(self.sidebar_toggle_btn)
        sidebar_layout.addLayout(brand_row)

        self.new_project_btn = QPushButton("+  New Project")
        self.new_project_btn.clicked.connect(self._switch_to_transcription_view)
        sidebar_layout.addWidget(self.new_project_btn)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("SidebarNav")
        for name in ("Transcription", "LLM", "Settings"):
            item = QListWidgetItem(name)
            self.nav_list.addItem(item)
        self.nav_list.currentRowChanged.connect(self._on_nav_item_changed)
        sidebar_layout.addWidget(self.nav_list, 1)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setObjectName("exitButton")
        self.exit_btn.clicked.connect(self.close)
        sidebar_layout.addWidget(self.exit_btn)

        root_layout.addWidget(self.sidebar_frame)

        self.main_stack = QStackedWidget()
        self.main_stack.setObjectName("MainStack")
        self.main_stack.addWidget(self._build_transcription_view())
        self.main_stack.addWidget(self._build_llm_workspace_view())
        self.main_stack.addWidget(self._build_settings_view())
        root_layout.addWidget(self.main_stack, 1)

        self.nav_list.setCurrentRow(0)
        self.setCentralWidget(root)

    def _build_transcription_view(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(14, 14, 14, 14)
        page_layout.setSpacing(8)

        main_surface = QFrame()
        main_surface.setObjectName("MainSurface")
        main_layout = QVBoxLayout(main_surface)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        title_row = QHBoxLayout()
        title_col = QVBoxLayout()

        title = QLabel("New Transcription")
        title.setObjectName("PageTitle")
        title_font = QFont(self.font())
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        subtitle = QLabel("Configure settings, drop media, and process.")
        subtitle.setObjectName("PageSubtitle")
        subtitle_font = QFont(self.font())
        subtitle_font.setPointSize(10)
        subtitle.setFont(subtitle_font)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        title_row.addLayout(title_col, 1)

        self.status_toggle_btn = QToolButton()
        self.status_toggle_btn.setObjectName("statusToggleButton")
        self.status_toggle_btn.setText("▶")
        self.status_toggle_btn.setToolTip("Hide right panel")
        self.status_toggle_btn.clicked.connect(self._toggle_status_panel_visibility)
        title_row.addWidget(self.status_toggle_btn)
        main_layout.addLayout(title_row)

        self.path_label = QLabel("No file selected")
        self.path_label.setObjectName("pathLabel")
        main_layout.addWidget(self.path_label)

        drop_card = QFrame()
        drop_card.setObjectName("Card")
        drop_layout = QVBoxLayout(drop_card)
        drop_layout.setContentsMargins(12, 12, 12, 12)
        self.drop_label = DropLabel()
        self.drop_label.file_dropped.connect(self.set_media_path)
        self.drop_label.browse_requested.connect(self.on_browse)
        drop_layout.addWidget(self.drop_label)
        main_layout.addWidget(drop_card)

        settings_grid = QGridLayout()
        settings_grid.setHorizontalSpacing(12)
        settings_grid.setVerticalSpacing(12)
        self.settings_grid = settings_grid

        general_card = QFrame()
        general_card.setObjectName("Card")
        self.general_settings_card = general_card
        general_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        general_layout = QVBoxLayout(general_card)
        general_layout.setContentsMargins(12, 12, 12, 12)
        general_layout.setSpacing(8)
        general_layout.addWidget(QLabel("General Settings"))
        general_layout.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        all_models = get_model_choices()
        self.model_combo.addItems(all_models)
        recommended = recommend_model(self.runtime)
        initial_model = self.config.last_model if self.config.last_model in all_models else recommended
        idx = self.model_combo.findText(initial_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        general_layout.addWidget(self.model_combo)
        self.model_hint_label = QLabel(f"Recommended: {recommended} ({self.runtime.device.upper()})")
        self.model_hint_label.setObjectName("hint")
        general_layout.addWidget(self.model_hint_label)

        self.transcribe_checkbox = QCheckBox("Transcribe audio")
        start_mode = str(self.config.run_mode or "full").strip().lower()
        self.transcribe_checkbox.setChecked(start_mode != "visual_only")
        self.transcribe_checkbox.toggled.connect(self._update_diar_ui_state)
        self.transcribe_checkbox.toggled.connect(self._update_service_visibility)
        general_layout.addWidget(self.transcribe_checkbox)
        general_layout.addStretch(1)

        advanced_card = QFrame()
        advanced_card.setObjectName("Card")
        self.advanced_options_card = advanced_card
        advanced_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        advanced_layout = QVBoxLayout(advanced_card)
        advanced_layout.setContentsMargins(12, 12, 12, 12)
        advanced_layout.setSpacing(8)
        advanced_layout.addWidget(QLabel("Advanced Options"))

        self.diar_checkbox = QCheckBox("Identify Speakers")
        self.diar_checkbox.setChecked(bool(self.config.use_diarization))
        self._update_diar_toggle_label(self.diar_checkbox.isChecked())
        self.diar_checkbox.toggled.connect(self._update_diar_toggle_label)
        self.diar_checkbox.toggled.connect(self._update_diar_ui_state)
        self.diar_checkbox.toggled.connect(self._update_service_visibility)
        advanced_layout.addWidget(self.diar_checkbox)

        diar_grid = QGridLayout()
        diar_grid.setHorizontalSpacing(8)
        diar_grid.setVerticalSpacing(6)
        diar_grid.addWidget(QLabel("Mode"), 0, 0)
        self.diar_backend_combo = QComboBox()
        self._populate_diar_backend_combo(self._diar_backends, preferred=self.config.diar_backend)
        diar_grid.addWidget(self.diar_backend_combo, 0, 1)
        diar_grid.addWidget(QLabel("Max Speakers"), 1, 0)
        self.max_speakers_input = QLineEdit()
        self.max_speakers_input.setPlaceholderText("auto")
        if self.config.max_speakers is not None:
            self.max_speakers_input.setText(str(self.config.max_speakers))
        diar_grid.addWidget(self.max_speakers_input, 1, 1)
        advanced_layout.addLayout(diar_grid)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        advanced_layout.addWidget(divider)

        self.visual_checkbox = QCheckBox("Analyze visuals (slides/chat OCR, beta)")
        self.visual_checkbox.setChecked(bool(self.config.use_visual_analysis))
        self.visual_checkbox.toggled.connect(self._update_visual_ui_state)
        self.visual_checkbox.toggled.connect(self._update_service_visibility)
        advanced_layout.addWidget(self.visual_checkbox)

        visual_grid = QGridLayout()
        visual_grid.setHorizontalSpacing(8)
        visual_grid.setVerticalSpacing(6)
        visual_grid.addWidget(QLabel("Mode"), 0, 0)
        self.visual_profile_combo = QComboBox()
        self.visual_profile_combo.addItems(["fast", "balanced", "accurate"])
        profile_idx = self.visual_profile_combo.findText(str(self.config.visual_profile or "balanced").lower())
        if profile_idx < 0:
            profile_idx = 1
        self.visual_profile_combo.setCurrentIndex(profile_idx)
        visual_grid.addWidget(self.visual_profile_combo, 0, 1)
        visual_grid.addWidget(QLabel("OCR Backend"), 1, 0)
        self.visual_backend_combo = QComboBox()
        self.visual_backend_combo.addItems(["auto", "rapidocr", "paddleocr", "surya", "pytesseract"])
        ocr_idx = self.visual_backend_combo.findText(str(self.config.visual_ocr_backend or "auto").lower())
        if ocr_idx < 0:
            ocr_idx = 0
        self.visual_backend_combo.setCurrentIndex(ocr_idx)
        visual_grid.addWidget(self.visual_backend_combo, 1, 1)
        visual_grid.addWidget(QLabel("Sample every (sec)"), 2, 0)
        self.visual_interval_input = QLineEdit()
        self.visual_interval_input.setText(f"{float(self.config.visual_sample_seconds or 1.0):.1f}")
        visual_grid.addWidget(self.visual_interval_input, 2, 1)
        advanced_layout.addLayout(visual_grid)
        advanced_layout.addStretch(1)

        self._update_transcription_card_columns()
        main_layout.addLayout(settings_grid)

        actions = QHBoxLayout()
        self.transcribe_btn = QPushButton("Process File")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_transcription)
        self.force_stop_btn = QPushButton("Force Stop")
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.clicked.connect(self.force_stop_transcription)
        self.save_btn = QToolButton()
        self.save_btn.setText("Save")
        self.save_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.save_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.save_menu = QMenu(self)
        self.save_all_action = self.save_menu.addAction("Save All (Transcript + OCR)")
        self.save_transcript_action = self.save_menu.addAction("Save Transcript Only")
        self.save_ocr_action = self.save_menu.addAction("Save OCR Only")
        self.save_all_action.triggered.connect(lambda: self.save_output("all"))
        self.save_transcript_action.triggered.connect(lambda: self.save_output("transcript"))
        self.save_ocr_action.triggered.connect(lambda: self.save_output("ocr"))
        self.save_btn.setMenu(self.save_menu)
        self.save_btn.clicked.connect(lambda: self.save_output("all"))
        self.save_btn.setEnabled(False)
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
        actions.addStretch(1)
        main_layout.addLayout(actions)

        progress_card = QFrame()
        progress_card.setObjectName("Card")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(12, 12, 12, 12)
        progress_layout.setSpacing(8)
        progress_layout.addWidget(QLabel("Transcription Progress"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(26)
        self.progress_bar.setFormat("Transcription %p%")
        progress_layout.addWidget(self.progress_bar)
        self.diar_progress_bar = QProgressBar()
        self.diar_progress_bar.setRange(0, 100)
        self.diar_progress_bar.setValue(0)
        self.diar_progress_bar.setFormat("Diarization %p%")
        progress_layout.addWidget(self.diar_progress_bar)

        self.terminal_log = QPlainTextEdit()
        self.terminal_log.setObjectName("TerminalLog")
        self.terminal_log.setReadOnly(True)
        self.terminal_log.setPlaceholderText("Live pipeline events...")
        self.terminal_log.setMinimumHeight(96)
        progress_layout.addWidget(self.terminal_log)
        main_layout.addWidget(progress_card)

        transcript_card = QFrame()
        transcript_card.setObjectName("Card")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setContentsMargins(12, 12, 12, 12)
        transcript_layout.setSpacing(8)
        transcript_layout.addWidget(QLabel("Transcript Output"))
        self.text_area = QPlainTextEdit()
        self.text_area.setPlaceholderText("Transcript appears here...")
        self.text_area.setMinimumHeight(110)
        self.text_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        transcript_layout.addWidget(self.text_area, 1)
        main_layout.addWidget(transcript_card, 1)

        status_panel = QFrame()
        status_panel.setObjectName("StatusPanel")
        status_panel.setMinimumWidth(220)
        status_panel.setMaximumWidth(420)
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(10)
        status_layout.addWidget(QLabel("Status"))
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        self.hf_token_status = QLabel(self._hf_token_status_text())
        self.hf_token_status.setObjectName("tokenLabel")
        status_layout.addWidget(self.hf_token_status)
        self.hw_metrics_label = QLabel("CPU: -- | RAM: -- | GPU: -- | VRAM: --")
        self.hw_metrics_label.setObjectName("metricsLabel")
        status_layout.addWidget(self.hw_metrics_label)
        metrics_card = QFrame()
        metrics_card.setObjectName("Card")
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        metrics_layout.setSpacing(6)
        self.transcription_time_label = QLabel("Transcription time: --")
        self.transcription_time_label.setObjectName("metricsLabel")
        self.diar_time_label = QLabel("Diarization time: --")
        self.diar_time_label.setObjectName("metricsLabel")
        self.visual_time_label = QLabel("Visual analysis time: --")
        self.visual_time_label.setObjectName("metricsLabel")
        metrics_layout.addWidget(self.transcription_time_label)
        metrics_layout.addWidget(self.diar_time_label)
        metrics_layout.addWidget(self.visual_time_label)
        status_layout.addWidget(metrics_card)
        status_layout.addStretch(1)

        self.status_panel = status_panel
        self.transcription_scroll = QScrollArea()
        self.transcription_scroll.setWidgetResizable(True)
        self.transcription_scroll.setFrameShape(QFrame.NoFrame)
        self.transcription_scroll.setWidget(main_surface)

        self.transcription_splitter = QSplitter(Qt.Horizontal)
        self.transcription_splitter.setChildrenCollapsible(False)
        self.transcription_splitter.addWidget(self.transcription_scroll)
        self.transcription_splitter.addWidget(status_panel)
        self.transcription_splitter.setStretchFactor(0, 1)
        self.transcription_splitter.setStretchFactor(1, 0)
        self.transcription_splitter.setSizes([900, 300])
        page_layout.addWidget(self.transcription_splitter, 1)
        return page

    def _build_llm_workspace_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        title = QLabel("LLM Workspace")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Open post-processing and connection management tools.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)
        card_layout.addWidget(QLabel("Post-Processing"))
        open_workspace_btn = QPushButton("Open LLM Post-Process")
        open_workspace_btn.clicked.connect(lambda _checked=False: self.open_llm_postprocess_dialog())
        manage_connections_btn = QPushButton("Open LLM Connections")
        manage_connections_btn.clicked.connect(self.open_llm_connections_dialog)
        card_layout.addWidget(open_workspace_btn)
        card_layout.addWidget(manage_connections_btn)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_settings_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Settings & Connections")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Manage API keys and defaults.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        api_card = QFrame()
        api_card.setObjectName("Card")
        api_layout = QGridLayout(api_card)
        api_layout.setContentsMargins(14, 14, 14, 14)
        api_layout.setHorizontalSpacing(8)
        api_layout.setVerticalSpacing(8)
        api_layout.addWidget(QLabel("API Keys"), 0, 0, 1, 4)

        openai_label = QLabel("OpenAI Key")
        self.openai_key_input = QLineEdit()
        self.openai_key_input.setPlaceholderText("sk-...")
        self.openai_key_input.setEchoMode(QLineEdit.Password)
        openai_show_btn = QPushButton("Show")
        openai_show_btn.clicked.connect(lambda: self._toggle_secret_field_visibility(self.openai_key_input, openai_show_btn))
        openai_test_btn = QPushButton("Test")
        openai_test_btn.clicked.connect(self.open_llm_connections_dialog)
        api_layout.addWidget(openai_label, 1, 0)
        api_layout.addWidget(self.openai_key_input, 1, 1)
        api_layout.addWidget(openai_show_btn, 1, 2)
        api_layout.addWidget(openai_test_btn, 1, 3)

        anthropic_label = QLabel("Anthropic Key")
        self.anthropic_key_input = QLineEdit()
        self.anthropic_key_input.setPlaceholderText("sk-ant-...")
        self.anthropic_key_input.setEchoMode(QLineEdit.Password)
        anthropic_show_btn = QPushButton("Show")
        anthropic_show_btn.clicked.connect(
            lambda: self._toggle_secret_field_visibility(self.anthropic_key_input, anthropic_show_btn)
        )
        anthropic_test_btn = QPushButton("Test")
        anthropic_test_btn.clicked.connect(self.open_llm_connections_dialog)
        api_layout.addWidget(anthropic_label, 2, 0)
        api_layout.addWidget(self.anthropic_key_input, 2, 1)
        api_layout.addWidget(anthropic_show_btn, 2, 2)
        api_layout.addWidget(anthropic_test_btn, 2, 3)

        defaults_card = QFrame()
        defaults_card.setObjectName("Card")
        defaults_layout = QGridLayout(defaults_card)
        defaults_layout.setContentsMargins(14, 14, 14, 14)
        defaults_layout.setHorizontalSpacing(8)
        defaults_layout.setVerticalSpacing(8)
        defaults_layout.addWidget(QLabel("Transcription Defaults"), 0, 0, 1, 2)
        defaults_layout.addWidget(QLabel("Default model"), 1, 0)
        self.default_model_input = QLineEdit(self.config.last_model or "")
        defaults_layout.addWidget(self.default_model_input, 1, 1)
        defaults_layout.addWidget(QLabel("Last open folder"), 2, 0)
        self.default_path_input = QLineEdit(self.last_open_dir or "")
        defaults_layout.addWidget(self.default_path_input, 2, 1)

        content_layout.addWidget(api_card)
        content_layout.addWidget(defaults_card)
        content_layout.addStretch(1)
        return page

    @Slot(int)
    def _on_nav_item_changed(self, index: int) -> None:
        if index < 0:
            return
        self.main_stack.setCurrentIndex(index)

    @Slot()
    def _switch_to_transcription_view(self) -> None:
        self.nav_list.setCurrentRow(0)

    @Slot()
    def _toggle_sidebar_collapsed(self) -> None:
        self._sidebar_collapsed = not self._sidebar_collapsed
        if self._sidebar_collapsed:
            self.sidebar_frame.setFixedWidth(52)
            self.sidebar_brand_label.setText("PS")
            self.new_project_btn.setVisible(False)
            self.nav_list.setVisible(False)
            self.exit_btn.setVisible(False)
            self.sidebar_toggle_btn.setText("▶")
            self.sidebar_toggle_btn.setToolTip("Show left panel")
            self._update_transcription_card_columns()
            return
        self.sidebar_frame.setFixedWidth(230)
        self.sidebar_brand_label.setText("PyScribe")
        self.new_project_btn.setVisible(True)
        self.nav_list.setVisible(True)
        self.exit_btn.setVisible(True)
        self.sidebar_toggle_btn.setText("◀")
        self.sidebar_toggle_btn.setToolTip("Hide left panel")
        self._update_transcription_card_columns()

    @Slot()
    def _toggle_status_panel_visibility(self) -> None:
        self._status_panel_hidden = not self._status_panel_hidden
        if not hasattr(self, "status_panel"):
            return
        self.status_panel.setVisible(not self._status_panel_hidden)
        if hasattr(self, "transcription_splitter"):
            if self._status_panel_hidden:
                self.transcription_splitter.setSizes([1200, 0])
            else:
                self.transcription_splitter.setSizes([900, 300])
        if hasattr(self, "status_toggle_btn"):
            if self._status_panel_hidden:
                self.status_toggle_btn.setText("◀")
                self.status_toggle_btn.setToolTip("Show right panel")
            else:
                self.status_toggle_btn.setText("▶")
                self.status_toggle_btn.setToolTip("Hide right panel")
        self._update_transcription_card_columns()

    def _toggle_secret_field_visibility(self, field: QLineEdit, button: QPushButton) -> None:
        if field.echoMode() == QLineEdit.Password:
            field.setEchoMode(QLineEdit.Normal)
            button.setText("Hide")
            return
        field.setEchoMode(QLineEdit.Password)
        button.setText("Show")

    def _append_terminal_log(self, text: str) -> None:
        if not hasattr(self, "terminal_log"):
            return
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.terminal_log.appendPlainText(f"[{timestamp}] {cleaned}")

    def _update_transcription_card_columns(self) -> None:
        if not hasattr(self, "settings_grid"):
            return
        while self.settings_grid.count():
            self.settings_grid.takeAt(0)

        available_width = self.width()
        if hasattr(self, "transcription_scroll"):
            available_width = int(self.transcription_scroll.viewport().width() or available_width)
        two_columns = available_width >= 900
        if two_columns:
            self.settings_grid.addWidget(self.general_settings_card, 0, 0)
            self.settings_grid.addWidget(self.advanced_options_card, 0, 1)
            self.settings_grid.setColumnStretch(0, 1)
            self.settings_grid.setColumnStretch(1, 1)
            return
        self.settings_grid.addWidget(self.general_settings_card, 0, 0)
        self.settings_grid.addWidget(self.advanced_options_card, 1, 0)
        self.settings_grid.setColumnStretch(0, 1)

    def _apply_responsive_layout_state(self) -> None:
        if hasattr(self, "status_panel") and hasattr(self, "status_toggle_btn"):
            should_hide_status = self.width() < 1080
            if should_hide_status != self._status_panel_hidden:
                self._status_panel_hidden = should_hide_status
                self.status_panel.setVisible(not self._status_panel_hidden)
            if hasattr(self, "transcription_splitter"):
                if self._status_panel_hidden:
                    self.transcription_splitter.setSizes([1200, 0])
                else:
                    self.transcription_splitter.setSizes([900, 300])
            if self._status_panel_hidden:
                self.status_toggle_btn.setText("◀")
                self.status_toggle_btn.setToolTip("Show right panel")
            else:
                self.status_toggle_btn.setText("▶")
                self.status_toggle_btn.setToolTip("Hide right panel")
        self._update_transcription_card_columns()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout_state()

    def _build_menus(self) -> None:
        tools_menu = self.menuBar().addMenu("&Tools")
        view_menu = self.menuBar().addMenu("&View")
        help_menu = self.menuBar().addMenu("&Help")

        hf_action = QAction("HF Token...", self)
        hf_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        hf_action.triggered.connect(self.configure_hf_token)
        tools_menu.addAction(hf_action)

        benchmark_action = QAction("Benchmark...", self)
        benchmark_action.setShortcut(QKeySequence("Ctrl+B"))
        benchmark_action.triggered.connect(self.open_benchmark_dialog)
        tools_menu.addAction(benchmark_action)

        llm_connections_action = QAction("LLM Connections...", self)
        llm_connections_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
        llm_connections_action.triggered.connect(self.open_llm_connections_dialog)
        tools_menu.addAction(llm_connections_action)

        llm_postprocess_action = QAction("LLM Post-Process...", self)
        llm_postprocess_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
        llm_postprocess_action.triggered.connect(lambda _checked=False: self.open_llm_postprocess_dialog())
        tools_menu.addAction(llm_postprocess_action)

        process_existing_action = QAction("Process Existing Transcript...", self)
        process_existing_action.triggered.connect(
            lambda _checked=False: self.open_llm_postprocess_dialog(prefer_loaded_transcript=True)
        )
        tools_menu.addAction(process_existing_action)

        theme_menu = view_menu.addMenu("Theme")
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)
        self.theme_actions: dict[str, QAction] = {}
        for mode, label in (("system", "System"), ("light", "Light"), ("dark", "Dark")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, m=mode: self._set_theme_mode(m))
            self.theme_action_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[mode] = action
        self.theme_actions[self.theme_mode].setChecked(True)

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

    def _apply_theme(self) -> None:
        applied = self._effective_theme_mode()
        self.setFont(QFont("Segoe UI", 10))
        if applied == "dark":
            window_bg = "#1e1e1e"
            surface_bg = "#242424"
            sidebar_bg = "#171717"
            card_bg = "#232323"
            border = "#3a3a3a"
            text = "#e6e6e6"
            muted = "#9aa4ad"
            input_bg = "#202020"
            accent = "#00a3a3"
            accent_hover = "#008080"
        else:
            window_bg = "#f7f9fb"
            surface_bg = "#f9fbfc"
            sidebar_bg = "#eef2f4"
            card_bg = "#ffffff"
            border = "#d7dfe5"
            text = "#1f2937"
            muted = "#5f6d7a"
            input_bg = "#ffffff"
            accent = "#008080"
            accent_hover = "#006b6b"

        self.setStyleSheet(
            f"""
            QWidget {{
                background: {window_bg};
                color: {text};
                font-family: "Segoe UI", "Roboto", "Helvetica", sans-serif;
            }}
            QPushButton, QFrame, QLineEdit {{
                border-radius: 8px;
            }}
            #Sidebar {{
                background: {sidebar_bg};
                border-right: 1px solid {border};
                padding: 12px;
            }}
            #SidebarBrand {{
                font-weight: 700;
                padding: 6px 2px;
                color: {text};
            }}
            #SidebarNav {{
                border: 1px solid {border};
                background: {sidebar_bg};
                outline: none;
                padding: 10px;
            }}
            #SidebarNav::item {{
                padding: 10px 12px;
                margin: 3px 0;
                border-radius: 8px;
            }}
            #SidebarNav::item:selected {{
                background: {accent};
                color: white;
                font-weight: 600;
            }}
            #MainStack {{
                background: {surface_bg};
                padding: 12px;
            }}
            #MainSurface, #StatusPanel {{
                background: {surface_bg};
                border: 1px solid {border};
                padding: 12px;
            }}
            #Card {{
                background: {card_bg};
                border: 1px solid {border};
                padding: 12px;
            }}
            #PageTitle {{
                font-weight: 700;
                padding-bottom: 2px;
            }}
            #PageSubtitle {{
                color: {muted};
                padding-bottom: 6px;
            }}
            #pathLabel {{
                background: {input_bg};
                border: 1px solid {border};
                padding: 10px 12px;
            }}
            #dropZone {{
                border: 2px dashed {accent};
                background: {surface_bg};
                padding: 15px;
            }}
            #dropZone[activeDrop="true"] {{
                border-color: {accent_hover};
                background: {card_bg};
            }}
            #dropTitle {{
                color: {accent};
                font-weight: 700;
                background: transparent;
            }}
            #dropSubtitle {{
                color: {muted};
                background: transparent;
            }}
            QPushButton, QToolButton {{
                background: {accent};
                color: white;
                border: 1px solid {accent};
                padding: 10px 14px;
                font-weight: 600;
            }}
            QToolButton#sidebarToggleButton, QToolButton#statusToggleButton {{
                min-width: 24px;
                max-width: 24px;
                min-height: 24px;
                max-height: 24px;
                padding: 2px;
                font-weight: 700;
            }}
            QPushButton:hover, QToolButton:hover {{
                background: {accent_hover};
                border-color: {accent_hover};
            }}
            QPushButton:disabled, QToolButton:disabled {{
                background: #95a5a6;
                border-color: #95a5a6;
                color: #f1f5f9;
            }}
            QPushButton#exitButton {{
                background: #b42318;
                border-color: #b42318;
            }}
            QPushButton#exitButton:hover {{
                background: #991b1b;
                border-color: #991b1b;
            }}
            QLineEdit, QComboBox, QPlainTextEdit, QTextEdit {{
                background: {input_bg};
                border: 1px solid {border};
                padding: 10px 12px;
                selection-background-color: {accent};
            }}
            QGroupBox {{
                background: {card_bg};
                border: 1px solid {border};
                border-radius: 8px;
                margin-top: 8px;
                padding: 12px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: {muted};
            }}
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
                border: 1px solid {accent};
            }}
            #hint, #metricsLabel {{
                color: {muted};
            }}
            #tokenLabel {{
                color: {accent};
                font-weight: 600;
            }}
            QProgressBar {{
                border: 1px solid {border};
                border-radius: 10px;
                background: {input_bg};
                text-align: center;
                min-height: 20px;
            }}
            QProgressBar::chunk {{
                background: {accent};
                border-radius: 8px;
                margin: 1px;
            }}
            QPlainTextEdit#TerminalLog {{
                background: #000000;
                color: #9df2a7;
                border: 1px solid #111111;
                font-family: "Consolas", "Courier New", monospace;
                padding: 10px;
            }}
            """
        )

    @staticmethod
    def _sanitize_theme_mode(value: str) -> str:
        mode = str(value or "system").strip().lower()
        if mode in {"system", "light", "dark"}:
            return mode
        return "system"

    def _effective_theme_mode(self) -> str:
        if self.theme_mode in {"light", "dark"}:
            return self.theme_mode
        lightness = QApplication.palette().color(QPalette.Window).lightness()
        return "dark" if lightness < 128 else "light"

    def _set_theme_mode(self, mode: str) -> None:
        self.theme_mode = self._sanitize_theme_mode(mode)
        if hasattr(self, "theme_actions") and self.theme_mode in self.theme_actions:
            self.theme_actions[self.theme_mode].setChecked(True)
        self._save_config(theme_mode=self.theme_mode)
        self._apply_theme()
        self._set_bar_color(self.progress_bar, self._progress_color(self.progress_bar.value()))
        self._set_bar_color(self.diar_progress_bar, self._progress_color(self.diar_progress_bar.value()))
        self._update_diar_ui_state(self.diar_checkbox.isChecked())

    @staticmethod
    def _default_diar_backends() -> list[str]:
        # Keep startup fast by deferring expensive capability checks until needed.
        return ["accurate", "fast", "sortformer"]

    def _set_window_title_status(self, status: str | None) -> None:
        if status:
            self.setWindowTitle(f"{self._window_title_base} - {status}")
            return
        self.setWindowTitle(self._window_title_base)

    def _populate_diar_backend_combo(self, backends: list[str], *, preferred: str | None = None) -> None:
        unique: list[str] = []
        for key in backends:
            backend = str(key or "").strip().lower()
            if not backend or backend in unique:
                continue
            unique.append(backend)
        if not unique:
            unique = ["accurate"]
        self._diar_backends = unique
        self.diar_backend_combo.clear()
        for key in unique:
            self.diar_backend_combo.addItem(get_backend_label(key), key)
        target = str(preferred or self.config.diar_backend or "").strip().lower()
        if target in unique:
            self.diar_backend_combo.setCurrentIndex(unique.index(target))
            return
        self.diar_backend_combo.setCurrentIndex(0)

    def _diar_probe_running(self) -> bool:
        return bool(self._diar_probe_thread and self._diar_probe_thread.isRunning())

    def _start_diar_backend_probe(self) -> None:
        if self._diar_backends_resolved or self._diar_probe_running():
            return
        self._diar_probe_status_before = self.status_label.text()
        self.status_label.setText("Loading speaker ID backends...")
        self._set_window_title_status("Loading speaker ID backends")
        self.diar_backend_combo.setEnabled(False)

        self._diar_probe_thread = QThread(self)
        self._diar_probe_worker = DiarBackendProbeWorker()
        self._diar_probe_worker.moveToThread(self._diar_probe_thread)
        self._diar_probe_thread.started.connect(self._diar_probe_worker.run)
        self._diar_probe_worker.finished.connect(self._on_diar_backend_probe_finished)
        self._diar_probe_worker.finished.connect(self._diar_probe_thread.quit)
        self._diar_probe_worker.finished.connect(self._diar_probe_worker.deleteLater)
        self._diar_probe_thread.finished.connect(self._on_diar_backend_probe_thread_finished)
        self._diar_probe_thread.finished.connect(self._diar_probe_thread.deleteLater)
        self._diar_probe_thread.start()

    @Slot(object, str)
    def _on_diar_backend_probe_finished(self, backends: object, error_text: str) -> None:
        backend_list = [str(item).strip().lower() for item in (backends if isinstance(backends, list) else [])]
        backend_list = [item for item in backend_list if item]
        current = str(self.diar_backend_combo.currentData() or "").strip().lower() or self.config.diar_backend
        self._populate_diar_backend_combo(backend_list or ["accurate"], preferred=current)
        self._diar_backends_resolved = True
        self._set_window_title_status(None)
        # Avoid clobbering active processing status text while a run is in progress.
        if self.transcribe_btn.isEnabled():
            if error_text:
                LOGGER.warning("Speaker backend probe failed; using fallback backend list. reason=%s", error_text)
                self.status_label.setText("Speaker backend check failed; using fallback backend list.")
            elif self._diar_probe_status_before.strip():
                self.status_label.setText(self._diar_probe_status_before)
            else:
                self.status_label.setText("Ready")
        self._update_diar_ui_state(self.diar_checkbox.isChecked())

    @Slot()
    def _on_diar_backend_probe_thread_finished(self) -> None:
        self._diar_probe_thread = None
        self._diar_probe_worker = None

    def set_media_path(self, path: str) -> None:
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
        self._append_terminal_log(f"Selected file: {os.path.basename(path)}")
        self._save_config()

    @Slot()
    def on_browse(self) -> None:
        start_dir = self.last_open_dir if os.path.isdir(self.last_open_dir) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select Media File", start_dir, AUDIO_VIDEO_FILTER)
        if path:
            self.set_media_path(path)

    @Slot()
    def start_transcription(self) -> None:
        if not self.media_path:
            QMessageBox.warning(self, "No file", "Please choose or drop a media file first.")
            return

        run_mode, allow_transcription, run_diarization, run_visual = self._effective_service_flags()
        if run_mode == "none":
            QMessageBox.warning(self, "No task selected", "Enable at least one processing option (Transcribe audio or Analyze visuals).")
            return

        model_name = self.model_combo.currentText().strip()
        model_name = normalize_model_name(model_name)
        if allow_transcription and not model_name:
            QMessageBox.warning(self, "No model", "Please select a model.")
            return
        if model_name:
            self.model_combo.setCurrentText(model_name)
        LOGGER.info("Qt start transcription mode=%s model=%s media=%s", run_mode, model_name, self.media_path)

        forced_language = None
        if allow_transcription:
            if not self._confirm_model_download(model_name):
                return
            forced_language = self._resolve_language_choice(model_name)
            if forced_language == "__cancel__":
                self._hide_download_progress_dialog()
                return

        self.diarization_warning = None
        self.transcript_only_text = ""
        self.visual_report_text = ""
        self._current_run_mode = run_mode
        self.progress_bar.setValue(0)
        self.diar_progress_bar.setValue(0)
        self.transcription_time_label.setText("Transcription time: --")
        self.diar_time_label.setText("Diarization time: --")
        self.visual_time_label.setText("Visual analysis time: --")
        self.status_label.setText("Starting...")
        self.terminal_log.clear()
        self._append_terminal_log(f"Starting job for: {os.path.basename(self.media_path or '')}")
        self.transcribe_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.force_stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.start_hw_monitor()

        max_speakers_text = self.max_speakers_input.text().strip()
        max_speakers = int(max_speakers_text) if max_speakers_text.isdigit() else None
        use_diarization = run_diarization
        self._current_use_diarization = use_diarization
        diar_backend = "off"
        diar_backend_for_run = "off"
        if use_diarization:
            diar_backend = str(self.diar_backend_combo.currentData() or "").strip().lower() or "accurate"
            diar_backend_for_run = diar_backend
            if self._diar_probe_running():
                # Don't block transcription while lazy backend discovery is still running.
                if diar_backend_for_run == "sortformer":
                    diar_backend_for_run = "accurate"
                self.status_label.setText(
                    f"Speaker backend detection still running; continuing with {get_backend_label(diar_backend_for_run)}."
                )
                self._append_terminal_log(self.status_label.text())
        use_visual_analysis = run_visual
        self._current_use_visual_analysis = use_visual_analysis
        visual_profile = self.visual_profile_combo.currentText().strip().lower() or "balanced"
        visual_ocr_backend = self.visual_backend_combo.currentText().strip().lower() or "auto"
        if use_visual_analysis:
            ready, reason = check_ocr_backend_ready(visual_ocr_backend)
            if not ready:
                fallback = ""
                for candidate in ("rapidocr", "paddleocr", "pytesseract", "surya"):
                    fallback_ready, _ = check_ocr_backend_ready(candidate)
                    if fallback_ready and candidate != visual_ocr_backend:
                        fallback = candidate
                        break
                if fallback:
                    answer = QMessageBox.question(
                        self,
                        "OCR backend unavailable",
                        (
                            f"Selected OCR backend '{visual_ocr_backend}' is not available.\n\n"
                            f"{reason}\n\n"
                            f"Switch to '{fallback}' and continue?"
                        ),
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if answer != QMessageBox.Yes:
                        self.status_label.setText("Visual analysis canceled: OCR backend unavailable.")
                        self.transcribe_btn.setEnabled(True)
                        self.cancel_btn.setEnabled(False)
                        self.force_stop_btn.setEnabled(False)
                        self.stop_hw_monitor()
                        return
                    visual_ocr_backend = fallback
                    fb_idx = self.visual_backend_combo.findText(fallback)
                    if fb_idx >= 0:
                        self.visual_backend_combo.setCurrentIndex(fb_idx)
                else:
                    QMessageBox.warning(
                        self,
                        "OCR backend unavailable",
                        (
                            f"Selected OCR backend '{visual_ocr_backend}' is not available.\n\n"
                            f"{reason}\n\n"
                            "Install dependencies or choose another OCR backend."
                        ),
                    )
                    self.status_label.setText("Visual analysis canceled: OCR backend unavailable.")
                    self.transcribe_btn.setEnabled(True)
                    self.cancel_btn.setEnabled(False)
                    self.force_stop_btn.setEnabled(False)
                    self.stop_hw_monitor()
                    return
        if use_visual_analysis and not self._confirm_visual_backend_download(visual_ocr_backend):
            self.status_label.setText("Visual analysis canceled by user.")
            self.transcribe_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.force_stop_btn.setEnabled(False)
            self.stop_hw_monitor()
            return
        visual_sample_seconds = self._parse_visual_sample_seconds()
        config_updates: dict[str, object] = {
            "run_mode": run_mode,
            "use_diarization": use_diarization,
            "max_speakers": max_speakers,
            "diar_backend": diar_backend,
            "use_visual_analysis": use_visual_analysis,
            "visual_profile": visual_profile,
            "visual_ocr_backend": visual_ocr_backend,
            "visual_sample_seconds": visual_sample_seconds,
        }
        if model_name:
            config_updates["last_model"] = model_name
        self._save_config(**config_updates)

        self.worker_thread = QThread()
        self.worker = TranscriptionWorker(
            self.media_path,
            model_name,
            run_mode=run_mode,
            use_diarization=use_diarization,
            diar_backend=diar_backend_for_run,
            max_speakers=max_speakers,
            use_visual_analysis=use_visual_analysis,
            visual_profile=visual_profile,
            visual_ocr_backend=visual_ocr_backend,
            visual_sample_seconds=visual_sample_seconds,
            language=forced_language,
        )
        self._update_service_visibility()
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
    def cancel_transcription(self) -> None:
        if self.worker is not None:
            self.worker.request_cancel()
        self.status_label.setText("Cancelling... waiting for current stage to yield.")
        self._append_terminal_log("Cancellation requested.")
        self.cancel_btn.setEnabled(False)

    @Slot()
    def force_stop_transcription(self) -> None:
        if self.worker is not None:
            self.worker.request_force_stop()
        self.status_label.setText("Force stop requested...")
        self._append_terminal_log("Force-stop requested.")
        self.cancel_btn.setEnabled(False)
        self.force_stop_btn.setEnabled(False)

    @Slot(str)
    def _on_transcript_update(self, text: str) -> None:
        self.transcript_text = text
        self.transcript_only_text = text
        self.text_area.setPlainText(text)

    @Slot(str)
    def _on_status_update(self, text: str) -> None:
        self.status_label.setText(text)
        self._append_terminal_log(text)
        if text.startswith("Diarization unavailable"):
            self.diarization_warning = text
        # Pyannote diarization can be a long blocking stage; show busy indicator instead of a stuck 25%.
        if "Running diarization" in text and self.diar_progress_bar.maximum() != 0:
            self.diar_progress_bar.setRange(0, 0)
        elif "Assigning speakers" in text and self.diar_progress_bar.maximum() == 0:
            self.diar_progress_bar.setRange(0, 100)
            self.diar_progress_bar.setValue(65)
            self._set_bar_color(self.diar_progress_bar, self._progress_color(65))

    @Slot(bool, str, str, str, float, float, float)
    def _on_worker_finished(
        self,
        cancelled: bool,
        transcript: str,
        transcript_only: str,
        visual_report: str,
        transcription_seconds: float,
        diarization_seconds: float,
        visual_analysis_seconds: float,
    ) -> None:
        LOGGER.info(
            "Qt worker finished cancelled=%s transcript_len=%s transcribe_seconds=%.2f diar_seconds=%.2f",
            cancelled,
            len(transcript or ""),
            transcription_seconds,
            diarization_seconds,
        )
        self.transcript_text = transcript
        self.transcript_only_text = transcript_only
        self.visual_report_text = visual_report
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
        done = "Cancelled."
        if not cancelled:
            if self._current_run_mode == "visual_only":
                done = "Visual analysis complete."
            elif self._current_run_mode == "transcribe_only":
                done = "Transcription complete."
            else:
                done = "Transcription complete."
        visual_unavailable = (
            self._current_use_visual_analysis
            and bool(visual_report)
            and "Unavailable:" in visual_report
        )
        if visual_unavailable and not cancelled:
            if self._current_run_mode == "visual_only":
                done = "Visual analysis unavailable."
            else:
                done = "Transcription complete (visual analysis unavailable)."
        self.status_label.setText(done)
        self._append_terminal_log(done)
        if self._current_run_mode in {"full", "transcribe_only"}:
            self.transcription_time_label.setText(f"Transcription time: {self._format_seconds(transcription_seconds)}")
        else:
            self.transcription_time_label.setText("Transcription time: n/a (visual-only)")
        if self._current_use_diarization and self._current_run_mode in {"full", "transcribe_only"}:
            self.diar_time_label.setText(f"Diarization time: {self._format_seconds(diarization_seconds)}")
        else:
            self.diar_time_label.setText("Diarization time: n/a (disabled)")
        if self._current_use_visual_analysis and self._current_run_mode in {"full", "visual_only"}:
            self.visual_time_label.setText(f"Visual analysis time: {self._format_seconds(visual_analysis_seconds)}")
        else:
            self.visual_time_label.setText("Visual analysis time: n/a (disabled)")
        if transcript or visual_report:
            self.save_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
        if visual_unavailable:
            lines = [line.strip() for line in visual_report.splitlines() if "Unavailable:" in line]
            detail = lines[0] if lines else "Visual analysis backend unavailable."
            QMessageBox.warning(self, "Visual analysis", detail)
        if self.diarization_warning:
            QMessageBox.warning(self, "Diarization", self.diarization_warning)
        self._hide_download_progress_dialog()
        self._update_service_visibility()
        self._cleanup_worker()

    @Slot(str)
    def _on_worker_failed(self, error_msg: str) -> None:
        LOGGER.error("Qt worker failed: %s", error_msg)
        self.visual_report_text = ""
        self.transcribe_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.force_stop_btn.setEnabled(False)
        self.stop_hw_monitor()
        self.progress_bar.setRange(0, 100)
        self.diar_progress_bar.setRange(0, 100)
        self.status_label.setText("Error")
        self._append_terminal_log(f"Error: {error_msg}")
        self.transcription_time_label.setText("Transcription time: --")
        self.diar_time_label.setText("Diarization time: --")
        self.visual_time_label.setText("Visual analysis time: --")
        QMessageBox.critical(self, "Transcription error", error_msg)
        self._hide_download_progress_dialog()
        self._update_service_visibility()
        self._cleanup_worker()

    @Slot(int)
    def _on_transcription_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)
        self._set_bar_color(self.progress_bar, self._progress_color(value))

    @Slot(int)
    def _on_diar_progress(self, value: int) -> None:
        if self.diar_progress_bar.maximum() == 0 and value < 100:
            # Keep busy state if backend does not emit granular values.
            return
        if self.diar_progress_bar.maximum() == 0:
            self.diar_progress_bar.setRange(0, 100)
        self.diar_progress_bar.setValue(value)
        self._set_bar_color(self.diar_progress_bar, self._progress_color(value))

    @Slot(bool)
    def _update_diar_ui_state(self, enabled: bool) -> None:
        del enabled
        _, allow_transcription, _, _ = self._effective_service_flags()
        diar_controls_enabled = bool(allow_transcription)
        if allow_transcription and not self._diar_backends_resolved:
            self._start_diar_backend_probe()
        # Keep selector usable immediately with fallback/default options while
        # lazy backend probing resolves final availability.
        combo_enabled = allow_transcription
        self.diar_backend_combo.setEnabled(combo_enabled)
        self.max_speakers_input.setEnabled(allow_transcription)
        self.diar_progress_bar.setEnabled(allow_transcription)
        if not allow_transcription:
            self.diar_progress_bar.setRange(0, 100)
            self.diar_progress_bar.setValue(0)
            self.diar_progress_bar.setFormat("Diarization disabled")
            self._set_bar_color(self.diar_progress_bar, "#94a3b8")
        else:
            self.diar_progress_bar.setFormat("Diarization %p%")
            self._set_bar_color(self.diar_progress_bar, self._progress_color(self.diar_progress_bar.value()))
        self._update_service_visibility()

    @Slot(bool)
    def _update_visual_ui_state(self, enabled: bool) -> None:
        del enabled
        visual_controls_enabled = True
        self.visual_profile_combo.setEnabled(visual_controls_enabled)
        self.visual_backend_combo.setEnabled(visual_controls_enabled)
        self.visual_interval_input.setEnabled(visual_controls_enabled)
        self._update_service_visibility()

    @Slot(bool)
    def _update_diar_toggle_label(self, enabled: bool) -> None:
        self.diar_checkbox.setText("Speaker Identification is On" if enabled else "Speaker Identification is Off")

    def _effective_service_flags(self) -> tuple[str, bool, bool, bool]:
        allow_transcription = self.transcribe_checkbox.isChecked()
        allow_visual = self.visual_checkbox.isChecked()
        run_diarization = allow_transcription and self.diar_checkbox.isChecked()
        run_visual = allow_visual
        mode = "none"
        if allow_transcription and run_visual:
            mode = "full"
        elif allow_transcription:
            mode = "transcribe_only"
        elif run_visual:
            mode = "visual_only"
        return mode, allow_transcription, run_diarization, run_visual

    @Slot()
    def _update_service_visibility(self) -> None:
        mode, allow_transcription, run_diarization, run_visual = self._effective_service_flags()
        show_main_progress = allow_transcription or run_visual
        self.progress_bar.setVisible(show_main_progress)
        self.transcription_time_label.setVisible(allow_transcription)
        self.diar_progress_bar.setVisible(run_diarization)
        self.diar_time_label.setVisible(run_diarization)
        self.visual_time_label.setVisible(run_visual)

        self.model_combo.setEnabled(allow_transcription)
        self.diar_checkbox.setEnabled(allow_transcription)
        if not allow_transcription:
            self.diar_backend_combo.setEnabled(False)
            self.max_speakers_input.setEnabled(False)
            self.diar_progress_bar.setEnabled(False)
            self.visual_profile_combo.setEnabled(False)
            self.visual_backend_combo.setEnabled(False)
            self.visual_interval_input.setEnabled(False)

        if mode == "visual_only":
            self.progress_bar.setFormat("Visual analysis %p%")
        else:
            self.progress_bar.setFormat("Transcription %p%")

    @Slot(int)
    def _on_model_download_progress(self, value: int) -> None:
        if self.download_progress_dialog is not None:
            self.download_progress_dialog.setValue(max(0, min(100, value)))
            if value >= 100:
                self._hide_download_progress_dialog()

    def _show_download_progress_dialog(self, model_name: str) -> None:
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

    def _hide_download_progress_dialog(self) -> None:
        if self.download_progress_dialog is None:
            return
        self.download_progress_dialog.close()
        self.download_progress_dialog = None

    def _cleanup_worker(self) -> None:
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

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
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
        if self._diar_probe_thread and self._diar_probe_thread.isRunning():
            self.status_label.setText("Waiting for speaker backend initialization to finish...")
            self._diar_probe_thread.quit()
            if not self._diar_probe_thread.wait(15000):
                QMessageBox.information(
                    self,
                    "Initializing backends",
                    "Speaker backend initialization is still running. Please try closing again in a few seconds.",
                )
                event.ignore()
                return
        self._set_window_title_status(None)
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

    def _confirm_visual_backend_download(self, backend: str) -> bool:
        backend = str(backend or "").strip().lower()
        if backend in {"pytesseract", "auto"}:
            return True
        if backend in self._confirmed_visual_backend_downloads:
            return True
        extra = ""
        if backend == "surya":
            extra = (
                "\n\nSurya is experimental in this app and may require a separate environment with newer Torch."
            )
        msg = (
            f"Visual OCR backend '{backend}' may download OCR model files on first run.\n\n"
            "This can take a few minutes depending on connection speed.\n\n"
            f"Continue?{extra}"
        )
        answer = QMessageBox.question(
            self,
            "Visual OCR model download",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self._confirmed_visual_backend_downloads.add(backend)
            self._save_config()
            return True
        return False

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

    def _parse_visual_sample_seconds(self) -> float:
        raw = (self.visual_interval_input.text() or "").strip()
        try:
            value = float(raw)
        except ValueError:
            value = 1.0
        value = min(10.0, max(0.5, value))
        self.visual_interval_input.setText(f"{value:.1f}")
        return value

    @staticmethod
    def _set_bar_color(bar: QProgressBar, color: str) -> None:
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
    def configure_hf_token(self) -> None:
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
    def show_model_help(self) -> None:
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
    def show_app_help(self) -> None:
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
    def show_about_dialog(self) -> None:
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
    def open_logs_folder(self) -> None:
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

    def start_hw_monitor(self) -> None:
        self.monitoring_active = True
        if self.metrics_thread and self.metrics_thread.is_alive():
            return
        self.metrics_thread = threading.Thread(target=self._hw_monitor_worker, daemon=True)
        self.metrics_thread.start()

    def stop_hw_monitor(self) -> None:
        self.monitoring_active = False
        self.hw_metrics.emit("CPU: -- | RAM: -- | GPU: -- | VRAM: --")

    def _hw_monitor_worker(self) -> None:
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

    def _save_payload_for_mode(self, mode: str) -> tuple[str, str] | None:
        transcript_part = (self.transcript_only_text or self.transcript_text or "").strip()
        ocr_part = (self.visual_report_text or "").strip()

        if mode == "transcript":
            if not transcript_part:
                QMessageBox.information(self, "Save", "No transcript text is available to save.")
                return None
            return transcript_part, "transcript"

        if mode == "ocr":
            if not ocr_part:
                QMessageBox.information(
                    self,
                    "Save OCR",
                    "No OCR/visual analysis output is available.\nRun with 'Analyze visuals' enabled.",
                )
                return None
            return ocr_part, "ocr"

        # default: all
        if not transcript_part and not ocr_part:
            QMessageBox.information(self, "Save", "Nothing is available to save.")
            return None
        if transcript_part and ocr_part:
            return f"{transcript_part}\n\n{ocr_part}".strip(), "all"
        if transcript_part:
            return transcript_part, "all"
        return ocr_part, "all"

    def save_output(self, mode: str = "all") -> None:
        payload = self._save_payload_for_mode(mode)
        if payload is None:
            return
        content, suffix = payload
        stem = "transcript"
        if self.media_path:
            stem = os.path.splitext(os.path.basename(self.media_path))[0]
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        suggested = f"{ts}_{stem}_{suffix}.txt"
        default_dir = os.path.dirname(self.media_path) if self.media_path else self.last_save_dir
        if not default_dir or not os.path.isdir(default_dir):
            default_dir = self.last_open_dir if os.path.isdir(self.last_open_dir) else os.path.expanduser("~")
        suggested_path = os.path.join(default_dir, suggested)
        title = "Save Transcript + OCR" if suffix == "all" else ("Save Transcript" if suffix == "transcript" else "Save OCR")
        path, _ = QFileDialog.getSaveFileName(self, title, suggested_path, "Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            self.last_save_dir = os.path.dirname(path) or self.last_save_dir
            self._save_config()
            self.status_label.setText(f"Saved: {os.path.basename(path)}")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    @Slot()
    def copy_transcript(self) -> None:
        if not self.transcript_text:
            return
        QApplication.clipboard().setText(self.transcript_text)
        self.status_label.setText("Transcript copied.")

    @Slot()
    def open_transcriptions_folder(self) -> None:
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
    def open_benchmark_dialog(self) -> None:
        dlg = BenchmarkDialog(parent=self, runtime=self.runtime)
        dlg.exec()

    @Slot()
    def open_llm_connections_dialog(self) -> None:
        dlg = LLMConnectionsDialog(config=self.config, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.config.llm_profiles = dlg.profiles()
        self.config.llm_default_profile = dlg.default_profile()
        self._save_config()
        self.status_label.setText("LLM connection profiles saved.")

    def open_llm_postprocess_dialog(self, prefer_loaded_transcript: bool = False) -> None:
        enabled_profiles = get_enabled_llm_profiles(self.config.llm_profiles)
        if not enabled_profiles:
            answer = QMessageBox.question(
                self,
                "No LLM profiles",
                (
                    "No enabled LLM profiles are configured.\n\n"
                    "Open LLM Connections now?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.open_llm_connections_dialog()
                enabled_profiles = get_enabled_llm_profiles(self.config.llm_profiles)
            if not enabled_profiles:
                self.status_label.setText("LLM post-process canceled: no enabled profiles.")
                return

        current_transcript = ""
        current_ocr = ""
        if not prefer_loaded_transcript:
            current_transcript = (self.transcript_only_text or self.transcript_text or "").strip()
            current_ocr = (self.visual_report_text or "").strip()

        dlg = LLMPostprocessDialog(
            config=self.config,
            current_transcript_text=current_transcript,
            current_ocr_text=current_ocr,
            is_transcription_running=self._is_transcription_running,
            prefer_loaded_transcript=prefer_loaded_transcript,
            parent=self,
        )
        dlg.exec()
        self._save_config()

    def _is_transcription_running(self) -> bool:
        return bool(self.worker_thread and self.worker_thread.isRunning())

    def _save_config(
        self,
        *,
        last_model: str | object = _UNSET,
        run_mode: str | object = _UNSET,
        theme_mode: str | object = _UNSET,
        use_diarization: bool | object = _UNSET,
        max_speakers: int | None | object = _UNSET,
        diar_backend: str | object = _UNSET,
        use_visual_analysis: bool | object = _UNSET,
        visual_profile: str | object = _UNSET,
        visual_ocr_backend: str | object = _UNSET,
        visual_sample_seconds: float | object = _UNSET,
    ) -> None:
        try:
            if last_model is not _UNSET:
                self.config.last_model = last_model
            if run_mode is not _UNSET:
                self.config.run_mode = str(run_mode)
            if theme_mode is not _UNSET:
                self.config.theme_mode = self._sanitize_theme_mode(str(theme_mode))
            if use_diarization is not _UNSET:
                self.config.use_diarization = use_diarization
            if max_speakers is not _UNSET:
                self.config.max_speakers = max_speakers
            if diar_backend is not _UNSET:
                self.config.diar_backend = diar_backend
            if use_visual_analysis is not _UNSET:
                self.config.use_visual_analysis = use_visual_analysis
            if visual_profile is not _UNSET:
                self.config.visual_profile = str(visual_profile)
            if visual_ocr_backend is not _UNSET:
                self.config.visual_ocr_backend = str(visual_ocr_backend)
            if visual_sample_seconds is not _UNSET:
                self.config.visual_sample_seconds = float(visual_sample_seconds)
            self.config.confirmed_visual_backends = sorted(self._confirmed_visual_backend_downloads)
            self.config.last_open_dir = self.last_open_dir if os.path.isdir(self.last_open_dir) else self.config.last_open_dir
            self.config.last_save_dir = self.last_save_dir if self.last_save_dir and os.path.isdir(self.last_save_dir) else self.config.last_save_dir
            save_config(self.config)
        except Exception as exc:
            LOGGER.warning("Qt config save failed: %s", exc, exc_info=True)


def run_qt_app() -> None:
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    LOGGER.info("Qt app event loop starting")
    app.exec()
    LOGGER.info("Qt app event loop exited")
