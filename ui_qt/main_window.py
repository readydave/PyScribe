"""PySide6 desktop frontend for PyScribe."""

from __future__ import annotations

import datetime
import os
import threading
import time

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
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
from ui_qt.benchmark_dialog import BenchmarkDialog
from utils import load_audio_waveform
AUDIO_VIDEO_FILTER = (
    "Media Files (*.m4a *.mp3 *.wav *.flac *.aac *.ogg *.wma *.mp4 *.mov *.mkv *.avi *.flv);;All Files (*.*)"
)


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


class TranscriptionWorker(QObject):
    status = Signal(str)
    transcript = Signal(str)
    progress = Signal(int)
    model_download_progress = Signal(int)
    diar_progress = Signal(int)
    finished = Signal(bool, str)
    failed = Signal(str)

    def __init__(
        self,
        media_path: str,
        model_name: str,
        cancel_event: threading.Event,
        use_diarization: bool,
        diar_backend: str,
        max_speakers: int | None,
        language: str | None,
    ):
        super().__init__()
        self.media_path = media_path
        self.model_name = model_name
        self.cancel_event = cancel_event
        self.use_diarization = use_diarization
        self.diar_backend = diar_backend
        self.max_speakers = max_speakers
        self.language = language
        self.runtime = detect_runtime()

    @Slot()
    def run(self):
        try:
            result = transcribe_media_file(
                media_path=self.media_path,
                model_name=self.model_name,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
                language=self.language,
                cancel_event=self.cancel_event,
                use_diarization=self.use_diarization,
                diar_backend=self.diar_backend,
                max_speakers=self.max_speakers,
                on_status=self.status.emit,
                on_text=self.transcript.emit,
                on_progress=lambda p: self.progress.emit(max(0, min(100, int(p)))),
                on_diar_progress=lambda p: self.diar_progress.emit(max(0, min(100, int(p)))),
                on_model_download_progress=lambda p: self.model_download_progress.emit(max(0, min(100, int(p)))),
            )
            self.finished.emit(result.cancelled, result.transcript)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    hw_metrics = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyScribe Qt")
        self.resize(980, 700)

        self.runtime = detect_runtime()
        self.config = load_config()
        self.media_path: str | None = None
        self.transcript_text = ""
        self.cancel_event = threading.Event()
        self.worker_thread: QThread | None = None
        self.worker: TranscriptionWorker | None = None
        self.monitoring_active = False
        self.metrics_thread: threading.Thread | None = None
        self.download_progress_dialog: QProgressDialog | None = None

        self._build_ui()
        self._apply_theme()
        self._set_bar_color(self.progress_bar, "#dc2626")
        self._set_bar_color(self.diar_progress_bar, "#dc2626")
        self.hw_metrics.connect(self.hw_metrics_label.setText)

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        self.path_label = QLabel("No file selected")
        self.path_label.setObjectName("pathLabel")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.on_browse)
        top_row.addWidget(self.path_label, 1)
        top_row.addWidget(browse_btn)

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
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_transcript)
        self.open_btn = QPushButton("Open Folder")
        self.open_btn.clicked.connect(self.open_transcriptions_folder)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self.copy_transcript)
        self.hf_token_btn = QPushButton("HF Token")
        self.hf_token_btn.clicked.connect(self.configure_hf_token)
        self.model_help_btn = QPushButton("Model Help")
        self.model_help_btn.clicked.connect(self.show_model_help)
        self.benchmark_btn = QPushButton("Benchmark")
        self.benchmark_btn.clicked.connect(self.open_benchmark_dialog)

        actions.addWidget(self.transcribe_btn)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.save_btn)
        actions.addWidget(self.open_btn)
        actions.addWidget(self.copy_btn)
        actions.addWidget(self.hf_token_btn)
        actions.addWidget(self.model_help_btn)
        actions.addWidget(self.benchmark_btn)

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
        layout.addWidget(self.diar_progress_bar)
        layout.addWidget(self.text_area, 1)
        self.setCentralWidget(root)

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
        self.media_path = path
        self.path_label.setText(path)
        self.status_label.setText(f"Selected: {os.path.basename(path)}")

    @Slot()
    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Media File", "", AUDIO_VIDEO_FILTER)
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

        if not self._confirm_model_download(model_name):
            return

        forced_language = self._resolve_language_choice(model_name)
        if forced_language == "__cancel__":
            self._hide_download_progress_dialog()
            return

        self.cancel_event.clear()
        self.progress_bar.setValue(0)
        self.diar_progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.transcribe_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.start_hw_monitor()

        max_speakers_text = self.max_speakers_input.text().strip()
        max_speakers = int(max_speakers_text) if max_speakers_text.isdigit() else None
        use_diarization = self.diar_checkbox.isChecked()
        diar_backend = self.diar_backend_combo.currentData() if use_diarization else "off"
        try:
            save_config(
                AppConfig(
                    last_model=model_name,
                    use_diarization=use_diarization,
                    max_speakers=max_speakers,
                    diar_backend=diar_backend,
                )
            )
        except Exception:
            pass

        self.worker_thread = QThread()
        self.worker = TranscriptionWorker(
            self.media_path,
            model_name,
            self.cancel_event,
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
        self.worker_thread.start()

    @Slot()
    def cancel_transcription(self):
        self.cancel_event.set()
        self.status_label.setText("Cancelling... waiting for current stage to yield.")
        self.cancel_btn.setEnabled(False)

    @Slot(str)
    def _on_transcript_update(self, text: str):
        self.transcript_text = text
        self.text_area.setPlainText(text)

    @Slot(str)
    def _on_status_update(self, text: str):
        self.status_label.setText(text)
        # Pyannote diarization can be a long blocking stage; show busy indicator instead of a stuck 25%.
        if "Running diarization" in text and self.diar_progress_bar.maximum() != 0:
            self.diar_progress_bar.setRange(0, 0)
        elif "Assigning speakers" in text and self.diar_progress_bar.maximum() == 0:
            self.diar_progress_bar.setRange(0, 100)
            self.diar_progress_bar.setValue(65)
            self._set_bar_color(self.diar_progress_bar, self._progress_color(65))

    @Slot(bool, str)
    def _on_worker_finished(self, cancelled: bool, transcript: str):
        self.transcript_text = transcript
        self.text_area.setPlainText(transcript)
        self.transcribe_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.stop_hw_monitor()
        self.progress_bar.setRange(0, 100)
        self.diar_progress_bar.setRange(0, 100)
        done = "Cancelled." if cancelled else "Transcription complete."
        self.status_label.setText(done)
        if transcript:
            self.save_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
        self._hide_download_progress_dialog()
        self._cleanup_worker()

    @Slot(str)
    def _on_worker_failed(self, error_msg: str):
        self.transcribe_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.stop_hw_monitor()
        self.progress_bar.setRange(0, 100)
        self.diar_progress_bar.setRange(0, 100)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Transcription error", error_msg)
        self._hide_download_progress_dialog()
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
        if QThread.currentThread() == self.worker_thread:
            self.worker_thread.quit()
            return
        self.worker_thread.quit()
        self.worker_thread.wait(2000)
        self.worker_thread = None
        self.worker = None

    def closeEvent(self, event):  # noqa: N802
        if self.worker_thread and self.worker_thread.isRunning():
            self.cancel_event.set()
            QMessageBox.information(
                self,
                "Transcription running",
                "A job is still running. Please wait for the current stage to finish, then close.",
            )
            event.ignore()
            return
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
            QMessageBox.warning(self, "Language detection failed", f"Continuing with auto-detect.\n\n{exc}")
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
        path, _ = QFileDialog.getSaveFileName(self, "Save Transcript", suggested, "Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.transcript_text)
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
        folder = os.path.dirname(self.media_path) if self.media_path else os.path.expanduser("~")
        try:
            open_folder(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Open folder failed", str(exc))

    @Slot()
    def open_benchmark_dialog(self):
        dlg = BenchmarkDialog(parent=self, runtime=self.runtime)
        dlg.exec()


def run_qt_app():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
