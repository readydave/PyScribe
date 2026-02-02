"""Benchmark dialog for Qt frontend."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import logging

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
)

from services import get_model_choices, load_model
from utils import convert_to_16k_mono, get_ffmpeg_cmd, load_audio_waveform
LOGGER = logging.getLogger(__name__)


class BenchmarkWorker(QObject):
    status = Signal(str)
    progress = Signal(int)
    result_line = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, selected_models: list[str], runtime, cancel_event: threading.Event, audio_language: str):
        super().__init__()
        self.selected_models = selected_models
        self.runtime = runtime
        self.cancel_event = cancel_event
        self.audio_language = audio_language

    @Slot()
    def run(self):
        temp_dir = tempfile.TemporaryDirectory()
        try:
            benchmark_files = {
                "en": os.path.join("assets", "benchmark-sherlock-holmes-en.mp3"),
                "es": os.path.join("assets", "benchmark-napoleon-es.mp3"),
            }
            audio_path = benchmark_files.get(self.audio_language)
            if not audio_path or not os.path.exists(audio_path):
                raise RuntimeError(f"Benchmark audio file missing for '{self.audio_language}'.")

            ffmpeg_cmd = get_ffmpeg_cmd(tool="ffmpeg")
            if not ffmpeg_cmd:
                raise RuntimeError("ffmpeg not found.")

            wav_path = convert_to_16k_mono(audio_path, temp_dir.name, ffmpeg_cmd)
            audio_np = load_audio_waveform(wav_path)
            for idx, model_name in enumerate(self.selected_models):
                if self.cancel_event.is_set():
                    self.result_line.emit("Benchmark cancelled.")
                    break

                self.status.emit(f"Testing model {idx + 1}/{len(self.selected_models)}: {model_name}")
                self.progress.emit(int((idx / max(len(self.selected_models), 1)) * 100))
                start = time.time()
                model = load_model(
                    model_name,
                    device=self.runtime.device,
                    compute_type=self.runtime.compute_type,
                    use_cache=False,
                )
                segments, _ = model.transcribe(audio_np, language=self.audio_language, beam_size=5)
                _ = [s.text for s in segments]
                elapsed = time.time() - start
                self.result_line.emit(f"- {model_name}: {elapsed:.2f} seconds")

            self.progress.emit(100)
            self.status.emit("Benchmark complete.")
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


class BenchmarkDialog(QDialog):
    def __init__(self, parent, runtime):
        super().__init__(parent)
        self.runtime = runtime
        self.cancel_event = threading.Event()
        self.worker_thread: QThread | None = None
        self.worker: BenchmarkWorker | None = None
        self.model_checks: dict[str, QCheckBox] = {}
        self.audio_language = "en"

        self.setWindowTitle("PyScribe Benchmark")
        self.resize(700, 560)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Audio language:"))
        self.en_btn = QPushButton("English")
        self.es_btn = QPushButton("Spanish")
        self.en_btn.clicked.connect(lambda: self._set_language("en"))
        self.es_btn.clicked.connect(lambda: self._set_language("es"))
        lang_row.addWidget(self.en_btn)
        lang_row.addWidget(self.es_btn)
        lang_row.addStretch(1)

        model_label = QLabel("Select models")
        model_box = QVBoxLayout()
        for model_name in get_model_choices():
            cb = QCheckBox(model_name)
            self.model_checks[model_name] = cb
            model_box.addWidget(cb)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start Benchmark")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_benchmark)
        self.cancel_btn.clicked.connect(self.cancel_benchmark)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.cancel_btn)
        controls.addStretch(1)

        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)

        layout.addLayout(lang_row)
        layout.addWidget(model_label)
        layout.addLayout(model_box)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.results_area, 1)

    def _set_language(self, lang: str):
        self.audio_language = lang
        self.status_label.setText(f"Language set: {lang}")

    @Slot()
    def start_benchmark(self):
        selected = [name for name, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "No models", "Select at least one model.")
            return

        self.cancel_event.clear()
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.results_area.clear()

        self.worker_thread = QThread()
        self.worker = BenchmarkWorker(selected, self.runtime, self.cancel_event, self.audio_language)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status_label.setText)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.result_line.connect(self._append_result_line)
        self.worker.finished.connect(self._finish)
        self.worker.failed.connect(self._fail)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.failed.connect(self._cleanup_worker)
        self.worker_thread.start()
        LOGGER.info("Benchmark started models=%s lang=%s", selected, self.audio_language)

    @Slot()
    def cancel_benchmark(self):
        self.cancel_event.set()
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Cancelling...")
        LOGGER.info("Benchmark cancel requested")

    @Slot(str)
    def _append_result_line(self, line: str):
        self.results_area.append(line)

    @Slot()
    def _finish(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Benchmark complete.")

    @Slot(str)
    def _fail(self, error_msg: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Benchmark failed.")
        QMessageBox.critical(self, "Benchmark error", error_msg)

    def _cleanup_worker(self):
        if not self.worker_thread:
            return
        self.worker_thread.quit()
        self.worker_thread.wait(3000)
        self.worker_thread = None
        self.worker = None
        LOGGER.info("Benchmark worker cleaned up")

    def closeEvent(self, event):  # noqa: N802
        if self.worker_thread and self.worker_thread.isRunning():
            LOGGER.warning("Benchmark dialog close requested while running; cancelling worker.")
            self.cancel_event.set()
            self.worker_thread.quit()
            if not self.worker_thread.wait(3000):
                QMessageBox.information(
                    self,
                    "Benchmark running",
                    "Benchmark is still shutting down. Please try closing again in a few seconds.",
                )
                event.ignore()
                return
            self.worker_thread = None
            self.worker = None
        event.accept()
