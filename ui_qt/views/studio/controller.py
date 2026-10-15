"""Studio controller: project loading, edit saving, speaker names, export, AI outputs and waveform peaks."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QWidget

from services import export_service
from services.config_service import AppConfig
from services.platform_service import open_folder
from services.project_store import Project, ProjectStore
from services.waveform_service import WaveformError, compute_peaks, preview_from
from ui_qt import keyring_worker
from ui_qt.llm_postprocess_dialog import LLMPostprocessDialog
from ui_qt.views.studio.studio_view import StudioView

LOGGER = logging.getLogger(__name__)

SAVE_DELAY_MS = 600


class StudioController(QObject):
    project_changed = Signal()
    status = Signal(str)

    def __init__(
        self,
        view: StudioView,
        store: ProjectStore,
        window: QWidget,
        *,
        get_config: Callable[[], AppConfig],
        ensure_llm_profiles: Callable[[], bool],
        is_busy: Callable[[], bool],
        get_save_dir: Callable[[], str],
        set_save_dir: Callable[[str], None],
        save_config: Callable[[], None],
    ) -> None:
        super().__init__(window)
        self._view = view
        self._store = store
        self._window = window
        self._get_config = get_config
        self._ensure_llm_profiles = ensure_llm_profiles
        self._is_busy = is_busy
        self._get_save_dir = get_save_dir
        self._set_save_dir = set_save_dir
        self._save_config = save_config
        self._peaks_handle: keyring_worker.TaskHandle | None = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self.flush)

        view.view.segment_model.text_edited.connect(lambda *_: self._save_timer.start())
        view.speaker_renamed.connect(self._rename_speaker)
        view.export_requested.connect(self.export)
        view.rename_requested.connect(self.rename_project)
        view.reveal_requested.connect(self.reveal)
        view.copy_requested.connect(self.copy_transcript)
        view.generate_requested.connect(self.generate)

    # --- loading ---------------------------------------------------------------------------------

    @property
    def project_id(self) -> str | None:
        return self._view.project_id

    def _project(self) -> Project | None:
        return self._store.load(self._view.project_id) if self._view.project_id else None

    def open_project(self, project_id: str) -> bool:
        self.flush()
        self._cancel_peaks()
        project = self._store.load(project_id)
        if project is None:
            return False
        peaks = self._store.load_peaks(project_id)
        self._view.load_project(project, peaks)
        if not peaks:
            self._compute_peaks(project)
        return True

    def close_project(self) -> None:
        self.flush()
        self._cancel_peaks()
        self._view.clear()

    def forget(self, project_id: str) -> None:
        """A project was deleted elsewhere; drop it from the Studio if it is the one open."""
        if self.project_id == project_id:
            self._save_timer.stop()
            self._cancel_peaks()
            self._view.clear()

    # --- waveform --------------------------------------------------------------------------------

    def _compute_peaks(self, project: Project) -> None:
        if not project.media_path or not os.path.isfile(project.media_path):
            return
        project_id, media = project.id, project.media_path
        self._peaks_handle = keyring_worker.start_task(
            lambda: compute_peaks(media),
            on_done=lambda peaks: self._on_peaks(project_id, peaks),
            on_error=lambda _message: LOGGER.info("Could not draw the waveform for %s.", project_id),
        )

    def _on_peaks(self, project_id: str, peaks: list[float]) -> None:
        self._peaks_handle = None
        if not isinstance(peaks, list) or not peaks:
            return
        try:
            self._store.set_peaks(project_id, peaks, preview_from(peaks))
        except (OSError, ValueError, WaveformError):
            LOGGER.warning("Could not cache the waveform.", exc_info=True)
        if self.project_id == project_id:
            self._view.set_peaks(peaks)
        self.project_changed.emit()

    def _cancel_peaks(self) -> None:
        if self._peaks_handle is not None:
            self._peaks_handle.cancel()
            self._peaks_handle = None

    # --- edits -----------------------------------------------------------------------------------

    def flush(self) -> None:
        """Write any pending transcript edits now."""
        self._save_timer.stop()
        project_id = self._view.project_id
        if project_id is None or not self._pending_edit():
            return
        try:
            self._store.update_segments(project_id, self._view.view.segment_model.segments())
        except (OSError, ValueError):
            LOGGER.warning("Could not save transcript edits.", exc_info=True)
            self.status.emit("Could not save your edits.")
            return
        self.project_changed.emit()

    def _pending_edit(self) -> bool:
        project = self._project()
        return project is not None and project.segments != self._view.view.segment_model.segments()

    def _rename_speaker(self, label: str, name: str) -> None:
        project_id = self._view.project_id
        if project_id is None:
            return
        try:
            project = self._store.rename_speaker(project_id, label, name)
        except (OSError, ValueError):
            LOGGER.warning("Could not save the speaker name.", exc_info=True)
            return
        if project is not None:
            self._view.refresh_speakers(project)
            self.project_changed.emit()

    def rename_project(self) -> None:
        project = self._project()
        if project is None:
            return
        title, accepted = QInputDialog.getText(self._window, "Rename project", "Title:", text=project.title)
        if not accepted or not title.strip():
            return
        try:
            self._store.rename(project.id, title)
        except (OSError, ValueError):
            LOGGER.warning("Could not rename the project.", exc_info=True)
            return
        self._view.set_title(title.strip())
        self.project_changed.emit()

    # --- export / copy / reveal ------------------------------------------------------------------

    def export(self, fmt: str) -> None:
        self.flush()
        project = self._project()
        if project is None or not project.segments or fmt not in export_service.FORMAT_BY_KEY:
            return
        start = os.path.join(self._get_save_dir() or os.path.expanduser("~"), export_service.suggested_filename(project, fmt))
        label = export_service.FORMAT_BY_KEY[fmt].label
        path, _ = QFileDialog.getSaveFileName(self._window, "Export transcript", start, label)
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != export_service.FORMAT_BY_KEY[fmt].extension:
            target = target.with_name(target.name + export_service.FORMAT_BY_KEY[fmt].extension)
        try:
            export_service.export_to_path(project, fmt, target)
        except OSError as exc:
            QMessageBox.warning(self._window, "Export failed", f"Could not write the file:\n{exc.strerror or exc}")
            return
        self._set_save_dir(str(target.parent))
        self.status.emit(f"Exported {target.name}")

    def copy_transcript(self) -> None:
        self.flush()
        project = self._project()
        if project is not None and project.segments:
            QApplication.clipboard().setText(export_service.export_txt(project))
            self.status.emit("Transcript copied.")

    def reveal(self) -> None:
        project_id = self._view.project_id
        if project_id is None:
            return
        try:
            open_folder(str(self._store.project_dir(project_id)))
        except (OSError, ValueError):
            LOGGER.warning("Could not open the project folder.", exc_info=True)

    # --- AI outputs ------------------------------------------------------------------------------

    def generate(self, template_id: str) -> None:
        """Open the existing LLM dialog on this transcript and keep what it generates with the project."""
        self.flush()
        project = self._project()
        if project is None:
            return
        text = export_service.transcript_for_llm(project).strip()
        if not text:
            QMessageBox.information(self._window, "No transcript", "This project has no transcript text to summarise.")
            return
        if not self._ensure_llm_profiles():
            return
        dialog = LLMPostprocessDialog(
            config=self._get_config(),
            current_transcript_text=text,
            current_ocr_text=project.visual_report,
            is_transcription_running=self._is_busy,
            initial_template_id=template_id,
            parent=self._window,
        )
        dialog.exec()
        self._save_config()
        output = dialog.output_text()
        if not output:
            return
        try:
            updated = self._store.set_ai_output(
                project.id, dialog.output_template_id() or template_id, output, dialog.output_profile_name()
            )
        except (OSError, ValueError):
            LOGGER.warning("Could not save the AI output.", exc_info=True)
            return
        if updated is not None and self.project_id == project.id:
            self._view.refresh_outputs(updated)
            self.project_changed.emit()
