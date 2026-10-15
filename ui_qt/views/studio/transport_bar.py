"""Playback controls under the transcript: play/pause, skip, speed, clock and the waveform."""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui_qt.components import AppButton, IconButton
from ui_qt.views.studio.transcript_view import format_clock
from ui_qt.views.studio.waveform_widget import WaveformWidget

LOGGER = logging.getLogger(__name__)

SKIP_MS = 10_000
SPEEDS = (("0.75×", 0.75), ("1×", 1.0), ("1.25×", 1.25), ("1.5×", 1.5), ("2×", 2.0))


class TransportBar(QWidget):
    position_changed = Signal(int)
    playing_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TransportBar")
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._loaded = False
        self._duration_hint_ms = 0
        self._pending_seek_ms: int | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 8, 16, 8)
        outer.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.play_button = AppButton("▶", "round")
        self.play_button.setFixedSize(40, 40)
        self.play_button.setAccessibleName("Play")
        self.play_button.clicked.connect(self.toggle)
        self.back_button = IconButton("↺", "Back 10 seconds")
        self.back_button.clicked.connect(lambda: self.seek_ms(self.position_ms() - SKIP_MS))
        self.forward_button = IconButton("↻", "Forward 10 seconds")
        self.forward_button.clicked.connect(lambda: self.seek_ms(self.position_ms() + SKIP_MS))
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setProperty("type", "caption")
        self.time_label.setAccessibleName("Playback position")
        self.waveform = WaveformWidget()
        self.waveform.seek_requested.connect(self.seek_ms)
        self.speed_combo = QComboBox()
        self.speed_combo.setAccessibleName("Playback speed")
        for label, rate in SPEEDS:
            self.speed_combo.addItem(label, rate)
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        for widget in (self.play_button, self.back_button, self.forward_button, self.time_label):
            row.addWidget(widget)
        row.addWidget(self.waveform, 1)
        row.addWidget(self.speed_combo)
        outer.addLayout(row)
        self.notice = QLabel("")
        self.notice.setProperty("type", "caption")
        self.notice.setVisible(False)
        outer.addWidget(self.notice)

        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.errorOccurred.connect(self._on_error)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self.unload("No recording is available for this project.")

    # --- loading ---------------------------------------------------------------------------------

    def load(self, media_path: str | None, duration_hint_ms: int = 0) -> bool:
        """Open ``media_path`` for playback. Returns False (and shows why) when it cannot be played."""
        self.unload()
        self._duration_hint_ms = max(0, int(duration_hint_ms))
        self.waveform.set_duration_ms(self._duration_hint_ms)
        if not media_path or not os.path.isfile(media_path):
            self._set_enabled(False, "The recording for this project is no longer available. The transcript can still be read and exported.")
            return False
        self._loaded = True
        self._player.setSource(QUrl.fromLocalFile(media_path))
        self._set_enabled(True, "")
        self._update_time()
        return True

    def unload(self, message: str = "") -> None:
        """Stop and release the current recording."""
        self._player.stop()
        self._player.setSource(QUrl())
        self._loaded = False
        self._pending_seek_ms = None
        self.waveform.set_position_ms(0)
        self.waveform.set_duration_ms(0)
        self._update_time()
        self._set_enabled(False, message)

    def shutdown(self) -> None:
        self.unload()

    def _set_enabled(self, enabled: bool, message: str) -> None:
        for widget in (self.play_button, self.back_button, self.forward_button, self.speed_combo):
            widget.setEnabled(enabled)
        self.waveform.setEnabled(enabled)
        self.notice.setText(message)
        self.notice.setVisible(bool(message))

    # --- controls --------------------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._loaded and self.play_button.isEnabled()

    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlayingState

    def position_ms(self) -> int:
        return int(self._player.position())

    def duration_ms(self) -> int:
        return int(self._player.duration()) or self._duration_hint_ms

    def play(self) -> None:
        if self.is_available():
            self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def toggle(self) -> None:
        if self.is_playing():
            self.pause()
        else:
            self.play()

    def seek_ms(self, position_ms: int) -> None:
        if not self.is_available():
            return
        duration = self.duration_ms()
        target = max(0, min(int(position_ms), duration if duration else int(position_ms)))
        if self._player.mediaStatus() in (QMediaPlayer.NoMedia, QMediaPlayer.LoadingMedia):
            self._pending_seek_ms = target  # applied once the file has finished opening
        else:
            self._player.setPosition(target)
        self._on_position(target)

    def set_peaks(self, peaks: list[float]) -> None:
        self.waveform.set_peaks(peaks)

    # --- player events ---------------------------------------------------------------------------

    def _on_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia) and self._pending_seek_ms is not None:
            target, self._pending_seek_ms = self._pending_seek_ms, None
            self._player.setPosition(target)

    def _on_speed_changed(self) -> None:
        self._player.setPlaybackRate(float(self.speed_combo.currentData() or 1.0))

    def _on_position(self, position_ms: int) -> None:
        self.waveform.set_position_ms(position_ms)
        self._update_time()
        self.position_changed.emit(int(position_ms))

    def _on_duration(self, duration_ms: int) -> None:
        if duration_ms > 0:
            self.waveform.set_duration_ms(duration_ms)
        self._update_time()

    def _on_state(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlayingState
        self.play_button.setText("⏸" if playing else "▶")
        self.play_button.setAccessibleName("Pause" if playing else "Play")
        self.playing_changed.emit(playing)

    def _on_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error == QMediaPlayer.NoError:
            return
        LOGGER.warning("Media playback error: %s %s", error, message)
        self._loaded = False
        self._set_enabled(False, "This recording can't be played on this system. The transcript can still be read and exported.")

    def _update_time(self) -> None:
        self.time_label.setText(f"{format_clock(self.waveform.position_ms() / 1000)} / {format_clock(self.waveform.duration_ms() / 1000)}")
