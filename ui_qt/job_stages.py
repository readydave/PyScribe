"""Job stage model for the Qt progress timeline (no Qt imports, so it is easy to test)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class Stage(str, Enum):
    LOAD = "load"
    TRANSCRIBE = "transcribe"
    SPEAKERS = "speakers"
    VISUALS = "visuals"
    SAVE = "save"


STAGE_LABELS: dict[Stage, str] = {
    Stage.LOAD: "Load model",
    Stage.TRANSCRIBE: "Transcribe",
    Stage.SPEAKERS: "Speakers",
    Stage.VISUALS: "Visuals",
    Stage.SAVE: "Save",
}

# Order in which stages are shown.
STAGE_ORDER: tuple[Stage, ...] = (Stage.LOAD, Stage.TRANSCRIBE, Stage.SPEAKERS, Stage.VISUALS, Stage.SAVE)

PENDING = "pending"
ACTIVE = "active"
DONE = "done"
FAILED = "failed"
DISABLED = "disabled"


@dataclass
class StageInfo:
    state: str = PENDING
    percent: int = 0
    started_at: float | None = None
    elapsed: float | None = None


class JobTracker:
    """Tracks per-stage state, percent and elapsed seconds for one job."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._stages: dict[Stage, StageInfo] = {stage: StageInfo() for stage in STAGE_ORDER}

    def reset(self, enabled: set[Stage]) -> None:
        """Start a new job; stages not in ``enabled`` are marked disabled."""
        self._stages = {
            stage: StageInfo(state=PENDING if stage in enabled else DISABLED) for stage in STAGE_ORDER
        }

    def info(self, stage: Stage) -> StageInfo:
        return self._stages[stage]

    def start(self, stage: Stage) -> None:
        """Mark a stage active without a percent (stages that have no progress value)."""
        info = self._stages[stage]
        if info.state in (DISABLED, FAILED):
            return
        if info.started_at is None:
            info.started_at = self._clock()
        info.state = ACTIVE

    def progress(self, stage: Stage, percent: int) -> None:
        """Record a progress value (0-100). The first value starts the stage; 100 completes it."""
        info = self._stages[stage]
        if info.state in (DISABLED, FAILED):
            return
        percent = max(0, min(100, int(percent)))
        if info.started_at is None:
            info.started_at = self._clock()
        info.percent = percent
        if percent >= 100:
            self._complete(info)
        else:
            info.state = ACTIVE

    def complete(self, stage: Stage, elapsed: float | None = None) -> None:
        """Mark a stage done, optionally with a measured elapsed time from the worker."""
        info = self._stages[stage]
        if info.state in (DISABLED, FAILED):
            return
        info.percent = 100
        self._complete(info)
        if elapsed is not None:
            info.elapsed = elapsed

    def fail_active(self) -> None:
        """Mark every in-progress stage as failed (pending stages stay pending)."""
        for info in self._stages.values():
            if info.state == ACTIVE:
                info.state = FAILED
                info.elapsed = self._elapsed_now(info)

    def cancel_active(self) -> None:
        """Return in-progress stages to pending after a cancel."""
        for info in self._stages.values():
            if info.state == ACTIVE:
                info.state = PENDING
                info.percent = 0
                info.started_at = None

    def _complete(self, info: StageInfo) -> None:
        info.state = DONE
        if info.elapsed is None:
            info.elapsed = self._elapsed_now(info)

    def _elapsed_now(self, info: StageInfo) -> float | None:
        if info.started_at is None:
            return None
        return max(0.0, self._clock() - info.started_at)
