"""Run one listener transcription job on a worker thread and stream stage updates.

``run_media_job`` wraps ``transcribe_media_file`` so a UI generator (Gradio) can show live
per-stage progress without importing Qt. Callbacks fire on the worker thread and push to a
queue; the generator drains it in the caller's thread and yields ``JobUpdate`` snapshots.

Caller-supplied ``on_*`` kwargs (``on_status``, ``on_text``, ``on_progress``, ``on_diar_progress``,
``on_visual_progress``, ``on_model_download_progress``, ``on_stage``) are wrapped, not dropped: they
are called from the worker thread after the job records the event, and their exceptions are logged.

Cancellation goes through ``cancel_event`` (created when absent). A cancelled job, including one that
ends in ``InterruptedError`` while ``cancel_event`` is set, ends with ``cancelled=True`` and no raise;
any other exception is yielded in ``error`` on the final update and then re-raised in the caller.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator

from services.job_stages import ACTIVE, DISABLED, DONE, STAGE_ORDER, JobTracker, Stage, StageInfo

LOGGER = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 0.25
JOIN_TIMEOUT_SECONDS = 30.0

_CALLBACK_STAGES: dict[str, Stage] = {
    "on_progress": Stage.TRANSCRIBE,
    "on_diar_progress": Stage.SPEAKERS,
    "on_visual_progress": Stage.VISUALS,
    "on_model_download_progress": Stage.LOAD,
}
_ALL_CALLBACKS = (*_CALLBACK_STAGES, "on_status", "on_text", "on_stage")


@dataclass(frozen=True)
class JobUpdate:
    """Snapshot of a running listener job."""

    stages: dict[Stage, StageInfo]
    status: str = ""
    transcript: str = ""
    overall_pct: int = 0
    result: Any | None = None
    error: BaseException | None = None
    cancelled: bool = False
    finished: bool = False


@dataclass
class _State:
    tracker: JobTracker
    status: str = ""
    transcript: str = ""
    result: Any | None = None
    error: BaseException | None = None
    cancelled: bool = False
    finished: bool = False
    enabled: set[Stage] = field(default_factory=set)

    def snapshot(self) -> JobUpdate:
        stages = {stage: replace(self.tracker.info(stage)) for stage in STAGE_ORDER}
        counted = [s for s in STAGE_ORDER if s is not Stage.SAVE and stages[s].state != DISABLED]
        total = sum(100 if stages[s].state == DONE else stages[s].percent for s in counted)
        pct = int(total / len(counted)) if counted else 0
        if self.finished and not self.cancelled and self.error is None:
            pct = 100
        return JobUpdate(
            stages=stages,
            status=self.status,
            transcript=self.transcript,
            overall_pct=max(0, min(100, pct)),
            result=self.result,
            error=self.error,
            cancelled=self.cancelled,
            finished=self.finished,
        )


def enabled_stages(run_mode: str, use_diarization: bool, use_visual_analysis: bool) -> set[Stage]:
    """Stages a job will run (Save is never enabled here; the listener saves elsewhere)."""
    mode = str(run_mode or "full").strip().lower()
    enabled: set[Stage] = set()
    if mode != "visual_only":
        enabled |= {Stage.LOAD, Stage.TRANSCRIBE}
        if use_diarization:
            enabled.add(Stage.SPEAKERS)
    if mode == "visual_only" or use_visual_analysis:
        enabled.add(Stage.VISUALS)
    if mode == "transcribe_only":
        enabled.discard(Stage.VISUALS)
    return enabled


def _wrap_callbacks(
    q: "queue.Queue[tuple[str, Any]]", user: dict[str, Callable[..., None] | None]
) -> dict[str, Callable[..., None]]:
    def make(name: str) -> Callable[..., None]:
        def cb(*args: Any) -> None:
            q.put((name, args))
            fn = user.get(name)
            if fn is not None:
                try:
                    fn(*args)
                except Exception as exc:  # caller callback must not kill the job
                    LOGGER.warning("listener_job: %s callback failed: %s", name, exc)

        return cb

    return {name: make(name) for name in _ALL_CALLBACKS}


def _apply(state: _State, name: str, args: tuple[Any, ...]) -> None:
    tracker = state.tracker
    if name == "on_status":
        state.status = str(args[0])
    elif name == "on_text":
        state.transcript = str(args[0])
    elif name == "on_stage":
        stage_name, phase = str(args[0]), str(args[1])
        try:
            stage = Stage(stage_name)
        except ValueError:
            return
        if phase == "start":
            tracker.start(stage)
        elif phase == "done":
            tracker.complete(stage)
    elif name in _CALLBACK_STAGES:
        stage = _CALLBACK_STAGES[name]
        pct = int(float(args[0]))
        # Download progress must not finish LOAD; only on_stage("load", "done") does.
        tracker.progress(stage, min(pct, 99) if stage is Stage.LOAD else pct)


def _finish(state: _State, kind: str, payload: Any, cancel_event: threading.Event) -> None:
    tracker = state.tracker
    state.finished = True
    if kind == "error":
        if isinstance(payload, InterruptedError) and cancel_event.is_set():
            state.cancelled = True
            tracker.cancel_active()
        else:
            state.error = payload
            tracker.fail_active()
        return
    state.result = payload
    if bool(getattr(payload, "cancelled", False)) or cancel_event.is_set():
        state.cancelled = True
        tracker.cancel_active()
        return
    for stage in state.enabled:
        if tracker.info(stage).state != DONE:
            tracker.complete(stage)


def run_media_job(
    *, transcribe_fn: Callable[..., Any] | None = None, **transcribe_kwargs: Any
) -> Iterator[JobUpdate]:
    """Run ``transcribe_fn`` (default ``transcribe_media_file``) on a daemon thread and yield updates.

    The last update has ``finished=True`` and either ``result``, ``cancelled`` or ``error`` set; an
    ``error`` is re-raised after that update is yielded.
    """
    if transcribe_fn is None:
        from services.transcription_service import transcribe_media_file as transcribe_fn

    cancel_event: threading.Event = transcribe_kwargs.get("cancel_event") or threading.Event()
    transcribe_kwargs["cancel_event"] = cancel_event
    enabled = enabled_stages(
        transcribe_kwargs.get("run_mode", "full"),
        bool(transcribe_kwargs.get("use_diarization", False)),
        bool(transcribe_kwargs.get("use_visual_analysis", False)),
    )
    state = _State(tracker=JobTracker(), enabled=enabled)
    state.tracker.reset(enabled)

    q: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    user = {name: transcribe_kwargs.pop(name, None) for name in _ALL_CALLBACKS}
    transcribe_kwargs.update(_wrap_callbacks(q, user))

    def work() -> None:
        try:
            q.put(("done", transcribe_fn(**transcribe_kwargs)))
        except BaseException as exc:  # delivered to the caller's thread
            q.put(("error", exc))

    thread = threading.Thread(target=work, name="listener-job", daemon=True)
    thread.start()
    try:
        yield state.snapshot()
        while not state.finished:
            try:
                events = [q.get(timeout=HEARTBEAT_SECONDS)]
            except queue.Empty:
                yield state.snapshot()
                continue
            while True:
                try:
                    events.append(q.get_nowait())
                except queue.Empty:
                    break
            for name, payload in events:
                if name in ("done", "error"):
                    _finish(state, name, payload, cancel_event)
                    break
                _apply(state, name, payload)
            yield state.snapshot()
        if state.error is not None:
            raise state.error
    finally:
        if not state.finished:
            cancel_event.set()
        thread.join(JOIN_TIMEOUT_SECONDS)
        if thread.is_alive():
            LOGGER.warning("listener_job: worker thread still alive after %ss join", JOIN_TIMEOUT_SECONDS)
