from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from services import listener_job
from services.job_stages import ACTIVE, DISABLED, DONE, FAILED, PENDING, Stage
from services.listener_job import enabled_stages, run_media_job


def _result(cancelled: bool = False) -> SimpleNamespace:
    return SimpleNamespace(transcript="hello", cancelled=cancelled)


def _job_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "listener-job" and t.is_alive()]


def test_enabled_stages_modes() -> None:
    assert enabled_stages("full", False, False) == {Stage.LOAD, Stage.TRANSCRIBE}
    assert enabled_stages("full", True, True) == {Stage.LOAD, Stage.TRANSCRIBE, Stage.SPEAKERS, Stage.VISUALS}
    assert enabled_stages("transcribe_only", False, True) == {Stage.LOAD, Stage.TRANSCRIBE}
    assert enabled_stages("visual_only", True, False) == {Stage.VISUALS}


def test_event_order_and_final_all_done() -> None:
    seen: list[str] = []

    def fake(**kw):
        kw["on_stage"]("load", "start")
        kw["on_model_download_progress"](100)
        kw["on_stage"]("load", "done")
        kw["on_status"]("Transcribing")
        kw["on_progress"](50)
        kw["on_text"]("partial")
        kw["on_progress"](99.5)
        kw["on_diar_progress"](40)
        return _result()

    updates = list(
        run_media_job(
            transcribe_fn=fake, media_path="a.wav", model_name="m", use_diarization=True, on_status=seen.append
        )
    )
    assert seen == ["Transcribing"]  # caller callback is wrapped, not dropped
    final = updates[-1]
    assert final.finished and final.result.transcript == "hello"
    assert not final.cancelled and final.error is None
    assert final.overall_pct == 100
    assert all(final.stages[s].state == DONE for s in (Stage.LOAD, Stage.TRANSCRIBE, Stage.SPEAKERS))
    assert final.stages[Stage.VISUALS].state == DISABLED and final.stages[Stage.SAVE].state == DISABLED
    assert not any(u.finished for u in updates[:-1])
    texts = [u.transcript for u in updates]
    assert texts.index("partial") > texts.index("")
    assert _job_threads() == []


def test_intermediate_states_and_load_stays_active() -> None:
    gate = threading.Event()
    release = threading.Event()

    def fake(**kw):
        kw["on_stage"]("load", "start")
        kw["on_model_download_progress"](100)
        gate.set()
        release.wait(5)
        return _result()

    gen = run_media_job(transcribe_fn=fake, media_path="a.wav", model_name="m")
    next(gen)
    assert gate.wait(5)
    mid = next(gen)
    while mid.stages[Stage.LOAD].percent < 99:
        mid = next(gen)
    assert mid.stages[Stage.LOAD].state == ACTIVE
    assert mid.stages[Stage.TRANSCRIBE].state == PENDING
    release.set()
    assert list(gen)[-1].finished
    assert _job_threads() == []


def test_visual_only_maps_visual_progress() -> None:
    def fake(**kw):
        kw["on_visual_progress"](30)
        kw["on_progress"](80)  # TRANSCRIBE disabled: ignored
        return _result()

    updates = list(run_media_job(transcribe_fn=fake, media_path="a", model_name="m", run_mode="visual_only"))
    final = updates[-1]
    assert final.stages[Stage.LOAD].state == DISABLED
    assert final.stages[Stage.TRANSCRIBE].state == DISABLED
    assert final.stages[Stage.VISUALS].state == DONE


def test_cancel_returns_cancelled_update_without_raise() -> None:
    cancel = threading.Event()

    def fake(**kw):
        kw["on_stage"]("load", "start")
        cancel.set()
        return _result(cancelled=True)

    final = list(run_media_job(transcribe_fn=fake, media_path="a", model_name="m", cancel_event=cancel))[-1]
    assert final.cancelled and final.finished and final.error is None
    assert final.stages[Stage.LOAD].state == PENDING
    assert _job_threads() == []


def test_interrupted_error_with_cancel_is_cancelled() -> None:
    cancel = threading.Event()

    def fake(**kw):
        cancel.set()
        raise InterruptedError("Cancelled during diarization.")

    final = list(run_media_job(transcribe_fn=fake, media_path="a", model_name="m", cancel_event=cancel))[-1]
    assert final.cancelled and final.error is None


def test_exception_yields_error_then_reraises() -> None:
    def fake(**kw):
        kw["on_stage"]("load", "start")
        raise RuntimeError("boom")

    seen: list = []
    gen = run_media_job(transcribe_fn=fake, media_path="a", model_name="m")
    with pytest.raises(RuntimeError, match="boom"):
        for update in gen:
            seen.append(update)
    final = seen[-1]
    assert final.finished and isinstance(final.error, RuntimeError)
    assert final.stages[Stage.LOAD].state == FAILED
    assert _job_threads() == []


def test_generator_close_sets_cancel_and_joins() -> None:
    started = threading.Event()
    cancel = threading.Event()

    def fake(**kw):
        started.set()
        kw["cancel_event"].wait(5)
        return _result(cancelled=True)

    gen = run_media_job(transcribe_fn=fake, media_path="a", model_name="m", cancel_event=cancel)
    next(gen)
    assert started.wait(5)
    gen.close()
    assert cancel.is_set()
    assert _job_threads() == []


def test_close_join_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(listener_job, "JOIN_TIMEOUT_SECONDS", 0.1)
    stop = threading.Event()

    def fake(**kw):
        stop.wait(5)
        return _result()

    gen = run_media_job(transcribe_fn=fake, media_path="a", model_name="m")
    next(gen)
    gen.close()  # returns despite the stuck worker
    stop.set()
    for t in _job_threads():
        t.join(2)
    assert _job_threads() == []
