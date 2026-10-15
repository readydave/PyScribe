from __future__ import annotations

import unittest

from ui_qt.job_stages import ACTIVE, DISABLED, DONE, FAILED, PENDING, STAGE_ORDER, JobTracker, Stage
from ui_qt.theme import PALETTES, build_qss, sanitize_mode


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class JobTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tracker = JobTracker(clock=self.clock)
        self.tracker.reset({Stage.TRANSCRIBE, Stage.SPEAKERS})

    def test_reset_marks_unselected_stages_disabled(self) -> None:
        self.assertEqual(self.tracker.info(Stage.TRANSCRIBE).state, PENDING)
        self.assertEqual(self.tracker.info(Stage.SPEAKERS).state, PENDING)
        self.assertEqual(self.tracker.info(Stage.VISUALS).state, DISABLED)

    def test_progress_activates_then_completes_with_elapsed(self) -> None:
        self.tracker.progress(Stage.TRANSCRIBE, 10)
        self.assertEqual(self.tracker.info(Stage.TRANSCRIBE).state, ACTIVE)
        self.clock.now += 12.5
        self.tracker.progress(Stage.TRANSCRIBE, 100)
        info = self.tracker.info(Stage.TRANSCRIBE)
        self.assertEqual(info.state, DONE)
        self.assertAlmostEqual(info.elapsed or 0.0, 12.5)

    def test_worker_elapsed_overrides_measured_time(self) -> None:
        self.tracker.progress(Stage.TRANSCRIBE, 5)
        self.clock.now += 3
        self.tracker.complete(Stage.TRANSCRIBE, elapsed=9.0)
        self.assertEqual(self.tracker.info(Stage.TRANSCRIBE).elapsed, 9.0)

    def test_disabled_stage_ignores_progress(self) -> None:
        self.tracker.progress(Stage.VISUALS, 50)
        self.assertEqual(self.tracker.info(Stage.VISUALS).state, DISABLED)

    def test_progress_is_clamped(self) -> None:
        self.tracker.progress(Stage.TRANSCRIBE, 250)
        self.assertEqual(self.tracker.info(Stage.TRANSCRIBE).percent, 100)
        self.assertEqual(self.tracker.info(Stage.TRANSCRIBE).state, DONE)

    def test_fail_marks_only_active_stages(self) -> None:
        self.tracker.progress(Stage.TRANSCRIBE, 40)
        self.tracker.fail_active()
        self.assertEqual(self.tracker.info(Stage.TRANSCRIBE).state, FAILED)
        self.assertEqual(self.tracker.info(Stage.SPEAKERS).state, PENDING)
        self.tracker.progress(Stage.TRANSCRIBE, 60)
        self.assertEqual(self.tracker.info(Stage.TRANSCRIBE).state, FAILED)

    def test_cancel_returns_active_stages_to_pending(self) -> None:
        self.tracker.progress(Stage.TRANSCRIBE, 40)
        self.tracker.cancel_active()
        info = self.tracker.info(Stage.TRANSCRIBE)
        self.assertEqual(info.state, PENDING)
        self.assertEqual(info.percent, 0)

    def test_start_marks_stage_active_without_percent(self) -> None:
        self.tracker.reset({Stage.LOAD, Stage.TRANSCRIBE})
        self.tracker.start(Stage.LOAD)
        self.assertEqual(self.tracker.info(Stage.LOAD).state, ACTIVE)
        self.tracker.complete(Stage.LOAD)
        self.assertEqual(self.tracker.info(Stage.LOAD).state, DONE)
        self.assertIsNotNone(self.tracker.info(Stage.LOAD).elapsed)

    def test_start_ignores_disabled_stage(self) -> None:
        self.tracker.reset({Stage.TRANSCRIBE})
        self.tracker.start(Stage.SAVE)
        self.assertEqual(self.tracker.info(Stage.SAVE).state, DISABLED)

    def test_load_and_save_bracket_the_pipeline(self) -> None:
        self.assertEqual(STAGE_ORDER[0], Stage.LOAD)
        self.assertEqual(STAGE_ORDER[-1], Stage.SAVE)


class ThemeTests(unittest.TestCase):
    def test_sanitize_mode(self) -> None:
        self.assertEqual(sanitize_mode("DARK"), "dark")
        self.assertEqual(sanitize_mode("bogus"), "system")
        self.assertEqual(sanitize_mode(None), "system")

    def test_qss_uses_palette_tokens_for_both_modes(self) -> None:
        for mode, palette in PALETTES.items():
            qss = build_qss(mode)
            self.assertIn(palette.page, qss)
            self.assertIn(palette.rubric, qss)
            self.assertIn('QProgressBar[state="done"]', qss)


if __name__ == "__main__":
    unittest.main()
