"""Tests for backend-aware transcription model resolution."""

from __future__ import annotations

import unittest

from services.model_service import model_supports_diarization, resolve_transcription_model


class ModelServiceTests(unittest.TestCase):
    def test_resolve_transcription_model_identifies_granite_repo(self) -> None:
        spec = resolve_transcription_model("ibm-granite/granite-4.0-1b-speech")

        self.assertEqual(spec.backend_kind, "granite_transformers")
        self.assertFalse(spec.supports_diarization)
        self.assertTrue(spec.is_experimental)

    def test_resolve_transcription_model_identifies_granite_url(self) -> None:
        spec = resolve_transcription_model("https://huggingface.co/ibm-granite/granite-4.0-1b-speech")

        self.assertEqual(spec.repo_id, "ibm-granite/granite-4.0-1b-speech")
        self.assertEqual(spec.backend_kind, "granite_transformers")

    def test_resolve_transcription_model_defaults_custom_repo_to_whisper_path(self) -> None:
        spec = resolve_transcription_model("owner/custom-ct2-model")

        self.assertEqual(spec.backend_kind, "faster_whisper")
        self.assertTrue(spec.supports_diarization)
        self.assertTrue(model_supports_diarization("owner/custom-ct2-model"))


if __name__ == "__main__":
    unittest.main()
