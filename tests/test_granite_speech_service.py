"""Tests for Granite Speech prompt and decode helpers."""

from __future__ import annotations

import unittest

from services.granite_speech_service import (
    GraniteSpeechModelBundle,
    build_granite_chat_prompt,
    build_granite_prompt,
    transcribe_granite_audio,
)


class _FakeInputIds:
    shape = (1, 3)


class _FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids="prompt-tokens")
        self.input_ids = _FakeInputIds()
        self.sent_to_device: str | None = None

    def to(self, device: str) -> "_FakeInputs":
        self.sent_to_device = device
        return self


class _FakeGeneratedIds:
    def __init__(self) -> None:
        self.slice_key = None

    def __getitem__(self, key):  # noqa: ANN001
        self.slice_key = key
        return "generated-tail"


class _FakeProcessor:
    def __init__(self) -> None:
        self.last_inputs: _FakeInputs | None = None
        self.last_prompt: str | None = None
        self.tokenizer = _FakeTokenizer()

    def __call__(self, prompt: str, audio_np: object, device: str, return_tensors: str) -> _FakeInputs:
        del audio_np, device, return_tensors
        self.last_prompt = prompt
        self.last_inputs = _FakeInputs()
        return self.last_inputs


class _FakeTokenizer:
    def __init__(self) -> None:
        self.last_chat = None
        self.last_decoded = None

    def apply_chat_template(self, chat: object, tokenize: bool, add_generation_prompt: bool) -> str:
        self.last_chat = (chat, tokenize, add_generation_prompt)
        return "templated prompt"

    def batch_decode(self, generated_ids: object, add_special_tokens: bool, skip_special_tokens: bool) -> list[str]:
        self.last_decoded = (generated_ids, add_special_tokens, skip_special_tokens)
        return ["decoded transcript"]


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.generated = _FakeGeneratedIds()

    def generate(self, **kwargs: object) -> _FakeGeneratedIds:
        self.calls.append(dict(kwargs))
        return self.generated


class GraniteSpeechServiceTests(unittest.TestCase):
    def test_build_granite_prompt_appends_keywords(self) -> None:
        prompt = build_granite_prompt(keywords=["canine", "feline"])

        self.assertIn("<|audio|>", prompt)
        self.assertIn("Keywords: canine, feline", prompt)

    def test_build_granite_chat_prompt_uses_chat_template(self) -> None:
        processor = _FakeProcessor()
        bundle = GraniteSpeechModelBundle(
            processor=processor,
            model=object(),
            model_name="ibm-granite/granite-4.0-1b-speech",
            device="cpu",
        )

        prompt = build_granite_chat_prompt(bundle, keywords=["vet"])

        self.assertEqual(prompt, "templated prompt")
        chat, tokenize, add_generation_prompt = processor.tokenizer.last_chat
        self.assertEqual(chat[0]["role"], "user")
        self.assertIn("Keywords: vet", chat[0]["content"])
        self.assertFalse(tokenize)
        self.assertTrue(add_generation_prompt)

    def test_transcribe_granite_audio_decodes_only_generated_tokens(self) -> None:
        processor = _FakeProcessor()
        model = _FakeModel()
        bundle = GraniteSpeechModelBundle(
            processor=processor,
            model=model,
            model_name="ibm-granite/granite-4.0-1b-speech",
            device="cpu",
        )

        transcript = transcribe_granite_audio(bundle, audio_np=[0.1, 0.2], keywords=["vet"])

        self.assertEqual(transcript, "decoded transcript")
        self.assertEqual(processor.last_prompt, "templated prompt")
        self.assertEqual(processor.last_inputs.sent_to_device, "cpu")
        self.assertEqual(model.generated.slice_key, (slice(None, None, None), slice(3, None, None)))
        self.assertEqual(processor.tokenizer.last_decoded, ("generated-tail", False, True))

    def test_transcribe_granite_audio_chunks_long_audio_and_reports_progress(self) -> None:
        processor = _FakeProcessor()
        model = _FakeModel()
        bundle = GraniteSpeechModelBundle(
            processor=processor,
            model=model,
            model_name="ibm-granite/granite-4.0-1b-speech",
            device="cpu",
        )
        progress: list[float] = []

        transcript = transcribe_granite_audio(
            bundle,
            audio_np=[0.0] * (31 * 16000),
            progress_cb=progress.append,
        )

        self.assertTrue(transcript)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(progress, [50.0, 100.0])


if __name__ == "__main__":
    unittest.main()
