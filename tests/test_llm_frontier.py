"""Tests for cloud/frontier providers, context budgeting, long-transcript handling, and transport safety."""

from __future__ import annotations

import http.server
import json
import threading
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib import request as urlrequest
from urllib.error import HTTPError

from services import llm_postprocess_service as svc
from services.llm_connection_service import (
    anthropic_endpoint_url,
    effective_context_tokens,
    effective_max_output_tokens,
    evaluate_profile_scope_policy,
    load_llm_profiles,
    open_url,
    openai_endpoint_url,
    openai_max_tokens_field,
    provider_auth_headers,
)
from services.llm_postprocess_service import (
    LLMPostprocessRequest,
    estimate_tokens,
    run_llm_postprocess,
    split_text_by_tokens,
)
from services.prompt_template_service import PromptTemplate


def _template() -> PromptTemplate:
    return PromptTemplate(
        id="t",
        name="T",
        version=1,
        description="",
        tags=(),
        output_format="markdown",
        enabled=True,
        built_in=True,
        system_prompt="SYSTEM",
        user_prompt_scaffold="SCAFFOLD",
        source_path="inline",
    )


def _profile(**overrides: object):
    raw = {
        "name": "p",
        "provider": "anthropic",
        "scope": "cloud",
        "base_url": "https://api.anthropic.com",
        "api_key_runtime": "sk-ant-test-key-123456",
        "default_model": "claude-sonnet-5-5",
        "cloud_acknowledged": True,
    }
    raw.update(overrides)
    profiles = load_llm_profiles([raw])
    assert profiles, raw
    return profiles[0]


class _Resp:
    def __init__(self, payload: object | None = None, lines: list[bytes] | None = None) -> None:
        self._body = json.dumps(payload).encode() if payload is not None else b""
        self._lines = lines or []

    def read(self) -> bytes:
        return self._body

    def __iter__(self):  # noqa: ANN204
        return iter(self._lines)

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class CloudPolicyTests(unittest.TestCase):
    def test_cloud_profile_defaults(self) -> None:
        profile = _profile(verify_tls=False)
        self.assertTrue(profile.verify_tls)  # cannot be switched off for cloud
        self.assertIsNone(profile.temperature)
        self.assertEqual(profile.timeout_seconds, 120.0)
        local = load_llm_profiles([{"name": "l", "provider": "ollama", "scope": "local",
                                    "base_url": "http://127.0.0.1:11434"}])[0]
        self.assertEqual(local.temperature, 0.2)

    def test_cloud_policy_checks(self) -> None:
        self.assertEqual(evaluate_profile_scope_policy(_profile()), (True, None, None))
        for overrides, code in (
            ({"cloud_acknowledged": False}, "policy_cloud_not_acknowledged"),
            ({"base_url": "http://api.anthropic.com"}, "policy_cloud_requires_https"),
            ({"base_url": "https://127.0.0.1:8443"}, "policy_cloud_not_remote"),
            ({"base_url": "https://192.168.1.5"}, "policy_cloud_not_remote"),
            ({"api_key_runtime": ""}, "policy_cloud_requires_key"),
        ):
            ok, found, _detail = evaluate_profile_scope_policy(_profile(**overrides))
            self.assertFalse(ok)
            self.assertEqual(found, code, overrides)

    def test_run_is_blocked_without_acknowledgement(self) -> None:
        result = run_llm_postprocess(_profile(cloud_acknowledged=False), _template(),
                                     LLMPostprocessRequest(transcript_text="hello"))
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "policy_cloud_not_acknowledged")

    def test_endpoint_helpers(self) -> None:
        self.assertEqual(openai_endpoint_url("http://h:1234", "/models"), "http://h:1234/v1/models")
        self.assertEqual(openai_endpoint_url("http://h/llm", "/models"), "http://h/llm/v1/models")
        self.assertEqual(openai_endpoint_url("https://api.openai.com", "/chat/completions", "cloud"),
                         "https://api.openai.com/v1/chat/completions")
        self.assertEqual(
            openai_endpoint_url("https://generativelanguage.googleapis.com/v1beta/openai", "/models", "cloud"),
            "https://generativelanguage.googleapis.com/v1beta/openai/models",
        )
        self.assertEqual(anthropic_endpoint_url("https://api.anthropic.com", "/messages"),
                         "https://api.anthropic.com/v1/messages")
        self.assertEqual(openai_max_tokens_field("https://api.openai.com"), "max_completion_tokens")
        self.assertEqual(openai_max_tokens_field("http://127.0.0.1:1234"), "max_tokens")

    def test_auth_headers_per_provider(self) -> None:
        self.assertEqual(provider_auth_headers("openai_compatible", "k"), {"Authorization": "Bearer k"})
        anthropic = provider_auth_headers("anthropic", "k")
        self.assertEqual(anthropic["x-api-key"], "k")
        self.assertIn("anthropic-version", anthropic)

    def test_default_limits(self) -> None:
        self.assertEqual(effective_context_tokens(_profile()), 180000)
        self.assertEqual(effective_max_output_tokens(_profile()), 8192)
        self.assertEqual(effective_context_tokens(_profile(context_tokens=5000)), 5000)
        local = load_llm_profiles([{"name": "l", "provider": "ollama", "scope": "local",
                                    "base_url": "http://127.0.0.1:11434"}])[0]
        self.assertEqual(effective_context_tokens(local), 16384)


class ProviderRequestTests(unittest.TestCase):
    def _run(self, profile, responses, **kwargs):
        sent: list[dict] = []

        def fake_open(request, timeout=None, context=None):  # noqa: ANN001, ARG001
            sent.append({"url": request.full_url, "headers": dict(request.header_items()),
                         "body": json.loads(request.data.decode())})
            return responses.pop(0) if isinstance(responses, list) else responses

        with patch("services.llm_postprocess_service.open_url", side_effect=fake_open):
            result = run_llm_postprocess(profile, _template(), LLMPostprocessRequest(transcript_text="Alice: hi"), **kwargs)
        return result, sent

    def test_anthropic_request_and_response(self) -> None:
        result, sent = self._run(_profile(), _Resp({"content": [{"type": "text", "text": "Summary"}], "stop_reason": "end_turn"}))
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.output_text, "Summary")
        request = sent[0]
        self.assertEqual(request["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(request["headers"].get("X-api-key"), "sk-ant-test-key-123456")
        self.assertIn("Anthropic-version", request["headers"])
        body = request["body"]
        self.assertEqual(body["model"], "claude-sonnet-5-5")
        self.assertEqual(body["system"], "SYSTEM")
        self.assertEqual(body["max_tokens"], 8192)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["messages"][0]["role"], "user")

    def test_anthropic_truncation_is_reported(self) -> None:
        result, _ = self._run(_profile(), _Resp({"content": [{"type": "text", "text": "Cut"}], "stop_reason": "max_tokens"}))
        self.assertEqual(result.status, "pass")
        self.assertIn("maximum output length", result.info_note or "")

    def test_anthropic_streaming(self) -> None:
        lines = [
            b"event: content_block_delta\n",
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n',
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n',
            b'data: {"type":"message_stop"}\n',
        ]
        seen: list[str] = []
        result, sent = self._run(_profile(), _Resp(lines=lines), on_output_chunk=seen.append)
        self.assertEqual(result.output_text, "Hello")
        self.assertEqual(seen, ["Hel", "lo"])
        self.assertTrue(sent[0]["body"]["stream"])

    def test_anthropic_stream_error_event(self) -> None:
        lines = [b'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n']
        result, _ = self._run(_profile(), _Resp(lines=lines), on_output_chunk=lambda _t: None)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "server_busy")

    def test_openai_cloud_omits_temperature_and_uses_completion_tokens(self) -> None:
        profile = _profile(provider="openai_compatible", base_url="https://api.openai.com", default_model="some-model")
        result, sent = self._run(profile, _Resp({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}))
        self.assertEqual(result.output_text, "ok")
        body = sent[0]["body"]
        self.assertNotIn("temperature", body)
        self.assertEqual(body["max_completion_tokens"], 8192)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(sent[0]["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(sent[0]["headers"].get("Authorization"), "Bearer sk-ant-test-key-123456")

    def test_ollama_gets_context_and_output_limits(self) -> None:
        profile = load_llm_profiles([{"name": "o", "provider": "ollama", "scope": "local",
                                      "base_url": "http://127.0.0.1:11434", "default_model": "llama3",
                                      "context_tokens": 12000, "max_output_tokens": 900}])[0]
        result, sent = self._run(profile, _Resp({"response": "done", "done_reason": "stop"}))
        self.assertEqual(result.output_text, "done")
        options = sent[0]["body"]["options"]
        self.assertEqual(options["num_ctx"], 12000)
        self.assertEqual(options["num_predict"], 900)
        self.assertEqual(options["temperature"], 0.2)


class ErrorTranslationTests(unittest.TestCase):
    def _error(self, code: int, body: dict | str) -> svc._LLMPostprocessException:
        raw = body if isinstance(body, str) else json.dumps(body)
        exc = HTTPError("https://x", code, "err", {}, BytesIO(raw.encode()))
        return svc._translate_error(exc)

    def test_rate_limit_and_overload(self) -> None:
        self.assertEqual(self._error(429, {"error": {"message": "Rate limit hit"}}).code, "rate_limited")
        self.assertEqual(self._error(529, {"type": "error", "error": {"message": "Overloaded"}}).code, "server_busy")

    def test_context_errors_are_recognised(self) -> None:
        err = self._error(400, {"error": {"message": "This model's maximum context length is 8192 tokens"}})
        self.assertEqual(err.code, "context_exceeded")
        err = self._error(400, {"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "prompt is too long: 250000 tokens"}})
        self.assertEqual(err.code, "context_exceeded")
        self.assertEqual(self._error(400, {"error": {"message": "bad field"}}).code, "server_error")

    def test_auth_message_is_included_and_keys_are_redacted(self) -> None:
        err = self._error(401, {"error": {"message": "Incorrect API key provided: sk-proj-abcdefghijkl1234"}})
        self.assertEqual(err.code, "auth_failed")
        self.assertIn("Incorrect API key provided", str(err))
        self.assertNotIn("sk-proj-abcdefghijkl1234", str(err))

    def test_unreadable_body_still_maps(self) -> None:
        exc = HTTPError("https://x", 500, "err", {}, None)
        self.assertEqual(svc._translate_error(exc).code, "server_error")


class LongTranscriptTests(unittest.TestCase):
    def test_estimate_and_split(self) -> None:
        self.assertGreater(estimate_tokens("a" * 320), 99)
        text = "\n".join(f"Speaker {i % 3}: sentence number {i} with some words" for i in range(400))
        parts = split_text_by_tokens(text, 500)
        self.assertGreater(len(parts), 1)
        self.assertEqual("\n".join(parts).split("\n"), text.split("\n"))  # nothing lost or reordered
        for part in parts:
            self.assertLessEqual(estimate_tokens(part), 520)

    def test_very_long_single_line_is_split(self) -> None:
        parts = split_text_by_tokens("x" * 10000, 300)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), "x" * 10000)

    def test_short_transcript_makes_a_single_call(self) -> None:
        calls: list[str] = []

        def fake_call(**kwargs):  # noqa: ANN003
            calls.append(kwargs["system_prompt"])
            return svc._CallOutcome(text="final")

        with patch.object(svc, "_call_model", side_effect=fake_call):
            result = run_llm_postprocess(_profile(), _template(), LLMPostprocessRequest(transcript_text="short"))
        self.assertEqual(result.output_text, "final")
        self.assertEqual(calls, ["SYSTEM"])

    def test_long_transcript_is_split_processed_and_merged(self) -> None:
        profile = _profile(context_tokens=2000, max_output_tokens=300)
        transcript = "\n".join(f"Speaker {i % 3}: sentence number {i} with some more words in it" for i in range(500))
        calls: list[dict] = []
        statuses: list[str] = []
        streamed: list[str] = []

        def fake_call(**kwargs):  # noqa: ANN003
            calls.append(kwargs)
            system = kwargs["system_prompt"]
            if "part " in system and " of " in system and "partial results" not in system:
                return svc._CallOutcome(text=f"partial-{len(calls)}")
            if kwargs["on_output_chunk"]:
                kwargs["on_output_chunk"]("FINAL")
            return svc._CallOutcome(text="FINAL")

        request = LLMPostprocessRequest(transcript_text=transcript, ocr_text="OCR-SLIDE", notes_text="NOTE-X")
        with patch.object(svc, "_call_model", side_effect=fake_call):
            result = run_llm_postprocess(profile, _template(), request, on_output_chunk=streamed.append,
                                         on_status=statuses.append)
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.output_text, "FINAL")
        self.assertIn("processed in", result.info_note or "")
        part_calls = [c for c in calls if "This is part" in c["system_prompt"]]
        self.assertGreaterEqual(len(part_calls), 2)
        for call in part_calls:
            self.assertIsNone(call["on_output_chunk"])  # only the final answer streams
            self.assertNotIn("OCR-SLIDE", call["user_payload"])
            self.assertLessEqual(estimate_tokens(call["system_prompt"]) + estimate_tokens(call["user_payload"]),
                                 svc._limits_for(profile).input_budget)
        final = calls[-1]
        self.assertIn("partial results", final["system_prompt"])
        self.assertIn("OCR-SLIDE", final["user_payload"])
        self.assertIn("NOTE-X", final["user_payload"])
        self.assertEqual(streamed, ["FINAL"])
        self.assertTrue(any("part 1 of" in status for status in statuses))
        self.assertTrue(any("final result" in status for status in statuses))

    def test_cancel_stops_between_parts(self) -> None:
        profile = _profile(context_tokens=2000, max_output_tokens=300)
        transcript = "\n".join(f"line {i} " + "word " * 12 for i in range(500))
        control = svc.LLMRunControl()
        calls: list[int] = []

        def fake_call(**kwargs):  # noqa: ANN003
            calls.append(1)
            control.request_cancel()
            return svc._CallOutcome(text="partial")

        with patch.object(svc, "_call_model", side_effect=fake_call):
            result = run_llm_postprocess(profile, _template(), LLMPostprocessRequest(transcript_text=transcript),
                                         run_control=control)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error_code, "cancelled")
        self.assertEqual(len(calls), 1)


class DialogAndListenerTests(unittest.TestCase):
    def test_listener_hides_cloud_profiles_unless_allowed(self) -> None:
        pytest = __import__("pytest")
        pytest.importorskip("gradio")
        import app
        from services.config_service import AppConfig

        profiles = [
            {"name": "local", "provider": "ollama", "scope": "local", "base_url": "http://127.0.0.1:11434"},
            {"name": "cloud", "provider": "anthropic", "scope": "cloud", "base_url": "https://api.anthropic.com",
             "api_key_runtime": "k", "cloud_acknowledged": True},
        ]
        for allow, expected in ((False, ["local"]), (True, ["local", "cloud"])):
            config = AppConfig(llm_profiles=profiles, llm_allow_cloud_in_listener=allow)
            with patch.object(app, "APP_CONFIG", config), patch.object(app, "_reload_listener_config"):
                self.assertEqual([p.name for p in app._enabled_llm_profiles()], expected)


class _RedirectHandler(http.server.BaseHTTPRequestHandler):
    seen_auth: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/start":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()
            return
        type(self).seen_auth.append(self.headers.get("x-api-key"))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        return


class RedirectTests(unittest.TestCase):
    def setUp(self) -> None:
        _RedirectHandler.seen_auth = []
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _RedirectHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def test_requests_with_credentials_do_not_follow_redirects(self) -> None:
        request = urlrequest.Request(self.base + "/start", headers={"x-api-key": "secret"})
        with self.assertRaises(HTTPError) as caught:
            open_url(request, timeout=5)
        self.assertEqual(caught.exception.code, 302)
        self.assertEqual(_RedirectHandler.seen_auth, [])

    def test_requests_without_credentials_still_follow_redirects(self) -> None:
        with open_url(urlrequest.Request(self.base + "/start"), timeout=5) as response:
            self.assertEqual(response.read(), b"ok")


if __name__ == "__main__":
    unittest.main()
