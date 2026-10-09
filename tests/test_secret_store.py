from __future__ import annotations

import types
from unittest import mock

import pytest

from services import config_service, secret_store
from services.llm_connection_service import (
    evaluate_profile_scope_policy,
    load_llm_profiles,
    resolve_profile_api_key,
    run_connection_test,
)


class _PasswordDeleteError(Exception):
    pass


class FakeKeyring:
    """In-memory stand-in for the keyring module; the real OS keyring is never touched."""

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}
        self.calls = 0
        self.fail: Exception | None = None
        self.hang: float = 0.0
        self.errors = types.SimpleNamespace(PasswordDeleteError=_PasswordDeleteError)

    def _enter(self) -> None:
        self.calls += 1
        if self.hang:
            import time

            time.sleep(self.hang)
        if self.fail:
            raise self.fail

    def set_password(self, service: str, name: str, value: str) -> None:
        self._enter()
        self.data[(service, name)] = value

    def get_password(self, service: str, name: str) -> str | None:
        self._enter()
        return self.data.get((service, name))

    def delete_password(self, service: str, name: str) -> None:
        self._enter()
        if (service, name) not in self.data:
            raise _PasswordDeleteError("missing")
        del self.data[(service, name)]

    def get_keyring(self) -> object:
        return object()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    fk = FakeKeyring()
    monkeypatch.setattr(secret_store, "keyring", fk)
    monkeypatch.setattr(secret_store, "_keyring_fail", types.SimpleNamespace(Keyring=type("Fail", (), {})))
    return fk


def _cloud_raw(api_key: str) -> dict[str, object]:
    return {
        "name": "claude", "provider": "anthropic", "scope": "cloud", "base_url": "https://api.anthropic.com",
        "api_key": api_key, "cloud_acknowledged": True,
    }


def test_roundtrip_missing_entry_and_delete(fake: FakeKeyring) -> None:
    ref = secret_store.new_ref()
    ref_id = secret_store.ref_id_from(ref)
    assert ref_id and secret_store.get_key(ref_id) is None
    secret_store.set_key(ref_id, "sk-secret-value")
    assert secret_store.get_key(ref_id) == "sk-secret-value"
    secret_store.delete_key(ref_id)
    secret_store.delete_key(ref_id)  # already gone: not an error
    assert secret_store.get_key(ref_id) is None
    assert secret_store.ref_id_from("env:X") is None and secret_store.ref_id_from("keyring:") is None


def test_missing_package_raises_plain_error_and_never_plaintext(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secret_store, "keyring", None)
    assert secret_store.is_available() is False
    with pytest.raises(secret_store.SecretStoreError, match="keyring"):
        secret_store.set_key("abc", "sk-secret-value")
    with pytest.raises(secret_store.SecretStoreError):
        secret_store.get_key("abc")


def test_backend_failure_error_hides_key_and_ref(fake: FakeKeyring) -> None:
    fake.fail = RuntimeError("backend exploded for entry abc with sk-secret-value")
    with pytest.raises(secret_store.SecretStoreError) as info:
        secret_store.set_key("abc", "sk-secret-value")
    assert "sk-secret-value" not in str(info.value) and "abc" not in str(info.value)


def test_hanging_backend_times_out(fake: FakeKeyring, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secret_store, "CALL_TIMEOUT_SECONDS", 0.1)
    fake.hang = 0.5
    with pytest.raises(secret_store.SecretStoreError, match="did not respond"):
        secret_store.get_key("abc")


def test_sanitizer_keeps_keyring_and_env_refs_and_blanks_plain_keys() -> None:
    out = config_service._sanitize_llm_profiles_for_storage(
        [{"api_key": "keyring:abc123"}, {"api_key": "env:X"}, {"api_key": "sk-plain", "api_key_runtime": "s"}]
    )
    assert [p["api_key"] for p in out] == ["keyring:abc123", "env:X", ""]
    assert "api_key_runtime" not in out[2]


def test_loading_profiles_does_not_touch_the_keyring(fake: FakeKeyring) -> None:
    profile = load_llm_profiles([_cloud_raw("keyring:abc123")])[0]
    assert fake.calls == 0
    assert profile.api_key is None and profile.api_key_ref == "keyring:abc123"
    assert evaluate_profile_scope_policy(profile)[0] is True  # "key configured" without unlocking
    assert fake.calls == 0


def test_session_key_overrides_ref_and_env_ref_unchanged(fake: FakeKeyring) -> None:
    raw = _cloud_raw("keyring:abc123") | {"api_key_runtime": "session-key"}
    profile = load_llm_profiles([raw])[0]
    assert profile.api_key == "session-key" and profile.api_key_ref is None
    assert load_llm_profiles([_cloud_raw("env:NOPE_NOT_SET")])[0].api_key_ref is None


def test_resolve_reads_keyring_and_reports_missing_entry(fake: FakeKeyring) -> None:
    profile = load_llm_profiles([_cloud_raw("keyring:abc123")])[0]
    with pytest.raises(secret_store.SecretStoreError, match="No API key is stored"):
        resolve_profile_api_key(profile)
    secret_store.set_key("abc123", "sk-secret-value")
    assert resolve_profile_api_key(profile) == "sk-secret-value"


def test_connection_test_fails_cleanly_without_network_when_keyring_broken(fake: FakeKeyring) -> None:
    fake.fail = RuntimeError("locked")
    profile = load_llm_profiles([_cloud_raw("keyring:abc123")])[0]
    with mock.patch("services.llm_connection_service._http_json_get", side_effect=AssertionError("no request")):
        result = run_connection_test(profile)
    assert result.status == "fail"
    assert "keyring" in (result.failure_detail or "").lower()


def test_postprocess_call_with_broken_keyring_raises_auth_failed_and_sends_no_request(fake: FakeKeyring) -> None:
    pytest.importorskip("ffmpeg")  # llm_postprocess_service imports multimodal_service, which needs ffmpeg-python
    from services import llm_postprocess_service as post

    fake.fail = RuntimeError("locked")
    profile = load_llm_profiles([_cloud_raw("keyring:abc123")])[0]
    with mock.patch.object(post, "open_url", side_effect=AssertionError("no request")) as opened:
        with pytest.raises(post._LLMPostprocessException) as info:
            post._call_model(
                profile=profile, model="m", system_prompt="s", user_payload="u", image_paths=(),
                limits=None, on_output_chunk=None, run_control=None,
            )
    assert info.value.code == "auth_failed"
    assert "keyring" in str(info.value).lower() and "abc123" not in str(info.value)
    opened.assert_not_called()
