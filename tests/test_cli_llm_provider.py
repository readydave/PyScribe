from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from services import cli_llm_provider as cli

FAKE = """#!{python}
import json, os, sys, time
data = sys.stdin.read()
record = {{"argv": sys.argv[1:], "cwd": os.getcwd(), "cwd_files": os.listdir("."), "env": sorted(os.environ), "stdin": data}}
open({record!r}, "w").write(json.dumps(record))
if "MODE:auth" in data:
    sys.stderr.write("Error: Not logged in. Please run /login\\n"); sys.exit(1)
if "MODE:rate" in data:
    sys.stdout.write("Usage limit reached"); sys.exit(1)
if "MODE:fail" in data:
    sys.stderr.write("secret stderr /home/dave/.ssh/id_rsa"); sys.exit(2)
if "MODE:sleep" in data:
    time.sleep(30)
if "MODE:big" in data:
    sys.stdout.write("x" * 5000)
    sys.stdout.flush()
    time.sleep(30)
if "MODE:empty" in data:
    sys.exit(0)
print("RESULT OK")
"""


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    record = tmp_path / "record.json"
    script = bindir / "claude"
    script.write_text(FAKE.format(python=sys.executable, record=str(record)))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    return record


def _record(path: Path) -> dict:
    return json.loads(path.read_text())


def test_success_uses_safe_flags_stdin_empty_cwd_and_clean_env(fake_claude: Path) -> None:
    prompt = cli.build_cli_prompt("Summarize.", "Transcript line with $(rm -rf /) and MODE:ok")
    result = cli.run_claude(prompt, model="sonnet", timeout_seconds=20)
    assert result.text == "RESULT OK"
    rec = _record(fake_claude)
    argv = rec["argv"]
    assert argv[:1] == ["-p"] and "--tools" in argv and argv[argv.index("--tools") + 1] == ""
    for flag in ("--strict-mcp-config", "--safe-mode", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert "--bare" not in argv
    assert not any("Transcript line" in a or "Summarize" in a for a in argv)  # prompt only on stdin
    assert "Transcript line" in rec["stdin"] and "untrusted" in rec["stdin"]
    assert rec["cwd_files"] == [] and "pyscribe-cli-" in rec["cwd"]
    assert "ANTHROPIC_API_KEY" not in rec["env"] and "OPENAI_API_KEY" not in rec["env"]
    assert not Path(rec["cwd"]).exists()  # temp dir removed afterwards


def test_default_model_adds_no_model_flag(fake_claude: Path) -> None:
    cli.run_claude("hi", model=cli.DEFAULT_MODEL, timeout_seconds=20)
    assert "--model" not in _record(fake_claude)["argv"]


def test_not_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(cli.CliError) as info:
        cli.run_claude("hi")
    assert info.value.code == "cli_not_installed"


@pytest.mark.parametrize(
    ("mode", "code"), [("MODE:auth", "auth_failed"), ("MODE:rate", "rate_limited"), ("MODE:fail", "cli_failed"), ("MODE:empty", "cli_failed")]
)
def test_failures_map_to_codes_without_echoing_stderr(fake_claude: Path, mode: str, code: str) -> None:
    with pytest.raises(cli.CliError) as info:
        cli.run_claude(mode, timeout_seconds=20)
    assert info.value.code == code
    assert ".ssh" not in info.value.detail and "secret" not in info.value.detail


def test_timeout_kills_process(fake_claude: Path) -> None:
    started = time.monotonic()
    with pytest.raises(cli.CliError) as info:
        cli.run_claude("MODE:sleep", timeout_seconds=1.0)
    assert info.value.code == "timeout" and time.monotonic() - started < 10


def test_cancel_kills_process(fake_claude: Path) -> None:
    flag = threading.Event()
    threading.Timer(0.7, flag.set).start()
    started = time.monotonic()
    with pytest.raises(cli.CliError) as info:
        cli.run_claude("MODE:sleep", timeout_seconds=30, cancel_check=flag.is_set)
    assert info.value.code == "cancelled" and time.monotonic() - started < 10


def test_output_cap(fake_claude: Path) -> None:
    with pytest.raises(cli.CliError, match="more output"):
        cli.run_claude("MODE:big", timeout_seconds=20, max_output_bytes=1000)


def test_env_allowlist_excludes_api_keys() -> None:
    env = cli.build_env({"PATH": "/bin", "HOME": "/h", "ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "t"})
    assert env == {"PATH": "/bin", "HOME": "/h"}


# --- plumbing: profile parsing, policy, connection test, post-processing -------------------------------------

import dataclasses  # noqa: E402

from services import llm_postprocess_service as post  # noqa: E402
from services.llm_connection_service import (  # noqa: E402
    evaluate_profile_scope_policy,
    load_llm_profiles,
    run_connection_test,
)


def _profile(**extra: object):  # noqa: ANN202
    raw = {"name": "my-claude", "provider": "claude_cli", "scope": "local", "base_url": "http://127.0.0.1:9",
           "api_key": "sk-ignored", "cloud_acknowledged": True, **extra}
    return load_llm_profiles([raw])[0]


def test_cli_profile_is_forced_to_cloud_and_ignores_url_and_key() -> None:
    profile = _profile()
    assert (profile.scope, profile.base_url, profile.api_key, profile.api_key_ref) == ("cloud", "", None, None)
    assert profile.default_model == cli.DEFAULT_MODEL and profile.verify_tls is True
    assert load_llm_profiles([{"name": "x", "provider": "claude_cli"}])  # no base_url needed


def test_cli_policy_needs_acknowledgement_and_rejects_any_endpoint() -> None:
    assert evaluate_profile_scope_policy(_profile())[0] is True
    assert evaluate_profile_scope_policy(_profile(cloud_acknowledged=False))[1] == "policy_cloud_not_acknowledged"
    tampered = dataclasses.replace(_profile(), base_url="https://evil.example")
    assert evaluate_profile_scope_policy(tampered)[0] is False
    local = dataclasses.replace(_profile(), scope="local")
    assert evaluate_profile_scope_policy(local)[0] is False


def test_connection_test_pass_and_failures(fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ok = run_connection_test(_profile())
    assert ok.status == "pass" and [s.stage for s in ok.stages] == ["scope_policy", "binary", "inference_smoke"]
    assert run_connection_test(_profile(cloud_acknowledged=False)).failure_code == "policy_cloud_not_acknowledged"
    monkeypatch.setattr(cli, "run_claude", lambda *a, **k: (_ for _ in ()).throw(cli.CliError("auth_failed", "not signed in")))
    failed = run_connection_test(_profile())
    assert failed.status == "fail" and failed.failure_code == "auth_failed"


def test_connection_test_cli_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    result = run_connection_test(_profile())
    assert result.status == "fail" and result.failure_code == "cli_not_installed"


def test_postprocess_call_uses_cli_and_streams_once(fake_claude: Path) -> None:
    chunks: list[str] = []
    outcome = post._call_model(
        profile=_profile(), model="default", system_prompt="Summarize.", user_payload="the transcript",
        image_paths=(), limits=None, on_output_chunk=chunks.append, run_control=None,
    )
    assert outcome.text == "RESULT OK" and chunks == ["RESULT OK"]
    rec = _record(fake_claude)
    assert "the transcript" in rec["stdin"] and "--model" not in rec["argv"]


def test_postprocess_maps_cli_errors_to_exception_codes(fake_claude: Path) -> None:
    with pytest.raises(post._LLMPostprocessException) as info:
        post._call_model(
            profile=_profile(), model="default", system_prompt="s", user_payload="MODE:rate",
            image_paths=(), limits=None, on_output_chunk=None, run_control=None,
        )
    assert info.value.code == "rate_limited"


@pytest.mark.parametrize("bad", ["--dangerously-skip-permissions", "-x", "a b", "a;rm", "$(x)", "x" * 101, "m\nn"])
def test_invalid_model_names_are_rejected_before_running(fake_claude: Path, bad: str) -> None:
    with pytest.raises(cli.CliError, match="model name") as info:
        cli.run_claude("hi", model=bad, timeout_seconds=20)
    assert info.value.code == "cli_failed" and not fake_claude.exists()


@pytest.mark.parametrize("good", ["sonnet", "claude-sonnet-5-5", "opus[1m]", "claude-opus-4.1:beta"])
def test_valid_model_names_are_passed(fake_claude: Path, good: str) -> None:
    cli.run_claude("hi", model=good, timeout_seconds=20)
    argv = _record(fake_claude)["argv"]
    assert argv[argv.index("--model") + 1] == good
