"""Shared listener auth and non-local bind validation helpers."""

from __future__ import annotations

import getpass
import os
import sys


def clean_env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_listener_auth(auth_user: str | None) -> tuple[str | None, str | None]:
    resolved_user = (auth_user or "").strip() or clean_env_value("PYSCRIBE_AUTH_USER")
    resolved_pass = clean_env_value("PYSCRIBE_AUTH_PASS")
    if resolved_user and not resolved_pass and sys.stdin and sys.stdin.isatty():
        prompted = getpass.getpass("Listener auth password (input hidden): ").strip()
        resolved_pass = prompted or None
    if bool(resolved_user) != bool(resolved_pass):
        raise SystemExit(
            "Listener auth requires both username and password "
            "(provide --auth-user and set PYSCRIBE_AUTH_PASS, or set both "
            "PYSCRIBE_AUTH_USER/PYSCRIBE_AUTH_PASS)."
        )
    return resolved_user, resolved_pass


def is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def validate_listener_security(
    host: str,
    *,
    auth_user: str | None,
    auth_pass: str | None,
    allow_nonlocal_host: bool,
) -> None:
    if is_loopback_host(host):
        return
    if not allow_nonlocal_host:
        raise SystemExit(
            "Refusing non-local listener bind. Use --host 127.0.0.1 for local-only access, "
            "or add --allow-nonlocal-host to explicitly expose the listener."
        )
    if not (auth_user and auth_pass):
        raise SystemExit(
            "Non-local listener bind requires authentication. "
            "Provide --auth-user and set PYSCRIBE_AUTH_PASS, or set "
            "PYSCRIBE_AUTH_USER/PYSCRIBE_AUTH_PASS."
        )


def reject_legacy_auth_pass_flag(argv: list[str]) -> None:
    for arg in argv[1:]:
        if arg == "--auth-pass" or arg.startswith("--auth-pass="):
            raise SystemExit(
                "`--auth-pass` is no longer supported to avoid credential leakage. "
                "Set PYSCRIBE_AUTH_PASS instead."
            )
