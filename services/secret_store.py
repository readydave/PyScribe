"""OS keyring storage for LLM API keys (optional ``keyring`` dependency).

A profile stores the reference ``keyring:<id>`` in its ``api_key`` field; the key itself lives only in the OS
keyring under service ``pyscribe``. There is never a fallback to plaintext: if the keyring is missing, locked or
broken, callers get a ``SecretStoreError`` with a plain-language message that contains no key and no ref id.

Keyring calls can block (D-Bus, KWallet/GNOME unlock prompts), so each call runs on a daemon thread with a
timeout. Callers on a GUI thread must still call these from a worker thread.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Callable

try:  # optional dependency
    import keyring
    from keyring.backends import fail as _keyring_fail
except Exception:  # missing package or a broken install
    keyring = None  # type: ignore[assignment]
    _keyring_fail = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

SERVICE_NAME = "pyscribe"
REF_PREFIX = "keyring:"
CALL_TIMEOUT_SECONDS = 5.0

_MISSING_MSG = "Secure key storage is not available. Install the optional 'keyring' package, or use env:NAME or a session key."
_BACKEND_MSG = "The system keyring could not be used (is it unlocked and running?). Use env:NAME or a session key instead."
_TIMEOUT_MSG = "The system keyring did not respond in time. Unlock it and try again, or use env:NAME or a session key."


class SecretStoreError(Exception):
    """Plain-language failure; never contains a key or a ref id."""


def new_ref() -> str:
    """A fresh opaque reference such as ``keyring:3f2a...``; renaming a profile keeps it."""
    return f"{REF_PREFIX}{uuid.uuid4().hex}"


def ref_id_from(value: object) -> str | None:
    """The id part of a ``keyring:<id>`` reference, or None if ``value`` is not one."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.lower().startswith(REF_PREFIX):
        ref_id = text[len(REF_PREFIX):].strip()
        return ref_id or None
    return None


def is_available() -> bool:
    """True when the package imports and a real (non-failing) backend is selected. May touch the backend."""
    if keyring is None:
        return False
    try:
        backend = keyring.get_keyring()
    except Exception:
        return False
    return not isinstance(backend, _keyring_fail.Keyring)


def _call(fn: Callable[[], Any]) -> Any:
    if keyring is None:
        raise SecretStoreError(_MISSING_MSG)
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # reported below without its text (may name the entry)
            outcome["error"] = exc

    thread = threading.Thread(target=run, name="pyscribe-keyring", daemon=True)
    thread.start()
    thread.join(CALL_TIMEOUT_SECONDS)
    if thread.is_alive():
        LOGGER.warning("Keyring call timed out after %ss", CALL_TIMEOUT_SECONDS)
        raise SecretStoreError(_TIMEOUT_MSG)
    if "error" in outcome:
        LOGGER.warning("Keyring call failed: %s", type(outcome["error"]).__name__)
        raise SecretStoreError(_BACKEND_MSG)
    return outcome.get("value")


def set_key(ref_id: str, key: str) -> None:
    ref_id, key = str(ref_id or "").strip(), str(key or "")
    if not ref_id or not key.strip():
        raise SecretStoreError("There is no key to store.")
    _call(lambda: keyring.set_password(SERVICE_NAME, ref_id, key))


def get_key(ref_id: str) -> str | None:
    """The stored key, or None when no entry exists."""
    ref_id = str(ref_id or "").strip()
    if not ref_id:
        return None
    value = _call(lambda: keyring.get_password(SERVICE_NAME, ref_id))
    return value if isinstance(value, str) and value.strip() else None


def delete_key(ref_id: str) -> None:
    """Remove an entry; a missing entry is not an error. Call only on an explicit user action."""
    ref_id = str(ref_id or "").strip()
    if not ref_id:
        return

    def run() -> None:
        try:
            keyring.delete_password(SERVICE_NAME, ref_id)
        except keyring.errors.PasswordDeleteError:
            pass

    _call(run)
