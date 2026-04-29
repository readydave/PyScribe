# main.py
# Entry point for PyScribe Qt desktop and listener modes.

import argparse
import getpass
import logging
import os
import socket
import sys
import time
import warnings
from services.listener_security_service import (
    clean_env_value,
    reject_legacy_auth_pass_flag,
    resolve_listener_auth,
    validate_listener_security,
)
from services.logging_service import configure_logging
from services.runtime_compat import ensure_platform_sys_version_compat
from services.runtime_env_service import (
    configure_runtime_environment,
    reexec_if_loader_env_changed,
)

ensure_platform_sys_version_compat()
RUNTIME_ENV = configure_runtime_environment()
reexec_if_loader_env_changed()
LOG_PATH = configure_logging()
LOGGER = logging.getLogger(__name__)

# Reduce noisy ONNX Runtime warning logs (keep errors/fatal only).
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("ORT_LOG_VERBOSITY_LEVEL", "0")

# Silence noisy dependency warning from ctranslate2.
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="ctranslate2",
)
warnings.filterwarnings(
    "ignore",
    message=r".*pynvml package is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.(set|get)_audio_backend has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*speechbrain\.pretrained.*deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*weights_only=False.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*std\(\): degrees of freedom is <= 0.*",
    category=UserWarning,
)

INTERACTIVE_LAN_AUTH_USER = os.environ.get("PYSCRIBE_LAN_AUTH_USER", "pyscribe")
LAUNCHER_DEFAULT_CHOICE = "1"
LAUNCHER_DEFAULT_TIMEOUT_SECONDS = 5.0


def _redact_sensitive_argv(argv: list[str]) -> list[str]:
    """Return argv with secret values redacted for logging."""
    redacted: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--auth-pass":
            redacted.append(arg)
            if i + 1 < len(argv):
                redacted.append("********")
                i += 2
                continue
            i += 1
            continue
        if arg.startswith("--auth-pass="):
            redacted.append("--auth-pass=********")
            i += 1
            continue
        redacted.append(arg)
        i += 1
    return redacted


def _as_bool_env(name: str) -> bool:
    value = clean_env_value(name)
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def run_listener(
    host: str,
    port: int,
    max_port_tries: int,
    share: bool,
    queue_size: int,
    auth_user: str | None,
    auth_pass: str | None,
) -> None:
    """Launches the Gradio listener with automatic port fallback."""
    LOGGER.info(
        "Launching listener host=%s port=%s max_port_tries=%s share=%s queue_size=%s auth=%s",
        host,
        port,
        max_port_tries,
        share,
        queue_size,
        bool(auth_user and auth_pass),
    )
    if bool(auth_user) != bool(auth_pass):
        raise SystemExit("Both auth username and password must be provided together.")
    if share and not (auth_user and auth_pass):
        raise SystemExit("Listener share mode requires authentication credentials.")

    def _announce_listener(chosen_port: int) -> None:
        print("PyScribe listener running:")
        if host in {"0.0.0.0", "::"}:
            lan_ip = _resolve_lan_ip()
            print(f"  Local: http://127.0.0.1:{chosen_port}")
            print(f"  LAN:   http://{lan_ip}:{chosen_port}  (share this on your local network)")
        else:
            print(f"  URL:   http://{host}:{chosen_port}")

    from app import launch_listener

    launch_listener(
        host=host,
        port=port,
        max_tries=max_port_tries,
        share=share,
        queue_size=queue_size,
        auth_user=auth_user,
        auth_pass=auth_pass,
        on_start=_announce_listener,
    )


def _resolve_lan_ip() -> str:
    """Best-effort LAN IP for copy/share when binding to 0.0.0.0."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "localhost"


def run_qt() -> None:
    """Launches the PySide6 desktop UI."""
    LOGGER.info("Launching Qt desktop UI")
    from ui_qt import run_qt_app

    run_qt_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyScribe entry point.")
    subparsers = parser.add_subparsers(dest="mode")

    serve_parser = subparsers.add_parser("serve", help="Run Gradio listener mode")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=7860, help="Preferred port (default: 7860)")
    serve_parser.add_argument("--max-port-tries", type=int, default=50, help="How many fallback ports to try")
    serve_parser.add_argument("--queue-size", type=int, default=16, help="Max queued listener requests")
    serve_parser.add_argument("--auth-user", default=None, help="Optional basic-auth username")
    serve_parser.add_argument(
        "--allow-nonlocal-host",
        action="store_true",
        help="Allow binding to non-local interfaces (requires auth).",
    )
    serve_parser.add_argument("--share", action="store_true", help="Enable Gradio public share URL")
    serve_parser.set_defaults(mode="serve")

    qt_parser = subparsers.add_parser("qt", help="Run PySide6 desktop GUI")
    qt_parser.set_defaults(mode="qt")

    return parser.parse_args()


def _input_with_timeout(prompt: str, timeout_seconds: float) -> str | None:
    """Reads a line from stdin, returning None when a TTY prompt times out."""
    if timeout_seconds <= 0 or not sys.stdin or not sys.stdin.isatty():
        return input(prompt)

    print(prompt, end="", flush=True)
    if os.name == "nt":
        import msvcrt

        deadline = time.monotonic() + timeout_seconds
        chars: list[str] = []
        while time.monotonic() < deadline:
            if not msvcrt.kbhit():
                time.sleep(0.05)
                continue
            char = msvcrt.getwch()
            if char in {"\r", "\n"}:
                print()
                return "".join(chars)
            if char == "\003":
                raise KeyboardInterrupt
            if char in {"\b", "\x7f"}:
                if chars:
                    chars.pop()
                    print("\b \b", end="", flush=True)
                continue
            chars.append(char)
            print(char, end="", flush=True)
        print()
        return None

    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if ready:
        return sys.stdin.readline()
    print()
    return None


def prompt_launch_mode() -> str:
    """Interactive launcher menu shown when no CLI mode is provided."""
    LOGGER.info("Starting interactive launcher menu")
    print("\nPyScribe launcher")
    print("  1) Desktop (Qt)")
    print("  2) Listener (Gradio web, localhost only)")
    while True:
        raw_choice = _input_with_timeout(
            f"Choose mode [1/2] (default 1, auto-start Desktop in {int(LAUNCHER_DEFAULT_TIMEOUT_SECONDS)}s): ",
            LAUNCHER_DEFAULT_TIMEOUT_SECONDS,
        )
        if raw_choice is None:
            LOGGER.info("Interactive launcher timed out; selecting Desktop (Qt)")
            print("No selection received; starting Desktop (Qt).")
            return LAUNCHER_DEFAULT_CHOICE
        choice = raw_choice.strip() or LAUNCHER_DEFAULT_CHOICE
        if choice in {"1", "2"}:
            return choice
        print("Please enter 1 or 2.")


def prompt_listener_scope() -> str:
    """Interactive listener exposure menu for option 2 in launcher mode."""
    LOGGER.info("Starting interactive listener scope menu")
    print("\nListener mode")
    print("  1) Localhost only (127.0.0.1)")
    print(f"  2) LAN share (0.0.0.0, auth user '{INTERACTIVE_LAN_AUTH_USER}')")
    while True:
        choice = input("Choose listener scope [1/2] (default 1): ").strip() or "1"
        if choice == "1":
            return "local"
        if choice == "2":
            return "lan"
        print("Please enter 1 or 2.")


def _resolve_interactive_lan_auth() -> tuple[str, str]:
    user = clean_env_value("PYSCRIBE_LAN_AUTH_USER") or INTERACTIVE_LAN_AUTH_USER
    password = clean_env_value("PYSCRIBE_LAN_AUTH_PASS")
    if not password and sys.stdin and sys.stdin.isatty():
        prompted = getpass.getpass("LAN listener password (input hidden): ").strip()
        password = prompted or None
    if not password:
        raise SystemExit(
            "Interactive LAN listener requires a password. "
            "Set PYSCRIBE_LAN_AUTH_PASS or enter one when prompted."
        )
    return user, password


def main() -> None:
    safe_argv = _redact_sensitive_argv(sys.argv)
    LOGGER.info(
        "PyScribe startup argv=%s log_path=%s cache_root=%s",
        safe_argv,
        LOG_PATH,
        RUNTIME_ENV.get("cache_root"),
    )
    reject_legacy_auth_pass_flag(sys.argv)
    args = parse_args()
    if len(sys.argv) == 1:
        selected = prompt_launch_mode()
        if selected == "1":
            run_qt()
            return
        scope = prompt_listener_scope()
        if scope == "lan":
            auth_user, auth_pass = _resolve_interactive_lan_auth()
            validate_listener_security(
                "0.0.0.0",
                auth_user=auth_user,
                auth_pass=auth_pass,
                allow_nonlocal_host=True,
                share=False,
            )
            LOGGER.warning(
                "Interactive LAN listener enabled for user=%s",
                auth_user,
            )
            run_listener(
                host="0.0.0.0",
                port=7860,
                max_port_tries=50,
                share=False,
                queue_size=16,
                auth_user=auth_user,
                auth_pass=auth_pass,
            )
            return

        auth_user, auth_pass = resolve_listener_auth(None)
        validate_listener_security(
            "127.0.0.1",
            auth_user=auth_user,
            auth_pass=auth_pass,
            allow_nonlocal_host=False,
            share=False,
        )
        run_listener(
            host="127.0.0.1",
            port=7860,
            max_port_tries=50,
            share=False,
            queue_size=16,
            auth_user=auth_user,
            auth_pass=auth_pass,
        )
        return

    if args.mode == "serve":
        auth_user, auth_pass = resolve_listener_auth(args.auth_user)
        allow_nonlocal_host = bool(args.allow_nonlocal_host or _as_bool_env("PYSCRIBE_ALLOW_NONLOCAL_HOST"))
        validate_listener_security(
            args.host,
            auth_user=auth_user,
            auth_pass=auth_pass,
            allow_nonlocal_host=allow_nonlocal_host,
            share=bool(args.share),
        )
        run_listener(
            host=args.host,
            port=args.port,
            max_port_tries=args.max_port_tries,
            share=args.share,
            queue_size=args.queue_size,
            auth_user=auth_user,
            auth_pass=auth_pass,
        )
        return
    if args.mode == "qt":
        run_qt()
        return
    run_qt()


if __name__ == "__main__":
    main()
