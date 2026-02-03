# main.py
# Entry point for PyScribe Qt desktop and listener modes.

import argparse
import logging
import os
import socket
import sys
import warnings
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


def run_listener(
    host: str,
    port: int,
    max_port_tries: int,
    share: bool,
    queue_size: int,
    auth_user: str | None,
    auth_pass: str | None,
):
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
        raise SystemExit("Both --auth-user and --auth-pass must be provided together.")

    def _announce_listener(chosen_port: int):
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


def run_qt():
    """Launches the PySide6 desktop UI."""
    LOGGER.info("Launching Qt desktop UI")
    from ui_qt import run_qt_app

    run_qt_app()


def parse_args():
    parser = argparse.ArgumentParser(description="PyScribe entry point.")
    subparsers = parser.add_subparsers(dest="mode")

    serve_parser = subparsers.add_parser("serve", help="Run Gradio listener mode")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=7860, help="Preferred port (default: 7860)")
    serve_parser.add_argument("--max-port-tries", type=int, default=50, help="How many fallback ports to try")
    serve_parser.add_argument("--queue-size", type=int, default=16, help="Max queued listener requests")
    serve_parser.add_argument("--auth-user", default=None, help="Optional basic-auth username")
    serve_parser.add_argument("--auth-pass", default=None, help="Optional basic-auth password")
    serve_parser.add_argument("--share", action="store_true", help="Enable Gradio public share URL")
    serve_parser.set_defaults(mode="serve")

    qt_parser = subparsers.add_parser("qt", help="Run PySide6 desktop GUI")
    qt_parser.set_defaults(mode="qt")

    return parser.parse_args()


def prompt_launch_mode() -> str:
    """Interactive launcher menu shown when no CLI mode is provided."""
    LOGGER.info("Starting interactive launcher menu")
    print("\nPyScribe launcher")
    print("  1) Desktop (Qt)")
    print("  2) Listener (Gradio web)")
    while True:
        choice = input("Choose mode [1/2] (default 1): ").strip() or "1"
        if choice in {"1", "2"}:
            return choice
        print("Please enter 1 or 2.")


def main():
    LOGGER.info(
        "PyScribe startup argv=%s log_path=%s cache_root=%s",
        sys.argv,
        LOG_PATH,
        RUNTIME_ENV.get("cache_root"),
    )
    args = parse_args()
    if len(sys.argv) == 1:
        selected = prompt_launch_mode()
        if selected == "1":
            run_qt()
            return
        run_listener(
            host="0.0.0.0",
            port=7860,
            max_port_tries=50,
            share=False,
            queue_size=16,
            auth_user=None,
            auth_pass=None,
        )
        return

    if args.mode == "serve":
        run_listener(
            host=args.host,
            port=args.port,
            max_port_tries=args.max_port_tries,
            share=args.share,
            queue_size=args.queue_size,
            auth_user=args.auth_user,
            auth_pass=args.auth_pass,
        )
        return
    if args.mode == "qt":
        run_qt()
        return
    run_qt()


if __name__ == "__main__":
    main()
