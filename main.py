# main.py
# Entry point for PyScribe desktop and listener modes.

import argparse
import os
import sys
import warnings

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


def run_desktop():
    """Checks desktop dependencies and launches the Tk GUI."""
    from utils import check_and_install_dependencies

    if not check_and_install_dependencies():
        sys.exit(1)

    from ui import PyScribeApp

    app = PyScribeApp()
    app.mainloop()


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
    if bool(auth_user) != bool(auth_pass):
        raise SystemExit("Both --auth-user and --auth-pass must be provided together.")

    from app import launch_listener

    chosen_port = launch_listener(
        host=host,
        port=port,
        max_tries=max_port_tries,
        share=share,
        queue_size=queue_size,
        auth_user=auth_user,
        auth_pass=auth_pass,
    )
    print(f"PyScribe listener running on http://{host}:{chosen_port}")


def run_qt():
    """Launches the PySide6 desktop UI."""
    from ui_qt import run_qt_app

    run_qt_app()


def parse_args():
    parser = argparse.ArgumentParser(description="PyScribe entry point.")
    parser.add_argument(
        "--gui",
        choices=["tk", "qt"],
        help="Quick GUI selector (equivalent to desktop/qt subcommands).",
    )
    subparsers = parser.add_subparsers(dest="mode")

    desktop_parser = subparsers.add_parser("desktop", help="Run local desktop GUI (default)")
    desktop_parser.set_defaults(mode="desktop")

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
    print("\nPyScribe launcher")
    print("  1) Desktop (Tk)")
    print("  2) Desktop (Qt)")
    print("  3) Listener (Gradio web)")
    while True:
        choice = input("Choose mode [1/2/3] (default 1): ").strip() or "1"
        if choice in {"1", "2", "3"}:
            return choice
        print("Please enter 1, 2, or 3.")


def main():
    args = parse_args()
    if len(sys.argv) == 1:
        selected = prompt_launch_mode()
        if selected == "1":
            run_desktop()
            return
        if selected == "2":
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

    if args.gui == "qt":
        run_qt()
        return
    if args.gui == "tk":
        run_desktop()
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
    run_desktop()


if __name__ == "__main__":
    main()
