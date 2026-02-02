# main.py
# Entry point for PyScribe desktop and listener modes.

import argparse
import sys
import warnings

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

    args = parser.parse_args()
    if not args.mode:
        args.mode = "desktop"
    return args


def main():
    args = parse_args()
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
