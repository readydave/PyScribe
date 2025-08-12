# main.py
# Entry point for the PyScribe application.

# --- FIX: Silence the noisy pkg_resources deprecation warning ---
# This warning comes from a dependency of faster-whisper (ctranslate2) and
# can be safely ignored. This filter prevents it from cluttering the terminal.
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="ctranslate2"
)

import sys
from utils import check_and_install_dependencies
from ui import PyScribeApp

def main():
    """
    Checks dependencies and launches the main application window.
    """
    # First, check if all required packages are installed.
    # The application will exit if they are not and the user declines to install.
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # Launch the GUI
    app = PyScribeApp()
    app.mainloop()

if __name__ == "__main__":
    main()
