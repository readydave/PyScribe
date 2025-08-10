# main.py
# Entry point for the PyScribe application.

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

