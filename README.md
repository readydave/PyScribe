# PyScribe - Local Transcription GUI

PyScribe is a modern, Windows-friendly GUI application for fast, local audio/video transcription using the powerful `faster-whisper` library. It's designed to provide a seamless and efficient transcription workflow, running entirely on your own hardware for maximum privacy and performance.

![PyScribe Main Interface](./images/2025-08-11_09-36-47.png)
![Standard Model Selection](./images/2025-08-11_09-37-06.png)

*Users can choose between standard multilingual models or specialized English-only models from Hugging Face.*

---

## Key Features

- **High-Speed Transcription:** Powered by `faster-whisper` for significant speed improvements over the original Whisper, especially on GPUs.
- **Hardware-Aware:** Automatically detects your GPU (NVIDIA/CUDA) or CPU and recommends the optimal model size for your hardware.
- **Live Progress & Transcription:** Watch the transcription appear in real-time and monitor progress with a live percentage bar.
- **Audio Playback & Cancellation:** Preview your audio files with a built-in player (Play/Stop) and cancel a transcription mid-process if it's taking too long.
- **Automatic Language Detection:** Detects the language of the audio and offers to translate to English if a non-English language is found.
- **Flexible Model Selection:**
    - Choose from standard Whisper models (`tiny`, `base`, `small`, `medium`, `large-v3`).
    - Select from a curated list of fine-tuned, `faster-whisper`-compatible models from Hugging Face.
    - Automatically caches and lists previously downloaded custom models for easy reuse.
- **Detailed Reporting:** Get a full summary upon completion, including time taken, detected language, and the model used.
- **Smart File Naming:** Automatically saves transcripts with a detailed, sortable filename that includes the timestamp and model name, perfect for A/B testing models.
- **User-Friendly Setup:** Automatically checks for missing dependencies and offers to install them on the first run.

---

## Requirements

- Python 3.9+
- **NVIDIA GPU (Recommended):** For significantly faster performance.
- **FFmpeg:** Must be installed and available in your system's PATH. You can install it easily on Windows with `winget install Gyan.FFmpeg`.

---

## Installation (One-Time Setup)

Follow these steps to set up the application for the first time.

1.  **Download and Extract:** Download the project ZIP from GitHub and extract it to a folder on your computer.

2.  **Create Virtual Environment:** Open a terminal (PowerShell or Command Prompt) inside the project folder and run the following command. **Using the name `.venv` is required for the `launch.bat` shortcut to work.**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Environment:**
    ```bash
    .\.venv\Scripts\activate
    ```

4.  **Install Dependencies:** Now, install the required packages.
    * **(Recommended for NVIDIA GPUs)** First, install the correct PyTorch version:
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        ```
    * Then, install the rest of the packages from the requirements file:
        ```bash
        pip install -r requirements.txt
        ```

---

## Usage

-   **For Regular Use:** Simply double-click the **`launch.bat`** file in the project folder.
-   **For Development:** Activate your virtual environment (`.\.venv\Scripts\activate`) and run `python main.py` from the terminal.

---

## License

This project is licensed under the terms of the GNU GPLv3.
