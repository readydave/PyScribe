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

## Choosing a Model

The application provides two types of models. Choosing the right one depends on your audio's language and your hardware.

### Standard (Multilingual) Models

These are the general-purpose models available in the first dropdown menu.

-   **Use Case:** These models are trained on many different languages. **You should use these for any audio that is not in English**, or if you are unsure of the language. The application's language detector relies on these models.

### Custom (English-Only) Models

These specialized models are available in the "Custom Model (HF)" dropdown.

-   **Use Case:** If you know your audio is **only in English**, using one of these models (e.g., `Systran/faster-whisper-small.en`) is often slightly **faster and more accurate**. They don't have the overhead of processing other languages.

### Model Hardware Requirements

Larger models are more accurate but require more GPU VRAM. Here are the recommended minimums for GPU users:

| Model Size | Required VRAM | Speed     | Accuracy |
| :--------- | :------------ | :-------- | :------- |
| `tiny`     | ~1 GB         | Fastest   | Low      |
| `base`     | ~1.5 GB       | Very Fast | Fair     |
| `small`    | ~2.5 GB       | Fast      | Good     |
| `medium`   | ~5 GB         | Medium    | High     |
| `large-v2` | ~8 GB         | Slow      | Highest  |
| `large-v3` | ~8 GB         | Slow      | Highest  |

---

## Requirements

- **Python 3.12 (Recommended):** This version is confirmed to be compatible with the required libraries.
- **FFmpeg:** Must be installed and available in your system's PATH. You can install it easily on Windows with `winget install Gyan.FFmpeg`.

---

## Installation (One-Time Setup for Windows)

This guide explains how to set up the project using an external virtual environment.

1.  **Install FFmpeg:** Open a terminal and run: `winget install Gyan.FFmpeg`
2.  **Download and Extract:** Download the project ZIP from GitHub and extract it to a folder (e.g., `C:\Code\PyScribe-main`).
3.  **Create an Environments Folder:** Create a central folder for your virtual environments, for example, `C:\Code\_envs`.
4.  **Create the Virtual Environment:** In a terminal, run the following command:
    ```bash
    py -3.12 -m venv C:\Code\_envs\pyscribe
    ```
5.  **Activate the Environment:**
    ```bash
    C:\Code\_envs\pyscribe\Scripts\activate
    ```
6.  **Navigate to Project Folder:** In the same terminal, change to your project directory.
    ```bash
    cd C:\Code\PyScribe-main
    ```
7.  **Install Dependencies:** Choose one of the following two paths.

    ---
    ### For Users with NVIDIA GPUs (Recommended)
    **A. Install GPU-Enabled PyTorch:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
    **B. Install Remaining Packages:**
    ```bash
    pip install -r requirements.txt
    ```
    ---
    ### For Users without GPUs (CPU-Only)
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Simply double-click the **`launch.bat`** file inside the project folder.

---

## Troubleshooting

-   **`ModuleNotFoundError` on launch:** The required packages were not installed. Activate your virtual environment and run `pip install -r requirements.txt`.
-   **App runs in "CPU Mode" on an NVIDIA system:** The CPU-only version of PyTorch is installed. To fix, activate your venv and run:
    ```bash
    pip uninstall torch
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

---

## License

This project is licensed under the terms of the GNU GPLv3.
