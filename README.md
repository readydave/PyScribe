# PyScribe

PyScribe is a simple GUI application for transcribing video files to text using [OpenAI Whisper](https://github.com/openai/whisper). The interface is built with `tkinter` and allows you to browse for a video, transcribe it and save the results.

## Features

- Supports common video formats: MP4, AVI, MOV, MKV, FLV
- Extracts audio with `ffmpeg`
- Uses Whisper for speech recognition
- Displays transcription in a scrollable text area
- Allows saving the transcription to a text file
- Progress indicator and status messages

## Requirements

- Python 3.8+
- `ffmpeg` available in your system path
- Python packages listed in `requirements.txt`

Install the dependencies with pip:

```bash
pip install -r requirements.txt
```

If you want to take advantage of GPU acceleration, make sure you install the appropriate version of PyTorch as described in the [Whisper documentation](https://github.com/openai/whisper#installation).

## Usage

Run the application with:

```bash
python pyscribe.py
```

1. Click **Browse** to select a video file.
2. Click **Transcribe** to start the transcription. The progress bar displays an estimated percentage as the audio is processed.
3. Once complete, the transcription text will be displayed. Click **Save Transcription** to store the text. The default filename uses the original video name with a `.txt` extension so transcripts stay next to their videos.

## Notes

- Large video files can take a while to process depending on your hardware.
- The application attempts to remove the temporary audio file after transcription.

## License

This project is licensed under the terms of the GNU GPLv3.
