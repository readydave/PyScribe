# PyScribe - Video to Text Transcription Tool
# Licensed under the GPLv3

import os
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import whisper


class PyScribeApp(tk.Tk):
    """Tkinter GUI application for transcribing video files using Whisper."""

    def __init__(self):
        super().__init__()
        self.title("PyScribe - Video Transcription")
        self.geometry("700x500")
        self._create_widgets()
        self.video_path = None
        self.transcription = None

    def _create_widgets(self):
        """Create and layout widgets."""
        # Frame for buttons
        button_frame = tk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        browse_btn = tk.Button(button_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT)

        transcribe_btn = tk.Button(button_frame, text="Transcribe", command=self.start_transcription)
        transcribe_btn.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(button_frame, text="Save Transcription", command=self.save_transcription)
        save_btn.pack(side=tk.LEFT)

        # Status label
        self.status_var = tk.StringVar(value="Select a video file to begin.")
        status_label = tk.Label(self, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=tk.X, padx=10)

        # Progress bar
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=10)

        # Text area for transcription
        self.text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD)
        self.text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

    def update_progress(self, percent):
        """Safely update the progress bar from a worker thread."""
        def _update():
            self.progress['value'] = percent
            self.status_var.set(f"Transcribing... {percent}%")

        self.after(0, _update)

    def browse_file(self):
        """Open a file dialog to select a video file."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select Video", filetypes=filetypes)
        if path:
            self.video_path = path
            self.status_var.set(f"Selected: {os.path.basename(path)}")

    def start_transcription(self):
        """Start transcription in a separate thread."""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return
        self.progress['value'] = 0
        self.status_var.set("Transcribing...")
        thread = threading.Thread(target=self.transcribe_video)
        thread.start()

    def transcribe_video(self):
        """Extract audio, transcribe it and display the result."""
        try:
            audio_file = self.extract_audio(self.video_path)
            model = whisper.load_model("base")

            audio = whisper.audio.load_audio(audio_file)
            total_seconds = audio.shape[0] / whisper.audio.SAMPLE_RATE
            segment_length = 30  # seconds
            step = int(segment_length * whisper.audio.SAMPLE_RATE)

            texts = []
            for i in range(0, len(audio), step):
                chunk = audio[i : i + step]
                result = model.transcribe(chunk, fp16=False)
                texts.append(result.get("text", ""))

                progress = min(100, int(((i + step) / len(audio)) * 100))
                self.update_progress(progress)

            self.transcription = "\n".join(texts).strip()
            self._update_text_area(self.transcription)
            self.update_progress(100)
            self.status_var.set("Transcription complete.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status_var.set("An error occurred during transcription.")
        finally:
            self.progress['value'] = 0
            if os.path.exists(audio_file):
                os.remove(audio_file)

    def _update_text_area(self, text):
        def _update():
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, text)

        self.after(0, _update)

    def extract_audio(self, video_path):
        """Use ffmpeg to extract audio from the video file."""
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")
        cmd = [
            "ffmpeg",
            "-y",  # overwrite without asking
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path

    def save_transcription(self):
        """Save the transcription to a text file."""
        if not self.transcription:
            messagebox.showwarning("Warning", "No transcription to save.")
            return
        default_name = "transcript.txt"
        if self.video_path:
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            default_name = base + ".txt"
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            initialfile=default_name,
                                            filetypes=[("Text files", "*.txt")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.transcription)
                self.status_var.set(f"Saved to {os.path.basename(path)}")
            except OSError as exc:
                messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    app = PyScribeApp()
    app.mainloop()
