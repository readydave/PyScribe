# ui.py
# The user interface for the PyScribe application.

import os
import sys
import threading
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import torch
from utils import get_available_hf_models
# The line "from transcriber import run_transcription" is removed from here.

# --- Dynamic Base Class for Theming ---
try:
    from ttkthemes import ThemedTk
    BaseAppClass = ThemedTk
except ImportError:
    BaseAppClass = tk.Tk

# --- Constants ---
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

class PyScribeApp(BaseAppClass):
    """Tkinter GUI app for transcription using faster-whisper."""

    def __init__(self):
        if issubclass(BaseAppClass, tk.Tk) and BaseAppClass != tk.Tk:
            super().__init__(theme="arc")
        else:
            super().__init__()

        self.title("PyScribe - Faster-Whisper GUI")
        self.geometry("900x650")

        # --- App State ---
        self.media_path: str | None = None
        self.transcription: str | None = None
        self.model = None
        self.current_model_name: str | None = None
        self.is_waiting_for_user = False
        self.choice_event = threading.Event()
        self.user_task_choice: str | None = None

        # --- Hardware Info ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "N/A"
        self.vram_gb = 0
        if self.device == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                self.vram_gb = round(props.total_memory / (1024 ** 3), 1)
            except Exception:
                pass
        self.cpu_count = os.cpu_count() or 1
        self.recommended_model = self._recommend_model()

        # --- UI Setup ---
        self._create_widgets()
        self._show_startup_status()
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _create_widgets(self):
        """Creates and lays out all the GUI widgets."""
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=6)

        tk.Button(top_frame, text="Browse File", command=self.browse_file).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(top_frame, text="Model:").pack(side=tk.LEFT, padx=(10, 2))
        self.model_var = tk.StringVar(value=self.recommended_model)
        self.model_combo = ttk.Combobox(
            top_frame, textvariable=self.model_var, values=MODEL_CHOICES, width=12, state="readonly"
        )
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind("<<ComboboxSelected>>", self._clear_hf_model)

        ttk.Label(top_frame, text="or Custom Model (HF):").pack(side=tk.LEFT, padx=(10, 2))
        self.hf_model_var = tk.StringVar()
        self.hf_model_combo = ttk.Combobox(
            top_frame, textvariable=self.hf_model_var, values=get_available_hf_models(), width=35
        )
        self.hf_model_combo.pack(side=tk.LEFT)
        self.hf_model_combo.bind("<<ComboboxSelected>>", self._clear_standard_model)

        self.transcribe_btn = tk.Button(top_frame, text="Transcribe", command=self.start_transcription, font=("Segoe UI", 9, "bold"))
        self.transcribe_btn.pack(side=tk.LEFT, padx=15)
        
        second_frame = tk.Frame(self)
        second_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.save_btn = tk.Button(second_frame, text="Save Transcript", command=self.save_transcription, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.copy_btn = tk.Button(second_frame, text="Copy", command=self.copy_to_clipboard, state=tk.DISABLED)
        self.copy_btn.pack(side=tk.LEFT, padx=6)

        self.open_btn = tk.Button(second_frame, text="Open Save Folder", command=self.open_transcriptions_folder, state=tk.NORMAL)
        self.open_btn.pack(side=tk.LEFT, padx=6)
        
        tk.Button(top_frame, text="Exit", command=self.on_exit).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Select an audio/video file to begin.")
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=10)
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=("Segoe UI", 10))
        self.text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=(0, 8))

    def _show_startup_status(self):
        """Displays hardware info and recommended model on startup."""
        info = f"GPU: {self.gpu_name} ({self.vram_gb} GB VRAM)" if self.device == "cuda" else f"CPU Mode ({self.cpu_count} cores)"
        self.status_var.set(f"{info} | Recommended model: {self.recommended_model}")

    def _recommend_model(self) -> str:
        """Recommends a Whisper model based on available hardware."""
        if self.device == "cuda":
            if self.vram_gb >= 10: return "large-v3"
            if self.vram_gb >= 8: return "large-v2"
            if self.vram_gb >= 5: return "medium"
            if self.vram_gb >= 3: return "small"
            return "base"
        if self.cpu_count >= 12: return "small"
        if self.cpu_count >= 8: return "base"
        return "tiny"

    def start_transcription(self):
        """Validates inputs and starts the transcription thread."""
        # --- FIX: Import locally to prevent circular dependency ---
        from transcriber import run_transcription

        if self.is_waiting_for_user:
            messagebox.showinfo("Info", "Please respond to the language prompt first.")
            return
        if not self.media_path:
            messagebox.showerror("Error", "Please select a file first.")
            return

        model_to_use = self.hf_model_var.get().strip() or self.model_var.get().strip()
        if not model_to_use:
            messagebox.showerror("Error", "Please select a model.")
            return

        if model_to_use in MODEL_CHOICES:
            try:
                selected_idx = MODEL_CHOICES.index(model_to_use)
                recommended_idx = MODEL_CHOICES.index(self.recommended_model)
                if selected_idx > recommended_idx + 1:
                    if not messagebox.askokcancel("Hardware Warning", f"The selected model '{model_to_use}' may be too large for your hardware.\n\nContinue?"):
                        return
            except ValueError:
                pass 

        self._toggle_busy(True)
        # The UI thread starts the transcriber logic in a separate thread
        thread = threading.Thread(target=run_transcription, args=(self, model_to_use, self.media_path), daemon=True)
        thread.start()

    # --- Callbacks for Transcriber ---
    def set_status(self, msg: str):
        self.after(0, lambda: self.status_var.set(msg))
        
    def update_progress(self, value: float):
        self.after(0, lambda: self.progress.config(value=value))

    def update_text_area(self, text: str):
        self.after(0, lambda: (self.text_area.delete(1.0, tk.END), self.text_area.insert(tk.END, text)))

    def show_error(self, msg: str):
        self.after(0, lambda: messagebox.showerror("Error", msg))

    def enable_save_buttons(self):
        self.after(0, lambda: (self.save_btn.config(state=tk.NORMAL), self.copy_btn.config(state=tk.NORMAL)))

    def finish_transcription_flow(self):
        self.is_waiting_for_user = False
        self._toggle_busy(False)

    def refresh_hf_model_list(self):
        self.after(0, lambda: self.hf_model_combo.config(values=get_available_hf_models()))

    def prompt_for_translation(self, lang_code: str):
        self.is_waiting_for_user = True
        self.choice_event.clear()
        self.after(0, self._ask_translation_prompt, lang_code)
        self._toggle_busy(False, keep_progress=True)
        self.choice_event.wait() # Wait for user to make a choice
        self._toggle_busy(True)

    def _ask_translation_prompt(self, lang_code: str):
        translate = messagebox.askyesno(
            "Language Detected",
            f"Detected language: '{lang_code}'.\n\nTranslate to English?"
        )
        self.user_task_choice = "translate" if translate else "transcribe"
        self.is_waiting_for_user = False
        self.choice_event.set()

    # --- UI Event Handlers ---
    def browse_file(self):
        filetypes = [("Audio/Video", " ".join(f"*{ext}" for ext in sorted(list(MODEL_CHOICES))))]
        path = filedialog.askopenfilename(title="Select Media File", filetypes=filetypes)
        if path:
            self.media_path = path
            self.status_var.set(f"Selected: {os.path.basename(path)}")
            self.save_btn.config(state=tk.DISABLED)
            self.copy_btn.config(state=tk.DISABLED)
            self.text_area.delete(1.0, tk.END)
            self.transcription = None

    def save_transcription(self):
        if not self.transcription or not self.media_path:
            return
        
        model_name_used = self.current_model_name or "unknown"
        safe_model_name = f"HF-{model_name_used.replace('/', '-')}" if "/" in model_name_used else model_name_used

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.media_path))[0]
        folder = os.path.dirname(self.media_path)
        save_path = os.path.join(folder, f"{base_name}_{ts}_{safe_model_name}.txt")
        
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(self.transcription)
            self.set_status(f"Saved to {os.path.basename(save_path)}")
        except OSError as e:
            self.show_error(f"Could not save file:\n\n{e}")

    def copy_to_clipboard(self):
        if self.transcription:
            self.clipboard_clear()
            self.clipboard_append(self.transcription)
            self.set_status("Transcript copied to clipboard.")

    def open_transcriptions_folder(self):
        folder = os.path.dirname(self.media_path) if self.media_path else os.path.expanduser("~")
        try:
            os.startfile(folder)
        except Exception as e:
            self.show_error(f"Could not open folder:\n\n{e}")

    def _clear_hf_model(self, event=None):
        self.hf_model_var.set("")

    def _clear_standard_model(self, event=None):
        self.model_var.set("")

    def _toggle_busy(self, busy: bool, keep_progress: bool = False):
        self.transcribe_btn.config(state=tk.DISABLED if busy else tk.NORMAL)
        if not keep_progress:
            self.progress.config(value=0)

    def on_exit(self):
        self.destroy()
