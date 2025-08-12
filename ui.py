# ui.py
# This module contains the user interface for the PyScribe application,
# built with tkinter. It handles window creation, widget layout, and user events.

import os
import sys
import threading
import datetime
import tempfile
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import torch
# Imports for psutil and pynvml are moved to the functions that use them.
from utils import get_available_hf_models, get_ffmpeg_cmd, convert_to_16k_mono
from transcriber import run_transcription
from benchmark import run_benchmark # Import the new benchmark function

# --- Dynamic Base Class for Theming ---
try:
    from ttkthemes import ThemedTk
    BaseAppClass = ThemedTk
except ImportError:
    BaseAppClass = tk.Tk

# --- Constants ---
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv"}
ALL_EXTS = AUDIO_EXTS | VIDEO_EXTS


class BenchmarkWindow(tk.Toplevel):
    """
    A new window for running transcription benchmarks.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("PyScribe Benchmark")
        self.geometry("600x600")
        
        self.cancel_event = threading.Event()
        self.monitoring_active = False # Flag for HW monitor thread
        
        # --- Create Widgets ---
        self.create_widgets()

    def create_widgets(self):
        """Creates and lays out the widgets for the benchmark window."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Audio File Selection ---
        audio_select_frame = ttk.LabelFrame(main_frame, text="1. Select Benchmark Audio")
        audio_select_frame.pack(fill=tk.X, pady=5)
        
        self.audio_choice_var = tk.StringVar(value="en")
        
        en_radio = ttk.Radiobutton(
            audio_select_frame, text="English (Sherlock Holmes)", variable=self.audio_choice_var,
            value="en", command=self.update_model_list
        )
        en_radio.pack(anchor="w", padx=5)
        
        es_radio = ttk.Radiobutton(
            audio_select_frame, text="Spanish (Napoleon)", variable=self.audio_choice_var,
            value="es", command=self.update_model_list
        )
        es_radio.pack(anchor="w", padx=5)

        # --- Model Selection ---
        self.model_frame = ttk.LabelFrame(main_frame, text="2. Select Models to Benchmark")
        self.model_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.model_vars = {}
        self.update_model_list() # Initial population of the model list

        # --- Controls ---
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Benchmark", command=self.start_benchmark)
        self.start_btn.pack(side=tk.LEFT)
        
        # --- FEATURE: Suppress Warning Checkbox ---
        self.suppress_warning_var = tk.BooleanVar(value=False)
        suppress_cb = ttk.Checkbutton(
            control_frame, text="Don't show download warning again", variable=self.suppress_warning_var
        )
        suppress_cb.pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar(value="Ready to start.")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        # --- Progress Bar ---
        self.progress = ttk.Progressbar(main_frame, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 5))

        # --- Hardware Metrics Display ---
        self.hw_metrics_text = tk.Text(main_frame, height=1, relief="flat", background=self.cget('bg'))
        self.hw_metrics_text.pack(fill=tk.X, pady=(0, 5))
        self.hw_metrics_text.config(state=tk.DISABLED)
        
        self.hw_metrics_text.tag_configure("cpu_color", foreground="green")
        self.hw_metrics_text.tag_configure("ram_color", foreground="green")
        self.hw_metrics_text.tag_configure("gpu_color", foreground="purple")
        self.hw_metrics_text.tag_configure("vram_color", foreground="purple")

        # --- Results Display ---
        results_frame = ttk.LabelFrame(main_frame, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, font=("Segoe UI", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.insert(tk.END, "Benchmark results will appear here.\n")
        self.results_text.config(state=tk.DISABLED)

        # --- Export Button ---
        self.export_btn = ttk.Button(main_frame, text="Export Results", command=self.export_results, state=tk.DISABLED)
        self.export_btn.pack(pady=10)

    def update_model_list(self):
        """
        Clears and repopulates the model selection checkboxes based on the
        chosen audio language (English or multilingual).
        """
        for widget in self.model_frame.winfo_children():
            widget.destroy()
        self.model_vars.clear()

        all_models = MODEL_CHOICES + get_available_hf_models()
        unique_models = sorted(list(set(all_models)))
        
        lang_choice = self.audio_choice_var.get()
        models_to_show = unique_models if lang_choice == "en" else [m for m in unique_models if ".en" not in m]

        for model_name in models_to_show:
            var = tk.BooleanVar(value=False)
            self.model_vars[model_name] = var
            cb = ttk.Checkbutton(self.model_frame, text=model_name, variable=var)
            cb.pack(anchor="w", padx=5)

    def start_benchmark(self):
        """Starts the benchmark process in a separate thread."""
        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        
        if not selected_models:
            messagebox.showwarning("No Models Selected", "Please select at least one model to benchmark.", parent=self)
            return

        # --- FEATURE: Suppress warning logic ---
        if not self.suppress_warning_var.get():
            proceed = messagebox.askokcancel(
                "Confirm Benchmark",
                "The benchmark will now download any selected models that are not already cached.\n\n"
                "This may take a significant amount of time and disk space.\n\nDo you want to continue?",
                parent=self
            )
            if not proceed:
                return

        self.start_btn.config(text="Cancel", command=self.cancel_benchmark)
        self.export_btn.config(state=tk.DISABLED)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Starting benchmark...\n")
        self.results_text.config(state=tk.DISABLED)
        
        self.cancel_event.clear()
        
        audio_language = self.audio_choice_var.get()
        
        thread = threading.Thread(
            target=run_benchmark,
            args=(self.parent, self, selected_models, audio_language),
            daemon=True
        )
        thread.start()

    def cancel_benchmark(self):
        """Signals the benchmark thread to stop."""
        self.cancel_event.set()
        self.status_var.set("Cancelling...")

    def export_results(self):
        """Saves the benchmark results to a text file."""
        results = self.results_text.get(1.0, tk.END).strip()
        if not results:
            messagebox.showwarning("No Results", "There are no benchmark results to export.", parent=self)
            return
        
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"{ts}-benchmark.txt"
            save_path = os.path.join(os.path.expanduser("~"), "Documents", filename)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(results)
            
            messagebox.showinfo("Export Successful", f"Benchmark results saved to:\n{save_path}", parent=self)
        except Exception as e:
            self.show_error(f"Failed to export results:\n\n{e}")

    def update_status(self, message: str):
        """Thread-safely updates the status label."""
        self.after(0, lambda: self.status_var.set(message))

    def update_progress(self, value: float):
        """Thread-safely updates the progress bar."""
        self.after(0, lambda: self.progress.config(value=value))

    def update_results(self, text: str):
        """Thread-safely appends text to the results area."""
        def _update():
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, text)
            self.results_text.see(tk.END) # Auto-scroll to the bottom
            self.results_text.config(state=tk.DISABLED)
        self.after(0, _update)

    def show_error(self, message: str):
        """Thread-safely shows an error message."""
        self.after(0, lambda: messagebox.showerror("Error", message, parent=self))

    def enable_start_button(self):
        """Thread-safely re-enables the start button."""
        self.after(0, lambda: self.start_btn.config(text="Start Benchmark", command=self.start_benchmark))
        self.after(0, lambda: self.export_btn.config(state=tk.NORMAL))

    def start_hw_monitor(self):
        """Starts the hardware monitoring thread."""
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._hw_monitor_worker, daemon=True)
        monitor_thread.start()

    def stop_hw_monitor(self):
        """Stops the hardware monitoring thread and clears the text."""
        self.monitoring_active = False
        self._update_hw_metrics([])

    def _hw_monitor_worker(self):
        """
        The worker function that runs in a separate thread to poll hardware stats.
        It runs in a loop until the `monitoring_active` flag is set to False.
        """
        import psutil
        import pynvml

        while self.monitoring_active:
            try:
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                metrics = [
                    (f"CPU: {cpu_percent:.1f}%", "cpu_color"),
                    (f" | RAM: {ram_percent:.1f}%", "ram_color")
                ]
                
                if self.parent.device == "cuda" and self.parent.gpu_handle:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.parent.gpu_handle).gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.parent.gpu_handle)
                    vram_used_gb = mem_info.used / (1024**3)
                    vram_total_gb = mem_info.total / (1024**3)
                    metrics.append((f" | GPU: {gpu_util}%", "gpu_color"))
                    metrics.append((f" | VRAM: {vram_used_gb:.1f}/{vram_total_gb:.1f} GB", "vram_color"))
                
                self.after(0, self._update_hw_metrics, metrics)
                time.sleep(1)
            except Exception:
                self.monitoring_active = False
    
    def _update_hw_metrics(self, metrics):
        """Thread-safely updates the hardware metrics Text widget with colors."""
        self.hw_metrics_text.config(state=tk.NORMAL)
        self.hw_metrics_text.delete(1.0, tk.END)
        for text, tag in metrics:
            self.hw_metrics_text.insert(tk.END, text, tag)
        self.hw_metrics_text.config(state=tk.DISABLED)


class PyScribeApp(BaseAppClass):
    """
    The main application class for the PyScribe GUI.
    This class encapsulates all the UI elements and application state.
    """

    def __init__(self):
        """Initializes the main application window and its state."""
        if issubclass(BaseAppClass, tk.Tk) and BaseAppClass != tk.Tk:
            super().__init__(theme="arc")
        else:
            super().__init__()

        self.title("PyScribe - Faster-Whisper GUI")
        self.geometry("900x650")

        # --- Application State Variables ---
        self.media_path: str | None = None
        self.prepared_audio_path: str | None = None
        self.temp_dir = None
        self.transcription: str | None = None
        self.model = None
        self.current_model_name: str | None = None
        self.playback_process = None
        
        # --- Threading and Synchronization Events ---
        self.is_waiting_for_user = False
        self.override_event = threading.Event()
        self.user_override_choice = False
        self.cancel_event = threading.Event()
        self.monitoring_active = False # Flag for HW monitor thread

        # --- Hardware Information ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "N/A"
        self.vram_gb = 0
        self.gpu_handle = None
        if self.device == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                props = torch.cuda.get_device_properties(0)
                self.vram_gb = round(props.total_memory / (1024 ** 3), 1)
            except (ImportError, Exception):
                self.device = "cpu"
        self.cpu_count = os.cpu_count() or 1
        self.recommended_model = self._recommend_model()

        # --- UI Initialization ---
        self._create_widgets()
        self._show_startup_status()
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)

    def _create_widgets(self):
        """Creates and arranges all the GUI widgets in the main window."""
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

        self.play_btn = tk.Button(second_frame, text="▶ Play", command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=6)

        tk.Button(second_frame, text="Run Benchmark", command=self.open_benchmark_window).pack(side=tk.LEFT, padx=6)
        
        tk.Button(top_frame, text="Exit", command=self.on_exit).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Select an audio/video file to begin.")
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=10)
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.hw_metrics_text = tk.Text(self, height=1, relief="flat", background=self.cget('bg'))
        self.hw_metrics_text.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.hw_metrics_text.config(state=tk.DISABLED)
        
        self.hw_metrics_text.tag_configure("cpu_color", foreground="green")
        self.hw_metrics_text.tag_configure("ram_color", foreground="green")
        self.hw_metrics_text.tag_configure("gpu_color", foreground="purple")
        self.hw_metrics_text.tag_configure("vram_color", foreground="purple")
        
        self.text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=("Segoe UI", 10))
        self.text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=(0, 8))

    def _show_startup_status(self):
        """Displays hardware info and recommended model in the status bar on startup."""
        info = f"GPU: {self.gpu_name} ({self.vram_gb} GB VRAM)" if self.device == "cuda" else f"CPU Mode ({self.cpu_count} cores)"
        self.status_var.set(f"{info} | Recommended model: {self.recommended_model}")

    def _recommend_model(self) -> str:
        """Recommends a Whisper model size based on available hardware."""
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
        """Validates user inputs and starts the transcription in a new thread."""
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
        self.cancel_event.clear()
        thread = threading.Thread(target=run_transcription, args=(self, model_to_use), daemon=True)
        thread.start()

    # --- Hardware Monitoring ---
    def start_hw_monitor(self):
        """Starts the hardware monitoring thread."""
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._hw_monitor_worker, daemon=True)
        monitor_thread.start()

    def stop_hw_monitor(self):
        """Stops the hardware monitoring thread and clears the text."""
        self.monitoring_active = False
        self._update_hw_metrics([])

    def _hw_monitor_worker(self):
        """
        The worker function that runs in a separate thread to poll hardware stats.
        It runs in a loop until the `monitoring_active` flag is set to False.
        """
        import psutil
        import pynvml

        while self.monitoring_active:
            try:
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                metrics = [
                    (f"CPU: {cpu_percent:.1f}%", "cpu_color"),
                    (f" | RAM: {ram_percent:.1f}%", "ram_color")
                ]
                
                if self.device == "cuda" and self.gpu_handle:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    vram_used_gb = mem_info.used / (1024**3)
                    vram_total_gb = mem_info.total / (1024**3)
                    metrics.append((f" | GPU: {gpu_util}%", "gpu_color"))
                    metrics.append((f" | VRAM: {vram_used_gb:.1f}/{vram_total_gb:.1f} GB", "vram_color"))
                
                self.after(0, self._update_hw_metrics, metrics)
                time.sleep(1)
            except Exception:
                self.monitoring_active = False
    
    def _update_hw_metrics(self, metrics):
        """Thread-safely updates the hardware metrics Text widget with colors."""
        self.hw_metrics_text.config(state=tk.NORMAL)
        self.hw_metrics_text.delete(1.0, tk.END)
        for text, tag in metrics:
            self.hw_metrics_text.insert(tk.END, text, tag)
        self.hw_metrics_text.config(state=tk.DISABLED)

    # --- Callbacks for Transcriber Thread ---
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
        self._toggle_busy(False)

    def refresh_hf_model_list(self):
        self.after(0, lambda: self.hf_model_combo.config(values=get_available_hf_models()))

    def prompt_for_english_override(self, lang_code: str, model_name: str):
        """Pauses the transcriber thread and prompts the user for a language override decision."""
        self.is_waiting_for_user = True
        self.override_event.clear()
        self.after(0, self._ask_english_override_prompt, lang_code, model_name)
        self._toggle_busy(False, keep_progress=True)
        self.override_event.wait()
        self._toggle_busy(True)

    def _ask_english_override_prompt(self, lang_code: str, model_name: str):
        """Shows the actual message box for the override and signals the waiting thread."""
        message = (f"The detected language is '{lang_code}', but you have selected an "
                   f"English-only model ('{model_name}').\n\n"
                   f"Force transcription in English anyway?")
        
        self.user_override_choice = messagebox.askyesno("Language Mismatch", message)
        
        self.is_waiting_for_user = False
        self.override_event.set()

    # --- UI Event Handlers ---
    def browse_file(self):
        self.stop_audio()
        self._cleanup_temp_dir()
        filetypes = [
            ("Audio/Video Files", " ".join(f"*{ext}" for ext in sorted(list(ALL_EXTS)))),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Media File", filetypes=filetypes)
        if path:
            self.media_path = path
            self.prepared_audio_path = None
            self.status_var.set(f"Selected: {os.path.basename(path)}")
            self.save_btn.config(state=tk.DISABLED)
            self.copy_btn.config(state=tk.DISABLED)
            self.play_btn.config(state=tk.NORMAL)
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

    def ensure_audio_is_prepared(self):
        if self.prepared_audio_path and os.path.exists(self.prepared_audio_path):
            return

        if not self.media_path:
            return

        try:
            self.set_status("Preparing audio for playback...")
            self.temp_dir = tempfile.TemporaryDirectory()
            ffmpeg_cmd = get_ffmpeg_cmd(tool="ffmpeg")
            if not ffmpeg_cmd:
                raise FileNotFoundError("ffmpeg not found.")
            self.prepared_audio_path = convert_to_16k_mono(self.media_path, self.temp_dir.name, ffmpeg_cmd)
            self.set_status("Audio ready.")
        except Exception as e:
            self.show_error(f"Could not prepare audio:\n\n{e}")
            self._cleanup_temp_dir()

    def play_audio(self):
        self.stop_audio()
        self.ensure_audio_is_prepared()
        if not self.prepared_audio_path:
            return
        
        def _play_monitor():
            ffplay_cmd = get_ffmpeg_cmd(tool="ffplay")
            if not ffplay_cmd:
                self.show_error("ffplay not found.")
                return

            try:
                self.after(0, self._set_play_button_state, True)
                self.set_status("Playing audio...")
                
                self.playback_process = subprocess.Popen(
                    [ffplay_cmd, "-nodisp", "-autoexit", self.prepared_audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                self.playback_process.wait()

                if self.playback_process:
                    self.set_status("Playback finished.")
            
            except Exception as e:
                self.show_error(f"Could not play audio:\n\n{e}")
            finally:
                self.playback_process = None
                if self.winfo_exists():
                    self.after(0, self._set_play_button_state, False)

        threading.Thread(target=_play_monitor, daemon=True).start()

    def stop_audio(self):
        if self.playback_process:
            try:
                self.playback_process.terminate()
                self.playback_process = None
                self.set_status("Playback stopped.")
            except Exception:
                pass

    def _set_play_button_state(self, is_playing: bool):
        if is_playing:
            self.play_btn.config(text="■ Stop", command=self.stop_audio)
        else:
            self.play_btn.config(text="▶ Play", command=self.play_audio)

    def cancel_transcription(self):
        self.cancel_event.set()
        self.set_status("Cancelling...")

    def open_benchmark_window(self):
        """Opens the benchmark tool window."""
        BenchmarkWindow(self)

    def _clear_hf_model(self, event=None):
        self.hf_model_var.set("")

    def _clear_standard_model(self, event=None):
        self.model_var.set("")

    def _toggle_busy(self, busy: bool, keep_progress: bool = False):
        if busy:
            if not keep_progress:
                self.progress.config(value=0)
            self.transcribe_btn.config(text="Cancel", command=self.cancel_transcription, fg="red")
        else:
            if not keep_progress:
                self.progress.config(value=0)
            self.transcribe_btn.config(text="Transcribe", command=self.start_transcription, fg="black")
            self.transcribe_btn.config(state=tk.NORMAL)

    def _cleanup_temp_dir(self):
        if self.temp_dir:
            try:
                self.temp_dir.cleanup()
            except Exception:
                pass
            self.temp_dir = None
            self.prepared_audio_path = None

    def on_exit(self):
        """Handles the window close event."""
        self.stop_audio()
        self._cleanup_temp_dir()
        if self.device == "cuda" and self.gpu_handle:
            try:
                import pynvml
                pynvml.nvmlShutdown() # Clean up NVML
            except ImportError:
                pass
        self.destroy()
