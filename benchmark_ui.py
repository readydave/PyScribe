# benchmark_ui.py
# This module contains the UI for the benchmark window.

import os
import threading
import datetime
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from utils import get_available_hf_models
from benchmark import run_benchmark

# This constant is shared with ui.py
MODEL_CHOICES = [
    "tiny",
    "base",
    "small",
    "small.en",
    "medium",
    "large-v2",
    "large-v3",
    "distil-whisper/large-v3",
    "Systran/faster-whisper-large-v3-turbo",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "openai/whisper-large-v3-turbo",
]

class BenchmarkWindow(tk.Toplevel):
    """
    A new window for running transcription benchmarks.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("PyScribe Benchmark")
        # --- CHANGE: Increased default window size ---
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
        
        self.suppress_warning_var = tk.BooleanVar(value=False)
        suppress_cb = ttk.Checkbutton(
            control_frame, text="Don't show download warning again", variable=self.suppress_warning_var
        )
        suppress_cb.pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar(value="Ready to start.")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        # --- FEATURE: Progress Bar ---
        self.progress = ttk.Progressbar(main_frame, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 5))

        # --- FEATURE: Hardware Metrics Display ---
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

        # --- FEATURE: Export Button ---
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
        """Signals the benchmark thread to stop and provides immediate user feedback."""
        self.cancel_event.set()
        self.status_var.set("Cancelling... waiting for current model to finish.")
        self.start_btn.config(state=tk.DISABLED) # Prevent multiple clicks
        self.stop_hw_monitor() # Stop the monitor immediately

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
        self.after(0, lambda: self.start_btn.config(text="Start Benchmark", command=self.start_benchmark, state=tk.NORMAL))
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
        import time

        gpu_handle = None
        if self.parent.device == "cuda":
            try:
                pynvml.nvmlInit()
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except (ImportError, Exception):
                pass

        while self.monitoring_active:
            try:
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                metrics = [
                    (f"CPU: {cpu_percent:.1f}%", "cpu_color"),
                    (f" | RAM: {ram_percent:.1f}%", "ram_color")
                ]
                
                if gpu_handle:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    vram_used_gb = mem_info.used / (1024**3)
                    vram_total_gb = mem_info.total / (1024**3)
                    metrics.append((f" | GPU: {gpu_util}%", "gpu_color"))
                    metrics.append((f" | VRAM: {vram_used_gb:.1f}/{vram_total_gb:.1f} GB", "vram_color"))
                
                self.after(0, self._update_hw_metrics, metrics)
                time.sleep(1)
            except Exception:
                self.monitoring_active = False
        
        if gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except (ImportError, Exception):
                pass
    
    def _update_hw_metrics(self, metrics):
        """Thread-safely updates the hardware metrics Text widget with colors."""
        self.hw_metrics_text.config(state=tk.NORMAL)
        self.hw_metrics_text.delete(1.0, tk.END)
        for text, tag in metrics:
            self.hw_metrics_text.insert(tk.END, text, tag)
        self.hw_metrics_text.config(state=tk.DISABLED)
