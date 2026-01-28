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
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter.font as tkfont
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from models import get_ranked_models, strip_badges, BADGES, TIER_ORDER
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    DND_FILES = None
    TkinterDnD = None

# Silence future warning from huggingface_hub resume_download deprecation
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*resume_download.*",
    module="huggingface_hub.file_download"
)

import torch
# Imports for psutil and pynvml are moved to the functions that use them.
from utils import get_ffmpeg_cmd, convert_to_16k_mono
from models import TIERS
from transcriber import run_transcription
from benchmark_ui import BenchmarkWindow # Import the new benchmark window

# --- Dynamic Base Class for Theming ---
try:
    from ttkthemes import ThemedTk
    BaseTk = ThemedTk
except ImportError:
    BaseTk = tk.Tk

if DND_AVAILABLE:
    BaseAppClass = TkinterDnD.Tk
    USE_THEME = False
else:
    BaseAppClass = BaseTk
    USE_THEME = BaseTk is not tk.Tk

# --- Constants ---
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv"}
ALL_EXTS = AUDIO_EXTS | VIDEO_EXTS
MODEL_REPO_MAP = {
    # Map shorthand -> confirmed public CT2 repos
    "tiny": "guillaumekln/whisper-tiny-ct2",
    "tiny.en": "guillaumekln/whisper-tiny.en-ct2",
    "base": "guillaumekln/whisper-base-ct2",
    "base.en": "guillaumekln/whisper-base.en-ct2",
    "small": "guillaumekln/whisper-small-ct2",
    "small.en": "guillaumekln/whisper-small.en-ct2",
    "medium": "guillaumekln/whisper-medium-ct2",
    "large-v2": "guillaumekln/whisper-large-v2-ct2",
    "large-v3": "guillaumekln/whisper-large-v3-ct2",
    "distil-whisper/distil-large-v3": "distil-whisper/distil-large-v3",
}


class PyScribeApp(BaseAppClass):
    """
    The main application class for the PyScribe GUI.
    This class encapsulates all the UI elements and application state.
    """

    def __init__(self):
        """Initializes the main application window and its state."""
        if USE_THEME:
            super().__init__(theme="arc")
        else:
            super().__init__()

        self.title("PyScribe - Faster-Whisper GUI")
        self.geometry("900x650")
        self.minsize(800, 600)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

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
        self.downloading_models = set()  # Track models flagged as downloading (future hook)

        # --- Hardware Information ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = "N/A"
        self.vram_gb = 0
        self.gpu_handle = None
        if self.device == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                self.vram_gb = round(props.total_memory / (1024 ** 3), 1)
            except (ImportError, Exception):
                self.device = "cpu"
        self.cpu_count = os.cpu_count() or 1

        self.cached_models = self._get_cached_hf_models()
        self.model_entries = self._load_ranked_models()
        self.recommended_model_name = self.model_entries[0]["name"] if self.model_entries else "small"
        base_label = self.model_entries[0]["label"] if self.model_entries else "🟢 small"
        self.recommended_model_label = f"{self._status_prefix(self.recommended_model_name)} {base_label}"
        self.selected_model_name = self.recommended_model_name
        self.selected_model_label = self.recommended_model_label
        self.model_font = tkfont.Font(family="Segoe UI", size=10)
        self.model_font_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.hf_xet_available = self._hf_xet_available()
        self.config_path = Path.home() / ".pyscribe_config.json"
        self.last_model_used = self._load_last_model()

        # --- UI Initialization ---
        self._create_widgets()
        self._apply_last_model_selection()
        self._show_startup_status()
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)

    def _create_widgets(self):
        """Creates and arranges all the GUI widgets in the main window."""
        top_frame = tk.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=6)
        top_frame.grid_columnconfigure(0, weight=0)
        top_frame.grid_columnconfigure(1, weight=0)
        top_frame.grid_columnconfigure(2, weight=1)
        top_frame.grid_columnconfigure(3, weight=0)
        top_frame.grid_columnconfigure(4, weight=0)
        top_frame.grid_columnconfigure(5, weight=0)

        tk.Button(top_frame, text="Browse File", command=self.browse_file).grid(row=0, column=0, sticky="w", padx=(0,8))

        self.drop_label = tk.Label(
            top_frame,
            text="Drag & Drop audio/video here",
            relief="solid",
            bd=2,
            highlightthickness=2,
            highlightbackground="red",
            width=28,
            pady=4
        )
        self.drop_label.grid(row=0, column=1, sticky="w", padx=(0,8))
        if DND_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_file_drop)
            self.drop_label.dnd_bind("<<DragEnter>>", self._on_drag_enter)
            self.drop_label.dnd_bind("<<DragLeave>>", self._on_drag_leave)

        model_frame = tk.Frame(top_frame)
        model_frame.grid(row=0, column=2, sticky="ew", padx=(4,8))
        model_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=(0,4))
        self.model_label = ttk.Label(model_frame, text=self._model_label_text(), font=("Segoe UI", 10, "bold"))
        self.model_label.grid(row=0, column=1, sticky="w")

        ttk.Button(top_frame, text="Choose Model", command=self.open_model_dialog).grid(row=0, column=3, sticky="w", padx=(6,8))
        self.transcribe_btn = tk.Button(top_frame, text="Transcribe", command=self.start_transcription, font=("Segoe UI", 9, "bold"))
        self.transcribe_btn.grid(row=0, column=4, sticky="e", padx=(0,8))
        tk.Button(top_frame, text="Exit", command=self.on_exit).grid(row=0, column=5, sticky="e")

        toolbar = tk.Frame(self)
        toolbar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,6))
        for i in range(6):
            toolbar.grid_columnconfigure(i, weight=0)
        self.save_btn = tk.Button(toolbar, text="Save Transcript", command=self.save_transcription, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=0, padx=4)
        self.copy_btn = tk.Button(toolbar, text="Copy", command=self.copy_to_clipboard, state=tk.DISABLED)
        self.copy_btn.grid(row=0, column=1, padx=4)
        self.open_btn = tk.Button(toolbar, text="Open Save Folder", command=self.open_transcriptions_folder, state=tk.NORMAL)
        self.open_btn.grid(row=0, column=2, padx=4)
        self.play_btn = tk.Button(toolbar, text="▶ Play", command=self.play_audio, state=tk.DISABLED)
        self.play_btn.grid(row=0, column=3, padx=4)
        tk.Button(toolbar, text="Run Benchmark", command=self.open_benchmark_window).grid(row=0, column=4, padx=4)
        tk.Button(toolbar, text="Rescan Cache", command=self._rescan_cache_and_refresh).grid(row=0, column=5, padx=4)
        
        second_frame = tk.Frame(self)
        second_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))

        self.save_btn = tk.Button(second_frame, text="Save Transcript", command=self.save_transcription, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.copy_btn = tk.Button(second_frame, text="Copy", command=self.copy_to_clipboard, state=tk.DISABLED)
        self.copy_btn.pack(side=tk.LEFT, padx=6)

        self.open_btn = tk.Button(second_frame, text="Open Save Folder", command=self.open_transcriptions_folder, state=tk.NORMAL)
        self.open_btn.pack(side=tk.LEFT, padx=6)

        self.play_btn = tk.Button(second_frame, text="▶ Play", command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=6)

        tk.Button(second_frame, text="Run Benchmark", command=self.open_benchmark_window).pack(side=tk.LEFT, padx=6)

        center_frame = tk.Frame(self)
        center_frame.grid(row=2, column=0, sticky="nsew")
        center_frame.grid_columnconfigure(0, weight=1)
        center_frame.grid_rowconfigure(4, weight=1)

        self.status_var = tk.StringVar(value="Select an audio/video file to begin.")
        tk.Label(center_frame, textvariable=self.status_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=10)
        self.progress = ttk.Progressbar(center_frame, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

        self.hw_metrics_text = tk.Text(center_frame, height=1, relief="flat", background=self.cget('bg'))
        self.hw_metrics_text.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.hw_metrics_text.config(state=tk.DISABLED)
        
        self.hw_metrics_text.tag_configure("cpu_color", foreground="green")
        self.hw_metrics_text.tag_configure("ram_color", foreground="green")
        self.hw_metrics_text.tag_configure("gpu_color", foreground="purple")
        self.hw_metrics_text.tag_configure("vram_color", foreground="purple")
        
        self.text_area = scrolledtext.ScrolledText(center_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        self.text_area.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 8))

    def _show_startup_status(self):
        """Displays hardware info and recommended model in the status bar on startup."""
        info = f"GPU: {self.gpu_name} ({self.vram_gb} GB VRAM)" if self.device == "cuda" else f"CPU Mode ({self.cpu_count} cores)"
        self.status_var.set(f"{info} | Recommended model: {self.recommended_model_name}")

    def start_transcription(self):
        """Validates user inputs and starts the transcription in a new thread."""
        if self.is_waiting_for_user:
            messagebox.showinfo("Info", "Please respond to the language prompt first.")
            return
        if not self.media_path:
            messagebox.showerror("Error", "Please select a file first.")
            return

        model_to_use = (self.selected_model_name or "").strip()
        if not model_to_use:
            messagebox.showerror("Error", "Please select a model.")
            return

        if model_to_use.startswith("openai/") and "faster-whisper" not in model_to_use and "ct2" not in model_to_use:
            messagebox.showerror(
                "Unsupported model format",
                "The selected model is the raw OpenAI checkpoint. Please choose a CTranslate2/faster-whisper export, e.g.:\n"
                "- Systran/faster-whisper-large-v3-turbo\n"
                "- deepdml/faster-whisper-large-v3-turbo-ct2\n"
                "- distil-whisper/large-v3",
                parent=self
            )
            return

        if not self._is_repo_cached(model_to_use):
            if not messagebox.askyesno("Model download required", f"'{model_to_use}' is not downloaded. Download now?", parent=self):
                return
            ok = self._download_models_with_progress([model_to_use], parent=self)
            if not ok:
                return

        entry = self._find_model_entry_by_name(model_to_use)
        if entry and not entry.get("fits", True):
            warn_msg = (f"The selected model '{model_to_use}' may exceed available VRAM "
                        f"({self.vram_gb} GB vs {entry.get('vram')} GB). Continue?")
            if not messagebox.askokcancel("Hardware Warning", warn_msg):
                return
        self._persist_last_model(model_to_use)

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

        gpu_handle = None
        if self.device == "cuda":
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
        def _refresh():
            self.cached_models = self._get_cached_hf_models()
            names = set(TIERS.keys()) | self.cached_models
            self.model_entries = self._build_entries_from_names(names)
            labels = self._build_model_labels()
            if labels:
                top = self.model_entries[0]
                self.recommended_model_name = top["name"]
                self.recommended_model_label = f"{self._status_prefix(top['name'])} {top['label']}"
                if self.selected_model_label not in labels:
                    self.selected_model_label = self.recommended_model_label
                    self.selected_model_name = self.recommended_model_name
                self.model_label.config(text=self._model_label_text())
        self.after(0, _refresh)

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

    def prompt_detected_language(self, lang_code: str, lang_prob: float):
        """
        Ask user to confirm detected language or force English.
        Returns 'en' if user opts to force English, otherwise the detected code.
        """
        self.is_waiting_for_user = True
        choice = messagebox.askyesno(
            "Confirm Language",
            f"Detected language: '{lang_code}' (confidence {lang_prob*100:.1f}%).\n\n"
            f"Yes = use detected language.\nNo = force US English.",
            parent=self
        )
        self.is_waiting_for_user = False
        return lang_code if choice else "en"

    # --- UI Event Handlers ---
    def _load_ranked_models(self):
        # Build entries from curated set + any cached repos
        names = set(TIERS.keys())
        names.update(self.cached_models)
        return self._build_entries_from_names(names)

    def _build_entries_from_names(self, names: set[str]):
        entries = []
        for name in names:
            tier = self._guess_tier_from_name(name)
            badge = BADGES.get(tier, "")
            hq = tier == "PRO"
            label = f"{badge}{' ★' if hq else ''} {name}"
            entries.append({
                "name": name,
                "tier": tier,
                "badge": badge,
                "hq": hq,
                "label": label,
                "tooltip": f"Tier: {tier}{' (HQ accuracy)' if hq else ''}",
                "fits": True,
                "verified": name in self.cached_models,
                "vram": None,
                "wer": None,
                "rt": None,
            })

        entries.sort(key=lambda x: (TIER_ORDER.get(x["tier"], 9), x["name"]))
        return entries

    def _get_cached_hf_models(self):
        """Return locally cached HF models (for Verified badge)."""
        cached = set()
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
        if os.path.isdir(cache_dir):
            for item in os.listdir(cache_dir):
                if item.startswith("models--"):
                    model_id = item.replace("models--", "").replace("--", "/")
                    if self._is_repo_cached(model_id):
                        cached.add(model_id)
                        # also mark shorthand if mapping points here
                        for short, repo in MODEL_REPO_MAP.items():
                            if repo == model_id:
                                cached.add(short)
        return cached

    def _scan_cache_repos(self) -> set[str]:
        """Return repo ids present in the local cache."""
        repos = set()
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
        if os.path.isdir(cache_dir):
            for item in os.listdir(cache_dir):
                if item.startswith("models--"):
                    model_id = item.replace("models--", "").replace("--", "/")
                    repos.add(model_id)
        return repos

    def _load_last_model(self):
        try:
            data = json.loads(self.config_path.read_text())
            return data.get("last_model")
        except Exception:
            return None

    def _persist_last_model(self, model_name: str):
        try:
            data = {"last_model": model_name}
            self.config_path.write_text(json.dumps(data))
        except Exception:
            pass

    def _apply_last_model_selection(self):
        if not self.last_model_used:
            return
        entry = self._find_model_entry_by_name(self.last_model_used)
        if entry:
            self.selected_model_name = entry["name"]
            self.selected_model_label = f"{self._status_prefix(entry['name'])} {entry['label']}"
        else:
            # keep recommended if not found
            self.selected_model_name = self.recommended_model_name
            self.selected_model_label = self.recommended_model_label

    def _is_repo_cached(self, model_id: str) -> bool:
        """Best-effort check if a HF repo snapshot exists locally (model.bin or safetensors)."""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
        repo_dir = os.path.join(cache_dir, f"models--{MODEL_REPO_MAP.get(model_id, model_id).replace('/', '--')}")
        if not os.path.isdir(repo_dir):
            return False
        # look for model binaries in snapshot folders
        for root, _, files in os.walk(repo_dir):
            for fname in files:
                if fname.endswith((".bin", ".safetensors")):
                    return True
        return False

    def _status_prefix(self, name: str):
        if name in self.downloading_models:
            return "●"
        if name in self.cached_models:
            return "☑"
        return "☐"

    def _build_model_labels(self, filter_text: str = ""):
        labels = []
        needle = filter_text.lower()
        for entry in self.model_entries:
            label = f"{self._status_prefix(entry['name'])} {entry['label']}"
            if needle and needle not in label.lower():
                continue
            labels.append(label)
        return labels

    def _populate_model_listbox(self, labels, listbox, recommended_name):
        listbox.delete(0, tk.END)
        for idx, lbl in enumerate(labels):
            base = self._strip_status(lbl)
            entry = self._find_model_entry_by_label(base)
            display = lbl
            if entry and entry["name"] == recommended_name:
                display = f"★ {display}"
            listbox.insert(tk.END, display)
            if entry and entry["name"] == recommended_name:
                listbox.itemconfig(idx, {'foreground': 'blue'})

    def _find_model_entry_by_label(self, label: str):
        label = self._strip_status(label)
        for entry in self.model_entries:
            if entry["label"] == label:
                return entry
        return None

    def _find_model_entry_by_name(self, name: str):
        for entry in self.model_entries:
            if entry["name"] == name:
                return entry
        return None

    def _normalize_model_label(self, value: str) -> str:
        cleaned = self._strip_status(strip_badges(value.strip()))
        if "• Verified" in cleaned:
            cleaned = cleaned.split("• Verified")[0].strip()
        return cleaned

    def _strip_status(self, label: str) -> str:
        if not label:
            return label
        if label[0] in {"☑", "☐", "●"}:
            return label[1:].strip()
        return label

    def _model_label_text(self):
        return self._strip_status(self.selected_model_label)

    def _tier_color(self, tier: str) -> str:
        return {"FAST": "green", "BALANCED": "darkorange", "PRO": "red"}.get(tier, "black")

    def open_model_dialog(self):
        dialog = tk.Toplevel(self)
        header_text = f"{self._suggested_text()} • Suggested: {self.recommended_model_name}"
        dialog.title("Select Model")
        dialog.transient(self)
        dialog.grab_set()
        dialog.geometry("560x520")
        dialog.minsize(520, 480)
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(2, weight=1)
        dialog.grid_rowconfigure(3, weight=0)

        header = tk.Label(
            dialog,
            text=header_text,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            justify="left",
            wraplength=520
        )
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 2))

        legend_frame = tk.Frame(dialog)
        legend_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        legend_frame.grid_columnconfigure((0,1,2,3,4), weight=1)
        legend_frame.grid_propagate(True)

        tk.Label(legend_frame, text="🟢 Fast (<2GB 100x+)", fg="green", anchor="w").grid(row=0, column=0, sticky="w", padx=(0,4))
        tk.Label(legend_frame, text="🟡 Balanced (2-6GB)", fg="darkorange", anchor="w").grid(row=0, column=1, sticky="w", padx=(4,4))
        tk.Label(legend_frame, text="🔴 Pro (6GB+)", fg="red", anchor="w").grid(row=0, column=2, sticky="w", padx=(4,4))

        tk.Label(legend_frame, text="☑ Downloaded", anchor="w").grid(row=1, column=0, sticky="w", pady=(2,0), padx=(0,4))
        tk.Label(legend_frame, text="● Downloading", anchor="w").grid(row=1, column=1, sticky="w", pady=(2,0), padx=(4,4))
        tk.Label(legend_frame, text="☐ Fetch", anchor="w").grid(row=1, column=2, sticky="w", pady=(2,0), padx=(4,4))
        tk.Label(legend_frame, text="! Error", fg="red", anchor="w").grid(row=1, column=3, sticky="w", pady=(2,0), padx=(4,4))
        tk.Label(legend_frame, text="★ Suggested (hardware)", font=("Segoe UI", 9, "bold"), anchor="w").grid(row=1, column=4, sticky="w", pady=(2,0), padx=(4,0))

        list_frame = tk.Frame(dialog)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_propagate(False)

        canvas = tk.Canvas(list_frame, highlightthickness=0)
        vscroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")

        inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        dialog.bind("<Configure>", lambda e: self._on_modal_resize(e, canvas, inner))

        labels = self._build_model_labels()
        self.model_choice_var = tk.StringVar(value=self.selected_model_name)
        self.download_vars = {}

        self.model_list_rows = []
        for idx, lbl in enumerate(labels):
            base = self._strip_status(lbl)
            entry = self._find_model_entry_by_label(base)
            name = entry["name"] if entry else base
            row = tk.Frame(inner)
            row.grid(row=idx, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)

            is_recommended = entry and entry["name"] == self.recommended_model_name
            radio_text = f"{'★ ' if is_recommended else ''}{lbl}"
            fg_color = self._tier_color(entry["tier"]) if entry else "black"
            radio = tk.Radiobutton(
                row,
                variable=self.model_choice_var,
                value=name,
                text=radio_text,
                anchor="w",
                justify="left",
                font=self.model_font_bold if is_recommended else self.model_font,
                fg=fg_color,
            )
            radio.grid(row=0, column=1, sticky="w")

            if name in self.cached_models:
                status_lbl = tk.Label(row, text="☑", fg="green")
                status_lbl.grid(row=0, column=0, padx=(0, 6))
            else:
                var = tk.BooleanVar(value=False)
                self.download_vars[name] = var
                chk = tk.Checkbutton(row, variable=var, text="☐", anchor="w")
                chk.grid(row=0, column=0, padx=(0, 6))
            self.model_list_rows.append(row)

        btn_frame = tk.Frame(dialog)
        btn_frame.grid(row=3, column=0, sticky="ew", padx=12, pady=8)
        btn_frame.grid_columnconfigure((0,1,2,3), weight=1)
        tk.Button(btn_frame, text="Download selected", width=18, command=lambda: self._download_selected(dialog)).grid(row=0, column=0, sticky="w")
        tk.Button(btn_frame, text="Filter to available", width=18, command=lambda: self._filter_available_models(dialog)).grid(row=0, column=1, sticky="w")
        tk.Button(btn_frame, text="Refresh models", width=16, command=lambda: self._refresh_models_online(dialog)).grid(row=0, column=2, sticky="w")
        tk.Button(btn_frame, text="Cancel", width=10, command=dialog.destroy).grid(row=0, column=3, sticky="e", padx=4)
        tk.Button(btn_frame, text="OK", width=10, command=lambda: self._apply_dialog_selection(dialog)).grid(row=0, column=4, sticky="e", padx=4)
    def _apply_dialog_selection(self, dialog):
        selected_name = self.model_choice_var.get()
        if not selected_name:
            dialog.destroy()
            return
        entry = self._find_model_entry_by_name(selected_name)
        label = None
        if entry:
            label = f"{self._status_prefix(entry['name'])} {entry['label']}"
        if entry and not entry.get("fits", True):
            warn_msg = (f"The model '{entry['name']}' may exceed available VRAM "
                        f"({self.vram_gb} GB vs {entry.get('vram')} GB). Continue?")
            if not messagebox.askokcancel("Hardware Warning", warn_msg, parent=dialog):
                return
        self.selected_model_name = selected_name
        self.selected_model_label = label or selected_name
        self.model_label.config(text=self._model_label_text())
        dialog.destroy()

    def _on_modal_resize(self, event, canvas, inner):
        canvas.configure(scrollregion=canvas.bbox("all"))
        width = event.width
        new_size = 10
        if width >= 800:
            new_size = 11
        if width >= 950:
            new_size = 12
        if new_size != self.model_font.cget("size"):
            self.model_font.config(size=new_size)
            self.model_font_bold.config(size=new_size)

    def _download_selected(self, dialog):
        targets = [name for name, var in (self.download_vars or {}).items() if var.get()]
        if not targets:
            messagebox.showinfo("Download", "No models selected for download.", parent=dialog)
            return
        self._download_models_with_progress(targets, parent=dialog)

    def _download_models_with_progress(self, targets, parent):
        progress_win = tk.Toplevel(parent)
        progress_win.title("Downloading models")
        progress_win.geometry("420x170")
        progress_win.transient(parent)
        progress_win.grab_set()
        tk.Label(progress_win, text="Downloading...", anchor="w").pack(anchor="w", padx=10, pady=(10, 4))
        status_var = tk.StringVar(value="")
        speed_var = tk.StringVar(value="")
        bar = ttk.Progressbar(progress_win, mode="indeterminate", maximum=100)
        bar.pack(fill=tk.X, padx=10, pady=6)
        tk.Label(progress_win, textvariable=status_var, anchor="w", justify="left", wraplength=380).pack(fill=tk.X, padx=10)
        tk.Label(progress_win, textvariable=speed_var, anchor="w").pack(fill=tk.X, padx=10, pady=(0,4))
        if not self._hf_xet_available():
            tk.Label(
                progress_win,
                text="Tip: install hf_xet for faster downloads: pip install huggingface_hub[hf_xet]",
                anchor="w", justify="left", fg="gray", wraplength=380
            ).pack(fill=tk.X, padx=10, pady=(0,6))

        done_event = threading.Event()
        first_progress = {"seen": False}

        def updater(pct, text, speed_text):
            if not first_progress["seen"]:
                first_progress["seen"] = True
                bar.config(mode="determinate")
            bar.config(value=pct)
            status_var.set(text)
            speed_var.set(speed_text)

        def make_tqdm(model):
            from tqdm import tqdm
            start_time = time.time()

            class UITqdm(tqdm):
                def update(self_inner, n=1):
                    super().update(n)
                    pct = (self_inner.n / self_inner.total * 100) if self_inner.total else 0
                    elapsed = max(time.time() - start_time, 0.001)
                    speed = self_inner.n / elapsed if elapsed else 0
                    text = f"{model}: {self._format_size(self_inner.n)} / {self._format_size(self_inner.total)}"
                    speed_text = f"{self._format_size(speed)}/s  ({pct:.1f}%)"
                    self.after(0, updater, pct, text, speed_text)
                def close(self_inner):
                    super().close()
                    if self_inner.total:
                        text = f"{model}: {self._format_size(self_inner.total)} / {self._format_size(self_inner.total)}"
                        self.after(0, updater, 100, text, "")
                    elif not first_progress["seen"]:
                        # ensure determinate shows completion even if total missing
                        self.after(0, updater, 100, f"{model}: complete", "")
            return UITqdm

        def worker():
            api = HfApi()
            for model in targets:
                try:
                    info = api.repo_info(model, files_metadata=True)
                    siblings = [s for s in info.siblings if getattr(s, "size", None)]
                    total_bytes = sum(s.size for s in siblings if s.size)
                except Exception as e:
                    self.after(0, lambda m=model, err=e: messagebox.showerror("Download failed", f"{m}: {err}", parent=parent))
                    continue

                self.downloading_models.add(model)

                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id=MODEL_REPO_MAP.get(model, model),
                        local_dir=None,
                        local_dir_use_symlinks=False,
                        tqdm_class=make_tqdm(model),
                    )
                    self.cached_models.add(model)
                except Exception as e:
                    self.after(0, lambda m=model, err=e: messagebox.showerror("Download failed", f"{m}: {err}", parent=parent))
                finally:
                    self.downloading_models.discard(model)

            self.after(0, progress_win.destroy)
            self.after(0, self.refresh_hf_model_list)
            done_event.set()

        threading.Thread(target=worker, daemon=True).start()
        progress_win.wait_visibility()
        progress_win.wait_window()
        return done_event.is_set()

    def _filter_available_models(self, dialog):
        def worker():
            api = HfApi()
            available = []
            for entry in self.model_entries:
                repo = MODEL_REPO_MAP.get(entry["name"], entry["name"])
                if entry["name"] in self.cached_models:
                    available.append(entry)
                    continue
                try:
                    api.model_info(repo)
                    available.append(entry)
                except Exception:
                    continue
            def apply():
                self.model_entries = available
                self.cached_models = self._get_cached_hf_models()
                self.refresh_hf_model_list()
                self._rebuild_model_dialog(dialog)
                messagebox.showinfo("Models filtered", f"{len(available)} models available", parent=dialog)
            self.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_models_online(self, dialog=None):
        def worker():
            api = HfApi()
            try:
                hf_models = api.list_models(
                    search="faster-whisper",
                    sort="downloads",
                    direction=-1,
                    limit=50,
                )
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Refresh failed", f"Could not fetch model list:\n{e}", parent=dialog or self))
                return

            names = set()
            for m in hf_models:
                mid = m.modelId
                # keep ct2/faster-whisper or distil- exports only
                if "faster-whisper" in mid or "ct2" in mid or mid.startswith("distil-whisper/"):
                    names.add(mid)

            # include cached models not in the list
            cached = self._get_cached_hf_models()
            cache_repos = self._scan_cache_repos()
            for c in cached:
                names.add(c)
            for repo in cache_repos:
                names.add(repo)

            def apply():
                self.cached_models = cached
                self.model_entries = self._build_entries_from_names(names)
                self.cached_models = cached
                self.refresh_hf_model_list()
                self._rebuild_model_dialog(dialog)
                messagebox.showinfo("Models refreshed", f"{len(self.model_entries)} models available.", parent=dialog or self)

            self.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _rescan_cache_and_refresh(self):
        names = set(TIERS.keys()) | self._scan_cache_repos()
        self.cached_models = self._get_cached_hf_models()
        names |= self.cached_models
        self.model_entries = self._build_entries_from_names(names)
        self.refresh_hf_model_list()
        messagebox.showinfo("Cache", f"Cache rescan complete. Models available: {len(self.model_entries)}")

    def _guess_tier_from_name(self, name: str) -> str:
        lname = name.lower()
        if "large" in lname:
            return "PRO"
        if "medium" in lname or "distil" in lname or "turbo" in lname:
            return "BALANCED"
        return "FAST"

    def _rebuild_model_dialog(self, dialog):
        # Rebuilds the scrolling list inside an open dialog after refresh/filter.
        list_frames = dialog.grid_slaves(row=2, column=0)
        if not list_frames:
            return
        list_frame = list_frames[0]
        canvas_widgets = list_frame.grid_slaves(row=0, column=0)
        if not canvas_widgets:
            return
        canvas = canvas_widgets[0]
        if not canvas.children:
            return
        inner = list(canvas.children.values())[0]
        for child in inner.winfo_children():
            child.destroy()

        labels = self._build_model_labels()
        self.model_choice_var.set(self.selected_model_name)
        self.download_vars = {}
        self.model_list_rows = []

        for idx, lbl in enumerate(labels):
            base = self._strip_status(lbl)
            entry = self._find_model_entry_by_label(base)
            name = entry["name"] if entry else base
            row = tk.Frame(inner)
            row.grid(row=idx, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)

            is_recommended = entry and entry["name"] == self.recommended_model_name
            radio_text = f"{'★ ' if is_recommended else ''}{lbl}"
            fg_color = self._tier_color(entry["tier"]) if entry else "black"
            radio = tk.Radiobutton(
                row,
                variable=self.model_choice_var,
                value=name,
                text=radio_text,
                anchor="w",
                justify="left",
                font=self.model_font_bold if is_recommended else self.model_font,
                fg=fg_color,
            )
            radio.grid(row=0, column=1, sticky="w")

            if name in self.cached_models:
                status_lbl = tk.Label(row, text="☑", fg="green")
                status_lbl.grid(row=0, column=0, padx=(0, 6))
            else:
                var = tk.BooleanVar(value=False)
                self.download_vars[name] = var
                chk = tk.Checkbutton(row, variable=var, text="☐", anchor="w")
                chk.grid(row=0, column=0, padx=(0, 6))

            self.model_list_rows.append(row)

        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _hf_xet_available(self):
        try:
            import hf_xet  # noqa: F401
            return True
        except Exception:
            return False

    def _format_size(self, num_bytes: float) -> str:
        if num_bytes is None or num_bytes == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(num_bytes)
        for u in units:
            if size < 1024:
                return f"{size:.1f} {u}"
            size /= 1024
        return f"{size:.1f} PB"

    def _suggested_text(self):
        if self.device == "cuda" and self.vram_gb:
            return f"{self.gpu_name} {self.vram_gb}GB Suggested"
        return "CPU Suggested"

    def _set_media_path(self, path: str):
        """Validate and apply a newly selected/dropped media file."""
        if not path:
            return
        self.stop_audio()
        self._cleanup_temp_dir()
        if not os.path.isfile(path):
            messagebox.showerror("Error", "File not found.")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in ALL_EXTS:
            messagebox.showerror("Error", "Unsupported file type. Please use an audio/video file.")
            return

        self.media_path = path
        self.prepared_audio_path = None
        self.status_var.set(f"Selected: {os.path.basename(path)}")
        self.save_btn.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.transcription = None

    def browse_file(self):
        filetypes = [
            ("Audio/Video Files", " ".join(f"*{ext}" for ext in sorted(list(ALL_EXTS)))),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Media File", filetypes=filetypes)
        if path:
            self._set_media_path(path)

    def _on_drag_enter(self, event):
        event.widget.config(relief="solid", bd=2)
        return event.action

    def _on_drag_leave(self, event):
        event.widget.config(relief="groove", bd=1)
        return event.action

    def _on_file_drop(self, event):
        try:
            paths = self.tk.splitlist(event.data)
            if not paths:
                return
            first = paths[0]
            if first.startswith("{") and first.endswith("}"):
                first = first[1:-1]
            self._set_media_path(first)
        finally:
            self._on_drag_leave(event)

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
