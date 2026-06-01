"""Modern Tkinter GUI utilities for live inference demos."""

from __future__ import annotations

import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable

import numpy as np
from PIL import Image, ImageTk


@dataclass
class PromptGuiStatus:
    prompt: str
    version: int
    source: str
    mode: str
    step: int = 0
    running: bool = True


class FullInferenceGui:
    """Single modern demo window: live camera view + prompt controls."""

    def __init__(
        self,
        get_status: Callable[[], PromptGuiStatus],
        on_prompt: Callable[[str, str], None],
        on_mode: Callable[[str], None],
        red_prompt: str,
        black_prompt: str,
        title: str = "VBTI Robot Inference Demo",
    ):
        self.get_status = get_status
        self.on_prompt = on_prompt
        self.on_mode = on_mode
        self.red_prompt = red_prompt
        self.black_prompt = black_prompt
        self.title = title

        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._closed = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_bgr: np.ndarray | None = None
        self._tk_image = None

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="FullInferenceGui", daemon=True)
        self._thread.start()

    def close(self):
        self._closed.set()
        root = self._root
        if root is not None:
            try:
                root.after(0, root.destroy)
            except Exception:
                pass

    def update_frame(self, bgr_frame: np.ndarray):
        with self._frame_lock:
            self._latest_bgr = bgr_frame.copy()

    def _pop_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return None if self._latest_bgr is None else self._latest_bgr.copy()

    def _run(self):
        root = tk.Tk()
        self._root = root
        root.title(self.title)
        root.geometry("1480x880")
        root.minsize(1180, 760)

        palette = {
            "bg": "#050816",
            "panel": "#0B1220",
            "panel2": "#0F172A",
            "card": "#111C31",
            "card_hi": "#17233A",
            "stroke": "#263655",
            "stroke_soft": "#1E2A44",
            "text": "#F8FAFC",
            "muted": "#94A3B8",
            "muted2": "#64748B",
            "cyan": "#22D3EE",
            "blue": "#2563EB",
            "blue_hi": "#3B82F6",
            "red": "#EF4444",
            "red_hi": "#F87171",
            "black": "#020617",
            "green": "#22C55E",
            "amber": "#F59E0B",
        }
        root.configure(bg=palette["bg"])

        families = set(tkfont.families(root))
        font = "Inter" if "Inter" in families else "Segoe UI" if "Segoe UI" in families else "DejaVu Sans"
        mono = "JetBrains Mono" if "JetBrains Mono" in families else "DejaVu Sans Mono"

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Mode.TRadiobutton", background=palette["card"], foreground=palette["text"], font=(font, 12))
        style.map("Mode.TRadiobutton", background=[("active", palette["card_hi"])], foreground=[("active", palette["text"])])

        def card(parent, **grid):
            f = tk.Frame(parent, bg=palette["card"], highlightbackground=palette["stroke_soft"], highlightthickness=1)
            f.grid(**grid)
            return f

        def label(parent, text, size=12, weight="normal", fg=None, bg=None, **kw):
            return tk.Label(
                parent,
                text=text,
                bg=bg or parent.cget("bg"),
                fg=fg or palette["text"],
                font=(font, size, weight),
                **kw,
            )

        def make_button(parent, text, command, bg, hover, fg="#FFFFFF", size=13, weight="bold", pady=12):
            btn = tk.Button(
                parent,
                text=text,
                command=command,
                bg=bg,
                fg=fg,
                activebackground=hover,
                activeforeground=fg,
                relief="flat",
                bd=0,
                highlightthickness=0,
                pady=pady,
                font=(font, size, weight),
                cursor="hand2",
            )
            btn.bind("<Enter>", lambda _e: btn.configure(bg=hover))
            btn.bind("<Leave>", lambda _e: btn.configure(bg=bg))
            return btn

        # ── Header ────────────────────────────────────────────────────────
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = tk.Frame(root, bg=palette["bg"])
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(20, 14))
        header.columnconfigure(0, weight=1)

        title_block = tk.Frame(header, bg=palette["bg"])
        title_block.grid(row=0, column=0, sticky="w")
        label(title_block, "VBTI", size=13, weight="bold", fg=palette["cyan"]).pack(anchor="w")
        label(title_block, "Live Robot Inference", size=30, weight="bold").pack(anchor="w")
        label(title_block, "Vision feedback + real-time language command control", size=12, fg=palette["muted"]).pack(anchor="w", pady=(3, 0))

        status_pill = tk.Label(
            header,
            text="● RUNNING",
            bg="#052E1A",
            fg="#86EFAC",
            font=(font, 12, "bold"),
            padx=18,
            pady=8,
        )
        status_pill.grid(row=0, column=1, sticky="e")

        # ── Main split ────────────────────────────────────────────────────
        main = tk.Frame(root, bg=palette["bg"])
        main.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 28))
        main.columnconfigure(0, weight=7)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        camera = card(main, row=0, column=0, sticky="nsew", padx=(0, 14))
        camera.rowconfigure(1, weight=1)
        camera.columnconfigure(0, weight=1)

        cam_head = tk.Frame(camera, bg=palette["card"])
        cam_head.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))
        cam_head.columnconfigure(0, weight=1)
        label(cam_head, "LIVE CAMERA GRID", size=12, weight="bold", fg=palette["cyan"]).grid(row=0, column=0, sticky="w")
        cam_meta = tk.StringVar(value="waiting for frames")
        tk.Label(cam_head, textvariable=cam_meta, bg=palette["card"], fg=palette["muted"], font=(mono, 10)).grid(row=0, column=1, sticky="e")

        video_shell = tk.Frame(camera, bg="#020617", highlightbackground=palette["stroke"], highlightthickness=1)
        video_shell.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        video_shell.rowconfigure(0, weight=1)
        video_shell.columnconfigure(0, weight=1)
        video_label = tk.Label(video_shell, bg="#020617", fg=palette["muted2"], text="Waiting for camera frames...", font=(font, 18, "bold"))
        video_label.grid(row=0, column=0, sticky="nsew")

        sidebar = tk.Frame(main, bg=palette["bg"])
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)

        # Active command card
        active_card = card(sidebar, row=0, column=0, sticky="ew", pady=(0, 14))
        active_card.columnconfigure(0, weight=1)
        label(active_card, "ACTIVE ROBOT COMMAND", size=11, weight="bold", fg=palette["cyan"]).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 6))
        active_var = tk.StringVar(value="")
        tk.Label(active_card, textvariable=active_var, bg=palette["card"], fg=palette["text"], font=(font, 16, "bold"), wraplength=410, justify="left").grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        meta_var = tk.StringVar(value="")
        tk.Label(active_card, textvariable=meta_var, bg=palette["card"], fg=palette["muted"], font=(mono, 10), justify="left").grid(row=2, column=0, sticky="w", padx=18, pady=(0, 16))

        controls = card(sidebar, row=1, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(5, weight=1)

        label(controls, "Quick Demo Targets", size=16, weight="bold").grid(row=0, column=0, sticky="w", padx=18, pady=(18, 10))
        red_btn = make_button(controls, "🔴  RED CUP", lambda: self.on_prompt(self.red_prompt, "gui-red"), palette["red"], palette["red_hi"], size=20, pady=20)
        red_btn.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))
        black_btn = make_button(controls, "⚫  BLACK CUP", lambda: self.on_prompt(self.black_prompt, "gui-black"), palette["black"], "#111827", size=20, pady=20)
        black_btn.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))

        label(controls, "Custom Command", size=16, weight="bold").grid(row=3, column=0, sticky="w", padx=18, pady=(0, 8))
        prompt_text = tk.Text(
            controls,
            height=7,
            bg="#020617",
            fg=palette["text"],
            insertbackground=palette["cyan"],
            relief="flat",
            bd=0,
            highlightbackground=palette["stroke"],
            highlightcolor=palette["cyan"],
            highlightthickness=1,
            padx=14,
            pady=12,
            font=(font, 12),
            wrap="word",
        )
        prompt_text.grid(row=4, column=0, sticky="nsew", padx=18, pady=(0, 12))

        def apply_custom():
            text = prompt_text.get("1.0", "end").strip()
            if text:
                self.on_prompt(text, "gui")

        send_btn = make_button(controls, "SEND PROMPT TO ROBOT", apply_custom, palette["blue"], palette["blue_hi"], size=12, pady=13)
        send_btn.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 18))

        mode_card = tk.Frame(controls, bg=palette["panel2"], highlightbackground=palette["stroke_soft"], highlightthickness=1)
        mode_card.grid(row=6, column=0, sticky="ew", padx=18, pady=(0, 16))
        mode_card.columnconfigure(0, weight=1)
        label(mode_card, "Update Mode", size=13, weight="bold", bg=palette["panel2"]).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))
        mode_var = tk.StringVar(value="smooth")
        mode_frame = tk.Frame(mode_card, bg=palette["panel2"])
        mode_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        ttk.Radiobutton(mode_frame, text="Smooth", variable=mode_var, value="smooth", style="Mode.TRadiobutton", command=lambda: self.on_mode(mode_var.get())).pack(anchor="w", pady=2)
        ttk.Radiobutton(mode_frame, text="Responsive", variable=mode_var, value="responsive", style="Mode.TRadiobutton", command=lambda: self.on_mode(mode_var.get())).pack(anchor="w", pady=2)
        tk.Label(mode_card, text="Smooth: waits for chunk refresh\nResponsive: resets action runner immediately", bg=palette["panel2"], fg=palette["muted"], font=(font, 10), justify="left").grid(row=2, column=0, sticky="w", padx=14, pady=(0, 12))

        stop_btn = make_button(controls, "STOP INFERENCE", root.destroy, "#334155", "#475569", fg=palette["text"], size=12, pady=12)
        stop_btn.grid(row=7, column=0, sticky="ew", padx=18, pady=(0, 18))

        def refresh_status():
            if self._closed.is_set():
                root.destroy()
                return
            try:
                s = self.get_status()
                active_var.set(s.prompt)
                meta_var.set(f"v{s.version}  •  source={s.source}  •  mode={s.mode}  •  step={s.step}")
                status_pill.configure(
                    text="● RUNNING" if s.running else "● STOPPED",
                    bg="#052E1A" if s.running else "#3A1A1A",
                    fg="#86EFAC" if s.running else "#FCA5A5",
                )
                if mode_var.get() != s.mode:
                    mode_var.set(s.mode)
                if not prompt_text.get("1.0", "end").strip():
                    prompt_text.insert("1.0", s.prompt)
            except Exception as exc:
                meta_var.set(f"GUI status error: {exc}")
            root.after(120, refresh_status)

        def refresh_frame():
            frame = self._pop_frame()
            if frame is not None:
                rgb = frame[:, :, ::-1]
                img = Image.fromarray(rgb)
                max_w = max(420, video_label.winfo_width() - 12)
                max_h = max(320, video_label.winfo_height() - 12)
                img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                self._tk_image = ImageTk.PhotoImage(img)
                video_label.configure(image=self._tk_image, text="")
                cam_meta.set(f"{frame.shape[1]}×{frame.shape[0]}  •  live")
            root.after(33, refresh_frame)

        root.protocol("WM_DELETE_WINDOW", root.destroy)
        refresh_status()
        refresh_frame()
        root.mainloop()
        self._closed.set()


# Backwards-compatible alias for older imports. The prompt GUI is now the full
# inference/demo GUI because the prompt controls are part of the robot UI.
PromptControlGui = FullInferenceGui
