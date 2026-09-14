#!/usr/bin/env python
"""Minimal Chrome DevTools Protocol client for the UI debug loop.

Why raw CDP instead of the chrome-devtools MCP tools:
  * the MCP `click`/`drag` tools address elements by DOM *uid* only.  A WASM app
    (Rerun, any egui/eframe build, Flutter web, Unity web) is a single <canvas>
    with no DOM children, so there is nothing to address.  We need pixel coords.
  * `Input.dispatchMouseEvent` produces **trusted** browser-level input (the
    events carry isTrusted=true and go through real hit-testing), which is what
    "real input events" means.  JS-dispatched `new PointerEvent(...)` is
    untrusted and is a strictly worse fallback.
  * shell scripts cannot call MCP tools; the toolkit has to stand alone.

Coordinate model (IMPORTANT):
  We pin `Emulation.setDeviceMetricsOverride(deviceScaleFactor=1)`, so
      CSS pixel == screenshot pixel == the number you pass to Input.*
  exactly, 1:1, regardless of the host display's devicePixelRatio (this machine
  reports dpr=1.25, which would otherwise silently scale every coordinate).

Only dependency: `websockets` (present in the robo env).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from websockets.sync.client import connect as ws_connect

ART = Path(os.environ.get("UITEST_DIR", "/tmp/rrd_ui_test"))
SESSION_FILE = ART / "session.json"
PIDS_FILE = ART / "pids"

DEFAULT_W, DEFAULT_H = 1600, 1000

CHROME_CANDIDATES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/opt/google/chrome/chrome",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.35)
        return s.connect_ex((host, port)) == 0


def wait_port(port: int, timeout: float = 30.0, host: str = "127.0.0.1") -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if port_open(port, host):
            return True
        time.sleep(0.2)
    return False


def track_pid(pid: int, label: str) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    with PIDS_FILE.open("a") as f:
        f.write(f"{pid}\t{label}\n")


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    raise RuntimeError(f"no chrome binary found, tried {CHROME_CANDIDATES}")


# --------------------------------------------------------------------------- #
# CDP session
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    port: int
    ws_url: str
    width: int = DEFAULT_W
    height: int = DEFAULT_H

    def save(self) -> None:
        ART.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(self.__dict__, indent=2))

    @staticmethod
    def load() -> "Session":
        if not SESSION_FILE.exists():
            raise RuntimeError(f"no session at {SESSION_FILE}; run launch.py first")
        return Session(**json.loads(SESSION_FILE.read_text()))


class CDP:
    """One websocket to one page target. Use as a context manager."""

    def __init__(self, ws_url: str, timeout: float = 30.0):
        self.ws_url = ws_url
        self.timeout = timeout
        self._id = 0
        self._ws = None

    def __enter__(self) -> "CDP":
        # max_size=None: screenshots arrive as multi-MB base64 payloads and the
        # 1 MiB default silently kills the connection.
        self._ws = ws_connect(self.ws_url, max_size=None, open_timeout=self.timeout)
        return self

    def __exit__(self, *exc) -> None:
        if self._ws is not None:
            self._ws.close()

    def send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        mid = self._id
        self._ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        end = time.time() + self.timeout
        while time.time() < end:
            raw = self._ws.recv(timeout=max(0.1, end - time.time()))
            msg = json.loads(raw)
            if msg.get("id") == mid:  # events have no "id"; skip them
                if "error" in msg:
                    raise RuntimeError(f"{method} failed: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(f"no CDP reply for {method}")

    # -- setup ------------------------------------------------------------- #
    def prepare(self, width: int, height: int) -> None:
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": width,
                "height": height,
                "deviceScaleFactor": 1,  # <- makes screenshot px == input px
                "mobile": False,
                "screenWidth": width,
                "screenHeight": height,
            },
        )

    def navigate(self, url: str) -> None:
        self.send("Page.navigate", {"url": url})

    def eval(self, expr: str) -> Any:
        r = self.send(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
        )
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"].get("text", "eval failed"))
        return r.get("result", {}).get("value")

    # -- observe ----------------------------------------------------------- #
    def screenshot(self, path: str | Path) -> Path:
        r = self.send(
            "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(base64.b64decode(r["data"]))
        return p

    # -- act --------------------------------------------------------------- #
    def _mouse(self, type_: str, x: float, y: float, button="none", buttons=0, clicks=0):
        self.send(
            "Input.dispatchMouseEvent",
            {
                "type": type_,
                "x": x,
                "y": y,
                "button": button,
                "buttons": buttons,
                "clickCount": clicks,
                "pointerType": "mouse",
            },
        )

    def move(self, x: float, y: float, buttons: int = 0) -> None:
        self._mouse("mouseMoved", x, y, buttons=buttons)

    def click(self, x: float, y: float, button: str = "left", clicks: int = 1,
              settle: float = 0.25) -> None:
        bmask = {"left": 1, "right": 2, "middle": 4}[button]
        # Hover first: egui resolves *which widget* from the previous pointer
        # position, so a press with no preceding move can land on nothing.
        self._mouse("mouseMoved", x, y)
        time.sleep(0.05)
        self._mouse("mousePressed", x, y, button=button, buttons=bmask, clicks=clicks)
        time.sleep(0.05)
        self._mouse("mouseReleased", x, y, button=button, buttons=0, clicks=clicks)
        time.sleep(settle)

    def drag(self, x1: float, y1: float, x2: float, y2: float, steps: int = 24,
             settle: float = 0.35) -> None:
        self._mouse("mouseMoved", x1, y1)
        time.sleep(0.05)
        self._mouse("mousePressed", x1, y1, button="left", buttons=1, clicks=1)
        time.sleep(0.05)
        for i in range(1, steps + 1):
            t = i / steps
            # intermediate moves are REQUIRED: egui starts a drag only after it
            # sees motion while the button is held.
            self._mouse("mouseMoved", x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, buttons=1)
            time.sleep(0.012)
        time.sleep(0.05)
        self._mouse("mouseReleased", x2, y2, button="left", buttons=0, clicks=1)
        time.sleep(settle)

    def wheel(self, x: float, y: float, dx: float = 0, dy: float = -120,
              settle: float = 0.3) -> None:
        self._mouse("mouseMoved", x, y)
        self.send(
            "Input.dispatchMouseEvent",
            {"type": "mouseWheel", "x": x, "y": y, "deltaX": dx, "deltaY": dy,
             "button": "none", "buttons": 0, "pointerType": "mouse"},
        )
        time.sleep(settle)

    def key(self, key: str, settle: float = 0.2) -> None:
        spec = KEYS.get(key)
        if spec is None:
            # printable character
            self.send("Input.dispatchKeyEvent",
                      {"type": "keyDown", "text": key, "key": key})
            self.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": key})
        else:
            code, vk = spec
            base = {"key": key, "code": code, "windowsVirtualKeyCode": vk,
                    "nativeVirtualKeyCode": vk}
            self.send("Input.dispatchKeyEvent", {"type": "rawKeyDown", **base})
            self.send("Input.dispatchKeyEvent", {"type": "keyUp", **base})
        time.sleep(settle)

    def text(self, s: str, settle: float = 0.2) -> None:
        self.send("Input.insertText", {"text": s})
        time.sleep(settle)


KEYS: dict[str, tuple[str, int]] = {
    "Enter": ("Enter", 13), "Escape": ("Escape", 27), "Tab": ("Tab", 9),
    "Backspace": ("Backspace", 8), "Delete": ("Delete", 46), "Space": ("Space", 32),
    "ArrowLeft": ("ArrowLeft", 37), "ArrowUp": ("ArrowUp", 38),
    "ArrowRight": ("ArrowRight", 39), "ArrowDown": ("ArrowDown", 40),
    "Home": ("Home", 36), "End": ("End", 35),
    "PageUp": ("PageUp", 33), "PageDown": ("PageDown", 34),
}


# --------------------------------------------------------------------------- #
# browser lifecycle
# --------------------------------------------------------------------------- #
def launch_chrome(width: int = DEFAULT_W, height: int = DEFAULT_H,
                  headless: bool = True, profile: Path | None = None) -> Session:
    """Start an isolated Chrome with a CDP port and return a Session.

    Isolated profile on purpose: never attach to the user's own browser.
    """
    port = free_port()
    profile = profile or (ART / "chrome-profile")
    profile.mkdir(parents=True, exist_ok=True)
    args = [
        find_chrome(),
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        f"--window-size={width},{height}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-features=Translate,MediaRouter",
        "--disable-search-engine-choice-screen",
        "--noerrdialogs",
        "--enable-unsafe-swiftshader",  # CPU fallback if the GPU path is refused
        "about:blank",
    ]
    if headless:
        # "new" headless keeps real GPU/WebGPU support, unlike old headless.
        args.insert(1, "--headless=new")
        args.insert(2, "--hide-scrollbars")

    ART.mkdir(parents=True, exist_ok=True)
    log = (ART / "chrome.log").open("ab")
    proc = subprocess.Popen(args, stdout=log, stderr=log, start_new_session=True)
    track_pid(proc.pid, "chrome")
    if not wait_port(port, 30):
        raise RuntimeError(f"chrome CDP port {port} never opened; see {ART/'chrome.log'}")

    ws = _page_ws(port)
    sess = Session(port=port, ws_url=ws, width=width, height=height)
    with CDP(ws) as c:
        c.prepare(width, height)
    sess.save()
    return sess


def _page_ws(port: int, tries: int = 40) -> str:
    import urllib.request

    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                for t in json.loads(r.read()):
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("no page target exposed by chrome")


def attach() -> tuple[Session, CDP]:
    """Attach to the session written by launch.py. Caller must `with` the CDP."""
    s = Session.load()
    return s, CDP(s.ws_url)


def kill(pid: int) -> bool:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False
