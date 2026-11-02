#!/usr/bin/env python3
"""Prove the native shell works — specifically, that its WebSocket connects.

This exists because of a documented hazard: WebSocket connections to localhost
have historically failed *inside* embedded webviews while working in a normal
browser. Porthole's answer is to serve the UI over http rather than file://, so
the page origin is a real http origin. This script checks that the answer holds
on this machine, with this backend.

The check is unambiguous and needs no screen scraping: if the webview's socket
connected, the Python bus sees a client. If it did not, it sees zero.

    p tools/verify-native-shell.py [--gui qt] [--seconds 10]

Opens a real window for a few seconds, then closes it and reports.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from porthole import PortholeBus  # noqa: E402
from porthole.mock import gradient_frame  # noqa: E402
from porthole.window import serve_ui  # noqa: E402

DIST = REPO_ROOT / "apps/demo/dist"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: object, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def screenshot(path: Path) -> bool:
    """Best-effort window capture. Not required for the verdict."""
    for command in (["grim", str(path)], ["spectacle", "-b", "-n", "-o", str(path)]):
        try:
            subprocess.run(command, check=True, timeout=15, capture_output=True)
            return path.exists()
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", default="qt")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()

    if not (DIST / "index.html").exists():
        print("frontend not built — cd apps/demo && npx vite build")
        return 1

    import webview

    # Must be the default port: the UI dials ws://<host>:8765 unless the app
    # overrides it, and this check exists to exercise the default path.
    bus = PortholeBus(app="native-check").start()
    bus.declare("camera/wrist", qos="stream", kind="image/jpeg")
    bus.declare("log/events", qos="event")
    bus.publish("camera/wrist", bus.jpeg_payload(gradient_frame(0.0, 320, 180)))
    bus.publish("log/events", {"level": "info", "msg": "native shell check"})

    url = serve_ui(DIST, port=args.port)

    state: dict[str, object] = {}

    def observe() -> None:
        deadline = time.monotonic() + args.seconds
        peak = 0
        while time.monotonic() < deadline:
            peak = max(peak, bus.clients)
            time.sleep(0.25)
        state["clients"] = peak
        state["shot"] = screenshot(REPO_ROOT / "tools/native-shell.png")
        for window in webview.windows:
            window.destroy()

    threading.Thread(target=observe, daemon=True).start()

    try:
        webview.create_window("porthole native check", url, width=1280, height=800)
        webview.start(gui=args.gui)
        state["started"] = True
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        state["started"] = False
        state["error"] = repr(exc)

    bus.stop()

    check("native window started", state.get("started"), str(state.get("error", "")))
    check(
        "webview connected over websocket",
        int(state.get("clients", 0) or 0) >= 1,
        f"peak clients={state.get('clients')}",
    )
    if state.get("shot"):
        print(f"screenshot: {REPO_ROOT / 'tools/native-shell.png'}")

    failed = 0
    for name, ok, detail in results:
        if not ok:
            failed += 1
        print(f"{'  ok  ' if ok else ' FAIL '} {name}{f'  ({detail})' if detail and not ok else ''}")
    print(f"\n{len(results) - failed}/{len(results)} native-shell checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
