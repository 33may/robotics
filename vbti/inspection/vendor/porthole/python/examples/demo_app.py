#!/usr/bin/env python3
"""The demo as a native desktop app — one command, one process.

    p python/examples/demo_app.py
    p python/examples/demo_app.py --browser     # same UI, no native shell

This is the shape a real Porthole app takes: start the bus, start whatever
produces data, then serve the built frontend and open a window on it. The window
is a client like any other — it connects over the same socket a browser would,
which is why "open it on my other machine" costs nothing.

Build the frontend first:  cd apps/demo && npx vite build
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from demo_backend import demo_mesh_assets, handle_demo_commands, start_demo_backend

from porthole import PortholeBus
from porthole.window import open_window, serve_ui

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST = REPO_ROOT / "apps/demo/dist"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--browser", action="store_true", help="skip the native shell")
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="serve and block, opening nothing — what the headless check runs",
    )
    parser.add_argument("--gui", default="qt", help="pywebview backend (qt, gtk)")
    parser.add_argument("--port", type=int, default=8766, help="http port for the UI")
    args = parser.parse_args()

    if not (DIST / "index.html").exists():
        print(f"frontend not built — run: cd {REPO_ROOT}/apps/demo && npx vite build")
        return 1

    assets, mesh_url = demo_mesh_assets()

    bus = PortholeBus(app="demo").start()
    start_demo_backend(bus, mesh_url)
    threading.Thread(target=handle_demo_commands, args=(bus,), daemon=True).start()

    url = serve_ui(DIST, port=args.port, assets=assets)
    print(f"bus  ws://127.0.0.1:{bus.port}")
    print(f"ui   {url}")
    print(f"mesh {mesh_url or 'none found — primitives only'}")

    if args.browser or args.serve_only:
        if args.browser:
            webbrowser.open(url)
        print("ctrl-c to stop", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    else:
        open_window(url, title="porthole demo", gui=args.gui)

    bus.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
