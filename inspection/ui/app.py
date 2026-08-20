#!/usr/bin/env python3
"""Run the inspection UI.

    p inspection/ui/app.py mock                       # fake run, no robot, no camera
    p inspection/ui/app.py mock --no_window           # same, browser at :8767
    p inspection/ui/app.py serve --run_dir=data/...   # window only; your loop owns the bus

`mock` owns a bus and drives every topic itself. `serve` owns no bus at all: it
serves the built frontend and opens the window, and the loop process publishes
into it. That split exists because there can be exactly one bus on a port, and
the bus belongs to whoever owns the robot.

Build the frontend first:
    cd ~/projects/porthole && npm run build -w @porthole/framework
    cd inspection/ui && npm install && npm run build
"""

from __future__ import annotations

import logging
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

from porthole import PortholeBus
from porthole.window import open_window, serve_ui

from inspection.ui.publisher import CAPTURE_MOUNT, MESH_MOUNT, InspectionPublisher, mesh_dir_for

UI_DIR = Path(__file__).resolve().parent
DIST = UI_DIR / "dist"


def _asset_mounts(run_dir: str | Path | None) -> dict[str, str | Path]:
    """Directories the browser fetches from: robot meshes and captured images.

    Neither belongs on the bus — 9 MB of DAE and an 848x480 PNG per view, all
    of it static, all of it something the browser already caches and decodes
    off the main thread. See AGENTS.md §7.
    """
    from inspection.cell.world import RobotCell

    mounts: dict[str, str | Path] = {}
    mesh_dir = mesh_dir_for(RobotCell())
    if mesh_dir and mesh_dir.is_dir():
        mounts[MESH_MOUNT] = mesh_dir
    else:
        print("WARNING: robot meshes not found — the 3D view will show boxes only")
    if run_dir:
        run_path = Path(run_dir).expanduser().resolve()
        if run_path.is_dir():
            mounts[CAPTURE_MOUNT] = run_path
        else:
            print(f"WARNING: run dir {run_path} does not exist — no capture images")
    return mounts


def _require_build() -> bool:
    if (DIST / "index.html").exists():
        return True
    print(f"frontend not built — run: cd {UI_DIR} && npm install && npm run build")
    return False


def _present(url: str, no_window: bool, gui: str) -> None:
    print(f"ui   {url}", flush=True)
    if no_window:
        webbrowser.open(url)
        print("ctrl-c to stop", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    else:
        open_window(url, title="inspection", width=1700, height=1000, gui=gui)


def mock(port: int = 8767, bus_port: int = 8765, seed: int = 0,
         no_window: bool = False, gui: str = "qt") -> int:
    """A real Supervisor + FakeRig run with no hardware, driven over the real bus.

    `bus_port` is passed to the UI as `?bus=`, so a mock can run beside a real
    loop without the two fighting over the default port.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not _require_build():
        return 1

    from inspection.ui.mock import start_mock

    bus = PortholeBus(app="inspection", port=bus_port).start()
    pub = InspectionPublisher(bus)
    print(f"bus  ws://127.0.0.1:{bus.port}", flush=True)

    start_mock(bus, pub, Path(tempfile.mkdtemp()) / "mock-run", seed=seed)

    url = serve_ui(DIST, port=port, assets=_asset_mounts(None))
    if bus_port != 8765:
        url = f"{url}/?bus={bus_port}"
    _present(url, no_window, gui)
    bus.stop()
    return 0


def serve(run_dir: str | None = None, port: int = 8767,
          no_window: bool = False, gui: str = "qt") -> int:
    """Serve the UI and open the window. No bus — the loop process owns that."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not _require_build():
        return 1
    print("no bus started here — the loop process owns ws://127.0.0.1:8765", flush=True)
    _present(serve_ui(DIST, port=port, assets=_asset_mounts(run_dir)), no_window, gui)
    return 0


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    sys.exit(fire.Fire({"mock": mock, "serve": serve}) or 0)
