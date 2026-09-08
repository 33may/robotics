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


def _present(url: str, no_window: bool, gui: str, open_browser: bool = True) -> None:
    print(f"ui   {url}", flush=True)
    if no_window:
        if open_browser:
            webbrowser.open(url)
        print("ctrl-c to stop", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    else:
        open_window(url, title="inspection", width=1700, height=1000, gui=gui)


def mock(port: int = 8767, bus_port: int = 8765, seed: int = 0,
         no_window: bool = False, gui: str = "qt", collect: bool = False,
         run_root: str | None = None, open_browser: bool = True,
         auto: bool = False) -> int:
    """A real Supervisor + FakeRig run with no hardware, driven over the real bus.

    `bus_port` is passed to the UI as `?bus=`, so a mock can run beside a real
    loop without the two fighting over the default port.

    `collect` swaps Flow A (`start_mock` — ask, survey, brain-driven views)
    for Flow B (`start_mock_collect` — no cognition, `SweepDriver` walks the
    whole shell) — same bus, same rig, same window entry point.

    `run_root`, given, replaces this command's own tempdir for the run
    directory's parent — lets a caller (the e2e harness) know the run dir up
    front instead of scraping stdout. `open_browser=False` skips the
    `webbrowser.open` call under `--no_window` — a harness driving its own
    browser (Playwright) does not want a second, real one popping up too.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not _require_build():
        return 1

    from inspection.ui.mock import start_mock, start_mock_collect

    bus = PortholeBus(app="inspection", port=bus_port).start()
    # The mock writes real capture files now, so give the publisher its run
    # dir and MOUNT it: without both, every capture-image URL is dropped and
    # the cloud/chain panels are blank in a way no check would notice.
    # NOT created here: `start_mock`'s RunWriter creates the run directory,
    # and it refuses one that already exists (evidence is never overwritten).
    mock_run = (Path(run_root) if run_root else Path(tempfile.mkdtemp())) / "mock-run"
    pub = InspectionPublisher(bus, run_dir=mock_run)
    print(f"bus  ws://127.0.0.1:{bus.port}", flush=True)

    if collect:
        # `auto` only means anything to Flow B — Flow A's gate is the point.
        start_mock_collect(bus, pub, mock_run, seed=seed, auto=auto)
    else:
        start_mock(bus, pub, mock_run, seed=seed)

    url = serve_ui(DIST, port=port, assets=_asset_mounts(mock_run))
    if bus_port != 8765:
        url = f"{url}/?bus={bus_port}"
    _present(url, no_window, gui, open_browser=open_browser)
    bus.stop()
    return 0


def serve(run_dir: str | None = None, port: int = 8767, bus_port: int = 8765,
          no_window: bool = False, gui: str = "qt") -> int:
    """Serve the UI and open the window. No bus — the loop process owns that.

    `run/app.py run` spawns exactly this as a child process for its native
    window: pywebview must own a main thread, and that process's main thread
    is the dispatcher. `bus_port` is passed through as `?bus=` so the window
    finds a loop that is not on the default port.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not _require_build():
        return 1
    print(f"no bus started here — the loop process owns "
          f"ws://127.0.0.1:{bus_port}", flush=True)
    url = serve_ui(DIST, port=port, assets=_asset_mounts(run_dir))
    if bus_port != 8765:
        url = f"{url}/?bus={bus_port}"
    _present(url, no_window, gui)
    return 0


def trace(run_dir: str, port: int = 8767, bus_port: int = 8765,
          no_window: bool = False, gui: str = "qt", poll: float = 0.5) -> int:
    """Open a brain run's trace: every stage of the agentic loop, as a stream.

        p inspection/ui/app.py trace --run_dir=inspection/data/runs/2408-cup1

    Owns a bus, but drives no robot and plans nothing — it reads the newest
    `<run>/ai/<seq>/trace.jsonl` and republishes it whenever the file changes.
    That is the whole live story too: a run in progress is just a file still
    being appended to, so watching one and re-opening a finished one are the
    same code path (see `brain/trace.py`).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not _require_build():
        return 1

    from inspection.brain.trace import TRACE_NAME, latest_trace_dir, read_trace

    run_path = Path(run_dir).expanduser().resolve()
    ai_dir = latest_trace_dir(run_path)
    if ai_dir is None:
        print(f"no trace under {run_path}/ai/*/{TRACE_NAME} — "
              f"run inspection/brain/loop.py first")
        return 1
    trace_file = ai_dir / TRACE_NAME

    bus = PortholeBus(app="inspection", port=bus_port).start()
    pub = InspectionPublisher(bus, run_dir=run_path)
    try:  # the viewed run says which flow it was — the UI shapes itself to it
        from inspection.record.run import Run
        rec = Run.load(run_path).record
        pub.publish_run_meta(source=rec.source, name=rec.name,
                             object=rec.object, question=rec.question)
    except Exception:
        logging.getLogger("inspection.ui").exception("run/meta publish failed")
    print(f"bus  ws://127.0.0.1:{bus.port}", flush=True)

    def watch() -> None:
        stamp = None
        while True:
            try:
                now = trace_file.stat().st_mtime_ns
                if now != stamp:
                    stamp = now
                    events = read_trace(ai_dir)
                    pub.publish_trace(events)
                    pub.log("info", f"trace: {len(events)} events")
            except Exception:
                logging.getLogger("inspection.ui").exception("trace watch failed")
            threading.Event().wait(poll)

    threading.Thread(target=watch, daemon=True).start()

    url = serve_ui(DIST, port=port, assets=_asset_mounts(run_path))
    if bus_port != 8765:
        url = f"{url}/?bus={bus_port}"
    _present(url, no_window, gui)
    bus.stop()
    return 0


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    sys.exit(fire.Fire({"mock": mock, "serve": serve, "trace": trace}) or 0)
