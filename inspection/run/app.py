#!/usr/bin/env python3
"""Loop v2 composition root. Design: inspection/2026-08-20-ui-driven-loop-design.md.

    p inspection/run/app.py run --outdir=data/runs/r1     # the real thing
    p inspection/run/app.py teach                         # save survey pose
"""
import json, logging, signal, subprocess, sys, threading, webbrowser
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

ROBOT_IP = "192.168.2.50"
SURVEY_POSE_FILE = Path(__file__).resolve().parent / "survey_pose.json"
UI_APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"


def install_sigint(sup):
    """First Ctrl-C: safe shutdown (stop arm, save, close). Second: hard exit.

    Install this as early as the Supervisor exists: everything after it —
    opening the camera, starting threads, waiting for a window — is time the
    operator may want to abort, and until it is installed the default handler
    tears the process down with the arm live.
    """
    def handler(signum, frame):
        signal.signal(signal.SIGINT, signal.default_int_handler)
        sup.request_shutdown()
    signal.signal(signal.SIGINT, handler)


def _open_ui(outdir, port, bus_port, no_window, gui):
    """Show the UI. Returns the window's child process, or None.

    The native window is a CHILD PROCESS, not a thread: pywebview refuses to
    start off the main thread, and this process's main thread is the
    dispatcher (and the only thread that may receive signals). So the window
    gets a main thread of its own — `inspection/ui/app.py serve`, which serves
    the built frontend and dials this process's bus over the socket like any
    other client. It owns no run state; killing it, or closing the window,
    cannot touch the run.
    """
    url = f"http://127.0.0.1:{port}" + (f"/?bus={bus_port}" if bus_port != 8765 else "")
    print(f"ui   {url}", flush=True)
    if no_window:
        from porthole.window import serve_ui
        from inspection.ui.app import DIST, _asset_mounts
        serve_ui(DIST, port=port, assets=_asset_mounts(outdir))
        webbrowser.open(url)
        return None
    cmd = [sys.executable, str(UI_APP), "serve", f"--run_dir={outdir}",
           f"--port={port}", f"--gui={gui}"]
    if bus_port != 8765:
        cmd.append(f"--bus_port={bus_port}")
    return subprocess.Popen(cmd)


def _close_ui(child):
    """Terminate the window process, bounded. Never raises into shutdown."""
    if child is None or child.poll() is not None:
        return
    try:
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            log.warning("ui window did not exit in 3 s — killing it")
            child.kill()
            child.wait(timeout=1)
    except Exception:
        log.exception("closing the ui window failed")


def teach(ip: str = ROBOT_IP):
    # unchanged from v1 (loop.py, retired — see the 2026-08-20 design doc)
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(ip)
    q = list(r.getActualQ())
    r.disconnect()
    SURVEY_POSE_FILE.write_text(json.dumps({"q_rad": q}, indent=2) + "\n")
    print(f"survey pose saved: {np.round(np.degrees(q), 1).tolist()} deg")


def run(outdir: str, ip: str = ROBOT_IP, r: float | None = None,
        port: int = 8767, bus_port: int = 8765, no_window: bool = False,
        seed: int = 0, gui: str = "qt"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from porthole import PortholeBus
    from inspection.motion.execute import preflight
    from inspection.run.machine import Supervisor
    from inspection.run.rigs import PoseStreamer, RealRig
    from inspection.run.segmenter import ObjectSegmenter
    from inspection.ui.app import _require_build
    from inspection.ui.publisher import InspectionPublisher

    pf = preflight(ip)
    if not pf["go"]:
        raise SystemExit(f"preflight NO-GO: {pf}")
    if not SURVEY_POSE_FILE.exists():
        raise SystemExit("no survey pose — run `p inspection/run/app.py teach`")
    if not _require_build():
        raise SystemExit(1)
    q_survey = np.array(json.loads(SURVEY_POSE_FILE.read_text())["q_rad"])

    outdir = Path(outdir).resolve()
    stop_event = threading.Event()
    # Everything that owns a socket, hardware or a thread is built inside the
    # try, so a failure part-way through setup still runs the teardown below
    # instead of leaking a listening bus, an open RTDE interface and a running
    # camera pipe.
    bus = rig = sup = poses = child = None
    try:
        bus = PortholeBus(app="inspection", port=bus_port).start()
        pub = InspectionPublisher(bus, run_dir=outdir)
        pub.declare()
        rig = RealRig(None, stop_event, outdir, ip)   # world set below
        # Object identity is settled in image space (run 2408-cup2): a cable
        # 20 mm from the cup is adjacent, so no distance rule can refuse it.
        # Weights load lazily on the first capture, not here.
        sup = Supervisor(rig, pub, outdir, q_survey, seed=seed, r=r,
                         segmenter=ObjectSegmenter())
        install_sigint(sup)                          # earliest safe Ctrl-C
        rig.world = sup.world
        rig.start_camera(pub)
        poses = PoseStreamer(rig.q, lambda q: pub.publish_pose(sup.world, q))
        poses.active = sup.pose_active               # dispatcher-gated
        sup.pose_quiesce = poses.quiesce             # settle-leg interlock
        poses.start()

        from inspection.brain.live import make_ask_handler
        start_brain = make_ask_handler(sup, pub, outdir)

        def pump():
            for c in bus.commands():
                # `brain/ask` is the ONLY command not forwarded to the
                # Supervisor: it starts a thinker, not a motion. Everything
                # that can move the arm still goes through the one queue.
                if c.get("cmd") == "brain/ask":
                    start_brain(str(c.get("question") or "").strip())
                else:
                    sup.events.put(c)
        threading.Thread(target=pump, name="cmd-pump", daemon=True).start()

        child = _open_ui(outdir, port, bus_port, no_window, gui)
        sup.run()                                    # blocks until done
    finally:
        # belt-and-braces shutdown, in this order (design §Safety):
        # stop event -> join worker (bounded) -> stopJ -> save -> close.
        if sup is not None:
            sup.stop_event.set()
            # Clearing the pose gate is what makes the exec worker's I4
            # handoff winnable on an ABNORMAL exit: if `sup.run()` died (an
            # escaped exception, a second Ctrl-C) with a worker waiting on
            # `_pose_ack`, no dispatcher is left to clear `pose_active`, so
            # its `quiesce()` could never succeed and it would walk into
            # `_recenter()` with the streamer still publishing. One line,
            # and the interlock holds on every path out of the run.
            sup.pose_active.clear()
            if not sup.join_workers(5.0):
                log.warning("a worker outlived the join budget — closing anyway")
        if poses is not None:
            poses.stop()
        if rig is not None:
            try:
                rig.arm.stop()
            except Exception:
                pass
        if sup is not None and len(sup.acc.points):
            np.save(outdir / "fused_cloud.npy", sup.acc.points)
        if rig is not None:
            rig.close()
        _close_ui(child)
        if bus is not None:
            bus.stop()
    print(f"run saved: {outdir}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run, "teach": teach})
