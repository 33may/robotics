#!/usr/bin/env python3
"""Loop v2 composition root. Design: inspection/2026-08-20-ui-driven-loop-design.md.

    p inspection/run/app.py run --outdir=data/runs/r1     # the real thing
    p inspection/run/app.py teach                         # save survey pose
"""
import json, logging, signal, threading, webbrowser
from pathlib import Path
import numpy as np

ROBOT_IP = "192.168.2.50"
SURVEY_POSE_FILE = Path(__file__).resolve().parent / "survey_pose.json"


def install_sigint(sup):
    """First Ctrl-C: safe shutdown (stop arm, save, close). Second: hard exit.

    Note: run/exit arriving while phase == "planning" shuts the dispatcher down
    immediately with the plan worker still running (daemon, benign — planning
    never moves the arm).
    """
    def handler(signum, frame):
        signal.signal(signal.SIGINT, signal.default_int_handler)
        sup.request_shutdown()
    signal.signal(signal.SIGINT, handler)


def teach(ip: str = ROBOT_IP):
    # unchanged from v1 (loop.py, retired — see the 2026-08-20 design doc)
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(ip)
    q = list(r.getActualQ())
    r.disconnect()
    SURVEY_POSE_FILE.write_text(json.dumps({"q_rad": q}, indent=2) + "\n")
    print(f"survey pose saved: {np.round(np.degrees(q), 1).tolist()} deg")


def run(outdir: str, ip: str = ROBOT_IP, r: float = 0.35,
        port: int = 8767, bus_port: int = 8765, no_window: bool = False,
        seed: int = 0, gui: str = "qt"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from porthole import PortholeBus
    from porthole.window import open_window, serve_ui
    from inspection.motion.execute import preflight
    from inspection.run.machine import Supervisor
    from inspection.run.rigs import PoseStreamer, RealRig
    from inspection.ui.app import DIST, _asset_mounts, _require_build
    from inspection.ui.publisher import InspectionPublisher

    pf = preflight(ip)
    if not pf["go"]:
        raise SystemExit(f"preflight NO-GO: {pf}")
    if not SURVEY_POSE_FILE.exists():
        raise SystemExit("no survey pose — run `p inspection/run/app.py teach`")
    if not _require_build():
        raise SystemExit(1)
    q_survey = np.array(json.loads(SURVEY_POSE_FILE.read_text())["q_rad"])

    outdir = Path(outdir)
    bus = PortholeBus(app="inspection", port=bus_port).start()
    pub = InspectionPublisher(bus, run_dir=outdir)
    pub.declare()
    stop_event = threading.Event()
    rig = RealRig(None, stop_event, outdir, ip)     # world set below
    sup = Supervisor(rig, pub, outdir, q_survey, seed=seed, r=r)
    rig.world = sup.world
    pub.publish_world(sup.world)
    rig.start_camera(pub)
    poses = PoseStreamer(rig.q, lambda q: pub.publish_pose(sup.world, q))
    poses.active = sup.pose_active                   # dispatcher-gated
    poses.start()
    install_sigint(sup)

    def pump():
        for c in bus.commands():
            sup.events.put(c)
    threading.Thread(target=pump, name="cmd-pump", daemon=True).start()

    url = serve_ui(DIST, port=port, assets=_asset_mounts(outdir))
    if bus_port != 8765:
        url = f"{url}/?bus={bus_port}"
    print(f"ui   {url}", flush=True)
    ui_thread = threading.Thread(
        target=lambda: webbrowser.open(url) if no_window else open_window(
            url, title="inspection", width=1700, height=1000, gui=gui),
        daemon=True)
    ui_thread.start()
    try:
        sup.run()                                    # blocks until done
    finally:
        # belt-and-braces shutdown, in this order (design §Safety)
        sup.stop_event.set()
        poses.stop()
        try:
            rig.arm.stop()
        except Exception:
            pass
        if len(sup.acc.points):
            np.save(outdir / "fused_cloud.npy", sup.acc.points)
        rig.close()
        bus.stop()
    print(f"run saved: {outdir}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run, "teach": teach})
