#!/usr/bin/env python3
"""Loop v2 composition root. Design: inspection/docs/2026-08-20-ui-driven-loop-design.md.

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


def _hashed(path: Path, embed: bool = False):
    """A `HashedFile` for a calib/config artifact — identity, not "latest"."""
    import hashlib
    p = Path(path)
    data = p.read_bytes()
    out = {"file": p.name, "sha256": hashlib.sha256(data).hexdigest()}
    if embed:
        out["content"] = data.decode()
    return out


def config_snapshot(models: dict[str, str] | None = None):
    """The world-in-effect at run start: code, cell, calib, tuning constants.

    Everything here is what makes a recorded run re-checkable after cell.yaml
    or the hand-eye artifact is edited — the questions "was this safe" and
    "why is that cloud shaped like that" are answered from this file plus the
    steps, never from today's source tree.
    """
    from inspection.cell import geometry as geo
    from inspection.cell.world import CELL_YAML, DEFAULT_PADDING
    from inspection.record.schema import ConfigSnapshot
    from inspection.record.writer import git_sha
    from inspection.run import settle
    from inspection.run.segmenter import MIN_MASK_PX, MIN_MASK_SCORE
    from inspection.view.grid import H_BINS, V_ELEVATIONS

    calib = sorted((Path(geo.__file__).resolve().parents[1] / "calib")
                   .glob("T_flange_cam_*.npy"))[-1]
    return ConfigSnapshot(
        git_sha=git_sha(),
        cell_yaml=_hashed(CELL_YAML, embed=True),
        calib=_hashed(calib),
        constants={
            "padding_m": DEFAULT_PADDING,
            "voxel_m": geo.VOXEL_M, "jump_gate_m": geo.JUMP_GATE_M,
            "box_pct": geo.BOX_PCT, "grow_eps_m": geo.GROW_EPS_M,
            "range_m": [geo.MIN_RANGE_M, geo.MAX_RANGE_M],
            "workspace": geo.WORKSPACE,
            "plane_z_tol_m": settle.PLANE_Z_TOL_M,
            "plane_tilt_tol_deg": settle.PLANE_TILT_TOL_DEG,
            "min_mask_score": MIN_MASK_SCORE, "min_mask_px": MIN_MASK_PX,
            "h_bins": H_BINS, "v_elevs": list(V_ELEVATIONS),
        },
        models=dict(models or {}),
    )


def viewsphere_method(r: float | None = None) -> dict:
    """The run's one view method, as `run.json` declares it.

    `r` is left open here and filled in by the Supervisor the moment the
    survey derives it (`machine.py:_recenter` -> `set_view_method_params`).
    """
    from inspection.run.machine import VIEW_METHOD
    from inspection.view.grid import H_BINS, V_ELEVATIONS
    return {"id": VIEW_METHOD, "kind": "viewsphere",
            "params": {"h_bins": H_BINS, "v_elevs": list(V_ELEVATIONS), "r": r}}


def finish_run(writer, acc) -> None:
    """Close a run: the fused cloud beside the steps, then the final run.json.

    Shared by both composition roots (`run/app.py` and `ui/mock.py`) so a mock
    run ends exactly the way a real one does. `fused/` is a directory because
    the cloud is derived from the steps, not one of them; `record/run.py`
    reads it there first. Status is read off the disk, not off a flag: a run
    is completed iff some AI session actually answered.

    `acc` may be None — a run whose setup died before the Supervisor existed
    still gets closed, because "created but never closed" is the signature of
    a crash and this one was not.
    """
    if acc is not None and len(acc.points):
        d = writer.dir / "fused"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "cloud.npy", acc.points)
        # Row-aligned with cloud.npy — uint8, mid-grey where a view
        # contributed points without a usable colour frame.
        np.save(d / "colors.npy", acc.colors)
    answered = any((writer.dir / "ai").glob("*/answer.json"))
    writer.close("completed" if answered else "aborted")


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
        seed: int = 0, gui: str = "qt", name: str | None = None,
        object: str | None = None):
    """One live run. `outdir` is `<runs root>/<run id>`; `name` defaults to
    the id's leaf and `object` is the free tag the archive is queried by."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from porthole import PortholeBus
    from inspection.eyes.cognition import real_cognition
    from inspection.eyes.verbs_local import Sam3Backend
    from inspection.motion.execute import preflight
    from inspection.run.machine import Supervisor
    from inspection.run.rigs import PoseStreamer, RealRig
    from inspection.run.segmenter import ObjectSegmenter
    from inspection.record.writer import RunWriter
    from inspection.ui.app import _require_build
    from inspection.ui.publisher import InspectionPublisher

    # FIRST, before the arm: `real_cognition()` raises if GEMINI_API_KEY is
    # unset, and there is no stub fallback any more (Anton 2026-08-26). A key
    # discovered missing at the first `brain/ask` is a key discovered with the
    # robot powered, the bus up and an operator waiting. Nothing loads here —
    # SAM 3, OCR and the Gemini client are all lazy.
    # ONE SAM 3 on the GPU: this same instance goes to the segmenter below, so
    # the checkpoint is loaded once for both the subagent's detect/segment and
    # the identity mask, instead of twice (`run/segmenter.py:77`).
    sam3 = Sam3Backend()
    cognition = real_cognition(sam3=sam3)

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
    bus = rig = sup = poses = child = writer = None
    try:
        bus = PortholeBus(app="inspection", port=bus_port).start()
        pub = InspectionPublisher(bus, run_dir=outdir)
        pub.declare()
        # The run directory is CREATED here, by the one writer that owns it —
        # `create` refuses an existing id, because evidence is never
        # overwritten. Everything downstream gets its paths from `writer.dir`.
        writer = RunWriter.create(
            outdir.parent, run_id=outdir.name, name=name or outdir.name,
            source="live", rig="real", object=object, question=None,
            config=config_snapshot({"vlm": cognition.label,
                                    "segmenter": "sam3"}),
            view_methods=[viewsphere_method(r)],
            q_survey=[float(v) for v in q_survey])
        # Retained, so a UI window opened at any moment knows which flow it
        # is rendering (data-engine runs drop the AI panel).
        pub.publish_run_meta(source="live", name=name or outdir.name,
                             object=object, question=None)
        rig = RealRig(None, stop_event, outdir, ip)   # world set below
        # Built by the rig (it opens the camera), written by the writer (it
        # owns every JSON in the run) — the split that ended the two-writers
        # problem for session.json.
        writer.write_session(rig.session)
        # Object identity is settled in image space (run 2408-cup2): a cable
        # 20 mm from the cup is adjacent, so no distance rule can refuse it.
        # Weights load lazily on the first capture or the first detect,
        # whichever comes first — never here.
        sup = Supervisor(rig, pub, writer, q_survey, seed=seed, r=r,
                         segmenter=ObjectSegmenter(backend=sam3))
        install_sigint(sup)                          # earliest safe Ctrl-C
        rig.world = sup.world
        rig.start_camera(pub)
        poses = PoseStreamer(rig.q, lambda q: pub.publish_pose(sup.world, q))
        poses.active = sup.pose_active               # dispatcher-gated
        sup.pose_quiesce = poses.quiesce             # settle-leg interlock
        poses.start()

        from inspection.brain.live import make_ask_handler
        start_brain = make_ask_handler(sup, pub, outdir, cognition, writer)

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
        if writer is not None:
            # Guarded like every other line of this teardown: `finish_run`
            # saves two arrays, hashes every binary in the run and validates
            # the final run.json. Losing the last bytes of a record must never
            # cost us `rig.close()` — a leaked RTDE interface and a camera pipe
            # left running are worse than an unclosed run, which the reader
            # already detects as crashed.
            try:
                finish_run(writer, sup.acc if sup is not None else None)
            except Exception:
                log.exception("could not close the run record")
        if rig is not None:
            rig.close()
        _close_ui(child)
        if bus is not None:
            bus.stop()
    print(f"run saved: {outdir}")


def collect(outdir: str, ip: str = ROBOT_IP, r: float | None = None,
           port: int = 8767, bus_port: int = 8765, no_window: bool = False,
           seed: int = 0, gui: str = "qt", name: str | None = None,
           object: str | None = None):
    """One data-collection sweep — Flow B. `outdir` is `<runs root>/<run id>`.

    Same composition as `run()` minus cognition/`brain/ask` — no VLM, no
    question, nothing under `ai/` — plus a `SweepDriver` in place of an
    operator picking cells: it walks the whole shell cheapest-path-first
    through the SAME approval gate every other mover uses. The closing
    status says whether it actually got there: `driver.finished` is True
    only once the candidate set ran out, never on a run cut short.

    `run()`'s own teardown (`finish_run`) closes on whether a question got
    answered, which has no meaning for a run that never asked one — so the
    close here is its own, not a call into `run()` or `finish_run` (the
    merge-safety rule for this task: additive-only in this file, `run()`
    itself untouched). It duplicates `finish_run`'s few lines of "save the
    fused cloud" rather than its "did the AI answer" status logic.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from porthole import PortholeBus
    from inspection.eyes.verbs_local import Sam3Backend
    from inspection.motion.execute import preflight
    from inspection.run.collect import SweepDriver
    from inspection.run.machine import Supervisor
    from inspection.run.rigs import PoseStreamer, RealRig
    from inspection.run.segmenter import ObjectSegmenter
    from inspection.record.writer import RunWriter
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
    # Same "everything live inside the try" shape as `run()`, for the same
    # reason: a failure part-way through setup must still hit the teardown
    # below rather than leak a socket, an open RTDE interface or a camera.
    bus = rig = sup = poses = child = writer = driver = None
    try:
        bus = PortholeBus(app="inspection", port=bus_port).start()
        pub = InspectionPublisher(bus, run_dir=outdir)
        pub.declare()
        writer = RunWriter.create(
            outdir.parent, run_id=outdir.name, name=name or outdir.name,
            source="data-engine", rig="real", object=object, question=None,
            config=config_snapshot({"segmenter": "sam3"}),
            view_methods=[viewsphere_method(r)],
            q_survey=[float(v) for v in q_survey])
        pub.publish_run_meta(source="data-engine", name=name or outdir.name,
                             object=object, question=None)
        rig = RealRig(None, stop_event, outdir, ip)   # world set below
        writer.write_session(rig.session)
        # Segmentation stays real (identity still matters for a sweep's
        # cloud), only the question-answering tier is gone.
        sup = Supervisor(rig, pub, writer, q_survey, seed=seed, r=r,
                         segmenter=ObjectSegmenter(backend=Sam3Backend()))
        install_sigint(sup)                          # earliest safe Ctrl-C
        rig.world = sup.world
        rig.start_camera(pub)
        poses = PoseStreamer(rig.q, lambda q: pub.publish_pose(sup.world, q))
        poses.active = sup.pose_active               # dispatcher-gated
        sup.pose_quiesce = poses.quiesce             # settle-leg interlock
        poses.start()

        def on_sweep_event(state, **kw):
            pub.log("info", f"sweep: {state} {kw.get('cell', '')}")

        driver = SweepDriver(sup, outdir, on_event=on_sweep_event)
        driver.start()

        def pump():
            # No `brain/ask` branch here — Flow B has no cognition command to
            # intercept, every bus command is the Supervisor's.
            for c in bus.commands():
                sup.events.put(c)
        threading.Thread(target=pump, name="cmd-pump", daemon=True).start()

        child = _open_ui(outdir, port, bus_port, no_window, gui)
        sup.run()                                    # blocks until done
    finally:
        # Same order as `run()`'s teardown (design §Safety): stop event ->
        # join worker (bounded) -> stopJ -> save -> close.
        if sup is not None:
            sup.stop_event.set()
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
        if writer is not None:
            acc = sup.acc if sup is not None else None
            if acc is not None and len(acc.points):
                d = writer.dir / "fused"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "cloud.npy", acc.points)
            writer.close("completed" if driver is not None and driver.finished
                        else "aborted")
        if rig is not None:
            rig.close()
        _close_ui(child)
        if bus is not None:
            bus.stop()
    print(f"collect saved: {outdir}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run, "teach": teach, "collect": collect})
