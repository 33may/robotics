"""The real Supervisor, wired behind the real bus — no robot, no camera.

Not a "demo mode": `start_mock` builds an ordinary `Supervisor` driving an
ordinary `FakeRig`, publishing through an ordinary `InspectionPublisher` on the
ordinary bus, so the frontend cannot tell and there is no scripted branch in
the UI (or here) to rot. Only the rig and the camera are faked — everything
else (the workcell, the collision world, IK, the viewsphere, reachability,
planning, the state machine) is the same code path a real run takes.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
import time

import numpy as np

from inspection.motion.plan import DEMO_PARK
from inspection.record.writer import RunWriter
from inspection.run.app import config_snapshot, finish_run, viewsphere_method
from inspection.run.machine import Supervisor
from inspection.run.rigs import CameraWorker, FakeRig, PoseStreamer


def mock_frame(t: float, w: int = 640, h: int = 360) -> np.ndarray:
    """A synthetic camera frame — moving ramp with a marker."""
    xs = np.linspace(0, 1, w, dtype="float32")
    ys = np.linspace(0, 1, h, dtype="float32")
    gx, gy = np.meshgrid(xs, ys)
    phase = 0.5 + 0.5 * math.sin(t)
    frame = np.empty((h, w, 3), dtype="float32")
    frame[:, :, 0] = gx * phase
    frame[:, :, 1] = gy
    frame[:, :, 2] = 1.0 - gx * phase
    cx, cy = int((0.5 + 0.3 * math.sin(t * 1.3)) * w), int((0.5 + 0.3 * math.cos(t)) * h)
    frame[max(0, cy - 14):cy + 14, max(0, cx - 14):cx + 14] = 1.0
    return (frame * 255).astype("uint8")


def _stub_segmenter():
    """The real `ObjectSegmenter` policy over a backend that needs no weights.

    `StubBackend.segment` fills the prompt box, so the mock's mask is the
    reprojected object's box — geometrically sensible on the synthetic scene
    and enough to drive every downstream artifact.
    """
    from inspection.eyes.verbs_local import StubBackend
    from inspection.run.segmenter import ObjectSegmenter
    return ObjectSegmenter(backend=StubBackend())


def start_mock(bus, pub, outdir, seed: int = 0) -> Supervisor:
    """Wire a real `Supervisor` to a real `FakeRig` behind `bus` — no robot,
    no camera. Starts the camera/pose workers, the command pump (bus ->
    `sup.events`), and `sup.run()` itself, each on its own daemon thread, and
    returns the supervisor already running.

    The rig starts off the survey pose (a small wrist offset) so the boot
    turn is an actual move, not a no-op; `speed=0.6` keeps that move — and
    every move after it — slow enough to watch and to stop mid-flight.

    `outdir` is `<root>/<run id>` and must NOT exist: the run's `RunWriter`
    creates it, the same way a real run's is created, because a mock run is a
    real recording of a fake robot (`rig="fake"` says so in run.json) and the
    e2e checks read it back through the same door.
    """
    q_survey = DEMO_PARK.copy()
    outdir = Path(outdir)
    rig = FakeRig(q_survey + np.radians([0, 0, 0, 0, 0, 8]), speed=0.6)
    writer = RunWriter.create(
        outdir.parent, run_id=outdir.name, name=outdir.name,
        source="live", rig="fake", question=None,
        config=config_snapshot({"vlm": "stub", "segmenter": "stub"}),
        view_methods=[viewsphere_method()],
        q_survey=[float(v) for v in q_survey], tags=["mock"])
    pub.publish_run_meta(source="live", name=outdir.name, object=None,
                         question=None)
    # A STUB segmenter, not a real one: the mock must stay GPU-free and
    # offline, but the identity path — prompt box, mask, masked lift, chain
    # overlays — is then the same code a real run takes, so the frontend has
    # something real to render and the checks can assert on it.
    sup = Supervisor(rig, pub, writer, q_survey=q_survey, seed=seed,
                     segmenter=_stub_segmenter())
    pub.publish_world(sup.world)

    # Daemon threads, deliberately never joined/stopped when `sup` reaches
    # "done": this process serves more than one run (each `npm run check`
    # invocation, each manual restart), and a finished Supervisor's camera/
    # pose threads are cheap to leave running until the process exits.
    camera = CameraWorker(grab=lambda: {"rgb": mock_frame(time.time())},
                          publish=pub.publish_frame)
    camera.start()
    poses = PoseStreamer(rig.q, lambda q: pub.publish_pose(sup.world, q))
    poses.active = sup.pose_active            # dispatcher-gated, like the real loop
    sup.pose_quiesce = poses.quiesce          # settle-leg interlock, like the real loop
    poses.start()

    # Same handler the real run uses: the mock is how the approval gate gets
    # rehearsed without hardware, and rehearsing against different wiring
    # rehearses nothing. The stubs are INJECTED, the same way the segmenter
    # above is — they are no longer what the handler falls back to when a key
    # is missing (Anton 2026-08-26), so a stub can only be reached by asking
    # for one, and `stub_cognition().label` marks the run's trace as one.
    from inspection.brain.live import make_ask_handler
    from inspection.eyes.cognition import stub_cognition
    ask = make_ask_handler(sup, pub, outdir, stub_cognition(), writer)

    def pump():
        for c in bus.commands():
            if c.get("cmd") == "brain/ask":
                ask(c.get("question"))
            else:
                sup.events.put(c)
    threading.Thread(target=pump, name="mock-cmd-pump", daemon=True).start()

    def run_and_close():
        # A mock run ENDS like a real one (`run/app.py`'s teardown): the fused
        # cloud beside the steps, then a closed run.json with a manifest. A
        # recording that is never closed is the one thing the e2e check could
        # not tell apart from a crash.
        try:
            sup.run()
        finally:
            try:
                finish_run(writer, sup.acc)
            except Exception:
                logging.getLogger(__name__).exception(
                    "could not close the mock run record")

    threading.Thread(target=run_and_close, daemon=True).start()
    return sup


def start_mock_collect(bus, pub, outdir, seed: int = 0) -> Supervisor:
    """`start_mock`'s Flow B twin: the same real `Supervisor`/`FakeRig`, but
    `source="data-engine"`, no `brain/ask` handler, and a `SweepDriver` in
    place of an operator picking cells one at a time.

    A separate function rather than a branch inside `start_mock` on purpose
    (merge-safety over DRY, same call as `run/app.py:collect` vs `run()`):
    the two runs close differently — `finish_run` reads "did a question get
    answered", which has no meaning for a run that never asked one, so this
    duplicates its few lines of "save the fused cloud" instead of its status
    logic, and leaves `start_mock` itself untouched for whatever else is
    landing in this file in parallel.
    """
    q_survey = DEMO_PARK.copy()
    outdir = Path(outdir)
    rig = FakeRig(q_survey + np.radians([0, 0, 0, 0, 0, 8]), speed=0.6)
    writer = RunWriter.create(
        outdir.parent, run_id=outdir.name, name=outdir.name,
        source="data-engine", rig="fake", question=None,
        config=config_snapshot({"segmenter": "stub"}),
        view_methods=[viewsphere_method()],
        q_survey=[float(v) for v in q_survey], tags=["mock"])
    pub.publish_run_meta(source="data-engine", name=outdir.name, object=None,
                         question=None)
    sup = Supervisor(rig, pub, writer, q_survey=q_survey, seed=seed,
                     segmenter=_stub_segmenter())
    pub.publish_world(sup.world)

    camera = CameraWorker(grab=lambda: {"rgb": mock_frame(time.time())},
                          publish=pub.publish_frame)
    camera.start()
    poses = PoseStreamer(rig.q, lambda q: pub.publish_pose(sup.world, q))
    poses.active = sup.pose_active
    sup.pose_quiesce = poses.quiesce
    poses.start()

    # Same writer-backed sink the real `run/app.py:collect` installs: the
    # sweep's approval chain (requested -> awaiting_approval -> approved) is a
    # record, and a mock that only logged it would rehearse a different write
    # path than the one it exists to rehearse. Two threads append to
    # `events.jsonl` from here (this one, and the dispatcher's stopped/fault
    # rows) — single-line O_APPEND writes, atomic at these sizes.
    from inspection.brain.live import make_event_sink
    from inspection.run.collect import SweepDriver
    driver = SweepDriver(sup, outdir,
                         on_event=make_event_sink(writer, pub, label="sweep"))
    driver.start()

    def pump():
        # No `brain/ask` branch: Flow B has no cognition command to
        # intercept, every bus command is the Supervisor's.
        for c in bus.commands():
            sup.events.put(c)
    threading.Thread(target=pump, name="mock-cmd-pump", daemon=True).start()

    def run_and_close():
        try:
            sup.run()
        finally:
            # The driver first, like the real teardown: it must not still be
            # ranking (and requesting) against a Supervisor that is done.
            driver.stop()
            driver.join(5.0)
            if sup.acc is not None and len(sup.acc.points):
                d = writer.dir / "fused"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "cloud.npy", sup.acc.points)
                # Row-aligned with cloud.npy, exactly as `finish_run` saves it.
                np.save(d / "colors.npy", sup.acc.colors)
            # Operator finish counts as completed, exactly like the real
            # teardown in `run/app.py:collect`.
            writer.close("completed" if driver.finished or sup.finish_requested
                         else "aborted")

    threading.Thread(target=run_and_close, daemon=True).start()
    return sup
