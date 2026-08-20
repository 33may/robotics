"""The real Supervisor, wired behind the real bus — no robot, no camera.

Not a "demo mode": `start_mock` builds an ordinary `Supervisor` driving an
ordinary `FakeRig`, publishing through an ordinary `InspectionPublisher` on the
ordinary bus, so the frontend cannot tell and there is no scripted branch in
the UI (or here) to rot. Only the rig and the camera are faked — everything
else (the workcell, the collision world, IK, the viewsphere, reachability,
planning, the state machine) is the same code path a real run takes.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from inspection.motion.plan import DEMO_PARK
from inspection.run.machine import Supervisor
from inspection.run.rigs import CameraWorker, FakeRig, PoseStreamer

# A mug-sized object standing on the table, in front of the robot.
OBJECT_CENTER = np.array([0.0, -0.45, 0.10])
OBJECT_DIMS = [0.09, 0.09, 0.11]


def cup_cloud(n: int, seed: int, coverage: float) -> np.ndarray:
    """Points on a mug-ish shell, revealed progressively as views accumulate.

    `coverage` in [0, 1] opens an azimuth wedge, so the cloud visibly grows as
    the run visits more cells instead of appearing complete on the first turn.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-math.pi, math.pi, n)
    keep = np.abs(theta) <= math.pi * max(coverage, 0.12)
    theta = theta[keep]
    z = rng.uniform(0.0, OBJECT_DIMS[2], theta.size)
    radius = OBJECT_DIMS[0] / 2 * (1.0 + 0.05 * np.sin(z * 40))
    points = np.stack([
        OBJECT_CENTER[0] + radius * np.cos(theta),
        OBJECT_CENTER[1] + radius * np.sin(theta),
        OBJECT_CENTER[2] - OBJECT_DIMS[2] / 2 + z,
    ], axis=1)
    return points.astype(np.float32)


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


def start_mock(bus, pub, outdir, seed: int = 0) -> Supervisor:
    """Wire a real `Supervisor` to a real `FakeRig` behind `bus` — no robot,
    no camera. Starts the camera/pose workers, the command pump (bus ->
    `sup.events`), and `sup.run()` itself, each on its own daemon thread, and
    returns the supervisor already running.

    The rig starts off the survey pose (a small wrist offset) so the boot
    turn is an actual move, not a no-op; `speed=0.6` keeps that move — and
    every move after it — slow enough to watch and to stop mid-flight.
    """
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_survey + np.radians([0, 0, 0, 0, 0, 8]), speed=0.6)
    sup = Supervisor(rig, pub, outdir, q_survey=q_survey, seed=seed)
    pub.publish_world(sup.world)

    camera = CameraWorker(grab=lambda: {"rgb": mock_frame(time.time())},
                          publish=pub.publish_frame)
    camera.start()
    poses = PoseStreamer(rig.q, lambda q: pub.publish_pose(sup.world, q))
    poses.active = sup.pose_active            # dispatcher-gated, like the real loop
    poses.start()

    def pump():
        for c in bus.commands():
            sup.events.put(c)
    threading.Thread(target=pump, name="mock-cmd-pump", daemon=True).start()

    threading.Thread(target=sup.run, daemon=True).start()
    return sup
