"""A scripted run that drives every topic, with no robot and no camera.

Not a "demo mode": it is an ordinary Python process publishing on the ordinary
socket, so the frontend cannot tell and there is no branch in the UI to rot.

What is real here: the workcell, the collision world, IK, the viewsphere,
reachability, direct-move validation and the replay. Only perception and the
arm are faked — the camera returns a synthetic frame and the "fused cloud" is
generated rather than measured. That means this exercises the publisher's real
paths, including the ones that are awkward to reach with hardware (a cell that
plans, a cell that refuses, a preview that freezes at impact).
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from inspection.cell.world import RobotCell, Q_PARK
from inspection.motion.direct import direct_move
from inspection.motion.ik import UR5eIK
from inspection.view.viewsphere import ViewSphere

from .publisher import InspectionPublisher

# A mug-sized object standing on the table, in front of the robot.
OBJECT_CENTER = np.array([0.0, -0.45, 0.10])
OBJECT_DIMS = [0.09, 0.09, 0.11]
QUESTION = "is there a logo on this cup?"


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


def _gloss(current, cell) -> str:
    """A stand-in for `run/decider.gloss`.

    Duplicated deliberately: nothing in this package imports from `run/`, and
    the real loop passes its own glosses into `publish_views`.
    """
    if current is None:
        return f"elevation index {cell[1]}"
    dh = (cell[0] - current[0] + 6) % 12 - 6
    dv = cell[1] - current[1]
    side = ("same side" if dh == 0 else
            "opposite side" if abs(dh) == 6 else
            f"{abs(dh)} step{'s' if abs(dh) > 1 else ''} {'right' if dh > 0 else 'left'}")
    height = "same height" if dv == 0 else ("higher" if dv > 0 else "lower")
    return f"{side}, {height}"


def run_mock(pub: InspectionPublisher, turns: int = 10, seed: int = 0) -> None:
    """Drive a whole scripted run. Blocks until it finishes."""
    rng = random.Random(seed)
    world = RobotCell()
    ik = UR5eIK()

    pub.declare()
    pub.status(phase="idle", question=QUESTION, step=0, max_turns=turns)
    pub.log("info", "mock: building the workcell")
    pub.publish_world(world)

    world.set_object("object", OBJECT_DIMS, [*OBJECT_CENTER.tolist(), 0, 0, 0])
    sphere = ViewSphere(OBJECT_CENTER, r=0.35)

    q_now = np.array(Q_PARK, dtype=float)
    pub.publish_pose(world, q_now)

    visited: set[tuple] = set()
    blocked: set[tuple] = set()
    captures: dict[tuple, dict] = {}
    current: tuple | None = None
    t0 = time.time()

    def refresh_views(reach):
        glosses = {c: _gloss(current, c) for c in sphere.cells()}
        pub.publish_views(sphere, reach, visited, blocked, current, captures, glosses)

    reach = sphere.reachability(world, ik)
    pub.publish_object(cup_cloud(6000, seed, 0.12), OBJECT_DIMS, OBJECT_CENTER)
    refresh_views(reach)
    pub.log("info", f"mock: {sum(1 for r in reach.values() if r is not None)}/36 cells reachable")

    for step in range(1, turns + 1):
        candidates = [c for c, r in reach.items()
                      if r is not None and c not in visited and c not in blocked]
        if not candidates:
            break
        cell = rng.choice(candidates)
        h, v = cell

        pub.status(phase="planning", step=step, detail=f"look {h} {v}")
        pub.log("info", f"turn {step}: look {h} {v} — {_gloss(current, cell)}")

        roll = reach[cell]
        T = sphere.flange_pose(h, v, roll)
        free = [q for q in ik.branches(T) if not world.is_colliding(q)]
        free.sort(key=lambda q: np.abs(q - q_now).sum())
        path = next((p for p in (direct_move(world, q_now, q) for q in free[:3]) if p), None)

        if path is None:
            blocked.add(cell)
            pub.log("warn", f"turn {step}: plan refused — cell blocked this round")
            refresh_views(reach)
            continue

        pub.status(phase="preview", detail=f"{len(path)} waypoints")
        clean = pub.replay_path(world, path)

        pub.status(phase="awaiting_approval")
        time.sleep(0.6)                       # the terminal owns the y/n
        if not clean:
            blocked.add(cell)
            refresh_views(reach)
            continue

        pub.status(phase="executing")
        pub.replay_path(world, path, hz=45)   # the move itself, same pose stream
        q_now = np.asarray(path[-1], dtype=float)
        visited.add(cell)
        current = cell
        blocked.clear()                       # new q — refused cells may work now

        pub.status(phase="capturing")
        pub.publish_frame(mock_frame(time.time() - t0))
        captures[cell] = {
            "step": step,
            "dir": None,                      # no run dir in mock: no image to show
            "comment": rng.choice([
                "handle visible, no logo on this side",
                "glare on the rim",
                "printed mark, partially occluded",
                "clean surface",
            ]),
        }

        pub.status(phase="fusing")
        cloud = cup_cloud(9000, seed, min(1.0, 0.12 + 0.11 * len(visited)))
        pub.publish_cloud(cloud)
        pub.publish_object(None, OBJECT_DIMS, OBJECT_CENTER)

        reach = sphere.reachability(world, ik)
        refresh_views(reach)
        pub.status(
            phase="idle",
            visited=len(visited),
            reachable=sum(1 for r in reach.values() if r is not None),
            total=len(reach),
        )
        pub.log("info", f"turn {step}: fused {len(cloud)} pts")
        time.sleep(0.8)

    pub.status(phase="done", detail=f"{len(visited)} views")
    pub.log("info", "mock: run finished")
