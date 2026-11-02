"""Synthetic publishers for developing panels without a backend.

A mock is not a UI mode. It is an ordinary Python process publishing ordinary
messages on the ordinary socket — the frontend cannot tell the difference, and
there is deliberately no way for it to ask. That means a panel developed against
a mock is a panel that works against the real thing, and it means there is no
"demo mode" branch to rot.

    from porthole import PortholeBus
    from porthole.mock import MockRig

    bus = PortholeBus(app="demo").start()
    rig = MockRig(bus)
    rig.every(15, "camera/wrist", lambda t: bus.jpeg_payload(noise_image(t)))
    rig.every(2,  "log/events",   lambda t: {"level": "info", "msg": f"tick {t:.1f}"})
    rig.run()
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable
from typing import Any

from .bus import PortholeBus

Producer = Callable[[float], Any]


class MockRig:
    """Runs periodic publishers on one scheduler thread.

    Uses a deadline heap rather than a thread per publisher, so a 30 Hz camera
    and a 0.2 Hz log line cost the same and stay in phase with each other.
    """

    def __init__(self, bus: PortholeBus) -> None:
        self.bus = bus
        self._jobs: list[tuple[float, str, Producer]] = []

    def every(self, hz: float, topic: str, produce: Producer) -> "MockRig":
        """Publish ``produce(elapsed_seconds)`` to ``topic`` at ``hz``."""
        if hz <= 0:
            raise ValueError("hz must be positive")
        self._jobs.append((1.0 / hz, topic, produce))
        return self

    def once(self, topic: str, payload: Any) -> "MockRig":
        """Publish a single retained message now — for static scene setup."""
        self.bus.publish(topic, payload)
        return self

    def run(self, duration: float | None = None) -> None:
        """Block, publishing on schedule. Ctrl-C or ``duration`` stops it."""
        start = time.monotonic()
        queue = [(start + period, i) for i, (period, _, _) in enumerate(self._jobs)]
        heapq.heapify(queue)

        try:
            while queue:
                due, index = heapq.heappop(queue)
                now = time.monotonic()
                if duration is not None and now - start >= duration:
                    return
                if due > now:
                    time.sleep(due - now)

                period, topic, produce = self._jobs[index]
                elapsed = time.monotonic() - start
                self.bus.publish(topic, produce(elapsed))
                # Schedule off the previous deadline so drift does not accumulate.
                heapq.heappush(queue, (due + period, index))
        except KeyboardInterrupt:
            pass


# ── generic synthetic sources ───────────────────────────────────────────────
# Domain-neutral on purpose: enough to exercise a panel's rendering path, not
# enough to pretend it is real data. App-specific fakes belong in the app.


def drifting_cloud(t: float, n: int = 4000, radius: float = 0.08) -> Any:
    """A wobbling point shell — exercises float32 array transport and 3D panels."""
    import numpy as np

    rng = np.random.default_rng(seed=int(t * 2) % 997)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    wobble = 1.0 + 0.08 * np.sin(3 * directions[:, 2] + t)
    return (directions * radius * wobble[:, None]).astype("float32")


def gradient_frame(t: float, w: int = 640, h: int = 360) -> Any:
    """A moving colour ramp with a moving marker — exercises the JPEG path."""
    import numpy as np

    xs = np.linspace(0, 1, w, dtype="float32")
    ys = np.linspace(0, 1, h, dtype="float32")
    gx, gy = np.meshgrid(xs, ys)
    phase = 0.5 + 0.5 * math.sin(t)

    frame = np.empty((h, w, 3), dtype="float32")
    frame[:, :, 0] = gx * phase
    frame[:, :, 1] = gy
    frame[:, :, 2] = 1.0 - gx * phase

    cx = int((0.5 + 0.35 * math.sin(t * 1.7)) * w)
    cy = int((0.5 + 0.35 * math.cos(t * 1.1)) * h)
    frame[max(0, cy - 12):cy + 12, max(0, cx - 12):cx + 12] = 1.0
    return (frame * 255).astype("uint8")


def wandering_series(t: float, channels: int = 3) -> dict:
    """One timestamped sample per channel — exercises streaming plots."""
    return {
        "t": t,
        "values": [math.sin(t * (0.6 + 0.3 * i) + i) * (1 + 0.2 * i) for i in range(channels)],
    }


# ── synthetic 3D scene ──────────────────────────────────────────────────────
# A three-segment linkage on a bench. Deliberately not a robot: it exercises
# every part of docs/scene-protocol.md — primitives, meshes, animated poses,
# per-frame overrides, background verdict — without giving Porthole an opinion
# about kinematics. Real robot scenes are built in the app, from its own model.

_BENCH = [
    ("bench/slab", (1.10, 0.70, 0.03), (0.0, 0.0, -0.015), "#6d737d", 1.0),
    ("bench/post_left", (0.04, 0.04, 0.70), (-0.50, -0.30, -0.37), "#3f444d", 1.0),
    ("bench/post_right", (0.04, 0.04, 0.70), (0.50, -0.30, -0.37), "#3f444d", 1.0),
    ("bench/keepout", (0.40, 0.30, 0.50), (0.55, 0.30, 0.25), "#ff8800", 0.25),
]


def _pose_matrix(rotation: Any, translation: Any) -> Any:
    import numpy as np

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _rz(angle: float) -> Any:
    import numpy as np

    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(angle: float) -> Any:
    import numpy as np

    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def linkage_scene(mesh_url: str | None = None) -> dict:
    """The static description: bench, linkage segments, origin triad.

    Pass ``mesh_url`` to hang a mesh off the tip. That path is worth exercising
    even with an arbitrary file: it is the only way to find out whether the
    asset mount, the loader and the Collada Z-up correction all agree.
    """
    from .scene import (
        axes_node,
        box_node,
        cylinder_node,
        mesh_node,
        points_node,
        scene_description,
    )

    nodes = [axes_node("world/axes", 0.2)]
    for path, size, centre, color, opacity in _BENCH:
        matrix = _pose_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], centre)
        nodes.append(box_node(path, size, matrix, color=color, opacity=opacity))

    nodes += [
        cylinder_node("arm/column", 0.05, 0.30, color="#4a5563"),
        box_node("arm/upper", (0.34, 0.07, 0.07), color="#8ab4f8"),
        box_node("arm/fore", (0.28, 0.06, 0.06), color="#8ab4f8"),
        cylinder_node("arm/tip", 0.035, 0.06, color="#7ee2a8"),
        axes_node("arm/tip_axes", 0.12),
        # Bulk data by reference: the cloud streams on its own topic and the
        # node's transform lifts it onto the bench.
        points_node(
            "sample/cloud",
            "scene/cloud",
            size=0.004,
            color="#9ecbff",
            transform=_pose_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.30, 0.20, 0.16]),
        ),
    ]
    if mesh_url:
        nodes.append(mesh_node("arm/payload", mesh_url, scale=(1.0, 1.0, 1.0)))
    return scene_description(nodes, frame="world")


def linkage_poses(t: float, mesh: bool = False) -> dict:
    """One animation frame of the linkage, with a verdict when the tip dips low.

    The verdict rides in the same message as the pose it describes — see
    `docs/scene-protocol.md` §1. Split across topics, a dropped message would
    leave the tip green while it is inside the bench.
    """
    import numpy as np

    from .scene import pose_frame

    yaw = 0.8 * math.sin(t * 0.5)
    shoulder = -0.6 + 0.5 * math.sin(t * 0.8)
    elbow = 1.1 + 0.6 * math.sin(t * 0.8 + 1.2)

    column_top = np.array([0.0, 0.0, 0.30])
    upper_len, fore_len = 0.34, 0.28

    r_base = _rz(yaw)
    r_upper = r_base @ _ry(shoulder)
    upper_centre = column_top + r_upper @ np.array([upper_len / 2, 0.0, 0.0])
    upper_end = column_top + r_upper @ np.array([upper_len, 0.0, 0.0])

    r_fore = r_upper @ _ry(elbow)
    fore_centre = upper_end + r_fore @ np.array([fore_len / 2, 0.0, 0.0])
    tip = upper_end + r_fore @ np.array([fore_len, 0.0, 0.0])

    poses = {
        "arm/column": _pose_matrix(np.eye(3), [0.0, 0.0, 0.15]),
        "arm/upper": _pose_matrix(r_upper, upper_centre),
        "arm/fore": _pose_matrix(r_fore, fore_centre),
        "arm/tip": _pose_matrix(r_fore, tip),
        "arm/tip_axes": _pose_matrix(r_fore, tip),
    }
    if mesh:
        poses["arm/payload"] = _pose_matrix(r_fore, tip)

    low = tip[2] < 0.08
    frame = pose_frame(
        poses,
        overrides={"arm/tip": {"color": "#f2777a"}} if low else None,
        background=["#5a2b2b", "#2a1414"] if low else None,
    )
    return frame
