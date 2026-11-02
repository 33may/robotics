"""The demo's synthetic backend, shared by every way of running it.

There are three entry points — the native app, the HMR publisher, and the
headless screenshot check — and they must publish identical data or the checks
stop meaning anything. So the backend lives here once and they all call it.

Nothing here is a "demo mode". It is a Python process publishing on the real
socket, and the frontend has no way to ask whether the data is synthetic.
"""

from __future__ import annotations

import random
import threading
from pathlib import Path

from porthole import PortholeBus
from porthole.mock import (
    MockRig,
    drifting_cloud,
    gradient_frame,
    linkage_poses,
    linkage_scene,
)

# Four views of different shapes, so the media grid's layout solver has
# something to solve. Phase offsets keep them visibly distinct.
CAMERAS = [
    ("wrist", 640, 360),
    ("scene", 640, 360),
    ("table", 480, 480),
    ("shelf", 640, 360),
]

LEVELS = ["info", "info", "info", "warn", "error"]
MESSAGES = [
    "planner: 8 candidate viewpoints",
    "planner: path accepted, 42 waypoints",
    "perception: cloud fused, 12043 points",
    "perception: centroid drift 1.8mm",
    "motion: shortcut removed 11 waypoints",
]

VIEWPOINT_OPTIONS = [
    {"id": "h3v1", "label": "one step right, same height", "meta": "h=3 v=1"},
    {"id": "h0v2", "label": "straight ahead, higher", "meta": "h=0 v=2"},
    {"id": "h9v1", "label": "one step left, same height", "meta": "h=9 v=1"},
    {"id": "h6v0", "label": "opposite side, lower", "meta": "h=6 v=0"},
]

# Any mesh on this machine exercises the asset-mount path end to end: URL fetch,
# loader dispatch, and the Collada Z-up correction. The demo runs fine without
# one — it just never proves that path works.
MESH_DIR = Path.home() / ".cache/robot_descriptions/ur_description/meshes/ur5e/visual"
MESH_MOUNT = "/meshes"
MESH_FILE = "wrist3.dae"


def demo_mesh_assets() -> tuple[dict[str, str | Path], str | None]:
    """Return ``(assets, mesh_url)`` — both empty when no mesh is available."""
    if MESH_DIR.is_dir() and (MESH_DIR / MESH_FILE).exists():
        return {MESH_MOUNT: MESH_DIR}, f"{MESH_MOUNT}/{MESH_FILE}"
    return {}, None


def start_demo_backend(bus: PortholeBus, mesh_url: str | None = None) -> None:
    """Declare every topic, publish retained state, and start the producers."""
    bus.declare("scene/description", qos="stream", kind="scene")
    bus.declare("scene/poses", qos="stream", kind="scene")
    bus.declare("scene/cloud", qos="stream", kind="ndarray")
    for name, _, _ in CAMERAS:
        bus.declare(f"camera/{name}", qos="stream", kind="image/jpeg")
    bus.declare("log/events", qos="event")
    bus.declare("decision/request", qos="event", history=1)

    # Retained state, published once. A UI opened an hour from now gets the whole
    # world before its first pose frame.
    bus.publish("scene/description", linkage_scene(mesh_url))
    bus.publish(
        "decision/request",
        {
            "id": "d-1",
            "prompt": "Where should the camera look next?",
            "options": VIEWPOINT_OPTIONS,
        },
    )

    rig = MockRig(bus)
    for index, (name, width, height) in enumerate(CAMERAS):
        # Bind the loop variables as defaults: a lambda closing over `index`
        # would give every camera the last one's phase and size.
        rig.every(
            15,
            f"camera/{name}",
            lambda t, i=index, w=width, h=height: bus.jpeg_payload(
                gradient_frame(t + i * 1.7, w, h)
            ),
        )
    rig.every(30, "scene/poses", lambda t: linkage_poses(t, mesh=mesh_url is not None))
    rig.every(10, "scene/cloud", lambda t: {"points": bus.array_payload(drifting_cloud(t))})
    rig.every(
        1.5,
        "log/events",
        lambda t: {"level": random.choice(LEVELS), "msg": random.choice(MESSAGES)},
    )
    threading.Thread(target=rig.run, name="demo-producers", daemon=True).start()


def handle_demo_commands(bus: PortholeBus, echo: bool = False) -> None:
    """Consume commands forever. Blocks — run it on a thread or the main one.

    The bus never acts on a command. This function is the application, and this
    is the layer where a real one would decide whether what a human asked for is
    legal before touching anything.
    """
    for command in bus.commands():
        if echo:
            print(f"command: {command}", flush=True)
        bus.publish(
            "log/events",
            {"level": "info", "msg": f"operator chose {command.get('choice')}"},
        )
