"""Live 3D preview of cell.yaml — CAD-less workcell modeling (MAY-183).

Usage (from repo root, robo env active):
    p inspection/cell/viewer.py            # boxes + UR5e model
    p inspection/cell/viewer.py --no-robot # boxes only, faster start
    p inspection/cell/viewer.py --once     # render once and exit (smoke test)

Opens a meshcat server and prints its URL — open it in a browser.
Then edit cell.yaml; every save re-renders the scene within ~0.5 s.

The modeling loop this enables: tape measure in one hand, editor in the other,
browser showing the boxes appear exactly where you typed them.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

CELL_YAML = Path(__file__).parent / "cell.yaml"

# name-substring -> (color, opacity); first match wins, top to bottom
STYLE = [
    ("shelf", (0xFF8800, 0.45)),   # keep-out envelope: loud + see-through
    ("bar", (0x4A4A4A, 1.0)),
    ("pedestal", (0x333333, 1.0)),
    ("slab", (0x8A8A8A, 1.0)),
]
DEFAULT_STYLE = (0x2288CC, 0.8)

UR5E_HOME = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def pose_to_T(pose):
    """[x, y, z, roll, pitch, yaw] -> 4x4 homogeneous transform."""
    x, y, z, roll, pitch, yaw = pose
    T = tf.euler_matrix(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


def frame_to_base(frames, name, _depth=0):
    """Resolve a frame's transform to the base frame by walking parents."""
    if name in (None, "base"):
        return np.eye(4)
    if _depth > 10:
        raise ValueError(f"frame cycle or chain too deep at '{name}'")
    if name not in frames:
        raise KeyError(f"box references unknown frame '{name}' (define it under frames:)")
    f = frames[name]
    return frame_to_base(frames, f.get("parent", "base"), _depth + 1) @ pose_to_T(f["pose"])


def style_for(name):
    for key, style in STYLE:
        if key in name:
            return style
    return DEFAULT_STYLE


def draw_cell(vis, cfg):
    """(Re)draw all boxes from a parsed cell.yaml dict."""
    vis["cell"].delete()
    frames = cfg.get("frames", {})
    boxes = cfg.get("boxes", {})
    for name, box in boxes.items():
        T = frame_to_base(frames, box.get("parent", "base")) @ pose_to_T(box["pose"])
        color, opacity = style_for(name)
        vis["cell"][name].set_object(
            g.Box(box["dims"]),
            g.MeshLambertMaterial(color=color, opacity=opacity, transparent=opacity < 1.0),
        )
        vis["cell"][name].set_transform(T)
    # base-frame axes triad for orientation sanity
    vis["cell"]["base_triad"].set_object(g.triad(0.15))
    return len(boxes)


def load_robot(vis):
    """Display the UR5e at a home pose for scale. Returns the pinocchio visualizer."""
    from pinocchio.visualize import MeshcatVisualizer
    from robot_descriptions.loaders.pinocchio import load_robot_description

    print("loading UR5e description (first run downloads it) ...", flush=True)
    robot = load_robot_description("ur5e_description")
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(viewer=vis)
    viz.loadViewerModel(rootNodeName="ur5e")
    viz.display(UR5E_HOME)
    print("UR5e displayed at home pose.", flush=True)
    return viz


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-robot", action="store_true", help="skip the UR5e model")
    ap.add_argument("--once", action="store_true", help="render once and exit")
    ap.add_argument("--yaml", type=Path, default=CELL_YAML, help="cell.yaml path")
    args = ap.parse_args()

    vis = meshcat.Visualizer()
    print(f"meshcat URL: {vis.url()}", flush=True)

    if not args.no_robot:
        try:
            load_robot(vis)
        except Exception as e:  # viewer must stay useful without the robot
            print(f"WARNING: robot model not loaded ({e}); showing boxes only.", flush=True)

    last_mtime = 0.0
    while True:
        try:
            mtime = args.yaml.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                cfg = yaml.safe_load(args.yaml.read_text())
                n = draw_cell(vis, cfg)
                print(f"rendered {n} boxes from {args.yaml.name}", flush=True)
        except Exception as e:
            # bad yaml mid-edit is normal — keep the old scene, keep watching
            print(f"cell.yaml not rendered: {e}", flush=True)
            last_mtime = mtime if "mtime" in dir() else 0.0
        if args.once:
            break
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
