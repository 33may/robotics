"""What the camera sees: old executed pose vs ray endpoint, per address.

Left panel  = the run's real rgb.png with the boxguard belief projected
              through the RECORDED optical T_base_cam — the wireframe
              hugging the photo box is the built-in sanity gate (residual
              5-10 mm offset = the rig's hand-eye/FK metrology floor).
Right panel = the same wireframe through the RAY endpoint camera, old
              footprint ghosted grey — footprint growth IS the pixel gain.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / "projects/robotics/inspection/investigation"))
from ray_views.ray_replay import (box_aabb, box_world, last_box,
                                  resolve_step, run_dir_for)
from inspection.motion.ik import UR5eIK
from inspection.record.run import Run
from inspection.view.faceview import _FACE_DEF

RUNS = ["1709-cam-ai-demo1", "1509-cam-ai_bench5", "1509-cam-ai_bench4",
        "1509-lan-ai_bench5"]
W, H = 1280, 720
SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                  for sz in (-1, 1)])
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
         if np.sum(SIGNS[a] != SIGNS[b]) == 1]


def face_ids(face):
    ax, sg, _ = _FACE_DEF[face]
    return [i for i in range(8) if SIGNS[i][ax] == sg]


def optical(T, kind):
    """(R_base_opt, t). kind 'opt' = stored optical; 'int' = x fwd / z up."""
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    if kind == "int":
        R = np.column_stack([-R[:, 1], -R[:, 2], R[:, 0]])
    return R, t


def poly_area(uv, ids):
    p = uv[ids][[0, 1, 3, 2]]
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def draw(ax, uv, z, face, color, lw, ls="-"):
    for a, b in EDGES:
        if z[a] > 0.02 and z[b] > 0.02:
            ax.plot(*zip(uv[a], uv[b]), color=color, lw=lw, ls=ls)
    if face:
        ring = uv[face_ids(face)][[0, 1, 3, 2, 0]]
        ax.plot(ring[:, 0], ring[:, 1], color="#ffd400", lw=2.2 * lw / 1.2)


def render_run(run_id):
    r = Run.load(run_dir_for(run_id))
    ci = r.session.intrinsics["color"]
    K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1.0]])
    got = last_box(r)
    if got is None:
        print(f"{run_id}: no boxguard belief, skipped")
        return
    _, prim = got
    world, ik = box_world(prim, 0.0), UR5eIK()
    aabb = box_aabb(prim)
    he = np.array([float(v) for v in prim.params["extents"]]) / 2.0
    corners = prim.center + (SIGNS * he) @ prim.axes.T

    def project(T, kind):
        R, t = optical(T, kind)
        pc = (corners - t) @ R
        uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2]
        return uv, pc[:, 2]

    steps = [s for s in r.steps if s.record.view.address]
    groups = [("frames_faces.png",
               [s for s in steps if s.record.view.method == "face"]),
              ("frames_cells.png",
               [s for s in steps if s.record.view.method != "face"])]
    for name, group in groups:
        if not group:
            continue
        n = len(group)
        fig, axes = plt.subplots(n, 2, figsize=(13, 3.7 * n))
        if n == 1:
            axes = axes[None, :]
        for row, s in zip(axes, group):
            v = s.record.view
            face = v.address[1] if v.method == "face" else None
            label = (f"face {face}" if face
                     else f"cell {v.address[0]},{v.address[1]}")
            res, _ = resolve_step(world, ik, prim, aabb, s, ci.fy, ci.height)

            uv_o, z_o = project(s.T_base_cam, "opt")
            if s.record.rgb_rotation_deg == 180:
                uv_o = np.array([W, H]) - uv_o   # raw -> upright disk frame
            row[0].imshow(np.asarray(Image.open(s.dir / "rgb.png")))
            draw(row[0], uv_o, z_o, face, "#2fbf4f", 1.6)
            row[0].set_title(f"{label} — OLD executed camera (real photo)",
                             fontsize=10)

            row[1].set_facecolor("#e8e8e8")
            draw(row[1], uv_o, z_o, None, "#999999", 1.0, ls="--")
            gain = ""
            if res is not None:
                uv_r, z_r = project(res.T_cam, "int")
                if np.isclose(float(res.roll), np.pi):   # roll is RADIANS
                    # roll-180 endpoints capture upside down; the pipeline
                    # de-rotates those images (rotated_artifacts), so show
                    # the frame the way every consumer would see it.
                    uv_r = np.array([W, H]) - uv_r
                draw(row[1], uv_r, z_r, face, "#2fbf4f", 1.8)
                if face:
                    ids = face_ids(face)
                    gain = (f"  face area x"
                            f"{poly_area(uv_r, ids) / max(poly_area(uv_o, ids), 1e-6):.1f}")
                extra = (f"θ={res.theta_deg:.0f}° shift={res.shift:.2f} "
                         f"d={res.depth:.2f}")
            else:
                extra = "INACCESSIBLE"
            row[1].set_title(f"RAY endpoint ({extra}){gain} — "
                             f"grey dashed = old footprint", fontsize=10)
            for ax in row:
                ax.set_xlim(0, W); ax.set_ylim(H, 0)
                ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out = Path(f"/tmp/ray_views/{run_id}/{name}")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote", out)


for rid in RUNS:
    render_run(rid)
