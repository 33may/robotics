"""Top-down viewsphere figure from a recorded step: 12 rays of one elevation ring.

Reads view_state.json + decomposition.json (read-only). Draws the fitted box
footprint, the 12 rays of ring `v`, the view point on each reachable ray,
dashed rays where no sample passed, and the 2 cm sampling on one ray. No text.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

step_dir, out = sys.argv[1], sys.argv[2]
V = int(sys.argv[3]) if len(sys.argv) > 3 else 1
HI = int(sys.argv[4]) if len(sys.argv) > 4 else 2
vs = json.load(open(f"{step_dir}/view_state.json"))
dec = json.load(open(f"{step_dir}/decomposition.json"))
prim = dec["primitives"][0]
T = np.array(prim["pose"]); ext = np.array(prim["params"]["extents"])
c = T[:3, 3]
R_MAX, MIN_RANGE, STEP = 0.40, 0.105, 0.02
ELEV = np.radians((10.0, 40.0, 70.0)[V])
cosv = np.cos(ELEV)

ORANGE, GREY, DARK, RED = "#d9731a", "#b5afa5", "#3d3d3d", "#c0392b"
fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

# box footprint (top-down)
h = ext / 2
sq = np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]])
R2 = T[:2, :2]
foot = (R2 @ sq.T).T + c[:2]
ax.add_patch(Polygon(foot, closed=True, facecolor="#f3dcc4", edgecolor=ORANGE, lw=1.6, zorder=3))

ring = [k for k in vs["candidates"]
        if isinstance(k["address"][0], int) and k["address"][1] == V]
for k in sorted(ring, key=lambda k: k["address"][0]):
    p = np.array(k["pose"])[:3, 3]
    d = p - c
    dist = float(np.linalg.norm(d))
    u2 = d[:2] / np.linalg.norm(d[:2])          # azimuth direction, top-down
    u3 = d / dist
    ub = T[:3, :3].T @ u3                        # direction in the box frame
    clear = float(np.min(h / np.maximum(np.abs(ub), 1e-9)))   # centre -> surface
    ub2 = R2.T @ u2                              # in-plane direction, box frame
    clear2 = float(np.min(h[:2] / np.maximum(np.abs(ub2), 1e-9)))
    start = c[:2] + u2 * clear2                  # drawn from the footprint edge
    end = c[:2] + u2 * (clear + R_MAX) * cosv
    if k["status"] == "unreachable":
        ax.plot(*zip(start, end), color=GREY, lw=1.2, ls=(0, (2, 3)), zorder=2)
        continue
    hi = k["address"][0] == HI
    ax.plot(*zip(start, end), color=DARK if hi else GREY, lw=1.8 if hi else 1.2, zorder=4 if hi else 2)
    view = c[:2] + u2 * dist * cosv
    if hi:
        for s in np.arange(clear + MIN_RANGE, clear + R_MAX + 1e-9, STEP):
            q = c[:2] + u2 * s * cosv
            if s < dist - 1e-6:
                ax.scatter(*q, s=55, marker="x", color=RED, lw=1.4, zorder=6)
            elif s > dist + 1e-6:
                ax.scatter(*q, s=14, color=GREY, zorder=5)
        ax.scatter(*view, s=160, color=ORANGE, edgecolor=DARK, lw=1.4, zorder=7)
    else:
        ax.scatter(*view, s=60, color=ORANGE, zorder=5)

lim = (R_MAX + 0.12) * cosv + 0.04
ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim)
ax.set_aspect("equal"); ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(out, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print("saved", out, "ring", V, "reachable",
      sum(k["status"] != "unreachable" for k in ring), "of", len(ring))
