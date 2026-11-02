"""Viewsphere figure from a recorded step: rays from the object, the view on each.

Reads view_state.json + decomposition.json of one step (read-only) and renders a
3D figure with no on-figure text: table, fitted box, one ray per cell, the view
sample on every reachable ray, unreachable rays dashed, one ray with its 2 cm
sampling shown.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

step_dir, out = sys.argv[1], sys.argv[2]
vs = json.load(open(f"{step_dir}/view_state.json"))
dec = json.load(open(f"{step_dir}/decomposition.json"))
prim = dec["primitives"][0]
T_box = np.array(prim["pose"])
ext = np.array(prim["params"]["extents"])
c = T_box[:3, 3]
R_MAX, MIN_RANGE, STEP = 0.40, 0.105, 0.02

cells = [k for k in vs["candidates"] if isinstance(k["address"][0], int)]

ORANGE, GREY, DARK, RED = "#d9731a", "#b8b2a8", "#4a4a4a", "#c0392b"
fig = plt.figure(figsize=(8, 8), dpi=200)
ax = fig.add_subplot(111, projection="3d")
ax.computed_zorder = False
ax.set_proj_type("ortho")
ax.set_facecolor("white"); fig.patch.set_facecolor("white")

# table: a flat patch under the object
zt = c[2] - ext[2] / 2
tx, ty = 0.30, 0.30
table = [[c[0]-tx, c[1]-ty, zt], [c[0]+tx, c[1]-ty, zt],
         [c[0]+tx, c[1]+ty, zt], [c[0]-tx, c[1]+ty, zt]]
tab = Poly3DCollection([table], facecolor="#f1ede6", edgecolor="#d8d2c8", linewidth=0.8)
tab.set_zorder(1); ax.add_collection3d(tab)
for g in np.linspace(-tx, tx, 9):
    ax.plot([c[0]+g, c[0]+g], [c[1]-ty, c[1]+ty], [zt, zt], color="#ddd7cd", lw=0.5, zorder=1)
    ax.plot([c[0]-tx, c[0]+tx], [c[1]+g, c[1]+g], [zt, zt], color="#ddd7cd", lw=0.5, zorder=1)

# the fitted box
h = ext / 2
corners = np.array([[sx*h[0], sy*h[1], sz*h[2]] for sx in (-1, 1)
                    for sy in (-1, 1) for sz in (-1, 1)])
corners = (T_box[:3, :3] @ corners.T).T + c
faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
bx = Poly3DCollection([corners[f] for f in faces], facecolor="#f3dcc4",
                      edgecolor=ORANGE, linewidth=1.0, alpha=0.95)
bx.set_zorder(2); ax.add_collection3d(bx)

# rays
hi = None
for k in cells:
    p = np.array(k["pose"])[:3, 3]
    d = p - c
    dist = np.linalg.norm(d)
    u = d / dist
    end = c + u * R_MAX
    if k["status"] == "unreachable":
        ax.plot(*zip(c, end), color=GREY, lw=0.9, ls=(0, (2, 3)), alpha=0.9, zorder=3)
        continue
    ax.plot(*zip(c, end), color=GREY, lw=0.9, alpha=0.9, zorder=3)
    ax.scatter(*p, s=28, color=ORANGE, depthshade=False, zorder=5)

# one ray with its sampling shown: failed samples as crosses, the view filled
if hi is not None:
    u, dist = hi
    ax.plot(*zip(c, c + u * R_MAX), color=DARK, lw=1.6, zorder=4)
    for s in np.arange(np.ceil(MIN_RANGE / STEP) * STEP, R_MAX + 1e-9, STEP):
        q = c + u * s
        if s < dist - 1e-6:
            ax.scatter(*q, s=26, marker="x", color=RED, lw=1.2, depthshade=False, zorder=6)
        elif abs(s - dist) < 1e-6 or (s > dist and s - dist < STEP):
            pass
        else:
            ax.scatter(*q, s=10, color=GREY, depthshade=False, zorder=4)
    ax.scatter(*(c + u * dist), s=90, color=ORANGE, edgecolor=DARK, lw=1.2,
               depthshade=False, zorder=7)

ax.set_xlim(c[0]-0.34, c[0]+0.34); ax.set_ylim(c[1]-0.34, c[1]+0.34)
ax.set_zlim(zt-0.02, zt + 0.44)
ax.set_box_aspect((0.68, 0.68, 0.46))
ax.view_init(elev=26, azim=-50)
ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(out, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print("saved", out, "cells", len(cells),
      "reachable", sum(k["status"] != "unreachable" for k in cells))
