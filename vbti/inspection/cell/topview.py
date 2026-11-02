#!/usr/bin/env python3
"""Top view of the cell yaml — the correctness check for every new shape.

Renders what the yaml SAYS, nothing else: box footprints (through the frame
tree), safety plane traces with their keep-out side hatched, the robot base
and its 850 mm reach circle. Tool boxes are flange-mounted, so they have no
fixed world pose and are skipped.

Usage:  p inspection/cell/topview.py [--yaml cell/cell.yaml] [-o topview.png]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

CELL_YAML = Path(__file__).parent / "cell.yaml"
REACH_M = 0.85          # UR5e nominal reach
INK, BOX, PLANE = "#333333", "#7a8ba6", "#c0392b"


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (f(a) for a in (roll, pitch, yaw)
                              for f in (np.cos, np.sin))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _pose_to_T(pose):
    T = np.eye(4)
    T[:3, :3] = _rot(*pose[3:])
    T[:3, 3] = pose[:3]
    return T


def _frame_to_base(frames, name):
    if name in (None, "base"):
        return np.eye(4)
    f = frames[name]
    return _frame_to_base(frames, f.get("parent", "base")) @ _pose_to_T(f["pose"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=str(CELL_YAML))
    ap.add_argument("-o", "--out", default=str(Path(__file__).parent / "topview.png"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.yaml).read_text())
    frames = cfg.get("frames", {})

    fig, ax = plt.subplots(figsize=(9, 9))

    # --- box footprints: 4 bottom corners through the frame tree
    for name, box in cfg.get("boxes", {}).items():
        T = _frame_to_base(frames, box.get("parent", "base")) @ _pose_to_T(box["pose"])
        dx, dy, dz = (d / 2 for d in box["dims"])
        loc = np.array([[sx * dx, sy * dy, -dz, 1.0]
                        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        xy = (loc @ T.T)[:, :2]
        ax.add_patch(plt.Polygon(xy, closed=True, facecolor=BOX, alpha=0.35,
                                 edgecolor=BOX, lw=1.2))
        ax.annotate(name, xy.mean(axis=0), ha="center", va="center",
                    fontsize=8, color=INK)

    # --- standalone safety planes: keep-out = whole half-space behind
    for name, pl in cfg.get("planes", {}).items():
        Tp = _frame_to_base(frames, pl.get("parent", "base"))
        p = (Tp @ np.array([*pl["point"], 1.0]))[:2]
        n = (Tp[:3, :3] @ np.array(pl["normal"], dtype=float))
        n_xy = n[:2] / np.linalg.norm(n[:2])           # safe side, top view
        t = np.array([-n_xy[1], n_xy[0]])              # along the wall
        L, W = 3.0, 0.12
        a, b = p - L * t, p + L * t
        ax.plot(*zip(a, b), color=PLANE, lw=2)
        ax.add_patch(plt.Polygon([a, b, b - W * n_xy, a - W * n_xy],
                                 closed=True, facecolor="none", edgecolor=PLANE,
                                 hatch="///", lw=0, alpha=0.45))
        ax.annotate("", xytext=p, xy=p + 0.2 * n_xy,
                    arrowprops=dict(arrowstyle="->", color=PLANE, lw=1.5))
        ax.annotate(name, p + 0.24 * n_xy, fontsize=8, color=PLANE,
                    ha="center", fontweight="bold")

    # --- corners: keep-out = INTERSECTION of the half-spaces behind the two
    # planes — hatch only the wedge where BOTH are violated
    for cname, corner in cfg.get("corners", {}).items():
        Tp = _frame_to_base(frames, corner.get("parent", "base"))
        ps, ns = [], []
        for pl in corner["planes"].values():
            ps.append((Tp @ np.array([*pl["point"], 1.0]))[:2])
            n = Tp[:3, :3] @ np.array(pl["normal"], dtype=float)
            ns.append(n[:2] / np.linalg.norm(n[:2]))
        t = [np.array([-n[1], n[0]]) for n in ns]      # trace directions
        # corner point in top view: intersection of the two traces
        s = np.linalg.solve(np.column_stack([t[0], -t[1]]), ps[1] - ps[0])
        c = ps[0] + s[0] * t[0]
        L = 3.0
        walls = []
        for i, j in ((0, 1), (1, 0)):
            # each wall ray runs from the corner into the OTHER plane's
            # violated side — that is where the physical wall extends
            ti = t[i] if t[i] @ ns[j] < 0 else -t[i]
            walls.append(ti)
            ax.plot(*zip(c, c + L * ti), color=PLANE, lw=2)
        wedge = [c, c + L * walls[0], c + L * (walls[0] + walls[1]),
                 c + L * walls[1]]
        ax.add_patch(plt.Polygon(wedge, closed=True, facecolor=PLANE,
                                 alpha=0.08, edgecolor=PLANE, hatch="///", lw=0))
        bis = -(walls[0] + walls[1])
        bis /= np.linalg.norm(bis)                     # safe-side bisector
        ax.annotate("", xytext=c, xy=c + 0.2 * bis,
                    arrowprops=dict(arrowstyle="->", color=PLANE, lw=1.5))
        ax.annotate(cname, c + 0.26 * bis, fontsize=8, color=PLANE,
                    ha="center", fontweight="bold")

    # --- robot base + reach
    ax.plot(0, 0, "o", color=INK, ms=8)
    ax.annotate("base", (0.03, 0.03), fontsize=8, color=INK)
    ax.add_patch(plt.Circle((0, 0), REACH_M, fill=False, ls="--",
                            color=INK, lw=0.8, alpha=0.5))
    ax.annotate("reach 0.85", (0, REACH_M + 0.03), ha="center",
                fontsize=7, color=INK, alpha=0.7)

    # frame bounds from boxes + reach only (planes are infinite)
    pts = [(-REACH_M, -REACH_M), (REACH_M, REACH_M)]
    for name, box in cfg.get("boxes", {}).items():
        T = _frame_to_base(frames, box.get("parent", "base")) @ _pose_to_T(box["pose"])
        r = np.linalg.norm(box["dims"][:2]) / 2
        pts += [(T[0, 3] - r, T[1, 3] - r), (T[0, 3] + r, T[1, 3] + r)]
    pts = np.array(pts)
    m = 0.15
    ax.set_xlim(pts[:, 0].min() - m, pts[:, 0].max() + m)
    ax.set_ylim(pts[:, 1].min() - m, pts[:, 1].max() + m)

    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlabel("x [m]  (controller base frame)")
    ax.set_ylabel("y [m]")
    ax.set_title(f"cell top view — {Path(args.yaml).name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys = __import__("sys")
    sys.exit(main())
