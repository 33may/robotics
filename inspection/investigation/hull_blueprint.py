#!/usr/bin/env python3
"""Draw the collision hull against the CAD hardware it is supposed to cover.

Two orthographic panels in the flange frame — side (X-Z) and top (X-Y) — with
every CAD part outlined, the shipped single box, and the proposed split hull
on top. The point is to make the phantom volume visible: the shipped box is
the union of each axis's worst case taken independently, so it wraps a
gripper that is simultaneously closed (long), wide open, and carrying
brackets at every station.

Anything the USD does NOT contain (Anton's camera brackets, the blue plate,
cables) is drawn from `cell.yaml`'s measured height only, and marked as
unverified along X — that gap is the whole safety question.

    p inspection/investigation/hull_blueprint.py
    p inspection/investigation/hull_blueprint.py --refresh   # re-read the USD
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data/tool"
CACHE = OUT / "usd_parts.json"

TCP_MM = 239.9                      # pendant 4-point wizard, 2026-08-18
LENS = (142.4, -8.3, -47.5)         # calibrated hand-eye, 2026-08-19

# The USD's default --slide-mm=-22 is a VISUAL nudge for the meshcat preview,
# not a physical placement. Reconciled 2026-08-24 against two independent
# measurements: at slide -22 the open fingers reach 198.1 mm and cell.yaml's
# closed-state estimate is 212; the pendant TCP is 239.9. At slide +6 the open
# reach is 226.1 and +13.8 closed lands on 239.9 EXACTLY. So the hardware sits
# 28 mm further out than the default preview draws it, and that 28 mm is the
# flange-side mount stack (X -18..+10) which the USD does not model at all.
MOUNT_STACK_MM = 28.0

CURRENT = [(-18, 240, 74.5, -100, 70)]

# Anton 2026-08-24: the stack is a CROSS, not a slab. One long shallow box for
# the gripper itself (mount stack through closed fingertips, only as tall as
# the gripper actually is), and one THIN VERTICAL slab for the camera/bracket
# assembly, centred on the calibrated lens station and 17 mm to either side.
# The shipped single box is precisely the bounding box of that cross, so every
# one of its four corners is volume the hardware never occupies — and the
# corners are what sweep into the upper arm under tilt.
# The tall part of the gripper is ONLY the base casting: base_1/4/5/11 reach
# Z -40.1..+35.2 but all of them end by real X=36. Past that the CAD never
# exceeds +-20.8. So the height step goes at X=40 — carrying -42..+37 out to
# the TCP (a first draft did) puts the phantom volume straight back at the
# front, which is the exact failure this hull exists to remove.
GRIPPER_BASE = (-18, 40, 45, -42, 37)      # mount stack + base casting
GRIPPER_BODY = (40, 240, 74.5, -25, 25)    # slender body -> closed fingertips
CAMERA_BOX = (LENS[0] - 17, LENS[0] + 17, 40, -100, 70)
PROPOSED = [GRIPPER_BASE, GRIPPER_BODY, CAMERA_BOX]


def coverage_gaps(parts, boxes, pitch=2.0):
    """CAD parts not fully inside the union of `boxes`.

    Samples each part's AABB on a `pitch` mm grid and asks whether every
    sample sits in some box. Union coverage, not per-box containment — a part
    straddling two boxes is legitimately covered. Returns
    {part: fraction_uncovered} for anything that leaks.
    """
    out = {}
    for name, (lo, hi) in parts.items():
        axes = [np.arange(lo[i], hi[i] + pitch, pitch) if hi[i] - lo[i] > 1e-9
                else np.array([lo[i]]) for i in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        covered = np.zeros(len(pts), bool)
        for x0, x1, hw, z0, z1 in boxes:
            covered |= ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                        (np.abs(pts[:, 1]) <= hw) &
                        (pts[:, 2] >= z0) & (pts[:, 2] <= z1))
        if not covered.all():
            out[name] = 1.0 - covered.mean()
    return out


def usd_parts(refresh=False):
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    from pxr import Usd
    from inspection.cell.usd_preview import DEFAULT_USD
    from inspection.investigation.tool_hull_from_usd import part_bounds
    stage = Usd.Stage.Open(str(DEFAULT_USD))
    parts = {k.rsplit("/", 1)[-1]: [(np.asarray(lo) * 1000).tolist(),
                                    (np.asarray(hi) * 1000).tolist()]
             for k, (lo, hi) in part_bounds(stage).items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(parts, indent=1))
    return parts


def draw(ax, parts, hulls, ai, bi, labels):
    """ai/bi index the axes to plot (0=X, 1=Y, 2=Z)."""
    # mount stack: 28 mm of hardware the USD has no geometry for
    ax.add_patch(mp.Rectangle((-18, -32 if bi == 2 else -32), 28, 64,
                              facecolor="tab:purple", alpha=0.30,
                              edgecolor="tab:purple", lw=1.0, zorder=3,
                              label="mount stack (not in USD)"))
    for lo, hi in parts.values():
        ax.add_patch(mp.Rectangle((lo[ai], lo[bi]), hi[ai] - lo[ai],
                                  hi[bi] - lo[bi], facecolor="0.55",
                                  edgecolor="0.25", lw=0.6, alpha=0.85, zorder=2))
    for boxes, colour, style, name in hulls:
        for j, (x0, x1, hw, z0, z1) in enumerate(boxes):
            span = {0: (x0, x1), 1: (-hw, hw), 2: (z0, z1)}
            a0, a1 = span[ai]
            b0, b1 = span[bi]
            ax.add_patch(mp.Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False,
                                      edgecolor=colour, lw=2.2, ls=style,
                                      zorder=4,
                                      label=name if j == 0 else None))
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.8, alpha=0.6)
    ax.plot([TCP_MM], [0], marker="x", ms=11, mew=2.5, color="crimson",
            zorder=6, label="pendant TCP 239.9")
    ax.plot([LENS[0]], [LENS[bi]], marker="o", ms=8, color="tab:blue",
            zorder=6, label="calibrated lens")
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(-60, 275)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    parts = usd_parts(args.refresh)
    # into REAL flange coordinates — see MOUNT_STACK_MM
    parts = {k: ([lo[0] + MOUNT_STACK_MM, lo[1], lo[2]],
                 [hi[0] + MOUNT_STACK_MM, hi[1], hi[2]])
             for k, (lo, hi) in parts.items()}
    hulls = [(CURRENT, "darkorange", "-", "shipped: one box"),
             (PROPOSED, "seagreen", "--", "proposed: CROSS (gripper + camera slab)")]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    draw(axes[0], parts, hulls, 0, 2, ("flange X  [mm]  ->  out of tool face",
                                       "flange Z  [mm]"))
    axes[0].set_title("SIDE view (X-Z).  grey = CAD gripper parts.  "
                      "the shipped box's -100..+70 height is Anton's measured "
                      "bracket span; CAD alone is only -40..+35")
    axes[0].set_ylim(-115, 90)
    for xy, xt in (((215, 55), (95, 80)), ((215, -85), (95, -70))):
        axes[0].annotate("phantom corner", xy=xy, xytext=xt, fontsize=9,
                         color="darkorange",
                         arrowprops=dict(arrowstyle="->", color="darkorange"))
    draw(axes[1], parts, hulls, 0, 1, ("flange X  [mm]", "flange Y  [mm]"))
    axes[1].set_title("TOP view (X-Y).  CAD half-width 57 mm; "
                      "the 74.5 mm box half-width is the MAX-OPEN finger span")
    axes[1].set_ylim(-90, 90)
    axes[0].legend(loc="lower left", fontsize=8, ncol=2)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "hull_blueprint.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"{len(parts)} CAD parts -> {path}")
    reach = max(hi[0] for _, hi in parts.values())
    print(f"CAD (shifted +{MOUNT_STACK_MM:.0f} mm) reaches X={reach:.1f} mm; "
          f"pendant TCP {TCP_MM} mm -> {TCP_MM - reach:.1f} mm of closed-finger "
          f"swing, consistent with cell.yaml's 212-vs-198 open/closed note")

    for label, boxes in (("shipped", CURRENT), ("proposed", PROPOSED)):
        gaps = coverage_gaps(parts, boxes)
        lens_ok = any(x0 <= LENS[0] <= x1 and abs(LENS[1]) <= hw
                      and z0 <= LENS[2] <= z1
                      for x0, x1, hw, z0, z1 in boxes)
        tcp_ok = any(x0 <= TCP_MM <= x1 for x0, x1, *_ in boxes)
        print(f"\n{label}: {'COVERS all 22 CAD parts' if not gaps else 'LEAKS'}"
              f" · lens {'in' if lens_ok else 'OUT'} · tcp {'in' if tcp_ok else 'OUT'}")
        for n, frac in sorted(gaps.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {n:<28} {frac*100:5.1f}% outside the hull")
    return 0


if __name__ == "__main__":
    sys.exit(main())
