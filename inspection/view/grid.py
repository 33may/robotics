#!/usr/bin/env python3
"""Grid vocabulary for the viewsphere — index math and the words for it.

The light half of the viewsphere: cell addressing (h wraps, v clamps), the
egocentric gloss both AI tiers speak, and the coverage bitmap's rendering.
numpy only — importing this must NEVER pull in IK/motion/pinocchio, because
the image tier (eyes/) and the decider both live on it (Anton 2026-08-24).

Move legality comes from motion (`viewsphere.reachability`), so `moves_from`
takes the feasible set as an argument rather than computing it.
"""
from dataclasses import dataclass

import numpy as np

H_BINS = 12                          # 30 deg each, h=0 faces the robot base
V_ELEVATIONS = (10.0, 40.0, 70.0)    # deg above the table (Anton 2026-08-18)
DEFAULT_R = 0.35                     # fallback only — the shell is DERIVED

# ---- shell radius: derived from the object, not declared -------------------
# The object should occupy the same fraction of frame height whether it is a
# 30 mm bolt or a 300 mm bottle — that is what keeps the system general.
# Computed ONCE from the survey cloud and then frozen: cells are the keys for
# coverage and evidence, so a shell that breathes as the cloud grows would
# silently re-point addresses that have already been visited.
# FILL_TARGET is NOMINAL, measured against the AABB diagonal — and no cell
# ever sees all three axes at once, so the real silhouette comes out ~1.4x
# smaller than nominal (fill_probe on 2408-seeded: nominal 0.43, measured
# 0.20-0.36, mean 0.31). 0.62 nominal is therefore ~0.45 of frame height in
# the actual images. Verify a change with:
#   p inspection/investigation/fill_probe.py --run <run> --fill <f>
#   p inspection/investigation/frame_preview.py --run <run>
FILL_TARGET = 0.62        # nominal fill of frame height, on the diagonal
EXTENT_RULE = "sphere"    # which dimension FILL_TARGET is measured against
R_MIN = 0.24              # measured floor — see below; NOT a tool clearance
R_MAX = 0.40              # past this reachability collapses (24/36 cells)
# R_MIN was 0.22, taken from a sweep on the DEMO cup at the DEMO position. With
# the real object where it actually sits (run 2408-cup1) that shell reaches
# only 19/36 cells against 28/36 at 0.24 — a cliff, and the run that used
# 0.226 wasted most of its orbit on blocked cells.
# The cliff is NOT tool geometry and NOT the cup: it is how much the object is
# inflated. A 94x115x111 mm cup becomes AABB+40 mm, and coal adds another
# 20 mm each side -> the solver sees ~174x195x191 mm. Thinning that recovers
# 0.22 completely (+20 mm box: 29/36) but the padding IS the standoff
# (Anton 2026-08-18, the inspected object is a hard obstacle), so the floor
# moves instead of the margin. Re-derive with investigation/block_report.py
# if the padding contract ever changes.


def object_extent(mn, mx, rule=EXTENT_RULE):
    """The object dimension the fill target is measured against [m].

    The three rules differ by which viewing cell they protect, and the choice
    moves the radius roughly 2x — more than the fill target itself does:
      "sphere"    AABB diagonal — the silhouette from the WORST cell, so no
                  cell can clip. Conservative; high rings under-fill.
      "upright"   AABB height — what a low side view sees.
      "footprint" largest horizontal extent — what a top-down view sees.
    """
    d = np.asarray(mx, dtype=float) - np.asarray(mn, dtype=float)
    if rule == "sphere":
        return float(np.linalg.norm(d))
    if rule == "upright":
        return float(d[2])
    if rule == "footprint":
        return float(np.max(d[:2]))
    raise ValueError(f"unknown extent rule {rule!r}")


def radius_for_extent(extent, fy, height_px, fill=FILL_TARGET):
    """Camera distance at which `extent` metres span `fill` of the frame.

    Pinhole similar triangles: a length L at distance r projects to fy*L/r
    pixels, so r = fy*L / (fill*height_px). Clamped to the reachable band —
    a clamped radius is an honest "as close as this cell allows", not a
    failure, so it returns silently.
    """
    r = fy * float(extent) / (fill * height_px)
    return float(np.clip(r, R_MIN, R_MAX))

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


@dataclass(frozen=True)
class Move:
    cell: tuple
    gloss: str


def step_delta(cur_h, h, h_bins=H_BINS):
    """Signed shortest azimuth steps cur -> h; +1 = one step right."""
    return (h - cur_h + h_bins // 2) % h_bins - h_bins // 2


def cell_gloss(cur, cell, h_bins=H_BINS, elevations=V_ELEVATIONS):
    """Egocentric label for cell relative to cur (D4 menu gloss)."""
    if cur is None:
        return f"elevation {elevations[cell[1]]:.0f} deg"
    dh = step_delta(cur[0], cell[0], h_bins)
    dv = cell[1] - cur[1]
    if abs(dh) == h_bins // 2:
        side = "opposite side"
    elif dh == 0:
        side = "same side"
    else:
        n = abs(dh)
        side = f"{_WORDS[n]} step{'s' if n > 1 else ''} " \
               f"{'right' if dh > 0 else 'left'}"
    height = "same height" if dv == 0 else ("higher" if dv > 0 else "lower")
    return f"{side}, {height}"


def neighbors(cell, h_bins=H_BINS, n_v=len(V_ELEVATIONS)):
    """4-connected neighbours: h wraps around the ring, v clamps at the ends."""
    h, v = cell
    out = [((h - 1) % h_bins, v), ((h + 1) % h_bins, v)]
    if v - 1 >= 0:
        out.append((h, v - 1))
    if v + 1 < n_v:
        out.append((h, v + 1))
    return out


def coverage_map(cov, cur=None, elevations=V_ELEVATIONS):
    """ASCII bitmap of a (n_v, h_bins) coverage array, highest ring first."""
    cov = np.asarray(cov, dtype=bool)
    n_v, n_h = cov.shape
    rows = []
    for v in range(n_v - 1, -1, -1):
        line = ""
        for h in range(n_h):
            if cur is not None and (int(cur[0]), int(cur[1])) == (h, v):
                line += "@"
            else:
                line += "#" if cov[v, h] else "."
        rows.append(f"v{v} {elevations[v]:>2.0f}deg |{line}|")
    rows.append(" " * 9 + "h" + "".join(str(h % 10) for h in range(n_h)))
    return "\n".join(rows)


def moves_from(cur, allowed, h_bins=H_BINS, elevations=V_ELEVATIONS):
    """Describe an allowed cell set from cur, nearest first. Feasibility is
    the caller's business — this only orders and names the options."""
    def dist(cell):
        if cur is None:
            return (0, cell[0], cell[1])
        return (abs(step_delta(cur[0], cell[0], h_bins)) + abs(cell[1] - cur[1]),
                cell[0], cell[1])

    return [Move(tuple(c), cell_gloss(cur, c, h_bins, elevations))
            for c in sorted(allowed, key=dist)]
