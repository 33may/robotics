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
DEFAULT_R = 0.35                     # camera-to-center distance

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
