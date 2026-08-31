#!/usr/bin/env python3
"""Corner taps -> oriented safety planes (MAY-190 workcell safety rebuild).

Input: a calib/planes_YYYY-MM-DD.yaml tap session. Each corner is 2 taps ON
the wall-wall junction line + 1 tap on each wall. A plane is exactly
determined by the line and its edge tap — no fitting slack, so the printed
diagnostics are the only quality signal:

  * line tilt from vertical — sanity only, nothing assumes verticality
  * edge-tap baseline (distance edge tap <-> line) — the lever arm; with
    ~3 mm tap noise the normal is wrong by about atan(noise / baseline)
  * dihedral angle between the corner's two planes

Convention: plane = {point, normal}, normal points to the SAFE side —
oriented so the robot base origin has positive signed distance. Keep-out
is behind the plane (negative side).

Usage:  p inspection/calib/plane_fit.py [session.yaml]
Prints a `planes:` snippet ready for the cell yaml.
"""

import sys
from pathlib import Path

import numpy as np
import yaml


def plane_from_corner(line_pts, edge_pt):
    """(2 line taps, 1 edge tap) -> (point, unit normal toward origin)."""
    p1, p2 = (np.asarray(p, dtype=float) for p in line_pts)
    e = np.asarray(edge_pt, dtype=float)
    d = p2 - p1
    n = np.cross(d, e - p1)
    n /= np.linalg.norm(n)
    if n @ (np.zeros(3) - p1) < 0:      # safe side = robot base origin
        n = -n
    return p1, n


def baseline(line_pts, edge_pt):
    """Distance from the edge tap to the corner line — the lever arm."""
    p1, p2 = (np.asarray(p, dtype=float) for p in line_pts)
    d = (p2 - p1) / np.linalg.norm(p2 - p1)
    v = np.asarray(edge_pt, dtype=float) - p1
    return np.linalg.norm(v - (v @ d) * d)


def tripod_planes(taps):
    """3 taps [wallA end, APEX, wallB end] -> two VERTICAL planes through
    the apex (taps projected to xy; the corner line is assumed plumb)."""
    e_a, apex, e_b = (np.asarray(p, dtype=float) for p in taps)
    out = []
    for e in (e_a, e_b):
        d = e - apex
        n = np.array([-d[1], d[0], 0.0])
        n /= np.linalg.norm(n)
        if n @ (np.zeros(3) - apex) < 0:
            n = -n
        out.append((apex, n, np.linalg.norm(d[:2])))   # point, normal, baseline
    return out


def main():
    default = sorted(Path(__file__).parent.glob("planes_*.yaml"))[-1]
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    cfg = yaml.safe_load(path.read_text())

    TAP_NOISE = 0.003  # m, ~table-probe accuracy
    snippet = {}
    for cname, c in cfg.get("corners", {}).items():
        print(f"\n== {cname} ==")
        fitted = []                    # (point, normal, baseline) per plane

        if c.get("style") == "tripod":
            e_a, apex, e_b = (np.asarray(p, dtype=float) for p in c["taps"])
            u, v = e_a - apex, e_b - apex
            apex_deg = np.degrees(np.arccos(
                (u[:2] @ v[:2]) / (np.linalg.norm(u[:2]) * np.linalg.norm(v[:2]))))
            print(f"  tripod: apex angle {apex_deg:.1f} deg (middle tap = apex),"
                  f" planes forced vertical")
            fitted = tripod_planes(c["taps"])
        else:
            p1, p2 = (np.asarray(p, dtype=float) for p in c["line"])
            d = p2 - p1
            tilt = np.degrees(np.arccos(abs(d[2]) / np.linalg.norm(d)))
            print(f"  line: {np.linalg.norm(d) * 1e3:.0f} mm long, "
                  f"{tilt:.2f} deg from vertical")
            for e in c["edges"]:
                pt, n = plane_from_corner(c["line"], e)
                fitted.append((pt, n, baseline(c["line"], e)))

        planes = {}
        for label, (pt, n, b) in zip("ab", fitted):
            err = np.degrees(np.arctan(TAP_NOISE / b))
            print(f"  {cname}_{label}: normal [{n[0]:+.4f}, {n[1]:+.4f}, "
                  f"{n[2]:+.4f}]  baseline {b * 1e3:.0f} mm -> "
                  f"~{err:.1f} deg normal error")
            planes[label] = {"point": [round(float(x), 4) for x in pt],
                             "normal": [round(float(x), 4) for x in n]}
        dihedral = np.degrees(np.arccos(
            np.clip(fitted[0][1] @ fitted[1][1], -1, 1)))
        print(f"  dihedral between the two planes: {dihedral:.1f} deg")
        snippet[cname] = {"parent": "base", "planes": planes}

    wall_snippet = {}
    for pname, pl in cfg.get("planes", {}).items():
        if pl.get("style") == "axis":
            # declared plumb + table-parallel: one tap fixes the offset only
            t0 = np.asarray(pl["taps"][0], dtype=float)
            n = np.zeros(3)
            ax_i = "xyz".index(pl["axis"])
            n[ax_i] = 1.0 if t0[ax_i] < 0 else -1.0    # toward the robot
            print(f"\n== {pname} (axis-snapped wall, 1 tap) ==")
            print(f"  normal [{n[0]:+.1f}, {n[1]:+.1f}, {n[2]:+.1f}] declared,"
                  f" offset {pl['axis']} = {t0[ax_i]:+.4f}")
            wall_snippet[pname] = {
                "parent": "base",
                "point": [round(float(x), 4) for x in t0],
                "normal": [float(x) for x in n],
            }
            continue
        t0, t1, t2 = (np.asarray(p, dtype=float) for p in pl["taps"])
        n = np.cross(t1 - t0, t2 - t0)
        area2 = np.linalg.norm(n)
        n /= area2
        if n @ (np.zeros(3) - t0) < 0:
            n = -n
        # conditioning: the smallest triangle altitude is the lever arm
        sides = [np.linalg.norm(b - a)
                 for a, b in ((t0, t1), (t1, t2), (t2, t0))]
        alt = area2 / max(sides)
        err = np.degrees(np.arctan(TAP_NOISE / alt))
        tilt = np.degrees(np.arcsin(abs(n[2])))
        print(f"\n== {pname} (standalone wall, 3 taps) ==")
        print(f"  normal [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]  "
              f"{tilt:.2f} deg from vertical")
        print(f"  triangle altitude {alt * 1e3:.0f} mm -> "
              f"~{err:.1f} deg normal error")
        wall_snippet[pname] = {
            "parent": "base",
            "point": [round(float(x), 4) for x in t0],
            "normal": [round(float(x), 4) for x in n],
        }

    print("\n--- cell yaml snippet ---")
    out = {}
    if snippet:
        out["corners"] = snippet
    if wall_snippet:
        out["planes"] = wall_snippet
    print(yaml.dump(out, sort_keys=False, default_flow_style=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
