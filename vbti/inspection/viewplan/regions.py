#!/usr/bin/env python3
"""Region-refined viewsphere — PROTOTYPE for Anton's eyeball review.

The cell stays an abstraction of WHERE; its concrete pose is recomputed per
step as the most-orthogonal view *inside its region of the shell*. No set
cover, no subset: all 36 cells, every step.

Mechanics (the constructive "normal idea", no blind sampling):
  step cloud -> oriented normals -> Poisson mesh -> 3000 surface samples
  -> extend each sample's normal ray until it pierces the shell (that
  intersection is the ideal shell position for that patch, boresight back
  along the ray) -> bin intersections into (h, v) cells -> cap per cell
  -> score every candidate with the UNCHANGED visibility machinery
  (frustum, occlusion raycast, 60 deg gate, cos-weighted visible area)
  -> argmax per cell. Cells with zero intersections keep the classic
  cell-center pose, flagged `empty`. The classic poses are scored through
  the same matrix, so orig-vs-best is a fair fight.

Review surface (rerun, blueprint baked in + .rbl sidecar):
  LEFT   one 3D space: cloud, the sphere GRID (region boundaries + cell
         labels), original cams (gray) vs region-best cams (colored by
         achieved mean incidence, green = orthogonal). Frusta 4x smaller
         than the record workspace so 72 stay readable.
  RIGHT  three tabs, one per elevation ring. Each tab: 12 azimuth rows,
         each row = [orig render | best render] side by side, rows aligned
         vertically. Scrub the `step` timeline; everything follows.

Shell: center = final accumulated cloud centroid, radius = median recorded
camera distance (the runs flew on their sphere, the cams encode it); h=0
faces the robot base, elevations from view/grid.py. The standoff gate's
R_max is widened to the shell radius — the stock [0.07, 0.20] window would
zero every pair at r~0.24. All other gates untouched.

Prototype honesty: not scalable, not wired into the loop, no reachability.

    p inspection/viewplan/regions.py --run 0809-box6 --open
    p inspection/viewplan/regions.py --run 0809-box2 --steps 5,12,26
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from inspection.view.grid import DEFAULT_R, H_BINS, V_ELEVATIONS
from inspection.viewplan import mesh as meshmod
from inspection.viewplan import planner
from inspection.viewplan import render as rendermod
from inspection.viewplan.contract import look_at_cv

DERIVED = planner.DATA / "derived"
FRUSTUM_M = 0.06 / 4.0        # record/rr FRUSTUM_M, 4x smaller (Anton)
CAP_PER_CELL = 20             # candidate cap; keeps the matrix at POC scale
V_EDGES = [(a + b) / 2 for a, b in zip(V_ELEVATIONS, V_ELEVATIONS[1:])]
RENDER_SCALE = 0.35           # 848x480 -> ~297x168 per pane; 72 panes/step


# ---------------------------------------------------------------- shell
def shell_of(run: str) -> Dict[str, float]:
    """center (final-cloud centroid), radius (median cam distance), az0."""
    plys = sorted((DERIVED / run / "steps").glob("step_*.ply"))
    if not plys:
        raise SystemExit(f"no derived step clouds for {run}")
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(str(plys[-1])).points)
    center = P.mean(axis=0)
    cams = planner.recorded_cams(run)
    r = (float(np.median(np.linalg.norm(cams - center, axis=1)))
         if len(cams) else DEFAULT_R)
    d = -center[:2]
    return {"center": center.tolist(), "r": r,
            "az0": float(np.arctan2(d[1], d[0]))}


def cell_dir(shell: Dict, h: float, v_el_deg: float) -> np.ndarray:
    az = shell["az0"] + h * (2 * np.pi / H_BINS)
    el = np.deg2rad(v_el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el)])


def orig_pose(shell: Dict, h: int, v: int) -> np.ndarray:
    c = np.asarray(shell["center"])
    pos = c + shell["r"] * cell_dir(shell, h, V_ELEVATIONS[v])
    return look_at_cv(pos, c - pos)


def bin_cells(shell: Dict, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(h, v) region of each shell position. v bands split at ring midpoints."""
    c = np.asarray(shell["center"])
    d = pos - c
    az = np.arctan2(d[:, 1], d[:, 0]) - shell["az0"]
    width = 2 * np.pi / H_BINS
    hh = np.floor((az + width / 2) / width).astype(int) % H_BINS
    el = np.rad2deg(np.arcsin(np.clip(d[:, 2] / shell["r"], -1, 1)))
    vv = np.digitize(el, V_EDGES)
    return hh, vv


def ray_shell_hits(S: np.ndarray, SN: np.ndarray, shell: Dict
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Outward normal-ray / shell intersections. -> (positions, sample idx)."""
    c, r = np.asarray(shell["center"]), shell["r"]
    pc = S - c
    b = np.einsum("ij,ij->i", SN, pc)
    disc = b ** 2 - (np.einsum("ij,ij->i", pc, pc) - r ** 2)
    ok = disc > 0
    t = -b + np.sqrt(np.where(ok, disc, 0.0))
    ok &= t > 1e-6                              # outward hit exists
    return S[ok] + t[ok, None] * SN[ok], np.nonzero(ok)[0]


# ---------------------------------------------------------------- planning
def territory_metrics(cosv: np.ndarray, cover: np.ndarray, area: float,
                      terr: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """score / n_seen / mean_inc restricted to territory columns.

    Same formulas as the global ones in refine_step; the restriction IS
    lane A — a pose is judged only on the samples whose normal ray landed
    in its cell, so the dense top face cannot outvote a holey side.
    """
    cv = cosv[:, terr]
    score = (cv * area).sum(axis=1)
    n_seen = cover[:, terr].sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean_inc = np.degrees(np.arccos(
            np.clip(cv.sum(axis=1) / np.maximum(n_seen, 1), 0, 1)))
    return score, n_seen, mean_inc


def refine_step(P: np.ndarray, run: str, shell: Dict, seed: int = 0,
                territory: bool = False, membrane: bool = False):
    """All 36 cells refined against one step cloud. -> (record, mesh).

    Two independent fixes for the top-attraction failure (2026-09-10 review),
    composable, each behind its own flag so the 2x2 matrix stays honest:
      territory  (lane A) score each cell ONLY on its own territory — the
                 samples whose normal-ray hit landed in that cell. The dense
                 top can no longer outvote a holey side.
      membrane   (lane B) loosen the Poisson density trim 6mm -> 20mm: the
                 membrane Poisson interpolates across partially-seen holes
                 survives, so holey sides contribute plane-continuing
                 normals and candidates. NOT trim 0 (that deletes every
                 vertex and the ladder silently falls back to alpha) and
                 not 50mm (table skirts) — 20mm measured as the sweet spot
                 on 0809-box6 steps 12/26 (/tmp/membrane_check sweep,
                 2026-09-10). Truly-unseen walls stay open — deliberate.
    """
    cams = planner.recorded_cams(run)
    m, info = meshmod.reconstruct(P, seed=seed,
                                  cams=cams if len(cams) else None,
                                  trim_m=0.02 if membrane else meshmod.TRIM_M)
    S, SN, area = planner.sample_surface(m, 3000, seed, cloud=P)
    table = meshmod.table_plane(P)

    hits, src = ray_shell_hits(S, SN, shell)
    if len(hits):                    # 10 mm table guard on camera positions
        above = hits @ np.asarray(table[:3]) + table[3] >= 0.01
        hits, src = hits[above], src[above]
    hh, vv = bin_cells(shell, hits) if len(hits) else (np.array([], int),) * 2

    cand_T, cand_cell = [], []
    rng = np.random.default_rng(seed)
    for v in range(len(V_ELEVATIONS)):
        for h in range(H_BINS):
            idx = np.nonzero((hh == h) & (vv == v))[0]
            if len(idx) > CAP_PER_CELL:
                idx = rng.choice(idx, CAP_PER_CELL, replace=False)
            for i in idx:
                cand_T.append(look_at_cv(hits[i], S[src[i]] - hits[i]))
                cand_cell.append((h, v))
    orig_i0 = len(cand_T)
    for v in range(len(V_ELEVATIONS)):
        for h in range(H_BINS):
            cand_T.append(orig_pose(shell, h, v))
            cand_cell.append((h, v))
    T = np.asarray(cand_T)

    params = dict(planner.params_for("base"))
    params["standoff_window_m"] = [0.07, shell["r"] + 0.05]   # widen R_max
    intr = planner.session_intrinsics(run)
    cover, cosv, _gate = planner.visibility(m, T, S, SN, intr, params)
    score = (cosv * area).sum(axis=1)                          # cos-wtd area
    n_seen = cover.sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean_inc = np.degrees(np.arccos(
            np.clip(cosv.sum(axis=1) / np.maximum(n_seen, 1), 0, 1)))

    # lane A: a cell's territory = unique sample indices whose normal-ray
    # hit binned into it. Precomputed once; the loop below swaps the
    # global metrics for territory-restricted ones per cell.
    if territory:
        terr_of = {(h, v): np.unique(src[(hh == h) & (vv == v)])
                   for v in range(len(V_ELEVATIONS))
                   for h in range(H_BINS)}

    cells = {}
    for v in range(len(V_ELEVATIONS)):
        for h in range(H_BINS):
            mine = [i for i in range(orig_i0) if cand_cell[i] == (h, v)]
            oi = orig_i0 + v * H_BINS + h
            if territory:      # orig and candidates judged on the SAME
                terr = terr_of[h, v]          # local ground: fair fight
                sc, ns, mi = territory_metrics(cosv, cover, area, terr)
            else:
                sc, ns, mi = score, n_seen, mean_inc
            rec = {"cell": [h, v], "n_candidates": len(mine),
                   "orig": {"T_base_cam": T[oi].tolist(),
                            "score": float(sc[oi]),
                            "n_seen": int(ns[oi]),
                            "mean_incidence_deg": float(mi[oi])}}
            if territory:
                rec["territory_n"] = int(len(terr))
            if mine:
                bi = mine[int(np.argmax(sc[mine]))]
                rec["best"] = {"T_base_cam": T[bi].tolist(),
                               "score": float(sc[bi]),
                               "n_seen": int(ns[bi]),
                               "mean_incidence_deg": float(mi[bi])}
                rec["empty"] = False
                rec["gain"] = float(sc[bi] / sc[oi]) if sc[oi] > 0 \
                    else float("inf")
            else:
                rec["best"] = rec["orig"]
                rec["empty"] = True
                rec["gain"] = 1.0
            cells[f"h{h}v{v}"] = rec
    record = {"run": run, "shell": shell, "mesh": info.get("method"),
              "n_hits": int(len(hits)), "cells": cells}
    return record, m


# ---------------------------------------------------------------- rerun
def _arc(shell: Dict, az_steps, el_steps) -> np.ndarray:
    c, r = np.asarray(shell["center"]), shell["r"]
    pts = [c + r * cell_dir(shell, a, e) for a, e in zip(az_steps, el_steps)]
    return np.asarray(pts)


def _log_grid(shell: Dict) -> None:
    """Static sphere grid: 12 meridian boundaries, ring-boundary parallels,
    and a labelled marker at every cell center."""
    import rerun as rr
    strips = []
    els = np.linspace(0.0, 88.0, 24)
    for h in range(H_BINS):                       # boundaries at h +/- 0.5
        strips.append(_arc(shell, np.full_like(els, h + 0.5), els))
    azs = np.linspace(0.0, H_BINS, 96)
    for el in [0.0, *V_EDGES]:                    # base + band boundaries
        strips.append(_arc(shell, azs, np.full_like(azs, el)))
    rr.log("world/grid", rr.LineStrips3D(
        strips, colors=[[90, 90, 110]], radii=[0.0006]), static=True)
    pos, labels = [], []
    c, r = np.asarray(shell["center"]), shell["r"]
    for v in range(len(V_ELEVATIONS)):
        for h in range(H_BINS):
            pos.append(c + 1.02 * r * cell_dir(shell, h, V_ELEVATIONS[v]))
            labels.append(f"h{h}v{v}")
    rr.log("world/grid/cells", rr.Points3D(
        pos, labels=labels, show_labels=False, colors=[[150, 150, 170]],
        radii=[0.002]), static=True)


def _inc_color(rec: Dict):
    if rec["empty"]:
        return [120, 120, 120]
    f = float(np.clip(1.0 - rec["best"]["mean_incidence_deg"] / 60.0, 0, 1))
    return [int(220 * (1 - f) + 40 * f), int(60 * (1 - f) + 200 * f), 60]


C_ORIG = [135, 135, 145]          # every original grid camera: flat gray
C_BEST = [235, 55, 235]           # every region-best camera: flat magenta


def _log_cam(path: str, T: np.ndarray, intr: Dict, color, mark_m: float
             ) -> None:
    import rerun as rr
    rr.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    rr.log(path, rr.Pinhole(
        resolution=[intr["width"], intr["height"]],
        focal_length=[intr["fx"], intr["fy"]],
        principal_point=[intr["ppx"], intr["ppy"]],
        camera_xyz=rr.ViewCoordinates.RDF,
        image_plane_distance=FRUSTUM_M,
        color=color))
    rr.log(path + "/mark", rr.Points3D([[0, 0, 0]], colors=[color],
                                       radii=[mark_m]))


def _ring_sheet(scene, rec: Dict, v: int, intr: Dict, window) -> np.ndarray:
    """One composite image per ring: 12 rows of [orig | best], labels burned
    in. One image entity scrubs fast; 24 view panes did not."""
    import cv2
    rows = []
    for h in range(H_BINS):
        c = rec["cells"][f"h{h}v{v}"]
        pair = []
        for tag in ("orig", "best"):
            img, _ = rendermod.render_view(
                scene, np.asarray(c[tag]["T_base_cam"]), intr,
                scale=RENDER_SCALE, window=window)
            img = np.ascontiguousarray(img)
            label = f"h{h} {tag}  {c[tag]['mean_incidence_deg']:.0f}deg"
            if tag == "best" and c["empty"]:
                label += " (empty)"
            cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
            pair.append(img)
        sep = np.full((pair[0].shape[0], 2, 3), 60, np.uint8)
        rows.append(np.concatenate([pair[0], sep, pair[1]], axis=1))
        rows.append(np.full((2, rows[-1].shape[1], 3), 60, np.uint8))
    return np.concatenate(rows[:-1], axis=0)


def _log_step(step: int, rec: Dict, m, P: np.ndarray, intr: Dict,
              window, colors: Optional[np.ndarray] = None) -> None:
    import rerun as rr
    rr.set_time("step", sequence=int(step))
    C = (np.asarray(colors) if colors is not None and len(colors) == len(P)
         else np.full((len(P), 3), [170, 170, 190]))
    rr.log("world/cloud", rr.Points3D(P, colors=C, radii=[0.0012]))
    scene = rendermod.scene_of(m)
    for key, c in rec["cells"].items():
        To = np.asarray(c["orig"]["T_base_cam"])
        Tb = np.asarray(c["best"]["T_base_cam"])
        _log_cam(f"world/orig/{key}", To, intr, C_ORIG, mark_m=0.003)
        _log_cam(f"world/best/{key}", Tb, intr, C_BEST, mark_m=0.006)
    for v in range(len(V_ELEVATIONS)):
        sheet = _ring_sheet(scene, rec, v, intr, window)
        rr.log(f"img/ring{v}", rr.Image(sheet).compress(jpeg_quality=80))


def _blueprint(run: str):
    import rerun.blueprint as rrb
    tabs = [rrb.Spatial2DView(origin=f"img/ring{v}",
                              name=f"ring v{v} ({V_ELEVATIONS[v]:.0f} deg)")
            for v in range(len(V_ELEVATIONS))]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name=f"{run} regions"),
            rrb.Tabs(*tabs),
            column_shares=[5, 3]),
        collapse_panels=True)


# ---------------------------------------------------------------- CLI
def _tag(territory: bool, membrane: bool) -> str:
    return "regions" + ("-terr" if territory else "") + \
        ("-memb" if membrane else "")


def run(run: str, steps: str = "all", seed: int = 0,
        out: Optional[str] = None, open: bool = False,   # noqa: A002
        territory: bool = False, membrane: bool = False):
    """Refine all 36 cells per step; write JSON + rrd (+.rbl); maybe open."""
    import open3d as o3d
    import rerun as rr
    run = str(run)
    tag = _tag(territory, membrane)
    shell = shell_of(run)
    print(f"{run}: shell r={shell['r']:.3f} m  center="
          f"{np.round(shell['center'], 3).tolist()}")
    plys = sorted((DERIVED / run / "steps").glob("step_*.ply"))
    if steps == "final":
        plys = plys[-1:]
    elif steps != "all":
        toks = steps if isinstance(steps, (tuple, list)) else \
            str(steps).split(",")
        want = {int(t) for t in toks if str(t).strip()}
        plys = [p for p in plys if int(p.stem.split("_")[1]) in want]

    app_id = f"viewplan-{tag}-{run}"
    rr.init(app_id)
    intr = planner.session_intrinsics(run)
    window = (0.07, shell["r"] + 0.05)
    _log_grid(shell)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    out_dir = DERIVED / run / "viewplan" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in plys:
        step = int(p.stem.split("_")[1])
        pcd = o3d.io.read_point_cloud(str(p))
        P = np.asarray(pcd.points)
        C = (np.asarray(pcd.colors) * 255).astype(np.uint8) \
            if pcd.has_colors() else None
        rec, m = refine_step(P, run, shell, seed=seed,
                             territory=territory, membrane=membrane)
        rec["params"] = {"territory": territory, "membrane": membrane}
        (out_dir / f"step_{step:03d}.json").write_text(
            json.dumps(rec, indent=1))
        _log_step(step, rec, m, P, intr, window, colors=C)
        moved = [c for c in rec["cells"].values() if not c["empty"]]
        gains = [c["gain"] for c in moved if np.isfinite(c["gain"])]
        print(f"  step {step:3d}: {rec['n_hits']:4d} shell hits, "
              f"{len(moved):2d}/36 cells refined, median gain "
              f"x{np.median(gains):.2f}" if gains else
              f"  step {step:3d}: no refinable cells")

    dst = Path(out) if out else DERIVED / run / "rrd" / f"{run}-{tag}.rrd"
    dst.parent.mkdir(parents=True, exist_ok=True)
    bp = _blueprint(run)
    rr.save(str(dst), default_blueprint=bp)
    bp.save(app_id, str(dst.with_suffix(".rbl")))
    print(f"-> {dst}")
    if open:
        import subprocess
        import sys
        viewer = Path(sys.executable).parent / "rerun"
        subprocess.Popen([str(viewer if viewer.exists() else "rerun"),
                          str(dst), str(dst.with_suffix(".rbl"))],
                         start_new_session=True)
    return None


def matrix(run: str, steps: str = "all", seed: int = 0):
    """The 2x2 review matrix: off / terr / memb / terr+memb — four rrds."""
    for terr, memb in ((False, False), (True, False),
                       (False, True), (True, True)):
        print(f"=== {_tag(terr, memb)} ===")
        globals()["run"](run, steps=steps, seed=seed,
                         territory=terr, membrane=memb)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run, "matrix": matrix})
