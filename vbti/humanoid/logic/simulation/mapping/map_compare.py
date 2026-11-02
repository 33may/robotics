"""map_compare — register a candidate occupancy grid to the GT map and score it.

Eval-oracle tooling (never in the deployment path): the candidate lives in the
bake MAP frame, which is REFLECTED vs world (`registration.json` in every
bake), so the trajectory fit allows det(R) = -1. Pipeline:

  1. `fit_traj_transform`     — 2D Umeyama (reflection allowed) from trajectory
                                 correspondences (map est xy ↔ GT xy) → seed
  2. `refine_grid_alignment`  — local grid-to-grid search (dx, dy, dθ) around
                                 the seed, maximizing occupied-cell agreement
  3. `score_grids`            — false-free% (smear ≤0.3 m / deep split),
                                 false-blocked% on observed cells, IoU
  4. `boundary_error`         — occupied-boundary distance distribution (m):
                                 the map's metric precision for navigation

All pure numpy. CLI at the bottom compares a baked artifact against the GT
nav map and writes an overlay PNG + report JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from humanoid.logic.oli.reason.mapping.costmap import OccupancyGrid
from humanoid.logic.oli.reason.mapping.occupancy_io import load_occupancy
from humanoid.logic.simulation.mapping.mesh_to_occupancy import _shift_or


# ── 2D Umeyama with reflection ────────────────────────────────────────────────


def fit_traj_transform(src: np.ndarray, dst: np.ndarray) -> Dict:
    """Least-squares 2D transform dst ≈ src @ R.T + t, det(R) = ±1 (whichever
    fits better — the bake map frame may be a reflection of world)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)

    best = None
    for d in (1.0, -1.0):
        R = Vt.T @ np.diag([1.0, d]) @ U.T
        t = cd - R @ cs
        res = float(np.linalg.norm(src @ R.T + t - dst, axis=1).mean())
        if best is None or res < best["residual_mean"]:
            best = {"R": R, "t": t, "mirrored": bool(np.linalg.det(R) < 0),
                    "residual_mean": res}
    assert best is not None
    return best


def apply_transform2d(pts: np.ndarray, T: Dict) -> np.ndarray:
    return np.asarray(pts, float) @ np.asarray(T["R"]).T + np.asarray(T["t"])


# ── grid helpers ──────────────────────────────────────────────────────────────


def _occupied_world(grid: OccupancyGrid) -> np.ndarray:
    rows, cols = np.nonzero(grid.grid)
    x = grid.origin[0] + (cols + 0.5) * grid.resolution
    y = grid.origin[1] + (rows + 0.5) * grid.resolution
    return np.stack([x, y], axis=1)


def _cells_world(grid: OccupancyGrid, mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    x = grid.origin[0] + (cols + 0.5) * grid.resolution
    y = grid.origin[1] + (rows + 0.5) * grid.resolution
    return np.stack([x, y], axis=1)


def _lookup(grid: OccupancyGrid, pts: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(hit, in_bounds) bool arrays: whether each world pt lands on a True cell
    of `mask` (same shape as grid)."""
    cols = np.floor((pts[:, 0] - grid.origin[0]) / grid.resolution).astype(int)
    rows = np.floor((pts[:, 1] - grid.origin[1]) / grid.resolution).astype(int)
    inb = (rows >= 0) & (rows < grid.nrows) & (cols >= 0) & (cols < grid.ncols)
    hit = np.zeros(len(pts), bool)
    hit[inb] = mask[rows[inb], cols[inb]]
    return hit, inb


def _rz(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]])


# ── grid-to-grid refinement ───────────────────────────────────────────────────


def refine_grid_alignment(
    cand: OccupancyGrid,
    gt: OccupancyGrid,
    seed: Dict,
    *,
    search_t: float = 0.3,
    step_t: float = 0.025,
    search_deg: float = 2.0,
    step_deg: float = 0.25,
    observed_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Exhaustive local search around `seed` maximizing the fraction of
    candidate-occupied cells landing on GT-occupied cells. Rotation is applied
    about the candidate centroid (decouples θ from translation).

    The artifact merges unknown→blocked; pass `observed_mask` so only REAL
    obstacles (blocked ∩ observed) drive the alignment — never coverage edges.
    """
    occ = cand.grid & observed_mask if observed_mask is not None else cand.grid
    pts = apply_transform2d(_cells_world(cand, occ), seed)
    centroid = pts.mean(axis=0)
    dts = np.arange(-search_t, search_t + 1e-9, step_t)
    dgs = np.arange(-search_deg, search_deg + 1e-9, step_deg)

    best = None
    for dg in dgs:
        rot = _rz(dg)
        p_rot = (pts - centroid) @ rot.T + centroid
        for dx in dts:
            for dy in dts:
                hit, _ = _lookup(gt, p_rot + [dx, dy], gt.grid)
                score = float(hit.mean())
                if best is None or score > best["score"]:
                    best = {"dx": float(dx), "dy": float(dy),
                            "dtheta_deg": float(dg), "score": score}
    assert best is not None
    rot = _rz(best["dtheta_deg"])
    R_seed, t_seed = np.asarray(seed["R"], float), np.asarray(seed["t"], float)
    R_tot = rot @ R_seed
    t_tot = rot @ (t_seed - centroid) + centroid + [best["dx"], best["dy"]]
    return {**best, "R": R_tot, "t": t_tot, "mirrored": bool(seed.get("mirrored", False))}


# ── scoring ───────────────────────────────────────────────────────────────────


def score_grids(
    cand: OccupancyGrid,
    gt: OccupancyGrid,
    T: Dict,
    *,
    observed_mask: Optional[np.ndarray] = None,
    smear_m: float = 0.3,
) -> Dict:
    """Navigation-safety metrics after mapping candidate cells into GT world.

    false_free: candidate says FREE where GT is occupied (collision risk),
    split into smear (≤ `smear_m` of GT free space — boundary discretization)
    and deep (real intrusions). false_blocked: candidate blocks GT-free cells,
    scored only on `observed_mask` cells (default: all).
    """
    free_mask = ~cand.grid
    obs = observed_mask if observed_mask is not None else np.ones_like(cand.grid)

    free_pts = apply_transform2d(_cells_world(cand, free_mask), T)
    on_gt_occ, inb = _lookup(gt, free_pts, gt.grid)
    false_free = on_gt_occ & inb
    n_free = int(inb.sum())

    # smear split: within smear_m of GT free space?
    r = max(1, int(round(smear_m / gt.resolution)))
    gt_free_dilated = _shift_or(~gt.grid, r, pad_value=False)
    near_free, _ = _lookup(gt, free_pts, gt_free_dilated)
    smear = false_free & near_free
    deep = false_free & ~near_free

    occ_mask = cand.grid & obs
    blocked_obs_pts = apply_transform2d(_cells_world(cand, occ_mask), T)
    on_gt_free, inb_b = _lookup(gt, blocked_obs_pts, ~gt.grid)
    false_blocked = on_gt_free & inb_b
    n_blocked_obs = int(inb_b.sum())

    occ_pts = apply_transform2d(_cells_world(cand, occ_mask), T)
    occ_hit, occ_inb = _lookup(gt, occ_pts, gt.grid)

    # frontier split: occupied cells touching unknown are coverage-edge
    # artifacts (TSDF range fringes) — blocked either way for the planner,
    # but they must not pollute the interior quality numbers.
    unknown_mask = cand.grid & ~obs
    frontier = occ_mask & _shift_or(unknown_mask, 2, pad_value=False)
    interior = occ_mask & ~frontier
    int_pts = apply_transform2d(_cells_world(cand, interior), T)
    int_hit, int_inb = _lookup(gt, int_pts, gt.grid)

    pct = lambda a, b: float(100.0 * a / b) if b else 0.0
    return {
        "false_free_pct": pct(int(false_free.sum()), n_free),
        "false_free_smear_pct": pct(int(smear.sum()), n_free),
        "false_free_deep_pct": pct(int(deep.sum()), n_free),
        "false_free_deep_cells": int(deep.sum()),
        "false_blocked_observed_pct": pct(int(false_blocked.sum()), n_blocked_obs),
        "occupied_precision_pct": pct(int(occ_hit.sum()), int(occ_inb.sum())),
        "occupied_precision_interior_pct": pct(int(int_hit.sum()), int(int_inb.sum())),
        "frontier_occ_cells": int(frontier.sum()),
        "interior_occ_cells": int(interior.sum()),
        "n_free_cells": n_free,
        "out_of_gt_bounds": int((~inb).sum()),
    }


def boundary_error(cand: OccupancyGrid, gt: OccupancyGrid, T: Dict,
                   *, max_rings: int = 12,
                   observed_mask: Optional[np.ndarray] = None) -> Dict:
    """Distance from each candidate-occupied cell (in world) to the nearest
    GT-occupied cell — the metric precision of the map's obstacle boundaries.
    Ring-dilation quantized to GT resolution; distances beyond
    `max_rings * res` are reported as a fraction, not a distance."""
    occ = cand.grid & observed_mask if observed_mask is not None else cand.grid
    pts = apply_transform2d(_cells_world(cand, occ), T)
    dist = np.full(len(pts), np.inf)
    mask = gt.grid.copy()
    for k in range(max_rings + 1):
        hit, _ = _lookup(gt, pts, mask)
        newly = hit & ~np.isfinite(dist)
        dist[newly] = k * gt.resolution
        if k < max_rings:
            mask = _shift_or(mask, 1, pad_value=False)
    finite = dist[np.isfinite(dist)]
    cap = max_rings * gt.resolution
    return {
        "p50_m": float(np.percentile(finite, 50)) if len(finite) else float("nan"),
        "p95_m": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
        "beyond_cap_frac": float(np.mean(~np.isfinite(dist))),
        "cap_m": cap,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def _load_traj_correspondences(tum_path: Path, dump_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """(map_xy, world_xy) matched by stamp: estimated trajectory vs GT base rows."""
    est = {}
    for line in Path(tum_path).read_text().splitlines():
        v = line.split()
        if len(v) >= 4:
            est[round(float(v[0]), 4)] = (float(v[1]), float(v[2]))
    gt = {}
    with open(dump_dir / "poses.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["cam"] == "base":
                gt[round(d["stamp_ns"] / 1e9, 4)] = (d["x"], d["y"])
    common = sorted(set(est) & set(gt))
    return (np.array([est[t] for t in common]), np.array([gt[t] for t in common]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", required=True, help="candidate occupancy artifact dir")
    ap.add_argument("--gt", required=True, help="GT occupancy artifact dir")
    ap.add_argument("--tum", required=True, type=Path,
                    help="estimated trajectory (map frame) for the seed fit")
    ap.add_argument("--dump", required=True, type=Path, help="dump dir with GT poses")
    ap.add_argument("--observed", default=None,
                    help="observed-mask .npy aligned with the candidate grid")
    ap.add_argument("--out", default=None, help="report JSON path")
    ap.add_argument("--overlay", default=None, help="overlay PNG path")
    a = ap.parse_args(argv)

    cand, gt = load_occupancy(a.cand), load_occupancy(a.gt)
    observed = np.load(a.observed) if a.observed else None
    map_xy, world_xy = _load_traj_correspondences(a.tum, a.dump)
    seed = fit_traj_transform(map_xy, world_xy)
    print(f"seed: mirrored={seed['mirrored']} residual={seed['residual_mean']:.3f} m "
          f"({len(map_xy)} correspondences)")
    T = refine_grid_alignment(cand, gt, seed, observed_mask=observed)
    print(f"refine: d=({T['dx']:.3f},{T['dy']:.3f}) m, {T['dtheta_deg']:.2f}°, "
          f"occupied-agreement={T['score']:.3f}")

    report = {
        "seed_residual_m": seed["residual_mean"],
        "refine": {k: T[k] for k in ("dx", "dy", "dtheta_deg", "score")},
        "mirrored": seed["mirrored"],
        **score_grids(cand, gt, T, observed_mask=observed),
        "boundary": boundary_error(cand, gt, T, observed_mask=observed),
    }
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    if a.overlay:
        _write_overlay(cand, gt, T, Path(a.overlay), observed_mask=observed)
        print("overlay ->", a.overlay)
    return 0


def _write_overlay(cand: OccupancyGrid, gt: OccupancyGrid, T: Dict, out_png: Path,
                   observed_mask: Optional[np.ndarray] = None) -> None:
    from PIL import Image  # lazy

    occ = cand.grid & observed_mask if observed_mask is not None else cand.grid
    img = np.full((gt.nrows, gt.ncols, 3), 255, np.uint8)
    img[gt.grid] = (220, 60, 60)                       # GT occupied: red
    occ_pts = apply_transform2d(_cells_world(cand, occ), T)
    cols = np.floor((occ_pts[:, 0] - gt.origin[0]) / gt.resolution).astype(int)
    rows = np.floor((occ_pts[:, 1] - gt.origin[1]) / gt.resolution).astype(int)
    inb = (rows >= 0) & (rows < gt.nrows) & (cols >= 0) & (cols < gt.ncols)
    on_gt = gt.grid[rows[inb], cols[inb]]
    r_in, c_in = rows[inb], cols[inb]
    img[r_in[~on_gt], c_in[~on_gt]] = (60, 60, 220)    # candidate-only: blue
    img[r_in[on_gt], c_in[on_gt]] = (120, 40, 140)     # agreement: purple
    Image.fromarray(np.flipud(img), "RGB").save(out_png)


if __name__ == "__main__":
    raise SystemExit(main())
