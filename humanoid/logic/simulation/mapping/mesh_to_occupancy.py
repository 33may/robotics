"""mesh_to_occupancy — bake a 2D occupancy artifact from a dense reconstruction mesh.

GUI-free replacement for the Isaac omap GUI bake (and the rejected nvblox
occupancy line). Input is a fused TSDF mesh of the scene in the MAP frame (the
frame the localizer reports in — e.g. `fuse_reconstruction.py --mirror-fix`
output); output is the brain's `occupancy_io` artifact + preview.

Method (the height-column projection every ecosystem tool implements —
pointcloud_to_grid, octomap multi-layer projection, grid_map_pcl):

  occupied = >= `min_occ_samples` surface samples in the robot's height column
             [floor+band_lo, floor+band_hi]   (tall band: porous racks stamp)
  free     = floor surface observed AND column empty
  unknown  = neither  →  BLOCKED  (Anton 16-07: plan only through observed floor)

Core is pure numpy (brain-testable); open3d is imported lazily in the CLI only.

CLI (isaac env for the PLY loader):
    ~/miniconda3/envs/isaac/bin/python logic/simulation/mapping/mesh_to_occupancy.py \
        --mesh <bake>/cusfm/dense_mesh.ply --out <bake>/occupancy_mesh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from humanoid.logic.oli.reason.mapping.costmap import OccupancyGrid
from humanoid.logic.oli.reason.mapping.occupancy_io import save_occupancy

_MAX_SUBDIV = 16


# ── surface sampling ──────────────────────────────────────────────────────────


def sample_surface(vertices: np.ndarray, triangles: np.ndarray, spacing: float) -> np.ndarray:
    """Sample points on the mesh surface no farther than ~`spacing` apart.

    Deterministic midpoint subdivision: triangles with a max edge > `spacing`
    split into 4; small triangles emit their corners + centroid.
    """
    tris = np.asarray(vertices, float)[np.asarray(triangles, int)]  # (T, 3, 3)
    out = [np.asarray(vertices, float)]
    for _ in range(_MAX_SUBDIV):
        if not len(tris):
            break
        e = np.stack([
            np.linalg.norm(tris[:, 0] - tris[:, 1], axis=1),
            np.linalg.norm(tris[:, 1] - tris[:, 2], axis=1),
            np.linalg.norm(tris[:, 2] - tris[:, 0], axis=1),
        ], axis=1).max(axis=1)
        small = e <= spacing
        done = tris[small]
        if len(done):
            out.append(done.reshape(-1, 3))                # corners
            out.append(done.mean(axis=1))                  # centroids
        big = tris[~small]
        if not len(big):
            break
        a, b, c = big[:, 0], big[:, 1], big[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tris = np.concatenate([
            np.stack([a, ab, ca], axis=1),
            np.stack([ab, b, bc], axis=1),
            np.stack([ca, bc, c], axis=1),
            np.stack([ab, bc, ca], axis=1),
        ])
    return np.concatenate(out)


# ── floor detection ───────────────────────────────────────────────────────────


def detect_floor_z(points: np.ndarray, *, bin_size: float = 0.02,
                   dominance: float = 0.3) -> float:
    """The floor is the LOWEST strongly-populated z level (not the global max —
    a ceiling/rack-top plane of equal area must not win)."""
    z = np.asarray(points, float)[:, 2]
    lo, hi = z.min(), z.max()
    nbins = max(1, int(np.ceil((hi - lo) / bin_size)))
    counts, edges = np.histogram(z, bins=nbins, range=(lo, lo + nbins * bin_size))
    threshold = dominance * counts.max()
    idx = int(np.argmax(counts >= threshold))  # first (lowest) qualifying bin
    center = (edges[idx] + edges[idx + 1]) / 2
    near = z[np.abs(z - center) <= bin_size]
    return float(near.mean())


# ── morphology (pure numpy) ───────────────────────────────────────────────────


def _shift_or(mask: np.ndarray, radius: int, pad_value: bool) -> np.ndarray:
    padded = np.pad(mask, radius, constant_values=pad_value)
    H, W = mask.shape
    acc = np.zeros_like(mask) if not pad_value else np.ones_like(mask)
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            win = padded[dr:dr + H, dc:dc + W]
            acc = (acc | win) if not pad_value else (acc & win)
    return acc


def binary_close(occ: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological closing (dilate then erode), square kernel (2r+1)².

    Out-of-bounds is treated as occupied during erosion (consistent with the
    OccupancyGrid convention) so walls touching the border don't shrink.
    """
    if radius <= 0:
        return occ
    dilated = _shift_or(occ, radius, pad_value=False)   # OR = dilation
    return _shift_or(dilated, radius, pad_value=True)   # AND = erosion


# ── grid build ────────────────────────────────────────────────────────────────


def build_grid(
    points: np.ndarray,
    floor_z: float,
    *,
    resolution: float = 0.05,
    band: Tuple[float, float] = (0.3, 1.4),
    floor_tol: float = 0.10,
    min_occ_samples: int = 2,
    close_radius: int = 1,
    free_close: int = 2,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    traj_xy: Optional[np.ndarray] = None,
    traj_radius: float = 0.25,
) -> Tuple[OccupancyGrid, Dict]:
    """Project surface samples into the trinary grid, export unknown=BLOCKED.

    `bounds` is (x0, y0, x1, y1); default = the samples' xy extent.
    Returns (OccupancyGrid [True = blocked], stats dict).
    """
    pts = np.asarray(points, float)
    if bounds is None:
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    else:
        x0, y0, x1, y1 = bounds
    ncols = max(1, int(np.ceil((x1 - x0) / resolution)))
    nrows = max(1, int(np.ceil((y1 - y0) / resolution)))

    def counts(mask: np.ndarray) -> np.ndarray:
        p = pts[mask]
        cols = np.floor((p[:, 0] - x0) / resolution).astype(int)
        rows = np.floor((p[:, 1] - y0) / resolution).astype(int)
        ok = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
        grid = np.zeros((nrows, ncols), np.int32)
        np.add.at(grid, (rows[ok], cols[ok]), 1)
        return grid

    z_rel = pts[:, 2] - floor_z
    occ_count = counts((z_rel >= band[0]) & (z_rel <= band[1]))
    floor_count = counts(np.abs(z_rel) <= floor_tol)

    occupied = occ_count >= min_occ_samples
    if close_radius > 0:
        occupied = binary_close(occupied, close_radius)
    free = (floor_count > 0) & ~occupied
    if free_close > 0:
        # seal thin unknown stripes inside observed floor (TSDF sampling gaps
        # at range) — they read as phantom walls to the planner. Obstacles
        # always win: closing free never overrides occupied.
        free = binary_close(free, free_close) & ~occupied

    # trajectory clearing: cells the robot physically occupied are free by the
    # strongest evidence there is (ground contact). The estimated (GT-free)
    # trajectory covers the start/end blind spot the cameras never see.
    traj_cleared = traj_overrode = 0
    if traj_xy is not None and len(traj_xy):
        rr = max(0, int(np.ceil(traj_radius / resolution)))
        stamp = np.zeros_like(occupied)
        cols = np.floor((np.asarray(traj_xy)[:, 0] - x0) / resolution).astype(int)
        rows = np.floor((np.asarray(traj_xy)[:, 1] - y0) / resolution).astype(int)
        ok = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
        stamp[rows[ok], cols[ok]] = True
        stamp = _shift_or(stamp, rr, pad_value=False) if rr else stamp
        traj_overrode = int((stamp & occupied).sum())
        traj_cleared = int((stamp & ~free).sum())
        occupied &= ~stamp
        free |= stamp
    unknown = ~occupied & ~free

    stats = {
        "floor_z": float(floor_z),
        "occupied_cells": int(occupied.sum()),
        "free_cells": int(free.sum()),
        "unknown_cells": int(unknown.sum()),
        "shape": [nrows, ncols],
        "resolution": resolution,
        "origin": [float(x0), float(y0)],
        "band": list(band),
        "traj_cleared_cells": traj_cleared,
        "traj_overrode_occupied_cells": traj_overrode,
        "observed_mask": occupied | free,  # array; CLI pops + saves as .npy
    }
    blocked = occupied | unknown
    return OccupancyGrid(blocked, resolution, (float(x0), float(y0))), stats


# ── CLI (isaac env: open3d loader) ────────────────────────────────────────────


def build_from_mesh(mesh_path: Path, out_dir: Path, **kw) -> Dict:
    import open3d as o3d  # lazy: CLI-only

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    resolution = kw.get("resolution", 0.05)
    pts = sample_surface(verts, tris, spacing=resolution / 2)
    floor_z = kw.pop("floor_z", None)
    if floor_z is None:
        floor_z = detect_floor_z(pts)
    grid, stats = build_grid(pts, floor_z, **kw)
    save_occupancy(grid, str(out_dir))
    observed = stats.pop("observed_mask")
    np.save(out_dir / "observed.npy", observed)
    _write_preview(grid, out_dir / "preview.png")
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def _write_preview(grid: OccupancyGrid, out_png: Path) -> None:
    from PIL import Image  # lazy

    img = np.where(grid.grid, 0, 255).astype(np.uint8)
    Image.fromarray(np.flipud(img), "L").save(out_png)  # north-up


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--band", type=float, nargs=2, default=(0.3, 1.4))
    ap.add_argument("--floor-tol", type=float, default=0.10)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--close", type=int, default=1)
    ap.add_argument("--free-close", type=int, default=2,
                    help="closing radius on FREE (seals unknown stripes in observed floor)")
    ap.add_argument("--traj", type=Path, default=None,
                    help="estimated trajectory TUM (map frame): stamp robot footprint free")
    ap.add_argument("--traj-radius", type=float, default=0.25)
    ap.add_argument("--floor-z", type=float, default=None,
                    help="override floor detection")
    a = ap.parse_args(argv)
    traj_xy = None
    if a.traj is not None:
        rows = [line.split() for line in a.traj.read_text().splitlines() if line.strip()]
        traj_xy = np.array([[float(v[1]), float(v[2])] for v in rows])
    stats = build_from_mesh(
        a.mesh, a.out, resolution=a.res, band=tuple(a.band), floor_tol=a.floor_tol,
        min_occ_samples=a.min_samples, close_radius=a.close, free_close=a.free_close,
        floor_z=a.floor_z, traj_xy=traj_xy, traj_radius=a.traj_radius)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
