"""occupancy_from_depth.py — robot-derived 2D occupancy grid (MAY-173 Phase 2.2).

The no-privileged-info replacement for the Isaac omap GUI export: everything the
grid is built from was either SENSED by the robot (uint16-mm depth PNGs from the
teleop dump) or ESTIMATED by the localization stack (the bake's cuVSLAM TUM
trajectory, MAP frame). GT rows / `T_world_cam` in `poses.jsonl` are never read.

Per joined stamp:  T_map_base (TUM) ∘ T_base_cam (static rig mount, USD camera
convention — the same construction as `recording/fk.T_base_cam_usd`) → backproject
depth → classify by height column:

  - z ∈ [zmin, zmax]  → obstacle HIT: endpoint cell + free-carve the 2D ray to it
  - z <  zmin         → FLOOR sighting: free-carve the ray INCLUDING the endpoint
  - z >  zmax         → ignored (overhead structure must not block ground cells)

Cells: hit-dominated → occupied; carved-only → free; NEVER OBSERVED → **blocked**
(Anton 16-07: the robot only plans through floor it has actually seen; revisit
with the deploy-time self-improvement loop). Export = `occupancy_io` artifact
(`occupancy.npy` + `occupancy.json`) — `StaticMapping`/MapPanel load it as-is.

Pure numpy + PIL + stdlib (brain-marker clean). Offline tooling:

    p logic/simulation/mapping/occupancy_from_depth.py \
        --dump data/coverage_drives/teleop_v1_demo \
        --tum  data/maps/teleop_v1_demo/latest/poses/slam_poses.tum \
        --out  data/maps/teleop_v1_demo/latest/occupancy
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from humanoid.logic.oli.reason.mapping.costmap import OccupancyGrid
from humanoid.logic.oli.reason.mapping.occupancy_io import save_occupancy

#: TUM↔frame-stamp join precision (seconds-decimals) — same slack as map_audit
_JOIN_DECIMALS = 4


# ── inputs ────────────────────────────────────────────────────────────────────


def load_tum_poses(path: Path | str) -> List[Tuple[float, np.ndarray]]:
    """TUM rows (`t x y z qx qy qz qw`) → [(stamp_s, T_map_base 4×4)]."""
    rows = np.loadtxt(Path(path))
    if rows.ndim == 1:
        rows = rows[None, :]
    out = []
    for t, x, y, z, qx, qy, qz, qw in rows:
        T = np.eye(4)
        T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
        T[:3, 3] = (x, y, z)
        out.append((float(t), T))
    return out


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def detect_heading_offset(poses: List[Tuple[float, np.ndarray]], *,
                          min_speed: float = 0.4, window: int = 10,
                          hz: float = 30.0, min_samples: int = 30) -> float:
    """GT-free convention calibration: the constant yaw offset between the TUM
    rotation's +X axis and the direction of travel (circular median over
    forward-driving segments). The teleop bake ships base_link yaw-FLIPPED
    (rotation x points backward) — invisible to position-only audits, fatal
    when composing with mount extrinsics. Returns 0.0 if the drive has too
    little motion to calibrate (trust the rig as-is)."""
    if len(poses) <= window:
        return 0.0
    P = np.array([T[:3, 3] for _, T in poses])
    d = P[window:] - P[:-window]
    speed = np.linalg.norm(d[:, :2], axis=1) * (hz / window)
    offs = []
    for i in np.where(speed > min_speed)[0]:
        heading = math.atan2(d[i, 1], d[i, 0])
        v = poses[i + window // 2][1][:3, :3] @ [1.0, 0.0, 0.0]
        yaw_r = math.atan2(v[1], v[0])
        offs.append(yaw_r - heading)
    if len(offs) < min_samples:
        return 0.0
    # circular mean — immune to ±π wrapping
    return math.atan2(np.mean(np.sin(offs)), np.mean(np.cos(offs)))


def rz(angle: float) -> np.ndarray:
    """4×4 rotation about z."""
    c, s = math.cos(angle), math.sin(angle)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def planarize(T: np.ndarray) -> np.ndarray:
    """Project a 4×4 pose onto the ground plane: keep (x, y, yaw), zero z and
    pitch/roll. The glide robot is planar by CONSTRUCTION (kinematic prior, not
    GT) — SLAM pitch/z drift otherwise tilts the reconstructed floor into the
    obstacle z-column and paints radial streak fans across the aisles."""
    yaw = math.atan2(T[1, 0], T[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    P = np.eye(4)
    P[0, 0], P[0, 1], P[1, 0], P[1, 1] = c, -s, s, c
    P[0, 3], P[1, 3] = T[0, 3], T[1, 3]
    return P


def t_base_cam(pos_base: Sequence[float], pitch_down_deg: float) -> np.ndarray:
    """Static base→camera transform from rig.json numbers — the dict-input twin of
    `recording/fk.T_base_cam_usd` (view +X tilted down, camera looks down -Z)."""
    th = math.radians(pitch_down_deg)
    view = np.array([math.cos(th), 0.0, -math.sin(th)])
    right = np.cross(view, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, view)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = right, up, -view
    T[:3, 3] = np.asarray(pos_base, dtype=float)
    return T


def backproject_depth(depth_mm: np.ndarray, K: Dict, stride: int) -> np.ndarray:
    """uint16-mm depth image → (N, 3) camera-frame points, USD convention
    (+X right, +Y up, view down -Z; depth = distance to image plane).
    Zero-depth (no return) pixels are dropped. `stride` subsamples both axes."""
    d = depth_mm[::stride, ::stride].astype(np.float64) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    valid = d > 0
    d, u, v = d[valid], u[valid], v[valid]
    x = (u - K["cx"]) / K["fx"] * d
    y = -(v - K["cy"]) / K["fy"] * d
    return np.column_stack([x, y, -d])


# ── the accumulator ───────────────────────────────────────────────────────────


class GridBuilder:
    """3-state occupancy accumulator over a fixed metric window.

    `hits` counts obstacle endpoints per cell; `free` counts ray pass-throughs.
    Export rule (unknown = blocked): a cell is FREE only if it was carved and is
    not hit-dominated — everything else (hits ≥ min_hits, or never observed) is
    True/blocked in the exported bool grid."""

    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float,
                 *, zmin: float = 0.15, zmax: float = 1.6, max_range: float = 6.0,
                 min_hits: int = 2) -> None:
        self.xmin, self.ymin, self.xmax, self.ymax = map(float, bounds)
        self.res = float(resolution)
        self.zmin, self.zmax = float(zmin), float(zmax)
        self.max_range = float(max_range)
        self.min_hits = int(min_hits)
        ncols = max(1, int(math.ceil((self.xmax - self.xmin) / self.res)))
        nrows = max(1, int(math.ceil((self.ymax - self.ymin) / self.res)))
        self.hits = np.zeros((nrows, ncols), dtype=np.int32)
        self.free = np.zeros((nrows, ncols), dtype=np.int32)

    # cell helpers (float arrays in, int arrays out; no bounds clipping here)
    def _cells(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        col = np.floor((xy[:, 0] - self.xmin) / self.res).astype(np.int64)
        row = np.floor((xy[:, 1] - self.ymin) / self.res).astype(np.int64)
        return row, col

    def _accumulate(self, counter: np.ndarray, xy: np.ndarray) -> None:
        if len(xy) == 0:
            return
        row, col = self._cells(xy)
        ok = (row >= 0) & (row < counter.shape[0]) & (col >= 0) & (col < counter.shape[1])
        np.add.at(counter, (row[ok], col[ok]), 1)

    def add_view(self, T_map_cam: np.ndarray, depth_mm: np.ndarray, K: Dict,
                 *, stride: int = 8) -> None:
        pts_cam = backproject_depth(depth_mm, K, stride)
        if len(pts_cam) == 0:
            return
        pts = (T_map_cam[:3, :3] @ pts_cam.T).T + T_map_cam[:3, 3]
        cam_xy = T_map_cam[:2, 3]
        delta = pts[:, :2] - cam_xy
        rng = np.linalg.norm(delta, axis=1)
        near = rng <= self.max_range
        carveable = near & (rng > 2 * self.res)  # rays too short to walk are skipped

        obstacle = carveable & (pts[:, 2] >= self.zmin) & (pts[:, 2] <= self.zmax)
        floor = near & (pts[:, 2] < self.zmin) & (pts[:, 2] > -0.5)

        self._accumulate(self.hits, pts[obstacle, :2])
        # floor ENDPOINTS are free evidence by themselves (covers straight-down
        # sightings whose 2D ray is too short to carve) …
        self._accumulate(self.free, pts[floor, :2])
        # … then carve pass-through free space: obstacle rays stop one cell SHORT
        # of the hit; floor rays run through (endpoint already credited above).
        self._carve(cam_xy, delta[obstacle], rng[obstacle], stop_short=self.res)
        fc = floor & carveable
        self._carve(cam_xy, delta[fc], rng[fc], stop_short=self.res / 2)

    def _carve(self, cam_xy: np.ndarray, delta: np.ndarray, rng: np.ndarray,
               *, stop_short: float) -> None:
        """Mark free samples along each 2D ray at half-cell spacing (vectorized
        ragged expansion: one flat sample array across all rays)."""
        if len(rng) == 0:
            return
        step = self.res / 2.0
        lengths = np.maximum(rng - stop_short, 0.0)
        counts = np.maximum((lengths / step).astype(np.int64), 0)
        total = int(counts.sum())
        if total == 0:
            return
        ray_idx = np.repeat(np.arange(len(rng)), counts)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        s = (np.arange(total) - offsets[ray_idx] + 1) * step
        dirs = delta / rng[:, None]
        xy = cam_xy + dirs[ray_idx] * s[:, None]
        self._accumulate(self.free, xy)

    def grid(self) -> OccupancyGrid:
        free_known = (self.free > 0) & (self.hits < self.min_hits)
        return OccupancyGrid(~free_known, self.res, (self.xmin, self.ymin))


# ── the flow: dump + TUM → artifact ──────────────────────────────────────────


def build_occupancy(dump_dir: Path | str, tum_path: Path | str, out_dir: Path | str,
                    *, streams: Sequence[str] = ("head",), resolution: float = 0.05,
                    stride: int = 8, frame_step: int = 5, zmin: float = 0.15,
                    zmax: float = 1.6, max_range: float = 6.0,
                    min_hits: int = 2, planar: bool = True) -> OccupancyGrid:
    """Build + save the occupancy artifact. Reads ONLY `rig.json`, the depth PNG
    stream(s) and the bake TUM — never `poses.jsonl` (GT stays out of the loop).
    `frame_step` subsamples stamps (mapping does not need every 30 Hz frame)."""
    from PIL import Image  # lazy: keep module import light

    dump_dir, out_dir = Path(dump_dir), Path(out_dir)
    rig = json.loads((dump_dir / "rig.json").read_text())
    poses = load_tum_poses(tum_path)

    # metric window: the driven trajectory ± sensor reach
    traj = np.array([T[:2, 3] for _, T in poses])
    margin = max_range + 1.0
    bounds = (traj[:, 0].min() - margin, traj[:, 1].min() - margin,
              traj[:, 0].max() + margin, traj[:, 1].max() + margin)
    builder = GridBuilder(bounds, resolution, zmin=zmin, zmax=zmax,
                          max_range=max_range, min_hits=min_hits)

    # convention auto-calibration: rotate every pose so its +X = travel heading
    offset = detect_heading_offset(poses)
    Rfix = rz(-offset)
    if abs(offset) > 0.05:
        print(f"heading-offset calibration: rotating poses by {-math.degrees(offset):+.1f}°")

    pose_by_key = {round(t, _JOIN_DECIMALS): T @ Rfix for t, T in poses}
    used = 0
    for name in streams:
        cam = rig["cameras"][name]
        T_bc = t_base_cam(cam["pos_base"], cam["pitch_down_deg"])
        K = cam["intrinsics"]
        files = sorted((dump_dir / "frames" / f"{name}_depth").glob("*.png"))
        for f in files[::frame_step]:
            key = round(int(f.stem) / 1e9, _JOIN_DECIMALS)
            T_mb = pose_by_key.get(key)
            if T_mb is None:
                continue
            if planar:
                T_mb = planarize(T_mb)
            depth = np.asarray(Image.open(f))
            builder.add_view(T_mb @ T_bc, depth, K, stride=stride)
            used += 1

    grid = builder.grid()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_occupancy(grid, str(out_dir))
    _save_preview(grid, builder, traj, out_dir / "preview.png")
    print(f"occupancy: {grid.nrows}x{grid.ncols} cells @ {resolution} m "
          f"({used} views) -> {out_dir}")
    return grid


def _save_preview(grid: OccupancyGrid, builder: GridBuilder, traj: np.ndarray,
                  path: Path) -> None:
    """Quick-look PNG: white=free, black=hit-occupied, grey=unknown, red=path."""
    from PIL import Image  # lazy

    img = np.full((*grid.grid.shape, 3), 128, dtype=np.uint8)   # unknown grey
    img[~grid.grid] = 255                                       # free white
    img[builder.hits >= builder.min_hits] = 0                   # obstacles black
    col = np.clip(((traj[:, 0] - builder.xmin) / builder.res).astype(int), 0, grid.ncols - 1)
    row = np.clip(((traj[:, 1] - builder.ymin) / builder.res).astype(int), 0, grid.nrows - 1)
    img[row, col] = (220, 40, 40)
    Image.fromarray(img[::-1]).save(path)  # +y up in the image


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Depth + SLAM poses -> occupancy artifact.")
    ap.add_argument("--dump", required=True, help="teleop dump dir (rig.json + frames/)")
    ap.add_argument("--tum", required=True, help="bake TUM trajectory (map frame)")
    ap.add_argument("--out", required=True, help="artifact dir (occupancy.npy/json)")
    ap.add_argument("--streams", default="head", help="comma list of depth streams")
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=8, help="pixel subsample")
    ap.add_argument("--frame-step", type=int, default=5, help="stamp subsample")
    ap.add_argument("--zmin", type=float, default=0.15)
    ap.add_argument("--zmax", type=float, default=1.6)
    ap.add_argument("--max-range", type=float, default=6.0)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--no-planar", action="store_true",
                    help="keep full SE(3) SLAM poses (default: project to x,y,yaw)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    build_occupancy(args.dump, args.tum, args.out,
                    streams=tuple(s.strip() for s in args.streams.split(",") if s.strip()),
                    resolution=args.resolution, stride=args.stride,
                    frame_step=args.frame_step, zmin=args.zmin, zmax=args.zmax,
                    max_range=args.max_range, min_hits=args.min_hits,
                    planar=not args.no_planar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
