"""layout_gen.py — occupancy grid → dressing placement list (MAY-173 dressing).

Extracts the building's perimeter walls from the nav occupancy grid (racks are
interior occupied islands and get nothing), walks each inward-facing wall run,
and emits plate/poster placements as a layout.yaml. Positions are the FINAL
quad centers — 2 cm off the wall along the inward normal — so the USD builder
does no spatial reasoning of its own.

Pure python (numpy + yaml + stdlib). Runs in the `hum` env.

    python -m humanoid.logic.simulation.mapping.dressing.layout_gen \
        --map assets/envs/warehouse_nvidia/nav_maps/v1 \
        --out assets/envs/warehouse_nvidia/dressing/layout.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

PLATE_SPACING_M = 1.75          # every 1.5–2 m (design: FINAL)
PLATE_HEIGHTS_M = (1.2, 2.5)    # alternating mount heights
PLATE_SIZES_M = (0.8, 0.6, 0.7, 0.5)  # cycled; big first → range for the far poses
PLATE_OFF_WALL_M = 0.02         # z-fighting offset
PLATE_END_MARGIN_M = 0.5        # keep off corners
POSTER_HEIGHT_M = 3.5           # above the high plate row — banner height
POSTER_SIZE_M = 2.0             # width; quad height = width * 3/4
POSTER_MIN_RUN_M = 6.0          # only on genuinely long blank stretches
POSTER_MAX = 6


@dataclass(frozen=True)
class WallSegment:
    """A contiguous inward-facing wall run.

    plane: world coord of the wall face along the normal axis.
    start/end: along-wall world coords (face-cell centers).
    normal: inward unit normal (into free space).
    """

    plane: float
    start: float
    end: float
    normal: tuple[float, float]

    @property
    def length(self) -> float:
        return self.end - self.start


def extract_perimeter(occ: np.ndarray) -> np.ndarray:
    """Occupied cells 4-connected to the grid border = the building wall.
    Interior occupied islands (racks) are excluded by construction."""
    h, w = occ.shape
    perim = np.zeros_like(occ)
    dq: deque = deque()

    def seed(r: int, c: int) -> None:
        if occ[r, c] and not perim[r, c]:
            perim[r, c] = True
            dq.append((r, c))

    for c in range(w):
        seed(0, c)
        seed(h - 1, c)
    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and occ[nr, nc] and not perim[nr, nc]:
                perim[nr, nc] = True
                dq.append((nr, nc))
    return perim


# Inward face directions: (dr, dc) to the free neighbor, (nx, ny) world normal.
_FACES = ((0, 1, 1.0, 0.0), (0, -1, -1.0, 0.0), (1, 0, 0.0, 1.0), (-1, 0, 0.0, -1.0))


def wall_segments(occ: np.ndarray, resolution: float, origin: tuple[float, float],
                  min_length: float = 1.0, max_gap_cells: int = 3) -> list[WallSegment]:
    """Contiguous inward-facing runs of perimeter wall, as world-coord segments."""
    h, w = occ.shape
    ox, oy = origin
    perim = extract_perimeter(occ)
    # faces[(normal, plane)] -> list of along-wall coords (face-cell centers)
    runs: dict[tuple[tuple[float, float], float], list[float]] = {}
    for r, c in zip(*np.where(perim)):
        for dr, dc, nx, ny in _FACES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or occ[nr, nc]:
                continue
            if nx != 0.0:  # wall face is a vertical plane x = const
                plane = float(ox + (c + (1.0 if nx > 0 else 0.0)) * resolution)
                along = float(oy + (r + 0.5) * resolution)
            else:          # horizontal plane y = const
                plane = float(oy + (r + (1.0 if ny > 0 else 0.0)) * resolution)
                along = float(ox + (c + 0.5) * resolution)
            runs.setdefault(((nx, ny), round(plane, 4)), []).append(along)

    segs: list[WallSegment] = []
    for (normal, plane), alongs in runs.items():
        alongs.sort()
        start = prev = alongs[0]
        for a in alongs[1:] + [None]:  # type: ignore[list-item]
            if a is not None and a - prev <= max_gap_cells * resolution:
                prev = a
                continue
            if prev - start >= min_length:
                segs.append(WallSegment(plane=plane, start=start, end=prev, normal=normal))
            if a is not None:
                start = prev = a
    return segs


def _placement_pos(seg: WallSegment, along: float, z: float) -> list[float]:
    nx, ny = seg.normal
    if nx != 0.0:
        x, y = seg.plane + nx * PLATE_OFF_WALL_M, along
    else:
        x, y = along, seg.plane + ny * PLATE_OFF_WALL_M
    return [round(x, 4), round(y, 4), z]


def build_layout(occ: np.ndarray, resolution: float,
                 origin: tuple[float, float]) -> dict:
    """Full dressing layout: numbered plates along every wall run + posters on
    the longest runs. Deterministic (segment order is sorted)."""
    segs = sorted(wall_segments(occ, resolution, origin),
                  key=lambda s: (s.normal, s.plane, s.start))
    plates = []
    number = 0
    for seg in segs:
        t = seg.start + PLATE_END_MARGIN_M
        i = 0
        while t <= seg.end - PLATE_END_MARGIN_M:
            plates.append({
                "number": number,
                "pos": _placement_pos(seg, t, PLATE_HEIGHTS_M[i % 2]),
                "normal": [seg.normal[0], seg.normal[1]],
                "size_m": PLATE_SIZES_M[number % len(PLATE_SIZES_M)],
                "texture": f"plate_{number:03d}.png",
            })
            number += 1
            i += 1
            t += PLATE_SPACING_M
    posters = []
    long_runs = sorted((s for s in segs if s.length >= POSTER_MIN_RUN_M),
                       key=lambda s: -s.length)[:POSTER_MAX]
    for seg in long_runs:
        mid = (seg.start + seg.end) / 2.0
        posters.append({
            "pos": _placement_pos(seg, mid, POSTER_HEIGHT_M),
            "normal": [seg.normal[0], seg.normal[1]],
            "size_m": POSTER_SIZE_M,
            "texture": "poster_vbti.png",
        })
    return {
        "resolution": float(resolution),
        "origin": [float(origin[0]), float(origin[1])],
        "plates": plates,
        "posters": posters,
    }


def quad_corners(pos, normal, size: float, height: float | None = None) -> list[list[float]]:
    """4 quad corners (world), wound so the face normal points along `normal`
    (into the room). Consumed by build_dressing_usd — kept here so the geometry
    math stays in the pure, tested module."""
    px, py, pz = pos
    nx, ny = normal
    ax, ay = -ny, nx  # horizontal along-wall axis
    hw = size / 2.0
    hh = (height if height is not None else size) / 2.0
    return [
        [px - ax * hw, py - ay * hw, pz - hh],
        [px + ax * hw, py + ay * hw, pz - hh],
        [px + ax * hw, py + ay * hw, pz + hh],
        [px - ax * hw, py - ay * hw, pz + hh],
    ]


def write_layout(layout: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(layout, sort_keys=False))


def load_layout(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dressing layout.yaml from a nav map.")
    ap.add_argument("--map", type=Path, required=True,
                    help="dir with occupancy.npy + occupancy.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    occ = np.load(args.map / "occupancy.npy")
    meta = json.loads((args.map / "occupancy.json").read_text())
    layout = build_layout(occ, resolution=meta["resolution"],
                          origin=tuple(meta["origin"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_layout(layout, args.out)
    print(f"[layout_gen] {len(layout['plates'])} plates, "
          f"{len(layout['posters'])} posters -> {args.out}")


if __name__ == "__main__":
    main()
