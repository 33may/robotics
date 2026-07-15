"""cell_coverage.py — cell-grid random coverage generator (MAY-173 locdev T2).

Anton's algorithm (15-07-2026): split the map extent into an R×C grid of cells;
in every cell rejection-sample K random FREE points (a rack/wall cell yields
fewer or none). Cells are visited by a FRONTIER RANDOM WALK (`neighbour_walk`):
from the current cell pick a random UNVISITED Moore-neighbour (8-connected, so
diagonal cell-to-cell passes are as likely as orthogonal); when the local ring
is exhausted, jump to the NEAREST unvisited cell. This keeps most moves local
and diagonal — the varied viewpoints a BoW/landmark map needs — while the
occasional expansion-jump adds novel long diagonal passes, WITHOUT the
whole-map criss-cross chaos of a pure-random cell order.

Lap 1 follows each cell's points in sampled sequence; lap 2 re-walks (fresh rng
draw) and follows each cell's sequence REVERSED — same positions, opposite
headings, doubling viewpoint directions.

`points_per_cell` is the collection-size knob (int, or a (row, col)->int for
zone weighting). The flat `targets` list feeds routes.plan_route, which stitches
every consecutive pair (intra-cell hops AND cell transitions) with the
deployment planner.

Pure numpy/stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from humanoid.logic.oli.reason.mapping.costmap import OccupancyGrid

Cell = Tuple[int, int]  # (row, col) in the coverage grid, row 0 = south
Point = Tuple[float, float]


def _ppc_from_spec(spec) -> Union[int, Callable[[int, int], int]]:
    """Build a points_per_cell value from a YAML spec node: either a plain int, or
    a `{default: N, zones: [{rows: [...], points: M}, ...]}` zone-weighting map."""
    if isinstance(spec, int):
        return spec
    default = int(spec["default"])
    row_points = {}
    for zone in spec.get("zones", []):
        for r in zone["rows"]:
            row_points[int(r)] = int(zone["points"])
    return lambda r, c: row_points.get(r, default)


@dataclass(frozen=True)
class CellCoverage:
    """The generated tour: per-cell samples, per-lap cell visit orders, and the
    flat target sequence (all laps concatenated)."""

    cells: Dict[Cell, List[Point]]
    cell_orders: List[List[Cell]]  # one visit order per lap (empty cells excluded)
    targets: List[Point] = field(default_factory=list)


def neighbour_walk(usable: List[Cell], rng: np.random.Generator, start: Cell) -> List[Cell]:
    """Order `usable` cells by a frontier random walk from `start`.

    At each step: pick a random unvisited cell among those at the smallest
    Chebyshev distance from the current cell. Chebyshev = Moore ring, so distance
    1 is the 8 neighbours (diagonals included); when all are visited the nearest
    ring with an unvisited cell supplies the (usually diagonal) jump. Deterministic
    for a given rng state.
    """
    remaining = set(usable)
    if start not in remaining:
        raise ValueError(f"start {start} not in usable cells")
    remaining.discard(start)
    order = [start]
    cur = start
    while remaining:
        dmin = min(max(abs(cur[0] - r), abs(cur[1] - c)) for r, c in remaining)
        ring = [cell for cell in remaining
                if max(abs(cur[0] - cell[0]), abs(cur[1] - cell[1])) == dmin]
        ring.sort()  # stable candidate order so rng choice is reproducible
        nxt = ring[int(rng.integers(len(ring)))]
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return order


def generate_cell_coverage(
    grid: OccupancyGrid,
    *,
    cells: Tuple[int, int] = (5, 5),
    points_per_cell: Union[int, Callable[[int, int], int]] = 10,
    margin_m: float = 1.0,
    seed: int = 33,
    laps: int = 2,
    max_tries_per_point: int = 200,
) -> CellCoverage:
    """Sample the per-cell points and build the multi-lap target sequence."""
    rng = np.random.default_rng(seed)
    free = grid.inflate(margin_m)
    rows, cols = cells
    x0, y0 = grid.origin
    w = grid.ncols * grid.resolution / cols
    h = grid.nrows * grid.resolution / rows

    ppc = points_per_cell if callable(points_per_cell) else (lambda r, c: points_per_cell)

    sampled: Dict[Cell, List[Point]] = {}
    for r in range(rows):
        for c in range(cols):
            want = int(ppc(r, c))
            pts: List[Point] = []
            tries = 0
            while len(pts) < want and tries < max_tries_per_point * max(want, 1):
                tries += 1
                x = float(rng.uniform(x0 + c * w, x0 + (c + 1) * w))
                y = float(rng.uniform(y0 + r * h, y0 + (r + 1) * h))
                if not free.is_occupied(x, y):
                    pts.append((x, y))
            sampled[(r, c)] = pts

    usable = [cell for cell, pts in sampled.items() if pts]
    orders: List[List[Cell]] = []
    targets: List[Point] = []
    # Start the first walk at the usable cell nearest the SW corner (row 0, col 0)
    # — the drive's natural spawn corner; each next lap continues from the last
    # cell of the previous one, so the recording is one unbroken sweep.
    start = min(usable, key=lambda rc: (rc[0] + rc[1], rc))
    for lap in range(laps):
        order = neighbour_walk(usable, rng, start)
        orders.append(order)
        for cell in order:
            seq = sampled[cell] if lap % 2 == 0 else list(reversed(sampled[cell]))
            targets.extend(seq)
        start = order[-1]

    return CellCoverage(cells=sampled, cell_orders=orders, targets=targets)


def build_coverage_route(path, grid: OccupancyGrid):
    """Load a committed coverage spec YAML and produce (CellCoverage, Route).

    The Route wraps the generated targets with the spec's clearance/speed so the
    drive can hand it straight to routes.plan_route. Deterministic per spec seed.
    """
    import yaml  # lazy: keeps the brain import path yaml-free

    from humanoid.logic.simulation.mapping.routes import Route

    raw = yaml.safe_load(__import__("pathlib").Path(path).read_text())
    cov = generate_cell_coverage(
        grid,
        cells=tuple(raw.get("cells", (5, 5))),
        points_per_cell=_ppc_from_spec(raw["points_per_cell"]),
        margin_m=float(raw.get("margin_m", 1.0)),
        seed=int(raw.get("seed", 33)),
        laps=int(raw.get("laps", 2)),
    )
    route = Route(
        name=str(raw["name"]),
        clearance_m=float(raw.get("clearance_m", 0.5)),
        speed=float(raw.get("speed", 0.8)),
        loop=bool(raw.get("loop", False)),
        waypoints=cov.targets,
    )
    return cov, route
