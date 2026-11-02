"""locbench/episodes.py — frozen, seeded episode sets (design.md D2/D3).

An evaluation must mean the same thing for every candidate, forever: spawn/goal pairs are
sampled ONCE (seeded), rendered for Anton's approval, frozen to `episodes/<scene>.json`
(versioned, committed), and every run reads the frozen file. Sampling rules:

  - points come from the occupancy free space with `clearance_m` of margin (no wall-huggers);
  - spawn↔goal must be ≥ `min_separation_m` apart (euclidean) AND the PLANNED route
    (A* on the `robot_radius_m`-inflated grid — what Nav actually drives) ≥ `min_route_m`;
  - every spawn is reachable from the boot origin (D3: no World-side teleport — Oli glides
    an unscored transit leg to each spawn), every goal reachable from its spawn.

Spawns carry NO yaw: the transit leg ends at whatever heading the follower arrives with, so
the warm-start pose (`LocalizationSetup.initial_pose`) is read from live GT at episode start.

The same file carries `coverage_goals` — the mapping pass's drive plan (design.md D9):
farthest-point-spread free-space points, ordered as a nearest-neighbor tour from the origin,
each reachable (one connected component ⇒ every leg drivable).

Pure numpy/stdlib on the emitted `OccupancyGrid`; consumed by the evaluator and `locbench
episodes`. Never imported by `logic/oli/`.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..oli.reason.mapping import OccupancyGrid
from ..oli.reason.nav import plan_path

FORMAT_VERSION = 1

Point = Tuple[float, float]

# Sampling gives up after this many rejected draws per episode — loud failure beats a hang.
_MAX_DRAWS_PER_EPISODE = 400


@dataclass(frozen=True)
class Episode:
    """One frozen scored leg: transit to `spawn` (unscored), then drive spawn→goal."""

    id: int
    spawn: Point           # (x, y) map frame — no yaw (read from GT at episode start)
    goal: Point            # (x, y) map frame
    route_m: float         # planned A* route length at freeze time (provenance/reporting)


@dataclass(frozen=True)
class EpisodeSet:
    scene: str
    seed: int
    map_dir: str                     # the baked map these episodes were sampled from
    constraints: Tuple[Tuple[str, float], ...]   # sampling params, frozen for provenance
    episodes: Tuple[Episode, ...]
    coverage_goals: Tuple[Point, ...]            # the mapping pass's ordered drive plan
    version: int = FORMAT_VERSION


def sample_episode_set(
    grid: OccupancyGrid,
    *,
    scene: str,
    map_dir: str,
    seed: int,
    n_episodes: int = 10,
    min_separation_m: float = 8.0,
    min_route_m: float = 10.0,
    clearance_m: float = 1.0,
    robot_radius_m: float = 0.30,
    origin_xy: Point = (0.0, 0.0),
    n_coverage: int = 8,
    zones: Optional[List[dict]] = None,
) -> EpisodeSet:
    """Sample a frozen episode set from the occupancy free space. Deterministic per seed.

    `zones` biases where episodes live (Anton, 13-07-2026: ~70% between the warehouse rails):
    each `{"rect": (x0, y0, x1, y1), "n": k}` draws k episodes whose spawn AND goal both lie
    inside the rect — the planned route then stays in that area. The remaining
    `n_episodes − Σk` episodes sample map-wide. Zone episodes come first (stable ids).
    """
    rng = random.Random(seed)
    candidates = _free_points(grid, clearance_m)
    reach = grid.inflate(robot_radius_m)

    zones = zones or []
    n_zoned = sum(int(z["n"]) for z in zones)
    if n_zoned > n_episodes:
        raise ValueError(f"zone episode counts ({n_zoned}) exceed n_episodes ({n_episodes})")
    pools: List[Tuple[List[Point], str]] = []
    for z in zones:
        pool = [p for p in candidates if _in_rect(p, z["rect"])]
        if not pool:
            raise ValueError(f"zone {z['rect']} contains no free space at "
                             f"clearance {clearance_m} m")
        pools.extend([(pool, f"zone {z['rect']}")] * int(z["n"]))
    pools.extend([(candidates, "map-wide")] * (n_episodes - n_zoned))

    episodes: List[Episode] = []
    for ep_id, (pool, label) in enumerate(pools):
        ep = _draw_episode(rng, pool, reach, ep_id, origin_xy,
                           min_separation_m, min_route_m)
        if ep is None:
            raise ValueError(
                f"could not sample episode {ep_id} ({label}) after "
                f"{_MAX_DRAWS_PER_EPISODE} draws — constraints too tight "
                f"(min_separation={min_separation_m} m, min_route={min_route_m} m, "
                f"clearance={clearance_m} m)"
            )
        episodes.append(ep)

    coverage = _coverage_tour(candidates, reach, origin_xy, n_coverage)

    constraints = (
        ("n_episodes", float(n_episodes)),
        ("min_separation_m", float(min_separation_m)),
        ("min_route_m", float(min_route_m)),
        ("clearance_m", float(clearance_m)),
        ("robot_radius_m", float(robot_radius_m)),
        ("origin_x", float(origin_xy[0])),
        ("origin_y", float(origin_xy[1])),
        ("n_coverage", float(n_coverage)),
    )
    return EpisodeSet(scene=scene, seed=seed, map_dir=map_dir, constraints=constraints,
                      episodes=tuple(episodes), coverage_goals=coverage)


# ── sampling internals ───────────────────────────────────────────────────────


def _in_rect(p: Point, rect) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def _free_points(grid: OccupancyGrid, clearance_m: float) -> List[Point]:
    """Cell-center world coords of every free cell with `clearance_m` of margin."""
    inflated = grid.inflate(clearance_m)
    rows, cols = np.nonzero(~inflated.grid)
    if len(rows) == 0:
        raise ValueError(f"no free space left at clearance {clearance_m} m")
    return [grid.cell_to_world(int(r), int(c)) for r, c in zip(rows, cols)]


def _route_length(path: List[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def _draw_episode(
    rng: random.Random,
    candidates: List[Point],
    reach: OccupancyGrid,
    ep_id: int,
    origin_xy: Point,
    min_separation_m: float,
    min_route_m: float,
) -> Optional[Episode]:
    for _ in range(_MAX_DRAWS_PER_EPISODE):
        spawn = candidates[rng.randrange(len(candidates))]
        goal = candidates[rng.randrange(len(candidates))]
        if math.dist(spawn, goal) < min_separation_m:
            continue
        route = plan_path(reach, spawn, goal)
        if route is None or _route_length(route) < min_route_m:
            continue
        if plan_path(reach, origin_xy, spawn) is None:
            continue  # D3: the transit leg must be drivable — no teleport
        return Episode(id=ep_id, spawn=spawn, goal=goal,
                       route_m=round(_route_length(route), 3))
    return None


def _coverage_tour(
    candidates: List[Point],
    reach: OccupancyGrid,
    origin_xy: Point,
    n_coverage: int,
) -> Tuple[Point, ...]:
    """Farthest-point-spread free points, reachable from origin, as a NN tour from origin.

    All chosen points share the origin's connected component (each is verified reachable),
    so every tour leg is drivable by construction.
    """
    if n_coverage <= 0:
        return ()
    pts = np.asarray(candidates, dtype=float)
    # min distance of every candidate to the chosen set — seeded with the origin, so the
    # first pick is the far side of the map and picks spread from there.
    min_d = np.linalg.norm(pts - np.asarray(origin_xy), axis=1)
    chosen: List[Point] = []
    dead = np.zeros(len(pts), dtype=bool)   # unreachable candidates, masked out
    while len(chosen) < n_coverage:
        order = np.where(~dead)[0]
        if len(order) == 0:
            raise ValueError("coverage sampling exhausted the free space — "
                             "is the origin walled off?")
        idx = order[int(np.argmax(min_d[order]))]
        p = (float(pts[idx, 0]), float(pts[idx, 1]))
        if plan_path(reach, origin_xy, p) is None:
            dead[idx] = True
            continue
        chosen.append(p)
        min_d = np.minimum(min_d, np.linalg.norm(pts - pts[idx], axis=1))
    # nearest-neighbor tour from the origin — the order the mapping drive visits them
    tour: List[Point] = []
    cur = origin_xy
    remaining = chosen[:]
    while remaining:
        nxt = min(remaining, key=lambda q: math.dist(cur, q))
        tour.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return tuple(tour)


# ── freeze / load ────────────────────────────────────────────────────────────


def save_episode_set(es: EpisodeSet, path: str | Path) -> None:
    doc = {
        "version": es.version,
        "scene": es.scene,
        "seed": es.seed,
        "map_dir": es.map_dir,
        "constraints": dict(es.constraints),
        "episodes": [
            {"id": e.id, "spawn": list(e.spawn), "goal": list(e.goal), "route_m": e.route_m}
            for e in es.episodes
        ],
        "coverage_goals": [list(g) for g in es.coverage_goals],
    }
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def load_episode_set(path: str | Path) -> EpisodeSet:
    doc = json.loads(Path(path).read_text())
    if doc.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"episode set {path}: version {doc.get('version')!r} != {FORMAT_VERSION} — "
            "regenerate (and re-approve) the set with this code"
        )
    constraints: Dict[str, float] = doc["constraints"]
    return EpisodeSet(
        scene=doc["scene"],
        seed=int(doc["seed"]),
        map_dir=doc["map_dir"],
        constraints=tuple((k, float(v)) for k, v in constraints.items()),
        episodes=tuple(
            Episode(id=int(e["id"]), spawn=tuple(e["spawn"]), goal=tuple(e["goal"]),
                    route_m=float(e["route_m"]))
            for e in doc["episodes"]
        ),
        coverage_goals=tuple(tuple(g) for g in doc["coverage_goals"]),
        version=int(doc["version"]),
    )
