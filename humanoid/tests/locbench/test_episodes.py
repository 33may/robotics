"""TDD for locbench episode sets (logic/locbench/episodes.py) — D2/D3.

Seeded sampling of spawn/goal pairs from the baked occupancy free space: every point keeps
clearance from walls, every pair is far enough apart (euclidean) AND long enough as a planned
route (A* length through aisles), and everything is reachable — including the unscored transit
leg from the robot's boot origin to each spawn (D3: no teleport, Oli glides there). Frozen to
a versioned JSON the evaluator and the mapping pass both read. Pure numpy/stdlib → `brain`.
"""

import json
import math

import numpy as np
import pytest

from humanoid.logic.locbench.episodes import (
    EpisodeSet,
    load_episode_set,
    sample_episode_set,
    save_episode_set,
)
from humanoid.logic.oli.reason.mapping import OccupancyGrid

pytestmark = pytest.mark.brain


def _open_grid(n=60, res=0.5):
    """A 30×30 m open room with a border wall — plenty of free space."""
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return OccupancyGrid(occ, res)


def _two_rooms(n=60, res=0.5):
    """Two rooms joined by a door — routes through the door are much longer than euclidean."""
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[:, n // 2] = True
    occ[n // 2 - 2 : n // 2 + 2, n // 2] = False  # the door
    return OccupancyGrid(occ, res)


def _sample(grid=None, **over):
    kw = dict(scene="testscene", map_dir="maps/test", seed=7, n_episodes=5,
              min_separation_m=8.0, min_route_m=10.0, clearance_m=1.0,
              origin_xy=(15.0, 15.0), n_coverage=6)
    kw.update(over)
    return sample_episode_set(grid if grid is not None else _open_grid(), **kw)


# ── sampling constraints ─────────────────────────────────────────────────────


def test_samples_the_requested_count():
    es = _sample()
    assert isinstance(es, EpisodeSet)
    assert len(es.episodes) == 5
    assert es.scene == "testscene" and es.seed == 7 and es.version == 1


def test_points_keep_clearance_from_walls():
    es = _sample(clearance_m=2.0)
    inflated = _open_grid().inflate(2.0)
    for ep in es.episodes:
        assert not inflated.is_occupied(*ep.spawn), f"spawn {ep.spawn} hugs a wall"
        assert not inflated.is_occupied(*ep.goal), f"goal {ep.goal} hugs a wall"


def test_pairs_respect_min_euclidean_separation():
    es = _sample(min_separation_m=12.0)
    for ep in es.episodes:
        d = math.dist(ep.spawn, ep.goal)
        assert d >= 12.0, f"episode {ep.id}: spawn-goal only {d:.1f} m apart"


def test_routes_respect_min_planned_length():
    # In the two-room world a pair can be 8 m apart euclidean but 20+ m through the door —
    # the constraint is on the PLANNED route, what Oli actually drives.
    es = _sample(grid=_two_rooms(), min_separation_m=5.0, min_route_m=18.0,
                 n_episodes=3, clearance_m=0.5)
    for ep in es.episodes:
        assert ep.route_m >= 18.0


def test_spawns_are_reachable_from_the_boot_origin():
    # D3: no teleport — every spawn must be glidable-to from where the robot boots.
    es = _sample(grid=_two_rooms(), origin_xy=(7.0, 15.0), n_episodes=3,
                 min_separation_m=5.0, min_route_m=6.0, clearance_m=0.5)
    from humanoid.logic.oli.reason.nav import plan_path
    inflated = _two_rooms().inflate(0.30)
    for ep in es.episodes:
        assert plan_path(inflated, (7.0, 15.0), ep.spawn) is not None


def test_spawn_carries_no_yaw():
    # The transit leg ends at whatever heading PurePursuit arrives with; the warm-start pose
    # is read from live GT at episode start — a frozen yaw would be a lie.
    es = _sample()
    for ep in es.episodes:
        assert len(ep.spawn) == 2 and len(ep.goal) == 2


def test_impossible_constraints_fail_loud():
    with pytest.raises(ValueError, match="episode"):
        _sample(min_separation_m=200.0)  # bigger than the room — cannot satisfy


# ── determinism (the whole point of freezing) ────────────────────────────────


def test_same_seed_same_set():
    assert _sample() == _sample()


def test_different_seed_different_set():
    assert _sample(seed=7) != _sample(seed=8)


# ── coverage goals for the mapping pass (3.2) ────────────────────────────────


def test_coverage_goals_spread_and_reachable():
    es = _sample(n_coverage=6)
    assert len(es.coverage_goals) == 6
    inflated = _open_grid().inflate(0.30)
    from humanoid.logic.oli.reason.nav import plan_path
    prev = (15.0, 15.0)  # the drive visits them in order, starting at the boot origin
    for g in es.coverage_goals:
        assert not inflated.is_occupied(*g)
        assert plan_path(inflated, prev, g) is not None
        prev = g


# ── freeze / load (versioned JSON) ───────────────────────────────────────────


def test_json_roundtrip(tmp_path):
    es = _sample()
    p = tmp_path / "testscene.json"
    save_episode_set(es, p)
    assert load_episode_set(p) == es
    doc = json.loads(p.read_text())          # the frozen file is honest about provenance
    assert doc["version"] == 1 and doc["seed"] == 7 and doc["scene"] == "testscene"
    assert doc["map_dir"] == "maps/test"
    assert "constraints" in doc


def test_load_rejects_unknown_version(tmp_path):
    es = _sample()
    p = tmp_path / "testscene.json"
    save_episode_set(es, p)
    doc = json.loads(p.read_text())
    doc["version"] = 99
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="version"):
        load_episode_set(p)
