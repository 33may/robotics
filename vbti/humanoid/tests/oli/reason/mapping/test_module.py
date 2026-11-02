"""`Map` contract + `MappingModule` protocol + `StaticMapping` (design.md D7).

The mapping module answers "what does the world look like" as a versioned snapshot: downstream
caches (the planner's derived robot layer) key on `version`, which bumps ONLY when grid content
changes. `StaticMapping` is the v1 realization over the baked artifacts — version constant
forever. Pure (`brain` env).
"""

from __future__ import annotations

import numpy as np
import pytest

from humanoid.logic.oli.reason.mapping import (
    Map,
    MappingModule,
    OccupancyGrid,
    StaticMapping,
    save_occupancy,
)

pytestmark = pytest.mark.brain


def _grid(n=8, res=0.5):
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = True
    return OccupancyGrid(occ, res, origin=(-1.0, -2.0))


# ── Map contract ─────────────────────────────────────────────────────────────

def test_map_holds_grid_version_stamp():
    m = Map(grid=_grid(), version=3, stamp_ns=7)
    assert m.version == 3 and m.stamp_ns == 7 and m.grid.resolution == 0.5


def test_map_is_frozen_and_coerces():
    m = Map(grid=_grid(), version=1.0, stamp_ns=2.0)
    assert isinstance(m.version, int) and isinstance(m.stamp_ns, int)
    with pytest.raises(Exception):
        m.version = 9  # type: ignore[misc]


# ── StaticMapping — the v1 realization ───────────────────────────────────────

def test_static_mapping_serves_baked_artifacts(tmp_path):
    save_occupancy(_grid(), tmp_path)
    mapping = StaticMapping(tmp_path)
    m = mapping.latest()
    assert m is not None
    assert m.grid.grid[0].all() and not m.grid.grid[1:].any()
    assert m.grid.origin == (-1.0, -2.0)


def test_static_mapping_version_never_bumps(tmp_path):
    save_occupancy(_grid(), tmp_path)
    mapping = StaticMapping(tmp_path)
    first = mapping.latest()
    for _ in range(3):
        again = mapping.latest()
        assert again.version == first.version
        assert again.grid is first.grid  # same snapshot object — zero per-call cost


def test_static_mapping_satisfies_protocol(tmp_path):
    save_occupancy(_grid(), tmp_path)
    assert isinstance(StaticMapping(tmp_path), MappingModule)
    assert not isinstance(object(), MappingModule)
