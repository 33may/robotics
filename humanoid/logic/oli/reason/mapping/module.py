"""mapping/module.py — the `MappingModule` protocol + the static v1 realization (design.md D7).

Pull seam: consumers ask for the newest snapshot; a static map costs nothing per call, a live
reconstructor serves its latest without anyone pacing it. `None` = no map yet (a live module at
boot) — Nav's hold semantics already cover it.

This module is the designated EMBRYO of a future `world_representation`: when dynamic
reconstruction / semantics / the reasoning demo need a 3D answer, `latest()` grows richer views
(the 2D grid becomes the planner's projection of it) — the seam stays put; widen the contract
when a second consumer with a 3D need actually exists, not before. Algorithm-private 3D maps
(RTAB-Map `.db` etc.) are NOT here — they live inside the localization module (design.md D12).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from .contracts import Map
from .occupancy_io import load_occupancy


@runtime_checkable
class MappingModule(Protocol):
    """`latest() -> Optional[Map]` — the newest world snapshot, or None before one exists."""

    def latest(self) -> Optional[Map]: ...


class StaticMapping:
    """v1 `MappingModule`: the baked artifacts (`occupancy.npy` + `occupancy.json`) as one
    immutable snapshot. Loads once at construction; `latest()` returns the SAME `Map` forever
    (version constant), so downstream version-keyed caches derive exactly once — today's
    behavior and cost, preserved by construction."""

    def __init__(self, map_dir: str | Path, *, version: int = 1) -> None:
        self._map = Map(grid=load_occupancy(str(map_dir)), version=version, stamp_ns=0)

    def latest(self) -> Optional[Map]:
        return self._map
