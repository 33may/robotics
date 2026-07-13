"""mapping/contracts.py — the versioned world-truth snapshot (design.md D7).

`Map` is what the mapping module emits and everyone else consumes — the planner receives it as
an explicit input value (`plan(pose, goal, map)`, design.md D8) and keys its derived robot layer
on `version`. The version bumps ONLY when grid content changes: with today's `StaticMapping`
that is never, so derived layers are built exactly once; a live reconstructor later bumps it on
real updates and downstream rebuilds exactly then. Pure: numpy/stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .costmap import OccupancyGrid


@dataclass(frozen=True)
class Map:
    """World truth as a snapshot: boolean occupancy + where it came from in time.

    `grid` is the raw baked/reconstructed map (world truth ONLY — footprint inflation and
    clearance are the planner's robot layer, never baked in here). Full-snapshot semantics by
    decision: partial/tile updates are a deferred optimization a future version bump would key.
    """

    grid: OccupancyGrid
    version: int    # monotonic; bumps only when grid CONTENT changes
    stamp_ns: int   # sim-time of the snapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
