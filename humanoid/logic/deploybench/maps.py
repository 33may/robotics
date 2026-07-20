"""deploybench/maps.py — resolve a baked map into the OccupancyGrid we render + validate on.

The picker draws this grid and start/goal are validated against it (FR2). This is the
DISPLAY/PLANNING occupancy map — distinct from the algorithm's private localization artifacts
(the cuVSLAM LMDB / cuVGL BoW under the same bake), which only the `LocalizationModule` reads.
Under the demo's map-frame-is-world rule (D1) they share a frame, so a point picked here means
the same thing to the planner and the localizer.
"""

from __future__ import annotations

from pathlib import Path

from ..oli.reason.mapping import OccupancyGrid, StaticMapping


def load_grid(map_dir: str | Path) -> OccupancyGrid:
    """Load the baked occupancy grid under `map_dir` (raises if none is baked there)."""
    snap = StaticMapping(str(map_dir)).latest()
    if snap is None:
        raise FileNotFoundError(f"no baked occupancy map under {map_dir!r}")
    return snap.grid
