"""nav/occupancy_io.py — load/save the baked 2D occupancy artifact.

The map is baked ONCE offline from the scene USD (Isaac `omap` generator, world-side) into a
plain artifact the brain loads at boot — so the brain never imports isaac, it just reads files.
The artifact is a directory holding:

  - `occupancy.npy`  : (H, W) bool array, True = occupied  (row-major, indexed [row, col])
  - `occupancy.json` : {"resolution": <meters/cell>, "origin": [x, y]}  — map-frame world coords
                       of the grid's (row=0, col=0) cell corner.

`save_occupancy`/`load_occupancy` are the runtime sides (pure numpy + stdlib json — the brain's
load path never imports anything heavier). `occupancy_from_image`/`convert_ros_map` are the
build-time bridge: they turn Isaac's Occupancy-Map GUI export (a ROS `map_server` PNG + YAML) into
that artifact, importing PIL + PyYAML *lazily* so those stay off the brain's load path. Run it as a
script:  `python -m humanoid.logic.oli.reason.nav.occupancy_io <png> <yaml> <out_dir> [--preview p]`.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .costmap import OccupancyGrid

_GRID_FILE = "occupancy.npy"
_META_FILE = "occupancy.json"


def save_occupancy(grid: OccupancyGrid, path: str) -> None:
    """Write an `OccupancyGrid` to `path/` as `occupancy.npy` + `occupancy.json`."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, _GRID_FILE), grid.grid)
    with open(os.path.join(path, _META_FILE), "w") as f:
        json.dump({"resolution": grid.resolution, "origin": list(grid.origin)}, f)


def load_occupancy(path: str) -> OccupancyGrid:
    """Load the baked artifact in `path/` back into an `OccupancyGrid`."""
    grid = np.load(os.path.join(path, _GRID_FILE))
    with open(os.path.join(path, _META_FILE)) as f:
        meta = json.load(f)
    ox, oy = meta["origin"]
    return OccupancyGrid(grid, float(meta["resolution"]), (float(ox), float(oy)))


# ── build-time bridge: ROS/omap map (PNG + YAML) → OccupancyGrid ──────────────
# Converting here (rather than a separate script) keeps all occupancy-artifact I/O in one module.
# PIL + PyYAML are imported lazily inside the build-time helpers, so the brain's load path above
# stays pure numpy + json.


def occupancy_from_image(
    gray,
    resolution: float,
    origin,
    *,
    negate: int = 0,
    occupied_thresh: float = 0.65,
    free_thresh: float = 0.196,
    unknown_occupied: bool = True,
) -> OccupancyGrid:
    """Convert a ROS map image + meta into an `OccupancyGrid`. Pure: numpy only.

    `gray` is the map image (PNG row 0 = top); RGBA/RGB collapse to one channel (omap writes equal
    RGB). ROS occupancy probability = `g/255 if negate else (255-g)/255`; a cell is occupied above
    `occupied_thresh`, free below `free_thresh`, else unknown. `unknown_occupied` folds unknown →
    occupied (conservative — the robot won't plan into unseen pockets / beyond the walls). Applies
    the vertical flip so the grid's row 0 is the origin corner (min-y), matching the costmap
    convention (`[row=+y, col=+x]`).
    """
    g = np.asarray(gray)
    if g.ndim == 3:
        g = g[..., 0]  # RGBA/RGB → one channel
    g = g.astype(np.float64)
    occ = g / 255.0 if negate else (255.0 - g) / 255.0
    occupied = occ > occupied_thresh
    if unknown_occupied:
        occupied = occupied | ((occ >= free_thresh) & (occ <= occupied_thresh))
    occupied = np.flipud(occupied)  # PNG top row = max-y → flip so row 0 = min-y (origin)
    return OccupancyGrid(occupied, float(resolution), (float(origin[0]), float(origin[1])))


def _load_gray(png_path: str) -> np.ndarray:
    from PIL import Image  # lazy: build-time only

    return np.array(Image.open(png_path))


def _load_meta(yaml_path: str) -> dict:
    import yaml  # lazy: build-time only

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _write_preview(grid: OccupancyGrid, out_png: str, mark_world=(0.0, 0.0)) -> None:
    from PIL import Image  # lazy: build-time only

    img = np.where(grid.grid, 0, 255).astype(np.uint8)  # occupied black, free white
    rgb = np.stack([img, img, img], axis=-1)
    if mark_world is not None:
        r, c = grid.world_to_cell(*mark_world)
        if grid.in_bounds(r, c):
            rgb[max(0, r - 3):r + 4, max(0, c - 3):c + 4] = (255, 0, 0)  # mark world (0,0)
    Image.fromarray(np.flipud(rgb), "RGB").save(out_png)  # flip to view north-up (max-y at top)


def convert_ros_map(
    png_path: str,
    yaml_path: str,
    out_dir: str,
    *,
    unknown_occupied: bool = True,
    preview: "str | None" = None,
) -> OccupancyGrid:
    """Load a ROS/omap PNG+YAML, convert to an `OccupancyGrid`, `save_occupancy` it to `out_dir`.

    Returns the grid. If `preview` is given, also writes a north-up PNG (occupied black, free white,
    a red mark at world (0,0)) for eyeballing orientation.
    """
    meta = _load_meta(yaml_path)
    grid = occupancy_from_image(
        _load_gray(png_path),
        meta["resolution"],
        meta["origin"],
        negate=int(meta.get("negate", 0)),
        occupied_thresh=float(meta.get("occupied_thresh", 0.65)),
        free_thresh=float(meta.get("free_thresh", 0.196)),
        unknown_occupied=unknown_occupied,
    )
    save_occupancy(grid, out_dir)
    if preview is not None:
        _write_preview(grid, preview)
    return grid


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert an Isaac omap / ROS map_server export (PNG + YAML) into the brain "
        "OccupancyGrid artifact (occupancy.npy + occupancy.json)."
    )
    ap.add_argument("png", help="ROS/omap grayscale map PNG")
    ap.add_argument("yaml", help="map_server YAML (resolution, origin, negate, thresholds)")
    ap.add_argument("out_dir", help="output dir for occupancy.npy + occupancy.json")
    ap.add_argument("--unknown-free", action="store_true",
                    help="treat unknown cells as free (default: occupied)")
    ap.add_argument("--preview", default=None, help="also write a north-up preview PNG here")
    a = ap.parse_args(argv)

    grid = convert_ros_map(
        a.png, a.yaml, a.out_dir, unknown_occupied=not a.unknown_free, preview=a.preview
    )
    ox, oy = grid.origin
    r, c = grid.world_to_cell(0.0, 0.0)
    print(f"grid {grid.nrows}x{grid.ncols}  res {grid.resolution}  origin {grid.origin}")
    print(f"extent x[{ox:.2f},{ox + grid.ncols * grid.resolution:.2f}] "
          f"y[{oy:.2f},{oy + grid.nrows * grid.resolution:.2f}]")
    print(f"occupied fraction {float(grid.grid.mean()):.3f}")
    print(f"world (0,0) -> cell (row={r}, col={c}) occupied={grid.is_occupied_cell(r, c)}")
    if a.preview:
        print("preview ->", a.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
