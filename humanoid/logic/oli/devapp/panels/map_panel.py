"""panels/map_panel.py — the 2D nav map: occupancy background + live robot / path / goal.

Pure I/O over the nav layer — NO planning logic lives here. It loads the baked `OccupancyGrid`
for the background + click un-projection, and each frame:
  • renders pose + planned path + goal from `AppState` (the path is produced by the Nav layer in
    the brain, not here), and
  • on LEFT click, un-projects the pixel to a world `GoalCoordinate` and hands it to the nav layer
    via `AppState.set_goal`. RIGHT click clears the goal.

The Nav reason (brain) consumes that goal, plans on its owned costmap, and publishes the path back
— this panel only draws it. Geometry (world↔pixel, overlay) is pure/unit-tested in `map_render.py`.
"""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, immvision

from ..imaging import fit_within
from ..map_render import base_rgb, compose, pixel_to_world
from ..panel import Panel
from ..state import AppState
from ...reason.nav import GoalCoordinate
from ...reason.nav.occupancy_io import load_occupancy


class MapPanel(Panel):
    title = "Nav Map"
    dock_space = "MainDockSpace"

    def __init__(self, map_dir: str) -> None:
        self._grid = load_occupancy(map_dir)   # raw map: render background + click un-projection
        self._base = base_rgb(self._grid)

    def setup(self) -> None:
        immvision.use_rgb_color_order()  # our raster is RGB

    def teardown(self) -> None:
        immvision.clear_texture_cache()  # free GL textures while the context is alive

    def draw(self, state: AppState) -> None:
        pose, path, goal = state.nav_snapshot()
        img = np.array(
            compose(self._grid, self._base, pose=pose, path=path, goal=goal),
            dtype=np.uint8, copy=True,  # ImmVision needs a writeable, contiguous buffer
        )
        h, w = img.shape[:2]
        # Fit the map to the live panel: fill the content region (minus one line for the
        # readout below) so the whole map is always visible when panels are docked/resized.
        avail = imgui.get_content_region_avail()
        disp = fit_within(w, h, avail.x, avail.y - imgui.get_frame_height_with_spacing())
        # "##"-prefixed id → no legend, so the widget rect == the image (exact click mapping).
        immvision.image_display("##oli_navmap", img, disp, True)
        self._handle_click(state, w, h)

        if pose is not None:
            cell = self._grid.world_to_cell(pose.x, pose.y)
            occ = self._grid.is_occupied(pose.x, pose.y)
            imgui.text(
                f"pose  x={pose.x:+.2f}  y={pose.y:+.2f}  yaw={pose.yaw:+.2f}   "
                f"cell={cell}  occupied={occ}"
            )
        else:
            imgui.text_disabled("no pose yet — launch the World with --debug-pose")
        if goal is not None:
            imgui.same_line()
            status = f"path {len(path)} pts" if path else "NO PATH (blocked/unreachable)"
            imgui.text_disabled(f"   goal={goal[0]:+.2f},{goal[1]:+.2f}  ·  {status}")
        else:
            imgui.same_line()
            imgui.text_disabled("   left-click: set goal · right-click: clear")

        # Arm gate (pure I/O — just forwards the toggle; the brain flips Teleop↔Nav on it).
        armed = state.get_armed()
        if imgui.button("Disengage" if armed else "Engage"):
            state.set_armed(not armed)
        imgui.same_line()
        if armed:
            imgui.text_colored(imgui.ImVec4(0.2, 0.9, 0.2, 1.0), "ARMED — Nav driving the path")
        else:
            imgui.text_disabled("disarmed — joystick drives (Engage to follow the path)")

    def _handle_click(self, state: AppState, w: int, h: int) -> None:
        """Un-project a click on the image → world `GoalCoordinate`, hand it to the nav layer."""
        if not imgui.is_item_hovered():
            return
        if imgui.is_mouse_clicked(1):            # right-click clears the goal
            state.set_goal(None)
            return
        if not imgui.is_mouse_clicked(0):
            return
        rmin, rmax = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        mouse = imgui.get_mouse_pos()
        # cursor → fraction of the displayed image → source-image pixel (no zoom/pan on
        # image_display, so this is exact) → world (x, y).
        pcol = (mouse.x - rmin.x) / max(rmax.x - rmin.x, 1e-6) * w
        prow = (mouse.y - rmin.y) / max(rmax.y - rmin.y, 1e-6) * h
        gx, gy = pixel_to_world(self._grid, prow, pcol)
        state.set_goal(GoalCoordinate(gx, gy))   # goal IN → the Nav layer plans; we just render
