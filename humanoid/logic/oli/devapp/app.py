"""app.py — the robot-brain dev app shell.

Builds the Hello ImGui runner: a full-screen dock space, menu + status bar, and one
dockable window per registered `Panel` (registry.py). The shell owns the window, the
docking layout and the main loop; panels only render themselves.

The app does NOT own the World — the robot is already spawned in some World and this app
attaches. Booting the brain worker thread (the Orchestrator loop) is wired here later;
for the camera milestone the app just renders panels.

Default dock layout — the space names panels target via `Panel.dock_space`:
    MainDockSpace   big central area (camera streams, later the 3D scene)
    LeftSpace       narrow left column (teleop / controls)
    BottomSpace     short bottom strip (diagnostics / logs)
"""

from __future__ import annotations

from typing import Optional

from imgui_bundle import hello_imgui, imgui

from .capture import enable_screenshot
from .registry import PanelRegistry
from .state import AppState

DOCK_MAIN = "MainDockSpace"
DOCK_LEFT = "LeftSpace"
DOCK_BOTTOM = "BottomSpace"


def _default_splits() -> list:
    """Partition the root dock space into MainDockSpace | LeftSpace | BottomSpace."""
    left = hello_imgui.DockingSplit()
    left.initial_dock = DOCK_MAIN
    left.new_dock = DOCK_LEFT
    left.direction = imgui.Dir.left
    left.ratio = 0.25

    bottom = hello_imgui.DockingSplit()
    bottom.initial_dock = DOCK_MAIN
    bottom.new_dock = DOCK_BOTTOM
    bottom.direction = imgui.Dir.down
    bottom.ratio = 0.22
    return [left, bottom]


def build_runner_params(
    registry: PanelRegistry,
    state: AppState,
    *,
    window_title: str = "Oli — Robot Brain",
    size=(1600, 1000),
) -> "hello_imgui.RunnerParams":
    """Assemble a Hello ImGui RunnerParams from the panel registry."""
    rp = hello_imgui.RunnerParams()
    rp.app_window_params.window_title = window_title
    rp.app_window_params.window_geometry.size = size

    rp.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    rp.imgui_window_params.show_menu_bar = True
    rp.imgui_window_params.show_status_bar = True

    dp = hello_imgui.DockingParams()
    dp.docking_splits = _default_splits()
    windows = []
    for panel in registry.panels():
        w = hello_imgui.DockableWindow()
        w.label = panel.title
        w.dock_space_name = panel.dock_space
        w.can_be_closed = panel.closable
        # Bind each window's gui to its panel; default-arg captures the current panel.
        w.gui_function = (lambda p=panel: p.draw(state))
        windows.append(w)
    dp.dockable_windows = windows
    rp.docking_params = dp

    prev_post_init = rp.callbacks.post_init
    prev_before_exit = rp.callbacks.before_exit
    prev_pre_new_frame = rp.callbacks.pre_new_frame

    def post_init() -> None:
        if prev_post_init is not None:
            prev_post_init()
        for panel in registry.panels():
            panel.setup()

    def before_exit() -> None:
        for panel in registry.panels():
            panel.teardown()
        if prev_before_exit is not None:
            prev_before_exit()

    def pre_new_frame() -> None:
        if prev_pre_new_frame is not None:
            prev_pre_new_frame()
        state.tick()

    rp.callbacks.post_init = post_init
    rp.callbacks.before_exit = before_exit
    rp.callbacks.pre_new_frame = pre_new_frame
    return rp


def run(
    registry: PanelRegistry,
    state: Optional[AppState] = None,
    *,
    screenshot: Optional[str] = None,
    n_frames: int = 20,
    **kw,
) -> None:
    """Run the dev app. If `screenshot` is set, capture a PNG after `n_frames` and exit."""
    state = state or AppState()
    rp = build_runner_params(registry, state, **kw)
    if screenshot:
        enable_screenshot(rp, screenshot, n_frames=n_frames)
    hello_imgui.run(rp)
