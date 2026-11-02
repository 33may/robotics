"""Shell-assembly test: build_runner_params wires one dockable window per panel.

Skipped where imgui_bundle is absent (e.g. the isaac env). No GL context / display is
needed here — RunnerParams is just a struct; hello_imgui.run() is never called.
"""

import pytest

pytest.importorskip("imgui_bundle")

from humanoid.logic.oli.devapp.app import (  # noqa: E402
    DOCK_LEFT,
    DOCK_MAIN,
    build_runner_params,
)
from humanoid.logic.oli.devapp.panel import Panel  # noqa: E402
from humanoid.logic.oli.devapp.registry import PanelRegistry  # noqa: E402
from humanoid.logic.oli.devapp.state import AppState  # noqa: E402

pytestmark = pytest.mark.brain


class _P(Panel):
    def __init__(self, title, dock):
        self.title = title
        self.dock_space = dock

    def draw(self, state):  # noqa: D401
        pass


def test_one_dockable_window_per_panel():
    reg = PanelRegistry()
    reg.register(_P("Cam", DOCK_MAIN))
    reg.register(_P("Teleop", DOCK_LEFT))
    rp = build_runner_params(reg, AppState())
    wins = rp.docking_params.dockable_windows
    assert [w.label for w in wins] == ["Cam", "Teleop"]
    assert [w.dock_space_name for w in wins] == [DOCK_MAIN, DOCK_LEFT]


def test_menu_and_dockspace_enabled():
    from imgui_bundle import hello_imgui

    rp = build_runner_params(PanelRegistry(), AppState())
    assert (
        rp.imgui_window_params.default_imgui_window_type
        == hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    assert rp.imgui_window_params.show_menu_bar is True
