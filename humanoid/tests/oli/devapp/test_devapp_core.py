"""Unit tests for the dev-app plugin core (registry, state, panel) — pure, no GUI."""

import pytest

from humanoid.logic.oli.devapp.panel import Panel
from humanoid.logic.oli.devapp.registry import PanelRegistry
from humanoid.logic.oli.devapp.state import AppState

pytestmark = pytest.mark.brain


class _P(Panel):
    def __init__(self, title, dock="MainDockSpace"):
        self.title = title
        self.dock_space = dock

    def draw(self, state):  # noqa: D401
        pass


def test_registry_register_returns_panel():
    reg = PanelRegistry()
    p = _P("A")
    assert reg.register(p) is p
    assert reg.panels() == [p]


def test_registry_rejects_duplicate_title():
    reg = PanelRegistry()
    reg.register(_P("A"))
    with pytest.raises(ValueError):
        reg.register(_P("A"))


def test_registry_panels_is_copy_in_order():
    reg = PanelRegistry()
    a, b = _P("A"), _P("B")
    reg.register(a)
    reg.register(b)
    got = reg.panels()
    assert got == [a, b]
    got.append(_P("C"))
    assert len(reg.panels()) == 2  # external mutation must not leak into the registry


def test_appstate_tick_increments():
    s = AppState()
    assert s.frame_index == 0
    s.tick()
    s.tick()
    assert s.frame_index == 2


def test_appstate_brain_snapshot_default_detached():
    s = AppState()
    attached, obs, pout, mode = s.brain_snapshot()
    assert attached is False and obs is None and pout is None and mode == "—"


def test_appstate_set_brain_publishes():
    s = AppState()
    s.set_brain(obs="OBS", policy_out="POUT", mode_name="WALK")
    attached, obs, pout, mode = s.brain_snapshot()
    assert attached is True and obs == "OBS" and pout == "POUT" and mode == "WALK"


def test_panel_base_draw_not_implemented():
    with pytest.raises(NotImplementedError):
        Panel().draw(AppState())
