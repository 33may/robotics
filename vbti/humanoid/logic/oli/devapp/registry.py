"""registry.py — the panel registry.

Add a panel in one line; the shell just iterates the registry to build dockable windows.
This is the single place panels are declared, so composing a different app (fewer panels,
a test harness) is a matter of registering a different set.
"""

from __future__ import annotations

from typing import List

from .panel import Panel


class PanelRegistry:
    """An ordered collection of panels with unique titles."""

    def __init__(self) -> None:
        self._panels: List[Panel] = []

    def register(self, panel: Panel) -> Panel:
        """Register `panel` and return it (so `p = reg.register(CameraPanel(...))` works).

        Raises ValueError on a duplicate title — ImGui keys windows by label, so titles
        must be unique.
        """
        if any(p.title == panel.title for p in self._panels):
            raise ValueError(f"duplicate panel title: {panel.title!r}")
        self._panels.append(panel)
        return panel

    def panels(self) -> List[Panel]:
        """The registered panels, in registration order (copy)."""
        return list(self._panels)
