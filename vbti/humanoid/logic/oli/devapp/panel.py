"""panel.py — the dev-app plugin contract.

A `Panel` is one dockable window. The shell (app.py) owns the window, docking layout and
main loop; a panel only renders itself each frame and, if it is a control panel, acts on
the brain's input seams. This seam is what makes surfaces movable / replaceable: add or
swap a panel without touching the shell.

Lifecycle (called by the shell):
    setup()      once, after the GL context exists (create textures, buffers)
    draw(state)  every frame, inside the panel's ImGui window (Begin/End done for you)
    teardown()   once, on app exit (release resources)

`state` is the shared AppState (state.py): latest-wins brain contracts written by the
brain worker thread and read here on the UI thread. Panels that need no live state
ignore it. Construction-time dependencies (a CameraSource, a launcher) are injected via
the panel's own __init__ — kept separate from per-frame `state` on purpose.
"""

from __future__ import annotations

from .state import AppState


class Panel:
    """Base class for a dockable dev-app panel. Subclass and override `draw`."""

    #: Window title — must be unique (ImGui identifies windows by label).
    title: str = "Panel"
    #: Dock space to place into (see app.py DOCK_* names).
    dock_space: str = "MainDockSpace"
    #: Whether the user may close it from the View menu.
    closable: bool = True

    def setup(self) -> None:
        """One-time init after the GL context exists. Override if needed."""

    def draw(self, state: AppState) -> None:  # noqa: D401
        """Render this panel's ImGui contents for the current frame. Must override."""
        raise NotImplementedError

    def teardown(self) -> None:
        """One-time cleanup on exit. Override if needed."""
