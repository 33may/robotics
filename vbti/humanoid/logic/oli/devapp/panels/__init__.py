"""panels — the concrete dockable windows of the dev app.

Each module defines one `Panel` subclass. Panels depend only on `panel.py`, `state.py`,
and (for data) the source protocols in `sources/` — never on Isaac or the World directly,
so they render identically against simulated or real sources.
"""
