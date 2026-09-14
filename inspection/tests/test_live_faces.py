#!/usr/bin/env python3
"""Faces reach the orchestrator: nav block + prompt language.
Run: p inspection/tests/test_live_faces.py"""


class _SupStub:
    _reach = {(0, 0): 0.0}
    visited = {(0, 0)}
    blocked = set()
    phase = "idle"
    target = None

    @staticmethod
    def _face_states():
        return {"x+": {"status": "ready", "off_angle_deg": None},
                "top": {"status": "visited", "off_angle_deg": 12.0}}


def test_nav_carries_faces():
    from inspection.brain.live import SupervisorMover
    mover = SupervisorMover.__new__(SupervisorMover)   # no bus, no run dir
    mover.sup = _SupStub()
    nav = mover.nav()
    assert nav["faces"]["x+"]["status"] == "ready"
    assert nav["faces"]["top"]["off_angle_deg"] == 12.0


def test_prompt_mentions_face_priority():
    import inspection.brain.loop as loop
    import inspect
    src = inspect.getsource(loop)
    assert "face views" in src and "mainly" in src, \
        "system prompt must carry the SOFT face-evidence priority"


class _StoreStub:
    @staticmethod
    def findings():
        return []


class _MoverStub:
    @staticmethod
    def nav():
        return {"reachable": set(), "visited": set(), "blocked": set(),
                "faces": {"x+": {"status": "ready", "off_angle_deg": None},
                          "top": {"status": "visited", "off_angle_deg": 12.0}}}


class _AiStub:
    def __init__(self):
        self.captured = None

    def menu_input(self, mi):
        self.captured = mi


def test_menu_input_faces_survive_unshredded():
    """`_menu_input` used to run EVERY nav() value through
    `sorted(map(list, v))`, which assumes a set of cell tuples. The faces
    block is a dict of dicts — under the old code `{"x+": ..., "top": ...}`
    shredded into `[['t','o','p'], ['x','+']]`, a silent record corruption
    with no exception raised. Face names must arrive intact."""
    from inspection.brain.loop import Brain
    brain = Brain.__new__(Brain)          # no disk, no SDK session
    brain.run = type("R", (), {"survey": None})()
    brain.cell = None
    brain.turn = 0
    brain.store = _StoreStub()
    brain.view_notes = {}
    brain.agent_seen = set()
    brain.mover = _MoverStub()
    brain.ai = _AiStub()

    brain._menu_input()

    faces = brain.ai.captured.agent_context["nav"]["faces"]
    assert faces["x+"]["status"] == "ready"
    assert faces["top"]["off_angle_deg"] == 12.0


def main():
    test_nav_carries_faces()
    test_prompt_mentions_face_priority()
    test_menu_input_faces_survive_unshredded()
    print("OK test_live_faces")


if __name__ == "__main__":
    main()
