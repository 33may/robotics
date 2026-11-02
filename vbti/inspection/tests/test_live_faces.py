#!/usr/bin/env python3
"""Faces reach the orchestrator: nav block + prompt language.
Run: p inspection/tests/test_live_faces.py"""
from inspection.view.rays import RayResolution

#: `sup._reach` holds resolved RAYS everywhere else (Stage B), and `nav()`
#: only ever reads `r is not None` off it — but the type has to be the real
#: one, not a float lookalike that would keep passing if the map's value
#: contract changed underneath these stubs (the same fix task 4 made in
#: test_collect.py). The fields are never inspected.
_RESOLVED = RayResolution(None, None, None, 0.0, 0.0, 0.0, 0.0, [])


class _SupStub:
    _reach = {(0, 0): _RESOLVED}
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
    assert "FACE VIEWS" in src and "BEFORE MINING A MEDIOCRE ONE" in src, \
        "system prompt must carry the NORMATIVE face-evidence priority"
    assert "A read off an oblique view is a lead, not evidence" in src, \
        "oblique reads are leads, never evidence (demo7 ruling)"


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

def test_move_waits_out_a_busy_machine_instead_of_bouncing(tmp_path):
    """Atomicity gate (Anton 2026-09-15): a move tool call during another
    mover's transaction must WAIT for idle, not return an instant bounce —
    the bounce gave the model a free turn mid-motion (bench3: it spat fresh
    moves and inspected stale frames while the operator's move executed)."""
    import queue
    import threading
    import time
    from inspection.brain.live import SupervisorMover

    class _BusySup:
        _reach = {(0, 0): _RESOLVED}
        visited = set()
        blocked = set()
        phase = "executing"          # operator's move in flight
        target = None
        events = queue.Queue()

    mover = SupervisorMover.__new__(SupervisorMover)
    mover.sup = _BusySup()
    mover.run_dir = tmp_path         # no run.json -> _captured_ids() == []
    mover.decider = "test"
    mover.on_event = lambda *a, **kw: None

    def finish_operator_move():
        time.sleep(0.3)
        mover.sup.phase = "idle"     # transaction closes

    t0 = time.monotonic()
    threading.Thread(target=finish_operator_move, daemon=True).start()
    ok, reason = mover.request((0, 0))
    waited = time.monotonic() - t0

    assert waited >= 0.3, "returned before the open transaction closed"
    assert ok is False               # stub machine never ACKs the request
    assert "was busy with another mover" in reason, reason
    assert "request was not sent" not in reason, "the bounce is gone"
