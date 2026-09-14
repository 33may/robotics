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


def main():
    test_nav_carries_faces()
    test_prompt_mentions_face_priority()
    print("OK test_live_faces")


if __name__ == "__main__":
    main()
