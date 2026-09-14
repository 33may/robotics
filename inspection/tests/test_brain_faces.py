#!/usr/bin/env python3
"""The missing test (final-review finding C1): a face address must survive
the WHOLE brain/eyes tier — gate, view lookup, coverage, gloss, evidence
citation, and the state/moves render — without raising.

Every site here was a real crash or a silent data-loss bug before the fix
round this test locks in: `_gate` cast `int("face")`, `ViewTools.coverage`/
`_gloss` unpacked `h, v = ("face", "top")`, `_collect_evidence` dropped a
face citation through the same int coercion, and `render.state_block`/
`moves_block` unpacked `self.cell` the same way once the agent stood at a
face. The Supervisor-side identity fix (`machine.py::begin_step`'s
`address = ["face", name]`) is what makes a face step's address round-trip
through `Run.at()` at all — proved separately by
`test_machine.py::test_face_request_full_cycle`, and exercised again here
from the READING side.

Run: p inspection/tests/test_brain_faces.py
"""
import asyncio
import tempfile
from pathlib import Path

import numpy as np

from inspection.brain import render
from inspection.brain.loop import Brain, _sorted_addresses
from inspection.eyes.tools import ViewTools
from inspection.record.run import Run
from inspection.tests.record_fixtures import make_legacy_run

T = np.eye(4)


def _face_fixture(tmp):
    """A run with a survey, one ordinary cell, and one captured FACE step —
    address `["face", "top"]`, the unified identity (architect ruling)."""
    root = Path(tmp)
    img = np.zeros((480, 848, 3), np.uint8)
    img[:, :, 1] = 77
    views = [
        {"cell": None, "pose_id": 0, "t": 0.0},
        {"cell": (2, 0), "pose_id": 1, "t": 1.0, "T_base_cam": T, "rgb": img},
        {"cell": ("face", "top"), "pose_id": 2, "t": 2.0,
         "T_base_cam": T, "rgb": img},
    ]
    make_legacy_run(root, views)
    return ViewTools(Run.load(root))


class _MoverStub:
    """Just enough of `SupervisorMover` for `Brain._gate`/`_nav()`."""

    def __init__(self, faces):
        self._faces = faces

    def nav(self):
        return {"faces": self._faces}


class _TraceStub:
    def event(self, *a, **kw):
        pass


def test_gate_passes_ready_face_denies_others():
    """`_gate`'s face branch (C1.1): ready passes, settling/unknown deny,
    and an ordinary cell still goes through the untouched int-bounds path —
    the face branch must not swallow cell traffic."""
    with tempfile.TemporaryDirectory() as tmp:
        brain = Brain.__new__(Brain)
        brain.tools = _face_fixture(tmp)
        brain.mover = _MoverStub({
            "top": {"status": "ready", "off_angle_deg": None},
            "x+": {"status": "settling", "off_angle_deg": None}})

        async def run():
            ok = await brain._gate(
                {"tool_input": {"cell": ["face", "top"]}}, "id", None)
            assert ok == {}, "a ready face must pass the gate"

            settling = await brain._gate(
                {"tool_input": {"cell": ["face", "x+"]}}, "id", None)
            assert (settling["hookSpecificOutput"]["permissionDecision"]
                    == "deny"), "a settling face must not pass"

            unknown = await brain._gate(
                {"tool_input": {"cell": ["face", "bogus"]}}, "id", None)
            assert (unknown["hookSpecificOutput"]["permissionDecision"]
                    == "deny"), "an unknown face name must not pass"

            cell_ok = await brain._gate(
                {"tool_input": {"cell": [2, 0]}}, "id", None)
            assert cell_ok == {}, "an ordinary in-bounds cell must still pass"

            cell_bad = await brain._gate(
                {"tool_input": {"cell": [999, 0]}}, "id", None)
            assert (cell_bad["hookSpecificOutput"]["permissionDecision"]
                    == "deny"), "an out-of-bounds cell must still be denied"

        asyncio.run(run())


def test_face_step_resolves_through_view_at_coverage_gloss():
    """C1.3/C1.4: a captured face step resolves through `view_at`, and
    `coverage`/`_gloss`/`get_view` name it instead of unpacking `h, v`."""
    with tempfile.TemporaryDirectory() as tmp:
        tools = _face_fixture(tmp)

        rec = tools.view_at(("face", "top"))
        assert rec is not None, "the face step must round-trip through Run.at"
        assert rec.cell == ("face", "top")

        gloss = tools._gloss(rec)
        assert "face" in gloss and "top" in gloss

        img = tools.get_view(("face", "top"))
        assert img.rgb.shape == (480, 848, 3)
        assert "face" in img.text and "top" in img.text

        cov = tools.coverage(cur=("face", "top"))  # must not raise
        assert "1 face view" in cov
        assert "1/36" in cov                       # the (2,0) cell only


def test_collect_evidence_keeps_a_face_citation():
    """C1.6: `_collect_evidence` used to coerce `e["cell"]` with
    `tuple(int(c) for c in ...)`, which raised ValueError on "face" and
    silently dropped the citation (caught, logged, `continue`d — the
    answer shipped with one less piece of evidence and no error). A face
    address must resolve to `("face", name)` and keep a cached hunt."""
    with tempfile.TemporaryDirectory() as tmp:
        brain = Brain.__new__(Brain)
        brain.tools = _face_fixture(tmp)
        brain.question = "is there a logo?"
        brain.trace = _TraceStub()
        rec = {"cell": ("face", "top"), "find": "is there a logo?",
              "found": True, "report": "yes, on the top face", "crop": None,
              "frame": None, "transcript": None, "transcript_id": "t1"}
        brain._hunts = {(("face", "top"), "is there a logo?"): rec}

        bounce, records = brain._collect_evidence([{"cell": ["face", "top"]}])
        assert bounce is None
        assert records == [rec], "the face citation must survive, not be dropped"


def test_render_blocks_survive_a_face_cell():
    """C1.2: once `self.cell` is a face tuple (the agent's location IS the
    face after a move), `state_block`/`moves_block` must render an honest
    line instead of raising on `h, v = ("face", "top")`."""
    with tempfile.TemporaryDirectory() as tmp:
        tools = _face_fixture(tmp)
        cell = ("face", "top")

        state = render.state_block(3, cell, tools._h_bins, tools._v_elevs,
                                   1, 10, 2)
        assert "face" in state and "top" in state

        moves = render.moves_block(3, tools, cell, set(), nav=None)
        assert moves.startswith("MOVES")
        assert "face" in moves and "top" in moves

        # The reverse mix: standing at a NORMAL cell with a face already in
        # agent_seen — the ring math must skip the face tuple, not compare
        # "face" to an int (0 <= c[0] raised before the fix).
        moves = render.moves_block(3, tools, (4, 1),
                                   {(0, 1), ("face", "top")}, nav=None)
        assert moves.startswith("MOVES")


def test_sorted_addresses_mixes_cells_and_faces():
    """C1.5: `agent_seen` can hold both kinds; plain `sorted()` over mixed
    `[h, v]` / `["face", name]` lists raises comparing `int` to `str`."""
    out = _sorted_addresses({(0, 0), ("face", "top"), (1, 2)})
    assert [0, 0] in out and ["face", "top"] in out and [1, 2] in out


def main():
    test_gate_passes_ready_face_denies_others()
    test_face_step_resolves_through_view_at_coverage_gloss()
    test_collect_evidence_keeps_a_face_citation()
    test_render_blocks_survive_a_face_cell()
    test_sorted_addresses_mixes_cells_and_faces()
    print("OK test_brain_faces")


if __name__ == "__main__":
    main()
