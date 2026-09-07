#!/usr/bin/env python3
"""Flow A end to end on the FakeRig: one approved turn writes a valid run.

The whole live write path, minus the wire: a real `Supervisor` over a real
`FakeRig`, the real settle pipeline, the real `RunWriter`. Only the publisher
is a stub — rendering poses drags pinocchio/hppfcl geometry into a test that
is about bytes on disk, and this sandbox's hppfcl trips over it
('Convex' object has no attribute 'halfSide').

Run: p -m pytest inspection/tests/test_flow_a.py
"""
import time

import numpy as np

from inspection.record.run import Run
from inspection.run.machine import Supervisor


class StubBus:
    """A bus that never delivers a command: the test drives `sup.events`
    directly, so the mock's command pump has nothing to pump."""

    def commands(self):
        return iter(())


class StubPub:
    """Records every publisher call, renders nothing."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*a, **kw):
            self.calls.append((name, a, kw))
        return _record


class DirectPlanSupervisor(Supervisor):
    """`_plan_worker` answers with a straight two-waypoint path.

    Not a shortcut around the state machine — every phase, event, generation
    check and worker handoff still runs; only OMPL is skipped. It has to be:
    in this checkout `plan_viewpoint` refuses the demo park pose outright
    ("no collision-free IK branch" — the taught `cell_plane_back` keep-out
    swallows it), which is exactly why `test_machine`'s cases time out here
    too. Planning is not what this test is about; the write path is.
    """

    def _plan_worker(self, gen, target):
        q = self.rig.q()
        goal = self.q_survey if target == "survey" else q
        self.events.put({"ev": "plan_done", "gen": gen, "target": target,
                         "path": [q, np.asarray(goal, float)], "detail": ""})


def wait_for(cond, timeout=120.0, msg=""):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, f"timeout: {msg}"
        time.sleep(0.02)


def test_one_approved_turn_writes_schema_run(tmp_path, monkeypatch):
    from inspection.ui import mock

    monkeypatch.setattr(mock, "Supervisor", DirectPlanSupervisor)
    run_dir = tmp_path / "0709-mock"
    sup = mock.start_mock(StubBus(), StubPub(), run_dir)
    try:
        sup.events.put({"cmd": "view/request", "target": "survey"})
        wait_for(lambda: sup.phase == "previewing", msg="previewing")
        sup.events.put({"cmd": "view/confirm", "target": "survey"})
        wait_for(lambda: sup.phase == "idle" and sup.survey_state == "visited",
                 msg="survey settled")
    finally:
        sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    # `close()` is the last thing the mock's run thread does — the manifest is
    # its receipt.
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")

    run = Run.load(run_dir)
    assert run.provenance == "native", "the run read back as legacy"
    assert run.record.source == "live" and run.record.rig == "fake"

    survey = run.survey
    assert survey is not None, "no step 0 on disk"
    assert survey.record.phase == "fused" and survey.record.outcome == "captured"
    assert survey.record.view.address is None
    assert survey.record.geometry.fused_points > 0
    assert survey.rgb() is not None, "the capture wrote no rgb.png"

    vstate = survey.view_state
    assert vstate is not None and vstate.candidates, "no view_state candidates"
    assert vstate.decider == "operator", vstate.decider

    # The old layout is DEAD: no NNN/ capture dirs, no run-root fused arrays.
    assert not (run_dir / "000").exists()
    assert not (run_dir / "fused_cloud.npy").exists()
    assert (run_dir / "fused" / "cloud.npy").exists()

    rep = run.validate()
    assert rep.ok, rep.problems


def _two_step_run(root):
    """A minimal native run: the survey plus one captured cell."""
    import numpy as np

    from inspection.record.schema import ViewState
    from inspection.record.writer import RunWriter
    from inspection.run.app import config_snapshot, viewsphere_method

    T4 = np.eye(4).tolist()
    w = RunWriter.create(root, run_id="0709-ans", name="ans", source="live",
                         rig="fake", config=config_snapshot(),
                         view_methods=[viewsphere_method(0.3)],
                         q_survey=[0.0] * 6)
    for address in (None, [3, 1]):
        sid, sdir = w.begin_step({"method": "vs", "address": address})
        (sdir / "rgb.png").write_bytes(b"not really a png")
        w.write_capture(sid, t_captured=1.0, joints_rad=[0.0] * 6,
                        T_base_flange=T4, T_base_cam=T4, rgb_rotation_deg=0)
        w.write_fused(sid, dict(offered=1, kept=1, dropped=0, fused_points=1,
                                source="depth"),
                      ViewState(step_id=sid, candidates=[]))
    return w


def test_the_brains_answer_validates_as_an_answer_record(tmp_path):
    """The AI tier's write boundary: hunt records in, `AnswerRecord` out.

    The loop's runtime answer carries `evidence` as one prose blob and
    `evidence_images` as raw hunt dicts — neither is what the schema declares.
    `_answer_record` is where the two meet, and a hunt with no image is
    dropped rather than faked (it stays in the trace).
    """
    from inspection.brain.loop import Brain
    from inspection.eyes.cognition import stub_cognition
    from inspection.record.run import Run
    from inspection.record.schema import AnswerRecord, validate_for_write

    w = _two_step_run(tmp_path)
    cog = stub_cognition()
    brain = Brain(w.dir, "is there a logo?", vlm=cog.vlm, verbs=cog.verbs)
    hunts = [
        {"cell": [3, 1], "find": "the wordmark", "found": True,
         "report": "yes, centred", "crop": "eyes/frames/a_01.png",
         "frame": "eyes/frames/a.png", "transcript": "transcripts/f000.json"},
        # No capture at that cell and no image: not citable as evidence.
        {"cell": [9, 0], "find": "a seam", "found": False, "report": "nothing",
         "crop": None, "frame": None, "transcript": None},
    ]
    rec = brain._answer_record(
        {"verdict": "yes", "reasoning": "the wordmark is legible",
         "evidence": "one prose blob"}, hunts)
    validate_for_write(AnswerRecord, rec)      # raises if the shape is wrong
    assert rec["evidence"] == ["one prose blob"]
    assert len(rec["evidence_images"]) == 1
    img = rec["evidence_images"][0]
    assert img["step_id"] == 1 and img["transcript_id"] == "f000"
    assert img["artifact"] == "eyes/frames/a_01.png"
    assert rec["step_ids"] == [1]

    brain.ai.answer(rec)
    run = Run.load(w.dir)
    assert run.ai[0].answer.verdict == "yes"
    # Everything the AI session produced lives under its own seq — trace,
    # store and verdict together.
    assert (run.ai[0].dir / "answer.json").exists()
    assert brain.trace.path == run.ai[0].dir / "trace.jsonl"
    assert brain.store.path == run.ai[0].dir
    assert run.validate().ok, run.validate().problems


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from inspection.ui import mock as _mock

    class _Patch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_one_approved_turn_writes_schema_run(Path(tempfile.mkdtemp()), _Patch())
    test_the_brains_answer_validates_as_an_answer_record(
        Path(tempfile.mkdtemp()))
    print("OK test_flow_a")
    del _mock
