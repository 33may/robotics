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


def start(tmp_path, monkeypatch, name="0709-mock"):
    """A running mock Supervisor over a fresh run dir. Returns (sup, run_dir).

    Everything the composition root wires — writer, rig, settle, ask handler —
    is the real one; only the planner and the publisher are stubbed.
    """
    from inspection.ui import mock

    monkeypatch.setattr(mock, "Supervisor", DirectPlanSupervisor)
    run_dir = tmp_path / name
    return mock.start_mock(StubBus(), StubPub(), run_dir), run_dir


def survey(sup, approve=True):
    """Drive one survey turn through the state machine, operator-style."""
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    if approve:
        sup.events.put({"cmd": "view/confirm", "target": "survey"})
        wait_for(lambda: sup.phase == "idle" and sup.target is None,
                 msg="settled")


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


def test_the_movers_approval_beats_land_in_events_jsonl(tmp_path, monkeypatch):
    """The approval chain is a RECORD, not a log line.

    "The brain may request, only a human confirms" is this project's central
    safety property, and `events.jsonl` is where it is provable after the
    fact. Drives the real `SupervisorMover` (the orchestrator's own path into
    the queue) against the real machine, with the real beat handler the ask
    handler installs.
    """
    import threading

    from inspection.brain.live import SupervisorMover, make_beat_handler
    from inspection.brain.trace import TraceWriter
    from inspection.record.ai_writer import AIRunWriter

    sup, run_dir = start(tmp_path, monkeypatch)
    ai = AIRunWriter.create(run_dir, orchestrator_model="test",
                            menu_id="brain-render", menu_hash="0" * 64)
    mover = SupervisorMover(sup, run_dir,
                            on_event=make_beat_handler(TraceWriter(ai.dir),
                                                       ai, StubPub()))
    out = {}
    th = threading.Thread(target=lambda: out.update(
        zip(("ok", "why"), mover.request("survey"))), daemon=True)
    th.start()
    # Approve only once the mover is actually AT the gate — it polls the
    # phase every 50 ms, so a confirm sent the instant `previewing` appears
    # can be executed before the mover ever observes it. The beat it writes
    # on arrival is the honest handshake, and it is the thing under test.
    wait_for(lambda: any(e.kind == "awaiting_approval"
                         for e in Run.load(run_dir).events),
             msg="mover reached the approval gate")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    th.join(120)
    assert out.get("ok"), out

    sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")

    run = Run.load(run_dir)
    kinds = [e.kind for e in run.events]
    assert kinds[:3] == ["requested", "awaiting_approval", "approved"]
    assert "captured" in kinds
    # The captured beat names the step it produced — the join between the
    # approval chain and the geometry it authorised.
    beat = [e for e in run.events if e.kind == "captured"][-1]
    assert beat.step_id == run.survey.record.step_id
    # M2: an AI-driven request is distinguishable from a clicked one on disk.
    assert run.survey.view_state.decider == "ai"
    assert run.validate().ok, run.validate().problems


def test_a_refused_view_is_a_rejected_step_on_disk(tmp_path, monkeypatch):
    """The plane gate's row of the write-path table: a settle that refuses
    the view writes an explicit `rejected` step — never a silent gap in the
    id space, and never a cell marked visited."""
    from inspection.run import machine as machine_mod
    from inspection.run.settle import SettleResult

    sup, run_dir = start(tmp_path, monkeypatch)
    monkeypatch.setattr(machine_mod, "settle_capture", lambda *a, **kw:
                        SettleResult(ok=False, detail="view rejected: table "
                                     "plane at z=90 mm", view=None, seg=None,
                                     npts=0, dropped=0))
    survey(sup)
    assert sup.survey_state != "visited", "a refused view was marked visited"
    sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")

    run = Run.load(run_dir)
    step = run.step(0)
    assert step.record.outcome == "rejected"
    assert "table plane" in step.record.detail
    assert step.record.geometry is None      # nothing was fused
    assert run.captured == []                # and it is not a captured view
    assert run.record.steps == [0], "the id space skipped the refusal"
    assert run.validate().ok, run.validate().problems


def test_a_crashing_capture_is_a_failed_step_on_disk(tmp_path, monkeypatch):
    """A rig that raises mid-capture ends the turn as `failed`, with the
    reason on disk, and the run carries on."""
    from inspection.run import machine as machine_mod

    sup, run_dir = start(tmp_path, monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("synthetic bad view")

    monkeypatch.setattr(machine_mod, "settle_capture", boom)
    survey(sup)
    sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")

    run = Run.load(run_dir)
    assert run.step(0).record.outcome == "failed"
    assert "synthetic bad view" in run.step(0).record.detail
    assert run.validate().ok, run.validate().problems


def test_a_failing_view_state_still_lands_the_run_in_idle(tmp_path, monkeypatch):
    """The fused write is built INSIDE `_write`'s guard, thunk and all.

    `_view_state` walks the whole shell — reachability, camera poses, the
    accumulator's AABB — and used to be evaluated as an ARGUMENT to `_write`,
    i.e. before the guard was entered. Anything it threw escaped
    `_on_settle_done` (the dispatcher logs and carries on, so nothing dies
    loudly), and the phase never came back: the machine sat in `fusing`
    forever with the arm on the table and no way out but Ctrl-C.
    """
    from inspection.run.machine import Supervisor

    sup, run_dir = start(tmp_path, monkeypatch)

    def boom(self, step_id, target):
        raise RuntimeError("synthetic view-state failure")

    monkeypatch.setattr(Supervisor, "_view_state", boom)
    survey(sup)                            # waits for idle — the wedge is here
    assert sup.phase == "idle" and sup.target is None
    assert sup.survey_state == "visited"   # the move itself was fine

    sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")

    run = Run.load(run_dir)
    step = run.step(0)
    # The capture stands; only the fused half was lost, so the step is
    # `captured` — never left claiming a phase it never finished.
    assert step.record.phase == "captured"
    assert step.record.outcome == "captured"
    assert step.view_state is None
    assert run.validate().ok, run.validate().problems


def test_a_stopped_move_is_an_event_and_leaves_no_step(tmp_path, monkeypatch):
    """`run/stop` mid-move: the arm halted before any camera was pointed
    anywhere, so the record gets an event and NO step directory."""
    sup, run_dir = start(tmp_path, monkeypatch)
    sup.rig.speed = 0.05                 # ~10 s of travel — time to stop it
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    wait_for(lambda: sup.phase == "executing", msg="executing")
    sup.events.put({"cmd": "run/stop"})
    wait_for(lambda: sup.phase == "idle", msg="stopped -> idle")

    run = Run.load(run_dir)
    stopped = [e for e in run.events if e.kind == "stopped"]
    assert stopped and "survey" in stopped[-1].detail
    assert stopped[-1].step_id is None
    assert not (run_dir / "steps").exists(), "a halted move left a step dir"
    sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    # Leave no run thread mid-write when the test ends: the mock closes its
    # writer on the way out, and the manifest is that receipt.
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")


def test_a_faulted_executor_is_an_event(tmp_path, monkeypatch):
    """The executor refusing (safety mode, a halted arm) is a run-level fault
    event; the machine parks in `fault` and only Exit gets out."""
    sup, run_dir = start(tmp_path, monkeypatch)
    monkeypatch.setattr(sup.rig, "move", lambda path: (_ for _ in ()).throw(
        RuntimeError("safety mode changed mid-path")))
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    wait_for(lambda: sup.phase == "fault", msg="fault")

    run = Run.load(run_dir)
    faults = [e for e in run.events if e.kind == "fault"]
    assert faults and "safety mode" in faults[-1].detail
    assert not (run_dir / "steps").exists()
    sup.events.put({"cmd": "run/exit"})
    wait_for(lambda: sup.phase == "done", msg="exit out of fault")
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")


def test_a_live_subagent_transcript_is_readable_through_the_run(tmp_path,
                                                                monkeypatch):
    """The eyes tier's transcripts are visible through the ONE read door.

    A transcript `Run.ai[].transcripts` cannot return is a transcript nobody
    will ever read — and the validator must be the thing that says so, which
    is why it now rejects any json in `transcripts/` it does not recognise.
    """
    from inspection.eyes.agents.inspect_agent import INSPECT
    from inspection.eyes.cognition import stub_cognition
    from inspection.eyes.store import FindingWriter, RunStore
    from inspection.eyes.tools import ViewTools
    from inspection.record.ai_writer import AIRunWriter
    from inspection.view.grid import DEFAULT_R, H_BINS, V_ELEVATIONS

    sup, run_dir = start(tmp_path, monkeypatch)
    survey(sup)
    sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")

    run = Run.load(run_dir)
    ai = AIRunWriter.create(run_dir, orchestrator_model="test",
                            menu_id="brain-render", menu_hash="0" * 64)
    store = RunStore.create(ai.dir, h_bins=H_BINS, v_elevs=V_ELEVATIONS,
                            r=DEFAULT_R)
    cog = stub_cognition()
    writer = FindingWriter(store, ai=ai)
    tools = ViewTools(run, writer=writer)
    finding = INSPECT.run(tools, cog.verbs, writer, cog.vlm,
                          task="what is this?", view_rec=tools._views()[0])
    assert finding.transcript_id == "t000"

    fresh = Run.load(run_dir)
    trs = fresh.ai[0].transcripts
    assert len(trs) == 1, "the read door cannot see the session's transcript"
    t = trs[0]
    assert t.transcript_id == "t000" and t.kind == "inspect"
    assert t.step_id == 0 and t.model == "StubVlm"
    # Declared relative to the AI session, and they resolve there.
    assert t.artifacts and all(
        (ai.dir / a).exists() for a in t.artifacts), t.artifacts
    assert all(a.startswith("artifacts/") for a in t.artifacts)
    assert fresh.validate().ok, fresh.validate().problems


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
    ai = brain.ai.dir.relative_to(w.dir).as_posix()
    hit = {"cell": [3, 1], "find": "the wordmark", "found": True,
           "report": "yes, centred", "crop": f"{ai}/artifacts/001_03_crop.png",
           "frame": f"{ai}/artifacts/001_frame.png",
           "transcript": "transcripts/t000.json", "transcript_id": "t000"}
    hunts = [
        hit,
        # No capture at that cell and no image: not citable as evidence.
        {"cell": [9, 0], "find": "a seam", "found": False, "report": "nothing",
         "crop": None, "frame": None, "transcript": None,
         "transcript_id": None},
        # The planner re-citing a hunt it already made (the "ship with the gap
        # on record" path) must not double the evidence.
        hit,
    ]
    rec = brain._answer_record(
        {"verdict": "yes", "reasoning": "the wordmark is legible",
         "evidence": "one prose blob"}, hunts)
    validate_for_write(AnswerRecord, rec)      # raises if the shape is wrong
    assert rec["evidence"] == ["one prose blob"]
    assert len(rec["evidence_images"]) == 1
    img = rec["evidence_images"][0]
    assert img["step_id"] == 1 and img["transcript_id"] == "t000"
    # Relative to the AI session, as the schema declares it — not to the run.
    assert img["artifact"] == "artifacts/001_03_crop.png"
    assert rec["step_ids"] == [1] and rec["transcript_ids"] == ["t000"]

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
