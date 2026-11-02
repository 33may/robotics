#!/usr/bin/env python3
"""Cross-tier e2e (task-6, boundary test): Flow A over a real ray
transaction on the fake rig.

Extends the e2e flow harness (`test_flows_e2e.py`, task-11) rather than
rebuilding it, but drives the machine the way `test_flow_a.py`'s own unit
suite already does: a real `Supervisor` over a real `FakeRig`
(`inspection.ui.mock.start_mock`), commanded directly on its event queue
(`sup.events.put({"cmd": ...})`) instead of over a live websocket — that
queue IS what `ui/mock.py:start_mock`'s `pump()` forwards every bus command
onto, so a dict landing there is the identical dispatch path a browser click
takes, with none of the Chromium overhead `test_flows_e2e.py` already pays
for the click-approve-write chain. The Playwright-driven tests prove that
wiring; this file proves what sits BEHIND it once Stage B's move
transactions are live.

**One thing this file does NOT reuse from the sibling suite**:
`test_flow_a.py`'s `DirectPlanSupervisor` (also imported by this dir's own
`_mock_launcher.py`). That stub exists because, when it was written, the real
`plan_viewpoint` refused the demo survey/park pose outright in this sandbox.
Probed live while writing this suite (2026-09-17): the REAL, unpatched
`Supervisor` now plans and transacts the survey and every cell/face move
fine — `1b7bc52`/`c8cc85a`'s ray/cell-map rework fixed whatever the stub was
worked around. Using the real planner here is the point: a stubbed
`_plan_worker` never calls `plan_transaction` at all, so it would prove
nothing about ray transactions.

Run:
    p -m pytest vbti/inspection/tests/e2e/test_flow_a_rays_e2e.py -m e2e -q
    p vbti/inspection/tests/e2e/test_flow_a_rays_e2e.py
"""
from __future__ import annotations

import queue
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ASSERT_TIMEOUT = 90.0


def wait_for(cond, timeout=ASSERT_TIMEOUT, msg=""):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, f"timeout: {msg}"
        time.sleep(0.02)


def _bg_request(mover, target):
    """Run `mover.request(target)` on its own thread; returns (thread, out).

    `out` is filled with {"ok", "why"} once the thread finishes — the same
    shape `test_flow_a.py::test_the_movers_approval_beats_land_in_events_jsonl`
    already reads, reused here rather than reinvented.
    """
    out = {}

    def go():
        ok, why = mover.request(target)
        out["ok"], out["why"] = ok, why

    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th, out


# === Part 1 — a deterministic pin on brain/live.py's one ruled line ========
#
# The cross-tier flow below (Part 2) drives real transactions on a real
# FakeRig, and empirically the `adjusting`-vs-`executing` race
# `brain/live.py`'s step 2 loses is exactly that — a RACE. Probing it while
# writing this suite: forcing a slide leg onto a real transaction reproduced
# the mis-report on 1 of 3 runs and "could not be planned" (a different,
# geometry-sensitive refusal) on the other 2 — real evidence the bug exists,
# but not a rig I can commit as a reliable RED/GREEN gate. Per the brief's
# own fallback ("otherwise pin the line with a focused unit assertion"),
# this scripts the exact phase sequence `SupervisorMover.request()` sees —
# previewing -> adjusting -> idle, NEVER visiting "executing" at all — deaf
# to FakeRig's timing, so it is deterministic in either direction.


class _FakeSup:
    """Just enough of `Supervisor` for `SupervisorMover.request()`: a phase/
    target pair and a command queue nothing reads. `request()` touches
    nothing else on `sup`."""

    def __init__(self):
        self.phase, self.target = "idle", None
        self.events = queue.Queue()


def _minimal_run_dir():
    """A real, openable (never closed) run directory — `_captured_ids()`
    calls `Run.load(...).captured` on every `request()`, and a run with no
    `run.json` would raise before the phase check under test ever runs."""
    from inspection.record.writer import RunWriter
    from inspection.run.app import config_snapshot, viewsphere_method

    root = Path(tempfile.mkdtemp())
    RunWriter.create(root, run_id="pin-run", name="pin-run", source="live",
                     rig="fake", config=config_snapshot(),
                     view_methods=[viewsphere_method(0.3)], q_survey=[0.0] * 6)
    return root / "pin-run"


def _drive_scripted_adjusting_race(sup, cell, adjusting_hold=0.3):
    """The dispatcher's side of the race, entirely scripted: previewing then
    STRAIGHT to `adjusting` — the slide leg the real dispatcher also reaches
    in well under a millisecond (task-3-report.md) — held for
    `adjusting_hold` seconds, far longer than the mover's 50 ms poll
    (`brain/live.py:POLL_S`), so the poll cannot help but land on it. Ends in
    `idle`/no target, the honest "the move finished, nothing was captured"
    terminal state for a run dir with no steps in it.
    """
    # `SupervisorMover.request()` normalizes its target to a TUPLE
    # (`cell = tuple(cell)`) and compares it against `sup.target` by
    # equality — `Supervisor._normalize_target` does the same on the real
    # dispatcher, so `sup.target` must be the tuple form here too, or every
    # `t == cell` check below silently reads False and the mover times out
    # instead of ever seeing the race this test exists to script.
    cell = tuple(cell)
    time.sleep(0.05)
    sup.phase, sup.target = "planning", cell
    time.sleep(0.05)
    sup.phase, sup.target = "previewing", cell
    time.sleep(0.15)                      # let the mover reach its step-2 wait
    sup.phase, sup.target = "adjusting", cell     # never "executing" at all
    time.sleep(adjusting_hold)
    sup.phase, sup.target = "idle", None


def test_pin_brain_live_does_not_mistake_a_slide_leg_for_a_refusal():
    """`brain/live.py` step 2 must accept `adjusting` (a slide leg) as proof
    the move started, exactly like `executing` (the hop) — the ruled
    one-line fix this task pins (task-3-report.md's concern #1)."""
    from inspection.brain.live import SupervisorMover

    sup = _FakeSup()
    run_dir = _minimal_run_dir()
    mover = SupervisorMover(sup, run_dir, decider="ai")
    cell = [1, 0]

    threading.Thread(target=_drive_scripted_adjusting_race,
                     args=(sup, cell), daemon=True).start()
    ok, why = mover.request(cell)

    assert ok is False, why       # this run dir never captures anything either way
    assert "not approved" not in why, (
        "brain/live.py step 2 read a slide leg (adjusting) as a refusal — "
        f"the arm was moving and the mover just told the orchestrator it "
        f"was not: {why!r}")
    assert "adjusting" not in why, why
    assert "no frame was captured" in why, (
        "expected the honest 'nothing was captured' outcome once the "
        f"approval check correctly let `adjusting` through: {why!r}")


# === Part 2 — the cross-tier flow: survey, AI moves, a face, a fault =======


def _make_mover(sup, run_dir):
    from inspection.brain.live import SupervisorMover, make_beat_handler
    from inspection.brain.trace import TraceWriter
    from inspection.record.ai_writer import AIRunWriter

    ai = AIRunWriter.create(run_dir, orchestrator_model="test",
                            menu_id="brain-render", menu_hash="0" * 64)
    trace = TraceWriter(ai.dir)
    pub = _NullPub()
    mover = SupervisorMover(sup, run_dir, on_event=make_beat_handler(
        trace, ai, pub), decider="ai")
    return mover, ai, trace


class _NullPub:
    def __getattr__(self, name):
        return lambda *a, **kw: None


class _PhaseTrace(_NullPub):
    """Records every `status(phase=...)` call — the dispatcher's own phase
    announcement (`machine.py:_set_phase`), issued synchronously and
    unconditionally on every transition, never polled.

    This is the "bus phase trace" real observable (the same field
    `ui/publisher.py` republishes onto the `run/status` topic). It is
    deliberately NOT `sup.phase` sampling: `SupervisorMover`'s own 50 ms
    poll of `sup.phase` can and does skip transitions (`_wait_for_gate`'s
    docstring below has the measured evidence), so re-sampling `sup.phase`
    here to check for an `adjusting` leg would just be a second copy of the
    exact race this file already worked around once. Recording the
    dispatcher's own announcement instead is not sampled at all — every
    phase it ever set is in `self.phases`, in order, so a leg cannot be
    missed by bad luck the way a poll can. Used as the machine's own `pub`
    (never the mover's beat-handler pub, which stays a plain `_NullPub`),
    so the main flow test can prove a move genuinely ran multiple legs
    instead of only checking its outcome.
    """

    def __init__(self):
        self.phases = []

    def status(self, **kw):
        self.phases.append(kw.get("phase"))


def _wait_for_gate(run_dir, n0, timeout=ASSERT_TIMEOUT, msg=""):
    """Block until the mover's OWN thread has written its `awaiting_approval`
    beat (index >= n0) — proof it already observed `previewing` itself,
    not merely that `sup.phase` did.

    `test_flow_a.py::test_the_movers_approval_beats_land_in_events_jsonl`
    documents exactly why this matters and is reused here rather than
    reinvented: `_wait` polls `sup.phase` every 50 ms
    (`brain/live.py:POLL_S`), and confirming the instant an OUTSIDE watcher
    sees `previewing` can execute before the mover's own poll ever does —
    it then never observes `previewing` at all and races straight through
    to `idle` once the (successful) move finishes, misreading a real
    success as "could not be planned" (empirically reproduced while writing
    this suite; a same-process racing confirm reacts far faster than any
    human ever would, which is why this class of race is invisible in
    production but was hit immediately here). Watching the mover's own beat
    instead of the phase sidesteps it structurally.
    """
    from inspection.record.run import Run

    wait_for(lambda: len(Run.load(run_dir).events) > n0 and
             any(e.kind == "awaiting_approval"
                for e in Run.load(run_dir).events[n0:]),
             timeout=timeout, msg=msg)


def _ai_move(sup, mover, run_dir, target, timeout=ASSERT_TIMEOUT):
    """One full AI-decider turn: request, wait for the mover's OWN approval
    gate (not `sup.phase` — see `_wait_for_gate`), approve, wait for the
    real outcome. Returns (ok, why)."""
    from inspection.record.run import Run

    n0 = len(Run.load(run_dir).events)
    th, out = _bg_request(mover, target)
    _wait_for_gate(run_dir, n0, timeout=timeout, msg=f"{target} awaiting_approval")
    sup.events.put({"cmd": "view/confirm", "target": target})
    th.join(timeout)
    assert not th.is_alive(), f"mover.request({target}) never returned"
    # The atomicity gate (brain/live.py's `request()`): whatever the mover
    # reports, it must not return while the machine is still mid-transaction.
    wait_for(lambda: sup.phase in ("idle", "fault"), timeout=timeout,
             msg=f"{target} settled")
    return out["ok"], out["why"]


def test_flow_a_ray_transaction_with_a_face_and_a_scripted_fault():
    """One Flow-A question run, fake rig, end to end: survey, two AI cell
    moves (opening the boxguard face gate), an AI face request that plans and
    executes as a real 3-leg transaction and captures, then a scripted
    mid-move fault on a further move — leaving the run alive and answerable.

    Assertions are on real observables only: `Run.load(...)`'s step records
    and events, `SupervisorMover.nav()` — the live picture `brain/live.py`
    itself hands the orchestrator — and the dispatcher's own phase
    announcements (`_PhaseTrace`, the "bus phase trace" observable) — never
    `Supervisor` internals.

    "Sweep step" (the brief's Step 1 wording) is these two AI-decider cell
    moves: Flow A's `ask` path never touches `run/collect.py`'s
    `SweepDriver`/sweep planner at all (that is Flow B's own driver, wired
    only by `ui/mock.py:start_mock_collect`/`run/app.py:collect`) — the only
    thing choosing an address here is `SupervisorMover.request()`, same as
    the face move below, just with `decider="ai"` over a plain grid cell.
    """
    from inspection.record.run import Run
    from inspection.ui import mock

    class _StubBus:
        def commands(self):
            return iter(())

    run_root = Path(tempfile.mkdtemp())
    run_dir = run_root / "0917-rays"
    # `_PhaseTrace` (not a plain `_NullPub`) as the MACHINE's own pub, so the
    # face move's leg sequence can be read off the dispatcher's own
    # announcements below rather than only inferred from its outcome.
    phase_trace = _PhaseTrace()
    sup = mock.start_mock(_StubBus(), phase_trace, run_dir)
    try:
        mover, ai, trace = _make_mover(sup, run_dir)

        ok, why = _ai_move(sup, mover, run_dir, "survey")
        assert ok, why

        def _reachable_unvisited():
            nav = mover.nav()
            return sorted(nav["reachable"] - nav["visited"])

        cell1 = list(_reachable_unvisited()[0])
        ok, why = _ai_move(sup, mover, run_dir, cell1)
        assert ok, why

        cell2 = list(_reachable_unvisited()[0])
        assert cell2 != cell1
        ok, why = _ai_move(sup, mover, run_dir, cell2)
        assert ok, why

        # Two fused boxguard fits with a small normal delta open the face
        # gate (`machine.py:_update_face_gate`) — the object here never
        # moves, so two real moves are always enough.
        nav = mover.nav()
        faces = nav["faces"]
        ready = [f for f, st in faces.items() if st["status"] == "ready"]
        assert ready, f"no face ready after two moves: {faces}"
        face = ["face", ready[0]]

        n_face0 = len(phase_trace.phases)
        ok, why = _ai_move(sup, mover, run_dir, face)
        assert ok, why

        # --- the face move genuinely ran as a multi-leg ray transaction ----
        # `_ai_move` above only proves the OUTCOME (approved, captured,
        # atomic) — this proves the MECHANISM the brief names ("3-leg
        # transaction"): at least one `adjusting` leg (a slide) actually
        # ran, not just the `executing` hop. Read off `_PhaseTrace` (the
        # dispatcher's own, unsampled announcements — see its docstring for
        # why this is the least-flaky of the options the review named: a
        # fresh `sup.phase` poll here would just reintroduce the exact race
        # `_wait_for_gate` exists to avoid).
        face_phases = phase_trace.phases[n_face0:]
        assert "adjusting" in face_phases, (
            f"the face move never announced an adjusting leg — not a "
            f"multi-leg transaction: {face_phases}")

        # --- step record carries {method, address} unchanged ---------------
        run = Run.load(run_dir)
        step = run.at(face)
        assert step is not None, f"{face} never landed a captured step"
        assert step.record.view.method == "face"
        assert step.record.view.address == face

        # --- the face finding's theta reaches the brain's live picture -----
        nav = mover.nav()
        st = nav["faces"][face[1]]
        assert st["status"] == "visited"
        assert st["off_angle_deg"] is not None, (
            "the resolved face's off-angle never reached "
            "SupervisorMover.nav() — the one thing brain/loop.py's "
            "FACES block and _gate read")
        # KNOWN GAP, ruled deferred to STAGE C (flagged, not fixed here —
        # see task-6 report): the same `RayResolution` also carries `shift`
        # (machine.py's `_face_find`), but `Supervisor._face_states()` only
        # ever forwards `theta` (as `off_angle_deg`) into
        # `nav()`/`_staple_state`'s FACES text. No brain-observable path
        # carries `shift` today. The ruling is that shift-to-brain is Stage
        # C prompt-reality work (machine.py's `_face_states()`/
        # `_view_state()` and brain/loop.py's FACES render both need it, and
        # both are outside this task's permitted edit — brain/live.py's one
        # ruled line only). Pinned here as an explicit absence rather than
        # silently dropped, so Stage C's own task closes this assertion
        # rather than discovering the gap cold.
        assert "shift" not in st, (
            "brain/live.py's nav()/_face_states() now carries 'shift' — "
            "task-6's report is stale, update it and this comment")

        # --- a scripted mid-move fault leaves the run alive and answerable -
        fault_target = list(_reachable_unvisited()[0])
        real_move = sup.rig.move
        calls = {"n": 0}

        def _fault_2nd(path, world="live"):
            calls["n"] += 1
            if calls["n"] == 2:      # let leg 1 run for real, fault leg 2
                raise RuntimeError("scripted mid-move fault (task-6 e2e)")
            return real_move(path, world=world)

        n0 = len(Run.load(run_dir).events)
        th, out = _bg_request(mover, fault_target)
        _wait_for_gate(run_dir, n0, msg="fault-target awaiting_approval")
        sup.rig.move = _fault_2nd
        sup.events.put({"cmd": "view/confirm", "target": fault_target})
        th.join(ASSERT_TIMEOUT)
        assert not th.is_alive(), "the atomicity gate hung on a mid-move fault"
        assert out["ok"] is False and "faulted" in out["why"], out

        wait_for(lambda: sup.phase == "fault", msg="fault phase")
        run = Run.load(run_dir)
        assert any(e.kind == "fault" for e in run.events)
        assert run.validate().ok, run.validate().problems   # alive mid-fault

        sup.events.put({"cmd": "run/exit"})
        wait_for(lambda: sup.phase == "done", msg="exit out of fault")
        wait_for(lambda: (run_dir / "manifest.json").exists(),
                 msg="writer closed")

        run = Run.load(run_dir)
        assert run.validate().ok, run.validate().problems   # alive after exit

        # answerable: the SAME ai session the approval chain landed in can
        # still take a verdict — production `Brain._commit_answer`, no SDK.
        from inspection.brain.loop import Brain
        from inspection.eyes.tools import ViewTools

        brain = Brain.__new__(Brain)
        brain.run = run
        brain.tools = ViewTools(run)
        brain.trace = trace
        brain.ai = ai
        brain.cell = tuple(face)
        brain.agent_seen = {tuple(face)}
        brain.answer = None
        brain._commit_answer(
            {"verdict": "yes", "reasoning": "task-6 e2e proof of answerability",
             "evidence": "scripted"}, [])

        run = Run.load(run_dir)
        assert run.ai and run.ai[0].answer is not None
        assert run.ai[0].answer.verdict == "yes"
        assert run.validate().ok, run.validate().problems
    finally:
        # No `plan_transaction`/rig monkeypatch to restore on the module
        # level (unlike test_machine.py's forced-slide-out fixtures): the
        # rig patch above lives on THIS `sup.rig` instance only, and the
        # real planner is never touched — that's the point of this suite.
        sup.request_shutdown()


def main():
    test_pin_brain_live_does_not_mistake_a_slide_leg_for_a_refusal()
    print("OK test_pin_brain_live_does_not_mistake_a_slide_leg_for_a_refusal")
    test_flow_a_ray_transaction_with_a_face_and_a_scripted_fault()
    print("OK test_flow_a_ray_transaction_with_a_face_and_a_scripted_fault")
    print("OK test_flow_a_rays_e2e")


if __name__ == "__main__":
    main()
