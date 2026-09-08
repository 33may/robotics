#!/usr/bin/env python3
"""Flow B: `rank_candidates`/`joint_l1` as pure logic, then the sweep driver
end to end on the FakeRig — same write path as Flow A (test_flow_a.py),
same stub-planner workaround for this sandbox's taught keep-outs.

Run: p -m pytest inspection/tests/test_collect.py
"""
import threading
import time

import numpy as np
import pytest

from inspection.brain.live import make_event_sink
from inspection.motion.plan import DEMO_PARK
from inspection.record.run import Run
from inspection.record.writer import RunWriter
from inspection.run.app import config_snapshot, viewsphere_method
from inspection.run.collect import SweepDriver, joint_l1, rank_candidates
from inspection.run.rigs import FakeRig
from inspection.run.segmenter import ObjectSegmenter
from inspection.tests.test_flow_a import DirectPlanSupervisor, StubPub, wait_for


def test_ranking_picks_cheapest_and_skips_unplannable():
    costs = {(0, 0): 3.0, (1, 0): 1.0, (2, 0): None}
    ranked = rank_candidates([(0, 0), (1, 0), (2, 0)],
                             planner=lambda c: costs[c])
    assert ranked == [(1, 0), (0, 0)]            # None never ranks


def test_joint_l1():
    path = np.array([[0.0] * 6, [1.0] + [0.0] * 5, [1.0, 2.0] + [0.0] * 4])
    assert joint_l1(path) == pytest.approx(3.0)


def test_joint_l1_single_waypoint_is_free():
    assert joint_l1(np.zeros((1, 6))) == 0.0


class AutoApprove(threading.Thread):
    """The human half of the gate, played by a thread that always says yes.

    Confirms once per generation — exactly what a person clicking "approve"
    once per preview does — so a `SweepDriver` can walk an entire shell
    unattended in a test while the real approval gate
    (`Supervisor._on_view_confirm`) still runs on every single step.

    `settle` holds the preview open briefly before confirming it. Without
    it this stub races `SupervisorMover._wait`'s own 50 ms poll
    (`brain/live.py:POLL_S`): the driver's stub planner resolves a plan
    synchronously, so a confirm sent the instant "previewing" is observed
    can clear it again inside the SAME 50 ms window the mover is sampling
    at — a real operator, planner and robot are never this fast, so the
    race never shows up outside a test this synthetic.
    """

    def __init__(self, sup, poll=0.02, settle=0.2):
        super().__init__(name="auto-approve", daemon=True)
        self.sup, self.poll, self.settle = sup, poll, settle
        self.stop_event = threading.Event()
        self._done_gen = None

    def run(self):
        while not self.stop_event.is_set():
            if self.sup.phase == "previewing" and self.sup.gen != self._done_gen:
                gen = self.sup.gen
                time.sleep(self.settle)
                if self.stop_event.is_set():
                    return
                self._done_gen = gen
                self.sup.events.put({"cmd": "view/confirm",
                                     "target": self.sup._target_json(self.sup.target)})
            time.sleep(self.poll)


def _stub_planner(world, ik, q_now, cell):
    """Stand-in for `SweepDriver`'s default planner.

    The taught `cell.yaml` keep-outs make the demo park pose collide (see
    `test_flow_a.py:DirectPlanSupervisor`), so real `plan_to_cell` refuses
    from wherever this sandbox's `FakeRig` starts — exactly the workaround
    `test_flow_a.py` uses for the Supervisor's own planner, applied here to
    the driver's independent ranking planner. Cost is deterministic (by
    cell address) so ranking order is reproducible without needing a real
    path.
    """
    h, v = cell
    return None, float(h + 10 * v)


def _stub_segmenter():
    from inspection.eyes.verbs_local import StubBackend
    return ObjectSegmenter(backend=StubBackend())


class StubSup:
    """The handful of attributes `SweepDriver.run` reads, and nothing else.

    Deliberately not a `Supervisor`: these two tests are about how the
    driver's own loop ENDS, and a real dispatcher would put the state
    machine, the taught keep-outs and a whole approval handshake between the
    test and the three lines under test.
    """

    class _Acc:
        def aabb(self, pct=None):
            return None                  # no object yet -> nothing to mirror

    class _Rig:
        def q(self):
            return np.zeros(6)

    def __init__(self, cells: int = 8):
        self.phase, self.target = "idle", None
        self.sphere, self.seed = None, 0
        self.acc, self.rig = self._Acc(), self._Rig()
        self.visited, self.blocked = set(), set()
        self._reach = {(h, 0): 0.0 for h in range(cells)}


class CountingMover:
    """A mover that approves the survey and refuses every cell.

    The shape of a run that has gone wrong: `SupervisorMover` refuses
    instantly while the machine is in `fault`, so nothing is ever visited and
    the candidate set never shrinks — which is exactly the state the driver
    used to re-rank forever.
    """

    def __init__(self, on_request):
        self.calls, self.asked, self.on_request = 0, [], on_request

    def request(self, cell):
        self.calls += 1
        self.asked.append(cell)
        self.on_request(self.calls, cell)
        return (True, "reached and captured") if cell == "survey" \
            else (False, "refused — the arm is faulted")


def test_a_faulted_run_ends_the_sweep(tmp_path):
    """`phase == "fault"` is terminal for the whole run: no request can ever
    be accepted again, so ranking on would spin `plan_to_cell` and `rig.q()`
    against hardware the teardown is already closing."""
    sup = StubSup()
    driver = SweepDriver(sup, tmp_path, planner=_stub_planner)
    # The fault lands MID-PASS (during the 3rd cell of a ranked pass of 8) —
    # the case a per-pass-only check would walk right past.
    driver.mover = CountingMover(
        lambda n, cell: setattr(sup, "phase", "fault") if n == 4 else None)

    driver.start()
    driver.join(10)
    assert not driver.is_alive(), "the sweep never noticed the fault"
    assert driver.finished is False, "a run cut short must not read completed"
    # survey + 3 cells: it checks between candidates, so it stops inside the
    # ranked pass rather than working through the remaining five.
    assert driver.mover.calls == 4, driver.mover.asked


def test_stop_ends_the_sweep(tmp_path):
    """`stop()` is what the composition root's teardown calls before the rig
    is closed — the driver must be gone before the RTDE interface is."""
    sup = StubSup()
    driver = SweepDriver(sup, tmp_path, planner=_stub_planner)
    driver.mover = CountingMover(
        lambda n, cell: driver.stop() if n == 2 else None)

    driver.start()
    driver.join(10)
    assert not driver.is_alive(), "the sweep ignored stop()"
    assert driver.finished is False
    assert driver.mover.calls == 2, driver.mover.asked


def test_sweep_driver_visits_every_reachable_cell(tmp_path):
    run_dir = tmp_path / "0709-sweep"
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_survey + np.radians([0, 0, 0, 0, 0, 8]))
    pub = StubPub()

    writer = RunWriter.create(
        run_dir.parent, run_id=run_dir.name, name=run_dir.name,
        source="data-engine", rig="fake", question=None,
        config=config_snapshot({"segmenter": "stub"}),
        view_methods=[viewsphere_method()],
        q_survey=[float(v) for v in q_survey])

    sup = DirectPlanSupervisor(rig, pub, writer, q_survey,
                               segmenter=_stub_segmenter())
    events = []
    # The REAL sink the composition roots install (`run/app.py:collect`,
    # `ui/mock.py:start_mock_collect`): the sweep's approval chain has to
    # land in `events.jsonl`, and a test that only collected the beats in a
    # list would prove nothing about the file that carries the safety
    # property.
    to_disk = make_event_sink(writer, pub, label="sweep")

    def on_event(state, **kw):
        events.append((state, kw))
        to_disk(state, **kw)

    driver = SweepDriver(sup, run_dir, on_event=on_event,
                         planner=_stub_planner)

    def run_and_close():
        # Mirrors `run/app.py:collect`'s own teardown (minus the hardware):
        # the fused cloud beside the steps, then a closed run.json — status
        # read off `driver.finished`, not off whether anything answered
        # (this run asks no question). `sup.run()` here plays the part
        # `Supervisor.run()` plays inside `collect`'s composition.
        try:
            sup.run()
        finally:
            driver.stop()
            driver.join(5.0)
            if sup.acc is not None and len(sup.acc.points):
                d = writer.dir / "fused"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "cloud.npy", sup.acc.points)
                np.save(d / "colors.npy", sup.acc.colors)
            writer.close("completed" if driver.finished else "aborted")

    th = threading.Thread(target=run_and_close, daemon=True)
    th.start()

    approver = AutoApprove(sup)
    approver.start()

    driver.start()

    wait_for(lambda: driver.finished, timeout=120.0, msg="sweep finished")
    driver.join(10)
    assert not driver.is_alive()

    approver.stop_event.set()
    sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")

    run = Run.load(run_dir)
    assert run.record.source == "data-engine"
    assert run.record.rig == "fake"
    assert run.record.question is None

    # Every cell the shell's own reachability found must have been visited
    # — the sweep's job is to exhaust the shell, not sample it. `sup` is
    # done running by now (phase == "done"), so `_reach`/`visited`/`blocked`
    # are no longer being written by anything and are safe to read plainly.
    reachable = {c for c, r in sup._reach.items() if r is not None}
    assert reachable, "no reachable cells on this shell — nothing to sweep"
    assert sup.visited == reachable
    assert not sup.blocked

    # The approval chain is on disk, not just in the log: the survey's own
    # beats come first, in order, and they are what proves a human gate was
    # opened for a machine-driven request.
    kinds = [e.kind for e in run.events]
    assert kinds[:3] == ["requested", "awaiting_approval", "approved"], kinds
    assert "captured" in kinds
    beat = [e for e in run.events if e.kind == "captured" and e.step_id == 0]
    assert beat, "no captured beat names the survey step"

    # The fused cloud carries its colours, row-aligned — a data-engine run's
    # fused output is the live one minus `ai/**` (flows-design §6).
    cloud = np.load(run_dir / "fused" / "cloud.npy")
    colors = np.load(run_dir / "fused" / "colors.npy")
    assert len(colors) == len(cloud)

    non_survey = [s for s in run.captured if s.record.view.address is not None]
    assert non_survey, "the sweep never captured a single cell"
    for step in non_survey:
        vstate = step.view_state
        assert vstate is not None and vstate.decider == "sweep-shortest", \
            (step.id, vstate.decider if vstate else None)

    rep = run.validate()
    # A question-less sweep owes no answer (validate ruling 2026-09-08):
    # the record must stand entirely clean.
    assert rep.ok, rep.problems


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_ranking_picks_cheapest_and_skips_unplannable()
    test_joint_l1()
    test_joint_l1_single_waypoint_is_free()
    test_sweep_driver_visits_every_reachable_cell(Path(tempfile.mkdtemp()))
    print("OK test_collect")
