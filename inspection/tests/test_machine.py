#!/usr/bin/env python3
"""Supervisor: request->planning->previewing, supersede, blocked.
Run: p inspection/tests/test_machine.py"""
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from inspection.motion.plan import DEMO_PARK
from inspection.run.machine import Supervisor
from inspection.run.rigs import FakeRig
from inspection.tests.test_publisher_v2 import BusSpy
from inspection.ui.publisher import InspectionPublisher


def wait_for(cond, timeout=30.0, msg=""):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, f"timeout: {msg}"
        time.sleep(0.01)


def make_sup(q_start=None, **kw):
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_start if q_start is not None else q_survey.copy())
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus),
                     Path(tempfile.mkdtemp()) / "run", q_survey, **kw)
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()
    return sup, rig, bus, th


def test_survey_request_reaches_previewing():
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    assert sup.phase == "idle"
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    assert sup.target == "survey"
    # preview loop is publishing poses
    n0 = len([1 for t, _ in bus.published if t == "scene/poses"])
    time.sleep(0.2)
    n1 = len([1 for t, _ in bus.published if t == "scene/poses"])
    assert n1 > n0, "preview replay is not looping"
    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_invalid_commands_dropped():
    sup, rig, bus, th = make_sup()
    sup.events.put({"cmd": "view/confirm", "target": "survey"})   # not previewing
    sup.events.put({"cmd": "nonsense"})
    sup.events.put({"cmd": "view/request", "target": [99, 99]})   # no sphere yet
    time.sleep(0.3)
    assert sup.phase == "idle"
    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_stale_plan_result_discarded():
    # NOTE: do not poll for phase == "planning" — trivial direct moves plan in
    # ~1 ms and the transient is unobservable. Test the guard from the stable
    # previewing state instead: a stale plan_done must not hijack it.
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    live_path = sup._path
    sup.events.put({"ev": "plan_done", "gen": sup.gen - 1, "target": "survey",
                    "path": [DEMO_PARK.copy()], "detail": "stale"})
    time.sleep(0.2)
    assert sup.phase == "previewing" and sup._path is live_path
    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


class SlowPlanSupervisor(Supervisor):
    """`_plan_worker` blocks on `self.release` (test-controlled) and counts
    how many planners are ever alive concurrently — proves F1a's
    supersede-during-planning defers rather than racing a second OMPL call
    against the same RobotCell."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._active_planners = 0
        self.max_concurrent = 0
        self.calls = 0   # total times a planner ever ran, incl. re-plans

    def _plan_worker(self, gen, target):
        with self._lock:
            self._active_planners += 1
            self.max_concurrent = max(self.max_concurrent, self._active_planners)
            self.calls += 1
        self.release.wait(10.0)
        with self._lock:
            self._active_planners -= 1
        q = self.rig.q()
        self.events.put({"ev": "plan_done", "gen": gen, "target": target,
                         "path": [q, q], "detail": "stub"})


class RefusingPlanSupervisor(Supervisor):
    """`_plan_worker` refuses deterministically — no real planner needed to
    exercise the blocked path."""

    def _plan_worker(self, gen, target):
        self.events.put({"ev": "plan_done", "gen": gen, "target": target,
                         "path": None, "detail": "stub refuses"})


def test_supersede_during_planning_defers():
    # Supersede requires two DISTINCT targets — pre-boot the only valid
    # target is "survey" itself (no sphere yet, so cell targets are dropped),
    # and a same-target duplicate is a no-op, not a supersede (see
    # test_duplicate_view_request_while_planning_is_ignored). So boot for
    # real first (stub released immediately — boot isn't what's under test
    # here), then supersede between two real, distinct reachable cells.
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_survey.copy() + np.radians([0, 0, 0, 0, 0, 8]))
    bus = BusSpy()
    sup = SlowPlanSupervisor(rig, InspectionPublisher(bus),
                             Path(tempfile.mkdtemp()) / "run", q_survey)
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()

    sup.release.set()
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="boot previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    wait_for(lambda: sup.phase == "idle" and sup.survey_state == "visited",
             timeout=60, msg="boot settled")
    sup.release.clear()   # re-arm the stub to block for the real test below

    views = bus.last("views/state")
    cells = [[c["h"], c["v"]] for c in views["cells"] if c["state"] == "available"]
    assert len(cells) >= 2, "need two distinct reachable cells to supersede between"
    target_a, target_b = cells[0], cells[1]

    sup.events.put({"cmd": "view/request", "target": target_a})
    wait_for(lambda: sup.phase == "planning", msg="planning (stub blocks here)")
    gen1 = sup.gen

    # supersede with a DIFFERENT target while the first plan is still in
    # flight -> must defer, not spawn a second planner
    sup.events.put({"cmd": "view/request", "target": target_b})
    time.sleep(0.1)
    assert sup.phase == "planning" and sup.gen == gen1, \
        "a superseding request must not start a second planner mid-plan"
    assert sup._pending_request == tuple(target_b)

    sup.release.set()   # let gen1's (now-stale-once-superseded) plan finish
    wait_for(lambda: sup.phase == "previewing", msg="previewing after deferred replan")
    assert sup.gen == gen1 + 1, "the deferred request must start its own generation"
    assert sup._pending_request is None
    assert sup.max_concurrent == 1, \
        f"two planners were alive at once: {sup.max_concurrent}"

    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_duplicate_view_request_while_planning_is_ignored():
    """Re-clicking the UI's `pending` (still-planning) button re-sends the
    same view/request. It must be a pure no-op — not a supersede-and-defer —
    or every extra click throws away a completed OMPL pass and pays for
    another one (the bug: an operator repeat-clicking "..." could keep the
    loop from ever reaching previewing)."""
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_survey.copy())
    bus = BusSpy()
    sup = SlowPlanSupervisor(rig, InspectionPublisher(bus),
                             Path(tempfile.mkdtemp()) / "run", q_survey)
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()

    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "planning", msg="planning (stub blocks here)")
    gen1 = sup.gen

    # same target again, mid-plan -> ignored outright, not queued
    sup.events.put({"cmd": "view/request", "target": "survey"})
    time.sleep(0.1)
    assert sup.phase == "planning" and sup.gen == gen1, \
        "a duplicate in-flight request must not disturb the running plan"
    assert sup._pending_request is None, \
        "a duplicate of the in-flight target must not be queued either"

    sup.release.set()   # let the one and only plan finish
    wait_for(lambda: sup.phase == "previewing", msg="previewing after single plan")
    assert sup.gen == gen1, "no replan happened, so gen must not have advanced"
    assert sup.calls == 1, f"expected exactly one plan, got {sup.calls}"

    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_blocked_plan_returns_to_idle():
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_survey.copy())
    bus = BusSpy()
    sup = RefusingPlanSupervisor(rig, InspectionPublisher(bus),
                                 Path(tempfile.mkdtemp()) / "run", q_survey)
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()

    # Don't poll for an intermediate phase (unobservable, see the earlier
    # coordinator ruling) — wait for the views/state topic to have been
    # republished twice more (planning, then idle-after-refusal), which is
    # monotonic and can't be missed regardless of how fast the stub runs.
    # Anchor n0 on the boot publish having landed, so a slow dispatcher
    # start can't make the count threshold trip early.
    wait_for(lambda: any(t == "views/state" for t, _ in bus.published),
             msg="boot views published")
    n0 = len([1 for t, _ in bus.published if t == "views/state"])
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: len([1 for t, _ in bus.published if t == "views/state"])
             >= n0 + 2, msg="plan+refusal views republished")

    assert sup.phase == "idle"
    assert sup.target is None, "target must not be left pending"
    assert sup._pending_request is None
    views = bus.last("views/state")
    assert views["survey"] == {"state": "available"}

    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def full_boot(sup):
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="boot previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    wait_for(lambda: sup.phase == "idle" and sup.survey_state == "visited",
             timeout=60, msg="boot settled")


def test_full_cycle_and_visited():
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    full_boot(sup)
    assert sup.sphere is not None and len(sup.acc.points)
    views = bus.last("views/state")
    assert views["survey"]["state"] == "visited"
    cells = [c for c in views["cells"] if c["state"] == "available"]
    assert cells, "no reachable cells after boot"
    target = [cells[0]["h"], cells[0]["v"]]
    sup.events.put({"cmd": "view/request", "target": target})
    wait_for(lambda: sup.phase == "previewing", timeout=60, msg="cell preview")
    sup.events.put({"cmd": "view/confirm", "target": target})
    wait_for(lambda: tuple(target) in sup.visited, timeout=60, msg="cell visited")
    assert sup.current == tuple(target)
    sup.request_shutdown(); th.join(10)


def test_stop_mid_execute_returns_to_idle():
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 30]))          # long boot move
    rig.speed = 0.05                    # slow: ~10 s — time to stop it
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    wait_for(lambda: sup.phase == "executing", msg="executing")
    assert sup.pose_active.is_set()
    sup.events.put({"cmd": "run/stop"})
    wait_for(lambda: sup.phase == "idle", msg="stopped -> idle")
    assert not sup.pose_active.is_set()
    assert not sup.stop_event.is_set(), "dispatcher must clear stop_event"
    assert sup.survey_state != "visited"
    sup.request_shutdown(); th.join(10)


class CaptureFailRig(FakeRig):
    def capture(self, pose_id):
        if pose_id == 1:
            raise RuntimeError("synthetic bad view")
        return super().capture(pose_id)


def test_capture_fail_not_visited():
    q_survey = DEMO_PARK.copy()
    rig = CaptureFailRig(q_survey.copy() + np.radians([0, 0, 0, 0, 0, 8]))
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus),
                     Path(tempfile.mkdtemp()) / "run", q_survey)
    th = threading.Thread(target=sup.run, daemon=True); th.start()
    full_boot(sup)
    views = bus.last("views/state")
    target = [c["h"] for c in views["cells"] if c["state"] == "available"][:1] and \
             [[c["h"], c["v"]] for c in views["cells"] if c["state"] == "available"][0]
    sup.events.put({"cmd": "view/request", "target": target})
    wait_for(lambda: sup.phase == "previewing", timeout=60, msg="preview")
    sup.events.put({"cmd": "view/confirm", "target": target})
    wait_for(lambda: sup.phase == "idle", timeout=60, msg="settle")
    assert tuple(target) not in sup.visited
    sup.request_shutdown(); th.join(10)


class FaultRig(FakeRig):
    def move(self, path):
        if getattr(self, "_boot_done", False):
            raise RuntimeError("safety mode changed mid-path")
        self._boot_done = True
        return super().move(path)


def test_executor_fault_enters_fault_and_exit_works():
    q_survey = DEMO_PARK.copy()
    rig = FaultRig(q_survey.copy() + np.radians([0, 0, 0, 0, 0, 8]))
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus),
                     Path(tempfile.mkdtemp()) / "run", q_survey)
    th = threading.Thread(target=sup.run, daemon=True); th.start()
    full_boot(sup)
    views = bus.last("views/state")
    target = [[c["h"], c["v"]] for c in views["cells"]
              if c["state"] == "available"][0]
    sup.events.put({"cmd": "view/request", "target": target})
    wait_for(lambda: sup.phase == "previewing", timeout=60, msg="preview")
    sup.events.put({"cmd": "view/confirm", "target": target})
    wait_for(lambda: sup.phase == "fault", timeout=60, msg="fault")
    sup.events.put({"cmd": "view/request", "target": target})   # dead in fault
    time.sleep(0.2)
    assert sup.phase == "fault"
    sup.events.put({"cmd": "run/exit"})
    th.join(10); assert not th.is_alive()
    assert (sup.outdir / "run.json").exists()


def test_sigint_saves_and_exits():
    import signal
    from inspection.run.app import install_sigint
    sup, rig, bus, th = make_sup()
    install_sigint(sup)
    try:
        signal.raise_signal(signal.SIGINT)
        th.join(15); assert not th.is_alive(), "SIGINT did not shut down"
        assert (sup.outdir / "run.json").exists()
        assert sup.stop_event.is_set() or sup.phase == "done"
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def main():
    test_survey_request_reaches_previewing()
    test_invalid_commands_dropped()
    test_stale_plan_result_discarded()
    test_supersede_during_planning_defers()
    test_duplicate_view_request_while_planning_is_ignored()
    test_blocked_plan_returns_to_idle()
    test_full_cycle_and_visited()
    test_stop_mid_execute_returns_to_idle()
    test_capture_fail_not_visited()
    test_executor_fault_enters_fault_and_exit_works()
    test_sigint_saves_and_exits()
    print("OK test_machine (task 5)")


if __name__ == "__main__":
    main()
