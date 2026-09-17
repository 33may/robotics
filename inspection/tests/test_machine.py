#!/usr/bin/env python3
"""Supervisor: request->planning->previewing, supersede, blocked.
Run: p inspection/tests/test_machine.py"""
import json
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


def make_writer(q_survey=None):
    """A real `RunWriter` over a throwaway directory.

    The Supervisor has no other way to write, so every one of these tests
    exercises the record layer whether it asserts on it or not — which is the
    point: a machine that cannot record is not the machine we ship.
    """
    from inspection.record.writer import RunWriter
    from inspection.run.app import config_snapshot, face_method, viewsphere_method
    q = DEMO_PARK.copy() if q_survey is None else q_survey
    return RunWriter.create(
        Path(tempfile.mkdtemp()), run_id="run", name="run", source="live",
        rig="fake", config=config_snapshot(), tags=["test"],
        view_methods=[viewsphere_method(), face_method()],
        q_survey=[float(v) for v in q])


def make_sup(q_start=None, rig_cls=FakeRig, **kw):
    q_survey = DEMO_PARK.copy()
    rig = rig_cls(q_start if q_start is not None else q_survey.copy())
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus), make_writer(q_survey),
                     q_survey, **kw)
    # Same final gate the REAL rig has: every move re-validated against the
    # live world unless the caller passes its own. Without this the fake
    # happily executed retreats the real rig refused ("world refuses path"
    # at segment 0 — cam_bench1/4), and four dead runs passed the suite.
    rig.world = sup.world
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


class GatedSupervisor(Supervisor):
    """Blocks the dispatcher *before* it handles anything, so a test can load
    several commands into the FIFO and control the moment any of them is
    processed. The only way to reproduce a queue-ordering race without
    racing: without it, "put a confirm, then Ctrl-C" is decided by whether
    the dispatcher happened to dequeue in the microsecond between."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = threading.Event()
        self.gate.set()

    def handle(self, ev):
        self.gate.wait(10.0)
        super().handle(ev)


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
                             make_writer(q_survey), q_survey)
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
                             make_writer(q_survey), q_survey)
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
                                 make_writer(q_survey), q_survey)
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


class _BlindBackend:
    """Segmenter that always returns nothing — a stand-in for weights that
    failed to load, an occluded object, or a confident-but-empty mask."""

    def segment(self, rgb, box):
        return np.zeros(rgb.shape[:2], dtype=bool), 0.0


def _stub_segmenter(backend=None):
    from inspection.eyes.verbs_local import StubBackend
    from inspection.run.segmenter import ObjectSegmenter
    return ObjectSegmenter(backend=backend or StubBackend())


def test_survey_runs_through_the_segmenter_when_injected():
    """The masked path end to end on the fake rig: a prompt box is built from
    the bootstrap cluster, the mask comes back, and the cloud is fused from
    masked depth rather than from grown depth."""
    seg = _stub_segmenter()
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]), segmenter=seg)
    full_boot(sup)
    assert seg.calls >= 1, "settle leg never asked for a mask"
    assert seg.misses == 0
    assert len(sup.acc.points), "masked lift produced no object"
    assert sup.sphere is not None
    sup.request_shutdown(); th.join(10)


def test_survey_falls_back_to_depth_when_the_mask_misses():
    """A model failure is a liveness cost, never a dead run: the survey still
    completes on the depth pipeline and the loop keeps its object."""
    seg = _stub_segmenter(_BlindBackend())
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]), segmenter=seg)
    full_boot(sup)                       # would hang or fail if it did not
    assert seg.misses >= 1, "blind backend should have been rejected"
    assert len(sup.acc.points), "fallback did not produce the object"
    sup.request_shutdown(); th.join(10)


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
    def capture(self, pose_id, out_dir=None):
        if pose_id == 1:
            raise RuntimeError("synthetic bad view")
        return super().capture(pose_id, out_dir)


class FrozenStampRig(FakeRig):
    """Stamps every capture with the boot-time q — the 2108-ui frozen-RTDE
    signature: the robot moved, the stamp did not. The survey capture is
    unaffected (the arm really is at q0 there)."""

    def __init__(self, q, *a, **k):
        super().__init__(q, *a, **k)
        self._q_frozen = np.asarray(q, dtype=float).copy()

    def capture(self, pose_id, out_dir=None):
        cap = super().capture(pose_id, out_dir)
        cap["q"] = self._q_frozen.copy()
        return cap


def test_bad_plane_view_rejected():
    """A bad-plane view with an UNCONFIRMED pose stamp is refused.

    The table is at z=0 by measurement; a plane fitted 24 cm up or tilted 48
    degrees, from a capture whose stamped q does NOT match the planned
    arrival, is the 2108-ui failure (a frozen RTDE q did exactly this to
    five views in a row). Such a view must not enter the accumulator, and
    the run must carry on. (With the pose CONFIRMED at target the gate only
    warns — see test_bad_plane_pose_confirmed_fuses.)
    """
    import inspection.run.settle as settle_mod

    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]), rig_cls=FrozenStampRig)
    full_boot(sup)                        # real geometry: boot view is sane
    n_boot = len(sup.acc.points)

    tilt = np.radians(48.0)
    poisoned = {
        "points": np.random.default_rng(0).normal(
            [0.3, 0.0, 0.24], 0.01, (200, 3)),
        "centroid": np.array([0.3, 0.0, 0.24]),
        "plane": np.array([np.sin(tilt), 0.0, np.cos(tilt), -0.2]),
        "n_scene": 200,
    }
    # Patch the seam the settle leg actually calls: identity resolution moved
    # into `run.segmenter.object_view`, imported by `run.settle` — which is
    # where the settle pipeline itself now lives (task 6). Returns
    # (view, segmentation).
    real = settle_mod.object_view
    settle_mod.object_view = lambda *a, **k: (poisoned, None)
    try:
        views = bus.last("views/state")
        target = [[c["h"], c["v"]] for c in views["cells"]
                  if c["state"] == "available"][0]
        sup.events.put({"cmd": "view/request", "target": target})
        wait_for(lambda: sup.phase == "previewing", timeout=60, msg="preview")
        sup.events.put({"cmd": "view/confirm", "target": target})
        wait_for(lambda: sup.phase == "idle", timeout=60, msg="settle")
    finally:
        settle_mod.object_view = real

    assert tuple(target) not in sup.visited, "poisoned view marked visited"
    assert len(sup.acc.points) == n_boot, "poisoned points entered the fusion"
    assert sup.phase == "idle", "run did not continue after the rejection"
    sup.request_shutdown(); th.join(10)


def test_bad_plane_pose_confirmed_fuses():
    """A bad plane with the pose CONFIRMED at target is waived, not rejected.

    The stamped q matches the planned arrival (the default FakeRig stamps
    truthfully), so the pose cannot be the frozen-RTDE failure the gate
    exists for — a contradicting plane is then a grazing-view FIT problem
    (1009-cup cell [10,0]: z=-71 mm from a correct pose, twice) and the
    view must still count (Anton 2026-09-10: "if the camera is already
    there, there is no point to reject the view").
    """
    import inspection.run.settle as settle_mod

    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    full_boot(sup)

    tilt = np.radians(48.0)
    poisoned = {
        "points": np.random.default_rng(0).normal(
            [0.3, 0.0, 0.24], 0.01, (200, 3)),
        "centroid": np.array([0.3, 0.0, 0.24]),
        "plane": np.array([np.sin(tilt), 0.0, np.cos(tilt), -0.2]),
        "n_scene": 200,
    }
    real = settle_mod.object_view
    settle_mod.object_view = lambda *a, **k: (poisoned, None)
    try:
        views = bus.last("views/state")
        target = [[c["h"], c["v"]] for c in views["cells"]
                  if c["state"] == "available"][0]
        sup.events.put({"cmd": "view/request", "target": target})
        wait_for(lambda: sup.phase == "previewing", timeout=60, msg="preview")
        sup.events.put({"cmd": "view/confirm", "target": target})
        wait_for(lambda: sup.phase == "idle", timeout=60, msg="settle")
    finally:
        settle_mod.object_view = real

    assert tuple(target) in sup.visited, \
        "pose-confirmed view was rejected instead of waived"
    assert sup.phase == "idle"
    sup.request_shutdown(); th.join(10)


def test_capture_fail_not_visited():
    q_survey = DEMO_PARK.copy()
    rig = CaptureFailRig(q_survey.copy() + np.radians([0, 0, 0, 0, 0, 8]))
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus), make_writer(q_survey),
                     q_survey)
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
    def move(self, path, world="live"):
        if getattr(self, "_boot_done", False):
            raise RuntimeError("safety mode changed mid-path")
        self._boot_done = True
        return super().move(path)


def test_executor_fault_enters_fault_and_exit_works():
    q_survey = DEMO_PARK.copy()
    rig = FaultRig(q_survey.copy() + np.radians([0, 0, 0, 0, 0, 8]))
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus), make_writer(q_survey),
                     q_survey)
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


def test_exit_during_planning_defers_until_plan_done():
    """I2a: `run/exit` (or Ctrl-C) while a planner is in flight must NOT tear
    the dispatcher down under it. `_plan_worker` calls `rig.q()` — on hardware
    that is the same RTDEReceiveInterface `app.py`'s teardown then
    disconnects, which is a segfault, not an exception. So the exit waits for
    plan_done and only then shuts down."""
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_survey.copy())
    bus = BusSpy()
    sup = SlowPlanSupervisor(rig, InspectionPublisher(bus),
                             make_writer(q_survey), q_survey)
    th = threading.Thread(target=sup.run, daemon=True); th.start()

    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "planning", msg="planning (stub blocks here)")

    sup.request_shutdown()
    time.sleep(0.4)
    assert sup.phase == "planning", \
        f"exit must not shut down mid-plan (phase={sup.phase})"
    assert sup._exit_after, "the exit must have been deferred, not dropped"
    assert th.is_alive()

    sup.release.set()                       # the planner finishes and reports
    th.join(10)
    assert not th.is_alive(), "the deferred exit never fired"
    assert sup.phase == "done"
    assert (sup.outdir / "run.json").exists()
    assert sup.join_workers(5.0), "a worker outlived the shutdown"
    assert not sup._action_thread.is_alive()


def test_exit_during_planning_fires_on_stale_plan_done():
    """Same deferral, but the plan_done that comes back is stale-gen. The gen
    guard drops the event's *content*; the deferred exit must still proceed,
    or the dispatcher waits for a release that will never come."""
    sup, rig, bus, th = make_sup()
    sup.phase = "planning"                  # test-only poke: no worker needed
    sup.events.put({"cmd": "run/exit"})
    time.sleep(0.2)
    assert sup.phase == "planning" and sup._exit_after, "exit was not deferred"
    sup.events.put({"ev": "plan_done", "gen": sup.gen - 1, "target": "survey",
                    "path": None, "detail": "stale"})
    th.join(10)
    assert not th.is_alive(), "a stale plan_done must still release the exit"
    assert (sup.outdir / "run.json").exists()


def test_queued_confirm_cannot_undo_a_ctrl_c():
    """I3: `request_shutdown` sets stop_event and queues run/exit, but a
    `view/confirm` already in the FIFO is dispatched FIRST, and
    `_on_view_confirm` clears stop_event and starts a motion — a Ctrl-C
    followed by the arm moving. The `_shutting_down` flag, set before the
    queue put, is what makes that ordering harmless."""
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_survey.copy() + np.radians([0, 0, 0, 0, 0, 30]), speed=0.05)
    bus = BusSpy()
    sup = GatedSupervisor(rig, InspectionPublisher(bus),
                          make_writer(q_survey), q_survey)
    th = threading.Thread(target=sup.run, daemon=True); th.start()
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    assert rig.moves == [], "nothing should have moved yet"

    # The race, reproduced deterministically: the confirm is in the queue
    # (and may already be dequeued) when the signal handler runs.
    sup.gate.clear()
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    time.sleep(0.05)
    sup.request_shutdown()
    sup.gate.set()

    th.join(10)
    assert not th.is_alive(), "shutdown did not complete"
    assert rig.moves == [], f"a queued confirm started a motion: {rig.moves}"
    assert sup.stop_event.is_set(), "the confirm cleared the Ctrl-C stop"
    assert sup.phase == "done" and sup.survey_state != "visited"
    assert (sup.outdir / "run.json").exists()


def test_stop_mid_execute_records_the_event_before_exit():
    """I5, in the record layer's terms: a stopped turn is on disk BEFORE any
    shutdown is requested, and it is an EVENT, not a step — the arm halted
    mid-move, so no camera was ever pointed anywhere and there is nothing for
    a step directory to hold."""
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 30]))
    rig.speed = 0.05
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    wait_for(lambda: sup.phase == "executing", msg="executing")
    sup.events.put({"cmd": "run/stop"})
    wait_for(lambda: sup.phase == "idle", msg="stopped -> idle")

    events = sup.outdir / "events.jsonl"
    assert events.exists(), "the stop was not recorded"
    last = json.loads(events.read_text().splitlines()[-1])
    assert last["kind"] == "stopped", last
    assert "survey" in last["detail"]
    assert not (sup.outdir / "steps").exists(), "a halted move left a step"

    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_join_workers_after_full_boot():
    """I2b: the design's "join worker (bounded)" step, which `app.py`'s
    teardown now runs before `rig.close()`."""
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    full_boot(sup)
    t0 = time.monotonic()
    assert sup.join_workers(5.0), "workers outlived the join budget"
    assert time.monotonic() - t0 < 5.0
    assert sup._action_thread is not None and not sup._action_thread.is_alive()
    assert sup._preview_thread is None or not sup._preview_thread.is_alive()
    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_malformed_target_is_rejected_not_raised():
    """M5: commands come off the bus untrusted. A malformed target is a
    refusal with a log line, not a traceback out of the handler."""
    sup, rig, bus, th = make_sup()
    for bad in (None, 5, [1, 2, 3], ["a", "b"], [1.5, 0], {"h": 1},
                # marker-id lookalikes: a 3D click sends a string, so a
                # near-miss must be refused rather than guessed at
                "h7v1", "h07v", "views/h07v1", "hxxvy", ""):
        sup.events.put({"cmd": "view/request", "target": bad})
        sup.events.put({"cmd": "view/confirm", "target": bad})
    time.sleep(0.4)
    assert sup.phase == "idle" and sup.target is None
    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_marker_id_target_from_a_3d_click():
    """Clicking a viewsphere marker in the 3D view sends the id the BACKEND
    published (`ui.publisher.cell_key`), so the scene panel never has to learn
    what a cell is. `_normalize_target` parses its own naming back."""
    from inspection.ui.publisher import cell_key
    norm = Supervisor._normalize_target
    for h, v in ((0, 0), (7, 1), (11, 2)):
        assert norm(cell_key(h, v)) == (h, v)
    assert norm("survey") == "survey"       # still its own thing
    assert norm((3, 1)) == (3, 1)           # the actions panel's form still works


def test_idle_publishes_the_real_pose():
    """The arm's actual configuration is part of "the current world" (FR1).

    Caught on the first hardware run: nothing published `scene/poses` at idle,
    so a window opened before any motion drew every robot mesh at its identity
    placement — the whole arm collapsed onto the base. The retained topic must
    carry a real frame from the moment the machine reaches idle, and again
    whenever it returns there (a cancelled preview otherwise leaves the last
    frame of a path the arm never took).
    """
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    wait_for(lambda: sup.phase == "idle", msg="idle")
    wait_for(lambda: any(t == "scene/poses" for t, _ in bus.published),
             msg="no pose frame published at idle")
    frame = bus.last("scene/poses")
    assert frame["names"], "pose frame carries no nodes"

    # `transforms` crosses the bus as a tagged ndarray (protocol §3.1):
    # {"$nd": {dtype, shape, data}} holding [N, 4, 4] float32.
    nd = frame["transforms"]["$nd"]
    mats = np.frombuffer(nd["data"], dtype=nd["dtype"]).reshape(nd["shape"])
    assert len(mats) == len(frame["names"])

    # ...and the frame must describe where the arm really is, not the identity
    # placement. At the survey pose the arm is extended, so at least one link
    # sits well clear of the base origin; an all-identity frame (the bug) has
    # every translation at zero.
    far = float(np.abs(mats[:, :3, 3]).max())
    assert far > 0.1, f"every link is at the base origin (max |t| = {far})"

    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


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


def test_three_plan_refusals_trigger_retreat():
    """The capture-refit self-trap (run 1409-lan-ai_bench1): the object box
    grows over the standing arm, the start config goes invalid, and every
    plan refuses — including cells that planned fine before. After three
    consecutive refusals the machine must retreat radially (validated
    against the box the stance was approved under) and become plannable
    again, instead of dead-ending the run."""
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    full_boot(sup)
    q_trapped = rig.q()
    # The trap, modelled bluntly: a refit "grew" the object box over the
    # stance. Small enough to leave most cell ENDPOINTS valid — in the real
    # trap the refusals pass the request gate exactly because reachability
    # (endpoint-only) stays green while the START is in collision.
    p = np.asarray(sup.ik.fk(q_trapped))[:3, 3]
    sup.world.set_object("object", [0.12, 0.12, 0.12],
                         [float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, 0.0])
    tried = set()
    for n in (1, 2, 3):
        avail = [c for c, r in sup._reach.items()
                 if r is not None and c not in sup.visited and c not in tried]
        assert avail, f"no requestable cell left before refusal {n}"
        cell = avail[0]
        tried.add(cell)
        sup.events.put({"cmd": "view/request", "target": list(cell)})
        if n < 3:
            wait_for(lambda n=n: sup._plan_fails == n and sup.phase == "idle",
                     timeout=60, msg=f"refusal {n} counted")
        else:
            # the third refusal triggers the escape move
            wait_for(lambda: len(rig.moves) == 2, timeout=60, msg="retreat move")
            wait_for(lambda: sup.phase == "idle", timeout=60,
                     msg="idle after retreat")
    assert sup._plan_fails == 0, "counter must reset after the retreat"
    assert sup.blocked == set(), "refusals were judged from a stance that is gone"
    assert sup.current is None, "the camera left the cell it was standing at"
    assert np.abs(rig.q() - q_trapped).max() > 0.005, "the arm actually moved"
    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_retreat_escapes_a_poisoned_stance_box():
    """1509-cam-ai_bench1: four retreats fired and none moved, while the
    escape provably existed — the stance box the worker validated against
    already swallowed q_now, so tier-1 direct_move failed at config 0 every
    time. The env-only fallback tier must still leave (the move is radially
    away from the object by construction), and both the attempt and the
    outcome must land in events.jsonl."""
    import json
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    full_boot(sup)
    q_trapped = rig.q()
    p = np.asarray(sup.ik.fk(q_trapped))[:3, 3]
    box = ([0.12, 0.12, 0.12],
           [float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, 0.0])
    sup.world.set_object("object", *box)
    sup._stance_box = box           # poisoned: swallows the standing arm
    tried = set()
    for n in (1, 2, 3):
        avail = [c for c, r in sup._reach.items()
                 if r is not None and c not in sup.visited and c not in tried]
        assert avail, f"no requestable cell left before refusal {n}"
        cell = avail[0]
        tried.add(cell)
        sup.events.put({"cmd": "view/request", "target": list(cell)})
        if n < 3:
            wait_for(lambda n=n: sup._plan_fails == n and sup.phase == "idle",
                     timeout=60, msg=f"refusal {n} counted")
        else:
            wait_for(lambda: len(rig.moves) == 2, timeout=60,
                     msg="fallback retreat move")
            wait_for(lambda: sup.phase == "idle", timeout=60,
                     msg="idle after retreat")
    assert np.abs(rig.q() - q_trapped).max() > 0.005, \
        "the arm escaped despite the poisoned stance box"
    rows = [json.loads(l) for l in
            (sup.writer.dir / "events.jsonl").read_text().splitlines()]
    retreats = [r for r in rows if r["kind"] == "retreat"]
    assert len(retreats) == 2, "attempt + outcome rows recorded"
    assert "stance_box rev" in retreats[0]["detail"]
    assert retreats[1]["detail"].startswith("moved"), retreats[1]["detail"]
    assert "env-only" in retreats[1]["detail"], \
        "the record says WHICH tier saved the run"
    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_retreat_retraces_when_radial_has_no_ik(monkeypatch):
    """1509-cam-ai_bench2: far-side stances stand at the workspace boundary —
    the radial escape target has ZERO IK branches ([5,0] flange at 0.751 m,
    +1 cm out: none), so both radial tiers die at '0/0 free' and the arm is
    wedged. The retrace tier must reverse the executed arrival path instead.
    Radial impossibility is forced here by a 5 m retreat distance — same
    0-branch outcome as the real boundary, independent of the test cell."""
    import json
    import inspection.run.machine as machine_mod
    monkeypatch.setattr(machine_mod, "RETREAT_M", 5.0)
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    full_boot(sup)
    q_trapped = rig.q()
    p = np.asarray(sup.ik.fk(q_trapped))[:3, 3]
    sup.world.set_object("object", [0.12, 0.12, 0.12],
                         [float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, 0.0])
    assert sup._arrival_path is not None, "confirm stored the arrival path"
    tried = set()
    for n in (1, 2, 3):
        avail = [c for c, r in sup._reach.items()
                 if r is not None and c not in sup.visited and c not in tried]
        assert avail, f"no requestable cell left before refusal {n}"
        cell = avail[0]
        tried.add(cell)
        sup.events.put({"cmd": "view/request", "target": list(cell)})
        if n < 3:
            wait_for(lambda n=n: sup._plan_fails == n and sup.phase == "idle",
                     timeout=60, msg=f"refusal {n} counted")
        else:
            wait_for(lambda: len(rig.moves) == 2, timeout=60,
                     msg="retrace retreat move")
            wait_for(lambda: sup.phase == "idle", timeout=60,
                     msg="idle after retreat")
    assert np.abs(rig.q() - q_trapped).max() > 0.005, "the arm actually moved"
    assert sup._arrival_path is None, "stale arrival path cleared after moving"
    rows = [json.loads(l) for l in
            (sup.writer.dir / "events.jsonl").read_text().splitlines()]
    outcome = [r for r in rows if r["kind"] == "retreat"][-1]
    assert "retrace" in outcome["detail"], outcome["detail"]
    sup.request_shutdown(); th.join(10)
    assert not th.is_alive()


def test_face_gate_opens_after_two_stable_fits():
    import numpy as np
    from inspection.fit.contract import Decomposition, Primitive

    def prim(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        pose = [[c, -s, 0, 0.1], [s, c, 0, -0.35], [0, 0, 1, 0.07],
                [0, 0, 0, 1]]
        return Primitive(kind="box", pose=pose,
                         params={"extents": [0.2, 0.1, 0.15]}, label="box")

    sup, rig, bus, th = make_sup()
    try:
        # Feed the gate directly — the unit under test is the counter.
        sup._update_face_gate(prim(0.0))
        assert sup._face_states()["x+"]["status"] == "settling"
        sup._update_face_gate(prim(np.deg2rad(1.0)))     # small delta #1
        assert sup._face_states()["x+"]["status"] == "settling"
        sup._update_face_gate(prim(np.deg2rad(2.0)))     # small delta #2
        assert sup._face_states()["x+"]["status"] == "ready"
        sup._update_face_gate(prim(np.deg2rad(40.0)))    # box jumped
        assert sup._face_states()["x+"]["status"] == "settling"
        sup.face_visited.add("top")
        sup._update_face_gate(prim(np.deg2rad(40.5)))
        sup._update_face_gate(prim(np.deg2rad(41.0)))
        st = sup._face_states()
        assert st["top"]["status"] == "visited"
        assert st["x+"]["status"] == "ready"
    finally:
        sup.request_shutdown(); th.join(10)


def test_face_request_full_cycle():
    import numpy as np
    from inspection.fit.contract import Primitive

    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        pose = [[1, 0, 0, 0.10], [0, 1, 0, -0.35], [0, 0, 1, 0.05],
                [0, 0, 0, 1]]
        prim = Primitive(kind="box", pose=pose,
                         params={"extents": [0.12, 0.08, 0.10]}, label="box")
        for _ in range(3):                      # open the gate
            sup._update_face_gate(prim)
        sup.events.put({"cmd": "view/request", "target": ["face", "top"]})
        wait_for(lambda: sup.phase == "previewing", timeout=60,
                 msg="face preview")
        assert sup.target == ("face", "top")
        sup.events.put({"cmd": "view/confirm", "target": ["face", "top"]})
        wait_for(lambda: "top" in sup.face_visited, timeout=60,
                 msg="face visited")
        find = sup._face_find.get("top")
        assert find is not None and find["theta"] is not None
        # The ray resolution's own findings, not a ladder rung name.
        assert set(find) == {"theta", "shift", "depth", "pos", "pose"}
        # The record carries the face identity.
        import json as _json
        from pathlib import Path as _P
        run_dir = _P(sup.outdir)
        steps = sorted((run_dir / "steps").iterdir())
        step = _json.load(open(steps[-1] / "step.json"))
        assert step["view"]["method"] == "face"
        assert step["view"]["address"] == ["face", "top"]
    finally:
        sup.request_shutdown(); th.join(10)


def test_face_refusal_does_not_enter_blocked():
    """I1: `_on_plan_done`'s refusal branch used to `self.blocked.add(target)`
    for EVERY refused target, faces included. `blocked` mixes with cell
    tuples in `nav()`, and `_menu_input`'s `sorted(map(list, v))` over it
    raises the moment a face tuple sits beside a cell tuple — silently
    losing the whole `MenuInput` record. A face refusal must stay entirely
    in `_face_unreachable`/`_face_theta`; `blocked` stays cell-only."""
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_survey.copy())
    bus = BusSpy()
    sup = RefusingPlanSupervisor(rig, InspectionPublisher(bus),
                                 make_writer(q_survey), q_survey)
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()
    try:
        import numpy as np
        from inspection.fit.contract import Primitive
        pose = [[1, 0, 0, 0.10], [0, 1, 0, -0.35], [0, 0, 1, 0.05],
                [0, 0, 0, 1]]
        prim = Primitive(kind="box", pose=pose,
                         params={"extents": [0.12, 0.08, 0.10]}, label="box")
        for _ in range(3):                      # open the gate
            sup._update_face_gate(prim)
        assert sup._face_states()["top"]["status"] == "ready"

        wait_for(lambda: any(t == "views/state" for t, _ in bus.published),
                 msg="boot views published")
        n0 = len([1 for t, _ in bus.published if t == "views/state"])
        sup.events.put({"cmd": "view/request", "target": ["face", "top"]})
        wait_for(lambda: len([1 for t, _ in bus.published
                              if t == "views/state"]) >= n0 + 2,
                 msg="plan+refusal views republished")

        assert sup.blocked == set(), \
            "a face refusal must never land in the cell-only `blocked` set"
        # The regression this guards: a mixed-type `blocked` used to raise
        # here — sortable-with-list is exactly what `_menu_input` needs.
        sorted(map(list, sup.blocked))
        # The menu must not say the face is BOTH blocked (absent) and
        # ready — its own vocabulary (`_face_states`) carries the refusal.
        assert "top" not in {t[1] for t in sup.blocked
                             if isinstance(t, tuple) and t and t[0] == "face"}
    finally:
        sup.request_shutdown(); th.join(10)


def test_views_payload_carries_faces():
    import numpy as np
    from inspection.fit.contract import Primitive

    sup, rig, bus, th = make_sup()
    try:
        full_boot(sup)
        views = bus.last("views/state")
        assert "faces" in views
        assert views["faces"]["x+"]["status"] == "settling"
        # Settling faces stay off the 3D scene: no standoff point, no marker.
        assert "pos" not in views["faces"]["x+"]
        desc = bus.last("scene/description")
        assert not [n for n in desc["nodes"]
                    if n["path"].startswith("views/face:")]
        pose = [[1, 0, 0, 0.1], [0, 1, 0, -0.35], [0, 0, 1, 0.05],
                [0, 0, 0, 1]]
        prim = Primitive(kind="box", pose=pose,
                         params={"extents": [0.1, 0.1, 0.1]}, label="box")
        for _ in range(3):
            sup._update_face_gate(prim)
        sup._publish_views()
        views = bus.last("views/state")
        assert views["faces"]["x+"]["status"] == "ready"
        # A ready face carries WHERE the camera would stand — the operator
        # sees the destination before approving, not after (bench gap #2).
        pos = views["faces"]["x+"]["pos"]
        assert len(pos) == 3
        # The x+ standoff sits beyond the face plane (center 0.1 + half 0.05).
        assert pos[0] > 0.15
        assert abs(pos[1] - (-0.35)) < 1e-6 and abs(pos[2] - 0.05) < 1e-6
        # No ladder has run yet, so this point is the rung-1 ESTIMATE and
        # says so — the marker draws dim until a plan resolves the truth
        # (Anton, bench: "the exact point, not hypothetical orthogonal").
        assert views["faces"]["x+"]["resolved"] is False
        # And each non-settling face is a scene marker the panel can click.
        desc = bus.last("scene/description")
        face_nodes = {n["path"]: n for n in desc["nodes"]
                      if n["path"].startswith("views/face:")}
        from inspection.view.faceview import FACES
        assert set(face_nodes) == {f"views/face:{f}" for f in FACES}
        assert face_nodes["views/face:x+"]["material"]["opacity"] < 0.9
    finally:
        sup.request_shutdown(); th.join(10)


def test_face_request_previews_on_the_wire():
    """Bench gap #1: while a face plan is in flight the payload must say
    pending, and previewing once the replay loops — the ActionsPanel approve
    button renders from exactly this block, so a status that stays "ready"
    is a face view no operator can ever confirm."""
    import numpy as np
    from inspection.fit.contract import Primitive

    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        pose = [[1, 0, 0, 0.10], [0, 1, 0, -0.35], [0, 0, 1, 0.05],
                [0, 0, 0, 1]]
        prim = Primitive(kind="box", pose=pose,
                         params={"extents": [0.12, 0.08, 0.10]}, label="box")
        for _ in range(3):                      # open the gate
            sup._update_face_gate(prim)
        sup.events.put({"cmd": "view/request", "target": ["face", "top"]})
        wait_for(lambda: sup.phase == "previewing", timeout=60,
                 msg="face preview")
        wait_for(lambda: bus.last("views/state")
                 ["faces"]["top"]["status"] == "previewing",
                 timeout=10, msg="previewing on the wire")
        hist = [p["faces"]["top"]["status"] for t, p in bus.published
                if t == "views/state" and "top" in p.get("faces", {})]
        assert "pending" in hist, "the planning publish must say pending"
        # At the approve moment the marker is the LADDER's endpoint, not the
        # rung-1 estimate: the plan resolved where the arm will actually go.
        face = bus.last("views/state")["faces"]["top"]
        assert face["resolved"] is True
        assert np.allclose(face["pos"], sup._face_find["top"]["pos"])
        # The marker mirrors it: enlarged, and solid now that it is truth.
        desc = bus.last("scene/description")
        node = next(n for n in desc["nodes"] if n["path"] == "views/face:top")
        assert node["geometry"]["radius"] >= 0.02
        assert node["material"]["opacity"] > 0.9
    finally:
        sup.request_shutdown(); th.join(10)


def test_request_gate_refuses_an_inaccessible_cell():
    """Stage B: `_reach` is the RAY cell map — a `RayResolution` per cell,
    `None` where the cell's ray has no provably-safe sample. The request
    gate reads exactly that key set, and a refusal says WHY: "inaccessible"
    is a fact about the ray (nothing safe on it), not the shrug the old
    "view/request (h, v) dropped" line was."""
    from inspection.view.rays import RayResolution
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        vals = list(sup._reach.values())
        assert vals, "the post-fuse refresh built no cell map"
        assert all(v is None or isinstance(v, RayResolution) for v in vals), \
            "_reach must hold ray resolutions, not shell rolls"
        avail = [c for c, r in sup._reach.items() if r is not None]
        assert avail, "no accessible cell after boot"
        cell = avail[0]
        sup._reach[cell] = None          # the ray resolved nothing safe
        n0 = len(bus.published)
        sup.events.put({"cmd": "view/request", "target": list(cell)})
        wait_for(lambda: any("inaccessible" in p.get("msg", "")
                             for _, p in bus.published[n0:]
                             if isinstance(p, dict)),
                 timeout=10, msg="honest refusal detail on the log topic")
        time.sleep(0.1)
        assert sup.phase == "idle" and sup.target is None
        assert cell not in sup.visited
    finally:
        sup.request_shutdown(); th.join(10)


def test_view_r_is_the_mean_resolved_cell_depth():
    """The Stage B honesty stopgap (plan Global Constraints): the schema's
    one `r` float, and the marker sphere built with it, carry the MEAN
    RESOLVED cell depth — cells are rays with their own depths now, so the
    old frozen fill-formula radius (`radius_for_extent`/`derive_r_floor`)
    is gone along with `self.r`/`self.r_floor`."""
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        depths = [r.depth for r in sup._reach.values() if r is not None]
        assert depths, "no accessible cell after boot"
        mean = float(np.mean(depths))
        assert sup.sphere is not None
        assert np.isclose(sup.sphere.r, mean), (sup.sphere.r, mean)
        assert np.isclose(sup._view_state(0, "survey").r, mean)
        assert not hasattr(sup, "r"), "self.r died with the shell radius"
        # `r_floor` died with the face ladder that was its last reader (task
        # 3): faces are rays now and carry their own resolved depth.
        assert not hasattr(sup, "r_floor")
    finally:
        sup.request_shutdown(); th.join(10)


def _acc_prim(sup):
    """A boxguard-shaped `Primitive` over the accumulator's own cloud — the
    UNPADDED belief box, which is what a real `_fit_step` installs."""
    from inspection.cell.geometry import BOX_PCT, min_yaw_aabb
    from inspection.fit.contract import Primitive
    center, dims, yaw = min_yaw_aabb(sup.acc.points, pct=BOX_PCT)
    c, s = np.cos(yaw), np.sin(yaw)
    pose = [[c, -s, 0.0, center[0]], [s, c, 0.0, center[1]],
            [0.0, 0.0, 1.0, center[2]], [0.0, 0.0, 0.0, 1.0]]
    return Primitive(kind="box", pose=pose,
                     params={"extents": dims.tolist()}, label="box")


def test_cell_map_measures_the_unpadded_belief_box():
    """`object_box` pads (+2 cm per dim, 5 cm floor) because it is the
    COLLISION box. Feeding that hull to the ray resolver pushes every cell's
    `clear_m` — and so its resolved depth, the marker radius and the interim
    plan target — outward from what the bench measured
    (`ray_replay.box_aabb`: "over-padding it would push every cell ray's
    sensor floor outward"). When a boxguard fit exists the map must measure
    the SAME prim the containment corners already come from."""
    import inspection.run.machine as machine_mod
    from inspection.view.rays import box_corners, cell_map
    from inspection.view.viewsphere import T_FLANGE_CAM
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    real, seen = machine_mod.cell_map, []

    def spy(*a, **kw):
        seen.append((a[4], kw.get("corners")))       # (aabb, corners)
        return real(*a, **kw)

    try:
        full_boot(sup)
        prim = _acc_prim(sup)
        corners = box_corners(prim)
        hull = (corners.min(0), corners.max(0))
        padded = sup._object_aabb
        # The two hulls really are different objects — the collision box's
        # margins are what must not reach the resolver.
        assert np.all(padded[0] <= hull[0] + 1e-9) and \
            np.all(padded[1] >= hull[1] - 1e-9)
        assert np.any(padded[1] - padded[0] > (hull[1] - hull[0]) + 1e-3)

        machine_mod.cell_map = spy
        sup._face_prim = prim
        got = sup._cell_map()
        aabb, passed_corners = seen[-1]
        assert np.allclose(aabb[0], hull[0]) and np.allclose(aabb[1], hull[1])
        assert passed_corners is not None
        sup._face_prim = None                        # no fit: padded fallback
        sup._cell_map()
        aabb, passed_corners = seen[-1]
        assert np.allclose(aabb[0], padded[0]) and np.allclose(aabb[1],
                                                               padded[1])
        assert passed_corners is None
        machine_mod.cell_map = real

        want = cell_map(sup.world, sup.ik, T_FLANGE_CAM, sup.acc.centroid,
                        hull, sup.rig.intr["fy"], sup.rig.intr["height"],
                        corners=corners)
        assert {c: None if r is None else round(r.depth, 6)
                for c, r in got.items()} == \
            {c: None if r is None else round(r.depth, 6)
             for c, r in want.items()}
    finally:
        machine_mod.cell_map = real
        sup.request_shutdown(); th.join(10)


def test_a_refit_that_resolves_nothing_reports_no_radius():
    """`ViewState.r` is Optional: a refit where NO cell resolved has no mean
    depth to report, and a constant dressed as a measurement would lie in
    `run.json` and in the record alike. The marker sphere still needs some
    radius to be drawn at all — it falls back to the grid default, and the
    log says that is what happened."""
    from inspection.view.grid import DEFAULT_R
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        cells = list(sup._reach)
        sup._cell_map = lambda: {c: None for c in cells}
        n0 = len(bus.published)
        sup._recenter()
        assert sup._reach and all(r is None for r in sup._reach.values())
        assert sup._view_state(0, "survey").r is None
        assert np.isclose(sup.sphere.r, DEFAULT_R), sup.sphere.r
        msgs = [p["msg"] for _, p in bus.published[n0:]
                if isinstance(p, dict) and "msg" in p]
        assert any("no cell resolved" in m for m in msgs), msgs
    finally:
        sup.request_shutdown(); th.join(10)


# --- Stage B: the three-leg move transaction ---------------------------------


class RecordingRig(FakeRig):
    """A `FakeRig` that keeps every move's WAYPOINTS and validation world.

    `FakeRig.moves` records path lengths only, and a transaction is exactly
    the question of which waypoints travelled under which world."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.paths, self.worlds = [], []

    def move(self, path, world="live"):
        self.paths.append([np.asarray(q, float) for q in path])
        self.worlds.append(world)
        return super().move(path, world=world)


class StopMidLegRig(RecordingRig):
    """Halts ONE waypoint into the `stop_at`-th move — the scripted stand-in
    for `run/stop` landing mid-slide, which is what leaves the arm parked
    between two ray depths with a half-executed transaction behind it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.stop_at, self.calls = 0, 0

    def move(self, path, world="live"):
        self.calls += 1
        if self.calls != self.stop_at:
            return super().move(path, world=world)
        assert len(path) > 2, "the scripted stop needs a leg to stop INSIDE"
        rep = super().move(path[:2], world=world)
        self.stop_event.set()
        return {**rep, "stopped": True}


class _RefuseFirstHops:
    """The real two-tier ladder with its first `n` answers refused.

    `plan_transaction` walks the current ray's out-depths current-depth-
    first, so "no slide-out at all" is always candidate 0 — and on this
    world the hop wins it every time. Refusing the first answers is how a
    test gets a THREE-leg transaction without inventing a world the arm is
    boxed into."""

    def __init__(self, n):
        self.n, self.calls = n, 0

    def __call__(self, *a, **kw):
        from inspection.motion.plan import plan_viewpoint
        self.calls += 1
        if self.calls <= self.n:
            return None, {"reason": "forced refusal", "tier": None}
        return plan_viewpoint(*a, **kw)


def _forced_slide_out(n=1):
    """A `plan_transaction` stand-in whose hop refuses its first `n` tries."""
    from inspection.motion.transaction import plan_transaction
    return lambda *a, **kw: plan_transaction(
        *a, planner=_RefuseFirstHops(n), **kw)


def _available_cells(sup):
    return [c for c, r in sup._reach.items()
            if r is not None and c not in sup.visited]


def _phases(bus, n0):
    """The phase strip as it was published from `n0` on."""
    return [p["phase"] for t, p in bus.published[n0:] if t == "run/status"]


def _at_previewing(sup, bus, cell, timeout=60):
    """Request `cell` and wait until the STATUS STRIP says previewing — not
    merely until `sup.phase` flips, which happens one line earlier and would
    let a phase-trace assertion anchor before its own publish."""
    sup.events.put({"cmd": "view/request", "target": list(cell)})
    wait_for(lambda: bus.last("run/status")["phase"] == "previewing",
             timeout=timeout, msg=f"preview {cell}")


def _cell_move(sup, bus, cell, timeout=60):
    """One full operator turn to `cell`: request, approve, wait for the fuse."""
    _at_previewing(sup, bus, cell, timeout)
    sup.events.put({"cmd": "view/confirm", "target": list(cell)})
    wait_for(lambda: sup.phase == "idle", timeout=timeout,
             msg=f"settled {cell}")


def test_first_move_after_a_survey_has_no_slide_out():
    """Legs 2-3 only (design §Move transaction): a survey leaves the arm on
    no ray, so there is nothing to slide out ALONG. Arrival then installs the
    ray state the NEXT move's slide-out is planned from — reconstructed from
    the resolution that was actually executed, never from the live cell map
    (which the fuse has already moved under it)."""
    from inspection.motion.transaction import RayState
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        assert sup._ray_state is None, "a survey leaves the arm on no ray"
        cell = _available_cells(sup)[0]
        _at_previewing(sup, bus, cell)
        legs, bounds, res = sup._legs, list(sup._leg_bounds), sup._target_res
        assert list(legs.out) == [], "no current ray, so no slide-out"
        assert bounds[0] == 0, f"the empty out leg must end at the start: {bounds}"
        assert bounds[-1] == len(sup._path) - 1, bounds
        sup.events.put({"cmd": "view/confirm", "target": list(cell)})
        wait_for(lambda: cell in sup.visited, timeout=60, msg="visited")

        st = sup._ray_state
        assert isinstance(st, RayState), st
        assert st.depth == res.depth and st.roll == res.roll
        assert st.samples is res.samples, "the slide-out corridor IS the ray's"
        # The ray is read back off the executed resolution: +X of a camera
        # pose looks AT the anchor (faceview.face_cam_pose), so anchor =
        # endpoint - depth * direction.
        T = np.asarray(res.T_cam, float)
        assert np.allclose(st.ray.direction, -T[:3, 0], atol=1e-9)
        assert np.allclose(np.asarray(st.ray.anchor)
                           + np.asarray(st.ray.direction) * st.depth,
                           T[:3, 3], atol=1e-9)
    finally:
        sup.request_shutdown(); th.join(10)


def test_a_move_off_a_ray_slides_out_hops_and_slides_in():
    """The full three-leg lifecycle on the fake rig: each leg is its own
    `rig.move`, run under the phase the plan gives it (slides `adjusting`,
    the hop `executing`) and re-validated against the world the LEGS were
    planned under — never the live world, which the next refit will move.
    The capture happens once, at the end of the slide-in."""
    import inspection.run.machine as machine_mod
    real = machine_mod.plan_transaction
    sup, rig, bus, th = make_sup(rig_cls=RecordingRig,
                                 q_start=DEMO_PARK.copy() + np.radians(
                                     [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        first = _available_cells(sup)[0]
        _cell_move(sup, bus, first)
        assert sup._ray_state is not None, "the arm stands on a ray now"

        machine_mod.plan_transaction = _forced_slide_out()
        cell = [c for c in _available_cells(sup) if c != first][0]
        _at_previewing(sup, bus, cell)
        legs = sup._legs
        assert len(legs.out) >= 1, "the fixture must force a slide-out"
        assert len(legs.hop) >= 2 and len(legs.slide_in) >= 2
        n_moves, n0 = len(rig.paths), len(bus.published)
        sup.events.put({"cmd": "view/confirm", "target": list(cell)})
        wait_for(lambda: cell in sup.visited and sup.phase == "idle",
                 timeout=60, msg="visited")

        phases = _phases(bus, n0)
        assert phases[:4] == ["executing", "adjusting", "executing",
                              "adjusting"], phases
        assert "capturing" in phases and phases[-1] == "idle", phases
        moved = rig.paths[n_moves:]
        assert len(moved) == 3, f"one rig.move per leg, got {len(moved)}"
        assert np.allclose(moved[0][1:], legs.out)
        assert np.allclose(moved[1], legs.hop)
        assert np.allclose(moved[2], legs.slide_in)
        assert rig.worlds[n_moves:] == [legs.world] * 3, \
            "each leg is judged by the world it was planned under"
        # The capture stands at the END of the slide-in, not at the hop's
        # landing: the step's stamped joints are the slide-in's last waypoint.
        step = json.load(open(sorted((sup.outdir / "steps").iterdir())[-1]
                              / "step.json"))
        assert np.allclose(step["joints_rad"], legs.slide_in[-1], atol=1e-6)
    finally:
        machine_mod.plan_transaction = real
        sup.request_shutdown(); th.join(10)


def test_a_transition_inaccessible_cell_is_refused_and_nothing_moves():
    """`plan_transaction` returning None is the honest answer "no out-depth
    from which the hop plans" — a refusal that NAMES that class, blocks the
    address, and never sends the arm anywhere (design §Failure semantics: no
    fallback substitution)."""
    import inspection.run.machine as machine_mod
    real = machine_mod.plan_transaction
    sup, rig, bus, th = make_sup(rig_cls=RecordingRig,
                                 q_start=DEMO_PARK.copy() + np.radians(
                                     [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        machine_mod.plan_transaction = lambda *a, **kw: None
        cell = _available_cells(sup)[0]
        n_moves, n0 = len(rig.paths), len(bus.published)
        sup.events.put({"cmd": "view/request", "target": list(cell)})
        wait_for(lambda: cell in sup.blocked and sup.phase == "idle",
                 timeout=60, msg="refusal returned to idle")
        msgs = [p["msg"] for _, p in bus.published[n0:]
                if isinstance(p, dict) and "msg" in p]
        assert any("transition-inaccessible" in m for m in msgs), msgs
        assert len(rig.paths) == n_moves, "a refused transition moved the arm"
        assert sup._ray_state is None and sup._legs is None
    finally:
        machine_mod.plan_transaction = real
        sup.request_shutdown(); th.join(10)


def test_a_stop_mid_slide_leaves_the_executed_waypoints_to_retrace():
    """`_arrival_path` is the EXECUTED transaction, not the planned one.

    A stop one waypoint into the slide-in leaves the arm between two ray
    depths with a hop and a slide-out behind it; the retreat's retrace tier
    reverses exactly those waypoints (design: the retreat machinery is the
    executor of a normal state). Retracing the PLANNED path instead would
    jump the arm from where it stopped to the second-to-last waypoint of a
    leg it never finished."""
    import inspection.run.machine as machine_mod
    real, real_m = machine_mod.plan_transaction, machine_mod.RETREAT_M
    sup, rig, bus, th = make_sup(rig_cls=StopMidLegRig,
                                 q_start=DEMO_PARK.copy() + np.radians(
                                     [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        first = _available_cells(sup)[0]
        _cell_move(sup, bus, first)

        machine_mod.plan_transaction = _forced_slide_out()
        cell = [c for c in _available_cells(sup) if c != first][0]
        _at_previewing(sup, bus, cell)
        legs, bounds, path = sup._legs, list(sup._leg_bounds), list(sup._path)
        assert len(legs.out) >= 1 and len(legs.slide_in) >= 3, \
            "the fixture must give all three legs something to travel"
        rig.stop_at = rig.calls + 3           # out, hop, then halt in slide-in
        sup.events.put({"cmd": "view/confirm", "target": list(cell)})
        wait_for(lambda: sup.phase == "idle", timeout=60, msg="stopped -> idle")

        # One waypoint into the slide-in: the concatenated path up to and
        # including the hop's landing, plus the one slide-in step travelled.
        executed = path[:bounds[1] + 2]
        assert len(sup._arrival_path) == len(executed), \
            (len(sup._arrival_path), len(executed))
        assert np.allclose(sup._arrival_path, executed)
        assert cell not in sup.visited and sup._ray_state is None
        assert np.allclose(rig.q(), executed[-1]), "the arm halted there"

        # Three refusals from that half-way stance, then the retrace: radial
        # escape is made impossible the way the workspace boundary makes it
        # impossible (no IK branch 5 m out).
        machine_mod.plan_transaction = lambda *a, **kw: None
        machine_mod.RETREAT_M = 5.0
        n_moves, tried = len(rig.paths), set()
        for n in (1, 2, 3):
            nxt = [c for c in _available_cells(sup) if c not in tried][0]
            tried.add(nxt)
            sup.events.put({"cmd": "view/request", "target": list(nxt)})
            if n < 3:
                wait_for(lambda n=n: sup._plan_fails == n and
                         sup.phase == "idle", timeout=60, msg=f"refusal {n}")
        wait_for(lambda: len(rig.paths) == n_moves + 1, timeout=60,
                 msg="retreat move")
        wait_for(lambda: sup.phase == "idle", timeout=60, msg="idle after")
        assert np.allclose(rig.paths[-1], list(reversed(executed))), \
            "the retreat must retrace exactly what was executed"
        rows = [json.loads(l) for l in
                (sup.writer.dir / "events.jsonl").read_text().splitlines()]
        assert "retrace" in [r for r in rows if r["kind"] == "retreat"][-1] \
            ["detail"]
    finally:
        machine_mod.plan_transaction, machine_mod.RETREAT_M = real, real_m
        sup.request_shutdown(); th.join(10)


def test_a_retreat_clears_the_ray_state():
    """The retreat moves the arm OFF the ray it was standing on, so the ray
    state (leg 1's `current`) is no longer true of anything — leaving it set
    would plan the next move's slide-out along a ray the camera has left."""
    import inspection.run.machine as machine_mod
    real = machine_mod.plan_transaction
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        cell = _available_cells(sup)[0]
        _cell_move(sup, bus, cell)
        assert sup._ray_state is not None and sup._arrival_path is not None

        machine_mod.plan_transaction = lambda *a, **kw: None
        tried = set()
        for n in (1, 2, 3):
            nxt = [c for c in _available_cells(sup) if c not in tried][0]
            tried.add(nxt)
            sup.events.put({"cmd": "view/request", "target": list(nxt)})
            if n < 3:
                wait_for(lambda n=n: sup._plan_fails == n and
                         sup.phase == "idle", timeout=60, msg=f"refusal {n}")
            else:
                wait_for(lambda: sup.phase == "idle" and sup._plan_fails == 0,
                         timeout=60, msg="idle after retreat")
        assert sup._ray_state is None, "the arm left the ray it stood on"
        assert sup._arrival_path is None
    finally:
        machine_mod.plan_transaction = real
        sup.request_shutdown(); th.join(10)


def main():
    test_survey_request_reaches_previewing()
    test_invalid_commands_dropped()
    test_stale_plan_result_discarded()
    test_supersede_during_planning_defers()
    test_duplicate_view_request_while_planning_is_ignored()
    test_blocked_plan_returns_to_idle()
    test_full_cycle_and_visited()
    test_survey_runs_through_the_segmenter_when_injected()
    test_survey_falls_back_to_depth_when_the_mask_misses()
    test_stop_mid_execute_returns_to_idle()
    test_capture_fail_not_visited()
    test_executor_fault_enters_fault_and_exit_works()
    test_exit_during_planning_defers_until_plan_done()
    test_bad_plane_view_rejected()
    test_bad_plane_pose_confirmed_fuses()
    test_exit_during_planning_fires_on_stale_plan_done()
    test_queued_confirm_cannot_undo_a_ctrl_c()
    test_stop_mid_execute_records_the_event_before_exit()
    test_join_workers_after_full_boot()
    test_malformed_target_is_rejected_not_raised()
    test_marker_id_target_from_a_3d_click()
    test_idle_publishes_the_real_pose()
    test_three_plan_refusals_trigger_retreat()
    test_sigint_saves_and_exits()
    test_face_gate_opens_after_two_stable_fits()
    test_face_request_full_cycle()
    test_face_refusal_does_not_enter_blocked()
    test_views_payload_carries_faces()
    test_face_request_previews_on_the_wire()
    test_request_gate_refuses_an_inaccessible_cell()
    test_view_r_is_the_mean_resolved_cell_depth()
    test_cell_map_measures_the_unpadded_belief_box()
    test_a_refit_that_resolves_nothing_reports_no_radius()
    test_first_move_after_a_survey_has_no_slide_out()
    test_a_move_off_a_ray_slides_out_hops_and_slides_in()
    test_a_transition_inaccessible_cell_is_refused_and_nothing_moves()
    test_a_stop_mid_slide_leaves_the_executed_waypoints_to_retrace()
    test_a_retreat_clears_the_ray_state()
    print("OK test_machine (task 5)")


if __name__ == "__main__":
    main()
