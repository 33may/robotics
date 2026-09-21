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
from inspection.cell.world import RobotCell
from inspection.fit.contract import Primitive
from inspection.motion.ik import UR5eIK
from inspection.motion.plan import DEMO_PARK
from inspection.record.run import Run
from inspection.record.writer import RunWriter
from inspection.run.app import config_snapshot, viewsphere_method
from inspection.run.collect import (SweepDriver, _default_planner, joint_l1,
                                    rank_candidates)
from inspection.run.rigs import FakeRig
from inspection.run.segmenter import ObjectSegmenter
from inspection.tests.test_flow_a import DirectPlanSupervisor, StubPub, wait_for
from inspection.view.rays import RayResolution, box_corners, resolve_cell
from inspection.view.viewsphere import T_FLANGE_CAM


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


# ---------------------------------------------------------------------------
# `_default_planner`: ranks a candidate by the REAL ray transaction's cost —
# `resolve_cell` + `plan_transaction`, both real, on a real collision world —
# never the old shell distance the KNOWN-DEGRADED note used to carry.

_BOX_CENTER = (0.05, -0.38, 0.0375)
_BOX_AABB = ((-0.025, -0.455, 0.0), (0.125, -0.305, 0.075))


def _transaction_world():
    """Real IK, real collision world, real ray samples (the Stage A box
    world, `test_transaction.py`'s fixture, copied) — the world
    `_default_planner`'s planner is meant to run on."""
    w = RobotCell()
    w.set_object("object", [0.15, 0.15, 0.075],
                 [0.05, -0.38, 0.0375, 0, 0, 0.17])
    return w, UR5eIK()


def _resolved(w, ik, h, v):
    res = resolve_cell(w, ik, T_FLANGE_CAM, _BOX_CENTER, _BOX_AABB, h, v,
                       fy=647.0, height_px=720)
    assert res is not None, f"fixture cell {(h, v)} must be accessible"
    return res


class _RankSup:
    """The four attributes `_default_planner` reads off a real
    Supervisor — `acc.centroid`/`_object_aabb`/`_face_prim`/`rig.intr` —
    the exact belief `machine._cell_map` feeds its OWN `resolve_cell`
    calls. Not a `Supervisor`: this is `_default_planner`'s own narrow
    contract, isolated from `StubSup` above (which the driver-loop tests
    own)."""

    class _Acc:
        def __init__(self, centroid):
            self.centroid = (None if centroid is None
                            else np.asarray(centroid, dtype=float))

    class _Rig:
        intr = {"fy": 647.0, "height": 720}

    def __init__(self, centroid=_BOX_CENTER, aabb=_BOX_AABB, face_prim=None):
        self.acc = self._Acc(centroid)
        self._object_aabb = aabb
        self._face_prim = face_prim
        self.rig = self._Rig()


def _prim(dims, pose6):
    """A synthetic boxguard `Primitive` at `(dims, pose6)` — a tighter,
    UNPADDED hull than the world's padded collision box, the way a real
    boxguard fit narrows `_object_aabb`'s margin (`test_machine.py`'s
    `_acc_prim`, built directly instead of fit from points so the test
    controls exactly how much tighter it is)."""
    cx, cy, cz, rx, ry, yaw = pose6
    c, s = np.cos(yaw), np.sin(yaw)
    pose = [[c, -s, 0.0, cx], [s, c, 0.0, cy],
           [0.0, 0.0, 1.0, cz], [0.0, 0.0, 0.0, 1.0]]
    return Primitive(kind="box", pose=pose, params={"extents": dims},
                     label="box")


def _always_refuses(world, ik, q_from, T_target, seed=0, **kw):
    """A hop planner that never finds a path — `plan_transaction`'s own
    `planner=` hook (`test_transaction.py`'s `_StubPlanner` pattern),
    stubbing only the hop so the transaction's real search and leg
    assembly still run."""
    return None, {"reason": "stub refusal"}


def test_default_planner_costs_the_real_transaction():
    """Cost is the transaction's joint-L1 over the WHOLE trip — q_now
    through every leg — not a shell-distance guess, and the transaction's
    own slide-in still lands on the resolved view pose."""
    w, ik = _transaction_world()
    q_now = np.asarray(_resolved(w, ik, 0, 2).q, dtype=float)
    planner = _default_planner(_RankSup())

    path, cost = planner(w, ik, q_now, (3, 2))

    assert path is not None
    assert np.allclose(path[0], q_now)
    assert cost == pytest.approx(joint_l1(path))
    target = _resolved(w, ik, 3, 2)
    assert np.allclose(ik.fk(path[-1]), target.T_flange, atol=1e-6)


def test_default_planner_ranks_cheaper_transaction_first():
    """Two real, resolvable cells cost differently by real joint travel —
    the whole point of replacing the shell-distance ranker — and a third,
    unresolvable cell never enters the ranking at all."""
    w, ik = _transaction_world()
    q_now = np.asarray(_resolved(w, ik, 0, 2).q, dtype=float)
    planner = _default_planner(_RankSup())
    assert resolve_cell(w, ik, T_FLANGE_CAM, _BOX_CENTER, _BOX_AABB, 0, 0,
                        fy=647.0, height_px=720) is None, \
        "fixture cell (0, 0) must be the unresolvable one"

    def cost_fn(cell):
        return planner(w, ik, q_now, cell)[1]

    ranked = rank_candidates([(6, 2), (3, 2), (0, 0)], cost_fn)

    assert ranked == [(3, 2), (6, 2)], "cheapest real transaction first"


def test_default_planner_drops_an_unresolvable_cell():
    """A cell whose ray never resolves costs (None, None) — dropped by
    `rank_candidates`, never crashed on."""
    w, ik = _transaction_world()
    q_now = np.asarray(_resolved(w, ik, 0, 2).q, dtype=float)
    planner = _default_planner(_RankSup())

    assert planner(w, ik, q_now, (0, 0)) == (None, None)


def test_default_planner_drops_a_cell_whose_transaction_refuses():
    """A cell that DOES resolve but whose transition cannot be planned —
    the KNOWN-DEGRADED gap the shell ranker used to disagree on — also
    costs (None, None): dropped, not crashed on."""
    w, ik = _transaction_world()
    q_now = np.asarray(_resolved(w, ik, 0, 2).q, dtype=float)
    planner = _default_planner(_RankSup(), hop_planner=_always_refuses)

    path, cost = planner(w, ik, q_now, (3, 2))     # (3, 2) resolves fine

    assert (path, cost) == (None, None)


def test_default_planner_needs_a_belief_to_rank_against():
    """No object yet -> no centroid/aabb -> honestly nothing to rank
    (`machine._cell_map`'s own guard, mirrored here) — never crashed on."""
    w, ik = _transaction_world()
    planner = _default_planner(_RankSup(centroid=None, aabb=None))

    assert planner(w, ik, np.zeros(6), (3, 2)) == (None, None)


def test_default_planner_tightens_the_belief_with_a_boxguard_fit():
    """Mirrors `machine._cell_map`'s ruled derivation exactly: `corners`
    feeds `resolve_cell`'s containment floor for a grid CELL
    unconditionally, so a fit tightens the belief this ranker measures too
    — not merely a face's. With `_face_prim` set, the belief comes from
    the prim's OWN unpadded hull; with none, the padded `_object_aabb`
    fallback holds. On this fixture the difference is real and visible:
    cell {3, 2} resolves 4 cm closer under the tighter hull, so ranking a
    cell against the padded box used to cost every candidate standoff it
    would not actually have to pay once a fit exists — the mismatch this
    task exists to kill."""
    w, ik = _transaction_world()
    q_now = np.asarray(_resolved(w, ik, 0, 2).q, dtype=float)
    prim = _prim([0.11, 0.11, 0.035], [*_BOX_CENTER, 0.0, 0.0, 0.17])
    corners = box_corners(prim)
    tight_hull = (corners.min(0), corners.max(0))

    with_fit = _default_planner(_RankSup(face_prim=prim))
    without_fit = _default_planner(_RankSup(face_prim=None))
    path_tight, _ = with_fit(w, ik, q_now, (3, 2))
    path_padded, _ = without_fit(w, ik, q_now, (3, 2))

    want_tight = resolve_cell(w, ik, T_FLANGE_CAM, _BOX_CENTER, tight_hull,
                              3, 2, fy=647.0, height_px=720, corners=corners)
    want_padded = resolve_cell(w, ik, T_FLANGE_CAM, _BOX_CENTER, _BOX_AABB,
                               3, 2, fy=647.0, height_px=720)

    assert want_tight.depth < want_padded.depth, \
        "fixture must make the tightened belief visibly closer"
    # The transaction's own slide-in always ends on the target's view pose
    # (proven by `test_default_planner_costs_the_real_transaction`), so the
    # planner's landing pose IS the resolution it used — equality with a
    # direct `resolve_cell` call proves which aabb/corners fed it.
    assert np.allclose(ik.fk(path_tight[-1]), want_tight.T_flange, atol=1e-6)
    assert np.allclose(ik.fk(path_padded[-1]), want_padded.T_flange,
                       atol=1e-6)
    assert not np.allclose(want_tight.T_flange, want_padded.T_flange), \
        "the two beliefs must actually resolve different view poses"


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
        # What `_mirror_object_box` reads now (`geometry.object_box` path):
        # an empty cloud -> no object yet -> nothing to mirror.
        points = np.empty((0, 3))

    class _Rig:
        def q(self):
            return np.zeros(6)

    def __init__(self, cells: int = 8):
        self.phase, self.target = "idle", None
        self.sphere, self.seed = None, 0
        self.acc, self.rig = self._Acc(), self._Rig()
        self.visited, self.blocked = set(), set()
        # RayResolution-shaped (Task 4): these driver-loop tests only ever
        # read `r is not None` off `_reach` — the fields below are never
        # inspected — but the type must be real, not a float lookalike,
        # now that `sup._reach` holds resolved rays everywhere else.
        self._reach = {(h, 0): RayResolution(None, None, None, 0.0, 0.0,
                                             0.0, 0.0, [])
                       for h in range(cells)}


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


class ApprovingMover:
    """A mover that approves everything and plays the Supervisor's part of
    the bargain: a successful request lands the cell in `visited`, which is
    how the candidate set actually shrinks in a real run."""

    def __init__(self, sup):
        self.sup, self.asked = sup, []

    def request(self, cell):
        self.asked.append(cell)
        if cell != "survey":
            self.sup.visited.add(cell)
        return True, "reached and captured"


def test_random_mode_shuffles_and_records_its_own_decider(tmp_path):
    """`--random` exists because nearest-first step order flatters
    convergence analysis (2026-09-14): the walk must not be the ranked
    order, must still exhaust the shell, and must say `sweep-random` in
    every step's `ViewState.decider` — the on-disk statement of policy."""
    sup = StubSup()
    driver = SweepDriver(sup, tmp_path, planner=_stub_planner,
                         random_order=True)
    assert driver.mover.decider == "sweep-random"
    driver.mover = ApprovingMover(sup)

    driver.start()
    driver.join(10)
    assert not driver.is_alive()
    assert driver.finished, "an approve-everything shell must be exhausted"
    cells = driver.mover.asked[1:]                    # after the survey
    assert set(cells) == set(sup._reach), "shuffling must not drop coverage"
    ranked_order = sorted(sup._reach, key=lambda c: c[0] + 10 * c[1])
    assert cells != ranked_order, "random order came out cheapest-first"

    # Seeded: the same sup/seed walks the same order — reruns are comparable.
    sup2 = StubSup()
    d2 = SweepDriver(sup2, tmp_path, planner=_stub_planner, random_order=True)
    d2.mover = ApprovingMover(sup2)
    d2.start()
    d2.join(10)
    assert d2.mover.asked == driver.mover.asked


def test_random_mode_retries_refused_passes_until_stopped(tmp_path):
    """Refusals can be transient (a missed approval window), so a refused
    pass retries from the top in random mode exactly as in cost mode — a
    first cut that ended the sweep on an unchanged candidate set died on
    that race. Hopeless runs end through `stop()`/fault, nothing else."""
    sup = StubSup()
    driver = SweepDriver(sup, tmp_path, planner=_stub_planner,
                         random_order=True)
    # Refuse everything; stop the driver mid-SECOND pass — proof it retried.
    second_pass_call = 1 + len(sup._reach) + 3
    driver.mover = CountingMover(
        lambda n, cell: driver.stop() if n == second_pass_call else None)

    driver.start()
    driver.join(10)
    assert not driver.is_alive()
    assert driver.finished is False
    assert driver.mover.calls == second_pass_call, driver.mover.asked


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
    test_default_planner_costs_the_real_transaction()
    test_default_planner_ranks_cheaper_transaction_first()
    test_default_planner_drops_an_unresolvable_cell()
    test_default_planner_drops_a_cell_whose_transaction_refuses()
    test_default_planner_needs_a_belief_to_rank_against()
    test_default_planner_tightens_the_belief_with_a_boxguard_fit()
    test_sweep_driver_visits_every_reachable_cell(Path(tempfile.mkdtemp()))
    # The driver-loop tests take pytest's `tmp_path`; a fresh mkdtemp is the
    # same contract, and without these four the script runner silently
    # covered less than the pytest run did.
    test_random_mode_shuffles_and_records_its_own_decider(
        Path(tempfile.mkdtemp()))
    test_random_mode_retries_refused_passes_until_stopped(
        Path(tempfile.mkdtemp()))
    test_a_faulted_run_ends_the_sweep(Path(tempfile.mkdtemp()))
    test_stop_ends_the_sweep(Path(tempfile.mkdtemp()))
    print("OK test_collect")
