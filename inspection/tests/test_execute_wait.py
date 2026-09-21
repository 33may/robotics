#!/usr/bin/env python3
"""_wait_async poll logic, no robot. Run: p inspection/tests/test_execute_wait.py"""
import threading
import time

from inspection.motion.execute import _wait_async


def _seq(values, after=False):
    """running_fn that yields `values` then `after` forever."""
    it = iter(values)
    return lambda: next(it, after)


def _counted(fn):
    """Wrap a running_fn; .calls tracks how many polls consumed."""
    def wrapped():
        wrapped.calls += 1
        return fn()
    wrapped.calls = 0
    return wrapped


def test_done_after_motion():
    r = _wait_async(_seq([True, True, False]), lambda: True, poll_s=0.001)
    assert r == "done"


def test_grace_prevents_premature_done():
    # controller hasn't registered the op yet: not-running at first poll,
    # then running, then finished — must NOT return done on the first poll
    wrapped = _counted(_seq([False, False, True, False]))
    r = _wait_async(wrapped, lambda: True, poll_s=0.001, grace_s=10.0)
    assert r == "done"
    assert wrapped.calls == 4  # graceless impl returns after 1 call


def test_never_ran_times_out_via_grace():
    wrapped = _counted(lambda: False)
    t0 = time.monotonic()
    r = _wait_async(wrapped, lambda: True, poll_s=0.001, grace_s=0.02)
    elapsed = time.monotonic() - t0
    assert r == "done"          # grace expired, op never registered
    assert wrapped.calls >= 2   # graceless version does exactly 1 poll
    assert elapsed >= 0.02      # grace window must have elapsed


def test_stopped():
    ev = threading.Event(); ev.set()
    r = _wait_async(lambda: True, lambda: True, stop_event=ev, poll_s=0.001)
    assert r == "stopped"


def test_unsafe():
    r = _wait_async(lambda: True, lambda: False, poll_s=0.001)
    assert r == "unsafe"


def test_stop_beats_a_move_that_has_already_finished():
    """A stop that lands on the same poll as the move's completion reports
    "stopped", never "done".

    `execute()` counts a waypoint only when the poll says "done", and the
    machine now RECONSTRUCTS where the arm is from that count: a three-leg
    transaction's executed waypoints are what the retreat retraces in
    reverse (`machine._exec_worker`). A poll that called a stopped waypoint
    done would record a waypoint the arm never reached, and the retrace's
    first segment would start from a configuration the arm is not in.
    """
    ev = threading.Event(); ev.set()
    r = _wait_async(lambda: False, lambda: True, stop_event=ev, poll_s=0.001,
                    grace_s=0.0)
    assert r == "stopped"


# ── execute(): one blended path move ────────────────────────────────────────
# The live demo's first feedback: per-waypoint moveJ decelerates to a full
# stop at every 2 cm slide sample — steppy motion. The fix sends the whole
# leg as ONE path-form moveJ with blend radii; these tests drive execute()
# with a fake controller (no robot).

import numpy as np

from inspection.motion.execute import BLEND_M, UR5eArm


class _Status:
    def __init__(self, running, progress):
        self.progress = progress
        self._r = running

    def isAsyncOperationRunning(self):
        return self._r


class _PathCtrl:
    """Records moveJ/stopJ; scripts the async progress polls. When the
    scripted statuses run out it repeats the last one and, if `on_exhaust`
    is set, fires it once (used to raise a stop mid-path)."""

    def __init__(self, statuses, on_exhaust=None):
        self.calls, self.stops = [], []
        self._seq = list(statuses)
        self._on_exhaust = on_exhaust

    def moveJ(self, *args):
        self.calls.append(args)

    def getAsyncOperationProgressEx(self):
        if len(self._seq) > 1:
            return self._seq.pop(0)
        if self._on_exhaust is not None:
            cb, self._on_exhaust = self._on_exhaust, None
            cb()
        return self._seq[0]

    def stopJ(self, a):
        self.stops.append(a)


def _arm(ctrl, q0):
    arm = UR5eArm.__new__(UR5eArm)
    arm.ctrl, arm.speed, arm.acc = ctrl, 0.5, 0.8
    arm._assert_fresh = lambda diagnose=True: None
    arm._safety_ok = lambda: True
    arm.q = lambda: np.asarray(q0, dtype=float)
    return arm


def _path(n, step=0.05):
    """n collinear waypoints, one joint stepping by `step` rad (a slide)."""
    return [np.array([0.1 * j for j in range(6)]) + [i * step, 0, 0, 0, 0, 0]
            for i in range(n)]


def test_execute_is_one_blended_path_move():
    path = _path(6)
    ctrl = _PathCtrl([_Status(True, 0), _Status(True, 4), _Status(False, -1)])
    rep = _arm(ctrl, path[0]).execute(path)
    assert len(ctrl.calls) == 1, "the whole leg must be ONE moveJ call"
    rows, asynchronous = ctrl.calls[0]
    assert asynchronous is True
    assert len(rows) == 5 and all(len(r) == 9 for r in rows)
    for r in rows:
        assert r[6] == 0.5 and r[7] == 0.8
    assert all(r[8] == BLEND_M for r in rows[:-1]), "interior waypoints blend"
    assert rows[-1][8] == 0.0, "the endpoint is reached exactly"
    assert rep["stopped"] is False and rep["waypoints_done"] == 5


def test_tiny_segment_zeroes_the_blend():
    path = _path(5)
    path.insert(2, path[2] - 1e-4)   # near-duplicate: adjacent segs shrink
    ctrl = _PathCtrl([_Status(True, 0), _Status(False, -1)])
    _arm(ctrl, path[0]).execute(path)
    rows, _ = ctrl.calls[0]
    # rows around the tiny segment must not blend (the controller refuses a
    # blend wider than half its segment); the far ones keep it
    assert rows[0][8] == BLEND_M and rows[-2][8] == BLEND_M
    assert rows[1][8] == 0.0 and rows[2][8] == 0.0
    assert rows[-1][8] == 0.0


def test_stop_mid_path_counts_only_reached_waypoints():
    ev = threading.Event()
    ctrl = _PathCtrl([_Status(True, 0), _Status(True, 1), _Status(True, 2)],
                     on_exhaust=ev.set)
    rep = _arm(ctrl, _path(6)[0]).execute(_path(6), stop_event=ev)
    assert rep["stopped"] is True
    # progress 2 = row 2 in flight: rows 0 and 1 are the only ones the
    # retrace may trust (undercount by design — an overcount would retrace
    # forward through a waypoint the stop prevented)
    assert rep["waypoints_done"] == 2
    assert ctrl.stops, "a stop must decelerate via stopJ"


def test_unsafe_mid_path_raises_and_stops():
    ctrl = _PathCtrl([_Status(True, 0), _Status(True, 1)])
    arm = _arm(ctrl, _path(4)[0])
    flips = iter([True, True, False])
    arm._safety_ok = lambda: next(flips, False)
    try:
        arm.execute(_path(4))
        assert False, "safety change mid-path must raise"
    except RuntimeError:
        pass
    assert ctrl.stops, "the except path must command a stop"


def main():
    test_done_after_motion(); test_grace_prevents_premature_done()
    test_never_ran_times_out_via_grace(); test_stopped(); test_unsafe()
    test_stop_beats_a_move_that_has_already_finished()
    test_execute_is_one_blended_path_move()
    test_tiny_segment_zeroes_the_blend()
    test_stop_mid_path_counts_only_reached_waypoints()
    test_unsafe_mid_path_raises_and_stops()
    print("OK test_execute_wait")


if __name__ == "__main__":
    main()
