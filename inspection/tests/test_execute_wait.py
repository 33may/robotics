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


def main():
    test_done_after_motion(); test_grace_prevents_premature_done()
    test_never_ran_times_out_via_grace(); test_stopped(); test_unsafe()
    test_stop_beats_a_move_that_has_already_finished()
    print("OK test_execute_wait")


if __name__ == "__main__":
    main()
