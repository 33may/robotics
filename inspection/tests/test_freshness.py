#!/usr/bin/env python3
"""RTDE freshness guard, no robot. Run: p inspection/tests/test_freshness.py

The failure this guards against is real and silent: on run 2108-ui the
receive interface's background thread died mid-run (ur_rtde #307) and every
getter — joints AND safety mode — returned the same cached state for 4.5
minutes with no exception. Six captures stamped bit-identical joints while
the arm demonstrably moved. The guard's contract: a `q()` or `_safety_ok()`
whose `getTimestamp()` does not advance within the poll window must RAISE,
never hand back a frozen value.
"""
import time

import numpy as np

from inspection.motion.execute import UR5eArm


class FrozenRecv:
    """A receive interface whose background thread has died: every getter
    keeps answering with cached state, exactly like ur_rtde #307."""

    def getTimestamp(self):
        return 1234.500

    def getActualQ(self):
        return [0.1, -1.2, 2.0, -1.5, 1.6, 0.3]

    def getSafetyMode(self):
        return 1        # cached NORMAL — the dangerous lie

    def getRobotMode(self):
        return 7        # cached RUNNING


class LiveRecv(FrozenRecv):
    """Healthy stream: the controller timestamp advances per packet."""

    def __init__(self):
        self._t = 1234.500

    def getTimestamp(self):
        self._t += 0.002        # 500 Hz packets
        return self._t


def make_arm(recv):
    """An UR5eArm with no hardware behind it: __new__ skips the connecting
    __init__, which is exactly the point — the guard must be testable
    without a robot."""
    arm = UR5eArm.__new__(UR5eArm)
    arm.recv = recv
    arm.stale_window = 0.05     # short window: keeps the failing case fast
    return arm


def test_q_fresh_passes():
    arm = make_arm(LiveRecv())
    q = arm.q()
    assert isinstance(q, np.ndarray) and q.shape == (6,)


def test_q_frozen_raises():
    arm = make_arm(FrozenRecv())
    try:
        arm.q()
    except RuntimeError as e:
        assert "stale" in str(e).lower(), f"unhelpful error: {e}"
    else:
        raise AssertionError("q() returned a frozen pose instead of raising")


def test_safety_fresh_passes():
    arm = make_arm(LiveRecv())
    assert arm._safety_ok() is True


def test_safety_frozen_raises():
    # A frozen recv freezes the safety mode too: a cached NORMAL would let a
    # move continue through a protective stop. Raising (not returning False)
    # is deliberate — execute()'s except path calls stop() on the way out.
    arm = make_arm(FrozenRecv())
    try:
        arm._safety_ok()
    except RuntimeError as e:
        assert "stale" in str(e).lower(), f"unhelpful error: {e}"
    else:
        raise AssertionError("_safety_ok() answered from a frozen recv")


def test_frozen_detection_is_bounded():
    # The guard must give up within ~its window, not hang a safety poll.
    arm = make_arm(FrozenRecv())
    t0 = time.monotonic()
    try:
        arm.q()
    except RuntimeError:
        pass
    assert time.monotonic() - t0 < 1.0, "staleness detection took too long"


def main():
    test_q_fresh_passes()
    test_q_frozen_raises()
    test_safety_fresh_passes()
    test_safety_frozen_raises()
    test_frozen_detection_is_bounded()
    print("OK test_freshness")


if __name__ == "__main__":
    main()
