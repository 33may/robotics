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
    keeps answering with cached state, exactly like ur_rtde #307. Its
    reconnect fails too — the unrecoverable case."""

    def __init__(self):
        self.reconnects = 0

    def getTimestamp(self):
        return 1234.500

    def getActualQ(self):
        return [0.1, -1.2, 2.0, -1.5, 1.6, 0.3]

    def getSafetyMode(self):
        return 1        # cached NORMAL — the dangerous lie

    def getRobotMode(self):
        return 7        # cached RUNNING

    def reconnect(self):
        self.reconnects += 1
        return False


class LiveRecv(FrozenRecv):
    """Healthy stream: the controller timestamp advances per packet."""

    def __init__(self):
        super().__init__()
        self._t = 1234.500

    def getTimestamp(self):
        self._t += 0.002        # 500 Hz packets
        return self._t


class HealingRecv(FrozenRecv):
    """Dead stream that a reconnect actually revives — the common case on
    this cell (the stream died in all three 2026-08-21 hardware runs)."""

    def getTimestamp(self):
        if not self.reconnects:
            return 1234.500
        self._t = getattr(self, "_t", 1234.500) + 0.002
        return self._t

    def reconnect(self):
        self.reconnects += 1
        return True


class LyingRecv(FrozenRecv):
    """reconnect() claims success but the stream stays frozen. Trust the
    timestamp, not the return value."""

    def reconnect(self):
        self.reconnects += 1
        return True


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


def test_dead_stream_heals_by_reconnect():
    # Stability is the operator's requirement: a dead stream that a
    # reconnect revives must hand back FRESH data, not abort the run.
    arm = make_arm(HealingRecv())
    q = arm.q()
    assert arm.recv.reconnects == 1, "guard did not attempt the reconnect"
    assert isinstance(q, np.ndarray) and q.shape == (6,)
    # ...and the healed stream serves later calls without reconnecting again.
    arm.q()
    assert arm.recv.reconnects == 1


def test_lying_reconnect_still_raises():
    arm = make_arm(LyingRecv())
    try:
        arm.q()
    except RuntimeError as e:
        assert "stale" in str(e).lower()
    else:
        raise AssertionError("a reconnect that healed nothing was believed")
    assert arm.recv.reconnects == 1


def main():
    test_q_fresh_passes()
    test_q_frozen_raises()
    test_safety_fresh_passes()
    test_safety_frozen_raises()
    test_frozen_detection_is_bounded()
    test_dead_stream_heals_by_reconnect()
    test_lying_reconnect_still_raises()
    print("OK test_freshness")


if __name__ == "__main__":
    main()


# ── D1: a healthy stream must never be declared dead by a slow scheduler ────

class SlowClockRecv(LiveRecv):
    """Healthy 500 Hz stream, but the first two reads land on the SAME
    packet — which is legitimate — and the process is then descheduled so
    that `time.sleep(0.002)` returns well after the poll window closed.

    This is the loaded-machine case. It reproduces on a busy box and never
    on an idle one, which is exactly the profile of a bug that survived 23
    isolated reproduction attempts and killed every real run.
    """

    def __init__(self, oversleep=0.5):
        super().__init__()
        self._reads = 0
        self._oversleep = oversleep
        self.real_sleep = time.sleep

    def getTimestamp(self):
        self._reads += 1
        if self._reads <= 2:
            return self._t          # same 2 ms packet — legitimate
        self._t += 0.002            # stream is alive and advancing
        return self._t


#: Captured before any test patches them. `execute` does `import time`, so
#: patching `ex.time.monotonic` mutates the shared module — restoring from a
#: value read *after* patching would reinstall the fake.
_REAL_MONOTONIC = time.monotonic
_REAL_SLEEP = time.sleep


class _descheduled:
    """Simulates a loaded machine: reads are cheap, but `sleep(0.002)` returns
    only after `oversleep` seconds because the thread lost the CPU.

    The overshoot must land on the SLEEP, not between two adjacent reads —
    that is the real failure. Two back-to-back `getTimestamp()` calls are
    microseconds apart; it is the sleep in between that can stretch.
    """

    def __init__(self, oversleep=0.5):
        self.oversleep = oversleep
        self.t = 0.0

    def _mono(self):
        self.t += 0.0001            # a read costs ~nothing
        return self.t

    def _sleep(self, _s):
        self.t += self.oversleep    # ... and then the scheduler forgets us

    def __enter__(self):
        time.sleep = self._sleep
        time.monotonic = self._mono
        return self

    def __exit__(self, *exc):
        time.sleep = _REAL_SLEEP
        time.monotonic = _REAL_MONOTONIC


def test_slow_scheduler_does_not_kill_a_live_stream():
    """The regression: `_ts_advancing` checked its deadline at the TOP of the
    loop, so an overshooting sleep returned False without ever re-reading —
    declaring a healthy stream dead and tearing down a working socket."""
    recv = SlowClockRecv()
    arm = make_arm(recv)
    with _descheduled():
        assert arm._ts_advancing() is True, \
            "healthy stream declared dead when the scheduler overshot"
    assert recv.reconnects == 0, "tore down a socket that was fine"


def test_frozen_stream_still_detected_under_a_slow_clock():
    """The fix must not blunt the guard: a genuinely frozen stream must still
    be caught even when the clock leaps."""
    arm = make_arm(FrozenRecv())
    with _descheduled():
        assert arm._ts_advancing() is False


# ── D3: the autopsy must stay out of the mid-move safety path ──────────────

def test_safety_poll_does_not_run_the_autopsy():
    """`_safety_ok` is polled every 50 ms DURING a move by the same thread
    that watches `stop_event`. Forking `ss` there (5 s timeout, under
    `_recv_lock`) would stall STOP with the arm travelling. Diagnostics
    belong on the q() path, where a pause is harmless."""
    import inspection.motion.execute as ex
    calls = []
    real = ex.stream_autopsy
    ex.stream_autopsy = lambda recv, ip=None: calls.append("ran") or "stub"
    try:
        arm = make_arm(HealingRecv())
        arm._safety_ok()
        assert calls == [], "autopsy ran inside the mid-move safety poll"
        arm2 = make_arm(HealingRecv())
        arm2.q()
        assert calls == ["ran"], "autopsy should still run on the q() path"
    finally:
        ex.stream_autopsy = real


# ── D2: a dead stream must not become a reconnect storm ────────────────────

def test_reconnect_is_rate_limited():
    """`PoseStreamer` swallows the RuntimeError and retries at 30 Hz, and the
    safety poll runs at 20 Hz. Without a backoff, an unrecoverable stream
    means ~50 reconnect attempts a second against the controller — a connect
    storm that is itself a plausible way to get hung up on."""
    recv = FrozenRecv()               # never heals
    arm = make_arm(recv)
    arm.reconnect_backoff = 5.0
    for _ in range(20):               # 20 rapid callers, as the threads do
        try:
            arm.q()
        except RuntimeError:
            pass
    assert recv.reconnects == 1, (
        f"{recv.reconnects} reconnect attempts in a burst — expected 1 "
        "until the backoff expires")


def test_backoff_expires_and_allows_a_later_retry():
    """The latch must not be permanent: a stream that comes back later still
    has to be healable."""
    recv = FrozenRecv()
    arm = make_arm(recv)
    arm.reconnect_backoff = 0.0       # expired immediately
    for _ in range(3):
        try:
            arm.q()
        except RuntimeError:
            pass
    assert recv.reconnects == 3
