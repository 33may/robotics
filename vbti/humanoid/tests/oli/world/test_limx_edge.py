"""TDD for the LimX edge loop's command logic (`edge_tick`).

`edge_tick` is the only new logic in the bus edge: freeze-until-first-cmd (D9),
hold-last between policy steps, and watchdog-damp when the brain goes stale. We drive
it with a fake WorldComm + fake body (no limxsdk, no sockets) so it's a pure brain test.
"""

import numpy as np
import pytest

from humanoid.logic.oli.contracts import NUM_JOINTS, PolicyOut

pytestmark = pytest.mark.brain

N = NUM_JOINTS


def _po(stamp=1):
    return PolicyOut(
        stamp_ns=stamp, q_des=np.arange(N, dtype=np.float32),
        dq_des=np.zeros(N), tau_ff=np.zeros(N),
        kp=np.full(N, 50.0), kd=np.full(N, 2.0), mode=np.zeros(N, dtype=np.int32),
    )


class FakeWorldComm:
    def __init__(self, cmd_script):
        self._script = list(cmd_script)  # per-tick receive_latest() returns
        self.published_stamps = []
        self.applied = []  # PolicyOuts applied (republished to bus)

    def publish(self, stamp_ns):
        self.published_stamps.append(stamp_ns)

    def receive_latest(self):
        return self._script.pop(0) if self._script else None

    def apply(self, policy_out):
        self.applied.append(policy_out)


class FakeBody:
    def __init__(self):
        self.damp_calls = []

    def latest_stamp_ns(self):
        return 7

    def read_joints_isaac(self):
        q = np.full(N, 0.3, dtype=np.float32)
        return q, np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

    def apply_isaac(self, q_des, dq_des, tau_ff, kp, kd):
        self.damp_calls.append({"q": np.asarray(q_des), "kp": np.asarray(kp),
                                "kd": np.asarray(kd)})


def _tick(wc, body, state):
    from humanoid.logic.simulation.mujoco.limx_world_main import edge_tick

    return edge_tick(wc, body, state, watchdog_ticks=3, kd_damp=5.0)


def _state():
    from humanoid.logic.simulation.mujoco.limx_world_main import EdgeState

    return EdgeState()


def test_freeze_until_first_command():
    wc, body, st = FakeWorldComm([None]), FakeBody(), _state()
    _tick(wc, body, st)
    assert wc.published_stamps == [7]      # always publishes the observation
    assert wc.applied == []                # but applies nothing pre-first-cmd (D9)
    assert body.damp_calls == []
    assert st.last is None


def test_applies_fresh_command():
    po = _po(stamp=11)
    wc, body, st = FakeWorldComm([po]), FakeBody(), _state()
    _tick(wc, body, st)
    assert wc.applied == [po]
    assert st.last is po and st.stale_ticks == 0 and st.n_cmds == 1


def test_holds_last_command_when_brain_silent():
    po = _po()
    wc, body, st = FakeWorldComm([po, None, None]), FakeBody(), _state()
    _tick(wc, body, st)   # fresh
    _tick(wc, body, st)   # silent → hold
    _tick(wc, body, st)   # silent → hold
    assert wc.applied == [po, po, po]      # republished each tick
    assert body.damp_calls == []           # not stale yet (watchdog_ticks=3)
    assert st.stale_ticks == 2


def test_watchdog_damps_when_stale():
    po = _po()
    # fresh, then 4 silent ticks → stale_ticks reaches 4 > watchdog_ticks(3)
    wc, body, st = FakeWorldComm([po, None, None, None, None]), FakeBody(), _state()
    for _ in range(5):
        _tick(wc, body, st)
    # hold-last republishes every tick while stale<=3 (ticks 1-4), then damps (tick 5)
    assert wc.applied == [po, po, po, po]
    assert len(body.damp_calls) == 1       # only the stale tick damped in place
    d = body.damp_calls[-1]
    np.testing.assert_allclose(d["kp"], np.zeros(N))      # kp=0 (pure damping)
    np.testing.assert_allclose(d["kd"], np.full(N, 5.0))  # small kd
    np.testing.assert_allclose(d["q"], np.full(N, 0.3))   # hold current q
