"""TDD for the Orchestrator (the brain loop).

`read → reason → act → write`, paced by the World stamp (one step per ≥policy_dt of
advancing sim time), latest-wins, with the first observation always stepping (this is
the freeze-until-cmd handshake: the brain bootstraps the World off the frozen pose).
Owns optional recording of every contract. Tested with fakes (no sockets, no ONNX) so
the loop logic is isolated. Pure: runs in the `brain` env.
"""

from collections import deque

import numpy as np
import pytest

from humanoid.logic.oli import Mode, Observation, PolicyIn, PolicyOut
from humanoid.logic.oli.reason.teleoperation.joystick import Teleop
from humanoid.logic.oli.reason.teleoperation.joystick.protocol import JoyPacket
from humanoid.logic.oli.runtime import Orchestrator

pytestmark = pytest.mark.brain

N = 31
MS = 1_000_000


def _obs(stamp):
    return Observation(
        stamp_ns=stamp, q=np.zeros(N), dq=np.zeros(N), tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _pout(stamp):
    z = np.zeros(N, dtype=np.float32)
    return PolicyOut(stamp_ns=stamp, q_des=z, dq_des=z, tau_ff=z,
                     kp=np.ones(N), kd=np.ones(N), mode=np.zeros(N, dtype=np.int32))


class FakeComm:
    def __init__(self, obs_list):
        self._q = deque(obs_list)
        self.sent = []
        self.connected = False

    def connect(self, timeout=10.0):
        self.connected = True

    def read_observation(self):
        return self._q.popleft() if self._q else None

    def send(self, msg):
        self.sent.append(msg)

    def close(self):
        pass


class FakeJoystick:
    def __init__(self, axes):
        self._pkt = JoyPacket(stamp_ns=0, axes=list(axes), buttons=[])

    def poll(self):
        return self._pkt


class FakeAction:
    def __init__(self):
        self.calls = []

    def step(self, policy_in):
        self.calls.append(policy_in)
        return _pout(policy_in.observation.stamp_ns)


def test_step_once_runs_full_cycle():
    comm = FakeComm([_obs(0)])
    action = FakeAction()
    orch = Orchestrator(comm, Teleop(mode=Mode.WALK), action,
                        joystick=FakeJoystick([0.0, 0.5, 0.0, 0.0]))
    out = orch.step_once()
    assert isinstance(out, PolicyOut)
    assert len(comm.sent) == 1
    assert action.calls[0].intent.mode == Mode.WALK
    assert action.calls[0].intent.v_x == pytest.approx(0.5)


def test_returns_none_when_no_observation():
    comm = FakeComm([])
    orch = Orchestrator(comm, Teleop(), FakeAction())
    assert orch.step_once() is None
    assert comm.sent == []


def test_first_obs_steps_then_paces_by_stamp():
    comm = FakeComm([_obs(0), _obs(5 * MS), _obs(10 * MS)])
    orch = Orchestrator(comm, Teleop(mode=Mode.WALK), FakeAction(), policy_dt_ns=10 * MS)
    orch.step_once()  # stamp 0   → first step (freeze handshake)
    orch.step_once()  # stamp 5ms → <10ms since last step → skip
    orch.step_once()  # stamp 10ms → ≥10ms → step
    assert [p.stamp_ns for p in comm.sent] == [0, 10 * MS]


def test_run_connects_then_loops():
    comm = FakeComm([_obs(0), _obs(20 * MS)])
    orch = Orchestrator(comm, Teleop(mode=Mode.WALK), FakeAction(), policy_dt_ns=10 * MS)
    orch.run(should_continue=lambda: len(comm.sent) < 2)
    assert comm.connected
    assert len(comm.sent) == 2


def test_recorder_receives_every_contract():
    rec = []
    comm = FakeComm([_obs(0)])
    orch = Orchestrator(comm, Teleop(mode=Mode.WALK), FakeAction(),
                        recorder=lambda o, pi, po, j: rec.append((o, pi, po)))
    orch.step_once()
    assert len(rec) == 1
    o, pi, po = rec[0]
    assert isinstance(o, Observation) and isinstance(pi, PolicyIn) and isinstance(po, PolicyOut)
