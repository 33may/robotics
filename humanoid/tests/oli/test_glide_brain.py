"""TDD for the brain-side glide path (MAY-172): GlideAction + the loop generalization.

Glide reuses the WHOLE Orchestrator loop; the only differences are (1) the Action forwards
the operator's velocity Intent as a `GlideCmd` instead of running the walk ONNX, and (2) the
Comm sends whatever the Action returns, dispatching `GlideCmd` → `GLIDE_CMD`. The walk path
(`PolicyOut` → `CMD`) is unchanged. Pure: runs in the `brain` env.
"""

from collections import deque

import numpy as np
import pytest

from humanoid.logic.oli import Mode, Observation, PolicyIn, PolicyOut
from humanoid.logic.oli.contracts import Intent
from humanoid.logic.oli.glide import GlideAction, GlideCmd
from humanoid.logic.oli.comm.base import Comm
from humanoid.logic.oli.reason.teleoperation.joystick import Teleop
from humanoid.logic.oli.reason.teleoperation.joystick.protocol import JoyPacket
from humanoid.logic.oli.runtime import Orchestrator

pytestmark = pytest.mark.brain

N = 31


def _obs(stamp):
    return Observation(
        stamp_ns=stamp, q=np.zeros(N), dq=np.zeros(N), tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _pin(stamp, v_x=0.0, v_y=0.0, w_z=0.0):
    return PolicyIn(observation=_obs(stamp),
                    intent=Intent(mode=Mode.WALK, v_x=v_x, v_y=v_y, w_z=w_z))


def _pout(stamp):
    z = np.zeros(N, dtype=np.float32)
    return PolicyOut(stamp_ns=stamp, q_des=z, dq_des=z, tau_ff=z,
                     kp=np.ones(N), kd=np.ones(N), mode=np.zeros(N, dtype=np.int32))


# ── GlideAction: forward the velocity Intent verbatim ─────────────────────────

def test_glide_action_forwards_intent_velocity_as_glide_cmd():
    cmd = GlideAction().step(_pin(1234, v_x=0.4, v_y=-0.1, w_z=0.2))
    assert isinstance(cmd, GlideCmd)
    assert cmd.stamp_ns == 1234
    assert (cmd.v_x, cmd.v_y, cmd.w_z) == pytest.approx((0.4, -0.1, 0.2))


def test_glide_action_speed_scale_multiplies_velocity():
    """speed_scale amplifies glide/turn (the demo boots at 3×); default stays verbatim."""
    cmd = GlideAction(speed_scale=3.0).step(_pin(7, v_x=0.4, v_y=-0.1, w_z=0.2))
    assert (cmd.v_x, cmd.v_y, cmd.w_z) == pytest.approx((1.2, -0.3, 0.6))


# ── Comm.send dispatches by message type ──────────────────────────────────────

class RecordingComm(Comm):
    def __init__(self):
        self.policy_outs = []
        self.glide_cmds = []

    def connect(self, timeout=10.0):
        pass

    def read_observation(self):
        return None

    def write_policy_out(self, po):
        self.policy_outs.append(po)

    def write_glide_cmd(self, cmd):
        self.glide_cmds.append(cmd)

    def close(self):
        pass


def test_comm_send_dispatches_policy_out_and_glide_cmd():
    comm = RecordingComm()
    comm.send(_pout(1))
    comm.send(GlideCmd(stamp_ns=2, v_x=0.1))
    assert len(comm.policy_outs) == 1 and comm.policy_outs[0].stamp_ns == 1
    assert len(comm.glide_cmds) == 1 and comm.glide_cmds[0].v_x == pytest.approx(0.1)


# ── Orchestrator in glide mode: same loop, GlideCmd out ───────────────────────

class SendOnlyComm:
    """Comm-shaped fake supporting ONLY send() — forces the loop off write_policy_out."""

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


def test_orchestrator_sends_glide_cmd_in_glide_mode():
    comm = SendOnlyComm([_obs(0)])
    orch = Orchestrator(comm, Teleop(mode=Mode.WALK), GlideAction(),
                        joystick=FakeJoystick([0.0, 0.5, 0.0, 0.0]))  # axis1 → v_x=0.5
    out = orch.step_once()
    assert isinstance(out, GlideCmd)
    assert len(comm.sent) == 1 and isinstance(comm.sent[0], GlideCmd)
    assert comm.sent[0].v_x == pytest.approx(0.5)


def test_glide_step_survives_obs_trace_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("OLI_TRACE", str(tmp_path / "trace.jsonl"))
    comm = SendOnlyComm([_obs(0)])
    orch = Orchestrator(comm, Teleop(mode=Mode.WALK), GlideAction(),
                        joystick=FakeJoystick([0.0, 0.3, 0.0, 0.0]))
    out = orch.step_once()  # the q_des trace must not choke on a GlideCmd
    assert isinstance(out, GlideCmd)
