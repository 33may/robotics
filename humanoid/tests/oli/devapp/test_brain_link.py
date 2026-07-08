"""Integration test: the dev app's BrainLink attaches to a World and drives it.

Uses the same import-pure `SimComm` + `FakeBody` loopback harness as the comm tests (a real
AF_UNIX socket, no Isaac) and a zero-action stub (no ONNX). Proves the "app IS the brain"
seam end-to-end: BrainLink connects on its worker thread, the Orchestrator steps, the fake
World receives commands, and the recorder publishes live contracts into AppState.
"""

import threading
import time

import numpy as np
import pytest

from humanoid.logic.oli import NUM_JOINTS as N
from humanoid.logic.oli import Mode, PolicyOut
from humanoid.logic.oli.devapp.brain_link import BrainLink
from humanoid.logic.oli.devapp.state import AppState
from humanoid.logic.oli.reason.teleoperation.joystick import FixedJoystick, Teleop
from humanoid.logic.simulation.isaacsim.sim_comm import SimComm

pytestmark = pytest.mark.brain


class _FakeBody:
    """Minimal slimmed-Oli stand-in; Isaac DOF order = reversed PR (non-trivial permute)."""

    def __init__(self):
        from humanoid.logic.oli import PR_ORDER
        self.dof_names = list(reversed(PR_ORDER))
        self.last_applied = None

    def read_joints_isaac(self):
        i = np.arange(N, dtype=np.float32)
        return i.copy(), i.copy(), i.copy()

    def read_imu(self):
        return (
            np.array([0.0, 0.0, -9.81], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def apply_isaac(self, q_des, dq_des, tau_ff, kp, kd):
        self.last_applied = np.asarray(q_des, dtype=np.float32).copy()

    def set_command_isaac(self, q_des, dq_des, tau_ff, kp, kd):
        self.apply_isaac(q_des, dq_des, tau_ff, kp, kd)


class _ZeroAction:
    """Action stub: emit a zero PolicyOut (avoids loading the walk ONNX in the test)."""

    def step(self, policy_in):
        z = np.zeros(N, dtype=np.float32)
        return PolicyOut(
            stamp_ns=policy_in.observation.stamp_ns,
            q_des=z, dq_des=z, tau_ff=z, kp=z, kd=z,
            mode=np.zeros(N, dtype=np.int32),
        )


def test_glide_adapter_maps_yaw_to_horizontal_right_is_right():
    """Glide adapter: yaw comes from the right-stick HORIZONTAL axis (2), drag-right = turn
    right (w_z < 0), and the vertical axis (3) no longer yaws. v_x/v_y stay on axes 1/0."""
    from humanoid.logic.oli.devapp.brain_link import GlideJoystickAdapter

    a = GlideJoystickAdapter()
    # drag right stick RIGHT (axis2 = +1) → turn right (sign verified live: w_z follows axis2)
    _, _, wz_right = a.axes_to_velocity([0.0, 0.0, 1.0, 0.0])
    assert wz_right > 0.0
    # drag LEFT (axis2 = -1) → turn left
    _, _, wz_left = a.axes_to_velocity([0.0, 0.0, -1.0, 0.0])
    assert wz_left < 0.0
    # the old vertical axis (3) must NO LONGER produce any yaw
    _, _, wz_vert = a.axes_to_velocity([0.0, 0.0, 0.0, 1.0])
    assert wz_vert == 0.0
    # forward/strafe unchanged: v_x←axis1, v_y←axis0
    vx, vy, _ = a.axes_to_velocity([0.2, 0.4, 0.0, 0.0])
    assert vx == pytest.approx(0.4) and vy == pytest.approx(0.2)


def test_brainlink_attaches_and_drives_fake_world(tmp_path):
    sock = str(tmp_path / "oli.sock")
    body = _FakeBody()
    server = SimComm(body, socket_path=sock)
    serve_t = threading.Thread(target=server.serve, kwargs={"timeout": 5.0}, daemon=True)
    serve_t.start()

    state = AppState()
    link = BrainLink(
        state,
        socket=sock,
        action=_ZeroAction(),                 # no ONNX
        reason=Teleop(mode=Mode.WALK),        # real reason
        joystick_source=FixedJoystick(0.1, 0.0, 0.0),  # real joystick
    )
    link.start()
    serve_t.join(timeout=5.0)
    assert not serve_t.is_alive(), "handshake never completed"

    try:
        # Drive a tiny World loop: publish advancing stamps, drain the brain's commands.
        stamp = 0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if state.brain_snapshot()[0] and body.last_applied is not None:
                break
            stamp += 20_000_000  # +20 ms so each obs clears the policy-dt gate
            server.publish(stamp_ns=stamp)
            cmd = server.receive_blocking(timeout=0.3)
            if cmd is not None:
                server.apply(cmd)

        assert link.error is None, f"brain thread errored: {link.error!r}"
        attached, obs, policy_out, mode = state.brain_snapshot()
        assert attached and obs is not None, "recorder never fed AppState"
        assert mode == "WALK"
        assert body.last_applied is not None, "World never received a command from the brain"
    finally:
        link.stop()
        server.close()


def test_brainlink_glide_forwards_velocity_to_world(tmp_path):
    """--mode glide → BrainLink builds GlideAction and the World receives a GlideCmd."""
    from humanoid.logic.oli.glide import GlideCmd

    sock = str(tmp_path / "oli.sock")
    body = _FakeBody()
    server = SimComm(body, socket_path=sock)
    serve_t = threading.Thread(target=server.serve, kwargs={"timeout": 5.0}, daemon=True)
    serve_t.start()

    state = AppState()
    link = BrainLink(
        state,
        socket=sock,
        mode="glide",                          # action defaults to GlideAction()
        reason=Teleop(mode=Mode.WALK),         # WALK intent → velocity forwarded
        joystick_source=FixedJoystick(0.3, 0.0, 0.1),
    )
    link.start()
    serve_t.join(timeout=5.0)
    assert not serve_t.is_alive(), "handshake never completed"

    try:
        got = None
        stamp = 0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and got is None:
            stamp += 20_000_000
            server.publish(stamp_ns=stamp)
            got = server.receive_glide_blocking(timeout=0.3)

        assert link.error is None, f"brain thread errored: {link.error!r}"
        assert isinstance(got, GlideCmd), "World never received a GlideCmd"
        assert abs(got.v_x) + abs(got.w_z) > 0.0, "velocity intent was not forwarded"
        attached, _obs, out, _mode = state.brain_snapshot()
        assert attached and isinstance(out, GlideCmd), "recorder did not publish the GlideCmd"
    finally:
        link.stop()
        server.close()
