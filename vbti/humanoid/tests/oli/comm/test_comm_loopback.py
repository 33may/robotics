"""Integration TDD for the Communication pair over a real UDS socket.

`BrainComm` (client) ⇄ `SimComm` (server) exchange contracts through the wire with
NO mocks — a real AF_UNIX socketpair and an injected `FakeBody`. This pins:
  - World=server / Brain=client connect + handshake (D10)
  - SimComm owns the PR↔Isaac permutation (D4): read permutes Isaac→PR, apply PR→Isaac
  - stamp is supplied by the World (sim time), not the body (D8)
  - latest-wins draining both directions

`SimComm` is import-pure (body injected, no isaacsim), so this runs in the `brain` env.
"""

import threading
import time

import numpy as np
import pytest

from humanoid.logic.oli import (
    NUM_JOINTS,
    PR_ORDER,
    CameraFrame,
    CameraIntrinsics,
    Mode,
    PolicyOut,
)
from humanoid.logic.oli.comm.client import BrainComm
from humanoid.logic.oli.comm.codec import encode_camera_frame
from humanoid.logic.oli.comm.frame_channel import FrameChannelServer
from humanoid.logic.oli.glide import GlideCmd
from humanoid.logic.simulation.isaacsim.sim_comm import SimComm

pytestmark = pytest.mark.brain

N = NUM_JOINTS


class FakeBody:
    """Stand-in for the slimmed Oli. Isaac DOF order = reversed PR, so the
    permutation is non-trivial: isaac index of PR joint pr_idx is (N-1-pr_idx)."""

    def __init__(self):
        self.dof_names = list(reversed(PR_ORDER))  # Isaac order
        self.last_applied = None
        self.last_command = None

    def read_joints_isaac(self):
        i = np.arange(N, dtype=np.float32)
        return i.copy(), i + 100.0, i + 200.0  # q, dq, tau (Isaac order)

    def read_imu(self):
        return (
            np.array([0.0, 0.1, -9.81], dtype=np.float32),
            np.array([0.3, -0.2, 0.1], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def apply_isaac(self, q_des, dq_des, tau_ff, kp, kd):
        self.last_applied = {
            "q": np.asarray(q_des, dtype=np.float32).copy(),
            "dq": np.asarray(dq_des, dtype=np.float32).copy(),
            "tau": np.asarray(tau_ff, dtype=np.float32).copy(),
            "kp": np.asarray(kp, dtype=np.float32).copy(),
            "kd": np.asarray(kd, dtype=np.float32).copy(),
        }

    def set_command_isaac(self, q_des, dq_des, tau_ff, kp, kd):
        self.last_command = {
            "q": np.asarray(q_des, dtype=np.float32).copy(),
            "kp": np.asarray(kp, dtype=np.float32).copy(),
        }


def _connected_pair(tmp_path):
    sock = str(tmp_path / "oli.sock")
    body = FakeBody()
    server = SimComm(body, socket_path=sock)
    t = threading.Thread(target=server.serve, kwargs={"timeout": 5.0}, daemon=True)
    t.start()
    client = BrainComm(socket_path=sock)
    client.connect(timeout=5.0)
    t.join(timeout=5.0)
    assert not t.is_alive(), "SimComm.serve did not complete handshake"
    return server, client, body


def _poll(fn, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.01)
    return None


def test_brain_reads_pr_ordered_observation_from_sim(tmp_path):
    server, client, body = _connected_pair(tmp_path)
    try:
        server.publish(stamp_ns=42)
        obs = _poll(client.read_observation)
        assert obs is not None, "brain never received the observation"
        assert obs.stamp_ns == 42  # World owns the stamp, not the body
        # Isaac order reversed → after Isaac→PR permute, q[pr_idx] = N-1-pr_idx
        expected_q = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        np.testing.assert_allclose(obs.q, expected_q, atol=1e-4)
        np.testing.assert_allclose(obs.dq, expected_q + 100.0, atol=1e-4)
        # IMU is base-frame, not permuted
        np.testing.assert_allclose(obs.acc, [0.0, 0.1, -9.81], atol=1e-4)
        np.testing.assert_allclose(obs.quat_wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-4)
    finally:
        client.close()
        server.close()


def test_sim_applies_pr_command_in_isaac_order(tmp_path):
    server, client, body = _connected_pair(tmp_path)
    try:
        q_des = np.arange(N, dtype=np.float32)  # PR order: q_des[pr_idx] = pr_idx
        po = PolicyOut(
            stamp_ns=7, q_des=q_des, dq_des=np.zeros(N), tau_ff=np.zeros(N),
            kp=np.full(N, 50.0), kd=np.full(N, 2.0), mode=np.zeros(N, dtype=np.int32),
        )
        client.write_policy_out(po)
        got = _poll(server.receive_latest)
        assert got is not None, "sim never received the command"
        server.apply(got)
        assert body.last_applied is not None
        # PR→Isaac permute: isaac[i] = q_des[N-1-i] = N-1-i
        expected_isaac_q = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        np.testing.assert_allclose(body.last_applied["q"], expected_isaac_q, atol=1e-4)
        np.testing.assert_allclose(body.last_applied["kp"], np.full(N, 50.0), atol=1e-4)
    finally:
        client.close()
        server.close()


def test_sim_stores_explicit_command_in_isaac_order(tmp_path):
    """Explicit-torque path: set_command uses the SAME PR→Isaac permutation as apply."""
    server, client, body = _connected_pair(tmp_path)
    try:
        q_des = np.arange(N, dtype=np.float32)  # PR order
        po = PolicyOut(
            stamp_ns=9, q_des=q_des, dq_des=np.zeros(N), tau_ff=np.zeros(N),
            kp=np.full(N, 50.0), kd=np.full(N, 2.0), mode=np.zeros(N, dtype=np.int32),
        )
        client.write_policy_out(po)
        got = _poll(server.receive_latest)
        assert got is not None, "sim never received the command"
        server.set_command(got)
        assert body.last_command is not None
        expected_isaac_q = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        np.testing.assert_allclose(body.last_command["q"], expected_isaac_q, atol=1e-4)
    finally:
        client.close()
        server.close()


def test_observation_latest_wins(tmp_path):
    server, client, body = _connected_pair(tmp_path)
    try:
        for s in (1, 2, 3):
            server.publish(stamp_ns=s)
        time.sleep(0.05)
        obs = _poll(client.read_observation)
        assert obs is not None and obs.stamp_ns == 3, "read must drain to the newest"
    finally:
        client.close()
        server.close()


def test_glide_cmd_round_trips_brain_to_world(tmp_path):
    """Glide mode (MAY-172): a GlideCmd travels the SAME socket as a new GLIDE_CMD frame."""
    server, client, body = _connected_pair(tmp_path)
    try:
        client.write_glide_cmd(GlideCmd(stamp_ns=7, v_x=0.4, v_y=-0.1, w_z=0.2))
        got = _poll(server.receive_glide_latest)
        assert got is not None, "World never received the glide command"
        assert isinstance(got, GlideCmd)
        assert got.stamp_ns == 7
        assert (got.v_x, got.v_y, got.w_z) == pytest.approx((0.4, -0.1, 0.2))
    finally:
        client.close()
        server.close()


def test_glide_receive_blocking_returns_latest(tmp_path):
    """Lock-step glide pacing: the World blocks until the brain's GlideCmd arrives."""
    server, client, body = _connected_pair(tmp_path)
    try:
        client.write_glide_cmd(GlideCmd(stamp_ns=11, v_x=0.25))
        got = server.receive_glide_blocking(timeout=2.0)
        assert got is not None and got.stamp_ns == 11
        assert got.v_x == pytest.approx(0.25)
    finally:
        client.close()
        server.close()


def test_receive_blocking_returns_latest_command(tmp_path):
    """Lock-step pacing: the World blocks until the brain's command arrives."""
    server, client, body = _connected_pair(tmp_path)
    try:
        client.write_policy_out(PolicyOut(
            stamp_ns=5, q_des=np.zeros(N), dq_des=np.zeros(N), tau_ff=np.zeros(N),
            kp=np.full(N, 1.0), kd=np.full(N, 1.0), mode=np.zeros(N, dtype=np.int32),
        ))
        got = server.receive_blocking(timeout=2.0)
        assert got is not None and got.stamp_ns == 5
    finally:
        client.close()
        server.close()


def test_receive_blocking_times_out_when_silent(tmp_path):
    """No command within the timeout → None (lets the World run its watchdog)."""
    server, client, body = _connected_pair(tmp_path)
    try:
        got = server.receive_blocking(timeout=0.2)  # brain sends nothing
        assert got is None
    finally:
        client.close()
        server.close()


def test_command_latest_wins(tmp_path):
    server, client, body = _connected_pair(tmp_path)
    try:
        for s in (10, 20, 30):
            client.write_policy_out(PolicyOut(
                stamp_ns=s, q_des=np.zeros(N), dq_des=np.zeros(N), tau_ff=np.zeros(N),
                kp=np.full(N, 1.0), kd=np.full(N, 1.0), mode=np.zeros(N, dtype=np.int32),
            ))
        time.sleep(0.05)
        got = _poll(server.receive_latest)
        assert got is not None and got.stamp_ns == 30, "receive must drain to the newest"
    finally:
        client.close()
        server.close()


# ── camera frame channel (§8): BrainComm reads CameraFrames off a 2nd socket ─────

def _cam_frame(stamp: int) -> CameraFrame:
    return CameraFrame(
        stamp_ns=stamp, name="chest",
        rgb=np.full((4, 8, 3), stamp % 256, dtype=np.uint8),
        depth=np.ones((4, 8), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=8, height=4, fx=5.0, fy=5.0, cx=4.0, cy=2.0),
    )


def test_brain_reads_camera_frame_from_world(tmp_path):
    """The brain connects BOTH the control socket and the frame socket; a frame
    published on the frame channel arrives as a decoded CameraFrame."""
    sock = str(tmp_path / "oli.sock")
    fsock = str(tmp_path / "frames.sock")
    body = FakeBody()
    server = SimComm(body, socket_path=sock)
    fserver = FrameChannelServer(socket_path=fsock)
    fserver.serve()
    t = threading.Thread(target=server.serve, kwargs={"timeout": 5.0}, daemon=True)
    t.start()
    client = BrainComm(socket_path=sock, frame_socket_path=fsock)
    client.connect(timeout=5.0)
    t.join(timeout=5.0)
    try:
        fserver.publish(encode_camera_frame(_cam_frame(11)))
        frame = _poll(client.read_camera_frame)
        assert frame is not None, "brain never received the camera frame"
        assert frame.stamp_ns == 11 and frame.name == "chest"
    finally:
        client.close()
        server.close()
        fserver.close()


def test_read_camera_frame_none_without_frame_channel(tmp_path):
    """A BrainComm with no frame_socket_path has no camera channel → always None."""
    server, client, body = _connected_pair(tmp_path)
    try:
        assert client.read_camera_frame() is None
    finally:
        client.close()
        server.close()
