"""TDD for the fenced ground-truth pose debug/eval channel (comm/debug_pose.py).

NOT the invariance spine: a separate AF_UNIX datagram channel carrying the World's ground-truth
base pose to the brain, for day-1 planner bring-up (before a real localizer exists) and later
estimated-vs-truth localizer evaluation. The codec is pure (struct); the server/client are
exercised over a real loopback UDS. Pure stdlib, so it runs in the `brain` env.
"""

import time

import pytest

from humanoid.logic.oli.comm.debug_pose import (
    POSE_NBYTES,
    DebugPoseClient,
    DebugPoseServer,
    decode_pose,
    encode_pose,
)

pytestmark = pytest.mark.brain


# ── codec (pure) ─────────────────────────────────────────────────────────────


def test_pose_frame_is_32_bytes():
    assert POSE_NBYTES == 32
    assert len(encode_pose(1, 0.0, 0.0, 0.0)) == 32


def test_encode_decode_roundtrip():
    assert decode_pose(encode_pose(123, 1.5, -2.25, 0.75)) == (123, 1.5, -2.25, 0.75)


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError):
        decode_pose(b"\x00" * 16)


# ── transport (real loopback UDS) ────────────────────────────────────────────


def _drain_until(client: DebugPoseClient, timeout: float = 1.0):
    deadline = time.monotonic() + timeout  # monotonic clock (not wall-clock)
    while time.monotonic() < deadline:
        p = client.latest()
        if p is not None:
            return p
        time.sleep(0.005)
    return None


def test_publish_then_latest_delivers_pose(tmp_path):
    path = str(tmp_path / "pose.sock")
    client = DebugPoseClient(path)  # binds first
    server = DebugPoseServer(path)
    try:
        server.publish(10, 1.0, 2.0, 0.3)
        assert _drain_until(client) == (10, 1.0, 2.0, 0.3)
    finally:
        server.close()
        client.close()


def test_latest_wins_drains_to_newest(tmp_path):
    path = str(tmp_path / "pose.sock")
    client = DebugPoseClient(path)
    server = DebugPoseServer(path)
    try:
        for i in range(5):
            server.publish(i, float(i), 0.0, 0.0)
        assert _drain_until(client) == (4, 4.0, 0.0, 0.0)  # one drain → freshest
    finally:
        server.close()
        client.close()


def test_latest_is_none_before_any_publish(tmp_path):
    path = str(tmp_path / "pose.sock")
    client = DebugPoseClient(path)
    try:
        assert client.latest() is None
    finally:
        client.close()
