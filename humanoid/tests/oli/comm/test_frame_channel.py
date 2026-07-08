"""Integration TDD for the camera frame channel over a real UDS SOCK_STREAM (§7).

`FrameChannelServer` (World side) ⇄ `FrameChannelClient` (brain side), no mocks. Pins
design.md D6/D7:
  - a separate stream channel from the SEQPACKET control socket
  - `publish()` is O(1) latest-wins (never blocks the control loop)
  - the client read drains to the newest frame
  - publishing with no client connected is a harmless no-op

Brain-pure (stdlib sockets + the pure wire): runs in the `brain` env.
"""

import time

import numpy as np
import pytest

from humanoid.logic.oli import CameraFrame, CameraIntrinsics
from humanoid.logic.oli.comm.codec import decode_camera_frame, encode_camera_frame
from humanoid.logic.oli.comm.frame_channel import FrameChannelClient, FrameChannelServer

pytestmark = pytest.mark.brain


def _frame(w: int = 8, h: int = 4, stamp: int = 0) -> CameraFrame:
    return CameraFrame(
        stamp_ns=stamp,
        name="chest",
        rgb=np.full((h, w, 3), stamp % 256, dtype=np.uint8),
        depth=np.full((h, w), 1.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(width=w, height=h, fx=5.0, fy=5.0, cx=4.0, cy=2.0),
    )


def _pair(tmp_path):
    sock = str(tmp_path / "frames.sock")
    server = FrameChannelServer(socket_path=sock)
    server.serve()
    client = FrameChannelClient(socket_path=sock)
    client.connect(timeout=5.0)
    return server, client


def _poll(fn, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.01)
    return None


def test_frame_roundtrip_over_channel(tmp_path):
    server, client = _pair(tmp_path)
    try:
        f = _frame(stamp=7)
        server.publish(encode_camera_frame(f))
        raw = _poll(client.read_latest)
        assert raw is not None, "brain never received the frame"
        g = decode_camera_frame(raw)
        assert g.stamp_ns == 7
        np.testing.assert_array_equal(g.rgb, f.rgb)
    finally:
        client.close()
        server.close()


def test_frame_latest_wins(tmp_path):
    server, client = _pair(tmp_path)
    try:
        for s in range(1, 21):
            server.publish(encode_camera_frame(_frame(stamp=s)))

        def newest():
            raw = client.read_latest()
            if raw is None:
                return None
            g = decode_camera_frame(raw)
            return g if g.stamp_ns == 20 else None

        assert _poll(newest, timeout=3.0) is not None, "the newest frame (20) must arrive"
    finally:
        client.close()
        server.close()


def test_read_latest_none_when_no_new_frame(tmp_path):
    server, client = _pair(tmp_path)
    try:
        server.publish(encode_camera_frame(_frame(stamp=1)))
        assert _poll(client.read_latest) is not None  # consume the one frame
        assert client.read_latest() is None  # nothing new pending
    finally:
        client.close()
        server.close()


def test_publish_without_client_is_noop(tmp_path):
    sock = str(tmp_path / "frames.sock")
    server = FrameChannelServer(socket_path=sock)
    server.serve()
    try:
        server.publish(encode_camera_frame(_frame(stamp=1)))  # no client connected
        time.sleep(0.05)  # no exception, server stays alive
    finally:
        server.close()


def test_publish_is_nonblocking(tmp_path):
    server, client = _pair(tmp_path)
    try:
        t0 = time.monotonic()
        for s in range(200):
            server.publish(encode_camera_frame(_frame(stamp=s)))
        dt = time.monotonic() - t0
        assert dt < 0.5, f"publish must be near-instant (mailbox drop), took {dt:.2f}s"
    finally:
        client.close()
        server.close()
