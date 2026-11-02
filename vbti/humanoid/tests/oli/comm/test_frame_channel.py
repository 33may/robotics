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


# ── multi-client (MAY-173 slam-demo-loop 1.6 root-cause fix) ─────────────────────
# The channel was documented for plural consumers ("dev app, brain/SLAM") but the
# server accepted exactly ONE client — a second consumer (the recorder; later the
# localizer) connected into the listen queue and starved silently. The server must
# fan out to every connected client, each with its own mailbox (a slow client never
# stalls a fast one), and survive any client dying.


def _poll(fn, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.01)
    return None


def test_two_clients_both_receive(tmp_path):
    path = str(tmp_path / "frames.sock")
    srv = FrameChannelServer(socket_path=path)
    srv.serve()
    c1 = FrameChannelClient(socket_path=path)
    c2 = FrameChannelClient(socket_path=path)
    c1.connect(timeout=5.0)
    c2.connect(timeout=5.0)
    try:
        for i in range(5):
            srv.publish(encode_camera_frame(_frame(stamp=100 + i)))
            time.sleep(0.02)
        f1 = _poll(c1.read_latest)
        f2 = _poll(c2.read_latest)
        assert f1 is not None and f2 is not None
        assert decode_camera_frame(f1).stamp_ns >= 100
        assert decode_camera_frame(f2).stamp_ns >= 100
    finally:
        c1.close()
        c2.close()
        srv.close()


def test_late_joining_client_receives(tmp_path):
    """A client that connects AFTER frames started flowing (Start Recording mid-
    session) must receive subsequent frames."""
    path = str(tmp_path / "frames.sock")
    srv = FrameChannelServer(socket_path=path)
    srv.serve()
    c1 = FrameChannelClient(socket_path=path)
    c1.connect(timeout=5.0)
    try:
        srv.publish(encode_camera_frame(_frame(stamp=1)))
        assert _poll(c1.read_latest) is not None
        late = FrameChannelClient(socket_path=path)
        late.connect(timeout=5.0)
        try:
            for i in range(5):
                srv.publish(encode_camera_frame(_frame(stamp=200 + i)))
                time.sleep(0.02)
            f = _poll(late.read_latest)
            assert f is not None and decode_camera_frame(f).stamp_ns >= 200
        finally:
            late.close()
    finally:
        c1.close()
        srv.close()


def test_dead_client_does_not_stop_others(tmp_path):
    path = str(tmp_path / "frames.sock")
    srv = FrameChannelServer(socket_path=path)
    srv.serve()
    c1 = FrameChannelClient(socket_path=path)
    c2 = FrameChannelClient(socket_path=path)
    c1.connect(timeout=5.0)
    c2.connect(timeout=5.0)
    try:
        c2.close()                                   # dies mid-session
        for i in range(5):
            srv.publish(encode_camera_frame(_frame(stamp=300 + i)))
            time.sleep(0.02)
        f = _poll(c1.read_latest)
        assert f is not None and decode_camera_frame(f).stamp_ns >= 300
    finally:
        c1.close()
        srv.close()
