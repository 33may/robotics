"""TDD for IsaacCameraSource — the dev app's LIVE camera source (MAY-149 ⇄ MAY-150 seam).

Drives real encoded frames through a real `FrameChannelServer` into the source (no Isaac),
proving the CameraPanel will see live chest/head RGBD exactly as it does the synthetic source.
Brain-pure.
"""

import time

import numpy as np
import pytest

from humanoid.logic.oli import CameraFrame, CameraIntrinsics
from humanoid.logic.oli.comm.codec import encode_camera_frame
from humanoid.logic.oli.comm.frame_channel import FrameChannelServer
from humanoid.logic.oli.devapp.sources.isaac_camera_source import IsaacCameraSource

pytestmark = pytest.mark.brain


def _frame(name: str, stamp: int, w: int = 8, h: int = 4) -> CameraFrame:
    return CameraFrame(
        stamp_ns=stamp,
        name=name,
        rgb=np.full((h, w, 3), stamp % 256, dtype=np.uint8),
        depth=np.full((h, w), float(stamp), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=w, height=h, fx=5.0, fy=5.0, cx=4.0, cy=2.0),
    )


def _poll(fn, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.01)
    return None


def test_reads_live_chest_and_head_without_clobber(tmp_path):
    sock = str(tmp_path / "frames.sock")
    server = FrameChannelServer(socket_path=sock)
    server.serve()
    src = IsaacCameraSource(sock, names=("chest", "head"), connect_timeout=5.0)
    try:
        for _ in range(3):  # both cameras every tick, a few ticks
            server.publish(encode_camera_frame(_frame("chest", stamp=1)))
            server.publish(encode_camera_frame(_frame("head", stamp=2)))
        chest = _poll(lambda: src.read("chest"))
        head = _poll(lambda: src.read("head"))
        assert chest is not None and chest.name == "chest" and chest.rgb.dtype == np.uint8
        assert head is not None and head.name == "head"
        assert chest.depth.dtype == np.float32
    finally:
        src.close()
        server.close()


def test_stream_names_advertised_before_any_frame(tmp_path):
    """The panel must be able to lay out its streams immediately, before frames flow."""
    sock = str(tmp_path / "frames.sock")
    server = FrameChannelServer(socket_path=sock)
    server.serve()
    src = IsaacCameraSource(sock, names=("chest", "head"), connect_timeout=5.0)
    try:
        assert src.stream_names() == ["chest", "head"]     # configured, up front
        assert src.read("chest") is None                    # nothing arrived yet → None
    finally:
        src.close()
        server.close()


def test_never_blocks_when_world_absent(tmp_path):
    """No frame socket → the source must construct + read instantly (UI never stalls)."""
    src = IsaacCameraSource(str(tmp_path / "nope.sock"), connect_timeout=0.1)
    try:
        t0 = time.monotonic()
        assert src.read("chest") is None
        assert src.stream_names() == ["chest", "head"]
        assert time.monotonic() - t0 < 0.2, "read/stream_names must not block on connect"
    finally:
        src.close()


def test_close_is_idempotent(tmp_path):
    src = IsaacCameraSource(str(tmp_path / "nope.sock"), connect_timeout=0.1)
    src.close()
    src.close()  # must not raise
