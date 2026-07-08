"""TDD for CameraPublisher — the World-side camera→frame-channel unit (C2, MAY-149).

`CameraPublisher` reads every camera off Oli each render tick, wraps each as a
`CameraFrame`, and serves them on the dedicated frame channel. It is import-pure
(duck-types Oli — no isaacsim), so it is loopback-tested in the `brain` env with a
`FakeOli`, the REAL `FrameChannelServer` (owned internally), and a real
`CameraStreamReader` on the other end:
  - both cameras arrive as decoded CameraFrames, keyed by name (no clobber)
  - the World supplies the stamp (design.md D8)
  - `every=N` throttles publishing to the render sub-tick (§4.3)
"""

import time

import numpy as np
import pytest

from humanoid.logic.oli import CameraIntrinsics
from humanoid.logic.oli.comm.camera_publisher import CameraPublisher
from humanoid.logic.oli.comm.camera_stream import CameraStreamReader

pytestmark = pytest.mark.brain


class FakeOli:
    """Stand-in for the camera-enabled Oli: two cameras, distinct constant frames."""

    def __init__(self, w: int = 8, h: int = 4) -> None:
        self._w, self._h = w, h
        self._val = {"chest": 10, "head": 20}

    @property
    def camera_names(self):
        return ["chest", "head"]

    def read_camera_rgbd(self, name: str):
        v = self._val[name]
        rgb = np.full((self._h, self._w, 3), v, dtype=np.uint8)
        depth = np.full((self._h, self._w), float(v) / 10.0, dtype=np.float32)
        return rgb, depth

    def camera_intrinsics(self, name: str):
        return CameraIntrinsics(width=self._w, height=self._h, fx=5.0, fy=5.0, cx=4.0, cy=2.0)


def _poll(fn, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.01)
    return None


def test_publisher_serves_both_streams(tmp_path):
    oli = FakeOli()
    sock = str(tmp_path / "frames.sock")
    pub = CameraPublisher(oli, socket_path=sock)
    reader = CameraStreamReader(socket_path=sock)
    reader.connect(timeout=5.0)
    try:
        for tick in range(3):
            pub.publish(tick, stamp_ns=100 + tick)
        chest = _poll(lambda: reader.read("chest"))
        head = _poll(lambda: reader.read("head"))
        assert chest is not None and chest.name == "chest"
        assert head is not None and head.name == "head"
        np.testing.assert_array_equal(chest.rgb, np.full((4, 8, 3), 10, dtype=np.uint8))
        np.testing.assert_array_equal(head.rgb, np.full((4, 8, 3), 20, dtype=np.uint8))
    finally:
        reader.close()
        pub.close()


def test_world_supplies_stamp(tmp_path):
    oli = FakeOli()
    sock = str(tmp_path / "frames.sock")
    pub = CameraPublisher(oli, socket_path=sock)
    reader = CameraStreamReader(socket_path=sock)
    reader.connect(timeout=5.0)
    try:
        pub.publish(0, stamp_ns=4242)
        f = _poll(lambda: reader.read("chest"))
        assert f is not None and f.stamp_ns == 4242
    finally:
        reader.close()
        pub.close()


def test_every_throttles_publishing(tmp_path):
    """every=5 → publish only on ticks that are multiples of 5 (§4.3 sub-tick)."""
    oli = FakeOli()
    sock = str(tmp_path / "frames.sock")
    pub = CameraPublisher(oli, socket_path=sock, every=5)
    reader = CameraStreamReader(socket_path=sock)
    reader.connect(timeout=5.0)
    try:
        for tick in (1, 2, 3, 4):  # none are multiples of 5
            pub.publish(tick, stamp_ns=tick)
        time.sleep(0.1)
        assert reader.read("chest") is None, "no frame should publish off the sub-tick"
        pub.publish(5, stamp_ns=5)
        assert _poll(lambda: reader.read("chest")) is not None
    finally:
        reader.close()
        pub.close()


class FlakyOli(FakeOli):
    """An Oli whose cameras aren't ready on the first read (Isaac annotators need a render
    tick to populate) — read_camera_rgbd raises until warmed."""

    def __init__(self) -> None:
        super().__init__()
        self._fail = {"chest": 1, "head": 1}

    def read_camera_rgbd(self, name: str):
        if self._fail[name] > 0:
            self._fail[name] -= 1
            raise RuntimeError("camera frame not ready yet")
        return super().read_camera_rgbd(name)


def test_publisher_survives_unready_camera(tmp_path):
    """A camera that isn't ready must NOT crash the World's loop — the publisher skips it
    and ships the frame once it renders. (Isaac kills the process on an uncaught error.)"""
    oli = FlakyOli()
    sock = str(tmp_path / "frames.sock")
    pub = CameraPublisher(oli, socket_path=sock)
    reader = CameraStreamReader(socket_path=sock)
    reader.connect(timeout=5.0)
    try:
        pub.publish(0, stamp_ns=0)  # cameras not ready → must NOT raise
        pub.publish(1, stamp_ns=1)  # warmed → frames flow
        chest = _poll(lambda: reader.read("chest"))
        head = _poll(lambda: reader.read("head"))
        assert chest is not None and chest.stamp_ns == 1
        assert head is not None and head.stamp_ns == 1
    finally:
        reader.close()
        pub.close()


def test_close_is_idempotent(tmp_path):
    oli = FakeOli()
    pub = CameraPublisher(oli, socket_path=str(tmp_path / "frames.sock"))
    pub.close()
    pub.close()  # must not raise
