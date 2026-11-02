"""TDD for the multi-stream camera reader (C1, MAY-149).

`CameraStreamReader` is the brain/consumer side of the frame channel. The raw
`FrameChannelClient` is SINGLE-SLOT: a chest frame arriving right after a head frame
overwrites it before the consumer reads (frame_channel.py `self._latest`). With TWO
cameras that clobbers. `CameraStreamReader` demuxes frames by camera name INSIDE its
receiver thread into a per-stream mailbox, so each stream's newest frame survives
independently.

Tested against a deterministic raw UDS server (both frames provably hit the wire), so
this pins the READER's demux — not any server mailbox policy. Brain-pure: runs in the
`brain` env.
"""

import os
import socket as _sock
import time

import numpy as np
import pytest

from humanoid.logic.oli import CameraFrame, CameraIntrinsics
from humanoid.logic.oli.comm.camera_stream import CameraStreamReader
from humanoid.logic.oli.comm.codec import encode_camera_frame
from humanoid.logic.oli.comm.frame_channel import FrameChannelServer

pytestmark = pytest.mark.brain


def _frame(name: str, stamp: int, w: int = 8, h: int = 4) -> CameraFrame:
    return CameraFrame(
        stamp_ns=stamp,
        name=name,
        rgb=np.full((h, w, 3), stamp % 256, dtype=np.uint8),
        depth=np.full((h, w), float(stamp), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=w, height=h, fx=5.0, fy=5.0, cx=4.0, cy=2.0),
    )


class _RawFrameServer:
    """Deterministic byte source: binds a UDS stream, sends whole encoded frames with
    `sendall` so every frame provably reaches the wire (no mailbox, no dropping)."""

    def __init__(self, path: str) -> None:
        self._path = path
        if os.path.exists(path):
            os.unlink(path)
        self._srv = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        self._srv.bind(path)
        self._srv.listen(1)
        self._conn = None

    def accept(self, timeout: float = 5.0) -> None:
        self._srv.settimeout(timeout)
        self._conn, _ = self._srv.accept()

    def send(self, frame: CameraFrame) -> None:
        self._conn.sendall(encode_camera_frame(frame))

    def close(self) -> None:
        for s in (self._conn, self._srv):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        if os.path.exists(self._path):
            try:
                os.unlink(self._path)
            except OSError:
                pass


def _reader(tmp_path):
    """Wire a reader to a raw server; connect() succeeds via the listen backlog, then
    the server dequeues with accept()."""
    sock = str(tmp_path / "frames.sock")
    server = _RawFrameServer(sock)
    reader = CameraStreamReader(socket_path=sock)
    reader.connect(timeout=5.0)
    server.accept(timeout=5.0)
    return server, reader


def _poll(fn, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.01)
    return None


def test_demux_two_streams_no_clobber(tmp_path):
    """The core fix: chest and head frames back-to-back both survive, keyed by name."""
    server, reader = _reader(tmp_path)
    try:
        server.send(_frame("chest", stamp=1))
        server.send(_frame("head", stamp=2))  # would clobber chest in a single slot
        chest = _poll(lambda: reader.read("chest"))
        head = _poll(lambda: reader.read("head"))
        assert chest is not None and chest.name == "chest" and chest.stamp_ns == 1
        assert head is not None and head.name == "head" and head.stamp_ns == 2
    finally:
        reader.close()
        server.close()


def test_read_returns_decoded_camera_frame(tmp_path):
    server, reader = _reader(tmp_path)
    try:
        server.send(_frame("chest", stamp=7))
        f = _poll(lambda: reader.read("chest"))
        assert isinstance(f, CameraFrame)
        assert f.rgb.shape == (4, 8, 3) and f.rgb.dtype == np.uint8
        assert f.depth.shape == (4, 8) and f.depth.dtype == np.float32
        assert f.intrinsics.width == 8 and f.intrinsics.height == 4
    finally:
        reader.close()
        server.close()


def test_latest_wins_per_stream(tmp_path):
    server, reader = _reader(tmp_path)
    try:
        for s in range(1, 11):
            server.send(_frame("chest", stamp=s))

        def newest():
            f = reader.read("chest")
            return f if (f is not None and f.stamp_ns == 10) else None

        got = _poll(newest)
        assert got is not None and got.stamp_ns == 10, "read must give the newest per stream"
    finally:
        reader.close()
        server.close()


def test_read_is_non_consuming(tmp_path):
    """A display feed re-reads the same latest frame until a newer one replaces it."""
    server, reader = _reader(tmp_path)
    try:
        server.send(_frame("chest", stamp=3))
        assert _poll(lambda: reader.read("chest")) is not None
        again = reader.read("chest")
        assert again is not None and again.stamp_ns == 3, "latest persists until replaced"
    finally:
        reader.close()
        server.close()


def test_read_unknown_stream_is_none(tmp_path):
    server, reader = _reader(tmp_path)
    try:
        assert reader.read("head") is None  # nothing ever arrived for head
    finally:
        reader.close()
        server.close()


def test_read_is_nonblocking(tmp_path):
    server, reader = _reader(tmp_path)
    try:
        t0 = time.monotonic()
        for _ in range(500):
            reader.read("chest")
        dt = time.monotonic() - t0
        assert dt < 0.5, f"read must be near-instant (dict lookup), took {dt:.2f}s"
    finally:
        reader.close()
        server.close()


def test_stream_names_discovered(tmp_path):
    server, reader = _reader(tmp_path)
    try:
        server.send(_frame("chest", stamp=1))
        server.send(_frame("head", stamp=2))
        _poll(lambda: reader.read("chest"))
        _poll(lambda: reader.read("head"))
        assert set(reader.stream_names()) == {"chest", "head"}
    finally:
        reader.close()
        server.close()


def test_close_is_idempotent(tmp_path):
    server, reader = _reader(tmp_path)
    reader.close()
    reader.close()  # must not raise
    server.close()


def test_server_and_reader_carry_two_streams(tmp_path):
    """Full loopback with the REAL FrameChannelServer: publishing chest then head in
    the same tick must not clobber chest — the server keeps a per-name mailbox, so the
    reader receives BOTH streams. (Single-slot server drops chest.)"""
    sock = str(tmp_path / "frames.sock")
    server = FrameChannelServer(socket_path=sock)
    server.serve()
    reader = CameraStreamReader(socket_path=sock)
    reader.connect(timeout=5.0)
    try:
        for _ in range(3):  # both cameras every tick, a few ticks
            server.publish(encode_camera_frame(_frame("chest", stamp=1)))
            server.publish(encode_camera_frame(_frame("head", stamp=2)))
        chest = _poll(lambda: reader.read("chest"))
        head = _poll(lambda: reader.read("head"))
        assert chest is not None and chest.name == "chest", "chest must not be clobbered by head"
        assert head is not None and head.name == "head"
    finally:
        reader.close()
        server.close()
