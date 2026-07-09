"""comm/debug_pose.py — the fenced ground-truth pose debug/eval channel (NOT the spine).

A separate AF_UNIX **datagram** channel carrying the World's ground-truth base pose to the
brain, for two uses:

  1. day-1 bring-up of the planner *before* a real localizer exists, and
  2. evaluating a real localizer's estimated-vs-truth pose error.

It is DELIBERATELY OUTSIDE the invariance spine (architecture.md §4): the real robot has no
ground-truth pose, so production simply never launches this channel and nothing on the spine
ever carries it. Gated by `--debug-pose`. Feeds a `GroundTruthLocalizer` via an injected reader,
so the planner still only ever sees a `RobotPose` from the `Localizer` seam.

Wire: a fixed 32-byte little-endian frame `<Q ddd>` = stamp_ns (uint64) + x, y, yaw (float64,
meters/rad, map frame). Stdlib only (struct + socket), so it is byte-identical across the
py3.8/py3.11 split and importable brain-side without pulling any world SDK. The codec speaks
primitive tuples — never `RobotPose` — to keep `comm` free of reason-internal types.
"""

from __future__ import annotations

import os
import socket
import struct
from typing import Optional, Tuple

_FMT = struct.Struct("<Qddd")
POSE_NBYTES: int = _FMT.size  # 32

PoseSample = Tuple[int, float, float, float]  # (stamp_ns, x, y, yaw)


def encode_pose(stamp_ns: int, x: float, y: float, yaw: float) -> bytes:
    """Pack a ground-truth pose sample into the fixed 32-byte wire frame."""
    return _FMT.pack(int(stamp_ns), float(x), float(y), float(yaw))


def decode_pose(buf: bytes) -> PoseSample:
    """Unpack a wire frame → (stamp_ns, x, y, yaw). Raises ValueError on a wrong-size frame."""
    if len(buf) != POSE_NBYTES:
        raise ValueError(f"debug pose frame must be {POSE_NBYTES} bytes, got {len(buf)}")
    stamp_ns, x, y, yaw = _FMT.unpack(buf)
    return (int(stamp_ns), float(x), float(y), float(yaw))


class DebugPoseServer:
    """World-side publisher: fire-and-forget the latest GT pose to the brain's socket path.

    AF_UNIX SOCK_DGRAM, non-blocking, best-effort: if no reader is bound yet or the buffer is
    full the send is silently dropped, so the sim loop never stalls on this debug channel. The
    World owns an instance and calls `publish(...)` each tick (or every N ticks). Stdlib only.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    def publish(self, stamp_ns: int, x: float, y: float, yaw: float) -> None:
        try:
            self._sock.sendto(encode_pose(stamp_ns, x, y, yaw), self._path)
        except (BlockingIOError, FileNotFoundError, ConnectionRefusedError):
            pass  # no reader / buffer full — the debug channel is best-effort by design

    def close(self) -> None:
        self._sock.close()


class DebugPoseClient:
    """Brain-side reader: bind the UDS path, drain to the newest datagram (latest-wins).

    `latest()` non-blocking-drains the socket and returns the freshest `PoseSample` seen since
    bind (or None if none yet), so the planner always reads the most recent GT pose and never
    lags behind a backlog. Inject `client.latest` (wrapped to a `RobotPose`) as a
    `GroundTruthLocalizer` pose_reader. Stdlib only.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        try:
            os.unlink(path)  # clear a stale socket file from a prior run
        except FileNotFoundError:
            pass
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(path)
        self._sock.setblocking(False)
        self._latest: Optional[PoseSample] = None

    def latest(self) -> Optional[PoseSample]:
        while True:
            try:
                buf = self._sock.recv(POSE_NBYTES * 8)
            except BlockingIOError:
                break
            try:
                self._latest = decode_pose(buf)
            except ValueError:
                continue  # skip a malformed frame, keep draining
        return self._latest

    def close(self) -> None:
        self._sock.close()
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
