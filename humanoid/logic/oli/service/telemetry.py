"""service/telemetry.py — W5: the brain's telemetry channel OUT.

Transport mirrors `comm/debug_pose.py` with a JSON payload: `TelemetryServer` (brain-side)
fire-and-forgets the latest `TelemetrySnapshot` to the reader's socket path — best-effort by
design, the brain loop never stalls because nobody listens; `TelemetryClient` (evaluator /
dev_app-later) binds the path and drains latest-wins, dropping malformed datagrams.

`TelemetryPublisher` is the assembly: a callable shaped for the Orchestrator's `recorder`
slot (`(obs, policy_in, action_out, joy)`). Per tick it reads Nav's observable surface
(`path` / `goal` / `last_pose` — read-only properties, the "path out" seam philosophy), pulls
the localization host's latest verdict via the injected `est_source` (None until §5 plugs the
host in), measures the brain-loop wall-clock rate over a sliding window (the GIL-stall metric,
locbench design.md D6), and publishes one snapshot.

Stdlib only; never isaacsim/limxsdk.
"""

from __future__ import annotations

import os
import socket
import time
from collections import deque
from typing import Callable, List, Optional, Protocol, Tuple

from ..reason.localization import LocalizationOut, RobotPose
from ..reason.nav import GoalCoordinate
from .protocol import TelemetrySnapshot, decode_telemetry, encode_telemetry

DEFAULT_TELEMETRY_SOCKET = "/tmp/oli-telemetry.sock"

# One JSON snapshot with a long planned path stays well under this.
_MAX_DATAGRAM = 65536

# Loop-rate window: rate = (N-1) ticks / wall-time span. Small enough to see a stall fast.
_RATE_WINDOW = 50


class NavObservable(Protocol):
    """Nav's read-only observable surface — what telemetry is allowed to see."""

    @property
    def path(self) -> Optional[List[Tuple[float, float]]]: ...
    @property
    def goal(self) -> Optional[GoalCoordinate]: ...
    @property
    def last_pose(self) -> Optional[RobotPose]: ...


class TelemetryServer:
    """Brain-side: fire-and-forget the latest snapshot to the reader's socket path."""

    def __init__(self, path: str = DEFAULT_TELEMETRY_SOCKET) -> None:
        self._path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    def publish(self, snap: TelemetrySnapshot) -> None:
        try:
            self._sock.sendto(encode_telemetry(snap), self._path)
        except (BlockingIOError, FileNotFoundError, ConnectionRefusedError):
            pass  # no reader / buffer full — telemetry is best-effort by design

    def close(self) -> None:
        self._sock.close()


class TelemetryClient:
    """Reader-side: bind the path, drain to the newest snapshot (latest-wins)."""

    def __init__(self, path: str = DEFAULT_TELEMETRY_SOCKET) -> None:
        self._path = path
        try:
            os.unlink(path)  # clear a stale socket file from a prior run
        except FileNotFoundError:
            pass
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(path)
        self._sock.setblocking(False)
        self._latest: Optional[TelemetrySnapshot] = None

    def latest(self) -> Optional[TelemetrySnapshot]:
        while True:
            try:
                buf = self._sock.recv(_MAX_DATAGRAM)
            except BlockingIOError:
                break
            try:
                self._latest = decode_telemetry(buf)
            except ValueError:
                continue  # malformed datagram — drop, keep draining
        return self._latest

    def close(self) -> None:
        self._sock.close()
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass


class TelemetryPublisher:
    """The Orchestrator-recorder-slot assembly: one tick in → one snapshot out."""

    def __init__(
        self,
        server,
        nav: NavObservable,
        est_source: Optional[Callable[[], Optional[LocalizationOut]]] = None,
    ) -> None:
        self._server = server
        self._nav = nav
        self._est_source = est_source
        self._tick_walltimes: deque = deque(maxlen=_RATE_WINDOW)

    def __call__(self, obs, policy_in, action_out, joy) -> None:
        self._tick_walltimes.append(time.monotonic())
        pose = self._nav.last_pose
        intent = policy_in.intent
        self._server.publish(TelemetrySnapshot(
            stamp_ns=obs.stamp_ns,
            pose=(pose.x, pose.y, pose.yaw) if pose is not None else None,
            path=self._nav.path,
            goal=self._nav.goal,
            est=self._est_source() if self._est_source is not None else None,
            intent=(intent.v_x, intent.v_y, intent.w_z),
            loop_hz=self._loop_hz(),
        ))

    def _loop_hz(self) -> Optional[float]:
        if len(self._tick_walltimes) < 2:
            return None
        span = self._tick_walltimes[-1] - self._tick_walltimes[0]
        if span <= 0.0:
            return None
        return (len(self._tick_walltimes) - 1) / span
