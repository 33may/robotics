"""service/loc_ctrl.py — the localization lifecycle channel (locbench D6, tasks 5.3).

The evaluator commands the in-brain localization host's per-episode lifecycle:
`start` (JSON: map_dir, warm-start pose, calibration blob) and `stop`. Unlike the goal
channel this is NOT latest-wins — a stop-then-start sequence must apply in order, so
`poll()` drains and applies EVERY valid command (malformed datagrams dropped). The server
only ENQUEUES onto the host (`request_start/request_stop`) — a heavy candidate `start()`
executes on the host's own thread, never here. There is no reply channel: the evaluator
watches the telemetry `loc_state` field (idle/starting/running/crashed) — the same one-way,
read-the-latest philosophy as the rest of the seam. Stdlib only.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Optional, Protocol, Tuple

from ..reason.localization import LocalizationSetup, RobotPose

DEFAULT_LOC_CTRL_SOCKET = "/tmp/oli-loc-ctrl.sock"

_MAX_DATAGRAM = 65536


class LifecycleSink(Protocol):
    """What this channel drives — `LocalizationHost` satisfies it structurally."""

    def request_start(self, setup: LocalizationSetup) -> None: ...
    def request_stop(self) -> None: ...


class LocCtrlServer:
    """Brain-side: bind, drain on poll(), enqueue every valid command onto the host."""

    def __init__(self, path: str, host: LifecycleSink) -> None:
        self._path = path
        self._host = host
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(path)
        self._sock.setblocking(False)

    def poll(self) -> int:
        """Apply all pending commands in arrival order; returns how many applied."""
        applied = 0
        while True:
            try:
                buf = self._sock.recv(_MAX_DATAGRAM)
            except BlockingIOError:
                break
            try:
                op, setup = _decode(buf)
            except ValueError:
                continue  # malformed — drop, keep draining
            if op == "start":
                assert setup is not None
                self._host.request_start(setup)
            else:
                self._host.request_stop()
            applied += 1
        return applied

    def close(self) -> None:
        self._sock.close()
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass


class LocCtrlClient:
    """Evaluator-side sender. Raises OSError when no brain listens (commands are loud)."""

    def __init__(self, path: str = DEFAULT_LOC_CTRL_SOCKET) -> None:
        self._path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def send_start(
        self,
        *,
        map_dir: str,
        initial_pose: Optional[Tuple[float, float, float]] = None,
        calibration: Optional[dict] = None,
    ) -> None:
        doc = {"op": "start", "map_dir": map_dir,
               "initial_pose": list(initial_pose) if initial_pose is not None else None,
               "calibration": calibration or {}}
        self._sock.sendto(json.dumps(doc).encode(), self._path)

    def send_stop(self) -> None:
        self._sock.sendto(json.dumps({"op": "stop"}).encode(), self._path)

    def close(self) -> None:
        self._sock.close()


def _decode(buf: bytes) -> Tuple[str, Optional[LocalizationSetup]]:
    try:
        doc = json.loads(buf)
        op = doc["op"]
        if op == "stop":
            return ("stop", None)
        if op != "start":
            raise ValueError(f"unknown op {op!r}")
        pose = doc.get("initial_pose")
        return ("start", LocalizationSetup(
            map_dir=doc["map_dir"],
            initial_pose=RobotPose(0, *pose) if pose is not None else None,
            calibration=dict(doc.get("calibration") or {}),
        ))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"malformed loc-ctrl datagram: {exc}") from exc
