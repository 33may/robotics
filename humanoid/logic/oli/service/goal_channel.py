"""service/goal_channel.py — W4: the goal command channel INTO the brain.

Brain-side `GoalChannelServer` binds an AF_UNIX datagram path (`comm/debug_pose.py` reader
pattern: unlink stale file, bind, non-blocking) and is polled from the brain loop. `poll()`
drains the backlog, decodes each frame (malformed → dropped), and applies ONLY the newest
valid command to Nav — latest-wins by design: every `Nav.set_goal` clears the planner's path
cache (full re-plan), so replaying a backlog would thrash the planner for no reason.

Client-side `GoalChannelClient` is the thin sender ANY goal source uses — the locbench
evaluator scripts episodes through it; dev_app later converts a map click and sends the same
frame (design.md D5: a goal source is anyone who can send). Sends RAISE when the brain is not
listening: goals are commands, not telemetry — a missing brain must be loud.

Stdlib only; never isaacsim/limxsdk.
"""

from __future__ import annotations

import os
import socket
from typing import Optional, Protocol

from ..reason.nav import GoalCoordinate
from .protocol import GOAL_NBYTES, decode_goal, encode_goal_clear, encode_goal_set

DEFAULT_GOAL_SOCKET = "/tmp/oli-goal.sock"


class GoalSink(Protocol):
    """The goal seam this channel drives — Nav satisfies it structurally."""

    def set_goal(self, goal: GoalCoordinate) -> None: ...
    def clear_goal(self) -> None: ...


class GoalChannelServer:
    """Brain-side: bind the UDS path, poll → apply the newest valid command to Nav."""

    def __init__(self, path: str, nav: GoalSink) -> None:
        self._path = path
        self._nav = nav
        try:
            os.unlink(path)  # clear a stale socket file from a crashed prior run
        except FileNotFoundError:
            pass
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(path)
        self._sock.setblocking(False)

    def poll(self) -> Optional[GoalCoordinate]:
        """Drain the backlog; apply the newest valid command (set → `Nav.set_goal`,
        clear → `Nav.clear_goal`). Returns the applied goal (None for clear/no traffic)."""
        newest: Optional[bytes] = None
        while True:
            try:
                buf = self._sock.recv(GOAL_NBYTES * 8)
            except BlockingIOError:
                break
            try:
                decode_goal(buf)  # validate now; only the survivor gets applied
            except ValueError:
                continue  # malformed frame — drop, keep draining
            newest = buf
        if newest is None:
            return None
        _, goal = decode_goal(newest)
        if goal is None:
            self._nav.clear_goal()
        else:
            self._nav.set_goal(goal)
        return goal

    def close(self) -> None:
        self._sock.close()
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass


class GoalChannelClient:
    """Client-side sender (evaluator, dev_app-later). Raises OSError if nobody listens."""

    def __init__(self, path: str = DEFAULT_GOAL_SOCKET) -> None:
        self._path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._stamp = 0  # informational send counter — the wire stamp slot

    def send_goal(self, x: float, y: float, yaw: Optional[float] = None) -> None:
        self._stamp += 1
        self._sock.sendto(encode_goal_set(self._stamp, x, y, yaw), self._path)

    def clear_goal(self) -> None:
        self._stamp += 1
        self._sock.sendto(encode_goal_clear(self._stamp), self._path)

    def close(self) -> None:
        self._sock.close()
