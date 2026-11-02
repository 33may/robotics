"""source.py — joystick SOURCES: where the stick state comes from.

A source answers one question each control step: *what is the stick doing right now?*
`poll()` returns a `JoyPacket` (axes + buttons) — axes in the LimX `walk_controller`
layout (`axis0 = v_y`, `axis1 = v_x`, `axis2` unused, `axis3 = w_z`) and buttons in the
PlayStation indices — which `Teleop` turns into velocity + mode. The Orchestrator only
depends on `poll()` (duck-typed), so any source satisfying `JoystickSource` drops in
without touching reason/action/runtime.

Sources here MUST stay brain-pure (no `isaacsim`, no `limxsdk`) — that is the
deployment invariant. The real LimX gamepad arrives as `limxsdk.SensorJoy` over the
MROS bus, which `limxsdk` can only read in the py3.8 World/RealComm process; a brain
source therefore reads the device directly (`evdev`/`pygame` on `/dev/input/js0`) or
receives packets across the Comm boundary. `FixedJoystick` is the no-device stand-in
used for the sim walk demo until that real source lands.
"""

from __future__ import annotations

import socket
from typing import Optional, Protocol, runtime_checkable

from .protocol import JoyPacket, JoyProtocolError, unpack_joy


@runtime_checkable
class JoystickSource(Protocol):
    """The seam the Orchestrator depends on: poll the latest stick state.

    `poll()` is non-blocking and latest-wins — it returns the most recent `JoyPacket`
    (axes + buttons) or `None` when no command is available, in which case `Teleop`
    holds zero velocity and the current mode. Implementations own their freshness.
    """

    def poll(self) -> Optional[JoyPacket]:
        ...


class FixedJoystick:
    """A constant joystick command (no physical device yet).

    Axes follow the `walk_controller` layout: axis0 = v_y, axis1 = v_x, axis3 = w_z.
    No buttons are pressed (mode is driven externally via `Teleop.set_mode`). Used for
    the sim walk demo (a steady `--vx`/`--vy`/`--wz` from the CLI) and as a deterministic
    source in tests, before a real LimX joystick source exists.
    """

    def __init__(self, v_x: float = 0.0, v_y: float = 0.0, w_z: float = 0.0) -> None:
        self._packet = JoyPacket(stamp_ns=0, axes=[v_y, v_x, 0.0, w_z], buttons=[])

    def poll(self) -> JoyPacket:
        return self._packet


class SocketJoystickSource:
    """Reads `JoyPacket` datagrams from the joystick app over UDP (latest-wins).

    Binds a non-blocking UDP socket; `poll()` drains every datagram waiting in the
    kernel buffer and keeps only the NEWEST stick state, so a slow brain step never
    backs up behind a fast app. When no new datagram has arrived it HOLDS the last
    sample (a centered app still sends ~50 Hz, so a gap means the app paused — holding
    avoids a velocity jerk mid-stride). Returns None only before the first packet.
    Brain-pure: stdlib `socket` + our `protocol`, no isaacsim/limxsdk.
    """

    DEFAULT_PORT: int = 9001

    def __init__(
        self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, recv_bufsize: int = 2048
    ) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self._bufsize = recv_bufsize
        self._last_pkt: Optional[JoyPacket] = None

    @property
    def port(self) -> int:
        """The actually-bound UDP port (useful when constructed with port=0)."""
        return self._sock.getsockname()[1]

    def poll(self) -> Optional[JoyPacket]:
        newest = None
        while True:
            try:
                data, _addr = self._sock.recvfrom(self._bufsize)
            except (BlockingIOError, OSError):
                break  # no more datagrams queued (or socket closing)
            try:
                newest = unpack_joy(data)
            except JoyProtocolError:
                continue  # drop a malformed datagram, keep draining
        if newest is not None:
            self._last_pkt = newest
        return self._last_pkt

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
