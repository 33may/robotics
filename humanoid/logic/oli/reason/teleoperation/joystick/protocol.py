"""protocol.py — the joystick app → brain wire (our own, separate from comm/).

The joystick link is latest-wins: each datagram carries the FULL stick state, so the
brain just keeps the newest and drops stale ones — no framing, no sequence handling.
That is why this is a standalone datagram wire and not part of the `comm/` state↔cmd
stream: different cadence (~50 Hz best-effort), different failure mode (a dropped joy
packet is harmless), different transport (UDP, so a remote/phone app only changes host).

Layout (little-endian, variable length — controllers differ in axis/button counts):

    +----------------------------------------------+
    | uint32  magic  = 'JOY1' (0x31594F4A)         |
    | uint64  stamp_ns                             |
    | uint16  n_axes                               |
    | uint16  n_buttons                            |
    +----------------------------------------------+
    | float32 × n_axes      (stick positions)      |
    | uint8   × n_buttons   (0/1 pressed)          |
    +----------------------------------------------+

Pure stdlib (`struct` only) — no isaacsim/limxsdk/numpy — so the app process can run
in any env. Axes follow the LimX `walk_controller` layout (axis0=v_y, axis1=v_x,
axis3=w_z); buttons follow the PlayStation indices `main.py` uses for mode combos.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Sequence

MAGIC: int = 0x31594F4A  # 'JOY1' little-endian
HEADER_FMT: str = "<IQHH"  # magic, stamp_ns, n_axes, n_buttons
HEADER_SIZE: int = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 16, f"header must be 16 bytes, got {HEADER_SIZE}"


class JoyProtocolError(ValueError):
    """Raised when a datagram violates the joystick wire contract."""


@dataclass(frozen=True)
class JoyPacket:
    """One full joystick sample: monotonic stamp + raw axes + button states."""

    stamp_ns: int
    axes: Sequence[float]
    buttons: Sequence[int]


def pack_joy(pkt: JoyPacket) -> bytes:
    n_axes = len(pkt.axes)
    n_buttons = len(pkt.buttons)
    header = struct.pack(HEADER_FMT, MAGIC, int(pkt.stamp_ns) & 0xFFFFFFFFFFFFFFFF,
                         n_axes, n_buttons)
    body = struct.pack(
        f"<{n_axes}f{n_buttons}B",
        *(float(v) for v in pkt.axes),
        *(1 if int(v) else 0 for v in pkt.buttons),
    )
    return header + body


def unpack_joy(buf: bytes) -> JoyPacket:
    if len(buf) < HEADER_SIZE:
        raise JoyProtocolError(f"buf too short for header: {len(buf)} < {HEADER_SIZE}")
    magic, stamp_ns, n_axes, n_buttons = struct.unpack_from(HEADER_FMT, buf, 0)
    if magic != MAGIC:
        raise JoyProtocolError(f"bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})")
    need = HEADER_SIZE + n_axes * 4 + n_buttons
    if len(buf) < need:
        raise JoyProtocolError(
            f"payload truncated: have {len(buf)} bytes, need {need} "
            f"(n_axes={n_axes}, n_buttons={n_buttons})"
        )
    fields = struct.unpack_from(f"<{n_axes}f{n_buttons}B", buf, HEADER_SIZE)
    axes: List[float] = list(fields[:n_axes])
    buttons: List[int] = [int(v) for v in fields[n_axes:n_axes + n_buttons]]
    return JoyPacket(stamp_ns=int(stamp_ns), axes=axes, buttons=buttons)
