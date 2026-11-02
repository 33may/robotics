"""frame_protocol.py — the wire for the dedicated camera-frame channel (§6, design.md D6).

The control channel (`protocol.py`) is fixed-size SEQPACKET datagrams (≤1 KB). Camera
frames are variable and large (a 720p RGB frame is ~2.8 MB), so they travel a separate
`SOCK_STREAM` channel with a length-prefixed framing: a fixed header (magic / version /
type / seq / stamp / camera name / resolution / intrinsics / two payload lengths),
followed by the RGB bytes then the depth bytes. The payload lengths make each message
self-delimiting on a byte stream (see `read_frame` for the reassembly helper §7 will use).

PURE: stdlib only (`struct`) — no numpy, no isaacsim/limxsdk. The dataclass↔bytes mapping
(and the depth uint16-mm quantization) lives in `codec.py`; this module only frames bytes.
"""

from __future__ import annotations

import struct
from typing import Tuple

MAGIC: int = 0x4F4C4943  # "OLIC"
FRAME_VERSION: int = 0
MSG_CAMERA_FRAME: int = 1
NAME_LEN: int = 16  # fixed-width NUL-padded ASCII camera name

# < little-endian | I magic | H ver | H type | I seq | q stamp_ns | 16s name |
#   H width | H height | f fx | f fy | f cx | f cy | I rgb_len | I depth_len
_HEADER = struct.Struct("<IHHIq16sHHffffII")
HEADER_SIZE: int = _HEADER.size  # 64 bytes


class FrameProtocolError(ValueError):
    """Malformed camera-frame wire buffer (bad magic/type, or truncated)."""


def pack_camera_frame(
    seq: int,
    stamp_ns: int,
    name: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_bytes: bytes,
    depth_bytes: bytes,
) -> bytes:
    """Frame one camera message: fixed header + RGB payload + depth payload."""
    name_b = name.encode("ascii")[:NAME_LEN].ljust(NAME_LEN, b"\x00")
    header = _HEADER.pack(
        MAGIC, FRAME_VERSION, MSG_CAMERA_FRAME, seq, stamp_ns, name_b,
        width, height, fx, fy, cx, cy, len(rgb_bytes), len(depth_bytes),
    )
    return header + rgb_bytes + depth_bytes


def unpack_camera_frame(
    buf: bytes,
) -> Tuple[int, int, str, int, int, float, float, float, float, bytes, bytes]:
    """Parse a complete camera-frame buffer. Raises `FrameProtocolError` if the header
    is bad or the payloads are truncated."""
    if len(buf) < HEADER_SIZE:
        raise FrameProtocolError(f"buffer too short for header: {len(buf)} < {HEADER_SIZE}")
    (magic, version, msg_type, seq, stamp_ns, name_b, width, height,
     fx, fy, cx, cy, rgb_len, depth_len) = _HEADER.unpack_from(buf)
    if magic != MAGIC:
        raise FrameProtocolError(f"bad magic 0x{magic:08X}")
    if msg_type != MSG_CAMERA_FRAME:
        raise FrameProtocolError(f"unexpected msg_type {msg_type}")
    end_rgb = HEADER_SIZE + rgb_len
    end_depth = end_rgb + depth_len
    if len(buf) < end_depth:
        raise FrameProtocolError(f"truncated payload: {len(buf)} < {end_depth}")
    name = name_b.rstrip(b"\x00").decode("ascii")
    rgb_bytes = buf[HEADER_SIZE:end_rgb]
    depth_bytes = buf[end_rgb:end_depth]
    return (seq, stamp_ns, name, width, height, fx, fy, cx, cy, rgb_bytes, depth_bytes)


def payload_lengths(header: bytes) -> Tuple[int, int]:
    """Given at least the fixed header, return (rgb_len, depth_len) — lets a stream
    reader know how many more bytes to pull before the message is complete (§7)."""
    if len(header) < HEADER_SIZE:
        raise FrameProtocolError("need the full header to read payload lengths")
    fields = _HEADER.unpack_from(header)
    return fields[12], fields[13]


def frame_name(header: bytes) -> str:
    """Extract the camera name from at least the fixed header — lets the World's frame
    server keep a per-stream mailbox (so two cameras never clobber each other §7)."""
    if len(header) < HEADER_SIZE:
        raise FrameProtocolError("need the full header to read the camera name")
    name_b = _HEADER.unpack_from(header)[5]
    return name_b.rstrip(b"\x00").decode("ascii")
