"""
protocol.py — the Communication wire between the World (server) and Brain (client).

Message types travel a single AF_UNIX SEQPACKET socket (D10/D11):

  - HELLO       Brain → World    once at startup, carries the World's DOF names
  - CMD         Brain → World    per-step PolicyOut (≅ RobotCmd, minus motor_names)
  - STATE_IMU   World → Brain    per physics tick, Observation (RobotState + IMU)
  - GLIDE_CMD   Brain → World    glide-mode base-velocity command (MAY-172) — additive;
                                 the walk frames above are untouched by its presence

`Observation` ≅ STATE_IMU and `PolicyOut` ≅ CMD (see `contracts.py`); `codec.py`
maps the dataclasses to/from these frames. All payloads are little-endian
fixed-size struct layouts. The same source produces byte-identical output in
CPython 3.8 (the future py3.8 `RealComm` edge) and 3.11 (the brain + `SimComm`)
— `struct.pack` is the canonical layer, so schema-invariance holds across the
version split. This module is PURE: stdlib only, no isaacsim/limxsdk, no numpy.

Header layout (8 bytes total):
    +-----------------------------------------------+
    | uint16  type_with_version                     |   bits 14-15 = version
    |                                               |   bits  0-13 = type_code
    +-----------------------------------------------+
    | uint16  payload_len    (redundant on          |
    |                          SEQPACKET, kept      |
    |                          for STREAM future)   |
    +-----------------------------------------------+
    | uint32  seq                                   |   monotonically increasing
    +-----------------------------------------------+

Payload sizes (computed at import time via `struct.calcsize`):
  HELLO_PAYLOAD_SIZE      996 B   = 4 (dof_count) + 31 × 32 (names)
  CMD_PAYLOAD_SIZE        690 B   = 8 (stamp) + 31 × (1+4+4+4+4+4+1)
  STATE_IMU_PAYLOAD_SIZE  420 B   = 8 (stamp) + 31 × 3 × 4 (q/dq/tau) + 12 + 12 + 16 (acc/gyro/quat)

Total wire sizes (header + payload):
  HELLO       1004 B
  CMD          698 B
  STATE_IMU    428 B

References:
  - Design: `humanoid/openspec/changes/may-147-oli-deployment-interface/design.md` (Contracts; D10, D11)
  - Origin: moved verbatim from the archived `may-147-isaac-limx-sdk-bridge` bridge wire
  - Wire reality: `humanoid/docs/vendor/humanoid-rl-deploy-python.md` § 11
  - Canonical structs: `oli-corpus://limxsdk#datatypes.h`
"""

from __future__ import annotations

import struct
from enum import IntEnum
from typing import List, Sequence, Tuple

# ── Constants ───────────────────────────────────────────────────────────────

PROTOCOL_VERSION: int = 0
NUM_JOINTS: int = 31
JOINT_NAME_LEN: int = 32  # NUL-padded fixed-width ASCII


class MsgType(IntEnum):
    HELLO = 0
    CMD = 1
    STATE_IMU = 2
    GLIDE_CMD = 3  # glide-mode base-velocity command (MAY-172) — additive


# Maximum representable type_code given that we steal 2 bits for the version
_TYPE_CODE_MASK: int = 0x3FFF
_VERSION_SHIFT: int = 14
_VERSION_MASK: int = 0x3


# ── Format strings ──────────────────────────────────────────────────────────

# 8-byte header: uint16 type_with_version, uint16 payload_len, uint32 seq
HEADER_FMT: str = "<HHI"
HEADER_SIZE: int = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 8, f"header must be 8 bytes, got {HEADER_SIZE}"

# HELLO: dof_count + 31×32-byte NUL-padded ASCII joint names
HELLO_PAYLOAD_FMT: str = f"<I{NUM_JOINTS * JOINT_NAME_LEN}s"
HELLO_PAYLOAD_SIZE: int = struct.calcsize(HELLO_PAYLOAD_FMT)

# CMD: stamp (u64) + 31×u8 mode + 31×f32 q + 31×f32 dq + 31×f32 tau
#                  + 31×f32 Kp + 31×f32 Kd + 31×u8 parallel_solve_required
CMD_PAYLOAD_FMT: str = (
    "<Q"
    f"{NUM_JOINTS}B"
    f"{NUM_JOINTS}f"
    f"{NUM_JOINTS}f"
    f"{NUM_JOINTS}f"
    f"{NUM_JOINTS}f"
    f"{NUM_JOINTS}f"
    f"{NUM_JOINTS}B"
)
CMD_PAYLOAD_SIZE: int = struct.calcsize(CMD_PAYLOAD_FMT)

# STATE_IMU: stamp (u64) + 31×f32 q + 31×f32 dq + 31×f32 tau
#          + 3×f32 acc + 3×f32 gyro + 4×f32 quat_wxyz
STATE_IMU_PAYLOAD_FMT: str = (
    "<Q"
    f"{NUM_JOINTS}f"
    f"{NUM_JOINTS}f"
    f"{NUM_JOINTS}f"
    "3f"
    "3f"
    "4f"
)
STATE_IMU_PAYLOAD_SIZE: int = struct.calcsize(STATE_IMU_PAYLOAD_FMT)

# GLIDE_CMD: stamp (u64) + v_x + v_y + w_z (f32) — body-frame base twist (glide mode)
GLIDE_CMD_PAYLOAD_FMT: str = "<Q3f"
GLIDE_CMD_PAYLOAD_SIZE: int = struct.calcsize(GLIDE_CMD_PAYLOAD_FMT)

# Total message sizes
HELLO_MSG_SIZE: int = HEADER_SIZE + HELLO_PAYLOAD_SIZE
CMD_MSG_SIZE: int = HEADER_SIZE + CMD_PAYLOAD_SIZE
STATE_IMU_MSG_SIZE: int = HEADER_SIZE + STATE_IMU_PAYLOAD_SIZE
GLIDE_CMD_MSG_SIZE: int = HEADER_SIZE + GLIDE_CMD_PAYLOAD_SIZE


# ── Errors ──────────────────────────────────────────────────────────────────

class ProtocolError(ValueError):
    """Raised when an incoming frame violates the wire contract."""


# ── Header pack / unpack ────────────────────────────────────────────────────

def pack_header(msg_type: MsgType, payload_len: int, seq: int) -> bytes:
    if msg_type & ~_TYPE_CODE_MASK:
        raise ProtocolError(f"msg_type {msg_type!r} exceeds 14-bit type_code field")
    if not (0 <= payload_len <= 0xFFFF):
        raise ProtocolError(f"payload_len {payload_len} out of uint16 range")
    if not (0 <= seq <= 0xFFFFFFFF):
        raise ProtocolError(f"seq {seq} out of uint32 range")
    type_with_version = (PROTOCOL_VERSION << _VERSION_SHIFT) | int(msg_type)
    return struct.pack(HEADER_FMT, type_with_version, payload_len, seq)


def unpack_header(buf: bytes) -> Tuple[MsgType, int, int, int]:
    """Returns ``(msg_type, version, payload_len, seq)``.

    Raises ``ProtocolError`` if the type_code does not match a known ``MsgType``
    or if the version is not the version this module was compiled with.
    """
    if len(buf) < HEADER_SIZE:
        raise ProtocolError(f"header buf too short: {len(buf)} < {HEADER_SIZE}")
    type_with_version, payload_len, seq = struct.unpack_from(HEADER_FMT, buf, 0)
    version = (type_with_version >> _VERSION_SHIFT) & _VERSION_MASK
    type_code = type_with_version & _TYPE_CODE_MASK
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version mismatch: received {version}, expected {PROTOCOL_VERSION}"
        )
    try:
        msg_type = MsgType(type_code)
    except ValueError as e:
        raise ProtocolError(f"unknown msg type code {type_code}") from e
    return msg_type, version, payload_len, seq


# ── HELLO ───────────────────────────────────────────────────────────────────

def pack_hello(seq: int, dof_names: Sequence[str]) -> bytes:
    """Driver → sidecar: announce Isaac DOF names in Isaac order."""
    if len(dof_names) != NUM_JOINTS:
        raise ProtocolError(
            f"HELLO requires {NUM_JOINTS} names, got {len(dof_names)}"
        )
    names_buf = b"".join(
        name.encode("ascii").ljust(JOINT_NAME_LEN, b"\x00")[:JOINT_NAME_LEN]
        for name in dof_names
    )
    payload = struct.pack(HELLO_PAYLOAD_FMT, NUM_JOINTS, names_buf)
    return pack_header(MsgType.HELLO, HELLO_PAYLOAD_SIZE, seq) + payload


def unpack_hello(buf: bytes) -> Tuple[int, int, List[str]]:
    """Returns ``(seq, dof_count, dof_names)``."""
    msg_type, _version, _payload_len, seq = unpack_header(buf)
    if msg_type is not MsgType.HELLO:
        raise ProtocolError(f"expected HELLO, got {msg_type.name}")
    if len(buf) < HEADER_SIZE + HELLO_PAYLOAD_SIZE:
        raise ProtocolError(
            f"HELLO buf too short: {len(buf)} < {HEADER_SIZE + HELLO_PAYLOAD_SIZE}"
        )
    dof_count, names_buf = struct.unpack_from(
        HELLO_PAYLOAD_FMT, buf, HEADER_SIZE
    )
    names = [
        names_buf[i * JOINT_NAME_LEN : (i + 1) * JOINT_NAME_LEN]
        .rstrip(b"\x00")
        .decode("ascii")
        for i in range(NUM_JOINTS)
    ]
    return seq, int(dof_count), names


# ── CMD ─────────────────────────────────────────────────────────────────────

def pack_cmd(
    seq: int,
    stamp_ns: int,
    mode: Sequence[int],
    q: Sequence[float],
    dq: Sequence[float],
    tau: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    parallel_solve_required: Sequence[int],
) -> bytes:
    """Sidecar → driver: per-tick cmd from ``limxsdk.subscribeRobotCmdForSim``.

    All array fields MUST have length ``NUM_JOINTS`` and be in PR-space order.
    ``parallel_solve_required`` is packed as ``uint8`` (0 or 1) — the canonical
    struct uses ``vector<bool>`` but we marshal as bytes for stable wire size.
    """
    for name, arr in (
        ("mode", mode),
        ("q", q),
        ("dq", dq),
        ("tau", tau),
        ("kp", kp),
        ("kd", kd),
        ("parallel_solve_required", parallel_solve_required),
    ):
        if len(arr) != NUM_JOINTS:
            raise ProtocolError(
                f"CMD field '{name}' must have {NUM_JOINTS} entries, got {len(arr)}"
            )
    payload = struct.pack(
        CMD_PAYLOAD_FMT,
        int(stamp_ns),
        *(int(v) & 0xFF for v in mode),
        *(float(v) for v in q),
        *(float(v) for v in dq),
        *(float(v) for v in tau),
        *(float(v) for v in kp),
        *(float(v) for v in kd),
        *(1 if v else 0 for v in parallel_solve_required),
    )
    return pack_header(MsgType.CMD, CMD_PAYLOAD_SIZE, seq) + payload


def unpack_cmd(
    buf: bytes,
) -> Tuple[
    int, int, List[int], List[float], List[float], List[float],
    List[float], List[float], List[int]
]:
    """Returns ``(seq, stamp_ns, mode, q, dq, tau, kp, kd, parallel_solve_required)``."""
    msg_type, _version, _payload_len, seq = unpack_header(buf)
    if msg_type is not MsgType.CMD:
        raise ProtocolError(f"expected CMD, got {msg_type.name}")
    if len(buf) < HEADER_SIZE + CMD_PAYLOAD_SIZE:
        raise ProtocolError(
            f"CMD buf too short: {len(buf)} < {HEADER_SIZE + CMD_PAYLOAD_SIZE}"
        )
    fields = struct.unpack_from(CMD_PAYLOAD_FMT, buf, HEADER_SIZE)
    stamp_ns = int(fields[0])
    idx = 1
    mode = [int(v) for v in fields[idx : idx + NUM_JOINTS]]
    idx += NUM_JOINTS
    q = list(fields[idx : idx + NUM_JOINTS])
    idx += NUM_JOINTS
    dq = list(fields[idx : idx + NUM_JOINTS])
    idx += NUM_JOINTS
    tau = list(fields[idx : idx + NUM_JOINTS])
    idx += NUM_JOINTS
    kp = list(fields[idx : idx + NUM_JOINTS])
    idx += NUM_JOINTS
    kd = list(fields[idx : idx + NUM_JOINTS])
    idx += NUM_JOINTS
    parallel_solve_required = [int(v) for v in fields[idx : idx + NUM_JOINTS]]
    return seq, stamp_ns, mode, q, dq, tau, kp, kd, parallel_solve_required


# ── STATE_IMU ───────────────────────────────────────────────────────────────

def pack_state_imu(
    seq: int,
    stamp_ns: int,
    q: Sequence[float],
    dq: Sequence[float],
    tau: Sequence[float],
    acc: Sequence[float],
    gyro: Sequence[float],
    quat_wxyz: Sequence[float],
) -> bytes:
    """Driver → sidecar: per-physics-tick state + IMU sample.

    ``quat_wxyz`` MUST be in `(w, x, y, z)` order to match both Isaac native
    and LimX's `ImuData.quat[4]` (`oli-corpus://limxsdk#datatypes.h`).
    """
    for name, arr, n in (
        ("q", q, NUM_JOINTS),
        ("dq", dq, NUM_JOINTS),
        ("tau", tau, NUM_JOINTS),
        ("acc", acc, 3),
        ("gyro", gyro, 3),
        ("quat_wxyz", quat_wxyz, 4),
    ):
        if len(arr) != n:
            raise ProtocolError(
                f"STATE_IMU field '{name}' must have {n} entries, got {len(arr)}"
            )
    payload = struct.pack(
        STATE_IMU_PAYLOAD_FMT,
        int(stamp_ns),
        *(float(v) for v in q),
        *(float(v) for v in dq),
        *(float(v) for v in tau),
        *(float(v) for v in acc),
        *(float(v) for v in gyro),
        *(float(v) for v in quat_wxyz),
    )
    return pack_header(MsgType.STATE_IMU, STATE_IMU_PAYLOAD_SIZE, seq) + payload


def unpack_state_imu(
    buf: bytes,
) -> Tuple[
    int, int, List[float], List[float], List[float],
    List[float], List[float], List[float]
]:
    """Returns ``(seq, stamp_ns, q, dq, tau, acc, gyro, quat_wxyz)``."""
    msg_type, _version, _payload_len, seq = unpack_header(buf)
    if msg_type is not MsgType.STATE_IMU:
        raise ProtocolError(f"expected STATE_IMU, got {msg_type.name}")
    if len(buf) < HEADER_SIZE + STATE_IMU_PAYLOAD_SIZE:
        raise ProtocolError(
            f"STATE_IMU buf too short: "
            f"{len(buf)} < {HEADER_SIZE + STATE_IMU_PAYLOAD_SIZE}"
        )
    fields = struct.unpack_from(STATE_IMU_PAYLOAD_FMT, buf, HEADER_SIZE)
    stamp_ns = int(fields[0])
    idx = 1
    q = list(fields[idx : idx + NUM_JOINTS])
    idx += NUM_JOINTS
    dq = list(fields[idx : idx + NUM_JOINTS])
    idx += NUM_JOINTS
    tau = list(fields[idx : idx + NUM_JOINTS])
    idx += NUM_JOINTS
    acc = list(fields[idx : idx + 3])
    idx += 3
    gyro = list(fields[idx : idx + 3])
    idx += 3
    quat_wxyz = list(fields[idx : idx + 4])
    return seq, stamp_ns, q, dq, tau, acc, gyro, quat_wxyz


# ── GLIDE_CMD ─────────────────────────────────────────────────────────────────

def pack_glide_cmd(
    seq: int, stamp_ns: int, v_x: float, v_y: float, w_z: float
) -> bytes:
    """Brain → World: a body-frame base-velocity command for kinematic glide (MAY-172).

    A NEW message type on the shared socket — additive, so the walk `CMD` frame is
    untouched. The World integrates this twist (accel/turn-limited) into base motion.
    """
    payload = struct.pack(
        GLIDE_CMD_PAYLOAD_FMT, int(stamp_ns), float(v_x), float(v_y), float(w_z)
    )
    return pack_header(MsgType.GLIDE_CMD, GLIDE_CMD_PAYLOAD_SIZE, seq) + payload


def unpack_glide_cmd(buf: bytes) -> Tuple[int, int, float, float, float]:
    """Returns ``(seq, stamp_ns, v_x, v_y, w_z)``."""
    msg_type, _version, _payload_len, seq = unpack_header(buf)
    if msg_type is not MsgType.GLIDE_CMD:
        raise ProtocolError(f"expected GLIDE_CMD, got {msg_type.name}")
    if len(buf) < HEADER_SIZE + GLIDE_CMD_PAYLOAD_SIZE:
        raise ProtocolError(
            f"GLIDE_CMD buf too short: "
            f"{len(buf)} < {HEADER_SIZE + GLIDE_CMD_PAYLOAD_SIZE}"
        )
    stamp_ns, v_x, v_y, w_z = struct.unpack_from(
        GLIDE_CMD_PAYLOAD_FMT, buf, HEADER_SIZE
    )
    return seq, int(stamp_ns), float(v_x), float(v_y), float(w_z)
