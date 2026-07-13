"""service/protocol.py — wire codecs for the brain service seam (W4 goals, W5 telemetry).

Two channels, two shapes:

  W4 (goal, client → brain)  fixed 33-byte struct datagram `<BQddd>` = op byte (SET/CLEAR) +
      stamp_ns + x, y, yaw — the `comm/debug_pose.py` pattern. `yaw` rides as NaN when the
      goal has no final heading (`GoalCoordinate.yaw is None`).
  W5 (telemetry, brain → client)  JSON datagram — the planned path is variable-length, so a
      fixed struct cannot carry it. One `TelemetrySnapshot` per message, latest-wins at the
      reader; JSON keeps the channel debuggable (log a raw datagram, read it).

Decoding is strict: a wrong-size/unknown-op goal frame or a malformed telemetry document
raises `ValueError`, so channel readers can drop bad datagrams and keep draining (the
malformed-tolerance the seam promises). The `est` field decodes back into a REAL
`LocalizationOut` — its self-validating contract (pose ⇔ not-LOST, status coerced through the
enum) filters invariant-breaking wire data for free.

Stdlib only (struct + json + dataclasses): importable by any client env, never isaacsim/limxsdk.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..reason.localization import LocalizationOut, LocalizationStatus, RobotPose
from ..reason.nav import GoalCoordinate

# ── W4: goal channel (fixed struct) ──────────────────────────────────────────

_GOAL_FMT = struct.Struct("<BQddd")
GOAL_NBYTES: int = _GOAL_FMT.size  # 33

_OP_SET = 1
_OP_CLEAR = 2


def encode_goal_set(stamp_ns: int, x: float, y: float, yaw: Optional[float] = None) -> bytes:
    """Pack a SET-goal frame; `yaw=None` (no final heading) rides as NaN on the wire."""
    wire_yaw = math.nan if yaw is None else float(yaw)
    return _GOAL_FMT.pack(_OP_SET, int(stamp_ns), float(x), float(y), wire_yaw)


def encode_goal_clear(stamp_ns: int) -> bytes:
    """Pack a CLEAR-goal frame (position slots are padding)."""
    return _GOAL_FMT.pack(_OP_CLEAR, int(stamp_ns), 0.0, 0.0, 0.0)


def decode_goal(buf: bytes) -> Tuple[int, Optional[GoalCoordinate]]:
    """Unpack a goal frame → (stamp_ns, goal); goal is None for CLEAR.

    Raises ValueError on wrong size, unknown op, or non-finite x/y (NaN is only meaningful
    in the yaw slot) — callers drop the datagram and keep draining.
    """
    if len(buf) != GOAL_NBYTES:
        raise ValueError(f"goal frame must be {GOAL_NBYTES} bytes, got {len(buf)}")
    op, stamp_ns, x, y, yaw = _GOAL_FMT.unpack(buf)
    if op == _OP_CLEAR:
        return (int(stamp_ns), None)
    if op != _OP_SET:
        raise ValueError(f"unknown goal op {op}")
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"goal position must be finite, got ({x}, {y})")
    return (int(stamp_ns), GoalCoordinate(x, y, None if math.isnan(yaw) else yaw))


# ── W5: telemetry (JSON datagram) ────────────────────────────────────────────

Point = Tuple[float, float]


@dataclass(frozen=True)
class TelemetrySnapshot:
    """One brain tick as seen from outside — the W5 message (latest-wins at the reader).

    `pose` is the pose Nav drives on (GT in Stage 1 shadow mode); `est` is the candidate
    module's latest verdict (None until the localization host produces one); `intent` is the
    commanded body twist from the recorder hook; `loop_hz` is the measured brain-loop rate
    (the GIL-stall metric, design.md D6). Everything except `stamp_ns` is Optional — a
    boot-time snapshot legitimately knows nothing yet.
    """

    stamp_ns: int                                    # tick stamp (sim clock, D8)
    pose: Optional[Tuple[float, float, float]] = None   # (x, y, yaw) Nav's driving pose
    path: Optional[List[Point]] = None                   # planned world waypoints
    goal: Optional[GoalCoordinate] = None                # the active goal, if any
    est: Optional[LocalizationOut] = None                # candidate's latest verdict
    intent: Optional[Tuple[float, float, float]] = None  # (v_x, v_y, w_z) commanded twist
    loop_hz: Optional[float] = None                      # measured brain-loop rate

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))


def encode_telemetry(snap: TelemetrySnapshot) -> bytes:
    """Serialize a snapshot to one JSON datagram."""
    doc = {
        "stamp_ns": snap.stamp_ns,
        "pose": list(snap.pose) if snap.pose is not None else None,
        "path": [list(p) for p in snap.path] if snap.path is not None else None,
        "goal": list(snap.goal) if snap.goal is not None else None,
        "est": _encode_est(snap.est),
        "intent": list(snap.intent) if snap.intent is not None else None,
        "loop_hz": snap.loop_hz,
    }
    return json.dumps(doc).encode()


def decode_telemetry(buf: bytes) -> TelemetrySnapshot:
    """Parse one JSON datagram → TelemetrySnapshot. Raises ValueError on any malformed
    document, including an `est` that breaks the LocalizationOut invariants."""
    try:
        doc = json.loads(buf)
        if not isinstance(doc, dict):
            raise ValueError("telemetry document must be a JSON object")
        goal = doc.get("goal")
        intent = doc.get("intent")
        pose = doc.get("pose")
        path = doc.get("path")
        return TelemetrySnapshot(
            stamp_ns=doc["stamp_ns"],
            pose=_triple(pose),
            path=[(float(x), float(y)) for x, y in path] if path is not None else None,
            goal=GoalCoordinate(
                float(goal[0]), float(goal[1]),
                None if goal[2] is None else float(goal[2]),
            ) if goal is not None else None,
            est=_decode_est(doc.get("est")),
            intent=_triple(intent),
            loop_hz=None if doc.get("loop_hz") is None else float(doc["loop_hz"]),
        )
    except ValueError:
        raise
    except Exception as exc:  # json errors, missing keys, wrong shapes → one error type
        raise ValueError(f"malformed telemetry datagram: {exc}") from exc


def _triple(vals: Optional[list]) -> Optional[Tuple[float, float, float]]:
    """A strict 3-tuple of floats (pose / intent) — wrong lengths raise, not truncate."""
    if vals is None:
        return None
    a, b, c = vals
    return (float(a), float(b), float(c))


def _encode_est(est: Optional[LocalizationOut]) -> Optional[dict]:
    if est is None:
        return None
    return {
        "stamp_ns": est.stamp_ns,
        "status": int(est.status),
        "pose": ([est.pose.stamp_ns, est.pose.x, est.pose.y, est.pose.yaw]
                 if est.pose is not None else None),
        "last_fix_stamp_ns": est.last_fix_stamp_ns,
    }


def _decode_est(doc: Optional[dict]) -> Optional[LocalizationOut]:
    if doc is None:
        return None
    pose = doc["pose"]
    # LocalizationOut.__post_init__ enforces pose ⇔ not-LOST and coerces the status int
    # through the enum — malformed wire data raises ValueError right here.
    return LocalizationOut(
        stamp_ns=doc["stamp_ns"],
        pose=RobotPose(*pose) if pose is not None else None,
        status=LocalizationStatus(doc["status"]),
        last_fix_stamp_ns=doc["last_fix_stamp_ns"],
    )
