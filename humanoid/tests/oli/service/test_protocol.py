"""TDD for the brain service seam's wire protocol (oli/service/protocol.py).

The service seam (locbench design.md D5) is the brain's goal-in / telemetry-out boundary:
W4 carries `GoalCoordinate` set/clear TO the brain (fixed-size struct datagram, debug_pose
pattern); W5 carries the brain's telemetry snapshot OUT (JSON datagram — the path is
variable-length). Codecs are pure (stdlib), so this runs in the `brain` env.
"""

import math

import pytest

from humanoid.logic.oli.reason.localization import (
    LocalizationOut,
    LocalizationStatus,
    RobotPose,
)
from humanoid.logic.oli.reason.nav import GoalCoordinate
from humanoid.logic.oli.service.protocol import (
    GOAL_NBYTES,
    TelemetrySnapshot,
    decode_goal,
    decode_telemetry,
    encode_goal_clear,
    encode_goal_set,
    encode_telemetry,
)

pytestmark = pytest.mark.brain


# ── goal channel codec (W4, fixed struct) ────────────────────────────────────


def test_goal_frame_is_fixed_size():
    assert len(encode_goal_set(1, 2.0, 3.0, 0.5)) == GOAL_NBYTES
    assert len(encode_goal_set(1, 2.0, 3.0, None)) == GOAL_NBYTES
    assert len(encode_goal_clear(1)) == GOAL_NBYTES


def test_goal_set_roundtrip_with_yaw():
    stamp, goal = decode_goal(encode_goal_set(123, 1.5, -2.25, 0.75))
    assert stamp == 123
    assert goal == GoalCoordinate(1.5, -2.25, 0.75)


def test_goal_set_roundtrip_without_yaw():
    stamp, goal = decode_goal(encode_goal_set(7, 4.0, 5.0, None))
    assert stamp == 7
    assert goal == GoalCoordinate(4.0, 5.0, None)  # NaN on the wire → None back


def test_goal_clear_roundtrip():
    stamp, goal = decode_goal(encode_goal_clear(99))
    assert stamp == 99
    assert goal is None


def test_decode_goal_rejects_wrong_length():
    with pytest.raises(ValueError):
        decode_goal(b"\x00" * 8)


def test_decode_goal_rejects_unknown_op():
    buf = bytearray(encode_goal_set(1, 0.0, 0.0, None))
    buf[0] = 42  # not a known op byte
    with pytest.raises(ValueError):
        decode_goal(bytes(buf))


def test_decode_goal_rejects_nonfinite_position():
    # A NaN x/y is a malformed goal (NaN is only meaningful in the yaw slot).
    buf = encode_goal_set(1, math.nan, 0.0, None)
    with pytest.raises(ValueError):
        decode_goal(buf)


# ── telemetry codec (W5, JSON datagram) ──────────────────────────────────────


def _full_snapshot() -> TelemetrySnapshot:
    return TelemetrySnapshot(
        stamp_ns=1_000_000,
        pose=(1.0, 2.0, 0.5),
        path=[(1.0, 2.0), (1.5, 2.5), (2.0, 3.0)],
        goal=GoalCoordinate(2.0, 3.0, None),
        est=LocalizationOut(
            stamp_ns=999_000,
            pose=RobotPose(999_000, 1.05, 2.02, 0.48),
            status=LocalizationStatus.TRACKING,
            last_fix_stamp_ns=990_000,
        ),
        intent=(0.3, 0.0, 0.1),
        loop_hz=99.5,
    )


def test_telemetry_roundtrip_full():
    snap = _full_snapshot()
    out = decode_telemetry(encode_telemetry(snap))
    assert out == snap
    # est comes back as a REAL LocalizationOut (self-validating contract), not a dict.
    assert isinstance(out.est, LocalizationOut)
    assert out.est.pose == RobotPose(999_000, 1.05, 2.02, 0.48)
    assert out.goal == GoalCoordinate(2.0, 3.0, None)


def test_telemetry_roundtrip_minimal():
    # Boot-time snapshot: nothing known yet except the tick stamp.
    snap = TelemetrySnapshot(stamp_ns=5)
    out = decode_telemetry(encode_telemetry(snap))
    assert out == snap
    assert out.pose is None and out.path is None and out.goal is None
    assert out.est is None and out.intent is None and out.loop_hz is None


def test_telemetry_roundtrip_lost_est():
    snap = TelemetrySnapshot(
        stamp_ns=10,
        est=LocalizationOut(stamp_ns=9, pose=None, status=LocalizationStatus.LOST),
    )
    out = decode_telemetry(encode_telemetry(snap))
    assert out.est is not None
    assert out.est.status is LocalizationStatus.LOST and out.est.pose is None


def test_decode_telemetry_rejects_garbage():
    with pytest.raises(ValueError):
        decode_telemetry(b"\xff\xfenot json")


def test_decode_telemetry_rejects_invariant_breaking_est():
    # A wire message claiming LOST but carrying a pose must NOT decode into a
    # contract-violating LocalizationOut — the self-validating contract is the filter.
    good = encode_telemetry(_full_snapshot())
    import json

    doc = json.loads(good)
    doc["est"]["status"] = int(LocalizationStatus.LOST)  # pose still present → invalid
    with pytest.raises(ValueError):
        decode_telemetry(json.dumps(doc).encode())
