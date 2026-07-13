"""localization/contracts.py — the localization module's in/out contracts.

Brain-internal module contracts (design.md D2–D6, change `may-173-reason-module-separation`),
distinct from the world-edge contracts in `oli/contracts.py`. The flow they describe:

    LocalizationIn (frame-paced bundle) ──► [LocalizationModule] ──► LocalizationOut
                                                   ▲
                              LocalizationSetup (map artifacts + hint + calibration, at start)

`RobotPose` lives here — it is localization's OUTPUT type; nav imports it from this package so
dependencies point nav→localization, never backwards. Pure: numpy/stdlib only, never
isaacsim/limxsdk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Mapping, Optional

from ...contracts import CameraFrame, Intent, Observation


@dataclass(frozen=True)
class RobotPose:
    """Planar robot pose in the map frame (SE(2)) — the Localizer's output, the planner's input.

    `map frame` = the fixed ground-truth scene frame the occupancy grid is baked in (the scene
    loaded by `--scene`). `stamp_ns` is SIM-time (the same D8 clock as `Observation`), so a pose
    aligns with the observation/camera frame it was derived from. Planar (x, y, yaw) for the 2D
    nav PoC; a future SE(3) pose extends this **without changing the `Localizer` seam** (the
    planner consumes whatever pose type the localizer emits).
    """

    stamp_ns: int       # SIM-time nanoseconds — the D8 pacing clock
    x: float = 0.0      # map-frame X [m]
    y: float = 0.0      # map-frame Y [m]
    yaw: float = 0.0    # map-frame heading [rad], CCW about +Z

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "yaw", float(self.yaw))


class LocalizationStatus(IntEnum):
    """The trust machine loc-mode SLAM actually is (design.md D3): a map-anchored fix, a pose
    propagated by odometry/VO since the last fix (usable but aging — the E2 dead-reckon regime),
    or nothing usable. An `IntEnum` so it crosses a wire untouched (debug-pose datagram pattern).
    """

    TRACKING = 0   # pose anchored to the map by a recent visual fix
    DRIFTING = 1   # propagated since the last map fix — usable, but aging
    LOST = 2       # no usable pose (pose is None)


@dataclass(frozen=True)
class LocalizationIn:
    """Frame-paced input bundle (design.md D2): one per camera tick, ~15–30 Hz.

    Localization is vision-paced — the ~100 Hz `Observation` nearest (≤) the frame stamp rides
    along (joints → FK camera extrinsics, IMU), plus the commanded twist as an optional motion
    prior. Candidates consume what they need and ignore the rest. This bundle is also the bag's
    record unit in the locbench harness (change b).
    """

    stamp_ns: int                        # = the pacing camera frame's stamp (sim clock)
    frames: Mapping[str, CameraFrame]    # "head" / "chest" — whatever arrived this tick
    observation: Observation             # nearest obs ≤ stamp_ns
    intent: Optional[Intent] = None      # commanded body twist — motion prior

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        if not self.frames:
            raise ValueError("LocalizationIn requires at least one camera frame")
        object.__setattr__(self, "frames", dict(self.frames))


@dataclass(frozen=True)
class LocalizationOut:
    """Map-frame verdict (design.md D3, D4). The module owns internal alignment (it received the
    map + initial-pose hint in `LocalizationSetup`), so `pose` is ALWAYS map-frame — downstream
    (bench scorer, live client, Nav) never anchors or refits, keeping any constant bias visible.

    Invariant: `pose is None` ⇔ `status is LOST`. `last_fix_stamp_ns` is the objective trust
    signal — `now − last_fix` maps onto the measured E2 budget (how long dead-reckoning stays
    safe). Fixed-size fields only, so this crosses a process boundary as a small datagram.
    """

    stamp_ns: int                            # stamp of the LocalizationIn it answers
    pose: Optional[RobotPose]                # map-frame SE(2); None iff LOST
    status: LocalizationStatus
    last_fix_stamp_ns: Optional[int] = None  # when the map last confirmed us

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        # Coerce status THROUGH the enum: a raw int decoded off a wire datagram (this enum is
        # designed to cross one) must land as a real LocalizationStatus or raise — never be
        # stored as a bare int that dodges the invariant below.
        object.__setattr__(self, "status", LocalizationStatus(self.status))
        if self.pose is not None and not isinstance(self.pose, RobotPose):
            raise TypeError(f"pose must be a RobotPose, got {type(self.pose).__name__}")
        if (self.pose is None) != (self.status is LocalizationStatus.LOST):
            raise ValueError(
                f"pose is None iff status is LOST (got pose={self.pose!r}, "
                f"status={self.status!r})"
            )
        if self.last_fix_stamp_ns is not None:
            object.__setattr__(self, "last_fix_stamp_ns", int(self.last_fix_stamp_ns))


@dataclass(frozen=True)
class LocalizationSetup:
    """Start-time materials (design.md D4, D5): the algorithm-private map artifacts (opaque dir —
    formats are mutually incompatible across candidates, so nobody else reads it), the known-start
    pose hint (no kidnapped-robot recovery in v1), and the calibration blob (camera mounts, depth
    encoding — the bag's meta in the bench, live config on the robot).
    """

    map_dir: Path
    initial_pose: Optional[RobotPose] = None
    calibration: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_dir", Path(self.map_dir))
