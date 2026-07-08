"""glide.py — glide-mode kinematic locomotion (MAY-172): the wire command + the model.

Glide is the demo-time stand-in for the LimX-gated dynamic walk. Instead of the walk
policy driving 31 joints and PhysX producing base motion, the brain forwards a body-frame
base-velocity command and the World glides Oli's base kinematically. Two pure pieces live
here:

  - `GlideCmd`   — the brain→World message (analogue of `PolicyOut`); crosses the SAME
                   Comm socket as a new `GLIDE_CMD` frame, so the walk contracts stay
                   byte-identical.
  - `GlideModel` — the "fake physics": commanded twist → accel-limited, integrated base
                   pose. This is the exact seam the MuJoCo-fitted model (or the real walk)
                   later replaces; the World instantiates it, the brain never does.

Pure: stdlib only, no isaacsim/limxsdk — the deployment-invariant boundary holds for
glide exactly as for the walk flow. When the real walk lands, glide is dropped and the
walk `PolicyOut` path runs unchanged (same Orchestrator loop, same joystick/Teleop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import PolicyIn


@dataclass(frozen=True)
class GlideCmd:
    """Body-frame base-velocity command, brain→World, for kinematic glide.

    `stamp_ns` is the Observation stamp this command was computed from (sim time — the
    same D8 pacing clock as `PolicyOut`). The velocities are the operator's `Intent`
    forwarded verbatim; the World integrates them (accel/turn-limited) into base motion.
    """

    stamp_ns: int      # SIM-time nanoseconds — the D8 pacing clock
    v_x: float = 0.0   # forward  [m/s]  (body frame)
    v_y: float = 0.0   # lateral  [m/s]
    w_z: float = 0.0   # yaw rate [rad/s]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "v_x", float(self.v_x))
        object.__setattr__(self, "v_y", float(self.v_y))
        object.__setattr__(self, "w_z", float(self.w_z))


def _approach(current: float, target: float, max_delta: float) -> float:
    """Move `current` toward `target` by at most ±`max_delta` (the accel clamp).

    Lands exactly on `target` once within reach, so a command both ramps up *and*
    settles without overshoot, and a zero command decays exactly to rest.
    """
    delta = target - current
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    return current + delta


class GlideModel:
    """The glide "fake physics": commanded body twist → accel-limited base pose.

    Holds the base pose (world frame: `x`, `y`, `yaw`) and the achieved body-frame twist
    (`vx`, `vy`, `wz`). Each `step` slews the twist toward the command under a per-axis
    acceleration limit (`lin_accel` for vx/vy, `yaw_accel` for wz), then forward-Euler
    integrates the pose. Stateful but pure — no isaacsim; the World owns an instance and
    feeds its pose/twist to the kinematic base each tick. Swapping in the MuJoCo-fitted
    response (or the real walk) means replacing this object, not the loop around it.
    """

    def __init__(
        self,
        lin_accel: float = 2.0,
        yaw_accel: float = 4.0,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        self.lin_accel = float(lin_accel)   # m/s²   — vx/vy slew rate
        self.yaw_accel = float(yaw_accel)   # rad/s² — wz slew rate
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

    def step(
        self, cmd_vx: float, cmd_vy: float, cmd_wz: float, dt: float
    ) -> tuple:
        """Advance one tick: accel-limit the twist toward the command, integrate the pose.

        Returns the new world-frame pose `(x, y, yaw)`. The achieved twist is on
        `self.vx/vy/wz` for the caller (the World sets the kinematic base from both).
        """
        lin = self.lin_accel * dt
        yaw_step = self.yaw_accel * dt
        self.vx = _approach(self.vx, float(cmd_vx), lin)
        self.vy = _approach(self.vy, float(cmd_vy), lin)
        self.wz = _approach(self.wz, float(cmd_wz), yaw_step)
        # Body → world at the current heading (forward Euler), then advance the heading.
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vx * cos_y - self.vy * sin_y) * dt
        self.y += (self.vx * sin_y + self.vy * cos_y) * dt
        self.yaw += self.wz * dt
        return (self.x, self.y, self.yaw)


class GlideAction:
    """Glide-mode Action: forward the operator's velocity Intent as a `GlideCmd`.

    A drop-in for `PolicyRunner` in the Orchestrator (same `.step(policy_in)` seam) — but
    instead of running the walk ONNX it passes the body-frame velocity command straight
    through, stamped with the observation it was computed from. The World integrates it
    (`GlideModel`) into base motion. Stateless and pure; `Intent.mode` is ignored (glide
    is the mode). Swapping in the real walk = swap this back for `PolicyRunner`.

    `speed_scale` multiplies the commanded velocity (default 1.0 = verbatim). The dev-app
    demo boots it at 3× (see `brain_link`) so the kinematic glide/turn feels lively without
    touching the joystick's shared max limits; the walk/real path is unaffected (it uses
    `PolicyRunner`, not this).
    """

    def __init__(self, speed_scale: float = 1.0) -> None:
        self._scale = float(speed_scale)

    def step(self, policy_in: PolicyIn) -> GlideCmd:
        intent = policy_in.intent
        s = self._scale
        return GlideCmd(
            stamp_ns=policy_in.observation.stamp_ns,
            v_x=intent.v_x * s, v_y=intent.v_y * s, w_z=intent.w_z * s,
        )
