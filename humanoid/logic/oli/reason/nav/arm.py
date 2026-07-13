"""nav/arm.py — the arm-gated navigation reason: teleop until you engage Nav.

Wraps a joystick `Teleop` and a `Nav` behind the single reason seam. **Disarmed** (default): the
operator drives via the joystick — position Oli, preview the path. **Armed**: `Nav` drives —
localize → plan → pure-pursuit → velocity `Intent`, following the path to the goal. `Nav` plans
regardless of arm state (so its `.path` stays fresh for the map render); the flag only gates which
`Intent` reaches the Action. Same `to_policy_in(obs, joy)` + `mode`/`set_mode` as `Teleop`, so it
drops into the Orchestrator unchanged. Pure: no isaacsim/limxsdk.
"""

from __future__ import annotations

from ...contracts import Mode, Observation, PolicyIn


class ArmedNav:
    """Compose `Teleop` (operator) + `Nav` (autonomy) behind an arm flag.

    Disarmed → `teleop.to_policy_in(obs, joy)`; armed → `nav.to_policy_in(obs)` (which holds
    zero-velocity when there is no goal / pose / path — so arming with nothing set is safe).
    """

    def __init__(self, teleop, nav) -> None:
        self._teleop = teleop
        self._nav = nav
        self._armed = False

    @property
    def mode(self) -> Mode:
        return self._teleop.mode

    def set_mode(self, mode: Mode) -> None:
        self._teleop.set_mode(mode)

    @property
    def armed(self) -> bool:
        return self._armed

    def set_armed(self, armed: bool) -> None:
        self._armed = bool(armed)

    def to_policy_in(self, observation: Observation, joy=None) -> PolicyIn:
        if self._armed:
            return self._nav.to_policy_in(observation)      # Nav drives (holds if no goal/pose/path)
        return self._teleop.to_policy_in(observation, joy)  # operator drives
