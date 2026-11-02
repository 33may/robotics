"""teleop.py — Teleoperation: a JoyPacket + Observation → PolicyIn (D5).

`JoystickAdapter` maps stick axes to a body-frame velocity command, faithful to
`walk_controller._update_commands_from_joy` (v_x←axis1, v_y←axis0, w_z←axis3, each
clipped to its max). `Teleop` consumes the operator's `JoyPacket` (axes + buttons) and
emits `PolicyIn`: axes drive velocity, and button combos switch the held mode the way
LimX `main.py` does (L1+Y → STAND, R1+X → WALK). Mode is **sticky** — a combo latches it
until the other combo is pressed, so a neutral poll keeps walking. Pure: no
isaacsim/limxsdk; the JoyPacket arrives from a source (see `source.py`).
"""

from __future__ import annotations

from typing import Optional, Sequence

from ....contracts import Intent, Mode, Observation, PolicyIn
from .protocol import JoyPacket

# PlayStation button indices (LimX main.py): a 2-button combo selects a mode.
_BTN_X, _BTN_Y, _BTN_L1, _BTN_R1 = 2, 3, 4, 7
# Checked in order; first satisfied combo wins if both are somehow held.
_MODE_COMBOS = (
    ((_BTN_L1, _BTN_Y), Mode.STAND),  # L1 + Y  (keyboard Q + I)
    ((_BTN_R1, _BTN_X), Mode.WALK),   # R1 + X  (keyboard U + J)
)


def _clip(x: float, limit: float) -> float:
    return max(-limit, min(limit, float(x)))


class JoystickAdapter:
    """Maps joystick axes to (v_x, v_y, w_z) in body frame, clipped to per-axis maxima."""

    def __init__(self, max_vx: float = 0.5, max_vy: float = 0.3, max_vz: float = 0.5) -> None:
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_vz = max_vz

    def axes_to_velocity(self, axes: Sequence[float]) -> tuple:
        # walk_controller convention: v_x←axis1, v_y←axis0, w_z←axis3
        return (
            _clip(axes[1], self.max_vx),
            _clip(axes[0], self.max_vy),
            _clip(axes[3], self.max_vz),
        )


class Teleop:
    """Reason foundation: combine held mode (+ button switches) and joystick velocity."""

    def __init__(self, mode: Mode = Mode.STAND, adapter: Optional[JoystickAdapter] = None) -> None:
        self._mode = mode
        self._adapter = adapter if adapter is not None else JoystickAdapter()

    @property
    def mode(self) -> Mode:
        return self._mode

    def set_mode(self, mode: Mode) -> None:
        self._mode = mode

    def _apply_buttons(self, buttons: Sequence[int]) -> None:
        """Latch the mode if a recognized combo is fully pressed; else hold."""
        def held(i: int) -> bool:
            return i < len(buttons) and bool(buttons[i])

        for combo, mode in _MODE_COMBOS:
            if all(held(i) for i in combo):
                self._mode = mode
                return

    def to_policy_in(
        self, observation: Observation, joy: Optional[JoyPacket] = None
    ) -> PolicyIn:
        if joy is None:
            v_x = v_y = w_z = 0.0
        else:
            self._apply_buttons(joy.buttons)
            v_x, v_y, w_z = self._adapter.axes_to_velocity(joy.axes)
        return PolicyIn(
            observation=observation,
            intent=Intent(mode=self._mode, v_x=v_x, v_y=v_y, w_z=w_z),
        )
