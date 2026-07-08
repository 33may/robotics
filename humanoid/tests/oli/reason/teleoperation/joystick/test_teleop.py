"""TDD for Teleop / JoystickAdapter (reason layer, joystick method).

Reasoning is the single producer of PolicyIn (D5): it takes an Observation plus the
operator's joystick (a `JoyPacket`) and emits PolicyIn directly. Axes map to velocity
(v_x=axis1, v_y=axis0, w_z=axis3, clipped); button combos switch the held mode the way
LimX `main.py` does (L1+Y → STAND, R1+X → WALK). Mode is sticky — held until the other
combo is pressed. Pure: runs in the `brain` env.
"""

import numpy as np
import pytest

from humanoid.logic.oli import Intent, Mode, Observation, PolicyIn
from humanoid.logic.oli.reason.teleoperation.joystick.protocol import JoyPacket
from humanoid.logic.oli.reason.teleoperation.joystick.teleop import JoystickAdapter, Teleop

pytestmark = pytest.mark.brain

N = 31


def _obs():
    return Observation(
        stamp_ns=1, q=np.zeros(N), dq=np.zeros(N), tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _joy(axes=(0.0, 0.0, 0.0, 0.0), *pressed):
    """JoyPacket with `axes` and an 8-button array with `pressed` indices set."""
    buttons = [0] * 8
    for i in pressed:
        buttons[i] = 1
    return JoyPacket(stamp_ns=1, axes=list(axes), buttons=buttons)


# PlayStation indices (LimX main.py): L1=4, Y=3 → stand; R1=7, X=2 → walk.
STAND = (4, 3)
WALK = (7, 2)


# ── JoystickAdapter (unchanged axis mapping) ─────────────────────────────────


def test_axes_to_velocity_mapping():
    adapter = JoystickAdapter()
    v_x, v_y, w_z = adapter.axes_to_velocity([0.2, 0.4, 0.0, -0.1])
    assert v_x == pytest.approx(0.4)   # v_x ← axis1
    assert v_y == pytest.approx(0.2)   # v_y ← axis0
    assert w_z == pytest.approx(-0.1)  # w_z ← axis3


def test_axes_clipped_to_limits():
    adapter = JoystickAdapter()  # defaults max_vx=0.5, max_vy=0.3, max_vz=0.5
    assert adapter.axes_to_velocity([10.0, 10.0, 0.0, 10.0]) == pytest.approx((0.5, 0.3, 0.5))
    assert adapter.axes_to_velocity([-10.0, -10.0, 0.0, -10.0]) == pytest.approx((-0.5, -0.3, -0.5))


# ── Teleop: axes → velocity ──────────────────────────────────────────────────


def test_teleop_emits_policy_in_with_held_mode():
    tele = Teleop()  # default mode STAND
    obs = _obs()
    pin = tele.to_policy_in(obs, _joy(axes=[0.1, 0.5, 0.0, 0.2]))
    assert isinstance(pin, PolicyIn)
    assert pin.observation is obs
    assert pin.intent.mode == Mode.STAND
    assert pin.intent.v_x == pytest.approx(0.5)
    assert pin.intent.w_z == pytest.approx(0.2)


def test_teleop_no_joystick_is_zero_velocity():
    tele = Teleop()
    tele.set_mode(Mode.WALK)
    pin = tele.to_policy_in(_obs())  # no joystick → zero command, hold mode
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)
    assert pin.intent.mode == Mode.WALK


def test_set_mode_still_works():
    tele = Teleop()
    tele.set_mode(Mode.WALK)
    pin = tele.to_policy_in(_obs(), _joy(axes=[0.0, 0.3, 0.0, 0.0]))
    assert pin.intent.mode == Mode.WALK
    assert pin.intent.v_x == pytest.approx(0.3)


# ── Teleop: buttons → mode (NEW) ─────────────────────────────────────────────


def test_stand_combo_switches_to_stand():
    tele = Teleop(mode=Mode.WALK)
    pin = tele.to_policy_in(_obs(), _joy((0.0, 0.0, 0.0, 0.0), *STAND))
    assert pin.intent.mode == Mode.STAND
    assert tele.mode == Mode.STAND  # sticky on the Teleop


def test_walk_combo_switches_to_walk():
    tele = Teleop(mode=Mode.STAND)
    pin = tele.to_policy_in(_obs(), _joy((0.0, 0.0, 0.0, 0.0), *WALK))
    assert pin.intent.mode == Mode.WALK
    assert tele.mode == Mode.WALK


def test_mode_is_sticky_across_neutral_polls():
    tele = Teleop(mode=Mode.STAND)
    tele.to_policy_in(_obs(), _joy((0.0, 0.0, 0.0, 0.0), *WALK))   # press R1+X
    pin = tele.to_policy_in(_obs(), _joy(axes=[0.0, 0.4, 0.0, 0.0]))  # release, just walk fwd
    assert pin.intent.mode == Mode.WALK
    assert pin.intent.v_x == pytest.approx(0.4)


def test_partial_combo_does_not_switch():
    tele = Teleop(mode=Mode.STAND)
    pin = tele.to_policy_in(_obs(), _joy((0.0, 0.0, 0.0, 0.0), 7))  # only R1, not X
    assert pin.intent.mode == Mode.STAND


def test_walk_combo_switches_and_drives_simultaneously():
    tele = Teleop(mode=Mode.STAND)
    pin = tele.to_policy_in(_obs(), _joy((0.0, 0.6, 0.0, 0.0), *WALK))
    assert pin.intent.mode == Mode.WALK
    assert pin.intent.v_x == pytest.approx(0.5)  # 0.6 clipped to max_vx
