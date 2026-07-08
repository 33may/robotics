"""TDD for the joystick app's keyboard→packet mapping (pure core).

The app is a keyboard-driven on-screen pad (ported from LimX's `robot-joystick`). Its
pygame window/loop is integration-only, but the mapping from held keys to a `JoyPacket`
is pure and tested here. Axes use the LimX `walk_controller` layout (axis0=v_y,
axis1=v_x, axis3=w_z); buttons use the PlayStation indices `main.py` reads for mode
combos (L1=4, Y=3 → STAND; R1=7, X=2 → WALK). Runs in the `brain` env (no pygame).
"""

import pytest

from humanoid.logic.oli.reason.teleoperation.joystick.app import keyboard_to_packet

pytestmark = pytest.mark.brain


def test_forward_sets_vx_axis1():
    pkt = keyboard_to_packet({"up"}, stamp_ns=5)
    assert pkt.stamp_ns == 5
    assert pkt.axes[1] == pytest.approx(1.0)   # axis1 = v_x
    assert pkt.axes[0] == pytest.approx(0.0)
    assert pkt.axes[3] == pytest.approx(0.0)


def test_back_is_negative_vx():
    assert keyboard_to_packet({"down"}, 0).axes[1] == pytest.approx(-1.0)


def test_strafe_sets_vy_axis0():
    assert keyboard_to_packet({"left"}, 0).axes[0] == pytest.approx(-1.0)
    assert keyboard_to_packet({"right"}, 0).axes[0] == pytest.approx(1.0)


def test_yaw_sets_axis3():
    assert keyboard_to_packet({"yaw_left"}, 0).axes[3] == pytest.approx(1.0)
    assert keyboard_to_packet({"yaw_right"}, 0).axes[3] == pytest.approx(-1.0)


def test_opposite_keys_cancel():
    assert keyboard_to_packet({"up", "down"}, 0).axes[1] == pytest.approx(0.0)


def test_neutral_is_all_zero_axes():
    pkt = keyboard_to_packet(set(), 0)
    assert all(a == pytest.approx(0.0) for a in pkt.axes)


def test_stand_combo_sets_l1_and_y_buttons():
    # LimX: L1+Y → stand. Indices buttons[4]=L1, buttons[3]=Y.
    pkt = keyboard_to_packet({"b_L1", "b_Y"}, 0)
    assert pkt.buttons[4] == 1 and pkt.buttons[3] == 1
    assert pkt.buttons[7] == 0 and pkt.buttons[2] == 0


def test_walk_combo_sets_r1_and_x_buttons():
    # LimX: R1+X → walk. Indices buttons[7]=R1, buttons[2]=X.
    pkt = keyboard_to_packet({"b_R1", "b_X"}, 0)
    assert pkt.buttons[7] == 1 and pkt.buttons[2] == 1


def test_buttons_default_unpressed():
    assert set(keyboard_to_packet(set(), 0).buttons) == {0}
