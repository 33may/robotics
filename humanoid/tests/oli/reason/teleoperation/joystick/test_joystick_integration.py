"""End-to-end joystick path (no mocks): app mapping → wire → source → Teleop.

Proves the whole reason-layer joystick chain agrees on conventions: a key press in the
app produces a JoyPacket whose axes, sent over real UDP and decoded by the socket
source, drive the correct body-frame velocity in the emitted PolicyIn. This guards the
axis-layout contract end to end (a single off-by-one in any link would flip an axis).
Runs in the `brain` env (pure: stdlib socket + our modules, no pygame).
"""

import socket
import time

import numpy as np
import pytest

from humanoid.logic.oli import Mode, Observation
from humanoid.logic.oli.reason.teleoperation.joystick.app import keyboard_to_packet
from humanoid.logic.oli.reason.teleoperation.joystick.protocol import pack_joy
from humanoid.logic.oli.reason.teleoperation.joystick.source import SocketJoystickSource
from humanoid.logic.oli.reason.teleoperation.joystick.teleop import Teleop

pytestmark = pytest.mark.brain

N = 31


def _obs():
    return Observation(
        stamp_ns=1, q=np.zeros(N), dq=np.zeros(N), tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _drive(src, held):
    """Simulate the app: map keys → packet → UDP datagram → source, return polled JoyPacket."""
    pkt = keyboard_to_packet(held, stamp_ns=1)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(pack_joy(pkt), ("127.0.0.1", src.port))
    finally:
        s.close()
    for _ in range(200):
        joy = src.poll()
        if joy is not None:
            return joy
        time.sleep(0.001)
    raise AssertionError("no datagram delivered")


def test_forward_key_drives_forward_velocity():
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    teleop = Teleop(mode=Mode.WALK)
    try:
        joy = _drive(src, {"up"})
        pin = teleop.to_policy_in(_obs(), joy)
        assert pin.intent.v_x > 0.0          # forward
        assert pin.intent.v_y == pytest.approx(0.0)
        assert pin.intent.w_z == pytest.approx(0.0)
        assert pin.intent.mode == Mode.WALK
    finally:
        src.close()


def test_turn_key_drives_yaw_only():
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    teleop = Teleop(mode=Mode.WALK)
    try:
        pin = teleop.to_policy_in(_obs(), _drive(src, {"yaw_left"}))
        assert pin.intent.w_z > 0.0
        assert pin.intent.v_x == pytest.approx(0.0)
        assert pin.intent.v_y == pytest.approx(0.0)
    finally:
        src.close()


def test_walk_combo_keys_switch_mode_end_to_end():
    # U+J (R1+X) pressed in the app must latch WALK by the time it reaches the PolicyIn.
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    teleop = Teleop(mode=Mode.STAND)
    try:
        pin = teleop.to_policy_in(_obs(), _drive(src, {"b_R1", "b_X"}))
        assert pin.intent.mode == Mode.WALK
        assert teleop.mode == Mode.WALK
    finally:
        src.close()
