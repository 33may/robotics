"""TDD for joystick sources (reason layer, joystick method).

A source answers "what is the stick doing now?" via `poll()` → a `JoyPacket` (axes in
the LimX `walk_controller` layout axis0=v_y, axis1=v_x, axis3=w_z; plus buttons), or
None. `FixedJoystick` is the no-device stand-in for the sim walk demo. Sources must
satisfy `JoystickSource` and stay brain-pure (no isaacsim/limxsdk). Runs in `brain`.
"""

import socket
import time

import pytest

from humanoid.logic.oli.reason.teleoperation.joystick.protocol import JoyPacket, pack_joy
from humanoid.logic.oli.reason.teleoperation.joystick.source import (
    FixedJoystick,
    JoystickSource,
    SocketJoystickSource,
)
from humanoid.logic.oli.reason.teleoperation.joystick.teleop import JoystickAdapter

pytestmark = pytest.mark.brain


def _send(port, axes, buttons=(0,), stamp=1):
    """Fire one JoyPacket datagram at a SocketJoystickSource bound on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(pack_joy(JoyPacket(stamp_ns=stamp, axes=list(axes), buttons=list(buttons))),
                 ("127.0.0.1", port))
    finally:
        s.close()


def _recv_until(src, attempts=200):
    """Poll until a packet has been delivered (UDP loopback is async). Returns JoyPacket."""
    for _ in range(attempts):
        pkt = src.poll()
        if pkt is not None:
            return pkt
        time.sleep(0.001)
    return None


def test_fixed_joystick_satisfies_source_protocol():
    assert isinstance(FixedJoystick(), JoystickSource)


def test_fixed_joystick_axes_layout():
    # axis0 = v_y, axis1 = v_x, axis2 unused, axis3 = w_z
    pkt = FixedJoystick(v_x=0.3, v_y=0.1, w_z=-0.2).poll()
    assert list(pkt.axes) == pytest.approx([0.1, 0.3, 0.0, -0.2])


def test_fixed_joystick_round_trips_through_adapter():
    # The source's layout must feed JoystickAdapter to recover (v_x, v_y, w_z).
    pkt = FixedJoystick(v_x=0.4, v_y=0.2, w_z=0.3).poll()
    v_x, v_y, w_z = JoystickAdapter().axes_to_velocity(pkt.axes)
    assert (v_x, v_y, w_z) == pytest.approx((0.4, 0.2, 0.3))


def test_fixed_joystick_defaults_to_zero():
    assert list(FixedJoystick().poll().axes) == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_fixed_joystick_has_no_buttons():
    # No physical buttons → mode is never switched by a FixedJoystick.
    assert list(FixedJoystick().poll().buttons) == []


# ── SocketJoystickSource (UDP, latest-wins) ──────────────────────────────────


def test_socket_source_satisfies_protocol():
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    try:
        assert isinstance(src, JoystickSource)
    finally:
        src.close()


def test_socket_source_returns_none_before_any_packet():
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    try:
        assert src.poll() is None
    finally:
        src.close()


def test_socket_source_returns_latest_axes():
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    try:
        _send(src.port, [0.1, 0.4, 0.0, -0.2])
        pkt = _recv_until(src)
        assert pkt is not None
        assert list(pkt.axes) == pytest.approx([0.1, 0.4, 0.0, -0.2], abs=1e-6)
    finally:
        src.close()


def test_socket_source_carries_buttons():
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    try:
        _send(src.port, [0.0, 0.0, 0.0, 0.0], buttons=[0, 0, 1, 0, 0, 0, 0, 1])
        pkt = _recv_until(src)
        assert pkt is not None
        assert list(pkt.buttons) == [0, 0, 1, 0, 0, 0, 0, 1]
    finally:
        src.close()


def test_socket_source_drains_to_newest():
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    try:
        # Queue three packets before polling; a single poll must yield the NEWEST.
        for i, v in enumerate([0.1, 0.2, 0.9]):
            _send(src.port, [v, 0.0, 0.0, 0.0], stamp=i + 1)
        time.sleep(0.02)  # let all three arrive in the socket buffer
        pkt = src.poll()
        assert pkt is not None
        assert pkt.axes[0] == pytest.approx(0.9, abs=1e-6)
    finally:
        src.close()


def test_socket_source_holds_last_when_no_new_packet():
    src = SocketJoystickSource(host="127.0.0.1", port=0)
    try:
        _send(src.port, [0.5, 0.0, 0.0, 0.0])
        first = _recv_until(src)
        assert first is not None
        # No new datagram → hold the last sample (avoids a jerk mid-stride).
        assert list(src.poll().axes) == pytest.approx(list(first.axes), abs=1e-6)
    finally:
        src.close()