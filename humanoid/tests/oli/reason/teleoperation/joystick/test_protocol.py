"""TDD for the joystick wire (reason layer, joystick method).

The joystick app → brain link is its OWN tiny protocol, separate from the comm/
state↔cmd wire: a single latest-wins datagram carrying the full stick state
`JoyPacket{stamp_ns, axes[], buttons[]}`. Variable-length (controllers differ in
axis/button counts), pure stdlib struct, no isaacsim/limxsdk/numpy. Runs in `brain`.
"""

import struct

import pytest

from humanoid.logic.oli.reason.teleoperation.joystick.protocol import (
    JoyPacket,
    JoyProtocolError,
    pack_joy,
    unpack_joy,
)

pytestmark = pytest.mark.brain


def test_round_trips_axes_and_buttons():
    pkt = JoyPacket(stamp_ns=123456789, axes=[0.1, -0.2, 0.0, 0.5], buttons=[0, 1, 0, 1])
    out = unpack_joy(pack_joy(pkt))
    assert out.stamp_ns == 123456789
    assert out.axes == pytest.approx([0.1, -0.2, 0.0, 0.5], abs=1e-6)
    assert list(out.buttons) == [0, 1, 0, 1]


def test_round_trips_variable_lengths():
    # A different controller: 6 axes, 12 buttons — the wire must not hard-code counts.
    pkt = JoyPacket(stamp_ns=7, axes=[0.0] * 6, buttons=[1] * 12)
    out = unpack_joy(pack_joy(pkt))
    assert len(out.axes) == 6
    assert len(out.buttons) == 12


def test_empty_axes_and_buttons_round_trip():
    out = unpack_joy(pack_joy(JoyPacket(stamp_ns=1, axes=[], buttons=[])))
    assert out.axes == []
    assert out.buttons == []


def test_short_buffer_raises():
    good = pack_joy(JoyPacket(stamp_ns=1, axes=[0.0, 0.0, 0.0, 0.0], buttons=[0]))
    with pytest.raises(JoyProtocolError):
        unpack_joy(good[:4])  # truncated header


def test_truncated_payload_raises():
    good = pack_joy(JoyPacket(stamp_ns=1, axes=[0.0, 0.0, 0.0, 0.0], buttons=[0, 1]))
    with pytest.raises(JoyProtocolError):
        unpack_joy(good[:-3])  # header says more axes/buttons than bytes present


def test_bad_magic_raises():
    payload = struct.pack("<I", 0xDEADBEEF) + b"\x00" * 16
    with pytest.raises(JoyProtocolError):
        unpack_joy(payload)
