"""TDD for the glide motion-command wire (GlideCmd ↔ GLIDE_CMD).

Glide mode carries a body-frame base-velocity command on the SAME Comm socket as the
walk flow, added as a NEW message type — so `CMD`/`PolicyOut` and `STATE_IMU` stay
byte-identical (the "don't alter the real flow" invariant, checked below). Pure: runs
in the `brain` env.
"""

import pytest

from humanoid.logic.oli.glide import GlideCmd
from humanoid.logic.oli.comm import codec
from humanoid.logic.oli.comm import protocol as p

pytestmark = pytest.mark.brain


# ── GlideCmd contract ─────────────────────────────────────────────────────────

def test_glide_cmd_coerces_stamp_and_holds_twist():
    cmd = GlideCmd(stamp_ns=42.9, v_x=0.3, v_y=-0.1, w_z=0.2)
    assert cmd.stamp_ns == 42 and isinstance(cmd.stamp_ns, int)  # int-coerced
    assert cmd.v_x == pytest.approx(0.3)
    assert cmd.v_y == pytest.approx(-0.1)
    assert cmd.w_z == pytest.approx(0.2)


# ── Raw wire frame (protocol) ─────────────────────────────────────────────────

def test_glide_cmd_wire_size_pinned():
    # 8 header + 8 stamp + 3×4 float = 28
    assert p.GLIDE_CMD_MSG_SIZE == 28


def test_glide_cmd_frame_round_trips():
    buf = p.pack_glide_cmd(seq=4, stamp_ns=123_456, v_x=0.5, v_y=-0.2, w_z=0.1)
    assert len(buf) == p.GLIDE_CMD_MSG_SIZE
    seq, stamp, vx, vy, wz = p.unpack_glide_cmd(buf)
    assert seq == 4 and stamp == 123_456
    for a, b in zip((vx, vy, wz), (0.5, -0.2, 0.1)):
        assert abs(a - b) < 1e-6


def test_glide_cmd_frame_is_tagged_glide_cmd():
    buf = p.pack_glide_cmd(seq=1, stamp_ns=0, v_x=0.0, v_y=0.0, w_z=0.0)
    msg_type, _version, _payload_len, _seq = p.unpack_header(buf)
    assert msg_type is p.MsgType.GLIDE_CMD


# ── Contract ↔ wire (codec) ───────────────────────────────────────────────────

def test_codec_glide_cmd_round_trips_through_wire():
    cmd = GlideCmd(stamp_ns=987, v_x=0.4, v_y=0.05, w_z=-0.3)
    got = codec.decode_glide_cmd(codec.encode_glide_cmd(cmd, seq=2))
    assert isinstance(got, GlideCmd)
    assert got.stamp_ns == cmd.stamp_ns
    for name in ("v_x", "v_y", "w_z"):
        assert abs(getattr(got, name) - getattr(cmd, name)) < 1e-4


# ── Real flow untouched: adding GLIDE_CMD must not perturb the walk frames ─────

def test_walk_frame_sizes_unchanged():
    assert p.HELLO_MSG_SIZE == 1004
    assert p.CMD_MSG_SIZE == 698
    assert p.STATE_IMU_MSG_SIZE == 428
