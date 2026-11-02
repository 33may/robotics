"""Frame wire + codec round-trip (§6): the dedicated camera-frame channel.

Separate from the fixed-size SEQPACKET control protocol — a 720p frame is ~2.8 MB, so
frames get their own variable-length stream framing (design.md D6). Depth is quantized
to uint16 **millimeters** on the wire (RealSense-native, halves bandwidth); the
round-trip is therefore lossy to ~1 mm by design. Brain-pure.
"""

import sys

import numpy as np
import pytest

from humanoid.logic.oli import CameraFrame, CameraIntrinsics
from humanoid.logic.oli.comm import frame_protocol as fp
from humanoid.logic.oli.comm.codec import decode_camera_frame, encode_camera_frame


def _frame(w: int = 8, h: int = 4, stamp: int = 42, name: str = "chest") -> CameraFrame:
    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    depth = np.linspace(0.1, 3.0, h * w, dtype=np.float32).reshape(h, w)
    intr = CameraIntrinsics(width=w, height=h, fx=5.0, fy=6.0, cx=4.0, cy=2.0)
    return CameraFrame(stamp_ns=stamp, name=name, rgb=rgb, depth=depth, intrinsics=intr)


# ── wire level ────────────────────────────────────────────────────────────────
def test_wire_roundtrip_fields():
    rgb_b = b"\x01\x02\x03" * 4
    depth_b = b"\x10\x00" * 4
    buf = fp.pack_camera_frame(7, 123, "head", 2, 2, 5.0, 6.0, 1.0, 1.0, rgb_b, depth_b)
    seq, stamp, name, w, h, fx, fy, cx, cy, rb, db = fp.unpack_camera_frame(buf)
    assert (seq, stamp, name, w, h) == (7, 123, "head", 2, 2)
    assert (fx, fy, cx, cy) == (5.0, 6.0, 1.0, 1.0)
    assert rb == rgb_b and db == depth_b


def test_wire_rejects_garbage():
    with pytest.raises(Exception):
        fp.unpack_camera_frame(b"not a real frame buffer")


# ── codec level ───────────────────────────────────────────────────────────────
def test_codec_roundtrip_rgb_exact():
    f = _frame()
    g = decode_camera_frame(encode_camera_frame(f))
    np.testing.assert_array_equal(g.rgb, f.rgb)


def test_codec_roundtrip_depth_within_1mm():
    f = _frame()
    g = decode_camera_frame(encode_camera_frame(f))
    np.testing.assert_allclose(g.depth, f.depth, atol=1e-3)  # uint16-mm quantization


def test_codec_roundtrip_metadata():
    f = _frame(stamp=999, name="head")
    g = decode_camera_frame(encode_camera_frame(f))
    assert g.stamp_ns == 999 and g.name == "head"
    assert (g.intrinsics.width, g.intrinsics.height) == (8, 4)
    assert (g.intrinsics.fx, g.intrinsics.fy) == (5.0, 6.0)
    assert (g.intrinsics.cx, g.intrinsics.cy) == (4.0, 2.0)


def test_codec_depth_zero_stays_zero():
    # RealSense convention: 0 = invalid / no return. Must survive the round-trip.
    f = _frame()
    d = np.array(f.depth)
    d[0, 0] = 0.0
    f2 = CameraFrame(stamp_ns=1, name="chest", rgb=f.rgb, depth=d, intrinsics=f.intrinsics)
    g = decode_camera_frame(encode_camera_frame(f2))
    assert g.depth[0, 0] == 0.0


def test_module_is_brain_pure():
    import humanoid.logic.oli.comm.frame_protocol  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules


# ── RGB-only frames (stereo pair, MAY-173 slam-demo-loop 1.1) ────────────────────
# The head stereo cameras have no depth annotator; their frames travel the same
# channel with an EMPTY depth payload (depth_len=0 is explicit in the header).


def test_wire_roundtrip_empty_depth():
    buf = fp.pack_camera_frame(7, 123, "head_left", 2, 2, 5.0, 6.0, 1.0, 1.0,
                               b"\x01\x02\x03" * 4, b"")
    *_, rb, db = fp.unpack_camera_frame(buf)
    assert rb == b"\x01\x02\x03" * 4
    assert db == b""


def test_contract_accepts_none_depth():
    f = _frame()
    g = CameraFrame(stamp_ns=1, name="head_left", rgb=f.rgb, depth=None,
                    intrinsics=f.intrinsics)
    assert g.depth is None


def test_contract_still_rejects_bad_depth_shape():
    f = _frame()
    with pytest.raises(ValueError):
        CameraFrame(stamp_ns=1, name="head_left", rgb=f.rgb,
                    depth=np.zeros((2, 2, 2), dtype=np.float32), intrinsics=f.intrinsics)


def test_codec_roundtrip_rgb_only():
    f = _frame()
    rgb_only = CameraFrame(stamp_ns=7, name="head_right", rgb=f.rgb, depth=None,
                           intrinsics=f.intrinsics)
    g = decode_camera_frame(encode_camera_frame(rgb_only))
    assert g.depth is None
    assert g.name == "head_right"
    np.testing.assert_array_equal(g.rgb, f.rgb)
