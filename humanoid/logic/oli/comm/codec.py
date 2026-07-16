"""codec.py — the pure contract↔wire seam.

Maps the dataclass contracts onto the frozen `protocol` frames and back:

    Observation  ↔  STATE_IMU   (World → Brain)
    PolicyOut    ↔  CMD          (Brain → World)
    GlideCmd     ↔  GLIDE_CMD    (Brain → World, glide mode — MAY-172)

Both the brain client and the World server use these; keeping the mapping in one
pure module (imports only `protocol` + `contracts` + numpy — no isaacsim/limxsdk,
no sockets) makes it unit-testable without a connection and keeps each side from
re-deriving the field order. `seq` is a wire-level detail supplied by the
transport; the contracts do not carry it (latest-wins uses the stamp, not seq).
"""

from __future__ import annotations

import numpy as np

from ..contracts import NUM_JOINTS, CameraFrame, CameraIntrinsics, Observation, PolicyOut
from ..glide import GlideCmd
from . import frame_protocol as fp
from . import protocol as p

# Sim runs serial PR space, so parallel-mechanism solving is not required; the
# wire carries the flag for the future RealComm edge but SimComm ignores it.
_NO_PARALLEL = [0] * NUM_JOINTS

# Depth is quantized to uint16 MILLIMETERS on the wire: RealSense-native, halves depth
# bandwidth vs float32, and 0 stays "invalid". Round-trip is lossy to ~1 mm by design.
_DEPTH_SCALE_MM = 1000.0


def encode_observation(obs: Observation, seq: int = 0) -> bytes:
    """Observation → a STATE_IMU wire frame."""
    return p.pack_state_imu(
        seq, obs.stamp_ns,
        obs.q, obs.dq, obs.tau, obs.acc, obs.gyro, obs.quat_wxyz,
    )


def decode_observation(buf: bytes) -> Observation:
    """A STATE_IMU wire frame → Observation."""
    _seq, stamp_ns, q, dq, tau, acc, gyro, quat_wxyz = p.unpack_state_imu(buf)
    return Observation(
        stamp_ns=stamp_ns,
        q=q, dq=dq, tau=tau, acc=acc, gyro=gyro, quat_wxyz=quat_wxyz,
    )


def encode_policy_out(po: PolicyOut, seq: int = 0) -> bytes:
    """PolicyOut → a CMD wire frame."""
    return p.pack_cmd(
        seq, po.stamp_ns,
        po.mode, po.q_des, po.dq_des, po.tau_ff, po.kp, po.kd, _NO_PARALLEL,
    )


def decode_policy_out(buf: bytes) -> PolicyOut:
    """A CMD wire frame → PolicyOut (drops the sim-ignored parallel-solve flag)."""
    _seq, stamp_ns, mode, q, dq, tau, kp, kd, _parallel = p.unpack_cmd(buf)
    return PolicyOut(
        stamp_ns=stamp_ns,
        q_des=q, dq_des=dq, tau_ff=tau, kp=kp, kd=kd, mode=mode,
    )


def encode_glide_cmd(cmd: GlideCmd, seq: int = 0) -> bytes:
    """GlideCmd → a GLIDE_CMD wire frame (glide mode — MAY-172)."""
    return p.pack_glide_cmd(seq, cmd.stamp_ns, cmd.v_x, cmd.v_y, cmd.w_z)


def decode_glide_cmd(buf: bytes) -> GlideCmd:
    """A GLIDE_CMD wire frame → GlideCmd."""
    _seq, stamp_ns, v_x, v_y, w_z = p.unpack_glide_cmd(buf)
    return GlideCmd(stamp_ns=stamp_ns, v_x=v_x, v_y=v_y, w_z=w_z)


def encode_camera_frame(frame: CameraFrame, seq: int = 0) -> bytes:
    """CameraFrame → a camera-frame wire buffer (depth → uint16 mm). RGB-only frames
    (depth None — the stereo pair) ship an EMPTY depth payload (depth_len=0)."""
    if frame.depth is None:
        depth_bytes = b""
    else:
        depth_bytes = np.clip(
            frame.depth * _DEPTH_SCALE_MM, 0.0, 65535.0).astype(np.uint16).tobytes()
    i = frame.intrinsics
    return fp.pack_camera_frame(
        seq, frame.stamp_ns, frame.name, i.width, i.height,
        i.fx, i.fy, i.cx, i.cy,
        frame.rgb.tobytes(), depth_bytes,
    )


def decode_camera_frame(buf: bytes) -> CameraFrame:
    """A camera-frame wire buffer → CameraFrame (uint16 mm depth → float32 m; an
    empty depth payload → depth None)."""
    (_seq, stamp_ns, name, w, h, fx, fy, cx, cy,
     rgb_bytes, depth_bytes) = fp.unpack_camera_frame(buf)
    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(h, w, 3)
    depth = None
    if depth_bytes:
        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(h, w).astype(np.float32)
        depth /= _DEPTH_SCALE_MM
    return CameraFrame(
        stamp_ns=stamp_ns, name=name, rgb=rgb, depth=depth,
        intrinsics=CameraIntrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy),
    )
