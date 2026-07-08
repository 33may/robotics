"""
test_protocol_cross_version.py — verify protocol.py is Py 3.8/3.11 byte-identical.

Exercises Spec R3 scenarios:
  - "Cross-version pack identity holds" (sha256 match)
  - "Round-trip is lossless within float32 tolerance"
  - "Protocol version mismatch is rejected cleanly"

Usage — run under both interpreters and diff the sha256 lines:

  /home/may33/miniconda3/envs/limx/bin/python   <this file>  > /tmp/proto.limx.txt
  /home/may33/miniconda3/envs/isaac/bin/python  <this file>  > /tmp/proto.isaac.txt
  diff /tmp/proto.{limx,isaac}.txt              # must be empty
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

BRIDGE_DIR = (
    Path("/home/may33/projects/ml_portfolio/robotics/humanoid/logic/simulation/"
         "isaacsim/bridge")
)
sys.path.insert(0, str(BRIDGE_DIR))

import protocol as p  # noqa: E402


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def canned_hello() -> bytes:
    names = [f"joint_{i:02d}" for i in range(p.NUM_JOINTS)]
    return p.pack_hello(seq=1, dof_names=names)


def canned_cmd() -> bytes:
    mode = [0] * p.NUM_JOINTS
    q = [0.01 * i for i in range(p.NUM_JOINTS)]
    dq = [0.0] * p.NUM_JOINTS
    tau = [0.0] * p.NUM_JOINTS
    kp = [139.41] * 4 + [0.0] * (p.NUM_JOINTS - 4)
    kd = [17.75] * 4 + [0.0] * (p.NUM_JOINTS - 4)
    parallel = [1] * p.NUM_JOINTS
    return p.pack_cmd(
        seq=42,
        stamp_ns=1_750_000_000_000_000_000,  # ns since epoch (mid-2025)
        mode=mode, q=q, dq=dq, tau=tau, kp=kp, kd=kd,
        parallel_solve_required=parallel,
    )


def canned_state_imu() -> bytes:
    q = [0.01 * i for i in range(p.NUM_JOINTS)]
    dq = [0.0] * p.NUM_JOINTS
    tau = [0.0] * p.NUM_JOINTS
    acc = [0.0, 0.0, 9.81]                     # gravity, body-up
    gyro = [0.0, 0.0, 0.0]                     # at rest
    quat_wxyz = [1.0, 0.0, 0.0, 0.0]           # identity rotation
    return p.pack_state_imu(
        seq=4242,
        stamp_ns=1_750_000_000_001_234_567,
        q=q, dq=dq, tau=tau, acc=acc, gyro=gyro, quat_wxyz=quat_wxyz,
    )


def print_header() -> None:
    print(f"# python {sys.version.split()[0]} on {sys.platform}")
    print(f"# protocol.PROTOCOL_VERSION = {p.PROTOCOL_VERSION}")
    print(f"# protocol.NUM_JOINTS = {p.NUM_JOINTS}")


def report_sizes() -> None:
    print(f"sizes  HEADER={p.HEADER_SIZE}  "
          f"HELLO={p.HELLO_MSG_SIZE}  "
          f"CMD={p.CMD_MSG_SIZE}  "
          f"STATE_IMU={p.STATE_IMU_MSG_SIZE}")


def report_packs() -> None:
    h = canned_hello()
    c = canned_cmd()
    s = canned_state_imu()
    print(f"hello      len={len(h):4d}  sha256_16={_h(h)}")
    print(f"cmd        len={len(c):4d}  sha256_16={_h(c)}")
    print(f"state_imu  len={len(s):4d}  sha256_16={_h(s)}")


def assert_roundtrip_hello() -> None:
    names_in = [f"joint_{i:02d}" for i in range(p.NUM_JOINTS)]
    buf = p.pack_hello(seq=99, dof_names=names_in)
    seq, dof_count, names_out = p.unpack_hello(buf)
    assert seq == 99, (seq, 99)
    assert dof_count == p.NUM_JOINTS, (dof_count, p.NUM_JOINTS)
    assert names_out == names_in, (names_out, names_in)


def assert_roundtrip_cmd() -> None:
    mode = list(range(p.NUM_JOINTS))           # 0..30 — exercise full uint8 range
    mode = [m & 0xFF for m in mode]
    q = [0.0001 * i for i in range(p.NUM_JOINTS)]
    dq = [-0.0001 * i for i in range(p.NUM_JOINTS)]
    tau = [0.5 * i for i in range(p.NUM_JOINTS)]
    kp = [100.0 + i for i in range(p.NUM_JOINTS)]
    kd = [10.0 + 0.1 * i for i in range(p.NUM_JOINTS)]
    parallel = [i % 2 for i in range(p.NUM_JOINTS)]

    buf = p.pack_cmd(
        seq=12345,
        stamp_ns=9_223_372_036_854_775_807,    # max int64-ish (uint64 large)
        mode=mode, q=q, dq=dq, tau=tau, kp=kp, kd=kd,
        parallel_solve_required=parallel,
    )
    s, st, m_o, q_o, dq_o, tau_o, kp_o, kd_o, par_o = p.unpack_cmd(buf)
    assert s == 12345, (s, 12345)
    assert st == 9_223_372_036_854_775_807, (st,)
    assert m_o == mode, (m_o, mode)
    assert par_o == parallel, (par_o, parallel)
    eps = 1e-6  # float32 round-trip tolerance
    for a, b in [(q_o, q), (dq_o, dq), (tau_o, tau), (kp_o, kp), (kd_o, kd)]:
        for x, y in zip(a, b):
            assert math.isclose(x, y, rel_tol=eps, abs_tol=eps), (x, y)


def assert_roundtrip_state_imu() -> None:
    q = [0.001 * i for i in range(p.NUM_JOINTS)]
    dq = [0.01 * i for i in range(p.NUM_JOINTS)]
    tau = [-0.5 * i for i in range(p.NUM_JOINTS)]
    acc = [0.1, -0.2, 9.81]
    gyro = [0.01, -0.02, 0.03]
    quat_wxyz = [0.999, 0.01, -0.02, 0.003]

    buf = p.pack_state_imu(
        seq=4294967295,                        # max uint32
        stamp_ns=12345678901234567,
        q=q, dq=dq, tau=tau, acc=acc, gyro=gyro, quat_wxyz=quat_wxyz,
    )
    s, st, q_o, dq_o, tau_o, acc_o, gyro_o, quat_o = p.unpack_state_imu(buf)
    assert s == 4294967295, (s,)
    assert st == 12345678901234567, (st,)
    eps = 1e-6
    for a, b in [(q_o, q), (dq_o, dq), (tau_o, tau),
                  (acc_o, acc), (gyro_o, gyro), (quat_o, quat_wxyz)]:
        for x, y in zip(a, b):
            assert math.isclose(x, y, rel_tol=eps, abs_tol=eps), (x, y)


def assert_version_mismatch_rejected() -> None:
    # Hand-craft a header with version=1 instead of 0
    import struct as _s
    bogus_type = (1 << 14) | int(p.MsgType.HELLO)
    bad_header = _s.pack(p.HEADER_FMT, bogus_type, p.HELLO_PAYLOAD_SIZE, 0)
    # Need full-size buf so length checks don't trip first
    bad_buf = bad_header + b"\x00" * p.HELLO_PAYLOAD_SIZE
    try:
        p.unpack_hello(bad_buf)
    except p.ProtocolError as e:
        assert "version mismatch" in str(e), str(e)
        return
    raise AssertionError("expected ProtocolError on version mismatch")


def assert_unknown_type_rejected() -> None:
    import struct as _s
    bogus_type = (0 << 14) | 0x3FFF  # max type_code, no MsgType defined
    bad_header = _s.pack(p.HEADER_FMT, bogus_type, 0, 0)
    try:
        p.unpack_header(bad_header)
    except p.ProtocolError as e:
        assert "unknown msg type" in str(e), str(e)
        return
    raise AssertionError("expected ProtocolError on unknown type_code")


def main() -> None:
    print_header()
    report_sizes()
    report_packs()
    # Round-trips
    assert_roundtrip_hello()
    assert_roundtrip_cmd()
    assert_roundtrip_state_imu()
    # Negative paths
    assert_version_mismatch_rejected()
    assert_unknown_type_rejected()
    print("roundtrip      OK  HELLO/CMD/STATE_IMU")
    print("version_check  OK")
    print("type_check     OK")


if __name__ == "__main__":
    main()
