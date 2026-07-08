"""Regression lock for the Communication wire (humanoid.logic.oli.comm.protocol).

The wire is moved verbatim from the archived bridge change (D11) and frozen — these
tests pin its sizes, round-trip fidelity, and guard rails so a future edit that
breaks 3.8∩3.11 byte-compat fails loudly. Pure stdlib: runs in the `brain` env.
"""

import struct

import pytest

from humanoid.logic.oli.comm import protocol as p

pytestmark = pytest.mark.brain

N = p.NUM_JOINTS


# ── Fixed wire sizes (3.8∩3.11 byte-compat depends on these) ──────────────────

def test_message_sizes_are_pinned():
    assert p.HEADER_SIZE == 8
    assert p.HELLO_MSG_SIZE == 1004
    assert p.CMD_MSG_SIZE == 698
    assert p.STATE_IMU_MSG_SIZE == 428


# ── Round trips ───────────────────────────────────────────────────────────────

def test_hello_round_trip():
    names = [f"joint_{i}" for i in range(N)]
    seq, count, got = p.unpack_hello(p.pack_hello(seq=7, dof_names=names))
    assert seq == 7 and count == N and got == names


def test_cmd_round_trip():
    mode = list(range(N))
    q = [0.1 * i for i in range(N)]
    dq = [-0.2 * i for i in range(N)]
    tau = [0.3 * i for i in range(N)]
    kp = [100.0 + i for i in range(N)]
    kd = [1.0 + i for i in range(N)]
    par = [i % 2 for i in range(N)]
    buf = p.pack_cmd(3, 123456789, mode, q, dq, tau, kp, kd, par)
    seq, stamp, m2, q2, dq2, tau2, kp2, kd2, par2 = p.unpack_cmd(buf)
    assert seq == 3 and stamp == 123456789
    assert m2 == mode and par2 == par
    for a, b in zip(q + dq + tau + kp + kd, q2 + dq2 + tau2 + kp2 + kd2):
        assert abs(a - b) < 1e-4  # float32 round-trip tolerance


def test_state_imu_round_trip():
    q = [0.01 * i for i in range(N)]
    dq = [0.02 * i for i in range(N)]
    tau = [0.03 * i for i in range(N)]
    acc = [0.0, 0.0, -9.81]
    gyro = [0.1, 0.2, 0.3]
    quat = [1.0, 0.0, 0.0, 0.0]
    buf = p.pack_state_imu(9, 42, q, dq, tau, acc, gyro, quat)
    seq, stamp, q2, dq2, tau2, acc2, gyro2, quat2 = p.unpack_state_imu(buf)
    assert seq == 9 and stamp == 42
    for a, b in zip(q + dq + tau + acc + gyro + quat,
                    q2 + dq2 + tau2 + acc2 + gyro2 + quat2):
        assert abs(a - b) < 1e-4


# ── Guard rails ───────────────────────────────────────────────────────────────

def test_version_mismatch_raises():
    buf = bytearray(p.pack_hello(0, [f"j{i}" for i in range(N)]))
    # Stomp the version bits (14-15) to a non-zero version.
    type_with_version, payload_len, seq = struct.unpack_from(p.HEADER_FMT, buf, 0)
    struct.pack_into(p.HEADER_FMT, buf, 0,
                     type_with_version | (1 << 14), payload_len, seq)
    with pytest.raises(p.ProtocolError):
        p.unpack_header(bytes(buf))


def test_unknown_type_code_raises():
    bad = struct.pack(p.HEADER_FMT, 0x0FFF, 0, 0)  # type_code 4095, version 0
    with pytest.raises(p.ProtocolError):
        p.unpack_header(bad)


def test_wrong_length_field_raises():
    with pytest.raises(p.ProtocolError):
        p.pack_state_imu(0, 0, [0.0] * (N - 1), [0.0] * N, [0.0] * N,
                         [0.0] * 3, [0.0] * 3, [1.0, 0.0, 0.0, 0.0])
