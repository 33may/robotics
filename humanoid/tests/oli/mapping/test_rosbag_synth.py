"""Tests for the dump → ROS 2 bag synthesizer (MAY-173 locdev T3).

Offline tooling: converts a coverage-drive dump (rig.json + frames/ + poses.jsonl,
written by DriveRecorder) into a ROS 2 bag consumable by NVIDIA's
`create_map_offline.py` (cuVGL map bake). Pure-Python `rosbags` — no ROS install;
runs in the `hum` env. Deliberately NOT part of the `brain` marker: this is
tooling, not invariant brain code.

The #1 gotcha under test: dump poses are USD camera convention (−Z view, +Y up)
and must land in the bag as ROS optical (+Z view, +X right, +Y down).
"""

import math

import numpy as np
import pytest
from PIL import Image as PILImage

pytest.importorskip("rosbags")  # tooling dep (hum env) — skip cleanly elsewhere

from pathlib import Path  # noqa: E402

from rosbags.highlevel import AnyReader  # noqa: E402

from humanoid.logic.simulation.mapping.recorder import DriveRecorder  # noqa: E402
from humanoid.logic.simulation.mapping.rosbag_synth import (  # noqa: E402
    BagSpec,
    base_pose_matrix,
    main,
    make_camera_info,
    quat_from_matrix,
    recover_static_mount,
    synthesize,
    usd_cam_to_optical,
)


# ─── frame conversion: USD camera → ROS optical ────────────────────────────────

def test_usd_to_optical_identity_pose():
    """USD identity camera (view −Z, up +Y) → optical axes flip Y and Z."""
    T_usd = np.eye(4)
    T_opt = usd_cam_to_optical(T_usd)
    # optical z (view dir) must be world −Z: same physical view direction
    assert np.allclose(T_opt[:3, 2], [0, 0, -1])
    # optical y (image down) must be world −Y (USD up was +Y)
    assert np.allclose(T_opt[:3, 1], [0, -1, 0])
    # optical x (image right) unchanged
    assert np.allclose(T_opt[:3, 0], [1, 0, 0])
    # translation untouched
    assert np.allclose(T_opt[:3, 3], 0)


def test_usd_to_optical_hand_worked_looking_along_plus_x():
    """Hand-worked: USD camera looking along world +X with up +Z.

    USD columns: x_cam=(0,−1,0), y_cam=(0,0,1), z_cam=(−1,0,0)  (view −z_cam=+X).
    Optical must view along +X: z_opt=(1,0,0); down = −Z: y_opt=(0,0,−1);
    right when facing +X with up +Z is −Y: x_opt=(0,−1,0).
    """
    T_usd = np.eye(4)
    T_usd[:3, 0] = [0, -1, 0]
    T_usd[:3, 1] = [0, 0, 1]
    T_usd[:3, 2] = [-1, 0, 0]
    T_usd[:3, 3] = [1.0, 2.0, 3.0]
    T_opt = usd_cam_to_optical(T_usd)
    assert np.allclose(T_opt[:3, 2], [1, 0, 0])   # view +X
    assert np.allclose(T_opt[:3, 1], [0, 0, -1])  # down −Z
    assert np.allclose(T_opt[:3, 0], [0, -1, 0])  # right −Y
    assert np.allclose(T_opt[:3, 3], [1.0, 2.0, 3.0])
    # still a rotation matrix
    R = T_opt[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


# ─── base pose rows → T_map_base ───────────────────────────────────────────────

def test_base_pose_matrix_yaw_quarter_turn():
    T = base_pose_matrix(x=1.0, y=2.0, yaw=math.pi / 2)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 0.0])
    # base +X maps to world +Y under yaw 90°
    assert np.allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(T[3], [0, 0, 0, 1])


# ─── quaternion helper ────────────────────────────────────────────────────────

def test_quat_from_matrix_identity_and_yaw90():
    assert np.allclose(quat_from_matrix(np.eye(3)), [0, 0, 0, 1])
    Rz = base_pose_matrix(0, 0, math.pi / 2)[:3, :3]
    q = quat_from_matrix(Rz)  # (x, y, z, w)
    s = math.sqrt(0.5)
    assert np.allclose(q, [0, 0, s, s], atol=1e-12)


# ─── CameraInfo from rig intrinsics ───────────────────────────────────────────

INTR = {"width": 12, "height": 8, "fx": 10.0, "fy": 10.0, "cx": 6.0, "cy": 4.0}


def test_camera_info_left_monocular_projection():
    ci = make_camera_info(INTR, frame_id="head_left_optical_frame", stamp_ns=7)
    assert ci.width == 12 and ci.height == 8
    assert ci.header.frame_id == "head_left_optical_frame"
    K = np.asarray(ci.k).reshape(3, 3)
    assert K[0, 0] == 10.0 and K[0, 2] == 6.0 and K[1, 2] == 4.0
    P = np.asarray(ci.p).reshape(3, 4)
    assert P[0, 3] == 0.0  # left camera: Tx = 0
    assert ci.distortion_model == "plumb_bob"
    assert np.allclose(ci.d, 0)
    assert np.allclose(np.asarray(ci.r).reshape(3, 3), np.eye(3))


def test_camera_info_right_encodes_stereo_baseline():
    ci = make_camera_info(
        INTR, frame_id="head_right_optical_frame", stamp_ns=7, baseline_m=0.05
    )
    P = np.asarray(ci.p).reshape(3, 4)
    assert np.isclose(P[0, 3], -10.0 * 0.05)  # Tx = −fx·baseline


# ─── static mount recovery from rendered poses ────────────────────────────────

def _mount_usd(y_off: float) -> np.ndarray:
    """Fixed base→cam(USD) mount: forward-looking (hand-worked case), offset y."""
    T = np.eye(4)
    T[:3, 0] = [0, -1, 0]
    T[:3, 1] = [0, 0, 1]
    T[:3, 2] = [-1, 0, 0]
    T[:3, 3] = [0.06, y_off, 0.65]
    return T


def test_recover_static_mount_returns_constant_base_to_optical():
    mount = _mount_usd(0.025)
    pairs = []
    for i in range(4):
        T_mb = base_pose_matrix(x=0.1 * i, y=0.02 * i, yaw=0.1 * i)
        pairs.append((T_mb, T_mb @ mount))
    T_base_opt = recover_static_mount(pairs)
    assert np.allclose(T_base_opt, usd_cam_to_optical(mount), atol=1e-9)


def test_recover_static_mount_rejects_inconsistent_mounts():
    mount = _mount_usd(0.025)
    wobble = mount.copy()
    wobble[:3, 3] += [0.0, 0.0, 0.05]  # 5 cm jump — not a fixed mount
    pairs = [
        (base_pose_matrix(0, 0, 0), mount),
        (base_pose_matrix(0.1, 0, 0), base_pose_matrix(0.1, 0, 0) @ wobble),
    ]
    with pytest.raises(ValueError):
        recover_static_mount(pairs)


# ─── end-to-end: tiny dump → bag ──────────────────────────────────────────────

RIG = {
    "camera_axes": "usd (-Z view, +Y up); convert to ROS optical (+Z view) at bag synth",
    "baseline_m": 0.05,
    "stereo_pair": ["head_left", "head_right"],
    "cameras": {
        "head_left": {"intrinsics": dict(INTR)},
        "head_right": {"intrinsics": dict(INTR)},
    },
}

N_STAMPS = 4
STAMP0 = 349_999_992
DT = 200_000_000


def _write_tiny_dump(root) -> "Path":
    """Real-writer fixture: use DriveRecorder so the dump layout is authoritative."""
    rec = DriveRecorder(root / "drive", RIG)
    rng = np.random.default_rng(33)
    mounts = {"head_left": _mount_usd(0.025), "head_right": _mount_usd(-0.025)}
    for i in range(N_STAMPS):
        stamp = STAMP0 + i * DT
        T_mb = base_pose_matrix(x=0.1 * i, y=0.0, yaw=0.0)
        rec.add_base_pose(stamp, x=0.1 * i, y=0.0, yaw=0.0)
        for cam, mnt in mounts.items():
            rgb = rng.integers(0, 255, size=(8, 12, 3), dtype=np.uint8)
            rec.add_frame(cam, stamp, rgb, T_mb @ mnt)
    rec.close()
    return root / "drive"


@pytest.fixture()
def tiny_dump(tmp_path):
    return _write_tiny_dump(tmp_path)


def _read_bag(bag_path):
    """Return {topic: [(t_ns, msg), …]} for the whole bag."""
    out = {}
    with AnyReader([Path(bag_path)]) as reader:
        for conn, t, raw in reader.messages():
            out.setdefault(conn.topic, []).append(
                (t, reader.deserialize(raw, conn.msgtype))
            )
    return out


def test_synthesize_bag_topics_counts_and_stamps(tiny_dump, tmp_path):
    spec = BagSpec()
    stats = synthesize(tiny_dump, tmp_path / "bag", spec)
    msgs = _read_bag(tmp_path / "bag")

    assert set(msgs) == {
        "/left/image_raw", "/left/camera_info",
        "/right/image_raw", "/right/camera_info",
        "/tf", "/tf_static", "/odom",
    }
    for topic in ("/left/image_raw", "/left/camera_info",
                  "/right/image_raw", "/right/camera_info", "/tf", "/odom"):
        assert len(msgs[topic]) == N_STAMPS, topic
        times = [t for t, _ in msgs[topic]]
        assert times == sorted(times) and len(set(times)) == N_STAMPS
    assert len(msgs["/tf_static"]) == 1
    assert stats["stamps"] == N_STAMPS


def test_synthesize_image_content_and_encoding(tiny_dump, tmp_path):
    synthesize(tiny_dump, tmp_path / "bag", BagSpec())
    msgs = _read_bag(tmp_path / "bag")
    _, img0 = msgs["/left/image_raw"][0]
    assert img0.encoding == "rgb8"
    assert img0.height == 8 and img0.width == 12 and img0.step == 36
    assert img0.header.frame_id == "head_left_optical_frame"
    assert img0.header.stamp.sec == STAMP0 // 1_000_000_000
    assert img0.header.stamp.nanosec == STAMP0 % 1_000_000_000
    # pixel-exact vs the PNG the recorder wrote
    png = np.asarray(
        PILImage.open(tiny_dump / "frames/head_left" / f"{STAMP0:019d}.png")
    )
    assert np.array_equal(
        np.frombuffer(img0.data, dtype=np.uint8).reshape(8, 12, 3), png
    )


def test_synthesize_tf_and_odom_geometry(tiny_dump, tmp_path):
    synthesize(tiny_dump, tmp_path / "bag", BagSpec())
    msgs = _read_bag(tmp_path / "bag")

    # dynamic tf: map→base_link tracks the base rows
    _, tf1 = msgs["/tf"][1]
    tr = tf1.transforms[0]
    assert tr.header.frame_id == "map" and tr.child_frame_id == "base_link"
    assert np.isclose(tr.transform.translation.x, 0.1)

    # static tf: both optical mounts, base_link parent, optical orientation
    _, tfs = msgs["/tf_static"][0]
    children = sorted(t.child_frame_id for t in tfs.transforms)
    assert children == ["head_left_optical_frame", "head_right_optical_frame"]
    left = next(t for t in tfs.transforms
                if t.child_frame_id == "head_left_optical_frame")
    assert left.header.frame_id == "base_link"
    assert np.isclose(left.transform.translation.y, 0.025)
    q = left.transform.rotation
    R_expect = usd_cam_to_optical(_mount_usd(0.025))[:3, :3]
    assert np.allclose([q.x, q.y, q.z, q.w], quat_from_matrix(R_expect), atol=1e-9)

    # odom mirrors base rows
    _, od1 = msgs["/odom"][1]
    assert od1.header.frame_id == "map" and od1.child_frame_id == "base_link"
    assert np.isclose(od1.pose.pose.position.x, 0.1)


def test_synthesize_every_n_subsamples(tiny_dump, tmp_path):
    stats = synthesize(tiny_dump, tmp_path / "bag", BagSpec(every_n=2))
    msgs = _read_bag(tmp_path / "bag")
    assert len(msgs["/left/image_raw"]) == 2
    assert stats["stamps"] == 2


def test_cli_main_writes_bag(tiny_dump, tmp_path):
    out = tmp_path / "cli_bag"
    rc = main(["--dump", str(tiny_dump), "--out", str(out), "--storage", "sqlite3"])
    assert rc == 0
    assert (out / "metadata.yaml").exists()
