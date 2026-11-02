"""rosbag_synth.py — coverage-drive dump → ROS 2 bag for the cuVGL map bake (MAY-173 locdev T3).

Offline tooling (NOT brain code): converts the neutral on-disk dump written by
`recorder.DriveRecorder` (rig.json + frames/<cam>/<stamp>.png + poses.jsonl) into
a ROS 2 bag consumable by NVIDIA's `create_map_offline.py` /
`rosbag_to_mapping_data` (ros-jazzy-isaac-mapping-ros). Pure-Python `rosbags`
library — no ROS installation; runs host-side in the `hum` env (py3.12).

Bag layout (all topic/frame names parameterized via BagSpec):

    /left/image_raw     sensor_msgs/Image  rgb8, stereo left  (rig stereo_pair[0])
    /left/camera_info   sensor_msgs/CameraInfo   P = [fx 0 cx 0 | 0 fy cy 0 | 0 0 1 0]
    /right/image_raw    sensor_msgs/Image  rgb8, stereo right (rig stereo_pair[1])
    /right/camera_info  sensor_msgs/CameraInfo   P[0,3] = Tx = −fx·baseline
    /tf                 tf2_msgs/TFMessage  map→base_link per stamp (base rows)
    /tf_static          tf2_msgs/TFMessage  base_link→<cam>_optical_frame (once)
    /odom               nav_msgs/Odometry   map→base_link (pose side-channel for
                        rosbag_to_mapping_data --pose_topic_name)

Frame convention (the #1 gotcha): dump poses are USD camera axes (−Z view, +Y up,
declared in rig.json `camera_axes`); everything written to the bag is ROS optical
(+Z view, +X right, +Y down) via a RIGHT-multiplied axis flip — re-labels the
camera's local axes without moving it in the world.

The static mounts are RECOVERED from the rendered per-stamp `T_world_cam` against
`map→base` (not read from rig mounts) and asserted constant — truthful even if a
later capture perturbs camera prims, and a loud failure instead of a wrong bag.

Verified against isaac_mapping_ros docs: topic names are configurable at bake
time via `--camera_topic_config` YAML; L/R sync threshold 50 µs (our pairs share
identical stamps); pose topics may be PoseStamped/Odometry/Path with child frame
`base_link`. Assumptions logged in MAY-173 T3 report: raw rgb8 accepted (H264
only documented for NVIDIA's own recordings), sim-time stamps acceptable offline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage

from rosbags.rosbag2 import Writer
from rosbags.rosbag2.enums import StoragePlugin
from rosbags.typesys import Stores, get_typestore

# ─── geometry ─────────────────────────────────────────────────────────────────

#: USD camera (−Z view, +Y up) → ROS optical (+Z view, +Y down): flip local Y & Z.
_USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])


def usd_cam_to_optical(T_world_cam_usd: np.ndarray) -> np.ndarray:
    """Re-express a USD-convention camera pose as a ROS-optical-convention pose."""
    return np.asarray(T_world_cam_usd, dtype=float) @ _USD_TO_OPTICAL


def base_pose_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    """T_map_base from a planar base row (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3] = x, y
    return T


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (x, y, z, w), Shepperd's method."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s,
                         (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                     0.25 * s, (R[1, 0] - R[0, 1]) / s])


# Tolerances sized from the v1 drive: steady-state mount noise ≤0.2 mm / 0.09°,
# settle transient 51 mm / 1.3° — these sit ~10× above noise, ~10× below transient.
_TOL_TRANS_M = 5e-3
_TOL_ROT_DEG = 0.3


def recover_static_mount(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    tol_trans_m: float = _TOL_TRANS_M,
    tol_rot_deg: float = _TOL_ROT_DEG,
) -> np.ndarray:
    """Recover the fixed base_link→optical mount from rendered world poses.

    ``pairs`` = [(T_map_base, T_world_cam_USD), …] per stamp. Raises ValueError
    if the recovered mount is not constant within tolerance (i.e. the camera is
    NOT rigidly mounted over this capture — do not write a static tf then).
    """
    if not pairs:
        raise ValueError("recover_static_mount: no pose pairs")
    mounts = [np.linalg.inv(T_mb) @ usd_cam_to_optical(T_wc) for T_mb, T_wc in pairs]
    # reference = median-nearest mount (same rule as _drop_transient_samples;
    # a first-sample reference would double the effective tolerance band)
    t = np.stack([m[:3, 3] for m in mounts])
    ref = mounts[int(np.argmin(np.linalg.norm(t - np.median(t, axis=0), axis=1)))]
    for i, m in enumerate(mounts):
        dt = np.linalg.norm(m[:3, 3] - ref[:3, 3])
        cos_ang = (np.trace(ref[:3, :3].T @ m[:3, :3]) - 1.0) / 2.0
        ang_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_ang))))
        if dt > tol_trans_m or ang_deg > tol_rot_deg:
            raise ValueError(
                f"mount not static at sample {i}: Δt={dt:.4f} m, Δrot={ang_deg:.3f}°"
            )
    return ref


# ─── bag spec ─────────────────────────────────────────────────────────────────

@dataclass
class BagSpec:
    """All names the bake tool might disagree with live here, not in code."""

    left_image_topic: str = "/left/image_raw"
    left_info_topic: str = "/left/camera_info"
    right_image_topic: str = "/right/image_raw"
    right_info_topic: str = "/right/camera_info"
    tf_topic: str = "/tf"
    tf_static_topic: str = "/tf_static"
    odom_topic: str = "/odom"
    map_frame: str = "map"
    base_frame: str = "base_link"
    optical_frame_suffix: str = "_optical_frame"
    storage: str = "mcap"  # mcap | sqlite3
    every_n: int = 1
    max_stamps: int | None = None
    #: IMU stream (imu.jsonl, written by DriveRecorder.add_imu). Topic/frame match
    #: isaac_mapping_ros vslam.launch.py defaults ('visual_slam/imu' remap +
    #: imu_frame param). NEVER subsampled by every_n — VIO wants the full rate.
    imu_topic: str = "/front_stereo_imu/imu"
    imu_frame: str = "front_stereo_camera_imu"
    include_imu: bool = True
    #: drop everything before dump_start + skip_seconds — the cold-start frames
    #: are UNCONSTRAINED in cuVSLAM's pose graph (no loop closure repairs them),
    #: so they must never reach the bake. Mid-drive losses are NOT trimmable
    #: here; those are the loop-closure/optimizer's job (and the T6 detector's).
    skip_seconds: float = 0.0


# ─── message assembly (rosbags typestore) ─────────────────────────────────────

_TS = get_typestore(Stores.ROS2_JAZZY)
_T = _TS.types

_MSG_IMAGE = "sensor_msgs/msg/Image"
_MSG_CAMINFO = "sensor_msgs/msg/CameraInfo"
_MSG_TF = "tf2_msgs/msg/TFMessage"
_MSG_ODOM = "nav_msgs/msg/Odometry"
_MSG_IMU = "sensor_msgs/msg/Imu"


def _stamp(stamp_ns: int):
    return _T["builtin_interfaces/msg/Time"](
        sec=int(stamp_ns) // 1_000_000_000, nanosec=int(stamp_ns) % 1_000_000_000
    )


def _header(stamp_ns: int, frame_id: str):
    return _T["std_msgs/msg/Header"](stamp=_stamp(stamp_ns), frame_id=frame_id)


def make_camera_info(
    intr: Dict[str, Any], *, frame_id: str, stamp_ns: int, baseline_m: float = 0.0
):
    """sensor_msgs/CameraInfo from rig intrinsics; right cam gets Tx = −fx·baseline."""
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float64)
    p = np.array([fx, 0, cx, -fx * baseline_m, 0, fy, cy, 0, 0, 0, 1, 0],
                 dtype=np.float64)
    return _T[_MSG_CAMINFO](
        header=_header(stamp_ns, frame_id),
        height=int(intr["height"]),
        width=int(intr["width"]),
        distortion_model="plumb_bob",
        d=np.zeros(5, dtype=np.float64),
        k=k,
        r=np.eye(3, dtype=np.float64).reshape(-1),
        p=p,
        binning_x=0,
        binning_y=0,
        roi=_T["sensor_msgs/msg/RegionOfInterest"](
            x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
        ),
    )


def _image_msg(png_path: Path, stamp_ns: int, frame_id: str):
    arr = np.asarray(PILImage.open(png_path), dtype=np.uint8)
    h, w = arr.shape[:2]
    return _T[_MSG_IMAGE](
        header=_header(stamp_ns, frame_id),
        height=h,
        width=w,
        encoding="rgb8",
        is_bigendian=0,
        step=3 * w,
        data=arr.reshape(-1),
    )


def _transform_stamped(T: np.ndarray, stamp_ns: int, parent: str, child: str):
    q = quat_from_matrix(T[:3, :3])
    return _T["geometry_msgs/msg/TransformStamped"](
        header=_header(stamp_ns, parent),
        child_frame_id=child,
        transform=_T["geometry_msgs/msg/Transform"](
            translation=_T["geometry_msgs/msg/Vector3"](
                x=T[0, 3], y=T[1, 3], z=T[2, 3]
            ),
            rotation=_T["geometry_msgs/msg/Quaternion"](
                x=q[0], y=q[1], z=q[2], w=q[3]
            ),
        ),
    )


def _odom_msg(T: np.ndarray, stamp_ns: int, map_frame: str, base_frame: str):
    q = quat_from_matrix(T[:3, :3])
    zero3 = _T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0)
    return _T[_MSG_ODOM](
        header=_header(stamp_ns, map_frame),
        child_frame_id=base_frame,
        pose=_T["geometry_msgs/msg/PoseWithCovariance"](
            pose=_T["geometry_msgs/msg/Pose"](
                position=_T["geometry_msgs/msg/Point"](
                    x=T[0, 3], y=T[1, 3], z=T[2, 3]
                ),
                orientation=_T["geometry_msgs/msg/Quaternion"](
                    x=q[0], y=q[1], z=q[2], w=q[3]
                ),
            ),
            covariance=np.zeros(36, dtype=np.float64),
        ),
        twist=_T["geometry_msgs/msg/TwistWithCovariance"](
            twist=_T["geometry_msgs/msg/Twist"](linear=zero3, angular=zero3),
            covariance=np.zeros(36, dtype=np.float64),
        ),
    )


def _imu_msg(row: dict, frame_id: str):
    """sensor_msgs/Imu from one imu.jsonl row. Orientation is deliberately
    unpopulated (covariance[0] = -1, the ROS 'no estimate' convention) —
    cuVSLAM consumes only angular_velocity + linear_acceleration."""
    no_cov = np.zeros(9, dtype=np.float64)
    ori_cov = no_cov.copy()
    ori_cov[0] = -1.0
    return _T[_MSG_IMU](
        header=_header(row["stamp_ns"], frame_id),
        orientation=_T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=ori_cov,
        angular_velocity=_T["geometry_msgs/msg/Vector3"](
            x=row["gyro"][0], y=row["gyro"][1], z=row["gyro"][2]),
        angular_velocity_covariance=no_cov,
        linear_acceleration=_T["geometry_msgs/msg/Vector3"](
            x=row["acc"][0], y=row["acc"][1], z=row["acc"][2]),
        linear_acceleration_covariance=no_cov,
    )


# ─── dump loading ─────────────────────────────────────────────────────────────

def _load_samples(dump_dir: Path, spec: BagSpec) -> Tuple[dict, List[dict]]:
    """rig dict + one record per stereo stamp: {stamp, base(T), left(row), right(row)}."""
    rig = json.loads((dump_dir / "rig.json").read_text())
    left_cam, right_cam = rig["stereo_pair"]
    rows: Dict[str, Dict[int, dict]] = {left_cam: {}, right_cam: {}, "base": {}}
    with open(dump_dir / "poses.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["cam"] in rows:
                rows[row["cam"]][row["stamp_ns"]] = row

    stamps = sorted(
        set(rows[left_cam]) & set(rows[right_cam]) & set(rows["base"])
    )
    dropped = len(set(rows[left_cam]) | set(rows[right_cam])) - len(stamps)
    if spec.skip_seconds > 0 and stamps:
        cutoff = stamps[0] + int(spec.skip_seconds * 1e9)
        stamps = [s for s in stamps if s >= cutoff]
    stamps = stamps[:: max(1, spec.every_n)]
    if spec.max_stamps is not None:
        stamps = stamps[: spec.max_stamps]

    samples = [
        {
            "stamp": s,
            "base": base_pose_matrix(
                rows["base"][s]["x"], rows["base"][s]["y"], rows["base"][s]["yaw"]
            ),
            "left": rows[left_cam][s],
            "right": rows[right_cam][s],
        }
        for s in stamps
    ]
    if dropped:
        print(f"[rosbag_synth] WARNING: {dropped} stamps lacked a full L/R/base triple")
    return rig, samples


def _load_imu(dump_dir: Path, spec: BagSpec, min_stamp_ns: int) -> List[dict]:
    """imu.jsonl rows at/after the first kept camera stamp (ascending). Empty
    list when the dump has no IMU stream or include_imu is off."""
    path = dump_dir / "imu.jsonl"
    if not spec.include_imu or not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows = [r for r in rows if r["stamp_ns"] >= min_stamp_ns]
    rows.sort(key=lambda r: r["stamp_ns"])
    return rows


def _drop_transient_samples(
    samples: List[dict],
    sides: Sequence[str] = ("left", "right"),
    *,
    tol_trans_m: float = _TOL_TRANS_M,
    tol_rot_deg: float = _TOL_ROT_DEG,
) -> Tuple[List[dict], int]:
    """Drop samples whose instantaneous mount deviates from the robust reference.

    Real drives leak a few pre-settle frames at the start (v1: samples 0–1,
    ~5 cm high): those contradict the static tf and must not enter the bag.
    Reference = the sample whose mount translation is nearest the elementwise
    median (immune to leading transients). A sample is dropped if EITHER stereo
    side deviates.
    """
    keep = np.ones(len(samples), dtype=bool)
    for side in sides:
        M = np.stack([
            np.linalg.inv(s["base"]) @ usd_cam_to_optical(np.array(s[side]["T_world_cam"]))
            for s in samples
        ])
        t = M[:, :3, 3]
        ref = M[int(np.argmin(np.linalg.norm(t - np.median(t, axis=0), axis=1)))]
        dt = np.linalg.norm(t - ref[:3, 3], axis=1)
        cos_ang = np.clip(
            (np.einsum("nij,ij->n", M[:, :3, :3], ref[:3, :3]) - 1.0) / 2.0, -1.0, 1.0
        )
        ang = np.degrees(np.arccos(cos_ang))
        keep &= (dt <= tol_trans_m) & (ang <= tol_rot_deg)
    kept = [s for s, k in zip(samples, keep) if k]
    return kept, int(len(samples) - len(kept))


# ─── synthesis ────────────────────────────────────────────────────────────────

def synthesize(
    dump_dir: Path | str, out_path: Path | str, spec: BagSpec | None = None
) -> Dict[str, Any]:
    """Write the bag; returns stats. Raises if the dump violates rig assumptions."""
    spec = spec or BagSpec()
    dump_dir = Path(dump_dir)
    rig, samples = _load_samples(dump_dir, spec)
    if not samples:
        raise ValueError(f"no complete stereo samples found in {dump_dir}")
    samples, n_transient = _drop_transient_samples(samples)
    if n_transient:
        print(f"[rosbag_synth] dropped {n_transient} pre-settle transient sample(s)")
    if not samples:
        raise ValueError("all samples classified as transient — dump unusable")
    left_cam, right_cam = rig["stereo_pair"]
    baseline = float(rig["baseline_m"])
    frames = {
        left_cam: f"{left_cam}{spec.optical_frame_suffix}",
        right_cam: f"{right_cam}{spec.optical_frame_suffix}",
    }

    # fixed mounts recovered from RENDERED poses (loud failure if not rigid)
    mounts = {
        cam: recover_static_mount(
            [(s["base"], np.array(s[side]["T_world_cam"])) for s in samples]
        )
        for side, cam in (("left", left_cam), ("right", right_cam))
    }

    plugin = {"mcap": StoragePlugin.MCAP, "sqlite3": StoragePlugin.SQLITE3}[spec.storage]
    n_msgs = 0
    with Writer(Path(out_path), version=9, storage_plugin=plugin) as writer:
        conn = {
            "li": writer.add_connection(spec.left_image_topic, _MSG_IMAGE, typestore=_TS),
            "lc": writer.add_connection(spec.left_info_topic, _MSG_CAMINFO, typestore=_TS),
            "ri": writer.add_connection(spec.right_image_topic, _MSG_IMAGE, typestore=_TS),
            "rc": writer.add_connection(spec.right_info_topic, _MSG_CAMINFO, typestore=_TS),
            "tf": writer.add_connection(spec.tf_topic, _MSG_TF, typestore=_TS),
            "tfs": writer.add_connection(spec.tf_static_topic, _MSG_TF, typestore=_TS),
            "od": writer.add_connection(spec.odom_topic, _MSG_ODOM, typestore=_TS),
        }
        imu_rows = _load_imu(dump_dir, spec, samples[0]["stamp"])
        if imu_rows:
            conn["imu"] = writer.add_connection(spec.imu_topic, _MSG_IMU, typestore=_TS)

        def put(key: str, msg: Any, msgtype: str, t_ns: int) -> None:
            nonlocal n_msgs
            writer.write(conn[key], t_ns, _TS.serialize_cdr(msg, msgtype))
            n_msgs += 1

        t0 = samples[0]["stamp"]
        static_tfs = [
            _transform_stamped(mounts[cam], t0, spec.base_frame, frames[cam])
            for cam in (left_cam, right_cam)
        ]
        if imu_rows:
            # synthesized IMU is base-frame → identity mount
            static_tfs.append(
                _transform_stamped(np.eye(4), t0, spec.base_frame, spec.imu_frame))
        put("tfs", _T[_MSG_TF](transforms=static_tfs), _MSG_TF, t0)

        imu_i = 0  # streaming merge: flush IMU up to each camera stamp (both sorted)

        def _flush_imu(up_to_ns: int) -> None:
            nonlocal imu_i
            while imu_i < len(imu_rows) and imu_rows[imu_i]["stamp_ns"] <= up_to_ns:
                put("imu", _imu_msg(imu_rows[imu_i], spec.imu_frame), _MSG_IMU,
                    imu_rows[imu_i]["stamp_ns"])
                imu_i += 1

        for s in samples:
            t = s["stamp"]
            _flush_imu(t)
            for side, cam, img_key, info_key, tx in (
                ("left", left_cam, "li", "lc", 0.0),
                ("right", right_cam, "ri", "rc", baseline),
            ):
                put(img_key,
                    _image_msg(dump_dir / s[side]["file"], t, frames[cam]),
                    _MSG_IMAGE, t)
                put(info_key,
                    make_camera_info(
                        rig["cameras"][cam]["intrinsics"],
                        frame_id=frames[cam], stamp_ns=t, baseline_m=tx,
                    ),
                    _MSG_CAMINFO, t)
            put("tf",
                _T[_MSG_TF](transforms=[
                    _transform_stamped(s["base"], t, spec.map_frame, spec.base_frame)
                ]),
                _MSG_TF, t)
            put("od", _odom_msg(s["base"], t, spec.map_frame, spec.base_frame),
                _MSG_ODOM, t)
        _flush_imu(2**63 - 1)  # trailing IMU after the last frame

    return {
        "stamps": len(samples),
        "dropped_transient": n_transient,
        "imu_rows": len(imu_rows),
        "messages": n_msgs,
        "out": str(out_path),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Coverage-drive dump → ROS 2 bag for the cuVGL map bake."
    )
    ap.add_argument("--dump", required=True, help="dump dir (rig.json, poses.jsonl)")
    ap.add_argument("--out", required=True, help="output bag directory")
    ap.add_argument("--storage", choices=("mcap", "sqlite3"), default="mcap")
    ap.add_argument("--every-n", type=int, default=1, help="keep every Nth stamp")
    ap.add_argument("--max-stamps", type=int, default=None, help="cap stamp count")
    ap.add_argument(
        "--skip-seconds", type=float, default=0.0,
        help="drop everything before dump_start + SKIP (cold-start pre-lock frames)",
    )
    ap.add_argument(
        "--no-imu", action="store_true",
        help="exclude the IMU stream even if the dump has imu.jsonl (ablation bags)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    stats = synthesize(
        args.dump,
        args.out,
        BagSpec(
            storage=args.storage,
            every_n=args.every_n,
            max_stamps=args.max_stamps,
            skip_seconds=args.skip_seconds,
            include_imu=not args.no_imu,
        ),
    )
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
