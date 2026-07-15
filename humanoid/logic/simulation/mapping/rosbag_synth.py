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


def recover_static_mount(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    tol_trans_m: float = 1e-3,
    tol_rot_deg: float = 0.1,
) -> np.ndarray:
    """Recover the fixed base_link→optical mount from rendered world poses.

    ``pairs`` = [(T_map_base, T_world_cam_USD), …] per stamp. Raises ValueError
    if the recovered mount is not constant within tolerance (i.e. the camera is
    NOT rigidly mounted over this capture — do not write a static tf then).
    """
    if not pairs:
        raise ValueError("recover_static_mount: no pose pairs")
    mounts = [np.linalg.inv(T_mb) @ usd_cam_to_optical(T_wc) for T_mb, T_wc in pairs]
    ref = mounts[0]
    for i, m in enumerate(mounts[1:], start=1):
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


# ─── message assembly (rosbags typestore) ─────────────────────────────────────

_TS = get_typestore(Stores.ROS2_JAZZY)
_T = _TS.types

_MSG_IMAGE = "sensor_msgs/msg/Image"
_MSG_CAMINFO = "sensor_msgs/msg/CameraInfo"
_MSG_TF = "tf2_msgs/msg/TFMessage"
_MSG_ODOM = "nav_msgs/msg/Odometry"


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

        def put(key: str, msg: Any, msgtype: str, t_ns: int) -> None:
            nonlocal n_msgs
            writer.write(conn[key], t_ns, _TS.serialize_cdr(msg, msgtype))
            n_msgs += 1

        t0 = samples[0]["stamp"]
        put(
            "tfs",
            _T[_MSG_TF](transforms=[
                _transform_stamped(mounts[cam], t0, spec.base_frame, frames[cam])
                for cam in (left_cam, right_cam)
            ]),
            _MSG_TF,
            t0,
        )

        for s in samples:
            t = s["stamp"]
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

    return {"stamps": len(samples), "messages": n_msgs, "out": str(out_path)}


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
    args = ap.parse_args(list(argv) if argv is not None else None)

    stats = synthesize(
        args.dump,
        args.out,
        BagSpec(storage=args.storage, every_n=args.every_n, max_stamps=args.max_stamps),
    )
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
