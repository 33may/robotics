"""slam_to_camera_tum — convert cuVSLAM vehicle poses (map frame M) to a
head_left CAMERA trajectory in optical convention, the format
`fuse_reconstruction.py --tum` expects (stamp_s tx ty tz qx qy qz qw,
pose = T_map_cam).

Input is a bake's `pycuvslam_map/slam_poses.tum`, written by
`logic/oli/reason/localization/realizations/cuvslam/build_map.py`:
`stamp = timestamp_ns / 1e9`, quaternion **xyzw**, pose = world_from_rig where
the rig is VEHICLE-frame (rig_from_camera = edex sensor_to_vehicle) — i.e. the
base_link pose in M (x fwd, y left, z up).

Per-line conversion:  T_map_cam = T_map_vehicle @ T_base_cam
where T_base_cam is the constant base←camera-optical SE(3) from the shared
mount table — the same `_base_from_camera` math the cuvslam realization's
`module.py` uses to build its rig (replicated here because that module imports
the `cuvslam` package, unavailable outside bench-cuvslam).

Run with the repo parent on PYTHONPATH (for `humanoid.*` imports):
    PYTHONPATH=<repo_parent> python logic/simulation/mapping/slam_to_camera_tum.py \
        --slam-tum <bake>/pycuvslam_map/slam_poses.tum \
        --out <bake>/slam_fused/slam_head_left.tum
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from humanoid.logic.oli.camera_mounts import CAMERAS, STEREO_CAMERAS, CameraMount


def _quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat_xyzw(m: np.ndarray) -> list:
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-8:
        return [(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                (m[1, 0] - m[0, 1]) / (4 * w), w]
    x = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0
    return [x, (m[0, 1] + m[1, 0]) / (4 * x), (m[0, 2] + m[2, 0]) / (4 * x),
            (m[2, 1] - m[1, 2]) / (4 * x)]


def _base_from_camera(mount: CameraMount) -> np.ndarray:
    """Constant base←camera-optical SE(3) — verbatim replica of
    `realizations/cuvslam/module.py::_base_from_camera` (mount-table math).

    Optical axes in the base frame (x fwd, y left, z up), level camera:
    x_opt(right) = -y_base, y_opt(down) = -z_base, z_opt(fwd) = +x_base; then
    the mount's pitch-down tilt about +Y_base.
    """
    r0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    a = math.radians(mount.pitch_down_deg)
    ry = np.array([
        [math.cos(a), 0.0, math.sin(a)],
        [0.0, 1.0, 0.0],
        [-math.sin(a), 0.0, math.cos(a)],
    ])
    t = np.eye(4)
    t[:3, :3] = ry @ r0
    t[:3, 3] = np.asarray(mount.pos_base, dtype=float)
    return t


def convert(slam_tum: Path, out: Path, camera: str = "head_left") -> int:
    mounts = {m.name: m for m in (*CAMERAS, *STEREO_CAMERAS)}
    if camera not in mounts:
        raise SystemExit(f"camera {camera!r} not in mount table {sorted(mounts)}")
    t_base_cam = _base_from_camera(mounts[camera])

    lines_out = []
    for line in slam_tum.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        stamp, t, q = v[0], v[1:4], v[4:8]
        t_map_vehicle = np.eye(4)
        t_map_vehicle[:3, :3] = _quat_xyzw_to_mat(q)
        t_map_vehicle[:3, 3] = t
        t_map_cam = t_map_vehicle @ t_base_cam
        qc = _mat_to_quat_xyzw(t_map_cam[:3, :3])
        tc = t_map_cam[:3, 3]
        lines_out.append(
            f"{stamp:.9f} {tc[0]:.9f} {tc[1]:.9f} {tc[2]:.9f} "
            f"{qc[0]:.9f} {qc[1]:.9f} {qc[2]:.9f} {qc[3]:.9f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines_out) + "\n")
    return len(lines_out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slam-tum", required=True, type=Path,
                    help="pycuvslam_map/slam_poses.tum (vehicle poses in M)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output camera TUM (T_map_cam, optical convention)")
    ap.add_argument("--camera", default="head_left")
    a = ap.parse_args(argv)
    n = convert(a.slam_tum, a.out, a.camera)
    print(f"wrote {n} camera poses ({a.camera}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
