#!/usr/bin/env python
"""Dense 3D reconstruction: fuse a dump's recorded RGB-D at an externally
optimized camera trajectory (e.g. cuSFM per-camera TUM) into a TSDF mesh.

No nvblox anywhere — open3d ScalableTSDFVolume, full depth range.

Inputs:
  --dump  recording dump dir (poses.jsonl, rig.json, frames/head*, frames/head_depth)
  --tum   per-camera TUM trajectory for `head_left` (cuSFM output_poses format:
          stamp_s tx ty tz qx qy qz qw, optical convention, pose = T_map_cam)
  --out   output mesh .ply

Head RGB-D sits 2.5 cm from head_left — sub-voxel at 3-5 cm voxels, fused as-is
(the same accepted offset as nvblox_inject).

Run in the `isaac` env (open3d lives there):
    ~/miniconda3/envs/isaac/bin/python logic/simulation/mapping/fuse_reconstruction.py \
        --dump data/coverage_drives/teleop_v1_demo \
        --tum  <bake>/cusfm/output_poses/<name>/camera_name-head_left_pose_file.tum \
        --out  <bake>/cusfm/dense_mesh.ply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

STAMP_TOL_S = 0.005  # dump↔TUM nearest-stamp match tolerance


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = np.linalg.norm([qx, qy, qz, qw])
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_tum(path: Path) -> list[tuple[float, np.ndarray]]:
    poses = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
        T = np.eye(4)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = [tx, ty, tz]
        poses.append((t, T))
    poses.sort(key=lambda p: p[0])
    return poses


def load_dump_head_frames(dump: Path) -> list[tuple[float, Path, Path]]:
    """(stamp_s, rgb_path, depth_path) for the head RGB-D cam."""
    frames = []
    with (dump / "poses.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            if d["cam"] == "head" and d.get("file") and d.get("depth_file"):
                frames.append((d["stamp_ns"] / 1e9, dump / d["file"], dump / d["depth_file"]))
    frames.sort(key=lambda p: p[0])
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--tum", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--max-depth", type=float, default=8.0, help="depth truncation [m]")
    ap.add_argument("--mirror-fix", action="store_true",
                    help="container bake frames are REFLECTED vs world (see "
                         "occupancy-bake-recipe memory): right-multiply each pose by "
                         "K=diag(1,1,-1) so relative motions match the real images")
    args = ap.parse_args()

    rig = json.loads((args.dump / "rig.json").read_text())
    intr = rig["cameras"]["head"]["intrinsics"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"])

    traj = load_tum(args.tum)
    frames = load_dump_head_frames(args.dump)
    stamps = np.array([s for s, _, _ in frames])
    print(f"trajectory: {len(traj)} poses | dump head frames: {len(frames)}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=3 * args.voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    K = np.diag([1.0, 1.0, -1.0, 1.0])
    used = skipped = 0
    for t, T_map_cam in traj:
        if args.mirror_fix:
            T_map_cam = T_map_cam @ K
        i = int(np.searchsorted(stamps, t))
        best = min((j for j in (i - 1, i) if 0 <= j < len(frames)),
                   key=lambda j: abs(stamps[j] - t), default=None)
        if best is None or abs(stamps[best] - t) > STAMP_TOL_S:
            skipped += 1
            continue
        _, rgb_path, depth_path = frames[best]
        color = o3d.io.read_image(str(rgb_path))
        depth = o3d.io.read_image(str(depth_path))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T_map_cam))
        used += 1

    print(f"integrated {used} frames, skipped {skipped} (no stamp match)")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.out), mesh)
    print(f"wrote {args.out}: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")


if __name__ == "__main__":
    main()
