#!/usr/bin/env python3
"""Inspection capture rig: UR5e pose + D405 frame bundle -> disk.

Owns the on-disk bundle format and the arm/camera glue. Camera layer in
``camera.py``; pose from RTDE joints -> UR5eIK flange FK -> calibrated
T_flange_cam (left-eye frame). Every bundle is base-frame placeable.

Bundle layout:
    {outdir}/session.json            intrinsics, IR baseline, depth scale
    {outdir}/{pose_id:03d}/
        rgb.png, ir_left.png, ir_right.png
        depth_raw.npy, depth_aligned.npy      uint16, hardware units
        meta.json                             joints, T_base_flange,
                                              T_base_cam, timestamp

Usage:
    p inspection/perception/capture.py snap --outdir=data/inspection/test
    p inspection/perception/capture.py snap --outdir=... --no_arm   # camera only
"""

import json
from pathlib import Path

import cv2
import numpy as np

from inspection.perception.camera import (
    WRIST_SERIAL, open_camera, session_metadata, capture_bundle, t_flange_cam,
)

ROBOT_IP = "192.168.2.50"


def save_bundle(outdir: Path, pose_id: int, bundle: dict,
                pose: dict | None) -> Path:
    """Write one frame bundle to {outdir}/{pose_id:03d}/.

    pose: {"joints_rad": [...], "T_base_flange": 4x4, "T_base_cam": 4x4}
    or None for camera-only captures.
    """
    d = Path(outdir) / f"{pose_id:03d}"
    d.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(bundle["rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(d / "ir_left.png"), bundle["ir_left"])
    cv2.imwrite(str(d / "ir_right.png"), bundle["ir_right"])
    np.save(d / "depth_raw.npy", bundle["depth_raw"])
    np.save(d / "depth_aligned.npy", bundle["depth_aligned"])

    meta = {"pose_id": pose_id, "timestamp": bundle["timestamp"]}
    if pose is not None:
        meta.update({k: np.asarray(v).tolist() for k, v in pose.items()})
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return d


def read_pose(robot_ip: str = ROBOT_IP) -> dict:
    """Current arm pose, all frames a bundle needs. Read-only."""
    from rtde_receive import RTDEReceiveInterface

    from inspection.motion.ik import UR5eIK

    q = np.array(RTDEReceiveInterface(robot_ip).getActualQ())
    T_bf = UR5eIK().fk(q)
    return {"joints_rad": q, "T_base_flange": T_bf,
            "T_base_cam": T_bf @ t_flange_cam()}


def snap(outdir: str, pose_id: int | None = None, serial: str = WRIST_SERIAL,
         no_arm: bool = False, robot_ip: str = ROBOT_IP) -> None:
    """Capture one bundle at the current (stationary) pose.

    pose_id defaults to the next free index in outdir.
    """
    outdir = Path(outdir)
    if pose_id is None:
        existing = [int(p.name) for p in outdir.glob("[0-9]*") if p.is_dir()]
        pose_id = max(existing, default=-1) + 1

    pose = None if no_arm else read_pose(robot_ip)

    pipe, profile, align, depth_scale = open_camera(serial)
    try:
        meta_path = outdir / "session.json"
        if not meta_path.exists():
            outdir.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(session_metadata(profile, depth_scale, serial),
                           indent=2) + "\n")

        bundle = capture_bundle(pipe, align)
        d = save_bundle(outdir, pose_id, bundle, pose)

        valid = (bundle["depth_raw"] > 0).mean() * 100
        cam_p = "" if pose is None else \
            f", cam at {np.round(pose['T_base_cam'][:3, 3], 3).tolist()} m"
        print(f"Saved {d}  (depth valid: {valid:.1f}%{cam_p})")
    finally:
        pipe.stop()


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"snap": snap})
