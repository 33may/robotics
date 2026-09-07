#!/usr/bin/env python3
"""Inspection capture rig: UR5e pose + D405 frame bundle -> disk.

Owns the on-disk bundle format and the arm/camera glue. Camera layer in
``camera.py``; pose from RTDE joints -> UR5eIK flange FK -> calibrated
T_flange_cam (left-eye frame). Every bundle is base-frame placeable.

Bundle layout (standalone `snap`; inside a recorded run the step directory
comes from `RunWriter.begin_step` and the facts from `steps/NNN/step.json`,
not from a meta.json):
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

from inspection.cell.geometry import is_half_turn, rotate180
from inspection.perception.camera import (
    WRIST_SERIAL, open_camera, session_metadata, capture_bundle, t_flange_cam,
)

ROBOT_IP = "192.168.2.50"


def save_bundle(outdir: Path, pose_id: int, bundle: dict,
                pose: dict | None, write_meta: bool = True,
                dest: Path | None = None) -> Path:
    """Write one frame bundle to `dest`, or to {outdir}/{pose_id:03d}/.

    pose: {"joints_rad": [...], "T_base_flange": 4x4, "T_base_cam": 4x4}
    or None for camera-only captures.

    `dest` is the step directory a `RunWriter.begin_step` handed out: a
    recorded run numbers its own steps, so the caller says WHERE and this
    writes only bytes. `write_meta=False` goes with it — inside a run, the
    pose/rotation facts belong to `steps/NNN/step.json` and a second
    hand-written copy beside it is exactly the drift the record layer exists
    to end. The camera-only `snap` below still writes its meta.json: it makes
    bundles, not runs.
    """
    d = Path(dest) if dest is not None else Path(outdir) / f"{pose_id:03d}"
    d.mkdir(parents=True, exist_ok=True)

    # ---- ORIENTATION (Anton 2026-08-24): the COLOUR frame is stored UPRIGHT.
    # Viewsphere roll is {0, 180}, so a half-turn capture is corrected here,
    # once, and every reader downstream (UI preview, label sheets, the VLM)
    # gets an upright image without each having to know about roll.
    #
    # `rgb` and `depth_aligned` share the colour viewport and rotate TOGETHER.
    # `depth_raw` and the IR pair are left RAW on purpose: they are what
    # `cell.geometry.object_in_base` deprojects against `T_base_cam` and the
    # session intrinsics, so the reconstruction path is untouched and old runs
    # stay replayable. Rotating one of a pixel-paired set is the bug here —
    # not rotating at all.
    rgb, depth_aligned = bundle["rgb"], bundle["depth_aligned"]
    rot = 0
    if pose is not None and is_half_turn(pose["T_base_cam"]):
        rgb, depth_aligned, rot = rotate180(rgb), rotate180(depth_aligned), 180

    cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(d / "ir_left.png"), bundle["ir_left"])
    cv2.imwrite(str(d / "ir_right.png"), bundle["ir_right"])
    np.save(d / "depth_raw.npy", bundle["depth_raw"])
    np.save(d / "depth_aligned.npy", depth_aligned)

    if write_meta:
        meta = {"pose_id": pose_id, "timestamp": bundle["timestamp"],
                # 0 or 180: how much rgb.png / depth_aligned.npy were rotated
                # relative to depth_raw and T_base_cam. Absent on runs captured
                # before 2026-08-24 — treat a missing key as "raw on disk".
                "rgb_rotation_deg": rot}
        if pose is not None:
            meta.update({k: np.asarray(v).tolist() for k, v in pose.items()})
        (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return d


def save_mask(capture_dir, mask: np.ndarray, pose: dict | None) -> None:
    """Add the object mask bitmap to an already-written capture dir.

    Segmentation happens after `save_bundle` (it needs the cloud accumulated
    so far to build its prompt), so the mask is a second, additive write
    rather than part of the bundle. A run stays fully replayable offline:
    `mask.png` is what decided which depth pixels became object points.

    Stored in the SAME orientation as `rgb.png` so the two overlay directly —
    callers hold masks in raw orientation, and the half-turn correction is
    applied here exactly as `save_bundle` applies it to rgb.

    Bytes only: the mask's score/box/px are a RECORD, and they live in the
    step (`schema.py:Segmentation`), written once by the run's writer.
    """
    d = Path(capture_dir)
    if not d.is_dir():
        return
    mask = np.asarray(mask, dtype=bool)
    if pose is not None and is_half_turn(pose["T_base_cam"]):
        mask = rotate180(mask)
    cv2.imwrite(str(d / "mask.png"), mask.astype(np.uint8) * 255)


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
