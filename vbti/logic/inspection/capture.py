#!/usr/bin/env python3
"""Inspection capture rig: arm pose + D405 frame bundle -> disk.

Owns the on-disk bundle format and the arm/camera glue. The camera layer
lives in ``vbti.logic.cameras.d405_still``.

Bundle layout:
    {outdir}/session.json            intrinsics, IR baseline, depth scale
    {outdir}/{pose_id:03d}/
        rgb.png, ir_left.png, ir_right.png
        depth_raw.npy, depth_aligned.npy      uint16, hardware units
        meta.json                             joints (deg + ticks), timestamp

Usage:
    python -m vbti.logic.inspection.capture snap --outdir=data/inspection/test
    python -m vbti.logic.inspection.capture snap --outdir=... --no_arm    # camera only
"""

import json
from pathlib import Path

import cv2
import numpy as np

from vbti.logic.cameras.d405_still import (
    open_camera, session_metadata, capture_bundle,
)

WRIST_SERIAL = "123622270367"  # gripper D405, see cameras.py CAMERA_PRESETS


def save_bundle(outdir: Path, pose_id: int, bundle: dict,
                joints_deg: dict | None, joints_ticks: dict | None) -> Path:
    """Write one frame bundle to {outdir}/{pose_id:03d}/."""
    d = Path(outdir) / f"{pose_id:03d}"
    d.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(bundle["rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(d / "ir_left.png"), bundle["ir_left"])
    cv2.imwrite(str(d / "ir_right.png"), bundle["ir_right"])
    np.save(d / "depth_raw.npy", bundle["depth_raw"])
    np.save(d / "depth_aligned.npy", bundle["depth_aligned"])

    meta = {"pose_id": pose_id, "timestamp": bundle["timestamp"],
            "joints_deg": joints_deg, "joints_ticks": joints_ticks}
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return d


def live(serial: str = WRIST_SERIAL) -> None:
    """Live viewer for the wrist camera: RGB | IR-left side by side. q quits."""
    pipe, profile, align, depth_scale = open_camera(serial)
    try:
        while True:
            fs = pipe.wait_for_frames(timeout_ms=2000)
            rgb = np.asanyarray(fs.get_color_frame().get_data())
            ir = np.asanyarray(fs.get_infrared_frame(1).get_data())
            view = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                              cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)])
            cv2.imshow("wrist D405  (q quits)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


def snap(outdir: str, pose_id: int | None = None, serial: str = WRIST_SERIAL,
         no_arm: bool = False, port: str = "/dev/ttyACM0") -> None:
    """Capture one bundle at the current (stationary) pose.

    pose_id defaults to the next free index in outdir.
    """
    outdir = Path(outdir)
    if pose_id is None:
        existing = [int(p.name) for p in outdir.glob("[0-9]*") if p.is_dir()]
        pose_id = max(existing, default=-1) + 1

    joints_deg = joints_ticks = None
    if not no_arm:
        from vbti.logic.servos.so101 import connect
        with connect(port) as arm:
            joints_ticks = arm.read_ticks()
            joints_deg = arm.read_degrees()

    pipe, profile, align, depth_scale = open_camera(serial)
    try:
        meta_path = outdir / "session.json"
        if not meta_path.exists():
            outdir.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(session_metadata(profile, depth_scale, serial),
                           indent=2) + "\n")

        bundle = capture_bundle(pipe, align)
        d = save_bundle(outdir, pose_id, bundle, joints_deg, joints_ticks)

        valid = (bundle["depth_raw"] > 0).mean() * 100
        print(f"Saved {d}  (depth valid: {valid:.1f}% of frame)")
    finally:
        pipe.stop()


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"snap": snap, "live": live})
