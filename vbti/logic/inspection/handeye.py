#!/usr/bin/env python3
"""Eye-in-hand calibration for the wrist D405 on the SO-ARM101.

Solves T_flange_cam: how the camera is bolted to the gripper flange.
Board detection runs on the IR-left stream — factory-rectified, zero
distortion, and the frame native depth lives in. (D405 color is UNRECTIFIED;
never feed its factory coefficients to solvePnP.)

Usage:
    python -m vbti.logic.inspection.handeye collect --outdir=data/inspection/handeye
    python -m vbti.logic.inspection.handeye solve   --outdir=data/inspection/handeye

Collect flow: the LEADER arm teleoperates the follower (tick-to-tick mirror,
see servos/teleop.py). Pose via the leader, press Enter — the mirror pauses so
the follower holds rigid during capture. Repeat 20-30x with varied rotations
(roll the wrist, tilt, look from left/right/high/low). 'q' ends the session.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from vbti.logic.cameras.d405_still import open_camera, session_metadata, capture_bundle
from vbti.logic.inspection.capture import WRIST_SERIAL
from vbti.logic.servos.fk_so101 import fk

# Board: must match the printed sheet (verified vs depth 2026-08-14, sq=20.07mm)
BOARD_SQUARES = (8, 6)
SQUARE_M = 0.020
MARKER_M = 0.015
DICT = cv2.aruco.DICT_4X4_50

MIN_CORNERS = 8


def make_board():
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_M, MARKER_M,
                                  cv2.aruco.getPredefinedDictionary(DICT))


def detect_board(ir_img: np.ndarray, K: np.ndarray):
    """ChArUco corners -> T_cam_target. Returns (T, n_corners) or (None, n)."""
    board = make_board()
    det = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = det.detectBoard(ir_img)
    n = 0 if ids is None else len(ids)
    if n < MIN_CORNERS:
        return None, n
    if board.checkCharucoCornersCollinear(ids):
        return None, n  # collinear corners -> solvePnP returns garbage
    obj, imgp = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, None)
    if not ok:
        return None, n
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec[:, 0]
    return T, n


def _K_from_meta(meta: dict) -> np.ndarray:
    i = meta["intrinsics"]["ir_left"]
    return np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1]])


# ── Collect ─────────────────────────────────────────────────────────────────

def collect(outdir: str, serial: str = WRIST_SERIAL,
            port: str = "/dev/ttyACM0",
            leader_port: str = "/dev/ttyACM1") -> None:
    """Interactive pair collection, leader-arm teleop.

    Drive the follower with the leader arm; the follower holds rigid
    (torque on) while each pair is captured. Enter=capture, q=quit.
    """
    import time as _time
    from vbti.logic.servos.so101 import SO101
    from vbti.logic.servos.teleop import TeleopMirror

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe, profile, align, depth_scale = open_camera(serial)
    meta = session_metadata(profile, depth_scale, serial)
    (outdir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
    K = _K_from_meta(meta)

    existing = sorted(outdir.glob("pair_*.npz"))
    idx = len(existing)
    print(f"{idx} existing pairs in {outdir}")

    board = make_board()
    det = cv2.aruco.CharucoDetector(board)

    follower, leader = SO101(port), SO101(leader_port)
    mirror = TeleopMirror(follower, leader)
    status = ""
    try:
        mirror.start()
        print("TELEOP LIVE — drive the follower with the leader arm.")
        print("Live window: SPACE=capture, q=quit.\n")

        while True:
            fs = pipe.wait_for_frames(timeout_ms=2000)
            rgb = np.asanyarray(fs.get_color_frame().get_data())
            ir = np.asanyarray(fs.get_infrared_frame(1).get_data())

            # Detection overlay on the IR view (the calibration stream)
            ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            corners, ids, _, _ = det.detectBoard(ir)
            n_live = 0 if ids is None else len(ids)
            if n_live:
                cv2.aruco.drawDetectedCornersCharuco(ir_bgr, corners, ids)
            color = (0, 255, 0) if n_live >= MIN_CORNERS else (0, 0, 255)
            cv2.putText(ir_bgr, f"corners: {n_live}  pairs: {idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if status:
                cv2.putText(ir_bgr, status, (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            view = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), ir_bgr])
            cv2.imshow("hand-eye collect  (SPACE=capture, q=quit)", view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key != ord(" "):
                continue

            mirror.pause()          # follower freezes at current goal
            _time.sleep(0.3)        # let it settle on the held goal
            try:
                joints_deg = follower.read_degrees()
                bundle = capture_bundle(pipe, align, settle_s=0.05, flush=2)
            finally:
                mirror.resume()

            T_cam_target, n = detect_board(bundle["ir_left"], K)
            if T_cam_target is None:
                status = f"REJECT: {n} corners"
                print(f"  REJECT: {n} corners (<{MIN_CORNERS} or collinear).")
                continue

            T_base_flange = fk(joints_deg)
            np.savez(outdir / f"pair_{idx:03d}.npz",
                     T_base_flange=T_base_flange,
                     T_cam_target=T_cam_target,
                     joints_deg=np.array([joints_deg[k] for k in sorted(joints_deg)]),
                     n_corners=n)
            cv2.imwrite(str(outdir / f"pair_{idx:03d}_ir.png"), bundle["ir_left"])
            dist = np.linalg.norm(T_cam_target[:3, 3]) * 1000
            status = f"saved pair {idx}: {n} corners @ {dist:.0f}mm"
            print(f"  saved pair {idx}: {n} corners, board at {dist:.0f} mm")
            idx += 1
    finally:
        mirror.stop()
        follower.close()
        leader.close()
        pipe.stop()
        cv2.destroyAllWindows()
    print(f"\n{idx} pairs total. Solve with:\n"
          f"  python -m vbti.logic.inspection.handeye solve --outdir={outdir}")


# ── Solve ───────────────────────────────────────────────────────────────────

METHODS = {
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
}


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def solve(outdir: str) -> None:
    """calibrateHandEye over all collected pairs; PARK primary + cross-check."""
    outdir = Path(outdir)
    pairs = sorted(outdir.glob("pair_*.npz"))
    if len(pairs) < 3:
        raise SystemExit(f"Only {len(pairs)} pairs — need >=3, want 20+.")

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for p in pairs:
        d = np.load(p)
        R_g2b.append(d["T_base_flange"][:3, :3])
        t_g2b.append(d["T_base_flange"][:3, 3])
        R_t2c.append(d["T_cam_target"][:3, :3])
        t_t2c.append(d["T_cam_target"][:3, 3])
    print(f"{len(pairs)} pairs loaded from {outdir}")

    results = {}
    for name, flag in METHODS.items():
        R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = R_c2g, t_c2g[:, 0]
        results[name] = X
        t = X[:3, 3] * 1000
        print(f"{name:>11}: cam at flange +[{t[0]:+.1f} {t[1]:+.1f} {t[2]:+.1f}] mm")

    # Cross-method agreement
    Xp = results["PARK"]
    for name in ("DANIILIDIS", "HORAUD"):
        d = results[name]
        dt = np.linalg.norm((d[:3, 3] - Xp[:3, 3])) * 1000
        dr = _rot_angle_deg(d[:3, :3].T @ Xp[:3, :3])
        print(f"PARK vs {name}: {dt:.2f} mm, {dr:.3f} deg")

    # Physical consistency: board pose in base frame must be identical for
    # every pair. Its spread IS the calibration residual.
    boards = []
    for p in pairs:
        d = np.load(p)
        boards.append(d["T_base_flange"] @ Xp @ d["T_cam_target"])
    ts = np.array([b[:3, 3] for b in boards]) * 1000
    dev = np.linalg.norm(ts - ts.mean(axis=0), axis=1)
    print(f"\nresidual (board-in-base spread, PARK): "
          f"median {np.median(dev):.2f} mm, max {dev.max():.2f} mm")
    print("recipe: <=1.5mm good for SO-101; >3mm = bad pairs, recollect worst")

    np.save(outdir / "T_flange_cam.npy", Xp)
    print(f"\nsaved {outdir}/T_flange_cam.npy (PARK)")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"collect": collect, "solve": solve})
