#!/usr/bin/env python3
"""Eye-in-hand calibration for the wrist D405 on the UR5e.

Solves T_flange_cam: how the camera is bolted to the flange. Board detection
runs on the IR-left stream — factory-rectified, zero distortion, and the frame
native depth lives in. (D405 color is UNRECTIFIED; never feed its factory
coefficients to solvePnP.)

Usage:
    p inspection/perception/handeye.py collect --outdir=data/inspection/handeye_ur
    p inspection/perception/handeye.py solve   --outdir=data/inspection/handeye_ur

Collect flow: FREEDRIVE, hands-free auto-capture. You hold the freedrive
button and pose the arm; whenever the arm has been still for DWELL_S and the
board is well-detected and the pose is novel, a pair is captured — accept
beep. No keyboard at the robot. 20-30 pairs, vary ROTATION (roll the wrist,
tilt, look from left/right/high/low); keep translation modest (PARK favors
rotation span). 'q' in the live window ends the session.

FK: joints via RTDE getActualQ() -> UR5eIK.fk() (nominal DH, flange frame,
verified vs pinocchio ur5e_description <1e-9 — the SAME frame cell.yaml welds
the tool boxes to). NOT the pendant TCP pose (that is the RG2-tip TCP and
would solve the wrong transform). Raw joints are saved per pair, so pairs can
be re-FK'd and re-solved if the factory DH calibration gets extracted later.
"""

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from inspection.perception.camera import (
    WRIST_SERIAL, open_camera, session_metadata, capture_bundle,
)

ROBOT_IP = "192.168.2.50"

# Board: must match the printed sheet (verified vs depth 2026-08-14, sq=20.07mm)
BOARD_SQUARES = (8, 6)
SQUARE_M = 0.020
MARKER_M = 0.015
DICT = cv2.aruco.DICT_4X4_50

MIN_CORNERS = 8

# Auto-capture gates (freedrive, hands-free)
QD_STILL = 0.010        # rad/s — every joint below this counts as "still"
DWELL_S = 0.6           # s of continuous stillness before a capture may fire
MIN_JOINT_DIST = 0.15   # rad L2 distance to EVERY captured pose (novelty gate)

SND_ACCEPT = "/usr/share/sounds/freedesktop/stereo/complete.oga"
SND_REJECT = "/usr/share/sounds/freedesktop/stereo/dialog-error.oga"


def _beep(path: str) -> None:
    subprocess.Popen(["paplay", path], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


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
            robot_ip: str = ROBOT_IP) -> None:
    """Hands-free pair collection under freedrive.

    Still for DWELL_S + board detected + pose novel -> capture + accept beep.
    Bad board while still -> one reject beep per stillness episode.
    """
    from rtde_receive import RTDEReceiveInterface

    from inspection.motion.ik import UR5eIK

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rtde = RTDEReceiveInterface(robot_ip)
    fk = UR5eIK()

    pipe, profile, align, depth_scale = open_camera(serial)
    meta = session_metadata(profile, depth_scale, serial)
    (outdir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
    K = _K_from_meta(meta)

    captured_q = []
    for p in sorted(outdir.glob("pair_*.npz")):
        captured_q.append(np.load(p)["q_rad"])
    idx = len(captured_q)
    print(f"{idx} existing pairs in {outdir}")

    board = make_board()
    det = cv2.aruco.CharucoDetector(board)

    still_since = None      # monotonic t when stillness began, None = moving
    episode_beeped = False  # one reject beep max per stillness episode
    status = ""
    WIN = "hand-eye collect  (freedrive, auto)  q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1400, 790)
    try:
        print("FREEDRIVE + auto-capture. Hold still ~1s at each pose.")
        print("Accept beep = pair saved. Double-low beep = fix the board view.")
        print("Live window: q=quit.\n")

        while True:
            fs = pipe.wait_for_frames(timeout_ms=2000)
            t_frame = time.monotonic()
            rgb = np.asanyarray(fs.get_color_frame().get_data())
            ir = np.asanyarray(fs.get_infrared_frame(1).get_data())

            qd = np.abs(rtde.getActualQd())
            if qd.max() > QD_STILL:
                still_since = None
                episode_beeped = False
            elif still_since is None:
                still_since = time.monotonic()

            # RGB display; detection runs on IR (the calibration stream).
            # Corner dots drawn with IR pixel coords — on the D405 the two
            # optical centres are 0.1 mm apart, close enough for display.
            view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            corners, ids, _, _ = det.detectBoard(ir)
            n_live = 0 if ids is None else len(ids)
            if n_live:
                cv2.aruco.drawDetectedCornersCharuco(view, corners, ids)
            # Big far-readable state line: motion state + corners + pair count
            if still_since is None:
                state, scol = "MOVING", (0, 200, 255)
            elif time.monotonic() - still_since < DWELL_S:
                state, scol = "HOLD...", (0, 255, 255)
            else:
                state, scol = "STILL", (0, 255, 0)
            ccol = (0, 255, 0) if n_live >= MIN_CORNERS else (0, 0, 255)
            cv2.putText(view, state, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, scol, 4)
            cv2.putText(view, f"{n_live} corners", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, ccol, 4)
            cv2.putText(view, f"pairs {idx}", (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
            if status:
                cv2.putText(view, status, (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            cv2.imshow(WIN, view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            # ---- auto-capture gate ----
            # Frame must have been exposed INSIDE the stillness window:
            # stillness began before the frame, and the arm is still still.
            if still_since is None or t_frame - still_since < DWELL_S:
                continue

            q = np.array(rtde.getActualQ())
            if captured_q and min(np.linalg.norm(q - c) for c in captured_q) \
                    < MIN_JOINT_DIST:
                continue  # parked on an already-captured pose — silent

            T_cam_target, n = detect_board(ir, K)
            if T_cam_target is None:
                if not episode_beeped:
                    _beep(SND_REJECT)
                    episode_beeped = True
                    status = f"REJECT: {n} corners"
                continue

            # Confirm the arm never moved while we processed
            if np.abs(rtde.getActualQd()).max() > QD_STILL:
                continue

            T_base_flange = fk.fk(q)
            np.savez(outdir / f"pair_{idx:03d}.npz",
                     T_base_flange=T_base_flange,
                     T_cam_target=T_cam_target,
                     q_rad=q, n_corners=n)
            cv2.imwrite(str(outdir / f"pair_{idx:03d}_ir.png"), ir)
            captured_q.append(q)
            dist = np.linalg.norm(T_cam_target[:3, 3]) * 1000
            status = f"saved pair {idx}: {n} corners @ {dist:.0f}mm"
            print(f"  saved pair {idx}: {n} corners, board at {dist:.0f} mm")
            _beep(SND_ACCEPT)
            idx += 1
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    print(f"\n{idx} pairs total. Solve with:\n"
          f"  p inspection/perception/handeye.py solve --outdir={outdir}")


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
    print("recipe: <=1mm good; 1-2mm = nominal-DH floor; >3mm = bad pairs")

    np.save(outdir / "T_flange_cam.npy", Xp)
    print(f"\nsaved {outdir}/T_flange_cam.npy (PARK)")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"collect": collect, "solve": solve})
