#!/usr/bin/env python3
"""D405 Tare depth calibration — the robot holds the camera straight down over the ChArUco board.

Protocol (Anton 2026-10-13, memory project/depth-calib-validation-protocol): perpendicular poses
at several heights over the board, ONE Tare candidate per height, nothing written to the camera
here. Candidates are scored on cell views (depth_val.py); only the winner is written (`apply`).

  p inspection/calib/depth_tare.py measure                   # no motion, no Tare: depth error here
  p inspection/calib/depth_tare.py scan                      # board in view, pendant in Remote
  p inspection/calib/depth_tare.py scan --heights='[0.18,0.24]'
  p inspection/calib/depth_tare.py apply --cand=<h*.json> --yes
  p inspection/calib/depth_tare.py restore --backup=<backup.json> --yes   | restore --factory --yes

Per height:
  1. normal mode (1280x720 ir_left + depth, the depth frame IS the left imager): ChArUco PnP ->
     GT = perpendicular distance left-imager origin -> board plane (what Tare wants); depth error
     before = depth vs the PnP plane in the central ROI; FK cross-check = camera -> probed table
     slab, which should be GT + board thickness.
  2. Tare mode (depth only 256x144@90, colour off — firmware calibration mode):
     run_tare_calibration(GT rounded to mm; the SDK truncates) -> candidate table + health.
  3. quick check: candidate set VOLATILE, depth error re-measured at the same pose, backup restored.
Out: inspection/calib/d405_tare/<MMDD-HHMM>/{backup.json, h<mm>.json, scan.json, *_ir.png}
"""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OUTDIR = Path(__file__).resolve().parent / "d405_tare"
PROBE = Path(__file__).resolve().parent / "table_probe_2026-08-18.yaml"

W, H, FPS = 1280, 720, 15
N_FRAMES = 15                      # frames averaged per measurement
ROI_FRAC = 0.2                     # central box, fraction of width and of height (Tare's ROI)
TARE_JSON = {"average step count": 20, "step count": 20, "accuracy": 2, "scan parameter": 0,
             "data sampling": 0, "apply preset": 1, "host assistance": 0}
TARE_TIMEOUT_MS = 30000
BOARD_DIMS = (0.20, 0.16, 0.006)   # collision box for the board, generous
HEIGHTS = (0.18, 0.24, 0.30)


# ---------------------------------------------------------------- camera
def _device():
    import pyrealsense2 as rs
    from inspection.perception.camera import WRIST_SERIAL
    for d in rs.context().query_devices():
        if d.get_info(rs.camera_info.serial_number) == WRIST_SERIAL:
            return d
    raise RuntimeError(f"D405 {WRIST_SERIAL} not found")


def _start(mode):
    """mode 'normal' = 1280x720 depth + ir_left; 'tare' = depth only 256x144@90."""
    import pyrealsense2 as rs
    from inspection.perception.camera import WRIST_SERIAL
    c = rs.config()
    c.enable_device(WRIST_SERIAL)
    if mode == "tare":
        c.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
    else:
        c.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        c.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    pipe = rs.pipeline()
    prof = pipe.start(c)
    for _ in range(30):                                     # auto-exposure settles
        pipe.wait_for_frames(timeout_ms=3000)
    return pipe, prof


def measure_here(n=N_FRAMES):
    """Board PnP + depth error at the current camera pose. Camera frame = left imager."""
    import pyrealsense2 as rs
    from inspection.perception.handeye import BOARD_SQUARES, SQUARE_M, detect_board
    pipe, prof = _start("normal")
    try:
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        i = prof.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
        K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
        Ts, depths, ir = [], [], None
        for _ in range(n):
            fs = pipe.wait_for_frames(timeout_ms=3000)
            ir = np.asanyarray(fs.get_infrared_frame(1).get_data()).copy()
            depths.append(np.asanyarray(fs.get_depth_frame().get_data()).astype(float) * scale)
            T, _ = detect_board(ir, K)
            if T is not None:
                Ts.append(T)
    finally:
        pipe.stop()
    if len(Ts) < n // 2:
        return {"ok": False, "why": f"board seen in {len(Ts)}/{n} frames"}, ir
    # board plane in the camera frame: normal = board z, through the board origin
    nrm = np.median([T[:3, 2] for T in Ts], 0)
    nrm /= np.linalg.norm(nrm)
    dist = float(np.median([abs(T[:3, 2] @ T[:3, 3]) for T in Ts]))   # m, origin -> plane
    T_cb = Ts[len(Ts) // 2]
    # depth: per-pixel median over the frames, zeros = invalid
    D = np.stack(depths)
    D[D == 0] = np.nan
    z = np.nanmedian(D, 0)
    x0, x1 = int(W * (0.5 - ROI_FRAC / 2)), int(W * (0.5 + ROI_FRAC / 2))
    y0, y1 = int(H * (0.5 - ROI_FRAC / 2)), int(H * (0.5 + ROI_FRAC / 2))
    vv, uu = np.mgrid[y0:y1, x0:x1]
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones(uu.shape)], -1)
    s = np.sign(nrm @ T_cb[:3, 3])                      # plane: (s*nrm) . p = dist
    z_exp = dist / (rays @ (s * nrm))
    zr = z[y0:y1, x0:x1]
    ok = np.isfinite(zr)
    err = zr[ok] - z_exp[ok]                            # + = depth reads long
    # does the board cover the ROI?
    bw, bh = BOARD_SQUARES[0] * SQUARE_M, BOARD_SQUARES[1] * SQUARE_M
    corners = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], float)
    pc = corners @ T_cb[:3, :3].T + T_cb[:3, 3]
    px = (pc[:, :2] / pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    poly = px.astype(np.float32).reshape(-1, 1, 2)
    covered = all(cv2.pointPolygonTest(poly, (float(u), float(v)), False) >= 0
                  for u, v in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    tilt = float(np.degrees(np.arccos(min(1.0, abs(nrm[2])))))
    return {"ok": True, "gt_mm": round(dist * 1000, 2), "frames_with_board": len(Ts),
            "tilt_deg": round(tilt, 2), "roi_covered_by_board": bool(covered),
            "roi_valid_frac": round(float(ok.mean()), 3),
            "depth_err_mm": round(float(np.median(err)) * 1000, 2) if ok.any() else None,
            "depth_err_pct": round(float(np.median(err)) / dist * 100, 2) if ok.any() else None,
            "T_cam_board": T_cb.tolist()}, ir


def run_tare(gt_mm):
    dev = _device()
    pipe, _ = _start("tare")
    try:
        ac = dev.as_auto_calibrated_device()
        table, health = ac.run_tare_calibration(float(round(gt_mm)), json.dumps(TARE_JSON),
                                                lambda p: None, TARE_TIMEOUT_MS)
    finally:
        pipe.stop()
    return list(table), [float(h) for h in health]


def _ac():
    return _device().as_auto_calibrated_device()


def _save_table(path, table, **meta):
    json.dump({"table": list(table), **meta}, open(path, "w"))


def _show(m):
    return json.dumps({k: v for k, v in m.items() if k != "T_cam_board"})


# ---------------------------------------------------------------- robot
def _table_plane():
    import yaml
    P = np.array([p["base"] for p in yaml.safe_load(open(PROBE))["points"]], float)
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][-1]
    return (n if n[2] > 0 else -n), c


def straight_down(T_base_cam_now, c_board, n_up, h):
    """Camera pose h above the board centre, optical axis = -n_up, image x kept from now."""
    z = -n_up
    x = T_base_cam_now[:3, 0] - (T_base_cam_now[:3, 0] @ z) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = c_board + n_up * h
    return T


def _confirm(msg):
    return input(f"{msg} [y/N] ").strip().lower() == "y"


def scan(heights=HEIGHTS, auto=False):
    """Board in view of the camera, pendant in Remote Control. Asks before every move."""
    from inspection.cell.geometry import load_T_flange_cam
    from inspection.cell.world import RobotCell
    from inspection.motion.execute import UR5eArm
    from inspection.motion.ik import UR5eIK
    from inspection.motion.plan import plan_viewpoint
    from inspection.perception.handeye import BOARD_SQUARES, SQUARE_M

    out = OUTDIR / datetime.now().strftime("%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    ac = _ac()
    backup = ac.get_calibration_table()
    _save_table(out / "backup.json", backup, note="flash table before this scan")
    print(f"backup -> {out / 'backup.json'}")

    ik, T_fc = UR5eIK(), load_T_flange_cam("cv")
    world = RobotCell()
    n_up, c_tab = _table_plane()
    rec = {"heights": list(heights), "tare_json": TARE_JSON, "steps": []}
    with UR5eArm() as arm:
        # 1. find the board from where the camera is now
        m0, ir0 = measure_here()
        cv2.imwrite(str(out / "start_ir.png"), ir0)
        if not m0["ok"]:
            raise SystemExit(f"board not found from the start pose: {m0['why']}")
        print(f"start pose: {_show(m0)}")
        T_bc = ik.fk(arm.q()) @ T_fc
        T_b_board = T_bc @ np.array(m0["T_cam_board"])
        c_board = T_b_board[:3, :3] @ [BOARD_SQUARES[0] * SQUARE_M / 2, BOARD_SQUARES[1] * SQUARE_M / 2, 0] \
            + T_b_board[:3, 3]
        n_board = T_b_board[:3, 2] if T_b_board[2, 2] > 0 else -T_b_board[:3, 2]
        above = float((c_board - c_tab) @ n_up * 1000)
        print(f"board centre {np.round(c_board, 4)}, tilt vs probed table "
              f"{np.degrees(np.arccos(min(1, n_board @ n_up))):.2f} deg, {above:.1f} mm above the slab")
        world.set_object("board", BOARD_DIMS, [*c_board, 0, 0, 0])
        rec.update(board_centre_base=c_board.tolist(), board_above_slab_mm=round(above, 2))

        for h in heights:
            T_target = straight_down(T_bc, c_board, n_board, h)
            path, rep = plan_viewpoint(world, ik, arm.q(), T_target @ np.linalg.inv(T_fc))
            if path is None:
                print(f"h={h}: no plan ({rep['reason']}) — skipped")
                rec["steps"].append({"h_m": h, "skipped": rep["reason"]})
                continue
            if not auto and not _confirm(f"h={h:.2f} m: move ({len(path)} waypoints, tier {rep['tier']})?"):
                break
            arm.execute(path, world=world)
            time.sleep(1.0)
            q = arm.q()
            T_bc_h = ik.fk(q) @ T_fc
            fk_slab_mm = float((T_bc_h[:3, 3] - c_tab) @ n_up * 1000)
            before, ir = measure_here()
            cv2.imwrite(str(out / f"h{int(h * 1000)}_ir.png"), ir)
            st = {"h_m": h, "q": q.tolist(), "fk_cam_to_slab_mm": round(fk_slab_mm, 2),
                  "before": {k: v for k, v in before.items() if k != "T_cam_board"}}
            print(f"h={h}: before {_show(before)}")
            if not before["ok"]:
                rec["steps"].append(st)
                continue
            st["fk_minus_gt_mm"] = round(fk_slab_mm - before["gt_mm"], 2)   # ~ board thickness
            if not before["roi_covered_by_board"]:
                print("  WARNING: board does not cover the central ROI — Tare also sees the table")
            try:
                table, health = run_tare(before["gt_mm"])
            except Exception as e:
                print(f"  Tare FAILED: {e}")
                st["tare_error"] = str(e)
                rec["steps"].append(st)
                continue
            cand = out / f"h{int(h * 1000)}.json"
            _save_table(cand, table, gt_mm=round(before["gt_mm"]), health=health, h_m=h)
            st.update(candidate=cand.name, health=health)
            print(f"  Tare health before/after {health} -> {cand.name}")
            # quick check at the same pose: candidate volatile, re-measure, restore
            try:
                ac.set_calibration_table(table)
                after, _ = measure_here()
            finally:
                ac.set_calibration_table(backup)
            st["after_same_pose"] = {k: v for k, v in after.items() if k != "T_cam_board"}
            print(f"  same pose with candidate: depth err {after.get('depth_err_mm')} mm "
                  f"({after.get('depth_err_pct')} %) vs before {before['depth_err_mm']} mm")
            rec["steps"].append(st)
            json.dump(rec, open(out / "scan.json", "w"), indent=1)
    json.dump(rec, open(out / "scan.json", "w"), indent=1)
    print(f"-> {out}")


def measure():
    """No motion, no Tare: board GT and depth error at the current pose."""
    m, _ = measure_here()
    print(_show(m))


def apply(cand: str, yes: bool = False):
    """Write a candidate to the camera EEPROM (persists). Backs up the flash table first."""
    if not yes:
        raise SystemExit("writes the camera EEPROM — pass --yes")
    ac = _ac()
    OUTDIR.mkdir(exist_ok=True)
    bk = OUTDIR / f"backup_before_apply_{datetime.now():%m%d-%H%M%S}.json"
    _save_table(bk, ac.get_calibration_table(), note="flash table before apply")
    ac.set_calibration_table(json.load(open(cand))["table"])
    ac.write_calibration()
    print(f"written {cand}; previous table -> {bk}")


def restore(backup: str = None, factory: bool = False, yes: bool = False):
    if not yes:
        raise SystemExit("writes the camera EEPROM — pass --yes")
    ac = _ac()
    if factory:
        ac.reset_to_factory_calibration()
        print("factory calibration restored")
        return
    ac.set_calibration_table(json.load(open(backup))["table"])
    ac.write_calibration()
    print(f"restored {backup}")


if __name__ == "__main__":
    import fire
    fire.Fire({"scan": scan, "measure": measure, "apply": apply, "restore": restore})
