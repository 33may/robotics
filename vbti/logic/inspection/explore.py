#!/usr/bin/env python3
"""Interactive exploration loop — azimuth-driven orbit of the object.

Dashboard window with three panels:
    [ live RGB | fused cloud side view | top-down + azimuth ring ]
Top-down ring shows every viewsphere cell as a dot on its projected ring
(one ring per elevation): green=available, red=blocked, blue=visited,
yellow=current target.

Keys: SPACE = one exploration step, a = autorun toggle, q = quit+save.
The first step is the survey capture; each further step visits the nearest
unvisited reachable cell, fuses the new view, re-centroids, re-maps.
The VLM later replaces exactly the cell picker, nothing else.

Usage:
    python -m vbti.logic.inspection.explore run --outdir=data/inspection/explore1
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from vbti.logic.cameras.d405_still import open_camera, session_metadata, capture_bundle
from vbti.logic.inspection.capture import WRIST_SERIAL, save_bundle
from vbti.logic.inspection.geometry import CloudAccumulator, object_from_view
from vbti.logic.inspection.viewsphere import (
    DEFAULT_R_M, V_ELEVATIONS, cell_cam_pose, reachability,
)
from vbti.logic.servos.so101 import connect

SURVEY_POSE = {"shoulder_pan": -10.0, "shoulder_lift": -75.0,
               "elbow_flex": 60.0, "wrist_flex": 25.0, "wrist_roll": 10.0}

FILL_FRACTION = 0.45        # object fills ~45% of frame height (D1)
FRAME_K = 1.108             # 2*tan(58deg/2): frame height per meter of range
FINGER_REACH_M = 0.13       # fingers end this far past the camera
CLEARANCE_M = 0.02          # margin beyond fingertips (incl. ~8mm kin. error)


def view_radius(aabb: tuple | None, r_default: float) -> float:
    """D1: framing radius vs mechanical floor — whichever is larger."""
    if aabb is None:
        return r_default
    ext = aabb[1] - aabb[0]
    d_max = float(ext.max())
    r_framing = d_max / (FRAME_K * FILL_FRACTION)
    r_floor = FINGER_REACH_M + float(np.linalg.norm(ext[:2])) / 2 + CLEARANCE_M
    return max(r_framing, r_floor, r_default)

PANEL = 360                 # px, square viz panels
SPAN_M = 0.28               # world meters shown across a panel
COL_AVAIL = (0, 220, 0)
COL_BLOCKED = (0, 0, 220)
COL_VISITED = (220, 120, 0)
COL_TARGET = (0, 220, 220)


# ── Rendering ───────────────────────────────────────────────────────────────

def _to_px(xy: np.ndarray, center: np.ndarray, size: int = PANEL,
           span: float = SPAN_M) -> np.ndarray:
    """World (n,2) -> pixel (n,2). y up -> v down."""
    scale = size / span
    uv = (xy - center) * scale
    uv[:, 1] *= -1
    return (uv + size / 2).astype(int)


def _scatter(canvas: np.ndarray, px: np.ndarray, color, thick: int = 1) -> None:
    ok = (px[:, 0] >= 0) & (px[:, 0] < canvas.shape[1]) & \
         (px[:, 1] >= 0) & (px[:, 1] < canvas.shape[0])
    canvas[px[ok, 1], px[ok, 0]] = color


def render_side(points: np.ndarray, center3: np.ndarray) -> np.ndarray:
    """x-z projection; table line at z=0."""
    c = np.zeros((PANEL, PANEL, 3), np.uint8)
    ground = _to_px(np.array([[center3[0], 0.0]]),
                    np.array([center3[0], center3[2]]))[0]
    cv2.line(c, (0, ground[1]), (PANEL, ground[1]), (60, 60, 60), 1)
    if len(points):
        z = points[:, 2]
        px = _to_px(points[:, [0, 2]], np.array([center3[0], center3[2]]))
        t = np.clip((z - z.min()) / max(float(z.max() - z.min()), 1e-6), 0, 1)
        for lo, hi, col in [(0.0, .33, (255, 120, 0)), (.33, .66, (0, 220, 0)),
                            (.66, 1.01, (0, 120, 255))]:
            m = (t >= lo) & (t < hi)
            _scatter(c, px[m], col)
    cv2.putText(c, "side x-z", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (200, 200, 200), 1)
    return c


def render_top(points: np.ndarray, center3: np.ndarray, reach: dict | None,
               visited: set, target: tuple | None, r: float) -> np.ndarray:
    """x-y projection + azimuth cell rings (one ring per elevation)."""
    c = np.zeros((PANEL, PANEL, 3), np.uint8)
    ctr2 = center3[:2]
    if len(points):
        _scatter(c, _to_px(points[:, :2], ctr2), (0, 200, 0))
    # centroid cross
    p0 = _to_px(np.array([ctr2]), ctr2)[0]
    cv2.drawMarker(c, tuple(p0), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    if reach is not None:
        for (h, v), q in reach.items():
            pos, _ = cell_cam_pose(center3, h, v, r)
            px = _to_px(pos[None, :2], ctr2)[0]
            if (h, v) == target:
                col, rad = COL_TARGET, 6
            elif (h, v) in visited:
                col, rad = COL_VISITED, 4
            elif q is not None:
                col, rad = COL_AVAIL, 4
            else:
                col, rad = COL_BLOCKED, 2
            cv2.circle(c, tuple(px), rad, col, -1)
            if v == 0:
                cv2.putText(c, str(h), (px[0] + 4, px[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, (150, 150, 150), 1)
    cv2.putText(c, "top x-y", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (200, 200, 200), 1)
    return c


def compose(rgb: np.ndarray, side: np.ndarray, top: np.ndarray,
            hud: list[str]) -> np.ndarray:
    left = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, PANEL))
    frame = np.hstack([left, side, top])
    bar = np.zeros((46, frame.shape[1], 3), np.uint8)
    for i, line in enumerate(hud):
        cv2.putText(bar, line, (10, 18 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 1)
    return np.vstack([frame, bar])


# ── Loop ────────────────────────────────────────────────────────────────────

def run(outdir: str, r: float = DEFAULT_R_M, max_steps: int = 24,
        serial: str = WRIST_SERIAL, port: str = "/dev/ttyACM0") -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe, profile, align, depth_scale = open_camera(serial)
    meta = session_metadata(profile, depth_scale, serial)
    (outdir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
    intr = meta["intrinsics"]["ir_left"]

    acc = CloudAccumulator()
    visited: set[tuple[int, int]] = set()
    log: list[dict] = []
    reach: dict | None = None
    target: tuple | None = None
    step = 0
    auto = False
    status = "SPACE = survey capture"
    r_used = r

    def do_step(arm) -> str:
        nonlocal reach, target, step, status, r_used
        joints = arm.read_degrees()
        if step == 0:
            arm.move_to(SURVEY_POSE)
            joints = arm.read_degrees()
            bundle = capture_bundle(pipe, align)
            save_bundle(outdir, 0, bundle, joints, arm.read_ticks())
            view = object_from_view(bundle["depth_raw"], intr, depth_scale, joints)
            if view["centroid"] is None:
                return "survey: NO OBJECT on table"
            acc.add(view["points"])
            step = 1
            return f"survey ok: {len(view['points'])} pts"

        r_used = view_radius(acc.aabb(), r)
        reach = reachability(acc.centroid, r=r_used, q0_deg=joints)
        # safety filter removed for now (Anton, 2026-08-14) — see safety.py;
        # real motion planning lands with the UR5e port
        open_cells = {c: q for c, q in reach.items()
                      if q is not None and c not in visited}
        if not open_cells or step > max_steps:
            return "exploration complete"

        target, q_target = min(open_cells.items(),
                               key=lambda kv: sum(abs(joints[k] - kv[1][k])
                                                  for k in kv[1]))
        status_r = f"r={r_used*1000:.0f}mm"
        arm.move_to(q_target)
        joints = arm.read_degrees()
        bundle = capture_bundle(pipe, align)
        save_bundle(outdir, step, bundle, joints, arm.read_ticks())
        visited.add(target)
        view = object_from_view(bundle["depth_raw"], intr, depth_scale, joints)
        if len(view["points"]):
            acc.add(view["points"])
        log.append({"step": step, "cell": list(target),
                    "n_view": len(view["points"]),
                    "n_fused": len(acc.points),
                    "centroid_mm": list(np.round(acc.centroid * 1000, 2))})
        (outdir / "state.json").write_text(json.dumps(
            {"visited": sorted(list(v) for v in visited), "log": log},
            indent=2) + "\n")
        step += 1
        return (f"cell h={target[0]} v={target[1]} {status_r}: "
                f"+{len(view['points'])} pts, fused {len(acc.points)}")

    try:
        with connect(port) as arm:
            while True:
                fs = pipe.wait_for_frames(timeout_ms=2000)
                rgb = np.asanyarray(fs.get_color_frame().get_data())

                center = acc.centroid if acc.centroid is not None \
                    else np.array([0.35, 0.0, 0.05])
                n_open = sum(1 for c, q in (reach or {}).items()
                             if q is not None and c not in visited)
                hud = [f"step {step}  visited {len(visited)}  open {n_open}  "
                       f"auto {'ON' if auto else 'off'}   {status}",
                       f"centroid {np.round(center*1000,1)} mm   "
                       "keys: SPACE=step  a=auto  q=quit"]
                frame = compose(rgb, render_side(acc.points, center),
                                render_top(acc.points, center, reach,
                                           visited, target, r_used), hud)
                cv2.imshow("explore  (SPACE=step, a=auto, q=quit)", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("a"):
                    auto = not auto
                if key == ord(" ") or auto:
                    status = do_step(arm)
                    if status == "exploration complete":
                        auto = False
                    time.sleep(0.05)
    finally:
        if len(acc.points):
            np.save(outdir / "fused_cloud.npy", acc.points)
        pipe.stop()
        cv2.destroyAllWindows()
        print(f"saved {outdir} — fused {len(acc.points)} pts, "
              f"{len(visited)} cells visited")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run})
