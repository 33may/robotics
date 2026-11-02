"""e06 common: the 2209 runs as views, the probed table as truth, frames and families.

Orientation rule (e03/e04): rgb / depth_aligned / mask on disk are upright; T_base_cam,
depth_raw and the intrinsics are raw; on rgb_rotation_deg == 180 steps the upright arrays
are flipped back to raw before any geometry. Everything here works in RAW orientation.

Smoke:  p inspection/investigation/iwm/e06_view_offset/common.py
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNS_DIR = ROOT / "inspection" / "data" / "runs"
EVID = HERE / "evidence"
DATA = ROOT / "inspection" / "data" / "iwm" / "e06_view_offset"
CALIB = ROOT / "inspection" / "calib"
W, H = 1280, 720

#: the e05 view ids on these runs (e01 evidence/frames_selected.json)
FRAME_IDS = {("2209-compute-ray-demo5", 2): "c5.2", ("2209-compute-ray-demo5", 6): "c5.6",
             ("2209-compute-ray-demo5", 7): "c5.7", ("2209-box-ray-demo1", 1): "b1.1",
             ("2209-box-ray-demo1", 4): "b1.4"}


def runs():
    """Every 2209 run with at least one captured step, in name order."""
    out = []
    for rd in sorted(RUNS_DIR.glob("2209-*")):
        if any((sd / "step.json").exists() for sd in (rd / "steps").glob("*")):
            out.append(rd.name)
    return out


def session(run):
    s = json.load(open(RUNS_DIR / run / "session.json"))
    return s["intrinsics"], float(s["depth_scale_m_per_unit"])


def K_of(intr):
    return np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])


def family(q_rad):
    """IK family of a joint configuration: A = elbow negative (the hand-eye calibration's
    region, 31/31 pairs), B = elbow positive."""
    return "A" if q_rad[2] < 0 else "B"


def steps_of(run):
    rd = RUNS_DIR / run
    return sorted(int(sd.name) for sd in (rd / "steps").glob("*") if (sd / "step.json").exists())


def load_view(run, step, arrays=True):
    """One captured step as a dict in RAW orientation. None when the step was not captured.

    T: base <- cam. K: colour intrinsics. Kd: depth intrinsics. depth: depth_aligned (colour
    grid, raw orientation). depth_raw: depth grid. mask: object mask (raw orientation) or None.
    """
    sd = RUNS_DIR / run / "steps" / f"{step:03d}"
    st = json.load(open(sd / "step.json"))
    if st.get("outcome") != "captured":
        return None
    intr, scale = session(run)
    rot = int(st.get("rgb_rotation_deg", 0) or 0)
    T = np.array(st["T_base_cam"], float)
    q = np.array(st["joints_rad"], float)
    v = {"run": run, "step": step, "id": FRAME_IDS.get((run, step), f"{run.split('-')[1][0]}{step}"),
         "T": T, "q": q, "family": family(q), "rot": rot, "K": K_of(intr["color"]), "Kd": K_of(intr["depth"]),
         "scale": scale, "address": st["view"].get("address"), "cam": T[:3, 3].copy(), "axis": T[:3, 2].copy(),
         "t_arrived": st.get("t_arrived"), "t_captured": st.get("t_captured"),
         "geometry": st.get("geometry") or {}, "dir": sd}
    if not arrays:
        return v
    depth = np.load(sd / "depth_aligned.npy")
    rgb = np.array(Image.open(sd / "rgb.png").convert("RGB"))
    mask = np.array(Image.open(sd / "mask.png")) > 0 if (sd / "mask.png").exists() else None
    if rot == 180:
        depth, rgb = depth[::-1, ::-1], rgb[::-1, ::-1]
        mask = None if mask is None else mask[::-1, ::-1]
    v.update(depth=np.ascontiguousarray(depth), rgb=np.ascontiguousarray(rgb),
             mask=None if mask is None else np.ascontiguousarray(mask),
             depth_raw=np.load(sd / "depth_raw.npy"))
    return v


def views(run_names=None):
    for run in (run_names or runs()):
        for s in steps_of(run):
            v = load_view(run, s)
            if v is not None:
                yield v


# ---------------------------------------------------------------- the table as truth

def table_truth():
    """The probed slab: plane [a, b, c, d] (n.p + d = 0, n up) through the five probe points
    of calib/table_probe_2026-08-18.yaml, plus the table frame pose from cell.yaml for the
    footprint. Returns (plane, T_base_table, (x_extent, y_extent))."""
    probe = yaml.safe_load(open(CALIB / "table_probe_2026-08-18.yaml"))
    P = np.array([p["base"] for p in probe["points"]], float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[-1]
    n = n if n[2] > 0 else -n
    plane = np.array([*n, -n @ c])
    cell = yaml.safe_load(open(ROOT / "inspection" / "cell" / "cell.yaml"))
    x, y, z, r, p, yw = cell["frames"]["table"]["pose"]
    T = np.eye(4)
    T[:3, :3] = _rpy(r, p, yw)
    T[:3, 3] = [x, y, z]
    return plane, T, (0.529, 1.0)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def plane_to_frame(plane, T_new_old):
    """Plane n.p + d = 0 given in frame `old` -> the same plane in frame `new`,
    with p_new = T_new_old p_old."""
    R, t = T_new_old[:3, :3], T_new_old[:3, 3]
    n = R @ plane[:3]
    return np.array([*n, plane[3] - n @ t])


def ray_hit(plane, origin, direction):
    """Point where the ray meets the plane (None if parallel or behind)."""
    den = plane[:3] @ direction
    if abs(den) < 1e-6:
        return None
    lam = -(plane[:3] @ origin + plane[3]) / den
    return None if lam <= 0 else origin + lam * direction


def deproject(depth_u16, K, scale, stride=2, lo=0.0, hi=np.inf):
    """Depth image -> (points (N,3) camera frame, pixels (N,2) as (v, u))."""
    z = depth_u16[::stride, ::stride].astype(np.float64) * scale
    vv, uu = np.mgrid[0:depth_u16.shape[0]:stride, 0:depth_u16.shape[1]:stride]
    ok = (z > lo) & (z < hi) & (depth_u16[::stride, ::stride] != 65535)
    z, uu, vv = z[ok], uu[ok], vv[ok]
    x = (uu - K[0, 2]) / K[0, 0] * z
    y = (vv - K[1, 2]) / K[1, 1] * z
    return np.column_stack([x, y, z]), np.column_stack([vv, uu])


def to_base(pc, T):
    return pc @ T[:3, :3].T + T[:3, 3]


def elevation_deg(T):
    """How far the optical axis points below the horizon (90 = straight down)."""
    return float(np.degrees(np.arcsin(np.clip(-T[2, 2], -1, 1))))


def elev_class(e):
    return "top" if e >= 65 else ("oblique" if e >= 25 else "grazing")


def run_record(out, used, settings, extra=None):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    rec = {"date": datetime.now().isoformat(timespec="seconds"), "commit": commit, "command": " ".join(sys.argv),
           "views": used, "settings": settings}
    if extra:
        rec.update(extra)
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(Path(out) / "run.json", "w"), indent=1)


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return super().default(o)


if __name__ == "__main__":
    plane, T_bt, ext = table_truth()
    print("runs:", len(runs()), runs()[:3], "...")
    print("table plane (base):", np.round(plane, 5), " z at table origin:", round(-(plane[:3] @ T_bt[:3, 3] + plane[3]) / plane[2], 5))
    n = 0
    for v in views(["2209-compute-ray-demo5"]):
        n += 1
        print(v["id"], v["family"], v["address"], "elev", round(elevation_deg(v["T"]), 1), v["depth"].shape, v["depth_raw"].shape,
              None if v["mask"] is None else int(v["mask"].sum()))
    print(n, "views")
