"""show — project a recorded run into a Rerun .rrd (user stories C1/D4).

Robot-free 3D inspection: camera frusta from T_base_cam + session intrinsics,
step images, fused cloud (colored when fused_colors.npy rides along).
Works on both generations via the legacy adapter. Read-only on the run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inspection.record.legacy import load_any


def show_run(run_dir: Path, out: Path) -> Path:
    import rerun as rr

    run_dir, out = Path(run_dir), Path(out)
    a = load_any(run_dir)
    rr.init(f"record-{a.run.id}", spawn=False)  # rerun 0.37: no '/' in app ids
    rr.save(str(out))

    intr = _color_intrinsics(run_dir)

    for sid in sorted(a.steps):
        step = a.steps[sid]
        if step.T_base_cam is None:
            continue
        rr.set_time("step", sequence=sid)
        T = np.asarray(step.T_base_cam)
        ent = f"world/cam/{sid:03d}"
        rr.log(ent, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        if intr is not None:
            rr.log(ent, rr.Pinhole(
                image_from_camera=intr["K"], width=intr["w"], height=intr["h"]))
        rgb = _step_file(run_dir, sid, "rgb.png")
        if rgb is not None:
            rr.log(f"{ent}/rgb", rr.EncodedImage(contents=rgb.read_bytes(),
                                                 media_type="image/png"))
        ply = _step_file(run_dir, sid, "cloud.ply")
        if ply is not None:
            rr.log(f"world/step_cloud/{sid:03d}", rr.Asset3D(path=str(ply)))

    _log_fused(rr, run_dir)
    return out


def _log_fused(rr, run_dir: Path) -> None:
    for cand in (run_dir / "fused" / "cloud.npy", run_dir / "fused_cloud.npy"):
        if cand.exists():
            pts = np.load(cand)
            colors = None
            cpath = cand.with_name(cand.name.replace("cloud", "colors"))
            if cpath.exists():
                c = np.load(cpath)
                if len(c) == len(pts):
                    colors = c
            rr.log("world/fused", rr.Points3D(pts, colors=colors, radii=0.002))
            return


def _step_file(run_dir: Path, sid: int, name: str) -> Path | None:
    for cand in (run_dir / "steps" / f"{sid:03d}" / name,
                 run_dir / f"{sid:03d}" / name):  # legacy layout
        if cand.exists():
            return cand
    return None


def _color_intrinsics(run_dir: Path) -> dict | None:
    sp = run_dir / "session.json"
    if not sp.exists():
        return None
    try:
        raw = json.loads(sp.read_text())
        c = raw["intrinsics"]["color"]
        return {"K": np.array([[c["fx"], 0, c["ppx"]],
                               [0, c["fy"], c["ppy"]], [0, 0, 1]]),
                "w": c["width"], "h": c["height"]}
    except Exception:
        return None
