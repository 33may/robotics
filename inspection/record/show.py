"""show — project a recorded run into a Rerun .rrd (user stories C1/D4).

Robot-free 3D inspection: camera frusta from T_base_cam + session intrinsics,
step images, fused cloud (colored when fused_colors.npy rides along).
Works on both generations via `Run` (record/run.py), which adapts legacy
runs transparently — no more separate per-layout file probing here.
Read-only on the run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inspection.record.run import Run


def show_run(run_dir: Path, out: Path) -> Path:
    import rerun as rr

    run_dir, out = Path(run_dir), Path(out)
    run = Run.load(run_dir)
    rr.init(f"record-{run.record.id}", spawn=False)  # rerun 0.37: no '/' in app ids
    rr.save(str(out))

    intr = _color_intrinsics(run_dir)

    for step in run.steps:
        if step.T_base_cam is None:
            continue
        rr.set_time("step", sequence=step.id)
        T = step.T_base_cam
        ent = f"world/cam/{step.id:03d}"
        rr.log(ent, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        if intr is not None:
            rr.log(ent, rr.Pinhole(
                image_from_camera=intr["K"], width=intr["w"], height=intr["h"]))
        rgb_path = step.dir / "rgb.png"
        if rgb_path.exists():
            rr.log(f"{ent}/rgb", rr.EncodedImage(contents=rgb_path.read_bytes(),
                                                 media_type="image/png"))
        ply_path = step.dir / "cloud.ply"
        if ply_path.exists():
            rr.log(f"world/step_cloud/{step.id:03d}", rr.Asset3D(path=str(ply_path)))

    fused = run.fused()
    if fused is not None:
        pts, colors = fused
        rr.log("world/fused", rr.Points3D(pts, colors=colors, radii=0.002))
    return out


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
