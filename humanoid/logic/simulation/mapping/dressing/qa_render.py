"""qa_render.py — render a scene at recorded drive poses (MAY-173 dressing QA).

Renders RGB at exact `poses.jsonl` camera poses (the loss-frame autopsy poses) so
before/after dressing can be compared pixel-for-pixel. No physics, no robot —
just the scene USD referenced into an empty stage + one camera prim driven to
each recorded T_world_cam.

Isaac env only (SimulationApp). The pose-row selection is pure python and
importable without isaacsim.

    conda run -n isaac python logic/simulation/mapping/dressing/qa_render.py \
        --scene assets/envs/warehouse_nvidia/Isaac/Environments/Simple_Warehouse/full_warehouse.usd \
        --poses data/coverage_drives/warehouse_coverage_v1/poses.jsonl \
        --times 71.0 164.6 271.0 322.9 --label before --out /tmp/dressing_qa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Drive-matching optics (rig.json of warehouse_coverage_v1): D435i RGB.
_WIDTH, _HEIGHT = 1280, 720
_FX = 931.2057783503649
_FOCAL_MM = 24.0
_CLIP_RANGE = (0.05, 1000.0)
_WARMUP_TICKS = 120   # texture/material streaming after stage open
_POSE_TICKS = 40      # RTX accumulation per pose


def select_pose_rows(poses_jsonl: Path, cam: str, times_s: list[float]) -> list[dict]:
    """Nearest `cam` row per requested time (seconds). Pure python.

    Returns one row per time, each augmented with "t_req" (requested seconds)
    and "t_actual" (matched stamp in seconds).
    """
    rows = []
    with open(poses_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("cam") == cam:
                rows.append(r)
    if not rows:
        raise ValueError(f"no rows for cam={cam!r} in {poses_jsonl}")
    out = []
    for t in times_s:
        t_ns = t * 1e9
        best = min(rows, key=lambda r: abs(r["stamp_ns"] - t_ns))
        row = dict(best)
        row["t_req"] = t
        row["t_actual"] = best["stamp_ns"] / 1e9
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a scene at recorded drive poses.")
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--cam", default="head_left")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--label", default="render", help="output filename prefix")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    picked = select_pose_rows(args.poses, args.cam, args.times)
    for row in picked:
        print(f"[qa_render] t={row['t_req']}s -> stamp {row['stamp_ns']} "
              f"({row['t_actual']:.3f}s)", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from PIL import Image
    from pxr import Gf, UsdGeom

    add_reference_to_stage(usd_path=str(args.scene.resolve()), prim_path="/World/Scene")
    stage = omni.usd.get_context().get_stage()

    cam = UsdGeom.Camera.Define(stage, "/World/qa_camera")
    h_ap = _FOCAL_MM * _WIDTH / _FX  # aperture from the drive's fx → same pixels/rad
    cam.GetFocalLengthAttr().Set(_FOCAL_MM)
    cam.GetHorizontalApertureAttr().Set(float(h_ap))
    cam.GetVerticalApertureAttr().Set(float(h_ap * _HEIGHT / _WIDTH))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(*_CLIP_RANGE))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp()

    # Replicator capture pattern (nvidia-corpus: replicator_tutorials getting-started
    # §14): capture at explicit orchestrator.step() calls, not on play.
    rep.orchestrator.set_capture_on_play(False)
    rp = rep.create.render_product("/World/qa_camera", (_WIDTH, _HEIGHT))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])

    for _ in range(_WARMUP_TICKS):  # let materials/textures stream in
        app.update()

    for row in picked:
        # poses.jsonl T_world_cam is column-vector convention; USD wants row-vector.
        m = np.array(row["T_world_cam"], dtype=np.float64).T
        op.Set(Gf.Matrix4d(m.tolist()))
        rep.orchestrator.step(rt_subframes=_POSE_TICKS)  # RTX accumulation
        data = annot.get_data()
        rgb = np.asarray(data)
        if rgb.ndim != 3 or rgb.shape[0] != _HEIGHT or rgb.shape[1] != _WIDTH:
            raise RuntimeError(f"annotator returned bad frame shape {rgb.shape} "
                               f"at t={row['t_req']}")
        rgb = rgb[..., :3]
        name = f"{args.label}_t{row['t_req']:g}.png"
        Image.fromarray(rgb.astype(np.uint8)).save(args.out / name)
        print(f"[qa_render] saved {args.out / name} rgb={rgb.shape}", flush=True)

    rep.orchestrator.wait_until_complete()
    annot.detach()
    rp.destroy()
    app.close()


if __name__ == "__main__":
    main()
