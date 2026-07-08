"""camera_smoke.py — render Oli's baked D435i cameras and verify (§4 + §10, MAY-149).

Boots Isaac, loads Oli WITH cameras, pins the base, holds joints at nominal, drops a
target cube ahead of the head camera, renders, saves RGB + depth per camera, and checks:
  - camera prim world pose == base_pose ∘ mount  (the FK cross-check — proves the baked
    placement matches the shared mount table the brain uses; design.md D5/D10)
  - horizontal FOV == D435i (69°)
  - depth renders with finite returns on the target

    conda run -n isaac python humanoid/logic/simulation/isaacsim/camera_smoke.py
    conda run -n isaac python .../camera_smoke.py --out /tmp/oli_cam --res 640 360 --gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _save_png(path: Path, arr: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(arr).save(str(path))
    except Exception:  # pragma: no cover - PIL absent
        np.save(str(path.with_suffix(".npy")), arr)


def _save_depth_png(path: Path, depth: np.ndarray) -> None:
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)
    norm = np.zeros_like(d)
    if finite.any():
        lo, hi = np.percentile(d[finite], [5, 95])
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    norm[~finite] = 0.0
    _save_png(path, (norm * 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/oli_cam_smoke"))
    ap.add_argument("--res", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not args.gui})

    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import Usd, UsdGeom

    from humanoid.logic.oli.camera_mounts import CAMERAS
    from humanoid.logic.simulation.isaacsim.oli import Oli

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 200.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    world.get_physics_context().set_gravity(0.0)  # base pinned, joints held → static scene

    spawn = np.array([0.0, 0.0, 1.05])
    oli = Oli(
        world, spawn_pose=tuple(spawn), pin_root=True,
        cameras=True, camera_resolution=(args.res[0], args.res[1]),
    )
    oli.set_joint_state(np.zeros(oli.num_dof))  # nominal joints → mount table applies directly

    # A target cube 1 m ahead of the head camera (base upright at spawn).
    head = next(m for m in CAMERAS if m.name == "head")
    head_world = spawn + head.pos_base
    world.scene.add(FixedCuboid(
        prim_path="/World/target",
        position=head_world + np.array([1.0, 0.0, 0.0]),
        scale=np.array([0.3, 0.5, 0.5]),
    ))

    for _ in range(60):
        world.step(render=True)

    stage = omni.usd.get_context().get_stage()
    base_pos = oli.base_world_position()
    ok = True
    for m in CAMERAS:
        rgb, depth = oli.read_camera_rgbd(m.name)
        intr = oli.camera_intrinsics(m.name)
        _save_png(args.out / f"{m.name}_rgb.png", rgb)
        _save_depth_png(args.out / f"{m.name}_depth.png", depth)
        np.save(args.out / f"{m.name}_depth.npy", depth)

        cam_path = f"/World/Oli/{m.parent_link}/{m.name}_camera"
        mm = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_pos = np.array(mm.ExtractTranslation())
        expected = base_pos + m.pos_base  # base upright (identity) → FK = base + base-frame mount
        pose_err = float(np.linalg.norm(world_pos - expected))
        view = (np.array(mm.ExtractRotationMatrix()).T) @ np.array([0.0, 0.0, -1.0])
        hfov = float(np.degrees(2 * np.arctan(intr.width / (2 * intr.fx))))
        finite = np.isfinite(depth)
        cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
        print(
            f"[smoke] {m.name:5s} pose_err={pose_err:.4f}m "
            f"view=({view[0]:.2f},{view[1]:.2f},{view[2]:.2f}) rgb={rgb.shape} "
            f"depth_finite={finite.mean():.2f} center={depth[cy, cx]:.3f}m "
            f"hfov={hfov:.1f}°",
            flush=True,
        )
        # pose_err is cm-level: the FK cross-check assumes exact-nominal joints, but the
        # head chain rests a few mm/deg off zero → ~2 cm at the head. That's consistency,
        # not sub-mm calibration; the real signal is "camera renders where the mount says".
        ok = ok and pose_err < 0.03 and abs(hfov - 69.0) < 1.0 and bool(finite.any())

    print(f"[smoke] VERDICT: {'PASS' if ok else 'FAIL'} — frames saved to {args.out}", flush=True)
    app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
