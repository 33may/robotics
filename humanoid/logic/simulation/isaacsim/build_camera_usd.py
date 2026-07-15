"""build_camera_usd.py — bake Oli's D435i cameras into the robot USD sensor layer.

Authors one `UsdGeom.Camera` prim per camera (chest, head RGBD + the head stereo
pair, MAY-173 T1) as a child of its parent link, into `HU_D04_01_sensor.usd` — the
(otherwise empty) Sensor-variant payload of the robot asset. Idempotent:
re-running overwrites the same prims, never dups.
Mounts + optics come from the shared `logic.oli.camera_mounts` table, so the USD
placement and the brain's FK derivation agree by construction (design.md D1/D3/D10).

Isaac env only (needs `pxr`). No SimulationApp — pure USD authoring.

    python logic/simulation/isaacsim/build_camera_usd.py         # default sensor layer
    python logic/simulation/isaacsim/build_camera_usd.py --rl     # the _rl sensor layer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

# Put the repo root on sys.path so the pure `humanoid.logic.oli` brain module imports
# when this script is run standalone in the isaac env (matches sim_world_main.py).
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.camera_mounts import (  # noqa: E402
    CAMERAS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    STEREO_CAMERAS,
    CameraMount,
    to_parent_local,
)

_ROBOT_ROOT = "/HU_D04_01"
_ASSET_USD = Path(__file__).resolve().parents[3] / "assets" / "oli" / "usd"
_FOCAL_MM = 24.0  # reference focal length; aperture is derived to hit the FOV
_CLIP_RANGE = (0.05, 1000.0)  # meters


def camera_prim_path(mount: CameraMount) -> str:
    """USD path of a camera prim: child of its parent link."""
    return f"{_ROBOT_ROOT}/{mount.parent_link}/{mount.name}_camera"


def _local_transform(mount: CameraMount) -> Gf.Matrix4d:
    """Camera local-to-parent transform. The parent chain is pure-translation at
    nominal pose, so the camera's rotation in the parent frame equals its world
    rotation: view axis (USD camera looks down local -Z) points forward (+X) tilted
    `pitch_down_deg` below horizontal, no roll.
    """
    th = np.radians(mount.pitch_down_deg)
    view = np.array([np.cos(th), 0.0, -np.sin(th)])          # optical axis (forward/down)
    right = np.cross(view, [0.0, 0.0, 1.0])                  # +X_local (horizontal, no roll)
    right = right / np.linalg.norm(right)
    up = np.cross(right, view)                               # +Y_local
    z_local = -view                                          # +Z_local (camera views down -Z)

    pos = to_parent_local(mount.pos_base, mount.parent_link)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(*right.tolist(), 0.0))              # row-vector convention:
    m.SetRow(1, Gf.Vec4d(*up.tolist(), 0.0))                 #   row i = image of local axis i
    m.SetRow(2, Gf.Vec4d(*z_local.tolist(), 0.0))
    m.SetRow(3, Gf.Vec4d(*pos.tolist(), 1.0))
    return m


def _apply_intrinsics(cam: UsdGeom.Camera, hfov_deg: float) -> None:
    """Set focal length + apertures so the horizontal FOV matches the D435i RGB
    sensor. Aspect follows the default render resolution (square pixels)."""
    h_aperture = 2.0 * _FOCAL_MM * np.tan(np.radians(hfov_deg) / 2.0)
    cam.GetFocalLengthAttr().Set(_FOCAL_MM)
    cam.GetHorizontalApertureAttr().Set(float(h_aperture))
    cam.GetVerticalApertureAttr().Set(float(h_aperture * DEFAULT_HEIGHT / DEFAULT_WIDTH))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(*_CLIP_RANGE))


def bake_cameras(sensor_layer_path) -> list[str]:
    """Author the two cameras into the given sensor-layer USD file. Idempotent.
    Returns the camera prim paths."""
    sensor_layer_path = str(sensor_layer_path)
    stage = Usd.Stage.Open(sensor_layer_path)
    paths = []
    for mount in CAMERAS + STEREO_CAMERAS:
        path = camera_prim_path(mount)
        cam = UsdGeom.Camera.Define(stage, path)
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()  # idempotent: don't stack ops on re-bake
        xf.AddTransformOp().Set(_local_transform(mount))
        _apply_intrinsics(cam, mount.hfov_deg)
        paths.append(path)
    stage.GetRootLayer().Save()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake Oli's D435i cameras into the robot USD.")
    ap.add_argument("--rl", action="store_true", help="target the _rl sensor layer")
    ap.add_argument("--sensor", type=Path, default=None, help="explicit sensor-layer USD path")
    args = ap.parse_args()
    if args.sensor is not None:
        sensor = args.sensor
    else:
        name = "HU_D04_01_rl_sensor.usd" if args.rl else "HU_D04_01_sensor.usd"
        sensor = _ASSET_USD / "configuration" / name
    paths = bake_cameras(sensor)
    print(f"[build_camera_usd] baked {len(paths)} cameras into {sensor}:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
