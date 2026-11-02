"""model_audit.py — dump Isaac USD physical properties to diff against the MJCF.

Systematic-debugging tool for the forward-walk failure: the brain walks in MuJoCo, so the
gap is Isaac sim fidelity. This dumps the things MuJoCo's MJCF defines that could differ —
total/per-link mass, joint armature, and the FOOT COLLIDER geometry (type/approximation/
extent) — so we compare against the MJCF reference (total 54.05 kg, joint damping 0.01,
foot = box 0.13×0.047×0.01 m + heel/center/tip contact spheres).

    conda run -n isaac python humanoid/logic/simulation/isaacsim/model_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from humanoid.logic.simulation.isaacsim.oli import Oli  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 1000.0, rendering_dt=1.0 / 50.0)
world.scene.add_default_ground_plane()
oli = Oli(world, pin_root=False, spawn_pose=(0.0, 0.0, 1.0))
view = oli._art._articulation_view
stage = world.stage

print("\n=== MASS (MJCF total = 54.05 kg) ===", flush=True)
try:
    masses = np.asarray(view.get_body_masses()).reshape(-1)
    names = list(view.body_names) if hasattr(view, "body_names") else [
        p.GetName() for p in []]
    print(f"total mass = {float(masses.sum()):.3f} kg  ({len(masses)} bodies)", flush=True)
    order = np.argsort(masses)[::-1]
    for i in order[:8]:
        nm = names[i] if i < len(names) else f"body{i}"
        print(f"  {nm:28s} {masses[i]:.3f} kg", flush=True)
except Exception as e:
    print(f"  mass read failed: {e}", flush=True)

print("\n=== ARMATURE (currently set on the drive) ===", flush=True)
try:
    arm = np.asarray(view.get_armatures()).reshape(-1)
    print(f"  min={arm.min():.4f} max={arm.max():.4f} (0 = USD default; "
          f"MJCF legs 0.141)", flush=True)
except Exception as e:
    print(f"  armature read failed: {e}", flush=True)

print("\n=== ALL prims with CollisionAPI (path + type + approximation) ===", flush=True)
n = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    n += 1
    tn = prim.GetTypeName()
    info = f"type={tn}"
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        info += f" approx={UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()}"
    try:
        if tn == "Mesh":
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            if pts is not None and len(pts):
                info += f" extent={np.round(pts.max(0) - pts.min(0), 3)} npts={len(pts)}"
        elif tn == "Cube":
            info += f" size={UsdGeom.Cube(prim).GetSizeAttr().Get()}"
        elif tn == "Sphere":
            info += f" radius={UsdGeom.Sphere(prim).GetRadiusAttr().Get()}"
    except Exception:
        pass
    print(f"  COL {prim.GetPath()} {info}", flush=True)
print(f"total colliders: {n}", flush=True)

app.close()
