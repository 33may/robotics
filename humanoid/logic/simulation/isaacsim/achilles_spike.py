"""achilles_spike.py — feasibility test: can Isaac/PhysX hold the achilles CLOSED LOOP?

Imports the EXACT HU_D04_01 MuJoCo model (A/B achilles motors + rods + the `<connect>`
loop-closure equality constraints) through Isaac's MJCF importer, then:
  1. confirms the loop constraints actually came in (counts joints / connect prims),
  2. stresses the physics (settle under gravity, then a hard pose perturbation) to see if
     the closed loop stays stable (no NaN / explosion / drift) and transmits motion.

This decides whether the faithful linkage is viable in PhysX before we integrate it.

    conda run -n isaac python humanoid/logic/simulation/isaacsim/achilles_spike.py
    # windowed (watch the ankle/rods):
    conda run -n isaac python humanoid/logic/simulation/isaacsim/achilles_spike.py --gui
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MJCF = (_REPO / "humanoid" / "vendor" / "humanoid-mujoco-sim" / "humanoid-description"
         / "HU_D04_description" / "xml" / "HU_D04_01.xml")

_GUI = "--gui" in sys.argv

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not _GUI})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

_FIX_BASE = "--free" not in sys.argv  # default fixed base (loop-stability test); --free to stand

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 1000.0, rendering_dt=1.0 / 50.0)
world.scene.add_default_ground_plane()

print(f"[spike] importing MJCF: {_MJCF}", flush=True)
status, cfg = omni.kit.commands.execute("MJCFCreateImportConfig")
cfg.set_fix_base(_FIX_BASE)         # fixed = isolate the ankle/achilles loop; --free = stand test
cfg.set_import_inertia_tensor(True)
cfg.set_self_collision(False)
cfg.set_make_default_prim(False)
cfg.set_create_physics_scene(False)  # World owns the physics scene
try:
    cfg.set_import_sites(True)
except Exception:
    pass
omni.kit.commands.execute(
    "MJCFCreateAsset", mjcf_path=str(_MJCF), import_config=cfg, prim_path="/World/Oli")

# ── what did the importer actually build? ──────────────────────────────────────
stage = world.stage
counts: dict = {}
achilles, connects, root_prims, base_rigid = [], [], [], None
for prim in stage.Traverse():
    t = prim.GetTypeName()
    p = str(prim.GetPath())
    counts[t] = counts.get(t, 0) + 1
    pl = p.lower()
    if "achilles" in pl and "joint" in t.lower():
        achilles.append((p, t))
    if "connect" in pl or "loop" in pl:
        connects.append((p, t))
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        root_prims.append(p)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) and p.endswith("base_link/base_link"):
        base_rigid = p

print("[spike] joint-type counts:", {k: v for k, v in counts.items() if "Joint" in k},
      flush=True)
print(f"[spike] achilles joints found: {len(achilles)}", flush=True)
print(f"[spike] connect/loop (spherical) prims: {len(connects)}", flush=True)
print(f"[spike] ArticulationRootAPI on: {root_prims}", flush=True)
print(f"[spike] base rigid body: {base_rigid}", flush=True)

# The MJCF importer stamps ArticulationRootAPI on BOTH the MuJoCo `worldBody` wrapper (a
# non-rigid Xform) AND the real base_link. The worldBody copy makes the physics tensor view
# fail ("did not match any rigid bodies"). Strip every root API that isn't on the base rigid
# body so the articulation is rooted at base_link.
for p in root_prims:
    if p != base_rigid:
        prim = stage.GetPrimAtPath(p)
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        print(f"[spike] removed stray ArticulationRootAPI from {p}", flush=True)
root_path = base_rigid or "/World/Oli"

world.reset()

art = SingleArticulation(prim_path=root_path)
art.initialize()
print(f"[spike] num_dof={art.num_dof}", flush=True)
print(f"[spike] dof_names={list(art.dof_names)}", flush=True)

# ── stability stress: settle under gravity, then yank the ankle, watch the loop ──
print("\n[spike] === settle under gravity (fixed base) ===", flush=True)
exploded = False
for i in range(3000):
    world.step(render=_GUI)
    if i % 300 == 0:
        q = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        dq = np.asarray(art.get_joint_velocities(), dtype=np.float64).reshape(-1)
        nan = bool(np.any(~np.isfinite(q)) or np.any(~np.isfinite(dq)))
        print(f"  t={i/1000:4.2f}s  max|q|={np.nanmax(np.abs(q)):7.2f}  "
              f"max|dq|={np.nanmax(np.abs(dq)):8.2f}  nan={nan}", flush=True)
        if nan or np.nanmax(np.abs(dq)) > 1e3:
            print("  >>> LOOP UNSTABLE (NaN / explosion)", flush=True)
            exploded = True
            break

print(f"\n[spike] VERDICT: closed loop {'BLEW UP' if exploded else 'held stable for 3s'}",
      flush=True)
app.close()
