"""
load_oli.py — Smoke-test loader for HU_D04_01 in Isaac Sim.

Goal:
  1. Boot Isaac Sim with an interactive viewport.
  2. Load HU_D04_01.usd at /World/Oli, pinned at the root.
  3. Print the joint name/index table (DOF order is what we'll use later
     for sweeps and policy I/O).
  4. Run the physics + render loop until the window is closed.

Run:
  conda activate isaac
  python humanoid/logic/simulation/isaacsim/load_oli.py

No joint commands are applied yet — Oli should hang motionless in the air,
held by the pinned root, while gravity pulls limp arms/legs to their
zero-effort poses (drives default to k_p≈1.7, k_d≈0.017, very soft).
"""

# ── 1. Bootstrap Kit BEFORE any other isaac/pxr imports ───────────────────
# Use the FULL experience (same .kit file the `isaacsim` CLI loads) so we
# get the editor UI: Stage panel, Property panel, Robotics/Physics/Sensors
# menus, asset browser, etc. The base.python experience boots faster but is
# bare — only viewport + physics. Worth the extra ~10s startup here.
from pathlib import Path as _Path

from isaacsim import SimulationApp

FULL_KIT = (
    _Path("/home/may33/miniconda3/envs/isaac/lib/python3.11/"
          "site-packages/isaacsim/apps/isaacsim.exp.full.kit")
)
SIM_APP = SimulationApp({"headless": False, "experience": str(FULL_KIT)})

# ── 2. Now safe to import the rest ────────────────────────────────────────
from pathlib import Path

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

# ── 3. Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/may33/projects/ml_portfolio/robotics/humanoid")
OLI_USD = PROJECT_ROOT / "assets/oli/usd/HU_D04_01.usd"
ROBOT_PRIM_PATH = "/World/Oli"
SPAWN_HEIGHT_M = 1.05  # lift base so legs clear the ground when pinned

# ── 4. Build the world ────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Reference the USD into the stage at /World/Oli
add_reference_to_stage(usd_path=str(OLI_USD), prim_path=ROBOT_PRIM_PATH)

# Lift /World/Oli before reset — fixRootLink pins the base at its current
# pose at world.reset(), so the translate must be authored *before* reset.
oli_xform = UsdGeom.Xformable(world.stage.GetPrimAtPath(ROBOT_PRIM_PATH))
existing_ops = {op.GetOpName(): op for op in oli_xform.GetOrderedXformOps()}
if "xformOp:translate" in existing_ops:
    translate_op = existing_ops["xformOp:translate"]
else:
    translate_op = oli_xform.AddTranslateOp()
translate_op.Set(Gf.Vec3d(0.0, 0.0, SPAWN_HEIGHT_M))
print(f"[load_oli] Spawn height: {SPAWN_HEIGHT_M} m")

# ── 5. Pin the base ───────────────────────────────────────────────────────
# The USD already has PhysxArticulationAPI applied to the articulation root.
# We flip fixRootLink = True on that prim before reset().
stage = world.stage
oli_root_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)

articulation_root = None
for prim in Usd.PrimRange(oli_root_prim):
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_root = prim
        break

assert articulation_root is not None, "No ArticulationRootAPI prim found under /World/Oli"
print(f"[load_oli] Articulation root: {articulation_root.GetPath()}")

# Ensure PhysxArticulationAPI is applied (it is in the source USD, but Apply
# is idempotent and ensures the schema accessor returns valid attrs).
physx_art = PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)

# Pin the base. Across Isaac Sim versions the attribute name has been stable
# but the schema accessor has shifted — use the raw attribute as the
# version-tolerant path.
fix_attr = articulation_root.GetAttribute("physxArticulation:fixRootLink")
if not fix_attr or not fix_attr.IsValid():
    fix_attr = articulation_root.CreateAttribute(
        "physxArticulation:fixRootLink", Sdf.ValueTypeNames.Bool
    )
fix_attr.Set(True)
print("[load_oli] physxArticulation:fixRootLink = True")

# ── 6. Reset world to materialize the articulation, then introspect joints
world.reset()

oli = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="oli")
oli.initialize()

print("\n[load_oli] DOF order (index → name):")
print("─" * 60)
for i, name in enumerate(oli.dof_names):
    print(f"  [{i:>2}] {name}")
print("─" * 60)
print(f"[load_oli] Total DOFs: {oli.num_dof}")

# ── 7. Main loop: step physics + render ───────────────────────────────────
print("\n[load_oli] Entering render loop. Close the Isaac Sim window to exit.")
while SIM_APP.is_running():
    world.step(render=True)

SIM_APP.close()
