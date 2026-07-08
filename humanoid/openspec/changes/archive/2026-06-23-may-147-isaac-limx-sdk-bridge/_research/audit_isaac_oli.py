"""
audit_isaac_oli.py — one-off Isaac-side facts for MAY-147 Phase 2.

Boots Isaac Sim headless, loads HU_D04_01.usd at /World/Oli, and dumps:
  - DOF index/name table (task 2.3) → _research/isaac_dof_dump.txt
  - All sensor/IMU prims under /World/Oli (task 2.5) → _research/imu_prim_audit.txt

Exits after the dumps; does not enter a render loop.

Run:
  /home/may33/miniconda3/envs/isaac/bin/python \
    humanoid/openspec/changes/may-147-isaac-limx-sdk-bridge/_research/audit_isaac_oli.py
"""

from pathlib import Path

from isaacsim import SimulationApp

FULL_KIT = Path(
    "/home/may33/miniconda3/envs/isaac/lib/python3.11/"
    "site-packages/isaacsim/apps/isaacsim.exp.base.kit"
)
# base.kit boots faster than full.kit and we don't need the editor UI for an audit
SIM_APP = SimulationApp({"headless": True, "experience": str(FULL_KIT)})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema  # noqa: E402

PROJECT_ROOT = Path("/home/may33/projects/ml_portfolio/robotics/humanoid")
OLI_USD = PROJECT_ROOT / "assets/oli/usd/HU_D04_01.usd"
ROBOT_PRIM_PATH = "/World/Oli"
SPAWN_HEIGHT_M = 1.05
RESEARCH_DIR = (
    PROJECT_ROOT / "openspec/changes/may-147-isaac-limx-sdk-bridge/_research"
)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=str(OLI_USD), prim_path=ROBOT_PRIM_PATH)

oli_xform = UsdGeom.Xformable(world.stage.GetPrimAtPath(ROBOT_PRIM_PATH))
existing_ops = {op.GetOpName(): op for op in oli_xform.GetOrderedXformOps()}
if "xformOp:translate" in existing_ops:
    translate_op = existing_ops["xformOp:translate"]
else:
    translate_op = oli_xform.AddTranslateOp()
translate_op.Set(Gf.Vec3d(0.0, 0.0, SPAWN_HEIGHT_M))

stage = world.stage
oli_root_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)

# Pin root for consistency with load_oli.py
articulation_root = None
for prim in Usd.PrimRange(oli_root_prim):
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_root = prim
        break
assert articulation_root is not None, "No ArticulationRootAPI under /World/Oli"
PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)
fix_attr = articulation_root.GetAttribute("physxArticulation:fixRootLink")
if not fix_attr or not fix_attr.IsValid():
    fix_attr = articulation_root.CreateAttribute(
        "physxArticulation:fixRootLink", Sdf.ValueTypeNames.Bool
    )
fix_attr.Set(True)

world.reset()
oli = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="oli")
oli.initialize()

# ── Task 2.3: dump Isaac DOF order ──────────────────────────────────────
dof_dump_path = RESEARCH_DIR / "isaac_dof_dump.txt"
with dof_dump_path.open("w") as f:
    f.write(f"# Isaac DOF order for {OLI_USD.name} loaded at {ROBOT_PRIM_PATH}\n")
    f.write(f"# Total DOFs: {oli.num_dof}\n")
    f.write("# Format: <index>\\t<dof_name>\n\n")
    for i, name in enumerate(oli.dof_names):
        f.write(f"{i}\t{name}\n")
print(f"[audit] DOF dump → {dof_dump_path}")
print(f"[audit] num_dof = {oli.num_dof}")

# ── Task 2.5: enumerate sensor / IMU prims under /World/Oli ─────────────
sensor_dump_path = RESEARCH_DIR / "imu_prim_audit.txt"
with sensor_dump_path.open("w") as f:
    f.write(f"# Prim-type + applied-API audit under {ROBOT_PRIM_PATH}\n")
    f.write("# Looking for IMU sensors, force sensors, anything sensor-like\n\n")
    f.write("## All prims with non-default applied APIs\n")
    for prim in Usd.PrimRange(oli_root_prim):
        applied = prim.GetAppliedSchemas()
        prim_type = prim.GetTypeName()
        is_sensor_like = (
            "Imu" in prim_type
            or "Sensor" in prim_type
            or any("Imu" in a or "Sensor" in a for a in applied)
        )
        if applied or is_sensor_like:
            tag = "  ← SENSOR-LIKE" if is_sensor_like else ""
            f.write(
                f"{prim.GetPath()}\ttype={prim_type}\tAPIs={list(applied)}{tag}\n"
            )

    f.write("\n## Direct check for IMU schema\n")
    found_imu = False
    for prim in Usd.PrimRange(oli_root_prim):
        if "IsaacImuSensor" in prim.GetTypeName() or any(
            "Imu" in a for a in prim.GetAppliedSchemas()
        ):
            f.write(f"IMU prim found: {prim.GetPath()} (type={prim.GetTypeName()})\n")
            found_imu = True
    if not found_imu:
        f.write("No IMU prim found in HU_D04_01.usd — need to attach one at runtime.\n")

print(f"[audit] Sensor/IMU prim audit → {sensor_dump_path}")

SIM_APP.close()
print("[audit] Done.")
