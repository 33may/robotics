"""
Convert clean duck mesh to a deformable body in the current Isaac Sim scene.
Run in Isaac Sim Script Editor AFTER opening object_1.usda in the viewport.

Steps before running:
  1. File → Open → select object_1.usda
  2. Window → Script Editor → paste/run this script
  3. File → Save As → duck_deformable.usda
"""
import omni.usd
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, PhysxSchema
from omni.physx.scripts import deformableUtils

# ── Config V7 ───────────────────────────────────────────────
YOUNGS_MODULUS = 5e5       # Pa — stiffer, deforms only under real force
POISSONS_RATIO = 0.499     # near-incompressible rubber
DENSITY = 300.0            # kg/m³ — hollow rubber toy
DYNAMIC_FRICTION = 1.0
ELASTICITY_DAMPING = 0.0001

SOLVER_POSITION_ITERATIONS = 20
VERTEX_VELOCITY_DAMPING = 0.005
SIMULATION_HEX_RESOLUTION = 50  # finer tet mesh = tighter fit to surface
# ─────────────────────────────────────────────────────────────

stage = omni.usd.get_context().get_stage()

# Find the mesh prim under /World
mesh_prim = None
world_prim = stage.GetPrimAtPath("/World")
for prim in Usd.PrimRange(world_prim):
    if prim.IsA(UsdGeom.Mesh):
        mesh_prim = prim
        break

if not mesh_prim:
    raise RuntimeError("No Mesh found under /World — is object_1.usda open?")

mesh_path = str(mesh_prim.GetPath())
print(f"[INFO] Found mesh at: {mesh_path}")

# ── Step 1: Apply deformable body via deformableUtils ────────
# This is the exact same function IsaacLab calls for MeshCuboidCfg.
# It applies PhysxDeformableBodyAPI, creates simulation tetrahedral mesh,
# sets up collision, and does the PhysX cooking — all in one call.
success = deformableUtils.add_physx_deformable_body(
    stage,
    prim_path=mesh_path,
    simulation_hexahedral_resolution=SIMULATION_HEX_RESOLUTION,
    solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
    vertex_velocity_damping=VERTEX_VELOCITY_DAMPING,
    self_collision=False,
)
if not success:
    raise RuntimeError("add_physx_deformable_body failed — check mesh quality")
print("[INFO] Applied deformable body to mesh")

# ── Step 2: Create and bind deformable material ──────────────
material_path = "/World/DeformableMaterial"
material_prim = UsdShade.Material.Define(stage, material_path)

physx_mat_api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(material_prim.GetPrim())
physx_mat_api.GetYoungsModulusAttr().Set(YOUNGS_MODULUS)
physx_mat_api.GetPoissonsRatioAttr().Set(POISSONS_RATIO)
physx_mat_api.GetDensityAttr().Set(DENSITY)
physx_mat_api.GetDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
physx_mat_api.GetElasticityDampingAttr().Set(ELASTICITY_DAMPING)

# Bind material to the mesh
UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(
    material_prim,
    UsdShade.Tokens.weakerThanDescendants,
    "physics"
)
print("[INFO] Created and bound deformable material")

print("\n[SUCCESS] Duck is now a deformable body!")
print("Press Play to test, then File → Save As → duck_deformable.usda")
