"""
Convert clean duck mesh to a SURFACE deformable body in Isaac Sim.
Run in Isaac Sim Script Editor AFTER opening object_1.usda in the viewport.

Surface deformable = FEM on triangle mesh surface (thin shell).
No internal tetrahedra = no ghost collision offset.
Correct for hollow objects like rubber duck.

Steps:
  1. File → Open → select the fixed duck USDA (object_1_fixed.usda or similar)
  2. Window → Script Editor → paste/run this script
  3. File → Save As → duck_surface_deformable.usda

Previous attempt used volumetric (add_physx_deformable_body) which fills interior
with tetrahedra → collision mesh extends beyond visual surface → ghost collision.
"""
import omni.usd
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, PhysxSchema
from omni.physx.scripts import deformableUtils

# ── Material Config ──────────────────────────────────────────
# Rubber duck: hollow thin shell, squishy but holds shape
YOUNGS_MODULUS = 5e5       # Pa — soft rubber. Range: 1e5 (very soft) to 3e6 (silicone 70)
POISSONS_RATIO = 0.45      # volume preservation. 0.499 = near-incompressible (can be unstable)
DENSITY = 300.0            # kg/m³ — hollow rubber toy (solid rubber ~1100)
DYNAMIC_FRICTION = 1.0     # rubber is grippy
THICKNESS = 0.002          # 2mm shell thickness — controls collision shell width

# Bending: how much the shell resists folding
BEND_STIFFNESS = 0.5       # higher = stiffer folds. 0 = cloth-like
BEND_DAMPING = 0.01        # damps folding vibrations
ELASTICITY_DAMPING = 0.001 # damps stretching vibrations

# Solver
SOLVER_POSITION_ITERATIONS = 20
VERTEX_VELOCITY_DAMPING = 0.005
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
    raise RuntimeError("No Mesh found under /World — is the duck USDA open?")

mesh_path = str(mesh_prim.GetPath())
print(f"[INFO] Found mesh at: {mesh_path}")

# ── Step 0: Strip any rigid body APIs ────────────────────────
# Surface deformable conflicts with rigid body — remove if present
if mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI):
    print("[WARN] Removing RigidBodyAPI from mesh (conflicts with deformable)")
    mesh_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

# Also remove rigid-body collision schemas if present
for api_name in ["PhysxSDFMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI",
                 "PhysxTriangleMeshSimplificationCollisionAPI", "PhysxConvexHullCollisionAPI"]:
    try:
        api_type = getattr(PhysxSchema, api_name, None)
        if api_type and mesh_prim.HasAPI(api_type):
            mesh_prim.RemoveAPI(api_type)
            print(f"[WARN] Removed {api_name}")
    except Exception:
        pass

# ── Step 1: Apply SURFACE deformable ────────────────────────
# This uses the triangle mesh directly — no tetrahedral volume.
# Collision IS the visual mesh. No ghost offset.
success = deformableUtils.add_physx_deformable_surface(
    stage,
    prim_path=mesh_path,
    solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
    vertex_velocity_damping=VERTEX_VELOCITY_DAMPING,
    self_collision=False,
)
if not success:
    raise RuntimeError(
        "add_physx_deformable_surface failed.\n"
        "Check: mesh must be triangulated, manifold, and not have RigidBodyAPI."
    )
print("[INFO] Applied SURFACE deformable to mesh")

# ── Step 2: Create and bind surface material ─────────────────
material_path = "/World/SurfaceDeformableMaterial"
material_prim = UsdShade.Material.Define(stage, material_path)

# Use Surface material (NOT Body material)
physx_mat_api = PhysxSchema.PhysxDeformableSurfaceMaterialAPI.Apply(material_prim.GetPrim())
physx_mat_api.CreateYoungsModulusAttr().Set(YOUNGS_MODULUS)
physx_mat_api.CreatePoissonsRatioAttr().Set(POISSONS_RATIO)
physx_mat_api.CreateDensityAttr().Set(DENSITY)
physx_mat_api.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
physx_mat_api.CreateThicknessAttr().Set(THICKNESS)

# Surface-specific: bending behavior
material_prim.GetPrim().CreateAttribute(
    "physxDeformableSurfaceMaterial:bendStiffness", Sdf.ValueTypeNames.Float
).Set(BEND_STIFFNESS)
material_prim.GetPrim().CreateAttribute(
    "physxDeformableSurfaceMaterial:bendDamping", Sdf.ValueTypeNames.Float
).Set(BEND_DAMPING)
material_prim.GetPrim().CreateAttribute(
    "physxDeformableSurfaceMaterial:elasticityDamping", Sdf.ValueTypeNames.Float
).Set(ELASTICITY_DAMPING)

# Bind material to mesh
UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(
    material_prim,
    UsdShade.Tokens.weakerThanDescendants,
    "physics"
)
print("[INFO] Created and bound surface deformable material")

# ── Summary ──────────────────────────────────────────────────
print(f"""
[SUCCESS] Duck is now a SURFACE deformable!

Material:
  youngs_modulus:  {YOUNGS_MODULUS:.0e} Pa
  poissons_ratio:  {POISSONS_RATIO}
  density:         {DENSITY} kg/m³
  thickness:       {THICKNESS*1000:.1f} mm
  dynamic_friction: {DYNAMIC_FRICTION}
  bend_stiffness:  {BEND_STIFFNESS}

Solver:
  iterations:      {SOLVER_POSITION_ITERATIONS}
  vertex_damping:  {VERTEX_VELOCITY_DAMPING}

Press Play to test, then File → Save As → duck_surface_deformable.usda

Tuning tips:
  - Duck too floppy? Increase youngs_modulus (try 1e6)
  - Duck collapses under gravity? Increase bend_stiffness (try 1.0-5.0)
  - Gripper slides off? Increase dynamic_friction (try 1.5-2.0)
  - Duck vibrates? Increase elasticity_damping (try 0.01)
  - Ghost collision? Should be gone — surface deformable has no tet mesh overshoot
""")
