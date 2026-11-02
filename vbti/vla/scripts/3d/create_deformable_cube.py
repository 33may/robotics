"""
Create a deformable cube in the current Isaac Sim scene.
Run in Isaac Sim Script Editor (Window → Script Editor).
"""

import numpy as np
import omni.usd
from pxr import UsdGeom, Usd, Sdf, UsdShade, Gf, Vt, UsdPhysics
from omni.physx.scripts import deformableUtils


def make_subdivided_cube_triangles(subdivisions: int = 10, half_size: float = 0.05):
    """Generate a subdivided cube mesh with triangular faces.

    Args:
        subdivisions: number of divisions per edge (10 → ~1200 triangles)
        half_size: half-extent in meters (0.05 → 10cm cube)

    Returns:
        vertices (N, 3), faces (M, 3) as numpy arrays
    """
    n = subdivisions + 1
    vertices = []
    faces = []

    def add_face_grid(origin, axis_u, axis_v):
        """Add a subdivided quad face of the cube."""
        base_idx = len(vertices)
        for i in range(n):
            for j in range(n):
                u = i / subdivisions
                v = j / subdivisions
                p = origin + axis_u * u + axis_v * v
                vertices.append(p)

        for i in range(subdivisions):
            for j in range(subdivisions):
                idx = base_idx + i * n + j
                # Two triangles per quad
                faces.append([idx, idx + 1, idx + n + 1])
                faces.append([idx, idx + n + 1, idx + n])

    s = half_size
    # 6 faces of cube
    add_face_grid(np.array([-s, -s, -s]), np.array([2*s, 0, 0]), np.array([0, 2*s, 0]))  # bottom (z=-s)
    add_face_grid(np.array([-s, -s,  s]), np.array([2*s, 0, 0]), np.array([0, 2*s, 0]))  # top (z=+s)
    add_face_grid(np.array([-s, -s, -s]), np.array([2*s, 0, 0]), np.array([0, 0, 2*s]))  # front (y=-s)
    add_face_grid(np.array([-s,  s, -s]), np.array([2*s, 0, 0]), np.array([0, 0, 2*s]))  # back (y=+s)
    add_face_grid(np.array([-s, -s, -s]), np.array([0, 2*s, 0]), np.array([0, 0, 2*s]))  # left (x=-s)
    add_face_grid(np.array([ s, -s, -s]), np.array([0, 2*s, 0]), np.array([0, 0, 2*s]))  # right (x=+s)

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def create_deformable_cube(
    stage: Usd.Stage,
    prim_path: str = "/World/SoftCube",
    position: tuple = (0.35, 0.55, 0.25),
    half_size: float = 0.03,
    subdivisions: int = 8,
    youngs_modulus: float = 1e5,
    poissons_ratio: float = 0.45,
    density: float = 500.0,
    dynamic_friction: float = 0.5,
    thickness: float = 0.005,
):
    """Create a deformable surface cube in the scene.

    Args:
        stage: USD stage
        prim_path: where to create the cube
        position: world position (x, y, z) in meters
        half_size: half-extent of cube in meters
        subdivisions: mesh subdivisions per edge
        youngs_modulus: stiffness in Pascals (1e4=jelly, 1e5=rubber, 1e7=plastic)
        poissons_ratio: volume preservation (0-0.5, higher = more incompressible)
        density: kg/m³ (water=1000, rubber duck ~500)
        dynamic_friction: friction coefficient
        thickness: surface thickness for collision
    """

    # --- 1. Generate subdivided cube mesh ---
    verts, faces = make_subdivided_cube_triangles(subdivisions, half_size)
    print(f"Mesh: {len(verts)} vertices, {len(faces)} triangles")

    # --- 2. Create mesh prim ---
    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
    mesh_prim.GetSubdivisionSchemeAttr().Set("none")
    mesh_prim.GetDoubleSidedAttr().Set(True)

    # Position
    xformable = UsdGeom.Xformable(mesh_prim)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*position))

    # Simple color
    UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
    ).Set(Vt.Vec3fArray([Gf.Vec3f(0.9, 0.7, 0.1)]))  # yellow

    # --- 3. Apply deformable surface body ---
    prim = mesh_prim.GetPrim()

    deformableUtils.add_physx_deformable_surface(
        stage=stage,
        prim_path=Sdf.Path(prim_path),
        solver_position_iteration_count=32,
        vertex_velocity_damping=0.005,
        sleep_damping=10.0,
        sleep_threshold=0.05,
        settling_threshold=0.1,
        self_collision=False,
    )

    # --- 4. Create deformable surface material ---
    material_path = f"{prim_path}/DeformableMaterial"

    deformableUtils.add_deformable_surface_material(
        stage=stage,
        path=Sdf.Path(material_path),
        density=density,
        dynamic_friction=dynamic_friction,
        poissons_ratio=poissons_ratio,
        thickness=thickness,
        youngs_modulus=youngs_modulus,
    )

    # --- 5. Bind material to mesh ---
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(material_path)),
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    print(f"Created deformable cube at {prim_path}")
    print(f"  Position: {position}")
    print(f"  Size: {half_size*2}m, Subdivisions: {subdivisions}")
    print(f"  Young's Modulus: {youngs_modulus}, Poisson: {poissons_ratio}")
    print(f"  Density: {density}, Friction: {dynamic_friction}")
    print(f"  Material: {material_path}")


# ─── Run ───
stage = omni.usd.get_context().get_stage()

# Remove old Xform cube if exists
old = stage.GetPrimAtPath("/World/Xform")
if old.IsValid():
    stage.RemovePrim("/World/Xform")
    print("Removed old /World/Xform")

create_deformable_cube(
    stage=stage,
    prim_path="/World/SoftCube",
    position=(0.35, 0.55, 0.25),   # above table
    half_size=0.03,                  # 6cm cube
    subdivisions=8,                  # 8x8 per face = enough for deformation
    youngs_modulus=1e5,              # rubber-like
    poissons_ratio=0.45,
    density=500.0,
)

print("\nDone! Hit Play to test.")
