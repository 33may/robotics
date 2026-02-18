#!/usr/bin/env python3
"""
Create a composed USD scene: GS visual splat + collision mesh with physics.

Takes a Gaussian Splat USDZ (visual) and a collision mesh (OBJ/PLY),
extracts the coordinate transform from the USDZ, applies it to the
collision mesh so both are aligned, and creates a single USD scene.

The key problem: 3DGRUT's ply_to_usd applies a coordinate system transform
(nerfstudio space → USD space) to the GS visual. The collision mesh from
Open3D Poisson reconstruction is still in nerfstudio space. This script
reads the transform from the USDZ and applies it to the collision mesh
so they match.

Usage:
    python create_scene_usd.py \
        --splat data/scenes/vbti_table/cleaned_v2/splat.usdz \
        --collision data/scenes/vbti_table/cleaned_v2/collision_mesh.obj \
        --output data/scenes/vbti_table/cleaned_v2/scene.usda \
        --target-faces 20000
"""

import argparse
import numpy as np
from pathlib import Path

import open3d as o3d
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Vt


COLMAP_TO_USD = np.array([
      [-1.,  0.,  0.,  0.],
      [ 0.,  0., -1.,  0.],
      [ 0., -1.,  0.,  0.],
      [ 0.,  0.,  0.,  1.],
])


def extract_usdz_transform(splat_path: str) -> np.ndarray:
    """
    Extract the coordinate transform from the USDZ volume prim.

    3DGRUT's ply_to_usd stores the coordinate system conversion as an
    xformOp:transform on the Volume prim. We read it so we can apply
    the same transform to the collision mesh.

    Returns:
        4x4 numpy transform matrix, or identity if none found.
    """
    stage = Usd.Stage.Open(splat_path)
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Volume":
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            for op in ops:
                if "transform" in op.GetOpName():
                    mat = op.Get()
                    # Gf.Matrix4d → numpy (USD stores row-major)
                    return np.array([list(mat.GetRow(i)) for i in range(4)])
            break

    print("[WARN] No transform found in USDZ, using identity")
    return np.eye(4)


def transform_vertices(verts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to Nx3 vertices."""
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    return verts @ rot.T + trans


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear [0,1]. Matches the IEC 61966-2-1 standard."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform normals using the inverse-transpose of the rotation."""
    rot = matrix[:3, :3]
    # For orthogonal matrices (rotation/reflection), inv(R).T = R
    # This handles the sign flip correctly
    transformed = normals @ rot.T
    # Re-normalize
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return transformed / norms


def decimate_mesh(mesh_path: str, target_faces: int) -> o3d.geometry.TriangleMesh:
    """Load and decimate a mesh to target face count."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    original_faces = len(mesh.triangles)
    print(f"Original mesh: {len(mesh.vertices)} verts, {original_faces} faces")

    if target_faces > 0 and original_faces > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        print(f"Decimated to: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")
    elif target_faces == 0:
        print("Keeping full resolution")

    mesh.compute_vertex_normals()
    return mesh


def create_scene_usd_old_nerfstudio(
    splat_path: str,
    collision_mesh_path: str,
    output_path: str,
    target_faces: int = 20000,
    static_friction: float = 0.7,
    dynamic_friction: float = 0.5,
    restitution: float = 0.1,
):
    splat_path = str(Path(splat_path).resolve())
    collision_mesh_path = str(Path(collision_mesh_path).resolve())
    output_path = str(Path(output_path).resolve())

    # --- 1. Extract transform from USDZ ---
    xform_matrix = extract_usdz_transform(splat_path)
    print(f"USDZ transform:\n{xform_matrix}")

    # --- 2. Decimate collision mesh ---
    mesh = decimate_mesh(collision_mesh_path, target_faces)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)

    print(f"Mesh bounds BEFORE transform: {verts.min(axis=0)} to {verts.max(axis=0)}")

    # --- 3. Apply the same coordinate transform as USDZ ---
    verts = transform_vertices(verts, xform_matrix)
    normals = transform_normals(normals, xform_matrix)

    print(f"Mesh bounds AFTER transform:  {verts.min(axis=0)} to {verts.max(axis=0)}")

    # --- 4. Create USD stage ---
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root xform
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Environment")

    # --- 5. Add GS visual as a reference ---
    visual_xform = UsdGeom.Xform.Define(stage, "/World/Environment/Visual")
    visual_xform.GetPrim().GetReferences().AddReference(splat_path)
    print(f"Added GS visual reference: {splat_path}")

    # --- 6. Create collision mesh (now in USDZ coordinate space) ---
    collision_prim_path = "/World/Environment/CollisionMesh"
    collision_mesh_geom = UsdGeom.Mesh.Define(stage, collision_prim_path)

    collision_mesh_geom.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts])
    )
    collision_mesh_geom.GetFaceVertexCountsAttr().Set(
        Vt.IntArray([3] * len(faces))
    )
    collision_mesh_geom.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray(faces.flatten().tolist())
    )
    collision_mesh_geom.GetNormalsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals])
    )
    collision_mesh_geom.SetNormalsInterpolation("vertex")

    # Matte object: transparent to camera but catches shadows from the GS proxy.
    # Must be visibility=inherited (NOT invisible) for matte compositing to work.
    # doNotCastShadows prevents the "black box" from dome lights.
    collision_prim_temp = stage.GetPrimAtPath(collision_prim_path)
    collision_prim_temp.CreateAttribute(
        "primvars:isMatteObject", Sdf.ValueTypeNames.Bool
    ).Set(True)
    collision_prim_temp.CreateAttribute(
        "primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool
    ).Set(True)

    # --- 7. Apply collision API ---
    collision_prim = stage.GetPrimAtPath(collision_prim_path)
    UsdPhysics.CollisionAPI.Apply(collision_prim)

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
    mesh_collision_api.GetApproximationAttr().Set("meshSimplification")

    # --- 8. Physics material ---
    material_path = "/World/Environment/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.GetStaticFrictionAttr().Set(static_friction)
    physics_material.GetDynamicFrictionAttr().Set(dynamic_friction)
    physics_material.GetRestitutionAttr().Set(restitution)

    binding_api = UsdShade.MaterialBindingAPI.Apply(collision_prim)
    binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

    # --- 9. Shadow proxy: link GS Volume → CollisionMesh ---
    # The GS Volume prim has a 'proxy' relationship that tells Isaac Sim
    # which mesh to use for shadow receiving. Point it at our collision mesh.
    volume_path = "/World/Environment/Visual/World/gauss/gauss"
    volume_prim = stage.OverridePrim(volume_path)
    proxy_rel = volume_prim.CreateRelationship("proxy")
    proxy_rel.SetTargets([collision_prim_path])
    print(f"Set shadow proxy: {volume_path} → {collision_prim_path}")

    # --- 10. Save ---
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()

    print(f"\nScene saved to: {output_path}")
    print(f"  /World/Environment/Visual         — GS splat (referenced)")
    print(f"  /World/Environment/CollisionMesh   — aligned, invisible, physics-enabled")
    print(f"  /World/Environment/PhysicsMaterial  — friction={static_friction}/{dynamic_friction}")
    print(f"  Shadow proxy: Volume → CollisionMesh")



COLMAP_TO_USD = np.array([
    [-1.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0., -1.,  0.,  0.],
    [ 0.,  0.,  0.,  1.],
])


def create_mesh_scene_usd(
    mesh_path: str,
    output_path: str,
    static_friction: float = 0.7,
    dynamic_friction: float = 0.5,
    restitution: float = 0.1,
    apply_colmap_transform: bool = True,
    apply_srgb_conversion: bool = True,
):
    """Create a USD scene from a MILo vertex-colored mesh.

    Single mesh prim serves as both visual (vertex colors) and collision
    (convex decomposition). No NuRec/USDZ — fully TiledCamera compatible.

    Args:
        apply_colmap_transform: Apply COLMAP→USD axis swap (x→-x, y→-z, z→-y).
            Set False if mesh is already in USD Z-up space (e.g. after clean_mesh cut).
        apply_srgb_conversion: Apply sRGB→linear. Set False if colors already linear.
    """
    mesh_path = str(Path(mesh_path).resolve())
    output_path = str(Path(output_path).resolve())

    # --- 1. Load mesh ---
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    normals = np.asarray(mesh.vertex_normals)

    print(f"Loaded: {len(verts)} verts, {len(faces)} faces, colors={colors.shape}")

    if apply_srgb_conversion:
        colors = srgb_to_linear(colors)
    else:
        print("Skipping sRGB→linear (colors assumed already linear)")

    # --- 2. COLMAP → USD coordinate transform ---
    if apply_colmap_transform:
        verts = transform_vertices(verts, COLMAP_TO_USD)
        normals = transform_normals(normals, COLMAP_TO_USD)
    else:
        print("Skipping COLMAP→USD transform (mesh already in USD space)")

    print(f"Bounds: {verts.min(axis=0)} → {verts.max(axis=0)}")

    # --- 3. Create USD stage ---
    stage: Usd.Stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Environment")

    # --- 4. Mesh prim: geometry + vertex colors ---
    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/Environment/Mesh")

    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
    mesh_prim.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
    mesh_prim.SetNormalsInterpolation("vertex")

    UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec3fArray([Gf.Vec3f(*c) for c in colors]))

    mesh_prim.GetDoubleSidedAttr().Set(True)

    # --- 5. Vertex color material (UsdPreviewSurface reading displayColor) ---
    material = UsdShade.Material.Define(stage, "/World/Environment/VertexColorMaterial")

    shader = UsdShade.Shader.Define(stage, "/World/Environment/VertexColorMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    primvar_reader = UsdShade.Shader.Define(stage, "/World/Environment/VertexColorMaterial/PrimvarReader")
    primvar_reader.CreateIdAttr("UsdPrimvarReader_float3")
    primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    primvar_reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        primvar_reader.GetOutput("result")
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    prim = mesh_prim.GetPrim()
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    # --- 6. Collision: convex decomposition ---
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("convexDecomposition")

    # --- 7. Physics material ---
    phys_material = UsdShade.Material.Define(stage, "/World/Environment/PhysicsMaterial")
    physics_mat = UsdPhysics.MaterialAPI.Apply(phys_material.GetPrim())
    physics_mat.GetStaticFrictionAttr().Set(static_friction)
    physics_mat.GetDynamicFrictionAttr().Set(dynamic_friction)
    physics_mat.GetRestitutionAttr().Set(restitution)

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        phys_material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )

    # --- 7. Save ---
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()

    print(f"\nSaved: {output_path}")
    print(f"  /World/Environment/Mesh            — visual + collision (convexDecomposition)")
    print(f"  /World/Environment/PhysicsMaterial  — friction={static_friction}/{dynamic_friction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create USD scene from MILo mesh")
    parser.add_argument("--mesh", required=True, help="Path to MILo mesh PLY")
    parser.add_argument("--output", required=True, help="Output .usda path")
    parser.add_argument("--static-friction", type=float, default=0.7)
    parser.add_argument("--dynamic-friction", type=float, default=0.5)
    parser.add_argument("--restitution", type=float, default=0.1)
    parser.add_argument("--no-colmap-transform", action="store_true",
                        help="Skip COLMAP→USD axis swap (mesh already oriented)")
    parser.add_argument("--no-srgb", action="store_true",
                        help="Skip sRGB→linear conversion (colors already linear)")
    args = parser.parse_args()

    create_mesh_scene_usd(
        mesh_path=args.mesh,
        output_path=args.output,
        static_friction=args.static_friction,
        dynamic_friction=args.dynamic_friction,
        restitution=args.restitution,
        apply_colmap_transform=not args.no_colmap_transform,
        apply_srgb_conversion=not args.no_srgb,
    )
