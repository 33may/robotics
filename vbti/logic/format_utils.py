#!/usr/bin/env python3
"""3D asset format conversion utilities."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData
from loguru import logger

import trimesh
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Vt, Gf


SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def _load_and_filter_splat(input_path: str, opacity_threshold: float = 0.5,
                           scale_percentile: float = 95.0,
                           outlier_nb: int = 20, outlier_std: float = 2.0):
    """Load a 3DGS PLY and return a filtered, colored Open3D point cloud.

    Filtering pipeline: opacity → scale → statistical outlier removal.
    Returns an o3d.geometry.PointCloud with colors set.
    """
    ply = PlyData.read(input_path)
    v = ply["vertex"]
    n_total = len(v["x"])

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1)
    colors = np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"]))  # sigmoid
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
    scales = np.exp(scales)  # stored as log-scale
    max_scale = scales.max(axis=-1)

    # --- Filter: opacity ---
    mask = opacities > opacity_threshold
    logger.info(f"Opacity filter: {mask.sum()}/{n_total} kept (threshold={opacity_threshold})")

    # --- Filter: scale (remove floaters) ---
    if scale_percentile < 100.0:
        scale_cutoff = np.percentile(max_scale[mask], scale_percentile)
        scale_mask = max_scale <= scale_cutoff
        mask = mask & scale_mask
        logger.info(f"Scale filter: {mask.sum()}/{n_total} kept (percentile={scale_percentile}, cutoff={scale_cutoff:.6f})")

    positions = positions[mask]
    colors = colors[mask]

    # --- Build point cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- Statistical outlier removal (catches isolated floater clusters) ---
    if outlier_nb > 0:
        pcd, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=outlier_nb, std_ratio=outlier_std
        )
        logger.info(f"Outlier removal: {len(inlier_idx)}/{len(positions)} kept (nb={outlier_nb}, std={outlier_std})")

    logger.success(f"Final point cloud: {len(pcd.points)} points")
    return pcd


def splat_to_pointcloud(input_path: str, output_path: str,
                        opacity_threshold: float = 0.5,
                        scale_percentile: float = 95.0,
                        outlier_nb: int = 20, outlier_std: float = 2.0):
    """Step 1: Filter a 3DGS PLY and export as a colored point cloud for review.

    Open the output in SuperSplat / CloudCompare / MeshLab to verify colors
    and that floaters are removed before running splat_to_mesh.
    """
    pcd = _load_and_filter_splat(input_path, opacity_threshold, scale_percentile,
                                 outlier_nb, outlier_std)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    logger.success(f"Saved colored point cloud to {output_path}")



def glb_to_usd(input_path: str, output_path: str, collision: str = "convexDecomposition", rigid_body: bool = True):
    """Convert a GLB (glTF binary) file to USD with materials and textures.

    Args:
        input_path: Path to input .glb file.
        output_path: Path to output .usda file.
        collision: Collision approximation type. Options: convexHull, convexDecomposition,
            meshSimplification, boundingCube, boundingSphere, none.
    """
    output_path = str(Path(output_path))
    output_dir = Path(output_path).parent

    # Load GLB with trimesh (preserves materials + UVs)
    scene = trimesh.load(input_path)

    # Handle single mesh or scene
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene.geometry.values())
    else:
        mesh = scene

    vertices = mesh.vertices
    faces = mesh.faces
    logger.info(f"Vertices: {vertices.shape}, Faces: {faces.shape}")

    # Create USD stage
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp()
    world.AddOrientOp()
    world.AddScaleOp()
    stage.SetDefaultPrim(world.GetPrim())

    # Define mesh
    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh_prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in vertices])
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(faces))
    mesh_prim.GetFaceVertexIndicesAttr().Set(faces.flatten().tolist())

    # UVs
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
        uv_primvar = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        uv_primvar.Set([Gf.Vec2f(*uv) for uv in uvs])
        logger.debug(f"UVs: {uvs.shape}")

    # Normals
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        mesh_prim.GetNormalsAttr().Set([Gf.Vec3f(*n) for n in mesh.vertex_normals])

    # Material + texture
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        mat = mesh.visual.material

        # Create material prim
        usd_mat = UsdShade.Material.Define(stage, "/World/Material")
        shader = UsdShade.Shader.Define(stage, "/World/Material/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Export texture if available
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            tex_path = output_dir / "texture.png"
            Image.fromarray(np.array(mat.baseColorTexture)).save(tex_path)
            logger.debug(f"Texture saved: {tex_path}")

            # Create texture reader shader
            tex_reader = UsdShade.Shader.Define(stage, "/World/Material/TextureReader")
            tex_reader.CreateIdAttr("UsdUVTexture")
            tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./texture.png")
            tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

            # UV reader
            uv_reader = UsdShade.Shader.Define(stage, "/World/Material/UVReader")
            uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
            uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            # Connect UV reader -> texture reader -> shader
            tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv_reader.GetOutput("result"))
            tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex_reader.GetOutput("rgb"))
        else:
            # Flat color
            color = mat.main_color[:3] / 255.0
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))

        # Metallic + roughness
        if hasattr(mat, 'metallicFactor'):
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(mat.metallicFactor or 0.0))
        if hasattr(mat, 'roughnessFactor'):
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(mat.roughnessFactor or 0.5))

        # Connect shader to material and bind
        usd_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(usd_mat)
        logger.debug("Material bound to mesh")

    # Physics
    if rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(world.GetPrim())
        logger.debug("Rigid body applied to /World")

    if collision != "none":
        UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
        mesh_collision.CreateApproximationAttr().Set(collision)
        logger.debug(f"Collision: {collision}")

    stage.GetRootLayer().Save()
    logger.success(f"Saved to {output_path}")


COLMAP_TO_USD = np.array([
    [-1.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0., -1.,  0.,  0.],
    [ 0.,  0.,  0.,  1.],
])


def _transform_vertices(verts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to Nx3 vertices."""
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    return verts @ rot.T + trans


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear [0,1]. Matches the IEC 61966-2-1 standard."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform normals using the inverse-transpose of the rotation."""
    rot = matrix[:3, :3]
    transformed = normals @ rot.T
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return transformed / norms


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation matrix that rotates unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.9999:
        # Nearly opposite — pick an arbitrary perpendicular axis
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, a) * a
        perp = perp / np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    skew = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def _align_and_center(verts: np.ndarray, normals: np.ndarray):
    """Align mesh so the flat surface faces Z-up, centered with bottom at Z=0.

    Uses PCA: the smallest variance direction is the surface normal of a flat
    object (table). Rotates that to Z-up, then centers the bounding box.
    """
    centroid = verts.mean(axis=0)
    centered = verts - centroid

    # PCA — smallest eigenvalue = thinnest axis = surface normal
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    surface_normal = eigenvectors[:, 0]

    # Make sure normal points "up" (positive Z after rotation)
    if surface_normal[2] < 0:
        surface_normal = -surface_normal

    # Rotate normal -> Z-up
    z_up = np.array([0.0, 0.0, 1.0])
    rot = _rotation_between_vectors(surface_normal, z_up)
    verts = (rot @ centered.T).T
    normals = (rot @ normals.T).T

    # Center XY at origin, bottom at Z=0
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    center[2] = bbox_min[2]  # shift so bottom sits at Z=0
    verts = verts - center

    logger.info(f"Aligned to Z-up, centered at origin, bottom at Z=0")
    logger.debug(f"Surface normal was: {surface_normal}")

    return verts, normals


def mesh_to_usd(
    mesh_path: str,
    output_path: str,
    static_friction: float = 0.7,
    dynamic_friction: float = 0.5,
    restitution: float = 0.1,
    apply_colmap_transform: bool = True,
    apply_srgb_conversion: bool = True,
):
    """Convert a MILo vertex-colored mesh PLY to a USD scene.

    Single mesh prim serves as both visual (vertex colors) and collision
    (convex decomposition). No NuRec/USDZ — fully TiledCamera compatible.

    Args:
        mesh_path: path to MILo mesh PLY (mesh_learnable_sdf.ply)
        output_path: output .usda path
        apply_colmap_transform: Apply COLMAP→USD axis swap (x→-x, y→-z, z→-y).
            Set False if mesh is already in USD Z-up space.
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

    logger.info(f"Loaded: {len(verts)} verts, {len(faces)} faces, colors={colors.shape}")

    if apply_srgb_conversion:
        colors = _srgb_to_linear(colors)
    else:
        logger.debug("Skipping sRGB->linear (colors assumed already linear)")

    # --- 2. COLMAP → USD coordinate transform ---
    if apply_colmap_transform:
        verts = _transform_vertices(verts, COLMAP_TO_USD)
        normals = _transform_normals(normals, COLMAP_TO_USD)
    else:
        logger.debug("Skipping COLMAP->USD transform (mesh already in USD space)")

    # --- 3. Align to ground plane + center at origin ---
    verts, normals = _align_and_center(verts, normals)
    logger.info(f"Bounds: {verts.min(axis=0)} -> {verts.max(axis=0)}")

    # --- 4. Create USD stage ---
    stage: Usd.Stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Environment")

    # --- 5. Mesh prim: geometry + vertex colors ---
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

    # --- 6. Vertex color material (UsdPreviewSurface reading displayColor) ---
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

    # --- 7. Collision: convex decomposition ---
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("convexDecomposition")

    # --- 8. Physics material ---
    phys_material = UsdShade.Material.Define(stage, "/World/Environment/PhysicsMaterial")
    physics_mat = UsdPhysics.MaterialAPI.Apply(phys_material.GetPrim())
    physics_mat.GetStaticFrictionAttr().Set(static_friction)
    physics_mat.GetDynamicFrictionAttr().Set(dynamic_friction)
    physics_mat.GetRestitutionAttr().Set(restitution)

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        phys_material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )

    # --- 9. Save ---
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()

    logger.success(f"Saved: {output_path}")
    logger.info(f"  /World/Environment/Mesh           — visual + collision (convexDecomposition)")
    logger.info(f"  /World/Environment/PhysicsMaterial — friction={static_friction}/{dynamic_friction}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "glb_to_usd": glb_to_usd,
        "mesh_to_usd": mesh_to_usd,
        "splat_to_pointcloud": splat_to_pointcloud,
    })
