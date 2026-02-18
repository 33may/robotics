#!/usr/bin/env python3
"""3D asset format conversion utilities."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

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
    print(f"Opacity filter: {mask.sum()}/{n_total} kept (threshold={opacity_threshold})")

    # --- Filter: scale (remove floaters) ---
    if scale_percentile < 100.0:
        scale_cutoff = np.percentile(max_scale[mask], scale_percentile)
        scale_mask = max_scale <= scale_cutoff
        mask = mask & scale_mask
        print(f"Scale filter: {mask.sum()}/{n_total} kept (percentile={scale_percentile}, cutoff={scale_cutoff:.6f})")

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
        print(f"Outlier removal: {len(inlier_idx)}/{len(positions)} kept (nb={outlier_nb}, std={outlier_std})")

    print(f"Final point cloud: {len(pcd.points)} points")
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
    print(f"Saved colored point cloud to {output_path}")


def splat_to_mesh(input_path: str, output_path: str,
                  opacity_threshold: float = 0.5,
                  scale_percentile: float = 95.0,
                  outlier_nb: int = 20, outlier_std: float = 2.0,
                  poisson_depth: int = 10,
                  normal_radius: float = 0.014,
                  density_quantile: float = 0.05,
                  target_faces: int = 0):
    """Step 2: Convert a filtered 3DGS PLY to a vertex-colored mesh via Poisson.

    Run splat_to_pointcloud first to verify filtering params look good.
    """
    pcd = _load_and_filter_splat(input_path, opacity_threshold, scale_percentile,
                                 outlier_nb, outlier_std)

    # --- Normal estimation ---
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=100)
    print("Normals estimated and oriented")

    # --- Poisson reconstruction ---
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )
    print(f"Raw Poisson mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # --- Remove low-density (hallucinated) regions ---
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))
    print(f"After density cleanup: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # --- Decimation ---
    if target_faces > 0 and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        print(f"After decimation: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
    print(f"Saved to {output_path}")


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
    print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")

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
        print(f"UVs: {uvs.shape}")

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
            print(f"Texture saved: {tex_path}")

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
        print("Material bound to mesh")

    # Physics
    if rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(world.GetPrim())
        print("Rigid body applied to /World")

    if collision != "none":
        UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
        mesh_collision.CreateApproximationAttr().Set(collision)
        print(f"Collision: {collision}")

    stage.GetRootLayer().Save()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "glb_to_usd": glb_to_usd,
        "splat_to_pointcloud": splat_to_pointcloud,
        "splat_to_mesh": splat_to_mesh,
    })
