#!/usr/bin/env python3
"""Fix non-manifold GLB mesh for PhysX deformable body using PyMeshLab repair filters."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pymeshlab
import trimesh


def check_watertight(vertices, faces):
    """Check if mesh is watertight using Open3D."""
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    return mesh_o3d.is_watertight()


def fix_manifold(input_path, output_path):
    """
    Fix non-manifold GLB mesh using PyMeshLab repair pipeline.

    Pipeline:
        1. Load GLB with trimesh (preserves materials)
        2. Export to temp PLY for PyMeshLab
        3. Apply repair filters (non-manifold edges, vertices, close holes)
        4. Re-apply original materials
        5. Export fixed GLB

    Args:
        input_path: Path to input GLB mesh
        output_path: Path to save fixed GLB mesh
    """
    print(f"Loading mesh from {input_path}...")

    # Step 1: Load with trimesh to preserve materials
    scene = trimesh.load(input_path)
    if isinstance(scene, trimesh.Scene):
        mesh_name = list(scene.geometry.keys())[0]
        mesh_original = scene.geometry[mesh_name]
    else:
        mesh_original = scene

    n_verts = len(mesh_original.vertices)
    n_faces = len(mesh_original.faces)
    is_watertight = check_watertight(mesh_original.vertices, mesh_original.faces)

    print(f"Original: {n_verts} verts, {n_faces} faces")
    print(f"Is watertight: {is_watertight}")

    if is_watertight:
        print("✓ Already watertight, no repair needed.")
        return True

    # Step 2: Export to temp file for PyMeshLab
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
        tmp_path = tmp.name
    mesh_original.export(tmp_path)

    # Step 3: Load in PyMeshLab and apply repair filters
    print("\nApplying PyMeshLab repair pipeline...")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(tmp_path)

    # Count non-manifold issues before repair
    ms.compute_selection_by_non_manifold_edges_per_face()
    ms.compute_selection_by_non_manifold_per_vertex()

    # Repair pipeline
    print("  1. Removing duplicate vertices...")
    ms.meshing_remove_duplicate_vertices()

    print("  2. Removing duplicate faces...")
    ms.meshing_remove_duplicate_faces()

    print("  3. Removing null/degenerate faces...")
    ms.meshing_remove_null_faces()

    print("  4. Repairing non-manifold edges...")
    ms.meshing_repair_non_manifold_edges()

    print("  5. Repairing non-manifold vertices...")
    ms.meshing_repair_non_manifold_vertices()

    print("  6. Re-orienting faces coherently...")
    ms.meshing_re_orient_faces_coherently()

    print("  7. Closing holes (maxholesize=30000)...")
    ms.meshing_close_holes(maxholesize=30000)

    print("  8. Merging close vertices...")
    ms.meshing_merge_close_vertices()

    print("  9. Removing unreferenced vertices...")
    ms.meshing_remove_unreferenced_vertices()

    # Second pass — closing holes can introduce new non-manifold geometry
    print("  10. Second pass: repair non-manifold edges...")
    ms.meshing_repair_non_manifold_edges()
    print("  11. Second pass: repair non-manifold vertices...")
    ms.meshing_repair_non_manifold_vertices()

    # Get repaired mesh
    repaired = ms.current_mesh()
    vertices_new = repaired.vertex_matrix()
    faces_new = repaired.face_matrix()

    n_verts_new = len(vertices_new)
    n_faces_new = len(faces_new)
    is_watertight_new = check_watertight(vertices_new, faces_new)

    print(f"\nAfter repair: {n_verts_new} verts, {n_faces_new} faces")
    print(f"Is watertight: {is_watertight_new}")

    # Step 4: Build output mesh and transfer texture UVs via nearest-neighbor
    mesh_fixed = trimesh.Trimesh(vertices=vertices_new, faces=faces_new)

    if hasattr(mesh_original, 'visual') and mesh_original.visual is not None:
        try:
            if mesh_original.visual.kind == 'texture' and mesh_original.visual.uv is not None:
                print("  Transferring texture UVs via nearest-neighbor...")
                from scipy.spatial import cKDTree
                tree = cKDTree(mesh_original.vertices)
                _, indices = tree.query(vertices_new)
                uv_new = mesh_original.visual.uv[indices]

                mesh_fixed.visual = trimesh.visual.TextureVisuals(
                    uv=uv_new,
                    material=mesh_original.visual.material,
                )
                print(f"  ✓ Transferred UVs for {len(uv_new)} vertices")
            else:
                mesh_fixed.visual = mesh_original.visual
        except Exception as e:
            print(f"  Warning: could not transfer materials: {e}")

    # Step 5: Export
    mesh_fixed.export(output_path)
    print(f"\n✓ Fixed mesh saved to {output_path}")
    print(f"  Vertices: {n_verts} → {n_verts_new}")
    print(f"  Faces: {n_faces} → {n_faces_new}")

    # Cleanup temp file
    Path(tmp_path).unlink(missing_ok=True)

    return is_watertight_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix non-manifold GLB mesh for PhysX deformable body"
    )
    parser.add_argument(
        "--input",
        default="vbti/data/so_v1/assets/duck/object_0.glb",
        help="Input GLB mesh path",
    )
    parser.add_argument(
        "--output",
        default="vbti/data/so_v1/assets/duck/object_0_fixed.glb",
        help="Output GLB mesh path",
    )

    args = parser.parse_args()
    fix_manifold(args.input, args.output)
