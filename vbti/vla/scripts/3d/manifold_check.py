#!/usr/bin/env python3
"""Check if PLY point cloud can be reconstructed into watertight mesh using Poisson reconstruction."""

import open3d as o3d


def check_manifold(input_path, depth=10):
    """
    Check if point cloud can be reconstructed into watertight mesh.

    Args:
        input_path: Path to PLY point cloud
        depth: Octree depth (higher = more detail, slower)
    """
    print(f"Loading point cloud from {input_path}...")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    n_points = len(pcd.points)
    print(f"Points: {n_points}")
    print(f"Has normals: {pcd.has_normals()}")

    # Estimate normals if missing
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Try Poisson reconstruction
    print(f"\nAttempting Poisson reconstruction (depth={depth})...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=0,
            linear_fit=False
        )

        n_verts = len(mesh.vertices)
        n_faces = len(mesh.triangles)
        is_watertight = mesh.is_watertight()

        print(f"✓ Reconstruction successful!")
        print(f"  Mesh: {n_verts} verts, {n_faces} faces")
        print(f"  Is watertight: {is_watertight}")

        if is_watertight:
            print("✓ Mesh is MANIFOLD — ready for deformable body!")
        else:
            print("⚠️ Mesh is NOT manifold — consider increasing depth parameter")

        return is_watertight

    except Exception as e:
        print(f"✗ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    is_watertight = check_manifold('vbti/data/so_v1/assets/duck/object_0.ply')
    exit(0 if is_watertight else 1)
