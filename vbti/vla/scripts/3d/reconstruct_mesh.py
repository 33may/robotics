#!/usr/bin/env python3
"""Reconstruct watertight mesh from Gaussian Splat point cloud using Poisson surface reconstruction."""

import argparse
import numpy as np
import open3d as o3d
import plyfile
import trimesh


def extract_colors_from_sh(vertex_data):
    """
    Extract RGB colors from Spherical Harmonics DC component (f_dc_0, f_dc_1, f_dc_2).

    Args:
        vertex_data: plyfile vertex element data

    Returns:
        RGB array of shape (N, 3) with values in [0, 1]
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))  # Clip to avoid overflow

    rgb = np.column_stack((
        sigmoid(vertex_data['f_dc_0']),
        sigmoid(vertex_data['f_dc_1']),
        sigmoid(vertex_data['f_dc_2'])
    ))
    return rgb


def load_ply_point_cloud(ply_path, estimate_normals=True, normal_radius=0.05, normal_nn=30):
    """
    Load point cloud from Gaussian Splat PLY file.

    Args:
        ply_path: Path to PLY file
        estimate_normals: If True, estimate normals from scratch
        normal_radius: Radius for normal estimation
        normal_nn: Number of neighbors for normal estimation

    Returns:
        Open3D point cloud with positions, colors, and normals
    """
    print(f"Loading point cloud from {ply_path}...")

    # Load with plyfile to access all attributes
    ply = plyfile.PlyData.read(ply_path)
    vertex = ply['vertex']

    # Extract positions
    positions = np.column_stack((vertex['x'], vertex['y'], vertex['z']))

    # Extract colors from SH DC component
    colors = extract_colors_from_sh(vertex)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Loaded {len(pcd.points)} points")
    print(f"Color range: {np.asarray(colors).min():.4f} - {np.asarray(colors).max():.4f}")

    # Handle normals
    if estimate_normals:
        print(f"Estimating normals (radius={normal_radius}, nn={normal_nn})...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=normal_nn
            )
        )
    else:
        # Use normals from PLY if available
        if 'nx' in vertex.dtype.names and 'ny' in vertex.dtype.names and 'nz' in vertex.dtype.names:
            normals = np.column_stack((vertex['nx'], vertex['ny'], vertex['nz']))
            pcd.normals = o3d.utility.Vector3dVector(normals)
            print(f"Loaded normals from PLY")
        else:
            print("No normals in PLY, estimating...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius,
                    max_nn=normal_nn
                )
            )

    return pcd


def reconstruct_poisson(pcd, depth=9, color_transfer=True):
    """
    Reconstruct mesh from point cloud using Poisson surface reconstruction.

    Args:
        pcd: Open3D point cloud with normals
        depth: Octree depth (higher = more detail)
        color_transfer: If True, transfer colors from point cloud to mesh vertices

    Returns:
        Tuple of (mesh, densities)
    """
    print(f"\nReconstructing mesh with Poisson (depth={depth})...")

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=0,
            linear_fit=False
        )

        print(f"✓ Reconstruction successful!")
        print(f"  Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

        # Transfer colors from point cloud to mesh vertices (nearest neighbor)
        if color_transfer and pcd.has_colors():
            print("Transferring colors via nearest-neighbor...")
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            colors = []
            for vertex in np.asarray(mesh.vertices):
                _, indices = pcd_tree.search_knn_vector_3d(vertex, 1)
                colors.append(np.asarray(pcd.colors)[indices[0]])

            mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
            print(f"  Colored {len(mesh.vertices)} vertices")

        return mesh, densities

    except Exception as e:
        print(f"✗ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main(input_ply, output_glb, depth=9, estimate_normals=True):
    """
    Full pipeline: Load PLY → Reconstruct → Export GLB.

    Args:
        input_ply: Input PLY point cloud
        output_glb: Output GLB mesh file
        depth: Poisson octree depth
        estimate_normals: Whether to estimate normals from scratch
    """
    # Load point cloud
    pcd = load_ply_point_cloud(input_ply, estimate_normals=estimate_normals)

    if pcd is None:
        print("Failed to load point cloud")
        return False

    # Reconstruct mesh
    mesh, densities = reconstruct_poisson(pcd, depth=depth, color_transfer=True)

    if mesh is None:
        print("Failed to reconstruct mesh")
        return False

    # Check watertight
    is_watertight = mesh.is_watertight()
    print(f"\nIs watertight: {is_watertight}")

    if not is_watertight:
        print("⚠️ Mesh is NOT manifold. Trying to fix...")
        # Try to fix by removing unreferenced vertices
        mesh.remove_unreferenced_vertices()
        is_watertight = mesh.is_watertight()
        print(f"After cleanup: watertight={is_watertight}")

    # Export to GLB using trimesh (to preserve colors)
    print(f"\nExporting to {output_glb}...")

    # Convert Open3D mesh to trimesh to export with colors
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh_tri.export(output_glb)

    print(f"✓ Mesh saved to {output_glb}")

    return is_watertight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct mesh from Gaussian Splat point cloud")
    parser.add_argument(
        "--input",
        default="vbti/data/so_v1/assets/duck/object_0.ply",
        help="Input PLY point cloud"
    )
    parser.add_argument(
        "--output",
        default="vbti/data/so_v1/assets/duck/object_0_reconstructed.glb",
        help="Output GLB mesh"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=9,
        help="Poisson octree depth (higher = more detail, slower)"
    )
    parser.add_argument(
        "--estimate-normals",
        action="store_true",
        default=True,
        help="Estimate normals from scratch"
    )

    args = parser.parse_args()
    success = main(args.input, args.output, depth=args.depth, estimate_normals=args.estimate_normals)
    exit(0 if success else 1)
