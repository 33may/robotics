#!/usr/bin/env python3
"""Interactive mesh cleaner using Polyscope + Open3D.

Features:
  - Remove small connected components (floating blobs)
  - Statistical outlier removal
  - Oriented bounding box crop with rotation + axis gizmo
  - Preview: see result of all filters + OBB crop without saving
  - On save: applies box rotation to mesh so scene comes out axis-aligned

Usage:
    python vbti/utils/clean_mesh.py \
        --mesh vbti/data/vbti_table/gs/milo_output/mesh_learnable_sdf.ply
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import polyscope as ps
import polyscope.imgui as psim


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear [0,1]."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def euler_to_rotation(rx, ry, rz):
    """Euler angles (degrees) → 3x3 rotation matrix. Order: Z * Y * X."""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def obb_corners(center, half_ext, rotation):
    """Compute 8 corners of an oriented bounding box."""
    axes = rotation @ np.diag(half_ext)
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(center + axes[:, 0] * sx + axes[:, 1] * sy + axes[:, 2] * sz)
    return np.array([
        corners[0], corners[1], corners[3], corners[2],  # bottom
        corners[4], corners[5], corners[7], corners[6],  # top
    ])


def slider_with_input(label, value, v_min, v_max, width_slider=180, width_input=80):
    """Slider + InputFloat on same line."""
    psim.PushItemWidth(width_slider)
    changed_s, value = psim.SliderFloat(f"##{label}_slider", value, v_min, v_max)
    psim.PopItemWidth()
    psim.SameLine()
    psim.PushItemWidth(width_input)
    changed_i, value = psim.InputFloat(f"##{label}_input", value, 0, 0, "%.3f")
    psim.PopItemWidth()
    psim.SameLine()
    psim.Text(label)
    return changed_s or changed_i, value


class MeshCleaner:
    def __init__(self, mesh_path: str):
        self.mesh_path = mesh_path
        self.original = o3d.io.read_triangle_mesh(mesh_path)
        self.original.compute_vertex_normals()

        self.verts = np.asarray(self.original.vertices)
        self.faces = np.asarray(self.original.triangles)
        self.colors = np.asarray(self.original.vertex_colors)

        # Convert to linear for Polyscope display only; keep original mesh sRGB for saving
        self.colors = srgb_to_linear(self.colors)
        # Store linear colors separately — do NOT modify self.original.vertex_colors
        # so PLY save keeps the original sRGB values

        bounds_min = self.verts.min(axis=0)
        bounds_max = self.verts.max(axis=0)
        self.scene_center = (bounds_min + bounds_max) / 2
        self.scene_extent = (bounds_max - bounds_min) / 2
        print(f"Loaded: {len(self.verts)} verts, {len(self.faces)} faces")
        print(f"Bounds: {bounds_min} → {bounds_max}")

        # --- Cleaning parameters ---
        self.min_component_ratio = 0.01
        self.use_component_filter = True

        self.use_sor = False
        self.sor_neighbors = 20.0
        self.sor_std = 2.0

        # OBB
        self.use_bbox = False
        self.obb_center = list(self.scene_center)
        self.obb_half = list(self.scene_extent + 1.0)
        self.obb_rot = [0.0, 0.0, 0.0]
        self._obb_base_center = np.array(self.obb_center, dtype=float)
        self._obb_local_offset = [0.0, 0.0, 0.0]
        self.total_rotation = np.eye(3)

        # Cleaned state
        self.clean_mesh = None
        self.clean_verts = self.verts
        self.clean_faces = self.faces
        self.clean_colors = self.colors
        self.removed_count = 0
        self.status_msg = "Ready"
        self._prev_obb = None
        self._previewing = False

    def _obb_rotation_matrix(self):
        return euler_to_rotation(*self.obb_rot)

    def _apply_obb_crop(self, mesh):
        """Crop mesh vertices to OBB (no rotation, just hide outside vertices)."""
        R = self._obb_rotation_matrix()
        center = np.array(self.obb_center)
        half = np.array(self.obb_half)

        verts = np.asarray(mesh.vertices)
        verts_local = (verts - center) @ R
        mask = np.all((verts_local >= -half) & (verts_local <= half), axis=1)

        faces = np.asarray(mesh.triangles)
        face_mask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
        mesh.triangles = o3d.utility.Vector3iVector(faces[face_mask])
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh

    def apply_cleaning(self):
        """Apply component filter + SOR + OBB crop. No rotation, just filtering."""
        self.status_msg = "Processing..."
        mesh = o3d.geometry.TriangleMesh(self.original)

        if self.use_component_filter and len(mesh.triangles) > 0:
            triangle_clusters, cluster_n_triangles, _ = (
                mesh.cluster_connected_triangles()
            )
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)

            min_triangles = int(self.min_component_ratio * len(self.original.triangles))
            small_clusters = cluster_n_triangles < max(min_triangles, 10)
            triangles_to_remove = small_clusters[triangle_clusters]
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()

        if self.use_sor and len(mesh.vertices) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            _, inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors=int(self.sor_neighbors),
                std_ratio=self.sor_std,
            )
            mesh = mesh.select_by_index(inlier_idx)

        if self.use_bbox:
            mesh = self._apply_obb_crop(mesh)

        mesh.compute_vertex_normals()
        self.clean_mesh = mesh
        self.clean_verts = np.asarray(mesh.vertices)
        self.clean_faces = np.asarray(mesh.triangles)
        raw_colors = (
            np.asarray(mesh.vertex_colors)
            if mesh.has_vertex_colors()
            else np.ones((len(self.clean_verts), 3)) * 0.7
        )
        # Convert to linear for Polyscope display only
        self.clean_colors = srgb_to_linear(raw_colors)
        self.removed_count = len(self.verts) - len(self.clean_verts)
        self.status_msg = f"Done — removed {self.removed_count:,} verts"

    def cut_obb(self):
        """Crop mesh to OBB, rotate mesh so OBB becomes axis-aligned, update baseline."""
        R = self._obb_rotation_matrix()
        center = np.array(self.obb_center)
        half = np.array(self.obb_half)

        mesh = o3d.geometry.TriangleMesh(self.original)
        verts = np.asarray(mesh.vertices)
        verts_local = (verts - center) @ R

        mask = np.all((verts_local >= -half) & (verts_local <= half), axis=1)

        faces = np.asarray(mesh.triangles)
        face_mask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
        mesh.triangles = o3d.utility.Vector3iVector(faces[face_mask])
        mesh.remove_unreferenced_vertices()

        new_verts = (np.asarray(mesh.vertices) - center) @ R
        mesh.vertices = o3d.utility.Vector3dVector(new_verts)
        mesh.compute_vertex_normals()

        self.total_rotation = R.T @ self.total_rotation

        self.original = mesh
        self.verts = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        raw_colors = (
            np.asarray(mesh.vertex_colors)
            if mesh.has_vertex_colors()
            else np.ones((len(self.verts), 3)) * 0.7
        )
        self.colors = srgb_to_linear(raw_colors)
        self.clean_mesh = mesh
        self.clean_verts = self.verts
        self.clean_faces = self.faces
        self.clean_colors = self.colors

        bounds_min = self.verts.min(axis=0)
        bounds_max = self.verts.max(axis=0)
        self.scene_center = (bounds_min + bounds_max) / 2
        self.scene_extent = (bounds_max - bounds_min) / 2
        self.obb_center = list(self.scene_center)
        self.obb_half = list(self.scene_extent + 1.0)
        self.obb_rot = [0.0, 0.0, 0.0]
        self._obb_base_center = np.array(self.obb_center, dtype=float)
        self._obb_local_offset = [0.0, 0.0, 0.0]

        self.removed_count = 0
        self.status_msg = f"Cut + rotated — {len(self.verts):,} verts"

    def _remove_obb_structures(self):
        for n in ["obb_edges", "obb_faces", "obb_axis_x", "obb_axis_y", "obb_axis_z"]:
            ps.remove_curve_network(n, error_if_absent=False)
            ps.remove_surface_mesh(n, error_if_absent=False)

    def update_obb_preview(self):
        """Show OBB as wireframe + transparent faces + RGB axis arrows."""
        if not self.use_bbox:
            self._remove_obb_structures()
            ps.set_transparency_mode("none")
            return

        ps.set_transparency_mode("simple")

        R = self._obb_rotation_matrix()
        center = np.array(self.obb_center)
        half = np.array(self.obb_half)
        corners = obb_corners(center, half, R)

        # Wireframe edges
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ])
        cn = ps.register_curve_network("obb_edges", corners, edges)
        cn.set_radius(0.02, relative=False)
        cn.set_color((1.0, 0.4, 0.1))

        # Transparent faces
        box_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 1], [1, 4, 5],
            [2, 6, 3], [3, 6, 7],
            [0, 3, 4], [4, 3, 7],
            [1, 5, 2], [2, 5, 6],
        ])
        sm = ps.register_surface_mesh("obb_faces", corners, box_faces)
        sm.set_color((1.0, 0.5, 0.2))
        sm.set_transparency(0.15)

        # Axis arrows at center (RGB = XYZ)
        arrow_len = min(half) * 0.5
        axis_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        axis_names = ["obb_axis_x", "obb_axis_y", "obb_axis_z"]
        for i, (name, color) in enumerate(zip(axis_names, axis_colors)):
            tip = center + R[:, i] * arrow_len
            cn = ps.register_curve_network(name, np.array([center, tip]), np.array([[0, 1]]))
            cn.set_radius(0.08, relative=False)
            cn.set_color(color)

    def update_visualization(self):
        """Re-register mesh in polyscope."""
        if len(self.clean_faces) > 0:
            m = ps.register_surface_mesh(
                "mesh", self.clean_verts, self.clean_faces, smooth_shade=True
            )
            m.add_color_quantity(
                "vertex_colors", self.clean_colors,
                defined_on="vertices", enabled=True,
            )
        else:
            ps.remove_all_structures()

    def gui_callback(self):
        """imgui panel."""
        psim.SetNextWindowSize((460, 800))
        psim.Begin("Mesh Cleaner", True)

        psim.Text(f"Original: {len(self.verts):,} verts / {len(self.faces):,} faces")
        psim.Text(f"Cleaned:  {len(self.clean_verts):,} verts / {len(self.clean_faces):,} faces")
        psim.Text(f"Removed:  {self.removed_count:,} verts")
        psim.TextColored((0.4, 1.0, 0.4, 1.0), self.status_msg)
        psim.Separator()

        # --- Connected components ---
        _, self.use_component_filter = psim.Checkbox(
            "Remove small components", self.use_component_filter
        )
        if self.use_component_filter:
            _, self.min_component_ratio = slider_with_input(
                "Min size (ratio)", self.min_component_ratio, 0.0001, 0.1
            )
            psim.Text(
                f"  = {int(self.min_component_ratio * len(self.original.triangles)):,} triangles"
            )

        psim.Separator()

        # --- Statistical outlier removal ---
        _, self.use_sor = psim.Checkbox("Statistical outlier removal", self.use_sor)
        if self.use_sor:
            _, self.sor_neighbors = slider_with_input("Neighbors", self.sor_neighbors, 5, 100)
            _, self.sor_std = slider_with_input("Std ratio", self.sor_std, 0.1, 5.0)

        psim.Separator()

        # --- OBB crop ---
        _, self.use_bbox = psim.Checkbox("Oriented bounding box", self.use_bbox)
        if self.use_bbox:
            ext = max(np.linalg.norm(self.scene_extent) * 2, 50)
            R = self._obb_rotation_matrix()

            psim.Text("Rotation (degrees):")
            rot_changed = False
            for i, label in enumerate(["rot X", "rot Y", "rot Z"]):
                c, self.obb_rot[i] = slider_with_input(label, self.obb_rot[i], -180, 180)
                rot_changed = rot_changed or c
            if rot_changed:
                R = self._obb_rotation_matrix()
                offset = np.array(self._obb_local_offset)
                new_center = self._obb_base_center + R @ offset
                self.obb_center = list(new_center)

            psim.Text("Center (world):")
            world_changed = False
            for i, label in enumerate(["cx", "cy", "cz"]):
                c, self.obb_center[i] = slider_with_input(label, self.obb_center[i], -ext, ext)
                world_changed = world_changed or c
            if world_changed:
                offset = np.array(self._obb_local_offset)
                self._obb_base_center = np.array(self.obb_center) - R @ offset

            psim.Text("Move (local axes):")
            local_changed = False
            for i, label in enumerate(["local X", "local Y", "local Z"]):
                c, self._obb_local_offset[i] = slider_with_input(
                    label, self._obb_local_offset[i], -ext, ext
                )
                local_changed = local_changed or c
            if local_changed:
                offset = np.array(self._obb_local_offset)
                new_center = self._obb_base_center + R @ offset
                self.obb_center = list(new_center)

            psim.Text("Half-extents:")
            _, self.obb_half[0] = slider_with_input("hx", self.obb_half[0], 0.1, ext)
            _, self.obb_half[1] = slider_with_input("hy", self.obb_half[1], 0.1, ext)
            _, self.obb_half[2] = slider_with_input("hz", self.obb_half[2], 0.1, ext)

            if psim.Button("Cut to OBB"):
                self.cut_obb()
                self.use_bbox = False
                self._remove_obb_structures()
                ps.set_transparency_mode("none")
                self._previewing = False
                self.update_visualization()

        # Live OBB preview
        cur_obb = (
            self.use_bbox, tuple(self.obb_center),
            tuple(self.obb_half), tuple(self.obb_rot),
        )
        if cur_obb != self._prev_obb:
            self.update_obb_preview()
            self._prev_obb = cur_obb

        psim.Separator()

        # --- Action buttons ---
        preview_label = "Back to full" if self._previewing else "Preview"
        if psim.Button(preview_label):
            if self._previewing:
                # Restore full mesh
                self._previewing = False
                self.clean_verts = self.verts
                self.clean_faces = self.faces
                self.clean_colors = self.colors
                self.clean_mesh = None
                self.removed_count = 0
                self.status_msg = "Back to original"
                self.update_visualization()
            else:
                # Show preview: apply all filters + OBB crop (no rotation)
                self._previewing = True
                self.apply_cleaning()
                self.update_visualization()

        psim.SameLine()
        if psim.Button("Reset"):
            self.clean_mesh = None
            self.clean_verts = self.verts
            self.clean_faces = self.faces
            self.clean_colors = self.colors
            self.removed_count = 0
            self._previewing = False
            self.status_msg = "Reset to original"
            self.update_visualization()

        psim.SameLine()
        if psim.Button("Save"):
            self.save()

        psim.End()

    def save(self):
        out_dir = Path(self.mesh_path).parent
        out_path = out_dir / "mesh_cleaned.ply"
        mesh_to_save = self.clean_mesh if self.clean_mesh else self.original

        # If OBB is active, apply rotation to saved mesh (same as Cut but on the clean mesh)
        if self.use_bbox and not np.allclose(self.obb_rot, [0, 0, 0]):
            mesh_to_save = o3d.geometry.TriangleMesh(mesh_to_save)
            R = self._obb_rotation_matrix()
            center = np.array(self.obb_center)
            verts = np.asarray(mesh_to_save.vertices)
            new_verts = (verts - center) @ R
            mesh_to_save.vertices = o3d.utility.Vector3dVector(new_verts)
            mesh_to_save.compute_vertex_normals()
            save_rotation = R.T @ self.total_rotation
        else:
            save_rotation = self.total_rotation

        o3d.io.write_triangle_mesh(str(out_path), mesh_to_save)
        self.status_msg = f"Saved to {out_path.name}"
        print(f"\nSaved: {out_path}")
        print(f"  {len(np.asarray(mesh_to_save.vertices)):,} verts, {len(np.asarray(mesh_to_save.triangles)):,} faces")
        if not np.allclose(save_rotation, np.eye(3)):
            print(f"  Total rotation applied:\n{save_rotation}")

    def run(self):
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")

        self.clean_verts = self.verts
        self.clean_faces = self.faces
        self.clean_colors = self.colors
        self.update_visualization()

        ps.set_user_callback(self.gui_callback)
        ps.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive mesh cleaner")
    parser.add_argument("--mesh", required=True, help="Path to mesh PLY")
    args = parser.parse_args()

    cleaner = MeshCleaner(args.mesh)
    cleaner.run()
