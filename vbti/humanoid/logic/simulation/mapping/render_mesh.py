#!/usr/bin/env python
"""Render a reconstruction mesh (.ply) to PNG views, headless.

Generic viz/audit tool for any bake's nvblox mesh (or any PLY):
top-down plan view + perspective orbit views, saved as PNGs.

Run in the `isaac` env (open3d lives there):
    ~/miniconda3/envs/isaac/bin/python logic/simulation/mapping/render_mesh.py \
        --mesh data/maps/<map>/mesh.ply --out-dir /tmp/mesh_views
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering


def make_views(center: np.ndarray, extent: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Named views: (eye, up). Z-up assumed (nvblox frame)."""
    span = float(max(extent[0], extent[1]))
    z_top = center[2] + span * 1.05
    views = {
        "top": (np.array([center[0], center[1] + 1e-3, z_top]), np.array([0.0, 1.0, 0.0])),
        "iso_ne": (center + np.array([span * 0.55, span * 0.55, extent[2] * 4.0]), np.array([0.0, 0.0, 1.0])),
        "iso_sw": (center + np.array([-span * 0.55, -span * 0.55, extent[2] * 4.0]), np.array([0.0, 0.0, 1.0])),
        "interior": (center + np.array([-span * 0.25, 0.0, extent[2] * 0.4]), np.array([0.0, 0.0, 1.0])),
    }
    return views


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crop-z-max", type=float, default=None,
                    help="drop geometry above this z (slice ceilings off for plan views)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if args.crop_z_max is not None:
        full = mesh.get_axis_aligned_bounding_box()
        crop_box = o3d.geometry.AxisAlignedBoundingBox(
            full.min_bound, np.array([full.max_bound[0], full.max_bound[1], args.crop_z_max]))
        mesh = mesh.crop(crop_box)
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
    print(f"bbox min {bbox.min_bound}, max {bbox.max_bound}, extent {extent}")

    renderer = rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"  # show baked vertex colors as-is
    renderer.scene.add_geometry("mesh", mesh, mat)

    for name, (eye, up) in make_views(center, extent).items():
        renderer.setup_camera(60.0, center.astype(np.float32), eye.astype(np.float32), up.astype(np.float32))
        img = renderer.render_to_image()
        path = out / f"{name}.png"
        o3d.io.write_image(str(path), img)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
