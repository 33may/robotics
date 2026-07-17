"""Tests for mesh_to_occupancy — occupancy grid from a dense reconstruction mesh.

The GUI-free replacement for both the Isaac omap GUI bake and the rejected
nvblox occupancy line: a fused TSDF mesh (map frame) is sliced into the robot's
height column and rasterized into the `occupancy_io` artifact.

Recipe (researched 2026-07-17, mirrors what pointcloud_to_grid / octomap
multi-layer projection / grid_map_pcl compute):

  occupied = any mesh surface in the column [floor+band_lo, floor+band_hi]
  free     = floor surface observed AND column empty
  unknown  = neither  →  BLOCKED (Anton 16-07: plan only through observed floor)

Core is pure numpy (brain-marked tests); the PLY loader lives only in the CLI.
"""

import numpy as np
import pytest

from humanoid.logic.simulation.mapping.mesh_to_occupancy import (
    binary_close,
    build_grid,
    detect_floor_z,
    sample_surface,
)

pytestmark = pytest.mark.brain


# ── helpers: tiny synthetic meshes ────────────────────────────────────────────


def quad(p0, p1, p2, p3):
    """Two triangles covering the quad p0-p1-p2-p3 (ccw)."""
    verts = np.array([p0, p1, p2, p3], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    return verts, tris


def merge(*meshes):
    verts, tris, off = [], [], 0
    for v, t in meshes:
        verts.append(v)
        tris.append(t + off)
        off += len(v)
    return np.concatenate(verts), np.concatenate(tris)


def floor_quad(x0, y0, x1, y1, z=0.0):
    return quad([x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z])


def wall_quad(x, y0, y1, z0, z1):
    """Vertical wall in the yz plane at fixed x."""
    return quad([x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1])


# ── sample_surface ────────────────────────────────────────────────────────────


class TestSampleSurface:
    def test_covers_large_triangle_densely(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        tris = np.array([[0, 1, 2]])
        pts = sample_surface(verts, tris, spacing=0.05)
        # every sample on the triangle plane, inside bounds
        assert np.allclose(pts[:, 2], 0.0)
        assert (pts[:, 0] >= -1e-9).all() and (pts[:, 1] >= -1e-9).all()
        # dense: no point of the triangle is farther than `spacing` from a sample.
        # probe a strict interior lattice
        probe = np.array([[x, y, 0.0] for x in np.arange(0.1, 0.8, 0.07)
                          for y in np.arange(0.1, 0.8, 0.07) if x + y < 0.9])
        d = np.linalg.norm(probe[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
        assert d.max() < 0.05

    def test_small_triangle_kept_as_is(self):
        verts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
        tris = np.array([[0, 1, 2]])
        pts = sample_surface(verts, tris, spacing=0.05)
        assert len(pts) >= 3  # at least its corners


# ── detect_floor_z ────────────────────────────────────────────────────────────


class TestDetectFloor:
    def test_finds_dominant_horizontal_plane(self):
        scene = merge(
            floor_quad(0, 0, 4, 4, z=-1.0),          # big floor
            wall_quad(2.0, 0, 4, -1.0, 1.0),          # wall (vertical spread)
            floor_quad(1, 1, 1.5, 1.5, z=-0.5),       # small shelf plane
        )
        pts = sample_surface(*scene, spacing=0.05)
        assert abs(detect_floor_z(pts) - (-1.0)) < 0.03

    def test_floor_not_fooled_by_high_surface_area(self):
        # ceiling-like plane ABOVE floor must not win even if same size:
        # floor = lowest dominant peak, not global max
        scene = merge(
            floor_quad(0, 0, 4, 4, z=0.0),
            floor_quad(0, 0, 4, 4, z=2.5),
        )
        pts = sample_surface(*scene, spacing=0.05)
        assert abs(detect_floor_z(pts) - 0.0) < 0.03


# ── build_grid ────────────────────────────────────────────────────────────────


def room_scene():
    """4x4 m floor at z=0, wall across x=2 for y in [0,4], z 0..2."""
    return merge(
        floor_quad(0, 0, 4, 4, z=0.0),
        wall_quad(2.0, 0.0, 4.0, 0.0, 2.0),
    )


class TestBuildGrid:
    def make(self, scene=None, **kw):
        verts, tris = scene if scene is not None else room_scene()
        pts = sample_surface(verts, tris, spacing=0.025)
        kw.setdefault("resolution", 0.05)
        kw.setdefault("band", (0.3, 1.4))
        return build_grid(pts, floor_z=0.0, **kw)

    def test_georeferencing(self):
        grid, stats = self.make()
        assert grid.resolution == 0.05
        # origin at (or just below) the mesh min corner
        assert grid.origin[0] <= 0.0 and grid.origin[1] <= 0.0
        assert grid.origin[0] > -0.1 and grid.origin[1] > -0.1

    def test_wall_occupied_floor_free_unknown_blocked(self):
        grid, stats = self.make()
        # wall cells blocked
        r, c = grid.world_to_cell(2.0, 2.0)
        assert grid.grid[r, c]
        # observed floor away from wall = free
        r, c = grid.world_to_cell(1.0, 1.0)
        assert not grid.grid[r, c]
        r, c = grid.world_to_cell(3.0, 3.0)
        assert not grid.grid[r, c]

    def test_unknown_is_blocked_outside_observed_floor(self):
        verts, tris = floor_quad(0, 0, 1, 1, z=0.0)  # tiny floor patch only
        pts = sample_surface(verts, tris, spacing=0.025)
        grid, _ = build_grid(pts, floor_z=0.0, resolution=0.05, band=(0.3, 1.4),
                             bounds=(0.0, 0.0, 2.0, 2.0))
        r, c = grid.world_to_cell(0.5, 0.5)
        assert not grid.grid[r, c]          # observed floor -> free
        r, c = grid.world_to_cell(1.5, 1.5)
        assert grid.grid[r, c]              # never observed -> blocked

    def test_band_excludes_overhead_and_fringe(self):
        scene = merge(
            floor_quad(0, 0, 4, 4, z=0.0),
            floor_quad(1.0, 1.0, 3.0, 3.0, z=2.0),    # overhead surface above band
            floor_quad(3.2, 3.2, 3.8, 3.8, z=0.05),   # low fringe below band
        )
        grid, _ = self.make(scene=scene)
        r, c = grid.world_to_cell(2.0, 2.0)
        assert not grid.grid[r, c]  # overhead ignored
        r, c = grid.world_to_cell(3.5, 3.5)
        assert not grid.grid[r, c]  # fringe ignored, floor below it seen -> free

    def test_min_samples_suppresses_specks(self):
        verts, tris = room_scene()
        pts = sample_surface(verts, tris, spacing=0.025)
        # one stray point mid-room inside the band
        stray = np.array([[1.0, 3.0, 0.8]])
        grid, _ = build_grid(np.vstack([pts, stray]), floor_z=0.0,
                             resolution=0.05, band=(0.3, 1.4), min_occ_samples=3)
        r, c = grid.world_to_cell(1.0, 3.0)
        assert not grid.grid[r, c]

    def test_stats_report(self):
        _, stats = self.make()
        for key in ("occupied_cells", "free_cells", "unknown_cells", "floor_z"):
            assert key in stats

    def test_free_close_fills_observation_stripes(self):
        # floor with thin unobserved stripes (TSDF fringe gaps at range):
        # planning-lethal if left unknown=blocked; free_close must seal them
        strips = merge(*[floor_quad(0, y, 4.0, y + 0.15, z=0.0)
                         for y in np.arange(0.0, 4.0, 0.25)])  # 0.10m gaps
        pts = sample_surface(*strips, spacing=0.025)
        grid, _ = build_grid(pts, floor_z=0.0, resolution=0.05, band=(0.3, 1.4),
                             free_close=2)
        # interior gap cells are free, not blocked
        r, c = grid.world_to_cell(2.0, 0.20)  # inside a gap stripe
        assert not grid.grid[r, c]

    def test_traj_clear_frees_unobserved_start(self):
        # robot start: floor beneath it never observed (head looks forward) ->
        # unknown -> blocked. Physical presence must clear it.
        verts, tris = floor_quad(1.0, 0.0, 4.0, 4.0, z=0.0)  # floor seen only ahead
        pts = sample_surface(verts, tris, spacing=0.025)
        traj = np.array([[0.5, 2.0], [1.5, 2.0], [2.5, 2.0]])
        grid, stats = build_grid(pts, floor_z=0.0, resolution=0.05, band=(0.3, 1.4),
                                 bounds=(0.0, 0.0, 4.0, 4.0),
                                 traj_xy=traj, traj_radius=0.25)
        r, c = grid.world_to_cell(0.5, 2.0)   # start, outside observed floor
        assert not grid.grid[r, c]
        assert stats["traj_cleared_cells"] > 0

    def test_traj_clear_overrides_phantom_obstacle_and_reports(self):
        scene = merge(
            floor_quad(0, 0, 4, 4, z=0.0),
            floor_quad(1.9, 1.9, 2.1, 2.1, z=0.8),  # phantom surface in band ON the path
        )
        pts = sample_surface(*scene, spacing=0.025)
        traj = np.array([[2.0, 2.0]])
        grid, stats = build_grid(pts, floor_z=0.0, resolution=0.05, band=(0.3, 1.4),
                                 min_occ_samples=2, traj_xy=traj, traj_radius=0.25)
        r, c = grid.world_to_cell(2.0, 2.0)
        assert not grid.grid[r, c]  # robot was here -> cannot be an obstacle
        assert stats["traj_overrode_occupied_cells"] > 0

    def test_free_close_never_overrides_obstacles(self):
        scene = merge(
            floor_quad(0, 0, 4, 4, z=0.0),
            wall_quad(2.0, 0.0, 4.0, 0.0, 2.0),
        )
        pts = sample_surface(*scene, spacing=0.025)
        grid, _ = build_grid(pts, floor_z=0.0, resolution=0.05, band=(0.3, 1.4),
                             free_close=3)
        r, c = grid.world_to_cell(2.0, 2.0)
        assert grid.grid[r, c]  # wall survives aggressive free closing


# ── binary_close ──────────────────────────────────────────────────────────────


class TestBinaryClose:
    def test_seals_one_cell_pinhole_in_wall(self):
        occ = np.zeros((9, 9), bool)
        occ[4, :] = True
        occ[4, 4] = False  # pinhole
        closed = binary_close(occ, radius=1)
        assert closed[4, 4]

    def test_does_not_grow_solid_regions(self):
        occ = np.zeros((9, 9), bool)
        occ[4, 4] = True
        closed = binary_close(occ, radius=1)
        assert closed.sum() == 1  # dilate then erode returns the single cell
