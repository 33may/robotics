"""Tests for occupancy_from_depth — robot-derived 2D occupancy (MAY-173 Phase 2.2).

The no-privileged-info occupancy builder: bake TUM poses (cuVSLAM, MAP frame)
∘ static rig mount FK → backproject recorded uint16-mm depth PNGs (USD camera
convention: view -Z, +Y up) → keep the robot's height column → 2D ray-carve
free space → 3-state counts → bool export via `occupancy_io`:

    unknown = BLOCKED (Anton 16-07) — the robot only plans through floor it saw.

Pure brain deps: numpy + PIL + stdlib. GT rows / T_world_cam are NEVER read.
"""

import json
import math

import numpy as np
import pytest
from PIL import Image

from humanoid.logic.oli.reason.mapping.occupancy_io import load_occupancy
from humanoid.logic.simulation.mapping.occupancy_from_depth import (
    GridBuilder,
    backproject_depth,
    build_occupancy,
    detect_heading_offset,
    load_tum_poses,
    planarize,
    t_base_cam,
)

pytestmark = pytest.mark.brain

K = {"fx": 100.0, "fy": 100.0, "cx": 4.0, "cy": 3.0, "width": 8, "height": 6}


def _depth(mm: int, shape=(6, 8)) -> np.ndarray:
    return np.full(shape, mm, dtype=np.uint16)


# ── TUM loader ────────────────────────────────────────────────────────────────


def test_load_tum_identity_and_yaw(tmp_path):
    s = math.sin(math.pi / 4)
    p = tmp_path / "poses.tum"
    p.write_text(
        "1.000000000 2.0 3.0 0.5 0 0 0 1\n"
        f"2.000000000 0 0 0 0 0 {s} {s}\n"  # 90° yaw
    )
    poses = load_tum_poses(p)
    assert len(poses) == 2
    t0, T0 = poses[0]
    assert t0 == pytest.approx(1.0)
    np.testing.assert_allclose(T0[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(T0[:3, 3], [2.0, 3.0, 0.5])
    _, T1 = poses[1]
    # 90° yaw: +X_base maps to +Y_map
    np.testing.assert_allclose(T1[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)


# ── backprojection (USD camera convention) ────────────────────────────────────


def test_backproject_center_pixel_looks_down_minus_z():
    pts = backproject_depth(_depth(2000), K, stride=1)
    # center pixel (u=cx, v=cy) → straight along the view axis: (0, 0, -depth)
    center = pts[np.argmin(np.abs(pts[:, 0]) + np.abs(pts[:, 1]))]
    np.testing.assert_allclose(center, [0.0, 0.0, -2.0], atol=1e-6)
    assert np.all(pts[:, 2] == pytest.approx(-2.0))


def test_backproject_image_axes_signs():
    d = np.zeros((6, 8), dtype=np.uint16)
    d[3, 6] = 1000  # right of center → +X_cam
    d[5, 4] = 1000  # below center → -Y_cam (USD +Y is up)
    pts = backproject_depth(d, K, stride=1)
    assert len(pts) == 2  # zero-depth pixels are dropped
    right = pts[pts[:, 0] > 0.01]
    below = pts[pts[:, 1] < -0.01]
    assert len(right) == 1 and right[0][0] == pytest.approx(0.02)
    assert len(below) == 1 and below[0][1] == pytest.approx(-0.02)


def test_backproject_stride_subsamples():
    assert len(backproject_depth(_depth(1000), K, stride=2)) == 3 * 4


# ── planar projection (glide kinematic prior) ────────────────────────────────


def test_planarize_strips_pitch_roll_and_z():
    """SLAM pitch/z drift tilts the reconstructed floor into the obstacle column
    (radial streak fans). The glide robot is planar by construction — planarize
    keeps only (x, y, yaw)."""
    yaw, pitch = 0.7, math.radians(5)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry
    T[:3, 3] = [1.0, 2.0, 0.08]  # z drift
    P = planarize(T)
    np.testing.assert_allclose(P[:3, 3], [1.0, 2.0, 0.0])       # z zeroed
    np.testing.assert_allclose(P[:3, :3] @ [0, 0, 1], [0, 0, 1], atol=1e-9)
    got_yaw = math.atan2(P[1, 0], P[0, 0])
    assert got_yaw == pytest.approx(yaw, abs=1e-6)               # yaw preserved


# ── heading-offset auto-calibration ──────────────────────────────────────────


def _drive_tum(yaw_offset: float, n: int = 60) -> list:
    """Synthetic forward drive along +X at 1 m/s, 30 Hz; rotations carry a
    constant convention offset vs true heading (the teleop bake ships 180°)."""
    poses = []
    for i in range(n):
        c, s = math.cos(yaw_offset), math.sin(yaw_offset)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[0, 3] = i / 30.0
        poses.append((i / 30.0, T))
    return poses


def test_detect_heading_offset_finds_convention_flip():
    off = detect_heading_offset(_drive_tum(math.pi))
    assert abs(abs(off) - math.pi) < 0.02          # 180° flip detected
    assert abs(detect_heading_offset(_drive_tum(0.0))) < 0.02   # clean rig: ~0
    off90 = detect_heading_offset(_drive_tum(math.pi / 2))
    assert off90 == pytest.approx(math.pi / 2, abs=0.02)


def test_detect_heading_offset_too_little_motion_is_zero():
    # a rig that never moves cannot be calibrated — fall back to trusting it
    still = [(i / 30.0, np.eye(4)) for i in range(60)]
    assert detect_heading_offset(still) == 0.0


# ── mount FK twin ─────────────────────────────────────────────────────────────


def test_t_base_cam_matches_fk_convention():
    T = t_base_cam([0.1, 0.2, 0.65], 0.0)
    # pitch 0: view +X_base → camera -Z axis is +X_base
    np.testing.assert_allclose(T[:3, :3] @ [0, 0, -1], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(T[:3, 3], [0.1, 0.2, 0.65])
    T90 = t_base_cam([0, 0, 1.0], 90.0)
    # pitch 90° down: view straight down -Z_base
    np.testing.assert_allclose(T90[:3, :3] @ [0, 0, -1], [0, 0, -1], atol=1e-9)


# ── GridBuilder: wall / free / unknown semantics ─────────────────────────────


def _wall_view():
    """Camera at (0,0,1) in map frame, level, looking along +X; wall 3 m ahead."""
    T_map_cam = np.eye(4) @ t_base_cam([0.0, 0.0, 1.0], 0.0)
    return T_map_cam, _depth(3000)


def test_wall_occupied_path_free_unknown_blocked():
    b = GridBuilder(bounds=(-1.0, -1.0, 5.0, 1.0), resolution=0.1,
                    zmin=0.15, zmax=1.6, max_range=6.0)
    T, d = _wall_view()
    b.add_view(T, d, K, stride=1)
    g = b.grid()
    assert g.is_occupied(3.05, 0.0)          # the wall
    assert not g.is_occupied(1.5, 0.0)       # carved free en route
    assert g.is_occupied(4.5, 0.0)           # behind the wall = unknown = blocked
    assert g.is_occupied(-0.5, 0.0)          # behind the camera = unknown = blocked
    assert g.is_occupied(1.5, 0.9)           # outside the FOV wedge = unknown


def test_floor_hits_carve_free_without_occupying():
    b = GridBuilder(bounds=(-1.0, -1.0, 1.0, 1.0), resolution=0.1,
                    zmin=0.15, zmax=1.6, max_range=6.0)
    # camera 1 m up looking straight down → all hits are floor (z≈0 < zmin)
    T_map_cam = t_base_cam([0.0, 0.0, 1.0], 90.0)
    b.add_view(T_map_cam, _depth(1000), K, stride=1)
    g = b.grid()
    assert not g.is_occupied(0.0, 0.0)       # floor sighting = free, not obstacle
    assert g.is_occupied(0.9, 0.9)           # never observed = blocked


def test_hits_beyond_max_range_ignored():
    b = GridBuilder(bounds=(-1.0, -1.0, 5.0, 1.0), resolution=0.1,
                    zmin=0.15, zmax=1.6, max_range=2.0)
    T, d = _wall_view()  # wall at 3 m > max_range
    b.add_view(T, d, K, stride=1)
    g = b.grid()
    assert g.is_occupied(3.05, 0.0)          # not a hit — but unknown = blocked
    assert g.is_occupied(1.5, 0.0)           # no carving from over-range rays


# ── full flow: dump + TUM → occupancy artifact ───────────────────────────────


def test_build_occupancy_end_to_end(tmp_path):
    dump = tmp_path / "dump"
    (dump / "frames" / "head_depth").mkdir(parents=True)
    rig = {"cameras": {"head": {"pos_base": [0.0, 0.0, 1.0], "pitch_down_deg": 0.0,
                                "intrinsics": K}}, "rgbd": ["head"]}
    (dump / "rig.json").write_text(json.dumps(rig))
    stamp_ns = 1_000_000_000
    Image.fromarray(_depth(3000), mode="I;16").save(
        dump / "frames" / "head_depth" / f"{stamp_ns:019d}.png")
    tum = tmp_path / "slam_poses.tum"
    tum.write_text("1.000000000 0 0 0 0 0 0 1\n")

    out = tmp_path / "occ"
    grid = build_occupancy(dump, tum, out, streams=("head",),
                           resolution=0.1, stride=1, frame_step=1)
    assert grid.is_occupied(3.05, 0.0) and not grid.is_occupied(1.5, 0.0)

    loaded = load_occupancy(str(out))     # round-trips through occupancy_io
    assert loaded.resolution == pytest.approx(0.1)
    np.testing.assert_array_equal(loaded.grid, grid.grid)
    assert (out / "preview.png").exists()


def test_build_occupancy_never_reads_gt(tmp_path):
    """poses.jsonl (GT) may be absent entirely — the builder must not touch it."""
    dump = tmp_path / "dump"
    (dump / "frames" / "head_depth").mkdir(parents=True)
    rig = {"cameras": {"head": {"pos_base": [0.0, 0.0, 1.0], "pitch_down_deg": 0.0,
                                "intrinsics": K}}, "rgbd": ["head"]}
    (dump / "rig.json").write_text(json.dumps(rig))
    Image.fromarray(_depth(2000), mode="I;16").save(
        dump / "frames" / "head_depth" / f"{1_000_000_000:019d}.png")
    tum = tmp_path / "t.tum"
    tum.write_text("1.000000000 0 0 0 0 0 0 1\n")
    grid = build_occupancy(dump, tum, tmp_path / "o", streams=("head",),
                           resolution=0.1, stride=1, frame_step=1)
    assert grid.is_occupied(2.05, 0.0)
