"""Tests for register_slam_to_mesh — GT-free bake-time alignment into M (MAY-173).

Two layers:
  * synthetic: a known SE(2) transform (with and without reflection) must be
    recovered exactly by the Umeyama fit, and the derived yaw/vertex/traj
    transforms must be self-consistent with R.
  * reference bake: the fit run against the real teleop_v1_demo artifacts must
    reproduce the numbers validated on 17-07 (mirrored=True, residual mean
    0.170 m over 320 timestamp-matched pairs) — this is the acceptance gate for
    the bake_map stage-7 implementation.

Pure numpy — runs in the `brain` env.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from humanoid.logic.simulation.mapping.register_slam_to_mesh import (
    fit_se2,
    load_tum,
    match_by_stamp,
    registration_dict,
    start_pose,
    transform_traj,
    transform_vertices,
    transform_xy,
    yaw_from_quat,
)

pytestmark = pytest.mark.brain

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "logic/simulation/mapping/register_slam_to_mesh.py"
REF_BAKE = REPO / "data/maps/teleop_v1_demo/2026-07-16_14-25-28_teleop_v1_demo_bag"
REF_SLAM = REF_BAKE / "pycuvslam_map/slam_poses.tum"
REF_CUSFM = REF_BAKE / "cusfm/pose_graph/vehicle_pose.tum"


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _make_pair(theta, t, mirrored, n=50, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    src = rng.uniform(-10, 10, size=(n, 2))
    R = _rot(theta) @ (np.diag([1.0, -1.0]) if mirrored else np.eye(2))
    dst = src @ R.T + np.asarray(t)
    if noise:
        dst = dst + rng.normal(0, noise, size=dst.shape)
    return src, dst, R


# ── synthetic recovery ────────────────────────────────────────────────────────


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("theta,t", [(0.0, (0.0, 0.0)), (0.7, (3.0, -2.0)),
                                     (-2.9, (-11.0, 0.4))])
def test_fit_recovers_known_transform_exactly(theta, t, mirrored):
    src, dst, R = _make_pair(theta, t, mirrored)
    fit = fit_se2(src, dst)
    assert fit["mirrored"] == mirrored
    np.testing.assert_allclose(fit["R"], R, atol=1e-12)
    np.testing.assert_allclose(fit["t"], t, atol=1e-11)
    assert fit["residuals"].max() < 1e-10
    # theta convention: R = Rot(theta) @ diag(1, -1 if mirrored else 1)
    assert math.isclose(math.atan2(math.sin(theta), math.cos(theta)),
                        fit["theta_rad"], abs_tol=1e-12)


def test_fit_with_noise_reports_residual_stats():
    src, dst, _ = _make_pair(0.3, (1.0, 2.0), True, n=500, noise=0.05)
    fit = fit_se2(src, dst)
    reg = registration_dict(fit, len(src))
    assert reg["mirrored"] is True
    assert 0.02 < reg["residual_mean_m"] < 0.12
    assert reg["residual_mean_m"] <= reg["residual_p95_m"] <= reg["residual_max_m"]
    assert reg["n"] == 500
    assert len(reg["R"]) == 2 and len(reg["t"]) == 2 and "theta_rad" in reg


def test_fit_needs_three_pairs():
    with pytest.raises(ValueError):
        fit_se2(np.zeros((2, 2)), np.zeros((2, 2)))


# ── stamp matching ────────────────────────────────────────────────────────────


def test_match_by_stamp_rounds_to_4_decimals():
    # sub-0.1ms bake stamp noise must still match; genuinely different stamps must not
    src = [[9.1129999999999995, 1.0, 2.0, 0, 0, 0, 0, 1],
           [10.5000000000000000, 3.0, 4.0, 0, 0, 0, 0, 1]]
    dst = [[9.113000, 5.0, 6.0, 0, 0, 0, 0, 1],
           [10.501000, 7.0, 8.0, 0, 0, 0, 0, 1]]
    s, d = match_by_stamp(src, dst)
    assert len(s) == len(d) == 1
    np.testing.assert_allclose(s[0], [1.0, 2.0])
    np.testing.assert_allclose(d[0], [5.0, 6.0])


# ── derived transforms consistent with R ──────────────────────────────────────


def _reg(theta, t, mirrored):
    src, dst, _ = _make_pair(theta, t, mirrored)
    return registration_dict(fit_se2(src, dst), len(src))


@pytest.mark.parametrize("mirrored", [False, True])
def test_traj_yaw_transforms_like_heading_vectors(mirrored):
    reg = _reg(0.9, (2.0, -1.0), mirrored)
    R = np.asarray(reg["R"])
    rows = []
    for i, yaw in enumerate(np.linspace(-3.0, 3.0, 7)):
        rows.append([float(i), 1.0 + i, -2.0 * i, 0.62,
                     0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    out = transform_traj(rows, reg)
    for src_row, dst_row in zip(rows, out):
        yaw_src = yaw_from_quat(*src_row[4:8])
        yaw_dst = yaw_from_quat(*dst_row[4:8])
        # the heading VECTOR must map by R — true for proper and mirrored fits
        np.testing.assert_allclose(
            R @ [math.cos(yaw_src), math.sin(yaw_src)],
            [math.cos(yaw_dst), math.sin(yaw_dst)], atol=1e-12)
        np.testing.assert_allclose(
            transform_xy(np.array([src_row[1:3]]), reg)[0], dst_row[1:3], atol=1e-12)
        assert dst_row[3] == src_row[3]  # z preserved
        assert dst_row[0] == src_row[0]  # stamp preserved


def test_transform_vertices_moves_xy_keeps_z():
    reg = _reg(1.2, (0.5, 0.5), True)
    V = np.array([[1.0, 2.0, 3.0], [-4.0, 0.0, -0.5]])
    out = transform_vertices(V, reg)
    np.testing.assert_allclose(out[:, 2], V[:, 2])
    np.testing.assert_allclose(out[:, :2], transform_xy(V[:, :2], reg))
    np.testing.assert_allclose(V, [[1, 2, 3], [-4, 0, -0.5]])  # input untouched


# ── start pose ────────────────────────────────────────────────────────────────


def test_start_pose_is_first_row_x_y_yaw():
    yaw = 2.5
    rows = [[8.846, 1.5, -0.25, 0.0, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)],
            [9.0, 99.0, 99.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    sp = start_pose(rows)
    assert sp["x"] == 1.5 and sp["y"] == -0.25
    assert math.isclose(sp["yaw"], yaw, abs_tol=1e-12)
    assert sp["frame"] == "M"


# ── reference bake acceptance (validated 17-07 numbers) ───────────────────────


@pytest.mark.skipif(not (REF_SLAM.exists() and REF_CUSFM.exists()),
                    reason="teleop_v1_demo reference bake not on disk")
def test_reference_bake_fit_reproduces_validated_numbers(tmp_path):
    out = tmp_path / "registration_mesh.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "fit",
         "--slam-tum", str(REF_SLAM), "--cusfm-tum", str(REF_CUSFM),
         "--out", str(out)],
        check=True, cwd=REPO)
    reg = json.loads(out.read_text())
    assert reg["mirrored"] is True
    assert reg["n"] == 320
    assert reg["residual_mean_m"] == pytest.approx(0.170, abs=0.005)
    assert reg["residual_p95_m"] == pytest.approx(0.227, abs=0.01)
    assert reg["residual_max_m"] == pytest.approx(0.240, abs=0.01)


@pytest.mark.skipif(not REF_SLAM.exists(),
                    reason="teleop_v1_demo reference bake not on disk")
def test_reference_bake_start_pose_matches_first_slam_row(tmp_path):
    rows = load_tum(REF_SLAM)
    sp = start_pose(rows)
    assert sp["x"] == rows[0][1] and sp["y"] == rows[0][2]
    assert sp["stamp_s"] == rows[0][0]
