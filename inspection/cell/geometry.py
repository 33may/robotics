#!/usr/bin/env python3
"""Scene geometry for the inspection loop — v1 sandbox.

Depth makes the geometry; detection only labels it (project principle).
Pipeline per view: deproject raw depth (IR-left frame) -> transform to base
frame via FK + hand-eye -> drop table plane -> object = largest above-table
cluster inside the workspace box. Detection-seeded growth replaces the
largest-cluster heuristic later, same interface.

All point clouds are (N, 3) float64 in the robot base frame, meters.
"""

from pathlib import Path

import numpy as np

from vbti.logic.servos.fk_so101 import fk

T_FLANGE_CAM_PATH = Path(__file__).resolve().parents[2] / "data" / "inspection" / "handeye" / "T_flange_cam.npy"

# Generous workspace box in base frame (m) — cuts the far room, keeps the table
WORKSPACE = {"x": (0.05, 0.60), "y": (-0.45, 0.45), "z": (-0.10, 0.50)}

TABLE_DIST_M = 0.006       # RANSAC inlier threshold for the table plane
ABOVE_TABLE_M = 0.008      # keep points this far above the fitted plane
VOXEL_M = 0.003            # accumulator downsample
DBSCAN_EPS_M = 0.02
DBSCAN_MIN_PTS = 20
MAX_RANGE_M = 0.60         # depth beyond this is noise for our cell
MIN_RANGE_M = 0.13         # gripper fingers live at 70-120mm from the camera —
                           # cut them out of every cloud (crude arm mask, v1)


def load_T_flange_cam() -> np.ndarray:
    return np.load(T_FLANGE_CAM_PATH)


def deproject(depth_u16: np.ndarray, intr: dict, depth_scale: float,
              max_range: float = MAX_RANGE_M, min_range: float = MIN_RANGE_M,
              stride: int = 2) -> np.ndarray:
    """Raw depth image -> (N,3) points in the camera (IR-left) frame.

    min_range acts as the v1 arm mask: the gripper fingers permanently hover
    7-12cm in front of the wrist camera and otherwise become the largest
    above-table cluster (observed 2026-08-14 — they track the camera).
    """
    z = depth_u16[::stride, ::stride].astype(np.float64) * depth_scale
    h, w = z.shape
    us = np.arange(0, depth_u16.shape[1], stride)[:w]
    vs = np.arange(0, depth_u16.shape[0], stride)[:h]
    uu, vv = np.meshgrid(us, vs)
    valid = (z > min_range) & (z < max_range)
    z = z[valid]
    x = (uu[valid] - intr["ppx"]) / intr["fx"] * z
    y = (vv[valid] - intr["ppy"]) / intr["fy"] * z
    return np.column_stack([x, y, z])


def cam_to_base(points_cam: np.ndarray, joints_deg: dict,
                T_flange_cam: np.ndarray | None = None) -> np.ndarray:
    """Camera-frame points -> base frame via FK and hand-eye."""
    if T_flange_cam is None:
        T_flange_cam = load_T_flange_cam()
    T = fk(joints_deg) @ T_flange_cam
    return points_cam @ T[:3, :3].T + T[:3, 3]


def crop_workspace(points: np.ndarray) -> np.ndarray:
    m = np.ones(len(points), dtype=bool)
    for i, ax in enumerate("xyz"):
        lo, hi = WORKSPACE[ax]
        m &= (points[:, i] >= lo) & (points[:, i] <= hi)
    return points[m]


def fit_table(points: np.ndarray) -> np.ndarray:
    """RANSAC plane fit -> [a,b,c,d] with unit normal pointing +z (up)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, _ = pcd.segment_plane(distance_threshold=TABLE_DIST_M,
                                 ransac_n=3, num_iterations=500)
    plane = np.asarray(plane)
    if plane[2] < 0:      # normal must point up
        plane = -plane
    return plane


def above_table(points: np.ndarray, plane: np.ndarray,
                margin: float = ABOVE_TABLE_M) -> np.ndarray:
    d = points @ plane[:3] + plane[3]
    return points[d > margin]


def rectify_to_table(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Rotate+shift so the fitted table plane becomes exactly z=0.

    The table is flat ground truth visible in every view; per-view FK
    orientation error (arm flex/backlash, observed 4-13 deg of plane tilt)
    is absorbed here. Only yaw about z remains uncorrected.
    """
    n = plane[:3] / np.linalg.norm(plane[:3])
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, target)
    s, c = np.linalg.norm(v), float(n @ target)
    if s < 1e-9:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    pts = points @ R.T
    # plane point closest to origin maps to z = -d after rotation
    pts[:, 2] += plane[3]
    return pts


def largest_cluster(points: np.ndarray) -> np.ndarray:
    """Largest DBSCAN cluster — v1 stand-in for detection-seeded growth."""
    import open3d as o3d
    if len(points) < DBSCAN_MIN_PTS:
        return points
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(pcd.cluster_dbscan(eps=DBSCAN_EPS_M,
                                           min_points=DBSCAN_MIN_PTS))
    if labels.max() < 0:
        return points
    counts = np.bincount(labels[labels >= 0])
    return points[labels == counts.argmax()]


def object_from_view(depth_u16: np.ndarray, intr: dict, depth_scale: float,
                     joints_deg: dict, plane: np.ndarray | None = None,
                     T_flange_cam: np.ndarray | None = None) -> dict:
    """One view -> object candidate. Returns dict with points/centroid/plane."""
    pts_cam = deproject(depth_u16, intr, depth_scale)
    pts = crop_workspace(cam_to_base(pts_cam, joints_deg, T_flange_cam))
    if len(pts) < 100:
        raise RuntimeError(f"only {len(pts)} workspace points — bad view?")
    if plane is None:
        plane = fit_table(pts)
    rect = rectify_to_table(pts, plane)          # table -> exactly z=0
    above = rect[rect[:, 2] > ABOVE_TABLE_M]
    obj = largest_cluster(above) if len(above) else above
    return {
        "points": obj,                            # table-rectified frame
        "centroid": obj.mean(axis=0) if len(obj) else None,
        "plane": plane,
        "n_scene": len(pts),
    }


class CloudAccumulator:
    """Grows the object cloud across views; centroid tracks the fused cloud."""

    def __init__(self, voxel: float = VOXEL_M):
        self.voxel = voxel
        self._points = np.empty((0, 3))

    def add(self, points: np.ndarray) -> None:
        import open3d as o3d
        if len(points) == 0:
            return
        merged = np.vstack([self._points, points])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged))
        pcd = pcd.voxel_down_sample(self.voxel)
        if len(pcd.points) > 50:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)
        self._points = np.asarray(pcd.points)

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def centroid(self) -> np.ndarray | None:
        return self._points.mean(axis=0) if len(self._points) else None

    def aabb(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not len(self._points):
            return None
        return self._points.min(axis=0), self._points.max(axis=0)
