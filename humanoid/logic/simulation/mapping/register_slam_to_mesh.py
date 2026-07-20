"""register_slam_to_mesh — GT-free bake-time alignment of cuSFM artifacts into M.

Anton's decision (17-07): every map artifact ships in the cuVSLAM map frame **M**
(the frame the runtime localizer reports in) because the pycuvslam landmark DB is
opaque/untransformable while the mesh is losslessly transformable vector data.
Runtime carries ZERO transform code; this script does the one-time alignment at
bake time — with NO ground truth anywhere: it fits the two trajectories of the
SAME teleop drive against each other.

    T_M←cusfm : rigid SE(2), reflection ALLOWED (the cuSFM/edex frame is
                MIRRORED vs M — see edex-world-is-mirrored memory), fitted by
                2D Umeyama on timestamp-matched pairs (stamps rounded to 4
                decimals) of
                    src = <bake>/cusfm/pose_graph/vehicle_pose.tum   (cusfm)
                    dst = <bake>/pycuvslam_map/slam_poses.tum        (M)

Validated on the teleop_v1_demo reference bake: mirrored=True, residual mean
0.170 m / p95 0.227 / max 0.240 over 320 matched keyframe pairs (the residual is
cuSFM PGO wiggle; the mesh walls carry the SAME wiggle — map and poses err
together).

Subcommands (thin CLI over pure-numpy functions; open3d only for mesh IO):

  fit         slam.tum + cusfm.tum → registration_mesh.json  (AUDIT artifact —
              runtime never reads it)
  apply       registration + mesh.ply/traj.tum → M-frame copies (new files,
              sources untouched). Mirrored fits flip triangle winding so
              normals stay outward.
  start-pose  first pose of slam_poses.tum → start_pose.json {x, y, yaw} — the
              demo's GT-free known-start hint, already in M.

Convention: R may include a reflection; theta_rad = atan2(R[1,0], R[0,0]) so
R = Rot(theta) @ diag(1, -1) when mirrored, R = Rot(theta) otherwise. Headings
transform as yaw' = theta - yaw (mirrored) / theta + yaw (proper). Transformed
trajectories are PLANAR: z is preserved, quaternions are rewritten as pure-z
rotations of the transformed yaw (only xy is consumed downstream by
mesh_to_occupancy --traj).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

STAMP_NDIGITS = 4  # timestamp match precision (s) — sub-ms bake stamp noise safe


# ── TUM IO ────────────────────────────────────────────────────────────────────


def load_tum(path: Path) -> List[List[float]]:
    """TUM rows [stamp, tx, ty, tz, qx, qy, qz, qw], sorted by stamp."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= 8:
            rows.append(v[:8])
    rows.sort(key=lambda r: r[0])
    return rows


def write_tum(path: Path, rows: List[List[float]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{r[0]:.9f} " + " ".join(f"{v:.9f}" for v in r[1:8]) + "\n")


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# ── fit (2D Umeyama, reflection allowed) ──────────────────────────────────────


def match_by_stamp(
    src_rows: List[List[float]], dst_rows: List[List[float]],
    ndigits: int = STAMP_NDIGITS,
) -> Tuple[np.ndarray, np.ndarray]:
    """(src_xy, dst_xy) for stamps present in both, rounded to `ndigits`."""
    src = {round(r[0], ndigits): (r[1], r[2]) for r in src_rows}
    dst = {round(r[0], ndigits): (r[1], r[2]) for r in dst_rows}
    common = sorted(set(src) & set(dst))
    return (np.array([src[t] for t in common], float),
            np.array([dst[t] for t in common], float))


def fit_se2(src: np.ndarray, dst: np.ndarray) -> Dict:
    """Least-squares dst ≈ src @ R.T + t with det(R) = ±1 (whichever fits
    better) — same algorithm as map_compare.fit_traj_transform, plus the full
    residual stats the audit artifact records."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 3:
        raise ValueError(f"need >=3 matched pairs, got {len(src)}")
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)

    best = None
    for d in (1.0, -1.0):
        R = Vt.T @ np.diag([1.0, d]) @ U.T
        t = cd - R @ cs
        res = np.linalg.norm(src @ R.T + t - dst, axis=1)
        if best is None or res.mean() < best["residuals"].mean():
            best = {"R": R, "t": t, "mirrored": bool(np.linalg.det(R) < 0),
                    "residuals": res}
    assert best is not None
    best["theta_rad"] = float(math.atan2(best["R"][1, 0], best["R"][0, 0]))
    return best


def registration_dict(fit: Dict, n: int, note: str = "") -> Dict:
    res = fit["residuals"]
    return {
        "R": np.asarray(fit["R"]).tolist(),
        "t": np.asarray(fit["t"]).tolist(),
        "mirrored": fit["mirrored"],
        "theta_rad": fit["theta_rad"],
        "residual_mean_m": float(res.mean()),
        "residual_p95_m": float(np.percentile(res, 95)),
        "residual_max_m": float(res.max()),
        "n": int(n),
        "note": note or ("T_M<-cusfm rigid SE(2), reflection allowed: "
                         "p_M = R@[x,y]_cusfm + t. AUDIT artifact only — "
                         "runtime never reads it."),
    }


# ── apply ─────────────────────────────────────────────────────────────────────


def transform_xy(pts: np.ndarray, reg: Dict) -> np.ndarray:
    return np.asarray(pts, float) @ np.asarray(reg["R"], float).T + np.asarray(reg["t"], float)


def transform_vertices(vertices: np.ndarray, reg: Dict) -> np.ndarray:
    """SE(2) on xy, z untouched — the planar frame alignment, mesh stays metric."""
    out = np.array(vertices, float, copy=True)
    out[:, :2] = transform_xy(out[:, :2], reg)
    return out


def transform_traj(rows: List[List[float]], reg: Dict) -> List[List[float]]:
    """TUM rows → M frame (planar): xy mapped, z kept, quat = pure-z of yaw'."""
    theta = float(reg["theta_rad"])
    sgn = -1.0 if reg["mirrored"] else 1.0
    out = []
    for stamp, tx, ty, tz, qx, qy, qz, qw in rows:
        (x, y), = transform_xy(np.array([[tx, ty]]), reg)
        yaw = theta + sgn * yaw_from_quat(qx, qy, qz, qw)
        out.append([stamp, x, y, tz,
                    0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
    return out


def apply_mesh(mesh_in: Path, mesh_out: Path, reg: Dict) -> Dict:
    import open3d as o3d  # lazy: CLI-only (isaac env)

    mesh = o3d.io.read_triangle_mesh(str(mesh_in))
    verts = transform_vertices(np.asarray(mesh.vertices), reg)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    if reg["mirrored"]:  # reflection flips winding — restore outward normals
        mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(mesh.triangles)[:, ::-1].copy())
    mesh.compute_vertex_normals()
    mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(mesh_out), mesh)
    return {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)}


# ── start pose ────────────────────────────────────────────────────────────────


def start_pose(slam_rows: List[List[float]]) -> Dict:
    stamp, tx, ty, _tz, qx, qy, qz, qw = slam_rows[0]
    return {"x": tx, "y": ty, "yaw": yaw_from_quat(qx, qy, qz, qw),
            "stamp_s": stamp, "frame": "M",
            "source": "first pose of pycuvslam_map/slam_poses.tum"}


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fit", help="fit T_M<-cusfm, write registration_mesh.json")
    p.add_argument("--slam-tum", required=True, type=Path,
                   help="pycuvslam_map/slam_poses.tum (M frame)")
    p.add_argument("--cusfm-tum", required=True, type=Path,
                   help="cusfm/pose_graph/vehicle_pose.tum (cusfm frame)")
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("apply", help="transform mesh/trajectory into M")
    p.add_argument("--registration", required=True, type=Path)
    p.add_argument("--mesh-in", type=Path)
    p.add_argument("--mesh-out", type=Path)
    p.add_argument("--traj-in", type=Path)
    p.add_argument("--traj-out", type=Path)

    p = sub.add_parser("start-pose", help="first slam pose → start_pose.json")
    p.add_argument("--slam-tum", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    a = ap.parse_args(argv)

    if a.cmd == "fit":
        src, dst = match_by_stamp(load_tum(a.cusfm_tum), load_tum(a.slam_tum))
        fit = fit_se2(src, dst)
        reg = registration_dict(fit, len(src))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(reg, indent=1))
        print(f"fit: n={reg['n']} mirrored={reg['mirrored']} "
              f"residual mean={reg['residual_mean_m']:.3f} "
              f"p95={reg['residual_p95_m']:.3f} max={reg['residual_max_m']:.3f} m "
              f"-> {a.out}")
        return 0

    if a.cmd == "apply":
        if not (a.mesh_in or a.traj_in):
            ap.error("apply: need --mesh-in and/or --traj-in")
        if bool(a.mesh_in) != bool(a.mesh_out) or bool(a.traj_in) != bool(a.traj_out):
            ap.error("apply: in/out flags must be paired")
        reg = json.loads(a.registration.read_text())
        if a.mesh_in:
            stats = apply_mesh(a.mesh_in, a.mesh_out, reg)
            print(f"mesh -> {a.mesh_out} ({stats['vertices']} verts, "
                  f"{stats['triangles']} tris, mirrored={reg['mirrored']})")
        if a.traj_in:
            rows = transform_traj(load_tum(a.traj_in), reg)
            a.traj_out.parent.mkdir(parents=True, exist_ok=True)
            write_tum(a.traj_out, rows)
            print(f"traj -> {a.traj_out} ({len(rows)} poses, planar)")
        return 0

    if a.cmd == "start-pose":
        rows = load_tum(a.slam_tum)
        if not rows:
            raise SystemExit(f"empty slam trajectory: {a.slam_tum}")
        sp = start_pose(rows)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(sp, indent=1))
        print(f"start pose: x={sp['x']:.3f} y={sp['y']:.3f} "
              f"yaw={sp['yaw']:.3f} -> {a.out}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
