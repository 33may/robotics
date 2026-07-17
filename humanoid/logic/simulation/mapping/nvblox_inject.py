"""nvblox_inject.py — feed recorded sensor depth into the NVIDIA occupancy step
(MAY-173 Phase 2.2, vendor-pipeline-first).

The container's `occupancy` step (`nvblox_ros fuse_cusfm`, driven by
`create_map_offline.py`) consumes `map_frames/rectified/frames_meta.json`
keyframes + per-keyframe uint16-mm depth PNGs in `map_frames/depth/<cam>/`.
FoundationStereo inference only ever existed to FILL that depth dir — our dump
already carries real sensor depth in the exact expected format, so this script
maps it in and NVIDIA's fusion does the rest (their poses, their conventions,
their ray-casting):

  1. `head` depth  → `depth/head_left/<stamp>.png` — the RGBD cam sits 2.5 cm
     from head_left with identical intrinsics: sub-cell registration error.
  2. `chest` depth → NEW `chest` keyframes appended to frames_meta:
     `camera_to_world = (bake's own head_left pose) ∘ (static FK offset)` —
     robot description only, never GT. Chest (35° down) supplies the near-field
     floor/low-obstacle evidence the level head cam cannot see.

Convention safety: before writing anything, the computed head_left optical
mount must reproduce the meta's own `sensor_to_vehicle` entry (runtime
self-check + pinned in tests). A bake with different conventions fails loudly.

Idempotent: frames_meta is backed up to `frames_meta.orig.json` once and always
rebuilt from that backup. Pure numpy + PIL + stdlib; `brain`-clean imports.

Usage:
    p logic/simulation/mapping/nvblox_inject.py \
        --dump data/coverage_drives/teleop_v1_demo \
        --map-dir data/maps/teleop_v1_demo/<bake> [--no-chest]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from humanoid.logic.oli.camera_mounts import CHEST_CAM, STEREO_CAMERAS, rgb_intrinsics
from humanoid.logic.oli.recording.fk import T_base_cam_usd

_CHEST_PARAMS_ID = "2"

#: USD camera (+X right, +Y up, view -Z) → optical (+X right, +Y down, view +Z)
_USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


# ── pure convention math ─────────────────────────────────────────────────────


def optical_from_usd(T_usd: np.ndarray) -> np.ndarray:
    """Re-express a USD-convention camera pose in the optical convention the
    bake's frames_meta uses (same origin, axes flipped)."""
    T = T_usd.copy()
    T[:3, :3] = T[:3, :3] @ _USD_TO_OPTICAL
    return T


def axis_angle_from_R(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rotation matrix → (unit axis, angle rad). Stable away from 0/π via the
    antisymmetric part; falls back to the symmetric form near π."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = math.acos(tr)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(v)
    if s > 1e-9:
        return v / s, ang
    if ang < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    # angle ≈ π: axis from the diagonal
    a = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
    i = int(np.argmax(a))
    a[(i + 1) % 3] = R[i, (i + 1) % 3] / (2.0 * a[i])
    a[(i + 2) % 3] = R[i, (i + 2) % 3] / (2.0 * a[i])
    return a / np.linalg.norm(a), ang


def meta_from_T(T: np.ndarray) -> Dict:
    """4×4 → the frames_meta `{axis_angle, translation}` block."""
    axis, ang = axis_angle_from_R(T[:3, :3])
    return {
        "axis_angle": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2]),
                       "angle_degrees": math.degrees(ang)},
        "translation": {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3])},
    }


def T_from_meta(block: Dict) -> np.ndarray:
    """frames_meta `{axis_angle, translation}` block → 4×4."""
    aa, tr = block["axis_angle"], block["translation"]
    a = np.array([aa["x"], aa["y"], aa["z"]], float)
    a /= np.linalg.norm(a)
    ang = math.radians(aa["angle_degrees"])
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
    T[:3, 3] = (tr["x"], tr["y"], tr["z"])
    return T


def _head_left_mount():
    return next(m for m in STEREO_CAMERAS if m.name == "head_left")


def t_headleft_chest_optical() -> np.ndarray:
    """Static head_left→chest transform, both in optical convention — pure
    robot description (camera_mounts FK), the demo-legal calibration."""
    T_hl = optical_from_usd(T_base_cam_usd(_head_left_mount()))
    T_ch = optical_from_usd(T_base_cam_usd(CHEST_CAM))
    return np.linalg.inv(T_hl) @ T_ch


# ── the injection ────────────────────────────────────────────────────────────


def _verify_convention(meta: Dict) -> None:
    """The bake's own head_left sensor_to_vehicle must match our computed
    optical mount — otherwise the convention bridge is wrong for this bake."""
    for params in meta["camera_params_id_to_camera_params"].values():
        smd = params["sensor_meta_data"]
        if smd.get("sensor_name") == "head_left":
            expect = T_from_meta(smd["sensor_to_vehicle_transform"])
            got = optical_from_usd(T_base_cam_usd(_head_left_mount()))
            if not np.allclose(got, expect, atol=1e-6):
                raise RuntimeError(
                    "convention self-check FAILED: computed head_left optical mount "
                    f"does not match the bake's sensor_to_vehicle.\n{got}\nvs\n{expect}")
            return
    raise RuntimeError("no head_left camera_params in frames_meta — cannot verify conventions")


def _chest_camera_params() -> Dict:
    intr = rgb_intrinsics(width=1280, height=720, hfov_deg=CHEST_CAM.hfov_deg)
    T_ch = optical_from_usd(T_base_cam_usd(CHEST_CAM))
    return {
        "sensor_meta_data": {
            "sensor_type": "CAMERA",
            "sensor_name": "chest",
            "frequency": 30,
            "sensor_to_vehicle_transform": meta_from_T(T_ch),
        },
        "calibration_parameters": {
            "image_width": intr.width,
            "image_height": intr.height,
            "camera_matrix": {
                "data": [intr.fx, 0, intr.cx, 0, intr.fy, intr.cy, 0, 0, 1],
                "row_count": 3, "column_count": 3},
            "distortion_coefficients": {
                "data": [0, 0, 0, 0, 0], "row_count": 1, "column_count": 5},
            "rectification_matrix": {
                "data": [1, 0, 0, 0, 1, 0, 0, 0, 1], "row_count": 3, "column_count": 3},
            "projection_matrix": {
                "data": [intr.fx, 0, intr.cx, 0, 0, intr.fy, intr.cy, 0, 0, 0, 1, 0],
                "row_count": 3, "column_count": 4},
        },
        "camera_projection_model_type": "PINHOLE",
    }


def inject(dump_dir: Path, map_dir: Path, *, chest: bool = True) -> Dict[str, int]:
    """Populate map_frames/depth (+ chest keyframes) from the dump. Returns
    counters for reporting/tests."""
    frames = dump_dir / "frames"
    mf = map_dir / "map_frames"
    meta_path = mf / "rectified" / "frames_meta.json"
    orig_path = mf / "rectified" / "frames_meta.orig.json"
    if not orig_path.exists():
        shutil.copy2(meta_path, orig_path)          # first run: preserve the bake's meta
    meta = json.loads(orig_path.read_text())         # idempotent: always from the original
    _verify_convention(meta)

    stats = {"head_depth": 0, "chest_depth": 0, "chest_keyframes": 0, "missing": 0}

    # 1) head RGBD depth onto head_left keyframes (identical intrinsics, 2.5 cm offset)
    (mf / "depth" / "head_left").mkdir(parents=True, exist_ok=True)
    hl_frames = [k for k in meta["keyframes_metadata"]
                 if k["image_name"].startswith("head_left/")]
    for kf in hl_frames:
        stamp = int(Path(kf["image_name"]).stem)
        src = frames / "head_depth" / f"{stamp:019d}.png"
        if not src.exists():
            stats["missing"] += 1
            continue
        shutil.copy2(src, mf / "depth" / "head_left" / f"{stamp}.png")
        stats["head_depth"] += 1

    # 2) chest: depth + color + synthesized keyframes off head_left's own poses
    if chest:
        (mf / "depth" / "chest").mkdir(parents=True, exist_ok=True)
        (mf / "rectified" / "chest").mkdir(parents=True, exist_ok=True)
        rel = t_headleft_chest_optical()
        next_id = 1 + max(int(k["id"]) for k in meta["keyframes_metadata"])
        for kf in hl_frames:
            stamp = int(Path(kf["image_name"]).stem)
            d_src = frames / "chest_depth" / f"{stamp:019d}.png"
            c_src = frames / "chest" / f"{stamp:019d}.jpg"
            if not (d_src.exists() and c_src.exists()):
                stats["missing"] += 1
                continue
            shutil.copy2(d_src, mf / "depth" / "chest" / f"{stamp}.png")
            shutil.copy2(c_src, mf / "rectified" / "chest" / f"{stamp}.jpg")
            T_map_chest = T_from_meta(kf["camera_to_world"]) @ rel
            entry = {
                "id": str(next_id),
                "camera_params_id": _CHEST_PARAMS_ID,
                "timestamp_microseconds": kf["timestamp_microseconds"],
                "image_name": f"chest/{stamp}.jpg",
                "camera_to_world": meta_from_T(T_map_chest),
            }
            if "synced_sample_id" in kf:
                entry["synced_sample_id"] = kf["synced_sample_id"]
            meta["keyframes_metadata"].append(entry)
            next_id += 1
            stats["chest_depth"] += 1
            stats["chest_keyframes"] += 1
        meta["camera_params_id_to_camera_params"][_CHEST_PARAMS_ID] = _chest_camera_params()
        session = next(iter(meta["camera_params_id_to_session_name"].values()))
        meta["camera_params_id_to_session_name"][_CHEST_PARAMS_ID] = session

    meta_path.write_text(json.dumps(meta, indent=1))
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inject recorded depth into an nvblox bake.")
    ap.add_argument("--dump", required=True, help="teleop dump dir (frames/)")
    ap.add_argument("--map-dir", required=True, help="bake output dir (map_frames/)")
    ap.add_argument("--no-chest", action="store_true", help="head depth only")
    args = ap.parse_args(list(argv) if argv is not None else None)
    stats = inject(Path(args.dump), Path(args.map_dir), chest=not args.no_chest)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
