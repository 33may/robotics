"""GATE A (D5 map-bridge): does the CONTAINER-baked cuvslam_map load in vendored
PyCuVSLAM `localize_in_map`?

The one unknown left from the 15-07 plan: container cuVSLAM (ROS build) wrote the LMDB
map; vendored v16 must read it. Probe uses ONLY bake-internal artifacts — the edex
keyframe images (the exact pixels the map was built from) and their `camera_to_world`
poses (the frame the map lives in) — so a failure isolates the bridge, never data prep.

Per probe frame: fresh Tracker(stereo rig from bake calibration, SlamConfig(sync_mode)),
one track() to init (KITTI SLAM example pattern), then localize_in_map with the frame's
own pose as hint (optionally perturbed). PASS = |t_est - t_true| <= --tol for every frame.

Run from humanoid/:
    conda run -n bench-cuvslam python \
        logic/oli/reason/localization/realizations/cuvslam/derisk_map_bridge.py
Exit 0 iff all probe frames localize within tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import cuvslam

BAKE_DEFAULT = "data/maps/teleop_v1_demo/2026-07-16_14-25-28_teleop_v1_demo_bag"


def axis_angle_to_quat_xyzw(aa: dict) -> list:
    ax = np.array([aa["x"], aa["y"], aa["z"]], dtype=float)
    n = np.linalg.norm(ax) or 1.0
    half = math.radians(aa["angle_degrees"]) / 2.0
    s = math.sin(half)
    return [ax[0] / n * s, ax[1] / n * s, ax[2] / n * s, math.cos(half)]


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat_xyzw(m: np.ndarray) -> list:
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-8:
        return [(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                (m[1, 0] - m[0, 1]) / (4 * w), w]
    # w≈0 fallback (180° rotations) — not expected for our hints
    x = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0
    return [x, (m[0, 1] + m[1, 0]) / (4 * x), (m[0, 2] + m[2, 0]) / (4 * x),
            (m[2, 1] - m[1, 2]) / (4 * x)]


def _se3(transform: dict) -> np.ndarray:
    """4x4 from an edex {axis_angle, translation} transform dict."""
    t = np.eye(4)
    t[:3, :3] = _quat_to_mat(axis_angle_to_quat_xyzw(transform["axis_angle"]))
    t[:3, 3] = [transform["translation"][a] for a in "xyz"]
    return t


def _inv(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def load_keyframes(edex: Path):
    """Group kf-pose entries by synced_sample_id -> {left, right} with pose + image."""
    meta = json.loads((edex / "frames_meta_kf_poses.json").read_text())
    cam_params = meta["camera_params_id_to_camera_params"]
    calib = next(iter(cam_params.values()))["calibration_parameters"]

    # the container built its rig in the VEHICLE frame: each camera carries a
    # sensor_to_vehicle transform in the edex calibration. Reuse them verbatim —
    # cross-rig localization is unsupported, the runtime rig must BE the bake rig.
    s2v = {}
    for params in cam_params.values():
        smd = params["sensor_meta_data"]
        s2v[smd["sensor_name"]] = _se3(smd["sensor_to_vehicle_transform"])
    if "head_left" not in s2v:  # left params may only appear in frames_meta.json
        full = json.loads((edex / "frames_meta.json").read_text())
        for params in full["camera_params_id_to_camera_params"].values():
            smd = params["sensor_meta_data"]
            s2v.setdefault(smd["sensor_name"], _se3(smd["sensor_to_vehicle_transform"]))
    assert {"head_left", "head_right"} <= set(s2v), f"missing extrinsics: {set(s2v)}"

    pairs = defaultdict(dict)
    for kf in meta["keyframes_metadata"]:
        # side from the image path — head_left entries omit camera_params_id
        side = kf["image_name"].split("/")[0]
        pairs[kf["synced_sample_id"]][side] = kf
    complete = {
        sid: p for sid, p in pairs.items()
        if "head_left" in p and "head_right" in p
    }
    return complete, calib, s2v


def make_rig(calib: dict, s2v: dict):
    """Vehicle-frame stereo rig — rig_from_camera = the bake's sensor_to_vehicle."""
    k = calib["camera_matrix"]["data"]
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    cams = [cuvslam.Camera(), cuvslam.Camera()]
    for c, side in zip(cams, ("head_left", "head_right")):
        c.size = (calib["image_width"], calib["image_height"])
        c.focal = (fx, fy)
        c.principal = (cx, cy)
        c.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
        t = s2v[side]
        c.rig_from_camera = cuvslam.Pose(
            translation=list(t[:3, 3]), rotation=_mat_to_quat_xyzw(t[:3, :3]))
    return cuvslam.Rig(cams)


def probe_one(map_dir: Path, rig, kf_pair: dict, edex: Path, s2v: dict, ts_ns: int,
              perturb_m: float, search_radius: float):
    left, right = kf_pair["head_left"], kf_pair["head_right"]
    imgs = [
        np.asarray(Image.open(edex / left["image_name"])),
        np.asarray(Image.open(edex / right["image_name"])),
    ]
    # hint must be a VEHICLE pose (map world = vehicle frame at t0):
    # T_world_vehicle = T_world_cam_left ∘ inv(sensor_to_vehicle[head_left])
    t_wv = _se3(left["camera_to_world"]) @ _inv(s2v["head_left"])
    t_true = t_wv[:3, 3].copy()
    q = _mat_to_quat_xyzw(t_wv[:3, :3])
    hint_t = t_true + np.array([perturb_m, -perturb_m / 2, 0.0])
    guess = cuvslam.Pose(translation=list(hint_t), rotation=q)

    odom_cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False, rectified_stereo_camera=True)
    slam_cfg = cuvslam.Tracker.SlamConfig(sync_mode=True)
    tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)
    tracker.track(ts_ns, imgs)

    loc = cuvslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=search_radius, vertical_search_radius=1.0,
        horizontal_step=0.25, vertical_step=0.25, angular_step_rads=0.1)

    done = threading.Event()
    result = {}

    def finish(pose, err):
        result["pose"], result["err"] = pose, err
        done.set()

    tracker.localize_in_map(str(map_dir), ts_ns, guess, imgs, loc,
                            lambda: None, finish)
    fired = done.wait(timeout=60)
    del tracker
    if not fired:
        return None, None, "finish_cb never fired (60 s)"
    pose = result.get("pose")
    if pose is None:
        return None, None, f"localize failed: {result.get('err')!r}"
    err_vec = np.array(pose.translation) - t_true
    return pose, float(np.linalg.norm(err_vec)), result.get("err")


def self_map_control(pairs: dict, calib: dict, s2v: dict, edex: Path,
                     n_frames: int, perturb_m: float, search_radius: float,
                     tol: float) -> int:
    """Positive control: PyCuVSLAM builds its OWN map from the same keyframes, then
    localizes against it with a hint from its own slam poses. Self-consistent frames
    by construction — isolates machinery bugs from container-map incompatibility."""
    sids = sorted(pairs, key=int)[:n_frames]
    rig = make_rig(calib, s2v)
    odom_cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False, rectified_stereo_camera=True)
    slam_cfg = cuvslam.Tracker.SlamConfig(sync_mode=True, max_map_size=0)
    tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)

    print(f"self-map control: tracking {len(sids)} stereo keyframes …")
    for sid in sids:
        left = pairs[sid]["head_left"]
        ts = int(left["timestamp_microseconds"]) * 1000
        imgs = [np.asarray(Image.open(edex / pairs[sid][s]["image_name"]))
                for s in ("head_left", "head_right")]
        est, _ = tracker.track(ts, imgs)
        if est.world_from_rig is None:
            print(f"  VO lost at sid {sid} — control aborted")
            return 1

    slam_poses = tracker.get_all_slam_poses()
    map_dir = "/tmp/cuvslam_selfmap"
    saved = {}
    tracker.save_map(map_dir, lambda ok: saved.setdefault("ok", ok))
    if not saved.get("ok"):
        print("save_map FAILED")
        return 1
    print(f"self-map saved ({len(slam_poses)} slam poses) -> {map_dir}")
    del tracker

    # probe the middle keyframe against the self-map, hint from the OWN pose stream
    mid = sids[len(sids) // 2]
    left = pairs[mid]["head_left"]
    ts_mid = int(left["timestamp_microseconds"]) * 1000
    own = min(slam_poses, key=lambda p: abs(p.timestamp_ns - ts_mid))
    t_true = np.array(own.pose.translation)
    imgs = [np.asarray(Image.open(edex / pairs[mid][s]["image_name"]))
            for s in ("head_left", "head_right")]
    guess = cuvslam.Pose(
        translation=list(t_true + np.array([perturb_m, -perturb_m / 2, 0.0])),
        rotation=list(own.pose.rotation))

    tracker2 = cuvslam.Tracker(rig, odom_cfg, slam_cfg)
    tracker2.track(1_000_000_000_000, imgs)
    loc = cuvslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=search_radius, vertical_search_radius=1.0,
        horizontal_step=0.25, vertical_step=0.25, angular_step_rads=0.1)
    done, result = threading.Event(), {}

    def finish(pose, err):
        result["pose"], result["err"] = pose, err
        done.set()

    tracker2.localize_in_map(map_dir, 1_000_000_000_000, guess, imgs, loc,
                             lambda: None, finish)
    done.wait(timeout=60)
    pose = result.get("pose")
    if pose is None:
        print(f"CONTROL FAIL — localize failed: {result.get('err')!r} "
              "(machinery bug, NOT the container map)")
        return 1
    err = float(np.linalg.norm(np.array(pose.translation) - t_true))
    verdict = "PASS" if err <= tol else "FAIL"
    print(f"CONTROL {verdict}  |err|={err:.3f} m  (sid {mid})")
    print("=> machinery works; container-map failure is a real bridge incompat"
          if verdict == "PASS" else "=> tune machinery on this fast loop")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default=BAKE_DEFAULT)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--perturb", type=float, default=0.15, help="hint offset [m]")
    ap.add_argument("--search-radius", type=float, default=4.0)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--self-map", type=int, default=0, metavar="N",
                    help="positive control: build own map from first N keyframes")
    args = ap.parse_args()

    bake = Path(args.bake)
    edex = bake / "edex"
    map_dir = bake / "cuvslam_map"
    assert map_dir.is_dir(), f"no cuvslam_map at {map_dir}"

    pairs, calib, s2v = load_keyframes(edex)
    if args.self_map:
        return self_map_control(pairs, calib, s2v, edex, args.self_map,
                                args.perturb, args.search_radius, args.tol)
    sids = sorted(pairs, key=int)
    picks = [sids[int(len(sids) * f)] for f in
             np.linspace(0.1, 0.85, args.samples)]
    rig = make_rig(calib, s2v)
    print(f"map: {map_dir}  |  {len(sids)} stereo keyframes, probing {picks}  "
          f"|  vehicle-frame rig, hint perturb {args.perturb} m")

    failures = 0
    ts = 1_000_000_000_000
    for sid in picks:
        pose, err, msg = probe_one(map_dir, rig, pairs[sid], edex, s2v, ts,
                                   args.perturb, args.search_radius)
        ts += 10_000_000_000  # fresh epoch per probe (strictly increasing anyway)
        left = pairs[sid]["head_left"]
        t_us = left["timestamp_microseconds"]
        if pose is None:
            print(f"  sid {sid} (t={t_us} us): FAIL — {msg}")
            failures += 1
            continue
        verdict = "PASS" if err <= args.tol else "FAIL"
        if verdict == "FAIL":
            failures += 1
        print(f"  sid {sid} (t={t_us} us): {verdict}  |err|={err:.3f} m  "
              f"est={[f'{x:.2f}' for x in pose.translation]}  msg={msg!r}")

    print("GATE A:", "PASS — container map loads in PyCuVSLAM" if failures == 0
          else f"FAIL — {failures}/{len(picks)} probes failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
