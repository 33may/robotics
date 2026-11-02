"""build_map — bench-only runtime-map builder for the cuvslam realization (D14).

D5 fallback, promoted to the primary runtime path by the 2026-07-17 Gate A verdict:
the container-baked `cuvslam_map` LMDB does NOT localize in vendored PyCuVSLAM v16
(see memory: cuvslam-container-map-not-pycuvslam-loadable), while a map rebuilt by
PyCuVSLAM `save_map` from the SAME keyframe images localizes at millimetres.

Builds from a bake's edex (rectified stereo pairs + calibration, the exact pixels the
container consumed), with the VEHICLE-frame rig (rig_from_camera = edex
sensor_to_vehicle). Consequence: the map's world frame == edex world == the occupancy
grid's frame — hints clicked on the occupancy map need no extra registration.

    conda run -n bench-cuvslam python \
        logic/oli/reason/localization/realizations/cuvslam/build_map.py \
        [--bake <bake_dir>] [--out <map_dir>] [--audit-samples 8]

Output: <out>/ cuVSLAM map + map_meta.json (rig, source, audit verdicts).
Audit: localize K spread keyframes with perturbed edex-pose hints; each must land
within --tol of its edex pose. Exit 0 iff all pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from derisk_map_bridge import (  # noqa: E402
    BAKE_DEFAULT, _inv, _mat_to_quat_xyzw, _se3, load_keyframes, make_rig)

import cuvslam  # noqa: E402


def _images(pair: dict, edex: Path):
    return [np.asarray(Image.open(edex / pair[s]["image_name"]))
            for s in ("head_left", "head_right")]


def _vehicle_pose(pair: dict, s2v: dict) -> np.ndarray:
    """Edex-world vehicle pose for a keyframe: T_wv = T_wc(left) ∘ inv(s2v(left))."""
    return _se3(pair["head_left"]["camera_to_world"]) @ _inv(s2v["head_left"])


def _configs():
    odom = cuvslam.Tracker.OdometryConfig(
        async_sba=False, rectified_stereo_camera=True)
    slam = cuvslam.Tracker.SlamConfig(sync_mode=True, max_map_size=0)
    return odom, slam


def build(bake: Path, out: Path, pairs: dict, calib: dict, s2v: dict) -> dict:
    edex = bake / "edex"
    sids = sorted(pairs, key=int)
    rig = make_rig(calib, s2v)
    odom_cfg, slam_cfg = _configs()
    tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)

    lost = 0
    for i, sid in enumerate(sids):
        ts = int(pairs[sid]["head_left"]["timestamp_microseconds"]) * 1000
        est, _ = tracker.track(ts, _images(pairs[sid], edex))
        if est.world_from_rig is None:
            lost += 1
        if i % 400 == 0:
            print(f"  tracked {i}/{len(sids)} (lost so far: {lost})")
    metrics = tracker.get_slam_metrics()
    print(f"tracked {len(sids)} keyframes, lost {lost}  |  metrics: {metrics}")

    saved = {}
    tracker.save_map(str(out), lambda ok: saved.setdefault("ok", ok))
    assert saved.get("ok"), "save_map failed"
    slam_poses = tracker.get_all_slam_poses()
    del tracker
    print(f"map saved -> {out}  ({len(slam_poses)} slam poses)")

    # dump the tracker's OWN trajectory (map frame) — the registration instrument
    with open(out / "slam_poses.tum", "w") as f:
        for p in slam_poses:
            t, q = p.pose.translation, p.pose.rotation
            f.write(f"{p.timestamp_ns / 1e9:.9f} {t[0]} {t[1]} {t[2]} "
                    f"{q[0]} {q[1]} {q[2]} {q[3]}\n")

    # map-world vs edex-world delta over the whole drive — if these frames are NOT
    # the same, this prints it (translation + yaw deltas per matched stamp)
    by_ts = {int(pairs[s]["head_left"]["timestamp_microseconds"]) * 1000: s
             for s in sids}
    dts, dyaws = [], []
    for p in slam_poses:
        sid = by_ts.get(p.timestamp_ns)
        if sid is None:
            continue
        t_wv = _vehicle_pose(pairs[sid], s2v)
        d = np.array(p.pose.translation) - t_wv[:3, 3]
        dts.append(np.linalg.norm(d))
        yaw_slam = np.arctan2(*_quat_wm(p.pose.rotation))
        yaw_edex = np.arctan2(t_wv[1, 0], t_wv[0, 0])
        dyaws.append(abs(np.arctan2(np.sin(yaw_slam - yaw_edex),
                                    np.cos(yaw_slam - yaw_edex))))
    if dts:
        dts, dyaws = np.array(dts), np.degrees(dyaws)
        print(f"map-world vs edex-world: |Δt| mean {dts.mean():.3f} m, "
              f"p95 {np.percentile(dts, 95):.3f}, max {dts.max():.3f}  |  "
              f"|Δyaw| mean {dyaws.mean():.1f}°, max {dyaws.max():.1f}°")
    return {"frames": len(sids), "lost": lost, "slam_poses": len(slam_poses),
            "vs_edex": {"dt_mean_m": float(dts.mean()) if len(dts) else None,
                        "dt_max_m": float(dts.max()) if len(dts) else None,
                        "dyaw_mean_deg": float(dyaws.mean()) if len(dts) else None}}


def _quat_wm(q):
    """(sin_yaw_num, cos_yaw_den) of the yaw of quaternion xyzw."""
    x, y, z, w = (float(v) for v in q)
    return 2 * (w * z + x * y), 1 - 2 * (y * y + z * z)


def _load_tum(path: Path) -> dict:
    """slam_poses.tum -> {stamp_ns: (t xyz, q xyzw)}."""
    out = {}
    for line in path.read_text().splitlines():
        v = [float(x) for x in line.split()]
        out[int(round(v[0] * 1e9))] = (np.array(v[1:4]), v[4:8])
    return out


def audit(out: Path, pairs: dict, calib: dict, s2v: dict, edex: Path,
          samples: int, perturb: float, tol: float, hint: str = "slam") -> list:
    sids = sorted(pairs, key=int)
    picks = [sids[int((len(sids) - 1) * f)]
             for f in np.linspace(0.05, 0.95, samples)]
    rig = make_rig(calib, s2v)
    odom_cfg, slam_cfg = _configs()
    loc = cuvslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=4.0, vertical_search_radius=1.0,
        horizontal_step=0.25, vertical_step=0.25, angular_step_rads=0.1)
    slam_tum = _load_tum(out / "slam_poses.tum") if hint == "slam" else {}

    results = []
    ts = 2_000_000_000_000
    for sid in picks:
        t_wv = _vehicle_pose(pairs[sid], s2v)
        if hint == "slam":
            kf_ts = int(pairs[sid]["head_left"]["timestamp_microseconds"]) * 1000
            near = min(slam_tum, key=lambda k: abs(k - kf_ts))
            t_true, q_true = slam_tum[near]
        else:  # edex-frame hint
            t_true, q_true = t_wv[:3, 3], _mat_to_quat_xyzw(t_wv[:3, :3])
        guess = cuvslam.Pose(
            translation=list(t_true + np.array([perturb, -perturb / 2, 0.0])),
            rotation=list(q_true))
        imgs = _images(pairs[sid], edex)

        tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)
        tracker.track(ts, imgs)
        done, res = threading.Event(), {}

        def finish(pose, err, res=res, done=done):
            res["pose"], res["err"] = pose, err
            done.set()

        tracker.localize_in_map(str(out), ts, guess, imgs, loc,
                                lambda: None, finish)
        done.wait(timeout=60)
        del tracker
        ts += 10_000_000_000

        pose = res.get("pose")
        err = (float(np.linalg.norm(np.array(pose.translation) - t_true))
               if pose is not None else None)
        ok = err is not None and err <= tol
        results.append({"sid": sid, "err_m": err, "pass": ok,
                        "msg": res.get("err")})
        print(f"  audit sid {sid}: "
              + (f"{'PASS' if ok else 'FAIL'} |err|={err:.3f} m" if err is not None
                 else f"FAIL — {res.get('err')!r}"))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default=BAKE_DEFAULT)
    ap.add_argument("--out", default=None,
                    help="output map dir (default: <bake>/pycuvslam_map)")
    ap.add_argument("--audit-samples", type=int, default=8)
    ap.add_argument("--perturb", type=float, default=0.15)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--hint", choices=("slam", "edex"), default="slam",
                    help="audit hint source: tracker's own trajectory or edex poses")
    ap.add_argument("--audit-only", action="store_true",
                    help="skip build; audit an existing map dir")
    args = ap.parse_args()

    bake = Path(args.bake)
    out = Path(args.out) if args.out else bake / "pycuvslam_map"
    edex = bake / "edex"
    pairs, calib, s2v = load_keyframes(edex)
    print(f"build_map: {len(pairs)} stereo keyframes from {edex}")

    build_stats = None
    if not args.audit_only:
        build_stats = build(bake, out, pairs, calib, s2v)
    audit_res = audit(out, pairs, calib, s2v, edex,
                      args.audit_samples, args.perturb, args.tol, args.hint)

    n_pass = sum(r["pass"] for r in audit_res)
    meta = {
        "source_bake": str(bake),
        "rig": "vehicle-frame (rig_from_camera = edex sensor_to_vehicle)",
        "build": build_stats,
        "audit": {"hint": args.hint, "perturb_m": args.perturb,
                  "tol_m": args.tol, "results": audit_res},
    }
    if args.audit_only and (out / "map_meta.json").exists():
        prior = json.loads((out / "map_meta.json").read_text())
        meta["build"] = prior.get("build")
    (out / "map_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"AUDIT ({args.hint} hints): {n_pass}/{len(audit_res)} within "
          f"{args.tol} m  -> {out / 'map_meta.json'}")
    return 0 if n_pass == len(audit_res) else 1


if __name__ == "__main__":
    sys.exit(main())
