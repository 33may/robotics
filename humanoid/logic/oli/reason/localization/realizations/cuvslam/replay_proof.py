"""replay_proof — offline accuracy instrument for the map_relative cuvslam module.

Replays a coverage-drive dump through `CuvslamModule` EXACTLY as the in-brain host
would (frame-paced LocalizationIn bundles, real stamps, hint = GT pose at the first
frame — the demo's known-start hint), then scores the emitted WORLD-frame poses
against the dump's GT (`poses.jsonl` base rows, same frame by decision 17-07).

HONESTY NOTE: replaying the dump the map was built FROM is a self-test — the true
exam is the live LOC MODE walk. This instrument proves machinery + the frame chain
(hint W→M, localize, anchored track, M→W emission) and gives the accuracy ceiling.

    conda run -n bench-cuvslam python \
        logic/oli/reason/localization/realizations/cuvslam/replay_proof.py \
        [--dump data/coverage_drives/teleop_v1_demo] [--map <bake>/pycuvslam_map]

Gate (daily 17-07): mean ATE <= 0.15 m, jumps(>0.5 m/frame) == 0.
Output: stats + verdict + plot PNG next to the dump's map artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))  # <repo-parent>/humanoid pkg

from humanoid.logic.oli.contracts import CameraFrame, CameraIntrinsics, Observation
from humanoid.logic.oli.reason.localization import (
    LocalizationIn,
    LocalizationSetup,
    RobotPose,
)
from humanoid.logic.oli.reason.localization.realizations.cuvslam.module import build

DUMP_DEFAULT = "data/coverage_drives/teleop_v1_demo"
MAP_DEFAULT = ("data/maps/teleop_v1_demo/2026-07-16_14-25-28_teleop_v1_demo_bag/"
               "pycuvslam_map")
GATE_MEAN_M = 0.15
JUMP_M = 0.5


def _obs(stamp_ns: int) -> Observation:
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _intrinsics(rig: dict, name: str) -> CameraIntrinsics:
    i = rig["cameras"][name]["intrinsics"]
    return CameraIntrinsics(width=i["width"], height=i["height"],
                            fx=i["fx"], fy=i["fy"], cx=i["cx"], cy=i["cy"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DUMP_DEFAULT)
    ap.add_argument("--map", dest="map_dir", default=MAP_DEFAULT)
    ap.add_argument("--every-n", type=int, default=1,
                    help="frame decimation for quick smoke runs")
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    args = ap.parse_args()

    dump, map_dir = Path(args.dump), Path(args.map_dir)
    rig = json.loads((dump / "rig.json").read_text())
    intr = {n: _intrinsics(rig, n) for n in ("head_left", "head_right")}

    gt = {}
    for line in open(dump / "poses.jsonl"):
        r = json.loads(line)
        if r["cam"] == "base":
            gt[r["stamp_ns"]] = (r["x"], r["y"], r["yaw"])

    stamps = sorted(int(p.stem) for p in (dump / "frames/head_left").glob("*.jpg"))
    stamps = [s for s in stamps if s in gt][:: args.every_n]
    if args.limit:
        stamps = stamps[: args.limit]
    print(f"replay: {len(stamps)} stereo ticks from {dump}  |  map: {map_dir}")

    cfg_path = Path(__file__).parent / "config.yaml"
    module = build(yaml.safe_load(cfg_path.read_text()) or {})
    g0 = gt[stamps[0]]
    module.start(LocalizationSetup(
        map_dir=map_dir,
        initial_pose=RobotPose(stamp_ns=stamps[0], x=g0[0], y=g0[1], yaw=g0[2]),
    ))

    est_xy, gt_xy, errs, statuses = [], [], [], []
    t0 = time.perf_counter()
    for ts in stamps:
        frames = {}
        for name in ("head_left", "head_right"):
            img = np.asarray(Image.open(dump / f"frames/{name}/{ts:019d}.jpg"))
            frames[name] = CameraFrame(stamp_ns=ts, name=name, rgb=img, depth=None,
                                       intrinsics=intr[name])
        out = module.step(LocalizationIn(stamp_ns=ts, frames=frames,
                                         observation=_obs(ts)))
        statuses.append(out.status)
        if out.pose is not None:
            g = gt[ts]
            est_xy.append((out.pose.x, out.pose.y))
            gt_xy.append((g[0], g[1]))
            errs.append(float(np.hypot(out.pose.x - g[0], out.pose.y - g[1])))
    wall = time.perf_counter() - t0
    module.stop()

    if not errs:
        print("NO POSES EMITTED — module dead? check stderr above")
        return 1

    e = np.array(errs)
    est, gta = np.array(est_xy), np.array(gt_xy)
    step_d = np.linalg.norm(np.diff(est, axis=0), axis=1)
    jumps = int((step_d > JUMP_M).sum())
    from collections import Counter
    st = Counter(s.name for s in statuses)
    cov = len(errs) / len(stamps)

    print(f"ATE  mean {e.mean():.3f} m  p95 {np.percentile(e, 95):.3f}  "
          f"max {e.max():.3f}   jumps>{JUMP_M}m: {jumps}   max-step {step_d.max():.3f}")
    print(f"coverage {cov:.1%}   status {dict(st)}   "
          f"rate {len(stamps) / wall:.1f} steps/s ({wall:.0f}s wall)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6))
    a.plot(gta[:, 0], gta[:, 1], "k-", lw=1, label="GT")
    a.plot(est[:, 0], est[:, 1], "r-", lw=1, alpha=0.75, label="est (W frame)")
    a.plot(*gta[0], "g^", ms=10, label="start (hint)")
    a.set_aspect("equal"); a.legend(); a.set_title("map_relative replay — XY overlay")
    b.plot(e, lw=0.8); b.axhline(GATE_MEAN_M, color="r", ls="--", label=f"gate {GATE_MEAN_M} m")
    b.set_xlabel("frame"); b.set_ylabel("|est−GT| m"); b.legend()
    b.set_title(f"error timeline — mean {e.mean():.3f} m")
    out_png = map_dir / "replay_proof.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=110)
    print(f"plot -> {out_png}")

    ok = e.mean() <= GATE_MEAN_M and jumps == 0
    print("REPLAY PROOF:", "PASS" if ok else "FAIL",
          f"(gate mean<={GATE_MEAN_M}, jumps==0)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
