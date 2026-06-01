"""Quick D405 gripper depth+RGB sample capture for v016 clip-range tuning.

Connects directly to the gripper RealSense D405 via pyrealsense2 (NOT the
lerobot camera abstraction — bypassed for speed). Captures ``--seconds`` of
synchronized color + depth frames at 30 fps, aligned to the color frame,
and saves:
  - <out>/depth_uint16.npy   (T, H, W) uint16 millimeters
  - <out>/color_uint8.npy    (T, H, W, 3) uint8 RGB
  - <out>/preview_<i>.png    a few sanity-check overlays
  - <out>/histogram.png      pooled depth distribution (mm)

Usage:
    conda run -n lerobot python -m vbti.logic.depth.capture_gripper_sample \\
        --serial 123622270367 \\
        --seconds 5 \\
        --out /home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v016/data/d405_gripper_sample
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="123622270367", help="D405 serial (gripper)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ── pipeline ──
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    # depth scale (m per uint16 unit)
    sensor = profile.get_device().first_depth_sensor()
    depth_scale = sensor.get_depth_scale()
    print(f"[device] D405 serial={args.serial}  depth_scale={depth_scale} m/unit")
    print(f"[capture] {args.seconds}s @ {args.fps} fps  → ~{int(args.seconds*args.fps)} frames")
    print("[capture] move gripper around duck/cup scene now")

    # warmup
    for _ in range(10):
        pipe.wait_for_frames()

    depth_buf, color_buf = [], []
    n_target = int(args.seconds * args.fps)
    t0 = time.perf_counter()
    while len(depth_buf) < n_target:
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        cf = aligned.get_color_frame()
        if not df or not cf:
            continue
        depth_buf.append(np.asanyarray(df.get_data()).copy())
        color_buf.append(np.asanyarray(cf.get_data()).copy())
    dt = time.perf_counter() - t0
    pipe.stop()

    depth = np.stack(depth_buf, axis=0)
    color = np.stack(color_buf, axis=0)
    print(f"[done] {len(depth)} frames in {dt:.1f}s ({len(depth)/dt:.1f} fps)")

    np.save(out / "depth_uint16.npy", depth)
    np.save(out / "color_uint8.npy", color)

    # convert depth uint16 → meters
    depth_m = depth.astype(np.float32) * depth_scale
    valid = depth_m[depth_m > 0]

    # stats
    if valid.size:
        pcts = np.percentile(valid, [1, 5, 25, 50, 75, 95, 99])
        print(
            f"[stats] valid={valid.size/depth_m.size:.1%}  "
            f"min={valid.min():.3f}  p1={pcts[0]:.3f}  p5={pcts[1]:.3f}  "
            f"p50={pcts[3]:.3f}  p95={pcts[5]:.3f}  p99={pcts[6]:.3f}  max={valid.max():.3f}  "
            f"(meters)"
        )

    # histogram + previews
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(valid, bins=80, range=(0.0, min(2.0, float(valid.max()) if valid.size else 2.0)))
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("pixel count")
    ax.set_title(f"D405 gripper depth — {len(depth)} frames @ {args.fps} fps")
    fig.tight_layout()
    fig.savefig(out / "histogram.png", dpi=120)
    plt.close(fig)

    n_prev = min(6, len(depth))
    idxs = np.linspace(0, len(depth) - 1, n_prev).astype(int)
    for i, k in enumerate(idxs):
        d_m = depth_m[k]
        clip = np.clip(d_m, 0.05, 0.5)
        u8 = ((clip - 0.05) / 0.45 * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        side = np.hstack([color[k], rgb])
        cv2.imwrite(str(out / f"preview_{i:02d}.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

    print(f"[out] {out}")


if __name__ == "__main__":
    main()
