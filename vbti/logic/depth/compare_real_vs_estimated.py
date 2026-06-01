"""exp02b — match real D405 vs DA-V2-Indoor estimated depth.

Workflow:
  1. Load a D405 capture produced by ``capture_gripper_sample`` (depth_uint16.npy +
     color_uint8.npy).
  2. Sample N frames of gripper RGB from an existing may-sim LeRobot dataset.
  3. Run Depth Anything v2 metric indoor (Small) on those RGB frames.
  4. Compute per-source histograms (real D405 vs estimated).
  5. Pick clip ranges from data: [p5, p95] of each source.
  6. Render a 2-column side-by-side panel: real (RGB | depth-colorized) and
     estimated (RGB | depth-colorized) using each source's own clip — so the
     visual character of the two streams can be compared.

Decision aim: pick ONE shared rescale strategy for the v016 backfill so that
estimated depth (backfilled into v013/v014/v015) and real depth (newly
captured) look visually consistent to the SmolVLA encoder.

Usage:
    conda run -n lerobot python -m vbti.logic.depth.compare_real_vs_estimated \\
        --d405-dir <v016/data/d405_gripper_sample> \\
        --src-dataset eternalmay33/01_02_03_merged_may-sim_detection \\
        --out <v016/results/exp02b_real_vs_estimated>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def turbo_colorize(depth_m: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    span = max(clip_max - clip_min, 1e-6)
    norm = np.clip((depth_m - clip_min) / span, 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d405-dir", required=True)
    ap.add_argument("--src-dataset", default="eternalmay33/01_02_03_merged_may-sim_detection")
    ap.add_argument("--src-root", default=None)
    ap.add_argument("--gripper-key", default="observation.images.gripper")
    ap.add_argument("--n-est-samples", type=int, default=12, help="Frames to keep after close-range filtering")
    ap.add_argument("--n-candidates", type=int, default=60,
                    help="Oversample this many evenly-spaced candidate frames before filtering")
    ap.add_argument("--max-median-depth-m", type=float, default=0.85,
                    help="Keep only frames whose DA-V2 median depth is below this — selects close-range/approach frames")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    # ── load D405 sample ──
    d_dir = Path(args.d405_dir).expanduser().resolve()
    print(f"[load] D405 sample {d_dir}")
    depth_u16 = np.load(d_dir / "depth_uint16.npy")
    color_u8 = np.load(d_dir / "color_uint8.npy")

    # depth_scale was printed at capture time but not saved — recompute from device default
    # D405 depth scale is 0.0001 m/unit (1e-4)
    depth_scale = 1e-4
    depth_m = depth_u16.astype(np.float32) * depth_scale
    valid_real = depth_m[depth_m > 0]
    print(f"[real]  frames={len(depth_u16)}  valid_pixels={valid_real.size:,}")

    # ── load source dataset, sample frames, run DA-V2 ──
    import torch
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from transformers import pipeline as hf_pipeline

    print(f"[load] {args.src_dataset}")
    src = LeRobotDataset(args.src_dataset, root=args.src_root)

    n_cand = min(args.n_candidates, len(src))
    cand_idxs = np.linspace(0, len(src) - 1, n_cand).astype(int)
    print(f"[depth] {args.model} on {args.device}")
    print(f"[scan]  oversampling {n_cand} candidates → keep {args.n_est_samples} with median depth < {args.max_median_depth_m}m")
    pipe = hf_pipeline(task="depth-estimation", model=args.model, device=args.device)

    cand_records = []  # (median_m, depth_arr, rgb_arr, src_idx)
    for i in cand_idxs:
        item = src[int(i)]
        rgb = item[args.gripper_key]
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        if rgb.ndim == 3 and rgb.shape[0] in (1, 3) and rgb.shape[-1] not in (1, 3):
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        with torch.no_grad():
            r = pipe(Image.fromarray(rgb))
        if "predicted_depth" in r:
            pd = r["predicted_depth"]
            d = pd.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(pd) else np.asarray(pd, dtype=np.float32)
            if d.ndim == 3:
                d = d[0]
        else:
            d = np.asarray(r["depth"], dtype=np.float32)
        if d.shape != rgb.shape[:2]:
            d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        cand_records.append((float(np.median(d)), d, rgb, int(i)))

    # filter by close-range, then if too few, take the closest n_est_samples regardless
    close = [r for r in cand_records if r[0] < args.max_median_depth_m]
    if len(close) >= args.n_est_samples:
        close.sort(key=lambda r: r[0])
        kept = close[: args.n_est_samples]
    else:
        cand_records.sort(key=lambda r: r[0])
        kept = cand_records[: args.n_est_samples]
        print(f"[scan]  only {len(close)} below threshold; using closest {args.n_est_samples} regardless")

    kept_idxs = [r[3] for r in kept]
    kept_meds = [r[0] for r in kept]
    print(f"[scan]  kept median-depth range: {min(kept_meds):.2f}–{max(kept_meds):.2f} m  src_idxs={kept_idxs[:6]}...")
    est_depth_m = np.stack([r[1] for r in kept], axis=0)
    est_rgb_list = [r[2] for r in kept]
    valid_est = est_depth_m.flatten()
    print(f"[est]   frames={len(est_depth_m)}  pixels={valid_est.size:,}")

    # ── stats ──
    def stats(x: np.ndarray) -> dict:
        if x.size == 0:
            return {}
        p = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
        return {
            "min": float(x.min()), "p1": float(p[0]), "p5": float(p[1]),
            "p25": float(p[2]), "p50": float(p[3]), "p75": float(p[4]),
            "p95": float(p[5]), "p99": float(p[6]), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std()),
        }

    s_real = stats(valid_real)
    s_est = stats(valid_est)
    print(f"[real]  {s_real}")
    print(f"[est]   {s_est}")

    # data-driven clip = [p5, p95]
    clip_real = (s_real["p5"], s_real["p95"])
    clip_est = (s_est["p5"], s_est["p95"])
    print(f"[clip]  real={clip_real}  est={clip_est}  (m, [p5,p95])")

    summary = {
        "model": args.model,
        "n_real_frames": int(len(depth_u16)),
        "n_est_frames": int(len(est_depth_m)),
        "max_median_depth_m_filter": args.max_median_depth_m,
        "kept_src_idxs": kept_idxs,
        "kept_median_depths_m": kept_meds,
        "real_stats_m": s_real,
        "est_stats_m": s_est,
        "suggested_clip_real_m": clip_real,
        "suggested_clip_est_m": clip_est,
        "rescale_strategy": "per-source linear to [0,1] using suggested_clip then shared turbo colormap",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── histogram plot ──
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    real_top = min(2.0, float(valid_real.max()) if valid_real.size else 2.0)
    axes[0].hist(valid_real, bins=80, range=(0.0, real_top), color="C0")
    axes[0].axvline(clip_real[0], color="k", ls="--", lw=1)
    axes[0].axvline(clip_real[1], color="k", ls="--", lw=1)
    axes[0].set_title(f"Real D405 — clip [p5,p95] = [{clip_real[0]:.2f}, {clip_real[1]:.2f}] m")
    axes[0].set_xlabel("depth (m)")
    est_top = min(3.5, float(valid_est.max()))
    axes[1].hist(valid_est, bins=80, range=(0.0, est_top), color="C1")
    axes[1].axvline(clip_est[0], color="k", ls="--", lw=1)
    axes[1].axvline(clip_est[1], color="k", ls="--", lw=1)
    axes[1].set_title(f"DA-V2 estimated — clip [p5,p95] = [{clip_est[0]:.2f}, {clip_est[1]:.2f}] m")
    axes[1].set_xlabel("depth (m)")
    fig.tight_layout()
    fig.savefig(plots / "histograms.png", dpi=130)
    plt.close(fig)

    # ── side-by-side panels ──
    n_panel = min(6, len(est_depth_m), len(depth_m))
    real_idxs = np.linspace(0, len(depth_m) - 1, n_panel).astype(int)
    est_idxs = np.linspace(0, len(est_depth_m) - 1, n_panel).astype(int)

    for k, (ri, ei) in enumerate(zip(real_idxs, est_idxs)):
        real_rgb = color_u8[ri]
        real_dvis = turbo_colorize(depth_m[ri], *clip_real)
        est_rgb = est_rgb_list[ei]
        est_dvis = turbo_colorize(est_depth_m[ei], *clip_est)

        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        axes[0, 0].imshow(real_rgb)
        axes[0, 0].set_title("Real D405 — RGB")
        axes[0, 1].imshow(real_dvis)
        axes[0, 1].set_title(f"Real D405 — depth (clip {clip_real[0]:.2f}-{clip_real[1]:.2f}m)")
        axes[1, 0].imshow(est_rgb)
        axes[1, 0].set_title("may-sim — RGB")
        axes[1, 1].imshow(est_dvis)
        axes[1, 1].set_title(f"DA-V2 estimated — depth (clip {clip_est[0]:.2f}-{clip_est[1]:.2f}m)")
        for a in axes.flat:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(plots / f"panel_{k:02d}.png", dpi=110)
        plt.close(fig)

    print(f"[out] {out}")
    print("[hint] inspect plots/histograms.png and plots/panel_*.png")


if __name__ == "__main__":
    main()
