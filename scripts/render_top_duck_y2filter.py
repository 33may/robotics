"""Render two diagnostic galleries of top-cam teacher-accepted bboxes split by
the y2-filter hypothesis:
  PASS (y2 <= 408): 50 images, prefer distinct episodes (up to 2/ep as fallback)
  FAIL (y2 >  408): 20 images, distinct episodes if possible

Each PNG = RGB+lime-bbox | blue-dominance+lime-bbox + red dashed line at y=408.

Efficiency: pick (episode, frame) rows up front from the parquet, then open
each needed mp4 exactly once per (episode, frame).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

from vbti.logic.dataset import resolve_dataset_path

SEED = 42
N_PASS = 50
N_FAIL = 20
H_IMG = 480
W_IMG = 640
Y2_THR = 0.85 * H_IMG  # 408
SHRINK = math.sqrt(0.5)  # inner 50% area (~0.707 per edge)
DATASET_REPO = "eternalmay33/01_02_03_merged_may-sim"
CLEAN_PARQUET = Path("/home/may33/.cache/vbti/detection_labels_clean.parquet")
GALLERY_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/data_analysis/gallery"
)
OUT_PASS = GALLERY_ROOT / "top_duck_y2pass"
OUT_FAIL = GALLERY_ROOT / "top_duck_y2fail"
FAR_APART_MIN_GAP = 60  # frames — second pick per episode must be far


def shrunk_xyxy(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * SHRINK
    bh = (y2 - y1) * SHRINK
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def blue_dominance(frame_rgb: np.ndarray) -> np.ndarray:
    r = frame_rgb[..., 0].astype(np.int16)
    g = frame_rgb[..., 1].astype(np.int16)
    b = frame_rgb[..., 2].astype(np.int16)
    dom = b - np.maximum(r, g)
    return np.clip(dom, 0, 255).astype(np.uint8)


def clear_out_dir(d: Path) -> None:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        return
    for p in d.iterdir():
        if p.is_file():
            p.unlink()


def sample_distinct_eps(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Pick up to n rows, prefer distinct episodes; fallback up to 2 per ep."""
    rng = np.random.default_rng(seed)
    per_ep = pool.groupby("episode_index", group_keys=False).sample(n=1, random_state=seed)
    if len(per_ep) >= n:
        return per_ep.sample(n=n, random_state=seed).reset_index(drop=True)

    used = set(zip(per_ep["episode_index"].tolist(), per_ep["frame_index"].tolist()))
    extras_rows = []
    ep_order = per_ep["episode_index"].sample(frac=1.0, random_state=seed).tolist()
    first_pick_frame = dict(zip(per_ep["episode_index"], per_ep["frame_index"]))
    for ep in ep_order:
        if len(per_ep) + len(extras_rows) >= n:
            break
        sub = pool[pool["episode_index"] == ep]
        f1 = int(first_pick_frame[ep])
        sub2 = sub[abs(sub["frame_index"] - f1) >= FAR_APART_MIN_GAP]
        sub2 = sub2[~sub2.apply(
            lambda r: (int(r["episode_index"]), int(r["frame_index"])) in used, axis=1
        )]
        if len(sub2) == 0:
            continue
        pick = sub2.sample(n=1, random_state=int(rng.integers(0, 10_000_000)))
        extras_rows.append(pick)
        used.add((int(pick["episode_index"].iloc[0]), int(pick["frame_index"].iloc[0])))
    if extras_rows:
        picked = pd.concat([per_ep, pd.concat(extras_rows)], ignore_index=True)
    else:
        picked = per_ep.copy()
    if len(picked) > n:
        picked = picked.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        picked = picked.reset_index(drop=True)
    return picked


def render_one(
    row: pd.Series,
    chunk_col: str,
    file_col: str,
    from_ts_col: str,
    ds_path: Path,
    cam_key: str,
    fps: float,
    out_dir: Path,
) -> dict | None:
    ep = int(row["episode_index"])
    fi = int(row["frame_index"])
    conf = float(row["top_duck_conf"])
    x1n = float(row["top_duck_x1"])
    y1n = float(row["top_duck_y1"])
    x2n = float(row["top_duck_x2"])
    y2n = float(row["top_duck_y2"])
    if not all(np.isfinite([x1n, y1n, x2n, y2n])):
        print(f"[skip] ep{ep:04d} fr{fi:04d}: bbox NaN")
        return None

    chunk_idx = int(row[chunk_col])
    file_idx = int(row[file_col])
    from_ts = float(row[from_ts_col])
    start_frame = int(round(from_ts * fps))
    tgt_frame = start_frame + fi

    video_path = (ds_path / "videos" / cam_key
                  / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")
    if not video_path.exists():
        print(f"[skip] ep{ep:04d} fr{fi:04d}: video missing {video_path}")
        return None

    try:
        dec = VideoDecoder(str(video_path))
        frame = dec[tgt_frame]
    except Exception as e:
        print(f"[skip] ep{ep:04d} fr{fi:04d}: torchcodec failed @ {tgt_frame}: {e}")
        return None
    rgb = frame.permute(1, 2, 0).cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    h, w = rgb.shape[:2]

    x1 = float(np.clip(x1n * w, 0, w - 1))
    y1 = float(np.clip(y1n * h, 0, h - 1))
    x2 = float(np.clip(x2n * w, 0, w - 1))
    y2 = float(np.clip(y2n * h, 0, h - 1))

    sx1n, sy1n, sx2n, sy2n = shrunk_xyxy(x1n, y1n, x2n, y2n)
    sx1 = int(np.clip(round(sx1n * w), 0, w - 1))
    sy1 = int(np.clip(round(sy1n * h), 0, h - 1))
    sx2 = int(np.clip(round(sx2n * w), 0, w))
    sy2 = int(np.clip(round(sy2n * h), 0, h))
    if sx2 <= sx1 or sy2 <= sy1:
        b_minus_max = float("nan")
        b_over_g = float("nan")
    else:
        patch = rgb[sy1:sy2, sx1:sx2].astype(np.float32)
        mean_rgb = patch.reshape(-1, 3).mean(axis=0)
        mg = float(mean_rgb[1])
        mb = float(mean_rgb[2])
        r_arr = patch[..., 0]
        g_arr = patch[..., 1]
        b_arr = patch[..., 2]
        dom_patch = np.clip(b_arr - np.maximum(r_arr, g_arr), 0, 255)
        b_minus_max = float(dom_patch.mean())
        b_over_g = mb / mg if mg > 0 else float("nan")

    dom_full = blue_dominance(rgb)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    ax_l, ax_r = axes[0], axes[1]
    ax_l.imshow(rgb)
    ax_r.imshow(dom_full, cmap="gray", vmin=0, vmax=255)
    for ax in (ax_l, ax_r):
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#00ff00", facecolor="none",
        )
        ax.add_patch(rect)
        ax.axhline(y=Y2_THR, color="red", linestyle="--", linewidth=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_l.set_title("RGB + teacher bbox", fontsize=9)
    ax_r.set_title("B - max(R,G) + y=408 thr", fontsize=9)
    fig.suptitle(
        f"ep{ep:04d} fr{fi:04d} | conf={conf:.3f} | y2={y2:.0f} | "
        f"BmaxRG_shrunk={b_minus_max:.1f} | B/G_shrunk={b_over_g:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    out_path = out_dir / f"ep{ep:04d}_fr{fi:04d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "ep": ep, "fi": fi, "conf": conf, "y2": y2,
        "b_minus_maxrg_shrunk": b_minus_max, "b_over_g_shrunk": b_over_g,
        "filename": out_path.name,
    }


def main() -> None:
    clear_out_dir(OUT_PASS)
    clear_out_dir(OUT_FAIL)

    # ---- Load clean parquet + filter accepted ----
    clean = pd.read_parquet(CLEAN_PARQUET)
    mask = (clean["top_duck_trust"] == 1) & (clean["top_duck_conf"] > 0.08)
    accepted = clean.loc[mask, ["episode_index", "frame_index", "top_duck_conf"]].copy()
    print(f"[info] accepted rows: {len(accepted):,}")

    # ---- Merge bboxes from detection_results.parquet ----
    ds_path = resolve_dataset_path(DATASET_REPO)
    det = pd.read_parquet(ds_path / "detection_results.parquet")
    det = det[["episode_index", "frame_index",
               "top_duck_x1", "top_duck_y1", "top_duck_x2", "top_duck_y2"]].copy()
    det = det.dropna(subset=["top_duck_x1", "top_duck_y1", "top_duck_x2", "top_duck_y2"])

    accepted_with_bbox = accepted.merge(det, on=["episode_index", "frame_index"], how="inner")
    print(f"[info] accepted rows with real bbox: {len(accepted_with_bbox):,}  "
          f"distinct eps: {accepted_with_bbox['episode_index'].nunique()}")

    # y2 in pixel coords (denormalize to H_IMG)
    accepted_with_bbox["y2_px"] = accepted_with_bbox["top_duck_y2"] * H_IMG
    pass_pool = accepted_with_bbox[accepted_with_bbox["y2_px"] <= Y2_THR].copy()
    fail_pool = accepted_with_bbox[accepted_with_bbox["y2_px"] > Y2_THR].copy()
    print(f"[info] PASS pool: {len(pass_pool):,} rows, "
          f"{pass_pool['episode_index'].nunique()} distinct eps")
    print(f"[info] FAIL pool: {len(fail_pool):,} rows, "
          f"{fail_pool['episode_index'].nunique()} distinct eps")

    picked_pass = sample_distinct_eps(pass_pool, N_PASS, SEED)
    picked_fail = sample_distinct_eps(fail_pool, N_FAIL, SEED)
    print(f"[info] picked PASS={len(picked_pass)}, FAIL={len(picked_fail)}")

    # ---- Episode video meta ----
    ep_files = sorted((ds_path / "meta" / "episodes").rglob("*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    cam_key = "observation.images.top"
    chunk_col = f"videos/{cam_key}/chunk_index"
    file_col = f"videos/{cam_key}/file_index"
    from_ts_col = f"videos/{cam_key}/from_timestamp"
    ep_meta_small = ep_meta[["episode_index", chunk_col, file_col, from_ts_col]].copy()

    picked_pass = picked_pass.merge(ep_meta_small, on="episode_index", how="left")
    picked_fail = picked_fail.merge(ep_meta_small, on="episode_index", how="left")

    with open(ds_path / "meta" / "info.json") as f:
        info = json.load(f)
    fps = float(info["fps"])
    feat = info["features"][cam_key]
    vid_w = int(feat["info"]["video.width"])
    vid_h = int(feat["info"]["video.height"])
    print(f"[info] fps={fps} top={vid_w}x{vid_h}")

    picked_pass = picked_pass.sort_values(by=["episode_index", "frame_index"]).reset_index(drop=True)
    picked_fail = picked_fail.sort_values(by=["episode_index", "frame_index"]).reset_index(drop=True)

    # ---- Render both sets ----
    for label, picked, out_dir in [
        ("PASS", picked_pass, OUT_PASS),
        ("FAIL", picked_fail, OUT_FAIL),
    ]:
        records: list[dict] = []
        for _, row in picked.iterrows():
            rec = render_one(row, chunk_col, file_col, from_ts_col,
                             ds_path, cam_key, fps, out_dir)
            if rec is not None:
                rec["set"] = label
                records.append(rec)
                print(f"[{label}] {rec['filename']}  conf={rec['conf']:.3f} "
                      f"y2={rec['y2']:.0f}")
        pd.DataFrame(records).to_csv(out_dir / "_metrics.csv", index=False)
        print(f"[done] {label}: {len(records)} images -> {out_dir}")


if __name__ == "__main__":
    main()
