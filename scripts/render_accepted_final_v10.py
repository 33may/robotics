"""Final v10 acceptance gallery — 100 stratified frames across 8 (cam, obj) combos.

Filter stack:
  - v8 trust: {cam}_{obj}_trust == 1
  - v9: top_duck additionally requires bbox y2 <= 408 (in pixel coords on 480H frame)
  - v10 NEW: for cams `left` and `right`, duck only:
        if bbox height > width, set y1 = y2 - w  (crop top; bbox becomes w x w, anchored to bottom)

Random seed 42, stratified: 13/12/12/13/13/12/12/13 across 8 combos -> 100.
Prefer distinct episodes within each combo.

Efficiency: pick rows up front, open each unique (cam, chunk, file) video only once.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2  # noqa: F401  (kept at top-level for pyright cleanliness, even though unused)
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

from vbti.logic.dataset import resolve_dataset_path


SEED = 42
H_IMG = 480
W_IMG = 640
Y2_THR_NORM = 408.0 / H_IMG  # 0.85
DATASET_REPO = "eternalmay33/01_02_03_merged_may-sim"
CLEAN_PARQUET = Path("/home/may33/.cache/vbti/detection_labels_clean.parquet")
GALLERY_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/data_analysis/gallery"
)
OUT_DIR = GALLERY_ROOT / "accepted_final_v10"

CAMS = ["left", "right", "top", "gripper"]
OBJS = ["duck", "cup"]
# Order: 13/12/12/13/13/12/12/13 = 100
COMBOS: list[tuple[str, str]] = [
    ("left", "duck"),    # 13
    ("left", "cup"),     # 12
    ("right", "duck"),   # 12
    ("right", "cup"),    # 13
    ("top", "duck"),     # 13
    ("top", "cup"),      # 12
    ("gripper", "duck"), # 12
    ("gripper", "cup"),  # 13
]
COUNTS: dict[tuple[str, str], int] = {
    ("left", "duck"): 13,
    ("left", "cup"): 12,
    ("right", "duck"): 12,
    ("right", "cup"): 13,
    ("top", "duck"): 13,
    ("top", "cup"): 12,
    ("gripper", "duck"): 12,
    ("gripper", "cup"): 13,
}


def clear_out_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def apply_v10_square(
    cam: str, obj: str, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float, bool]:
    """For left/right cams, duck only: if h > w, crop top so bbox becomes w x w anchored at bottom.
    Returns (x1, y1, x2, y2, was_squared).
    """
    if cam not in ("left", "right") or obj != "duck":
        return x1, y1, x2, y2, False
    w = x2 - x1
    h = y2 - y1
    if h > w:
        y1 = y2 - w
        return x1, y1, x2, y2, True
    return x1, y1, x2, y2, False


def decode_frame(
    ds_path: Path,
    cam_key: str,
    chunk_idx: int,
    file_idx: int,
    tgt_frame: int,
    decoder_cache: dict[tuple[str, int, int], VideoDecoder],
) -> np.ndarray | None:
    video_path = (
        ds_path / "videos" / cam_key
        / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    )
    if not video_path.exists():
        print(f"[skip] video missing: {video_path}")
        return None
    key = (cam_key, chunk_idx, file_idx)
    dec = decoder_cache.get(key)
    if dec is None:
        try:
            dec = VideoDecoder(str(video_path))
        except Exception as e:
            print(f"[skip] decoder open failed {video_path.name}: {e}")
            return None
        decoder_cache[key] = dec
    try:
        frame = dec[tgt_frame]
    except Exception as e:
        print(f"[skip] decode failed @ {tgt_frame} in {video_path.name}: {e}")
        return None
    rgb = frame.permute(1, 2, 0).cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def render_one(
    rgb: np.ndarray,
    cam: str,
    obj: str,
    ep: int,
    fr: int,
    conf: float,
    bbox_px: tuple[float, float, float, float],
    squared: bool,
    out_path: Path,
) -> None:
    x1, y1, x2, y2 = bbox_px
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=150)
    ax.imshow(rgb)
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="#00ff00", facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    squared_tag = " [squared]" if squared else ""
    fig.suptitle(
        f"{cam}/{obj} ep{ep:04d} fr{fr:04d} conf={conf:.3f}{squared_tag}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    clear_out_dir(OUT_DIR)

    clean = pd.read_parquet(CLEAN_PARQUET)
    print(f"[info] clean parquet: {len(clean):,} rows")

    ds_path = resolve_dataset_path(DATASET_REPO)
    det = pd.read_parquet(ds_path / "detection_results.parquet")

    # Compute accepted row counts per (cam, obj) under the full v10 mask
    per_combo_accept: dict[tuple[str, str], int] = {}
    for cam, obj in COMBOS:
        trust_col = f"{cam}_{obj}_trust"
        mask = clean[trust_col] == 1
        if (cam, obj) == ("top", "duck"):
            sub = clean.loc[mask, ["episode_index", "frame_index"]].copy()
            det_td = det[["episode_index", "frame_index", "top_duck_y2"]]
            merged = sub.merge(det_td, on=["episode_index", "frame_index"], how="left")
            # BBOX-ONLY filter interpretation: drop only rows where bbox exists AND y2>0.85
            drop = merged["top_duck_y2"].notna() & (merged["top_duck_y2"] > Y2_THR_NORM)
            per_combo_accept[(cam, obj)] = int((~drop).sum())
        else:
            per_combo_accept[(cam, obj)] = int(mask.sum())

    for combo, n in per_combo_accept.items():
        print(f"[info] accept {combo[0]}/{combo[1]}: {n:,}")

    # Episode meta
    ep_files = sorted((ds_path / "meta" / "episodes").rglob("*.parquet"))
    ep_meta_list: list[pd.DataFrame] = [pd.read_parquet(p) for p in ep_files]
    ep_meta = pd.concat(ep_meta_list, ignore_index=True)

    with open(ds_path / "meta" / "info.json") as f:
        info = json.load(f)
    fps = float(info["fps"])

    # Cam key map
    cam_key_map: dict[str, str] = {}
    for cam in CAMS:
        candidates = [f"observation.images.{cam}", f"observation.images.cam_{cam}"]
        for c in candidates:
            if c in info["features"]:
                cam_key_map[cam] = c
                break
        if cam not in cam_key_map:
            raise KeyError(f"cam {cam} not in info.json features; tried {candidates}")
    print(f"[info] cam_key_map: {cam_key_map}")

    # Pick rows per combo
    all_picks: list[pd.DataFrame] = []
    for combo in COMBOS:
        cam, obj = combo
        trust_col = f"{cam}_{obj}_trust"
        conf_col = f"{cam}_{obj}_conf"
        base_mask = clean[trust_col] == 1
        base = clean.loc[base_mask, ["episode_index", "frame_index", conf_col]].copy()
        base = base.rename(columns={conf_col: "conf"})

        bbox_cols = [f"{cam}_{obj}_x1", f"{cam}_{obj}_y1",
                     f"{cam}_{obj}_x2", f"{cam}_{obj}_y2"]
        det_sub = det[["episode_index", "frame_index"] + bbox_cols].dropna(
            subset=[f"{cam}_{obj}_x1", f"{cam}_{obj}_y1",
                    f"{cam}_{obj}_x2", f"{cam}_{obj}_y2"]
        )
        merged = base.merge(det_sub, on=["episode_index", "frame_index"], how="inner")

        if (cam, obj) == ("top", "duck"):
            merged = merged[merged[f"{cam}_{obj}_y2"] <= Y2_THR_NORM].copy()

        merged = merged.rename(columns={
            f"{cam}_{obj}_x1": "x1n",
            f"{cam}_{obj}_y1": "y1n",
            f"{cam}_{obj}_x2": "x2n",
            f"{cam}_{obj}_y2": "y2n",
        })

        n = COUNTS[combo]
        pool = merged
        if len(pool) == 0:
            print(f"[warn] empty pool for {cam}/{obj}")
            continue

        # Prefer distinct episodes: first one-per-ep, then fill if short
        distinct = pool.groupby("episode_index", as_index=False).sample(
            n=1, random_state=SEED
        )
        if len(distinct) >= n:
            picked = distinct.sample(n=n, random_state=SEED).reset_index(drop=True)
        else:
            taken_keys = set(
                zip(distinct["episode_index"].tolist(), distinct["frame_index"].tolist())
            )
            leftover_mask = ~pool.set_index(
                ["episode_index", "frame_index"]
            ).index.isin(taken_keys)
            leftover = pool.loc[leftover_mask]
            need = n - len(distinct)
            if len(leftover) >= need:
                extras = leftover.sample(n=need, random_state=SEED)
                picked = pd.concat([distinct, extras], ignore_index=True)
            else:
                picked = pd.concat([distinct, leftover], ignore_index=True)
        picked = picked.reset_index(drop=True)
        picked["cam"] = cam
        picked["obj"] = obj
        print(f"[pick] {cam}/{obj}: {len(picked)}/{n} (pool={len(pool)})")
        all_picks.append(picked)

    picks = pd.concat(all_picks, ignore_index=True)

    # Decode + render
    records: list[dict[str, object]] = []
    squared_count_leftright = 0
    sample_leftright_total = 0
    decoder_cache: dict[tuple[str, int, int], VideoDecoder] = {}

    for cam in CAMS:
        cam_key = cam_key_map[cam]
        chunk_col = f"videos/{cam_key}/chunk_index"
        file_col = f"videos/{cam_key}/file_index"
        from_ts_col = f"videos/{cam_key}/from_timestamp"
        if chunk_col not in ep_meta.columns:
            print(f"[warn] missing ep_meta col {chunk_col}")
            continue
        ep_meta_sm = ep_meta[["episode_index", chunk_col, file_col, from_ts_col]].copy()
        ep_meta_sm = ep_meta_sm.rename(columns={
            chunk_col: "chunk_idx",
            file_col: "file_idx",
            from_ts_col: "from_ts",
        })

        sub = picks[picks["cam"] == cam].merge(
            ep_meta_sm, on="episode_index", how="left"
        )
        for (chunk_idx_g, file_idx_g), grp in sub.groupby(["chunk_idx", "file_idx"]):
            chunk_idx = int(chunk_idx_g)  # type: ignore[arg-type]
            file_idx = int(file_idx_g)    # type: ignore[arg-type]
            grp = grp.sort_values(by="frame_index")
            for i in range(len(grp)):
                row = grp.iloc[i]
                ep = int(row["episode_index"])
                fr = int(row["frame_index"])
                conf = float(row["conf"])
                from_ts = float(row["from_ts"])
                start_frame = int(round(from_ts * fps))
                tgt_frame = start_frame + fr
                rgb = decode_frame(
                    ds_path, cam_key, chunk_idx, file_idx, tgt_frame, decoder_cache
                )
                if rgb is None:
                    continue
                h, w = rgb.shape[:2]
                x1n, y1n, x2n, y2n = (
                    float(row["x1n"]), float(row["y1n"]),
                    float(row["x2n"]), float(row["y2n"]),
                )
                x1 = float(np.clip(x1n * w, 0, w - 1))
                y1 = float(np.clip(y1n * h, 0, h - 1))
                x2 = float(np.clip(x2n * w, 0, w - 1))
                y2 = float(np.clip(y2n * h, 0, h - 1))

                # v10 delta
                x1, y1, x2, y2, squared = apply_v10_square(cam, str(row["obj"]), x1, y1, x2, y2)
                x1 = float(np.clip(x1, 0, w - 1))
                y1 = float(np.clip(y1, 0, h - 1))
                x2 = float(np.clip(x2, 0, w - 1))
                y2 = float(np.clip(y2, 0, h - 1))

                if cam in ("left", "right"):
                    sample_leftright_total += 1
                    if squared:
                        squared_count_leftright += 1

                out_path = OUT_DIR / f"{cam}_{row['obj']}_ep{ep:04d}_fr{fr:04d}.png"
                render_one(
                    rgb, cam, str(row["obj"]), ep, fr, conf,
                    (x1, y1, x2, y2), squared, out_path,
                )
                records.append({
                    "cam": cam, "obj": str(row["obj"]),
                    "ep": ep, "fr": fr, "conf": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "squared": bool(squared),
                    "filename": out_path.name,
                })
                print(f"[save] {out_path.name} conf={conf:.3f} squared={squared}")

    pd.DataFrame(records).to_csv(OUT_DIR / "_metrics.csv", index=False)

    stats = {
        "per_combo_accept_v10_mask": {
            f"{c}_{o}": n for (c, o), n in per_combo_accept.items()
        },
        "n_rendered": len(records),
        "sample_leftright_total": sample_leftright_total,
        "squared_count_leftright": squared_count_leftright,
    }
    with open(OUT_DIR / "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] {len(records)} PNGs -> {OUT_DIR}")
    print(f"[done] left/right squared: {squared_count_leftright}/{sample_leftright_total}")


if __name__ == "__main__":
    main()
