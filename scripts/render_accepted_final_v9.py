"""Final v9 acceptance gallery — 100 stratified frames across 8 (cam, obj) combos
under full v8 trust filter + new top_duck y2 <= 408 rule.

Random seed 42, stratified: 13 each from 4 combos + 12 each from 4 combos -> 100.
Prefer distinct episodes within each combo.

Efficiency: pick rows up front, open each unique (cam, episode) video only once.
"""
from __future__ import annotations

import json
import shutil
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
H_IMG = 480
W_IMG = 640
Y2_THR_NORM = 408.0 / H_IMG  # 0.85
DATASET_REPO = "eternalmay33/01_02_03_merged_may-sim"
CLEAN_PARQUET = Path("/home/may33/.cache/vbti/detection_labels_clean.parquet")
GALLERY_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/data_analysis/gallery"
)
OUT_DIR = GALLERY_ROOT / "accepted_final_v9"

CAMS = ["left", "right", "top", "gripper"]
OBJS = ["duck", "cup"]
COMBOS = [(c, o) for c in CAMS for o in OBJS]  # 8 combos
# 4 combos get 13, 4 get 12 -> 100 total. Give 13 to first 4 deterministically.
COUNTS = {combo: (13 if i < 4 else 12) for i, combo in enumerate(COMBOS)}


def clear_out_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def pick_rows_for_combo(
    pool: pd.DataFrame, n: int, seed: int
) -> pd.DataFrame:
    """Pick n rows, prefer distinct episodes; fallback to duplicates per ep."""
    if len(pool) == 0:
        return pool.copy()
    # One row per distinct episode first
    per_ep = pool.groupby("episode_index").sample(n=1, random_state=seed)
    if len(per_ep) >= n:
        return per_ep.sample(n=n, random_state=seed).reset_index(drop=True)
    # Need extras
    taken = set(zip(per_ep["episode_index"].tolist(), per_ep["frame_index"].tolist()))
    leftover = pool[~pool.apply(
        lambda r: (int(r["episode_index"]), int(r["frame_index"])) in taken, axis=1
    )]
    need = n - len(per_ep)
    if len(leftover) == 0:
        return per_ep.reset_index(drop=True)
    extras = leftover.sample(n=min(need, len(leftover)), random_state=seed)
    out = pd.concat([per_ep, extras], ignore_index=True)
    return out.reset_index(drop=True)


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
    bbox_norm: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    h, w = rgb.shape[:2]
    x1n, y1n, x2n, y2n = bbox_norm
    x1 = float(np.clip(x1n * w, 0, w - 1))
    y1 = float(np.clip(y1n * h, 0, h - 1))
    x2 = float(np.clip(x2n * w, 0, w - 1))
    y2 = float(np.clip(y2n * h, 0, h - 1))

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=150)
    ax.imshow(rgb)
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="#00ff00", facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(
        f"{cam}/{obj} ep{ep:04d} fr{fr:04d} conf={conf:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    clear_out_dir(OUT_DIR)

    # Load clean parquet
    clean = pd.read_parquet(CLEAN_PARQUET)
    print(f"[info] clean parquet: {len(clean):,} rows")

    # Load detection_results for bboxes
    ds_path = resolve_dataset_path(DATASET_REPO)
    det = pd.read_parquet(ds_path / "detection_results.parquet")

    # Compute accepted counts per (cam, obj) + top_duck before/after y2 filter
    per_combo_accept: dict[tuple[str, str], int] = {}
    top_duck_before = int((clean["top_duck_trust"] == 1).sum())

    # Two interpretations of the top_duck y2 filter:
    #  A) STRICT: drop ALL rows that don't have y2 <= 0.85 (NaN bbox treated as reject)
    #  B) BBOX-ONLY: drop ONLY rows where bbox exists AND y2 > 0.85
    # Rationale: trust=1 rows without bbox (~87k) are center-only teacher labels.
    # They can't be arm-lock FPs (no bbox = no arm-lock). So B) is the right read.
    sub = clean.loc[clean["top_duck_trust"] == 1, ["episode_index", "frame_index"]].copy()
    det_td = det[["episode_index", "frame_index", "top_duck_y2"]]
    td_merge = sub.merge(det_td, on=["episode_index", "frame_index"], how="left")

    td_with_bbox = int(td_merge["top_duck_y2"].notna().sum())
    td_without_bbox = int(td_merge["top_duck_y2"].isna().sum())
    td_bbox_y2ok = int((td_merge["top_duck_y2"].notna() & (td_merge["top_duck_y2"] <= Y2_THR_NORM)).sum())
    td_bbox_y2fail = int((td_merge["top_duck_y2"].notna() & (td_merge["top_duck_y2"] > Y2_THR_NORM)).sum())
    top_duck_strict = td_bbox_y2ok  # only bbox+y2ok kept
    top_duck_bbox_only = top_duck_before - td_bbox_y2fail  # drop only bbox y2 fails

    for cam, obj in COMBOS:
        trust_col = f"{cam}_{obj}_trust"
        mask = clean[trust_col] == 1
        if (cam, obj) == ("top", "duck"):
            per_combo_accept[(cam, obj)] = top_duck_bbox_only
        else:
            per_combo_accept[(cam, obj)] = int(mask.sum())

    print(f"[info] top_duck before y2 filter:         {top_duck_before:,}")
    print(f"[info] top_duck trust=1 WITH bbox        : {td_with_bbox:,}")
    print(f"[info] top_duck trust=1 WITHOUT bbox     : {td_without_bbox:,}")
    print(f"[info] top_duck bbox & y2 <= 0.85        : {td_bbox_y2ok:,}")
    print(f"[info] top_duck bbox & y2 >  0.85 (DROP) : {td_bbox_y2fail:,}")
    print(f"[info] top_duck STRICT (bbox+y2ok only)  : {top_duck_strict:,}")
    print(f"[info] top_duck BBOX-ONLY filter kept    : {top_duck_bbox_only:,}")
    for combo, n in per_combo_accept.items():
        print(f"[info] accept {combo[0]}/{combo[1]}: {n:,}")

    # Load episode meta
    ep_files = sorted((ds_path / "meta" / "episodes").rglob("*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)

    with open(ds_path / "meta" / "info.json") as f:
        info = json.load(f)
    fps = float(info["fps"])

    # Build cam_key map — the dataset uses observation.images.{cam}
    # Verify against info.json features
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
    rng = np.random.default_rng(SEED)
    all_picks: list[pd.DataFrame] = []
    for combo in COMBOS:
        cam, obj = combo
        trust_col = f"{cam}_{obj}_trust"
        conf_col = f"{cam}_{obj}_conf"
        # Start from rows in clean parquet with trust == 1
        base_mask = clean[trust_col] == 1
        base = clean.loc[base_mask, ["episode_index", "frame_index", conf_col]].copy()
        base = base.rename(columns={conf_col: "conf"})

        # Merge with det to bring in bbox
        bbox_cols = [f"{cam}_{obj}_x1", f"{cam}_{obj}_y1",
                     f"{cam}_{obj}_x2", f"{cam}_{obj}_y2"]
        det_sub = det[["episode_index", "frame_index"] + bbox_cols].dropna(subset=bbox_cols)
        merged = base.merge(det_sub, on=["episode_index", "frame_index"], how="inner")

        # Apply y2 filter for top_duck
        if (cam, obj) == ("top", "duck"):
            merged = merged[merged[f"{cam}_{obj}_y2"] <= Y2_THR_NORM].copy()

        # Rename bbox cols to common names
        merged = merged.rename(columns={
            f"{cam}_{obj}_x1": "x1n",
            f"{cam}_{obj}_y1": "y1n",
            f"{cam}_{obj}_x2": "x2n",
            f"{cam}_{obj}_y2": "y2n",
        })

        n = COUNTS[combo]
        # Use combo-specific seed for more diversity
        seed_i = SEED + hash((cam, obj)) % 10_000
        picked = pick_rows_for_combo(merged, n, seed=int(rng.integers(0, 1_000_000)))
        picked["cam"] = cam
        picked["obj"] = obj
        print(f"[pick] {cam}/{obj}: {len(picked)}/{n} (pool={len(merged)})")
        all_picks.append(picked)

    picks = pd.concat(all_picks, ignore_index=True)

    # Merge ep meta for each cam
    records = []
    missing_streams = []
    decoder_cache: dict[tuple[str, int, int], VideoDecoder] = {}
    # Process by (cam, episode) to open each video once
    for cam in CAMS:
        cam_key = cam_key_map[cam]
        chunk_col = f"videos/{cam_key}/chunk_index"
        file_col = f"videos/{cam_key}/file_index"
        from_ts_col = f"videos/{cam_key}/from_timestamp"
        if chunk_col not in ep_meta.columns:
            print(f"[warn] missing ep_meta col {chunk_col}")
            missing_streams.append(cam)
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
        # Group by (chunk, file) to open each video once (episodes may share mp4 files)
        for (chunk_idx, file_idx), grp in sub.groupby(["chunk_idx", "file_idx"]):
            chunk_idx = int(chunk_idx)
            file_idx = int(file_idx)
            # Sort by frame for sequential seek efficiency
            grp = grp.sort_values(by="frame_index")
            for _, row in grp.iterrows():
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
                bbox = (
                    float(row["x1n"]), float(row["y1n"]),
                    float(row["x2n"]), float(row["y2n"]),
                )
                out_path = OUT_DIR / f"{cam}_{row['obj']}_ep{ep:04d}_fr{fr:04d}.png"
                render_one(rgb, cam, str(row["obj"]), ep, fr, conf, bbox, out_path)
                records.append({
                    "cam": cam, "obj": str(row["obj"]),
                    "ep": ep, "fr": fr, "conf": conf,
                    "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                    "filename": out_path.name,
                })
                print(f"[save] {out_path.name} conf={conf:.3f}")

    pd.DataFrame(records).to_csv(OUT_DIR / "_metrics.csv", index=False)

    # Write stats for findings
    stats = {
        "per_combo_accept_bbox_only_interp": {
            f"{c}_{o}": n for (c, o), n in per_combo_accept.items()
        },
        "top_duck_before_y2": top_duck_before,
        "top_duck_trust1_with_bbox": td_with_bbox,
        "top_duck_trust1_without_bbox": td_without_bbox,
        "top_duck_bbox_y2ok": td_bbox_y2ok,
        "top_duck_bbox_y2fail_dropped": td_bbox_y2fail,
        "top_duck_strict_kept": top_duck_strict,
        "top_duck_bbox_only_kept": top_duck_bbox_only,
        "n_rendered": len(records),
    }
    with open(OUT_DIR / "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] {len(records)} PNGs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
