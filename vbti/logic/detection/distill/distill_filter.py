"""Phase 0.3+0.4 — Apply filters A+B+E (v7) to G-DINO detections and render review galleries.

V8 delta from v7: CONF_THRESH_DEFAULT lowered 0.05 -> 0.02. v7 audit still showed
68% FN in `rejected_lowconf_E` — teacher emits conf 0.02-0.05 for many visible
objects across non-gripper cams. Below 0.02 is mostly true noise (teacher
boundary). PI's "no false rejections" stance prefers admitting more noise over
losing visible objects.

V7 delta from v6: CONF_THRESH_DEFAULT lowered 0.08 -> 0.05 (rescued 20k labels).

V6 filters (2026-04-20):

  A: gripper-cam duck during grasp/transport phases
     Step 1 (NaN bbox handling):
       if bbox is NaN (regardless of teacher_conf):
         interpolate (cx,cy,w,h) from nearest valid bbox within +/-15 frames
         (weighted avg of nearest-backwards and nearest-forwards by 1/distance)
         if no neighbor found -> reject (phase_gripper_duck_no_neighbor)
       (R_PHASE_OCCLUDED retained in enum for stats compat but no longer produced.)
     Step 2 (blue-dominance on INNER 50% of bbox):
       shrink bbox to inner 50% area (w,h scaled by sqrt(0.5), centered)
       compute mean (R,G,B) on shrunk region
       v4 update: FN audit showed orange duck head + red cup contamination
         systematically pulls R up, causing B/R<1.25 while B/G>=1.25 on real ducks.
         We now gate on B/G alone (empty jaw has B/G~1.00-1.22, real duck B/G>1.3).
       blue_ok = (B > thr_bg[cam_mode] * G)    # B/R constraint dropped
       if NOT blue_ok -> reject (phase_gripper_duck_no_blue)
       else -> accept

  B: gripper-cam CUP (pure geometry, color rule dropped)
     if conf >= 0.08:
       area_norm = (bbox_w / 640) * (bbox_h / 480)
       in_jaw_region = (cx >= 0.70) AND (cy >= 0.65) AND (area_norm <= 0.30)
       if in_jaw_region -> reject (gripper_cup_jaw_region)
       else -> accept

  E: gripper_duck in grasp/transport >= 0.15, gripper_duck in release >= 0.02,
     all others >= 0.02.  (v8: 0.05 -> 0.02 — non-gripper cams emit low conf for
     distant but visible objects; below 0.02 is pure teacher noise.)

Priority A > B > E. Each (cam, obj, frame) row gets at most one reason code.

Usage:
    python -m vbti.logic.detection.distill_filter \\
        --dataset eternalmay33/01_02_03_merged_may-sim \\
        --output /home/may33/.cache/vbti/detection_labels_clean.parquet \\
        --gallery-dir "/path/to/data_analysis/"
"""
from __future__ import annotations

import argparse
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from vbti.logic.dataset import resolve_dataset_path
from vbti.logic.detection.phases import PHASE_NAMES, detect_phases_episode
from vbti.logic.detection.process_dataset import (
    VideoReader,
    get_video_path,
    load_dataset_meta,
)

CAMERAS = ["left", "right", "top", "gripper"]
OBJECTS = ["duck", "cup"]
CLUSTER_NAMES = ("realsense_v1", "realsense_v2", "opencv_raw")

# Locked config (v3)
INTERP_CONF_GATE = 0.3           # fixed — do not negotiate
INTERP_WINDOW = 15               # fixed — do not negotiate
SHRINK_RATIO = float(np.sqrt(0.5))   # ~0.707 — shrink to inner 50% area
BLUE_RATIO_THRESHOLDS_REPORT = (1.15, 1.25, 1.35)   # reported for sensitivity table
BLUE_RATIO_FALLBACK = 1.25           # fallback when per-cluster thresholds don't diverge
BLUE_RATIO_MEANINGFUL_DELTA = 0.05   # apply per-cluster only if max-min diff > this
# Cup jaw-region geometry.
# v3 spec started at (cx>=0.70, cy>=0.65, area<=0.30).
# v5 audit showed ~25% FN rate on that rect — real cups on the right side of frame
#   (cx>0.7, cy~0.66-0.70) got caught. Manual re-inspection of TPs shows jaw tooling
#   sits tightly at cx~0.74-0.78, cy~0.81-0.86, area~0.06-0.14. Jaw scan CSV at
#   cx>=0.74, cy>=0.81, area<=0.14 gives catch=15.1%, fp_acc=1.45% (vs 24.6%/3.2% at v3).
# v11 (stage1 audit 2026-04-21): 518 jaw false-accepts found clustered tightly at
#   cx=0.734-0.740 (just below the v5 cutoff). Lowering to 0.73 catches them all.
#   Distribution has a clean gap [0.72, 0.73] separating jaw cluster from real cups.
CUP_JAW_CX_MIN = 0.73
CUP_JAW_CY_MIN = 0.81
CUP_JAW_AREA_MAX = 0.14
CONF_THRESH_GRIPPER_DUCK = 0.15
CONF_THRESH_DEFAULT = 0.02  # v8: lowered from 0.05 — v7 audit still 68% FN in lowconf_E
INVESTIGATION_CSV = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/investigation_v3/episode_modes.csv"
)

# Reason codes
R_ACCEPTED = "accepted"
R_LOWCONF = "low_confidence"
R_PHASE_OCCLUDED = "phase_gripper_duck_occluded"        # A, bbox NaN & conf<=0.3
R_PHASE_NO_NEIGHBOR = "phase_gripper_duck_no_neighbor"  # A, bbox NaN & conf>0.3 & no valid neighbor
R_PHASE_NO_BLUE = "phase_gripper_duck_no_blue"          # A, shrunk bbox not blue-dominant
R_CUP_JAW = "gripper_cup_jaw_region"                    # B, geometric jaw region
R_TOP_DUCK_ARMLOCK = "top_duck_armlock_y2"              # v9, top_duck bbox.y2 > 0.85
R_NO_DETECTION = "no_detection"
R_SIDE_DUCK_RELEASE = "side_duck_release"               # stage1: left/right duck in release phase
R_LEFT_DUCK_TOP_STRIP = "left_duck_top_strip"           # stage1: left_duck y2<0.10 fixed artifact
R_LEFT_DUCK_FIXED_BLOB = "left_duck_fixed_blob"         # stage1: left_duck fixed pixel-blob artifact (cx≈0.28 cy≈0.74)
R_RIGHT_DUCK_ARM_BASE = "right_duck_arm_base"           # stage1: right_duck x1<0.124 (left-edge arm base)
R_TOP_DUCK_FIXED_BLOB = "top_duck_fixed_blob"           # stage1: top_duck fixed pixel-blob artifact
R_CUP_INSIDE_DUCK = "cup_inside_duck"                   # stage1: pregrasp/grasp cup bbox ≥80% inside duck bbox (DINO double-label)
R_INTERPOLATED = "interpolated"                          # stage2: rescued via linear interp between anchors

REASON_CATEGORIES = [
    R_ACCEPTED,
    R_LOWCONF,
    R_PHASE_OCCLUDED,
    R_PHASE_NO_NEIGHBOR,
    R_PHASE_NO_BLUE,
    R_CUP_JAW,
    R_TOP_DUCK_ARMLOCK,
    R_NO_DETECTION,
    R_SIDE_DUCK_RELEASE,
    R_LEFT_DUCK_TOP_STRIP,
    R_LEFT_DUCK_FIXED_BLOB,
    R_RIGHT_DUCK_ARM_BASE,
    R_TOP_DUCK_FIXED_BLOB,
    R_CUP_INSIDE_DUCK,
    R_INTERPOLATED,
]

Y2_ARMLOCK_THRESHOLD = 0.85  # v9: top_duck bbox bottom-edge reject threshold


# ------------------------------------------------------------------
# Phase computation
# ------------------------------------------------------------------

def compute_phases(ds_path: Path, fps: int) -> pd.DataFrame:
    """Return DF[(frame_index, episode_index, phase_name)] — phase_name in PHASE_NAMES or 'unknown'."""
    data_files = sorted((ds_path / "data").rglob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in data_files]
    df = pd.concat(dfs, ignore_index=True)

    rows = []
    ok = fail = 0
    for ep_idx in tqdm(sorted(df["episode_index"].unique()), desc="  phases", unit="ep"):
        ep_data = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        states = np.stack(ep_data["observation.state"].values)
        phases = detect_phases_episode(states, fps=fps)
        if np.all(phases == -1):
            fail += 1
        else:
            ok += 1
        for fi, ph in zip(ep_data["frame_index"].values, phases):
            rows.append({
                "frame_index": int(fi),
                "episode_index": int(ep_idx),
                "phase": PHASE_NAMES[ph] if ph >= 0 else "unknown",
            })
    print(f"[phases] {ok} ok, {fail} failed")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Cluster / cam_mode join
# ------------------------------------------------------------------

def load_cluster_modes() -> pd.DataFrame:
    """Load episode -> cluster_name (cam_mode) mapping."""
    if not INVESTIGATION_CSV.exists():
        raise FileNotFoundError(f"Missing: {INVESTIGATION_CSV}")
    em = pd.read_csv(INVESTIGATION_CSV)
    return em[["episode_index", "cluster_name"]].rename(columns={"cluster_name": "cam_mode"})


# ------------------------------------------------------------------
# Bbox interpolation for NaN high-conf gripper_duck frames
# ------------------------------------------------------------------

def interpolate_gripper_duck_bboxes(df: pd.DataFrame) -> pd.DataFrame:
    """For gripper_duck frames with NaN bbox (any conf), fill
    (cx, cy, w, h) via weighted-by-1/distance average of nearest valid bboxes
    within +/-INTERP_WINDOW frames.  v6: conf>0.3 gate removed — always interpolate.

    Adds columns:
      gripper_duck_cx_interp, gripper_duck_cy_interp,
      gripper_duck_w_interp,  gripper_duck_h_interp,
      gripper_duck_bbox_filled (bool: True if we filled it here)
    """
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    cx = df["gripper_duck_cx"].to_numpy(dtype=np.float64)
    cy = df["gripper_duck_cy"].to_numpy(dtype=np.float64)
    x1 = df["gripper_duck_x1"].to_numpy(dtype=np.float64)
    y1 = df["gripper_duck_y1"].to_numpy(dtype=np.float64)
    x2 = df["gripper_duck_x2"].to_numpy(dtype=np.float64)
    y2 = df["gripper_duck_y2"].to_numpy(dtype=np.float64)
    conf = df["gripper_duck_conf"].to_numpy(dtype=np.float64)
    fi = df["frame_index"].to_numpy(dtype=np.int64)
    ep = df["episode_index"].to_numpy(dtype=np.int64)

    w = x2 - x1
    h = y2 - y1
    valid_bbox = ~(np.isnan(x1) | np.isnan(y1) | np.isnan(x2) | np.isnan(y2))

    cx_out = cx.copy()
    cy_out = cy.copy()
    w_out = w.copy()
    h_out = h.copy()
    filled = np.zeros(len(df), dtype=bool)

    n = len(df)
    # Per-episode slices (df is pre-sorted)
    ep_start = {}
    ep_end = {}
    cur_ep = None
    for i in range(n):
        e = ep[i]
        if e != cur_ep:
            ep_start[int(e)] = i
            if cur_ep is not None:
                ep_end[int(cur_ep)] = i
            cur_ep = e
    ep_end[int(cur_ep)] = n

    for i in range(n):
        if valid_bbox[i]:
            continue
        # v6: no conf gate — interpolate for all NaN bboxes.
        e = int(ep[i])
        s = ep_start[e]
        eend = ep_end[e]
        cur_fi = fi[i]

        # Scan backwards for nearest valid within window
        back = None
        for j in range(i - 1, s - 1, -1):
            if fi[j] < cur_fi - INTERP_WINDOW:
                break
            if valid_bbox[j]:
                back = j
                break
        # Scan forwards
        fwd = None
        for j in range(i + 1, eend):
            if fi[j] > cur_fi + INTERP_WINDOW:
                break
            if valid_bbox[j]:
                fwd = j
                break

        picks = []
        if back is not None:
            dist = abs(cur_fi - fi[back])
            if dist > 0:
                picks.append((back, 1.0 / dist))
        if fwd is not None:
            dist = abs(fi[fwd] - cur_fi)
            if dist > 0:
                picks.append((fwd, 1.0 / dist))

        if not picks:
            continue

        wsum = sum(wt for _, wt in picks)
        cx_val = sum(cx[j] * wt for j, wt in picks) / wsum
        cy_val = sum(cy[j] * wt for j, wt in picks) / wsum
        w_val = sum((x2[j] - x1[j]) * wt for j, wt in picks) / wsum
        h_val = sum((y2[j] - y1[j]) * wt for j, wt in picks) / wsum

        cx_out[i] = cx_val
        cy_out[i] = cy_val
        w_out[i] = w_val
        h_out[i] = h_val
        filled[i] = True

    df["gripper_duck_cx_interp"] = cx_out
    df["gripper_duck_cy_interp"] = cy_out
    df["gripper_duck_w_interp"] = w_out
    df["gripper_duck_h_interp"] = h_out
    df["gripper_duck_bbox_filled"] = filled

    # Effective bbox (interp if filled, else original)
    x1e = np.where(filled, cx_out - w_out / 2.0, x1)
    y1e = np.where(filled, cy_out - h_out / 2.0, y1)
    x2e = np.where(filled, cx_out + w_out / 2.0, x2)
    y2e = np.where(filled, cy_out + h_out / 2.0, y2)

    df["gripper_duck_x1_eff"] = x1e
    df["gripper_duck_y1_eff"] = y1e
    df["gripper_duck_x2_eff"] = x2e
    df["gripper_duck_y2_eff"] = y2e
    return df


# ------------------------------------------------------------------
# Bbox mean RGB helper (on inner 50% shrunk region)
# ------------------------------------------------------------------

def _shrunk_bbox_mean_rgb(frame: np.ndarray, x1n, y1n, x2n, y2n,
                          shrink: float = SHRINK_RATIO) -> tuple[float, float, float] | None:
    """Mean (R,G,B) inside INNER <shrink^2>=50% area of normalized bbox. None if invalid."""
    h, w = frame.shape[:2]
    if any(np.isnan([x1n, y1n, x2n, y2n])):
        return None
    cx = 0.5 * (x1n + x2n)
    cy = 0.5 * (y1n + y2n)
    bw = max(0.0, x2n - x1n) * shrink
    bh = max(0.0, y2n - y1n) * shrink
    sx1 = cx - bw / 2.0
    sy1 = cy - bh / 2.0
    sx2 = cx + bw / 2.0
    sy2 = cy + bh / 2.0
    x1 = int(np.clip(sx1 * w, 0, w - 1))
    y1 = int(np.clip(sy1 * h, 0, h - 1))
    x2 = int(np.clip(sx2 * w, 0, w - 1))
    y2 = int(np.clip(sy2 * h, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0:
        return None
    mean = crop.reshape(-1, 3).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


# ------------------------------------------------------------------
# Gripper-cam pixel sampling (shrunk bbox, duck only — cup no longer samples)
# ------------------------------------------------------------------

def compute_gripper_duck_pixel_stats(
    det_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    ds_path: Path,
    episodes: pd.DataFrame,
    info: dict,
) -> dict:
    """Sample mean RGB (on INNER 50% shrunk bbox) for every gripper_duck candidate
    during grasp/transport with either a valid bbox or a successfully-interpolated bbox.

    Returns: {(ep, fi): {"r","g","b"}}
    """
    cam = "gripper"
    cam_key = f"observation.images.{cam}"
    fps = info["fps"]
    feat = info["features"][cam_key]
    vid_w = feat["info"]["video.width"]
    vid_h = feat["info"]["video.height"]

    chunk_col = f"videos/{cam_key}/chunk_index"
    file_col = f"videos/{cam_key}/file_index"
    from_ts_col = f"videos/{cam_key}/from_timestamp"

    merged = det_df.merge(phase_df, on=["frame_index", "episode_index"], how="left")

    has_bbox_eff = ~(
        merged["gripper_duck_x1_eff"].isna()
        | merged["gripper_duck_y1_eff"].isna()
        | merged["gripper_duck_x2_eff"].isna()
        | merged["gripper_duck_y2_eff"].isna()
    )
    # v11 (stage1): Sample RGB for every candidate frame regardless of phase.
    # Stage1 runs blue-gate universally (not just grasp/transport), so RGB
    # must be available for reach/pregrasp/release too.
    cand = merged[
        (merged["gripper_duck_conf"] > 0.0)
        & has_bbox_eff
    ][[
        "episode_index", "frame_index",
        "gripper_duck_x1_eff", "gripper_duck_y1_eff",
        "gripper_duck_x2_eff", "gripper_duck_y2_eff",
    ]].set_index(["episode_index", "frame_index"])

    cand_set = set(cand.index)
    print(f"[pixel-sample] gripper_duck candidates (A, incl. interp): {len(cand_set)}")

    duck_stats: dict[tuple[int, int], dict] = {}
    if not cand_set:
        return duck_stats

    video_groups = episodes.groupby([chunk_col, file_col])
    t0 = time.perf_counter()
    total_streamed = 0

    for (chunk_idx, file_idx), group_eps in tqdm(list(video_groups), desc="  gripper RGB", unit="vid"):
        video_path = get_video_path(ds_path, cam_key, int(chunk_idx), int(file_idx))
        if not video_path.exists():
            print(f"  Warning: missing {video_path}")
            continue
        group_eps = group_eps.sort_values(from_ts_col)

        per_ep_need: dict[int, list[int]] = defaultdict(list)
        eps_in_file = set(group_eps["episode_index"].astype(int))
        for (e, f_i) in cand_set:
            if e in eps_in_file:
                per_ep_need[e].append(f_i)
        for e in per_ep_need:
            per_ep_need[e].sort()

        if not per_ep_need:
            continue

        with VideoReader(video_path, vid_w, vid_h) as reader:
            for _, ep_row in group_eps.iterrows():
                ep_idx = int(ep_row["episode_index"])
                n_ep_frames = int(ep_row["length"])
                from_ts = float(ep_row[from_ts_col])
                start_frame = round(from_ts * fps)

                target_fis = per_ep_need.get(ep_idx, [])
                if not target_fis:
                    reader.skip(n_ep_frames)
                    continue

                if reader._pos < start_frame:
                    reader.skip(start_frame - reader._pos)

                next_idx = 0
                cur_local = 0
                while cur_local < n_ep_frames and next_idx < len(target_fis):
                    target = target_fis[next_idx]
                    gap = target - cur_local
                    if gap > 0:
                        reader.skip(gap)
                        cur_local += gap
                    frame = reader.read_one()
                    if frame is None:
                        break
                    total_streamed += 1
                    cur_local += 1

                    key = (ep_idx, target)
                    if key in cand_set:
                        row = cand.loc[key]
                        rgb = _shrunk_bbox_mean_rgb(
                            frame,
                            float(row["gripper_duck_x1_eff"]),
                            float(row["gripper_duck_y1_eff"]),
                            float(row["gripper_duck_x2_eff"]),
                            float(row["gripper_duck_y2_eff"]),
                        )
                        if rgb is not None:
                            r, g, b = rgb
                            duck_stats[key] = {"r": r, "g": g, "b": b}
                    next_idx += 1

                remaining = n_ep_frames - cur_local
                if remaining > 0:
                    reader.skip(remaining)

    elapsed = time.perf_counter() - t0
    print(f"[pixel-sample] Decoded {total_streamed} gripper frames in {elapsed:.1f}s")
    print(f"[pixel-sample] duck_stats={len(duck_stats)}")
    return duck_stats


# ------------------------------------------------------------------
# Measure per-cluster thresholds
# ------------------------------------------------------------------

def measure_cluster_thresholds(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Choose per-cluster (thr_br, thr_bg) keeping >=95% of tentative-accepted duck frames.

    Tentative-accepted = phase in {grasp,transport} AND conf>0 AND shrunk-bbox RGB available
                         AND B/R >= fallback (1.25) AND B/G >= fallback (1.25)
    Then for each cluster, 5th percentile of B/R and B/G in the tentative set
    becomes the per-cluster threshold (rounded to 2 decimals, floored at fallback).

    Returns (thr_br_per_cluster, thr_bg_per_cluster, measurement_log).
    """
    thr_br_per = {}
    thr_bg_per = {}
    log = {}

    r = df["gripper_duck_mean_r_shrunk"].to_numpy()
    g = df["gripper_duck_mean_g_shrunk"].to_numpy()
    b = df["gripper_duck_mean_b_shrunk"].to_numpy()
    have = ~np.isnan(r) & ~np.isnan(g) & ~np.isnan(b)
    # avoid division by 0
    with np.errstate(divide="ignore", invalid="ignore"):
        br = np.where((r > 0) & have, b / r, np.nan)
        bg = np.where((g > 0) & have, b / g, np.nan)
    phase_mask = df["phase"].isin(["grasp", "transport"]).to_numpy()
    conf = df["gripper_duck_conf"].to_numpy()
    tentative = (
        have & phase_mask & (conf > 0)
        & (br >= BLUE_RATIO_FALLBACK) & (bg >= BLUE_RATIO_FALLBACK)
    )

    modes = df["cam_mode"].to_numpy()
    for cluster in CLUSTER_NAMES:
        m = tentative & (modes == cluster)
        n = int(m.sum())
        if n < 50:
            thr_br_per[cluster] = BLUE_RATIO_FALLBACK
            thr_bg_per[cluster] = BLUE_RATIO_FALLBACK
            log[cluster] = {"n": n, "br_p05": None, "bg_p05": None, "reason": "too few tentative accepts"}
            continue
        br_p05 = float(np.quantile(br[m], 0.05))
        bg_p05 = float(np.quantile(bg[m], 0.05))
        # Floor at fallback to avoid loosening below 1.25
        thr_br_per[cluster] = max(round(br_p05, 2), BLUE_RATIO_FALLBACK - 0.10)
        thr_bg_per[cluster] = max(round(bg_p05, 2), BLUE_RATIO_FALLBACK - 0.10)
        log[cluster] = {
            "n": n,
            "br_p05": br_p05,
            "bg_p05": bg_p05,
            "br_p25": float(np.quantile(br[m], 0.25)),
            "bg_p25": float(np.quantile(bg[m], 0.25)),
            "br_median": float(np.quantile(br[m], 0.5)),
            "bg_median": float(np.quantile(bg[m], 0.5)),
        }

    # Spec: "Only apply the per-mode rule if measured thresholds differ meaningfully (>0.05)
    # across clusters. Else fall back to single threshold 1.25."
    br_vals = list(thr_br_per.values())
    bg_vals = list(thr_bg_per.values())
    br_range = max(br_vals) - min(br_vals)
    bg_range = max(bg_vals) - min(bg_vals)
    apply_per_mode = (br_range > BLUE_RATIO_MEANINGFUL_DELTA) or (bg_range > BLUE_RATIO_MEANINGFUL_DELTA)
    log["_decision"] = {
        "br_range": br_range,
        "bg_range": bg_range,
        "meaningful_delta": BLUE_RATIO_MEANINGFUL_DELTA,
        "apply_per_mode": apply_per_mode,
    }
    if not apply_per_mode:
        thr_br_per = {c: BLUE_RATIO_FALLBACK for c in CLUSTER_NAMES}
        thr_bg_per = {c: BLUE_RATIO_FALLBACK for c in CLUSTER_NAMES}

    return thr_br_per, thr_bg_per, log


# ------------------------------------------------------------------
# Assemble cleaned df
# ------------------------------------------------------------------

def build_cleaned_df(
    det_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    modes_df: pd.DataFrame,
    duck_stats: dict,
) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    """Assemble cleaned dataframe with trust + reason columns per (cam, obj)."""
    df = det_df.merge(phase_df, on=["frame_index", "episode_index"], how="left")
    df["phase"] = df["phase"].fillna("unknown").astype("category")

    df = df.merge(modes_df, on="episode_index", how="left")
    df["cam_mode"] = df["cam_mode"].fillna("realsense_v1").astype("category")

    # Interpolate bboxes (high-conf NaN fill)
    df = interpolate_gripper_duck_bboxes(df)

    # Join shrunk-bbox RGB
    if duck_stats:
        idx = pd.MultiIndex.from_tuples(list(duck_stats.keys()),
                                        names=["episode_index", "frame_index"])
        vals = np.array([[v["r"], v["g"], v["b"]] for v in duck_stats.values()], dtype=np.float32)
        duck_rgb = pd.DataFrame({
            "gripper_duck_mean_r_shrunk": vals[:, 0],
            "gripper_duck_mean_g_shrunk": vals[:, 1],
            "gripper_duck_mean_b_shrunk": vals[:, 2],
        }, index=idx).reset_index()
        df = df.merge(duck_rgb, on=["episode_index", "frame_index"], how="left")
    else:
        df["gripper_duck_mean_r_shrunk"] = np.nan
        df["gripper_duck_mean_g_shrunk"] = np.nan
        df["gripper_duck_mean_b_shrunk"] = np.nan

    # Measure per-cluster thresholds
    thr_br_per, thr_bg_per, measure_log = measure_cluster_thresholds(df)
    print(f"[measure] thr_br per-cluster: {thr_br_per}")
    print(f"[measure] thr_bg per-cluster: {thr_bg_per}")
    print(f"[measure] decision: {measure_log['_decision']}")

    # Compute blue-ok using per-cluster thresholds.
    # v4: orange head of duck + red cup contamination systematically pulls R up.
    # Empirically, FNs have B/R<1.25 but B/G>=1.25, while TPs (empty jaw) have
    # B/G<1.22. Dropping the B/R requirement recovers orange-head FNs without
    # admitting empty-jaw TPs. Keep thr_br in reason plumbing for telemetry
    # but only gate on B/G.
    r = df["gripper_duck_mean_r_shrunk"].to_numpy()
    g = df["gripper_duck_mean_g_shrunk"].to_numpy()
    b = df["gripper_duck_mean_b_shrunk"].to_numpy()
    have_rgb = ~np.isnan(r) & ~np.isnan(g) & ~np.isnan(b)
    modes = df["cam_mode"].to_numpy()

    thr_br_arr = np.full(len(df), BLUE_RATIO_FALLBACK, dtype=np.float64)
    thr_bg_arr = np.full(len(df), BLUE_RATIO_FALLBACK, dtype=np.float64)
    for cluster, thr in thr_br_per.items():
        thr_br_arr[modes == cluster] = thr
    for cluster, thr in thr_bg_per.items():
        thr_bg_arr[modes == cluster] = thr

    with np.errstate(invalid="ignore"):
        blue_ok = have_rgb & (b > thr_bg_arr * g)
    df["gripper_duck_blue_ok"] = blue_ok

    # Also keep sensitivity flags (single-threshold) for reporting
    for thr in BLUE_RATIO_THRESHOLDS_REPORT:
        with np.errstate(invalid="ignore"):
            mk = have_rgb & (b > thr * r) & (b > thr * g)
        df[f"gripper_duck_is_blue_{int(thr*100):03d}"] = mk

    # Cup geometry feature: in-jaw region
    cx_cup = df["gripper_cup_cx"].to_numpy()
    cy_cup = df["gripper_cup_cy"].to_numpy()
    w_cup_norm = (df["gripper_cup_x2"].to_numpy() - df["gripper_cup_x1"].to_numpy())
    h_cup_norm = (df["gripper_cup_y2"].to_numpy() - df["gripper_cup_y1"].to_numpy())
    # bboxes are normalized to [0,1]; area_norm already in [0,1]^2 so matches spec
    cup_area_norm = w_cup_norm * h_cup_norm
    with np.errstate(invalid="ignore"):
        cup_in_jaw = (
            (cx_cup >= CUP_JAW_CX_MIN)
            & (cy_cup >= CUP_JAW_CY_MIN)
            & (cup_area_norm <= CUP_JAW_AREA_MAX)
        )
    df["gripper_cup_area_norm"] = cup_area_norm
    df["gripper_cup_in_jaw"] = cup_in_jaw

    # ------------------------------------------------------------------
    # Apply per (cam, obj) rules. Priority: A > B > E.
    # ------------------------------------------------------------------
    for cam in CAMERAS:
        for obj in OBJECTS:
            conf_col = f"{cam}_{obj}_conf"
            trust_col = f"{cam}_{obj}_trust"
            reason_col = f"{cam}_{obj}_reason"

            conf = df[conf_col].to_numpy(dtype=np.float32)
            reason = np.full(len(df), R_ACCEPTED, dtype=object)

            no_det = ~(conf > 0.0)
            reason[no_det] = R_NO_DETECTION

            # v6: per-phase threshold for gripper_duck.
            # RELEASE phase uses 0.08 (duck still visible at low conf during drop).
            # Other phases (grasp/transport/etc) stay at 0.15.
            if cam == "gripper" and obj == "duck":
                phase_arr = df["phase"].to_numpy()
                thresh = np.where(
                    phase_arr == "release",
                    CONF_THRESH_DEFAULT,
                    CONF_THRESH_GRIPPER_DUCK,
                ).astype(np.float32)
            else:
                thresh = np.full(len(df), CONF_THRESH_DEFAULT, dtype=np.float32)

            # --- Filter A (gripper_duck, grasp/transport)
            if cam == "gripper" and obj == "duck":
                phase_mask = df["phase"].isin(["grasp", "transport"]).to_numpy()
                candidate = phase_mask & (conf > 0.0)

                orig_bbox_nan = (
                    df["gripper_duck_x1"].isna()
                    | df["gripper_duck_y1"].isna()
                    | df["gripper_duck_x2"].isna()
                    | df["gripper_duck_y2"].isna()
                ).to_numpy()
                filled = df["gripper_duck_bbox_filled"].to_numpy()

                # v6: conf>0.3 gate removed. Any NaN bbox that was NOT filled by
                # interpolation -> R_PHASE_NO_NEIGHBOR (regardless of conf).
                # R_PHASE_OCCLUDED retained in enum for stats compat but unused.
                no_neighbor = candidate & orig_bbox_nan & ~filled
                # blue gate fails on valid/interp bbox -> R_PHASE_NO_BLUE
                has_eff = have_rgb  # sampling only happened for frames with eff bbox
                not_blue = candidate & has_eff & ~blue_ok

                reason[no_neighbor & (reason == R_ACCEPTED)] = R_PHASE_NO_NEIGHBOR
                reason[not_blue & (reason == R_ACCEPTED)] = R_PHASE_NO_BLUE
                # Frames with candidate but no RGB at all (e.g. bbox was NaN, interp succeeded,
                # but pixel sampler couldn't read) — extremely rare. Leave accepted — E will
                # handle low-conf and by construction high-conf + filled + no RGB means we
                # couldn't decode; safest is to keep them (they're few).

            # --- Filter B (gripper_cup, pure geometry)
            if cam == "gripper" and obj == "cup":
                in_jaw = df["gripper_cup_in_jaw"].to_numpy()
                cup_jaw_mask = in_jaw & (conf >= CONF_THRESH_DEFAULT)
                reason[cup_jaw_mask & (reason == R_ACCEPTED)] = R_CUP_JAW

            # --- Filter v9 (top_duck, arm-lock geometric reject)
            if cam == "top" and obj == "duck":
                y2_top = df["top_duck_y2"].to_numpy(dtype=np.float32)
                armlock = np.isfinite(y2_top) & (y2_top > Y2_ARMLOCK_THRESHOLD)
                reason[armlock & (reason == R_ACCEPTED)] = R_TOP_DUCK_ARMLOCK

            # --- Filter E — conf floor
            lowconf = (conf > 0.0) & (conf < thresh)
            reason[lowconf & (reason == R_ACCEPTED)] = R_LOWCONF

            trust = (reason == R_ACCEPTED).astype(np.int8)
            df[trust_col] = trust
            df[reason_col] = pd.Categorical(reason, categories=REASON_CATEGORIES)

    # Blue-threshold sensitivity report
    blue_tune = {}
    phase_mask = df["phase"].isin(["grasp", "transport"]).to_numpy()
    conf = df["gripper_duck_conf"].to_numpy()
    candidate = phase_mask & (conf > 0.0)
    bbox_ok = candidate & have_rgb
    blue_tune["candidates_total"] = int(candidate.sum())
    blue_tune["bbox_with_rgb"] = int(bbox_ok.sum())
    for thr in BLUE_RATIO_THRESHOLDS_REPORT:
        mk = df[f"gripper_duck_is_blue_{int(thr*100):03d}"].to_numpy()
        kept = int((bbox_ok & mk).sum())
        rej = int((bbox_ok & ~mk).sum())
        blue_tune[f"thr_{thr:.2f}_kept_blue"] = kept
        blue_tune[f"thr_{thr:.2f}_rejected_no_blue"] = rej

    return df, blue_tune, thr_br_per, thr_bg_per, measure_log


# ------------------------------------------------------------------
# Trust-gated interpolation + stable re-pass + v10 cy shift
# ------------------------------------------------------------------

_INTERP_FIELDS = ["cx", "cy", "conf", "x1", "y1", "x2", "y2"]


def apply_trust_gated_interp_and_v10(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw interpolation with a trust-gated version, then run the
    filter rules one more time ("stable re-pass") to catch interpolated rows
    that drift into reject zones. Finally apply v10 cy shift for left/right duck.

    Flow per (cam, obj):
      1. Null cx/cy/conf/bbox on rows where trust=0.
      2. Linearly interpolate those cols per episode.
      3. Rows that were trust=0 and now have valid cx/cy become candidates for
         trust=1, subject to re-checking Filter E and v9.
      4. Force trust=0 for rows that had NO valid raw detection (conf=0 OR raw
         bbox missing/zero). These must NEVER be rescued by interpolation — doing
         so produces mismatched cx/bbox pairs (observed 2026-04-20: ep=113 fr=402
         had raw all-zero, got interpolated arm-lock bbox with cx pointing at
         empty sky). Exception: gripper_duck with bbox_filled=True from Filter
         A's +/-15-frame interp is a legitimate rescue path.
      5. Apply v10 cy-shift for left/right duck (target correction): if h>w,
         set cy = y2 - w/2.
    """
    df = df.copy()
    for cam in CAMERAS:
        for obj in OBJECTS:
            trust_col = f"{cam}_{obj}_trust"
            reason_col = f"{cam}_{obj}_reason"
            cols = [f"{cam}_{obj}_{f}" for f in _INTERP_FIELDS]
            trust = df[trust_col].to_numpy().astype(np.int8)

            # 0. Snapshot "did this row have a real raw detection?" BEFORE we
            #    nuke anything. Permanent-untrust mask — checked at the end to
            #    undo any interpolation-based rescue.
            raw_conf = df[f"{cam}_{obj}_conf"].to_numpy()
            raw_x1 = df[f"{cam}_{obj}_x1"].to_numpy()
            raw_y1 = df[f"{cam}_{obj}_y1"].to_numpy()
            raw_x2 = df[f"{cam}_{obj}_x2"].to_numpy()
            raw_y2 = df[f"{cam}_{obj}_y2"].to_numpy()
            conf_invalid = ~(np.isfinite(raw_conf) & (raw_conf > 0))
            bbox_nan = (
                ~np.isfinite(raw_x1) | ~np.isfinite(raw_y1)
                | ~np.isfinite(raw_x2) | ~np.isfinite(raw_y2)
            )
            bbox_zero = (
                np.isfinite(raw_x1) & np.isfinite(raw_y1)
                & np.isfinite(raw_x2) & np.isfinite(raw_y2)
                & (raw_x1 == 0) & (raw_y1 == 0) & (raw_x2 == 0) & (raw_y2 == 0)
            )
            bbox_invalid = bbox_nan | bbox_zero
            # Gripper_duck Filter A legitimately interpolates missing bboxes;
            # those rows have bbox_filled=True and should remain eligible for trust.
            if cam == "gripper" and obj == "duck":
                filled_ok = df["gripper_duck_bbox_filled"].to_numpy().astype(bool)
                bbox_invalid = bbox_invalid & ~filled_ok
            no_raw_detection = conf_invalid | bbox_invalid

            # 1. Null cx/cy/conf/bbox on trust=0 rows
            untrusted_mask = trust == 0
            df.loc[untrusted_mask, cols] = np.nan

            # 2. Linear interpolate per episode
            df[cols] = (
                df.groupby("episode_index")[cols]
                .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            )

            # 3. Re-check Filter E and v9 on newly-filled rows.
            # Any row that was untrusted + now has a valid cx => recovered candidate.
            cx_after = df[f"{cam}_{obj}_cx"].to_numpy()
            conf_after = df[f"{cam}_{obj}_conf"].to_numpy()
            has_value = np.isfinite(cx_after)

            if cam == "gripper" and obj == "duck":
                phase_arr = df["phase"].to_numpy()
                thresh = np.where(
                    phase_arr == "release",
                    CONF_THRESH_DEFAULT,
                    CONF_THRESH_GRIPPER_DUCK,
                ).astype(np.float32)
            else:
                thresh = np.full(len(df), CONF_THRESH_DEFAULT, dtype=np.float32)

            conf_ok = np.isfinite(conf_after) & (conf_after >= thresh)

            if cam == "top" and obj == "duck":
                y2_after = df[f"top_duck_y2"].to_numpy(dtype=np.float32)
                v9_ok = ~(np.isfinite(y2_after) & (y2_after > Y2_ARMLOCK_THRESHOLD))
            else:
                v9_ok = np.ones(len(df), dtype=bool)

            recovered = untrusted_mask & has_value & conf_ok & v9_ok
            trust = np.where(recovered, np.int8(1), trust)

            # 3b. Permanent untrust: rows that had no valid raw detection
            # (raw conf=0 OR raw bbox missing/zero) must stay trust=0, even if
            # interpolation produced plausible values. Without this the stable
            # re-pass rescues rows where teacher genuinely saw nothing.
            trust = np.where(no_raw_detection, np.int8(0), trust)
            df[trust_col] = trust

            # update reason for recovered rows
            reason_arr = np.array(df[reason_col].astype(object).tolist(), dtype=object)
            reason_arr[recovered] = R_ACCEPTED
            reason_arr[no_raw_detection] = R_NO_DETECTION
            df[reason_col] = pd.Categorical(reason_arr, categories=REASON_CATEGORIES)

    # 4. v10 cy-shift for left/right duck
    for cam in ["left", "right"]:
        x1 = df[f"{cam}_duck_x1"].to_numpy(dtype=np.float32)
        y1 = df[f"{cam}_duck_y1"].to_numpy(dtype=np.float32)
        x2 = df[f"{cam}_duck_x2"].to_numpy(dtype=np.float32)
        y2 = df[f"{cam}_duck_y2"].to_numpy(dtype=np.float32)
        w = x2 - x1
        h = y2 - y1
        should_square = np.isfinite(w) & np.isfinite(h) & (h > w)
        new_cy = y2 - w / 2.0
        cy_col = f"{cam}_duck_cy"
        cy_cur = df[cy_col].to_numpy(dtype=np.float32)
        cy_new = np.where(should_square, new_cy, cy_cur)
        df[cy_col] = cy_new.astype(np.float32)

    return df


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

def write_stats(
    df: pd.DataFrame,
    out_dir: Path,
    blue_tune: dict,
    thr_br: dict,
    thr_bg: dict,
    measure_log: dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Filter stats — A + B + E (v8)")
    md.append("")
    md.append(f"Total frames: **{len(df)}** across {df['episode_index'].nunique()} episodes.")
    md.append("")
    md.append("## V8 filter semantics (v8 delta: conf floor 0.05 -> 0.02)")
    md.append("- **A** (gripper_duck, grasp/transport):")
    md.append(f"  - NaN bbox (any conf): interpolate from +/-{INTERP_WINDOW} neighbors.")
    md.append("    - success -> blue gate; failure -> `phase_gripper_duck_no_neighbor`.")
    md.append("  - R_PHASE_OCCLUDED retained in enum for stats-compat; no longer produced.")
    md.append(f"  - Blue gate on INNER 50% shrunk bbox (scale={SHRINK_RATIO:.3f}):")
    md.append("    - v4/v5: B > thr_bg[cam_mode]*G  (B/R constraint dropped — audit "
              "showed orange duck head + red cup pull R up, causing false rejections).")
    md.append("- **B** (gripper_cup): geometry only.")
    md.append(f"  - in_jaw = cx>={CUP_JAW_CX_MIN} AND cy>={CUP_JAW_CY_MIN} AND area<={CUP_JAW_AREA_MAX} "
              "-> `gripper_cup_jaw_region`.")
    md.append(f"- **E**: gripper_duck grasp/transport>={CONF_THRESH_GRIPPER_DUCK}, "
              f"gripper_duck release>={CONF_THRESH_DEFAULT}, others>={CONF_THRESH_DEFAULT}.")
    md.append("")

    # Per-cluster threshold table
    md.append("## Per-cluster blue-dominance thresholds (measured)")
    md.append("")
    md.append("| cluster | n | B/R p05 | B/R p50 | B/G p05 | B/G p50 | thr_br | thr_bg |")
    md.append("|---------|----:|-------:|-------:|-------:|-------:|------:|------:|")
    for c in CLUSTER_NAMES:
        lg = measure_log.get(c, {})
        n = lg.get("n", 0)
        br05 = lg.get("br_p05")
        br50 = lg.get("br_median")
        bg05 = lg.get("bg_p05")
        bg50 = lg.get("bg_median")
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "-"
        md.append(
            f"| {c} | {n} | {fmt(br05)} | {fmt(br50)} | {fmt(bg05)} | {fmt(bg50)} | "
            f"{thr_br[c]:.2f} | {thr_bg[c]:.2f} |"
        )
    md.append("")
    dec = measure_log.get("_decision", {})
    md.append(f"thr range across clusters: B/R={dec.get('br_range', 0):.3f}, "
              f"B/G={dec.get('bg_range', 0):.3f}; meaningful-delta={BLUE_RATIO_MEANINGFUL_DELTA} -> "
              f"**apply_per_mode={dec.get('apply_per_mode', False)}**.")
    md.append("")

    # Phase distribution
    md.append("## Phase distribution")
    pcounts = df["phase"].value_counts().sort_index()
    md.append("| phase | count | % |")
    md.append("|------|------|---|")
    for p, c in pcounts.items():
        md.append(f"| {p} | {c} | {100 * c / len(df):.1f}% |")
    md.append("")

    # Per (cam, obj) breakdown
    md.append("## Per (cam, obj) trust + reason breakdown")
    md.append("")
    md.append("| cam | obj | total | accepted | lowconf | phase_occl | no_neighbor | no_blue | cup_jaw | no_det | % acc |")
    md.append("|-----|-----|------:|---------:|--------:|-----------:|------------:|--------:|--------:|-------:|------:|")
    for cam in CAMERAS:
        for obj in OBJECTS:
            reason_col = f"{cam}_{obj}_reason"
            vc = df[reason_col].value_counts()
            acc = int(vc.get(R_ACCEPTED, 0))
            lc = int(vc.get(R_LOWCONF, 0))
            po = int(vc.get(R_PHASE_OCCLUDED, 0))
            pn = int(vc.get(R_PHASE_NO_NEIGHBOR, 0))
            pnb = int(vc.get(R_PHASE_NO_BLUE, 0))
            cj = int(vc.get(R_CUP_JAW, 0))
            nd = int(vc.get(R_NO_DETECTION, 0))
            total = len(df)
            pct = 100 * acc / total if total else 0
            md.append(f"| {cam} | {obj} | {total} | {acc} | {lc} | {po} | {pn} | {pnb} | {cj} | {nd} | {pct:.1f}% |")
    md.append("")

    # Blue-threshold sensitivity
    md.append("## Filter A blue-dominance sensitivity (single-threshold report)")
    md.append(f"gripper_duck candidates in grasp/transport (conf>0): {blue_tune['candidates_total']}")
    md.append(f"- frames with shrunk-bbox RGB decoded: {blue_tune['bbox_with_rgb']}")
    md.append("")
    md.append("| threshold | kept blue | rejected no_blue |")
    md.append("|----------:|---------:|-----------------:|")
    for thr in BLUE_RATIO_THRESHOLDS_REPORT:
        kept = blue_tune[f"thr_{thr:.2f}_kept_blue"]
        rej = blue_tune[f"thr_{thr:.2f}_rejected_no_blue"]
        md.append(f"| {thr} | {kept} | {rej} |")
    md.append("")

    # Conf histograms
    md.append("## Confidence histograms (per cam/obj)")
    md.append("")
    for cam in CAMERAS:
        for obj in OBJECTS:
            conf_col = f"{cam}_{obj}_conf"
            trust_col = f"{cam}_{obj}_trust"
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            vals_all = df[conf_col].values
            vals_all = vals_all[vals_all > 0]
            axes[0].hist(vals_all, bins=50, color="gray")
            axes[0].set_title(f"{cam}_{obj} — all (conf>0)")
            axes[0].axvline(
                CONF_THRESH_GRIPPER_DUCK if (cam == "gripper" and obj == "duck") else CONF_THRESH_DEFAULT,
                color="red", linestyle="--", label="threshold",
            )
            axes[0].legend()
            vals_kept = df.loc[df[trust_col] == 1, conf_col].values
            axes[1].hist(vals_kept, bins=50, color="green")
            axes[1].set_title(f"{cam}_{obj} — accepted only")
            fig.tight_layout()
            png = out_dir / f"hist_{cam}_{obj}.png"
            fig.savefig(png, dpi=300)
            plt.close(fig)
            md.append(f"![{cam}_{obj}](hist_{cam}_{obj}.png)")
    md.append("")

    (out_dir / "filter_stats.md").write_text("\n".join(md))
    print(f"[stats] wrote {out_dir / 'filter_stats.md'}")


# ------------------------------------------------------------------
# Gallery rendering
# ------------------------------------------------------------------

def _load_font(size: int):
    for path in [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _render_overlay(
    frame: np.ndarray,
    bbox_norm: tuple[float, float, float, float] | None,
    accepted: bool,
    title_lines: list[str],
    out_path: Path,
    extra_bbox_norm: tuple[float, float, float, float] | None = None,
    extra_color: tuple[int, int, int] = (40, 140, 240),
    target_width: int = 512,
):
    h, w = frame.shape[:2]
    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)

    pil = Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR)
    header_h = 22 * len(title_lines) + 12
    canvas = Image.new("RGB", (new_w, new_h + header_h), (20, 20, 20))
    canvas.paste(pil, (0, header_h))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(13)
    y = 6
    for ln in title_lines:
        draw.text((8, y), ln, fill=(255, 255, 255), font=font)
        y += 20

    def _draw_box(bn, color):
        x1n, y1n, x2n, y2n = bn
        if any(np.isnan([x1n, y1n, x2n, y2n])):
            return
        x1 = int(np.clip(x1n * new_w, 0, new_w - 1))
        y1 = int(np.clip(y1n * new_h, 0, new_h - 1)) + header_h
        x2 = int(np.clip(x2n * new_w, 0, new_w - 1))
        y2 = int(np.clip(y2n * new_h, 0, new_h - 1)) + header_h
        for w_off in range(3):
            draw.rectangle([x1 - w_off, y1 - w_off, x2 + w_off, y2 + w_off], outline=color)

    if bbox_norm is not None:
        color = (0, 220, 0) if accepted else (220, 40, 40)
        _draw_box(bbox_norm, color)
    if extra_bbox_norm is not None:
        _draw_box(extra_bbox_norm, extra_color)

    canvas.save(out_path, format="PNG")


class GalleryRenderer:
    """Renders sample frames + overlays by streaming videos once per camera."""

    def __init__(self, ds_path: Path, episodes: pd.DataFrame, info: dict):
        self.ds_path = ds_path
        self.episodes = episodes
        self.info = info
        self.fps = info["fps"]
        self.cam_wh = {}
        for cam in CAMERAS:
            feat = info["features"][f"observation.images.{cam}"]
            self.cam_wh[cam] = (feat["info"]["video.width"], feat["info"]["video.height"])

    def render_samples(self, samples: list[dict], out_root: Path):
        by_cam: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            by_cam[s["cam"]].append(s)

        for cam, cam_samples in by_cam.items():
            cam_key = f"observation.images.{cam}"
            chunk_col = f"videos/{cam_key}/chunk_index"
            file_col = f"videos/{cam_key}/file_index"
            from_ts_col = f"videos/{cam_key}/from_timestamp"

            per_ep_samples: dict[int, list[dict]] = defaultdict(list)
            for s in cam_samples:
                per_ep_samples[s["ep"]].append(s)
            for ep in per_ep_samples:
                per_ep_samples[ep].sort(key=lambda x: x["fi"])

            vid_w, vid_h = self.cam_wh[cam]
            eps_needed = set(per_ep_samples.keys())
            ep_rows = self.episodes[self.episodes["episode_index"].isin(eps_needed)]
            video_groups = ep_rows.groupby([chunk_col, file_col])

            desc = f"  gallery[{cam}]"
            for (chunk_idx, file_idx), group_eps in tqdm(list(video_groups), desc=desc, unit="vid"):
                video_path = get_video_path(self.ds_path, cam_key, int(chunk_idx), int(file_idx))
                if not video_path.exists():
                    continue
                group_eps = group_eps.sort_values(from_ts_col)

                with VideoReader(video_path, vid_w, vid_h) as reader:
                    for _, ep_row in group_eps.iterrows():
                        ep_idx = int(ep_row["episode_index"])
                        n_ep_frames = int(ep_row["length"])
                        from_ts = float(ep_row[from_ts_col])
                        start_frame = round(from_ts * self.fps)

                        wanted = per_ep_samples.get(ep_idx, [])
                        if not wanted:
                            reader.skip(n_ep_frames)
                            continue

                        if reader._pos < start_frame:
                            reader.skip(start_frame - reader._pos)

                        cur_local = 0
                        wi = 0
                        while cur_local < n_ep_frames and wi < len(wanted):
                            tgt_fi = wanted[wi]["fi"]
                            gap = tgt_fi - cur_local
                            if gap > 0:
                                reader.skip(gap)
                                cur_local += gap
                            frame = reader.read_one()
                            if frame is None:
                                break
                            cur_local += 1
                            out_subdir = out_root / wanted[wi]["subfolder"]
                            out_subdir.mkdir(parents=True, exist_ok=True)
                            _render_overlay(
                                frame,
                                wanted[wi]["bbox_norm"],
                                wanted[wi]["accepted"],
                                wanted[wi]["title_lines"],
                                out_subdir / wanted[wi]["filename"],
                                extra_bbox_norm=wanted[wi].get("extra_bbox_norm"),
                            )
                            wi += 1
                        remaining = n_ep_frames - cur_local
                        if remaining > 0:
                            reader.skip(remaining)


def _pick_samples(df: pd.DataFrame, n: int, rng: random.Random) -> list[pd.Series]:
    if len(df) == 0:
        return []
    idx = list(range(len(df)))
    rng.shuffle(idx)
    pick = idx[:n]
    return [df.iloc[i] for i in pick]


def _shrunk_box_from(x1, y1, x2, y2, shrink=SHRINK_RATIO):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * shrink
    bh = (y2 - y1) * shrink
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def build_galleries(
    df: pd.DataFrame,
    renderer: GalleryRenderer,
    gallery_root: Path,
    n_per_folder: int = 24,
    seed: int = 42,
):
    rng = random.Random(seed)
    samples: list[dict] = []

    def _title_base(cam, obj, ep, fi, conf, phase, reason, cam_mode=""):
        return [
            f"cam={cam}  obj={obj}  ep={ep}  fr={fi}",
            f"teacher_conf={conf:.3f}  phase={phase}  reason={reason}  mode={cam_mode}",
        ]

    # --- blue-gate helper text
    def _bluegate_line(s):
        r = s.get("gripper_duck_mean_r_shrunk")
        g = s.get("gripper_duck_mean_g_shrunk")
        b = s.get("gripper_duck_mean_b_shrunk")
        if pd.isna(r):
            return "meanRGB(shrunk)=N/A"
        br = b / r if r > 0 else float("nan")
        bg = b / g if g > 0 else float("nan")
        return (f"shrunk meanRGB=({r:.0f}, {g:.0f}, {b:.0f})  "
                f"B/R={br:.2f}  B/G={bg:.2f}")

    def _eff_bbox(s):
        x1 = s["gripper_duck_x1_eff"]
        y1 = s["gripper_duck_y1_eff"]
        x2 = s["gripper_duck_x2_eff"]
        y2 = s["gripper_duck_y2_eff"]
        return (x1, y1, x2, y2)

    # 1. rejected_phase_A_no_neighbor
    mask = df["gripper_duck_reason"] == R_PHASE_NO_NEIGHBOR
    for s in _pick_samples(df[mask], n_per_folder, rng):
        tl = _title_base("gripper", "duck", int(s["episode_index"]), int(s["frame_index"]),
                         float(s["gripper_duck_conf"]), s["phase"], R_PHASE_NO_NEIGHBOR,
                         cam_mode=s.get("cam_mode", ""))
        tl.append(f"NaN bbox, no valid neighbor in +/-{INTERP_WINDOW} frames")
        samples.append({
            "subfolder": "rejected_phase_A_no_neighbor",
            "cam": "gripper",
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": None,
            "accepted": False,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_gripper_duck.png",
            "title_lines": tl,
        })

    # 2. rejected_phase_A_occluded (low-conf NaN)
    mask = df["gripper_duck_reason"] == R_PHASE_OCCLUDED
    for s in _pick_samples(df[mask], n_per_folder, rng):
        tl = _title_base("gripper", "duck", int(s["episode_index"]), int(s["frame_index"]),
                         float(s["gripper_duck_conf"]), s["phase"], R_PHASE_OCCLUDED,
                         cam_mode=s.get("cam_mode", ""))
        tl.append(f"NaN bbox, conf<=interp gate ({INTERP_CONF_GATE})")
        samples.append({
            "subfolder": "rejected_phase_A_occluded",
            "cam": "gripper",
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": None,
            "accepted": False,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_gripper_duck.png",
            "title_lines": tl,
        })

    # 3. rejected_phase_A_no_blue
    mask = df["gripper_duck_reason"] == R_PHASE_NO_BLUE
    for s in _pick_samples(df[mask], n_per_folder, rng):
        bbox_full = _eff_bbox(s)
        bbox_shrunk = _shrunk_box_from(*bbox_full)
        tl = _title_base("gripper", "duck", int(s["episode_index"]), int(s["frame_index"]),
                         float(s["gripper_duck_conf"]), s["phase"], R_PHASE_NO_BLUE,
                         cam_mode=s.get("cam_mode", ""))
        tl.append(_bluegate_line(s))
        interp_flag = " [INTERP]" if bool(s.get("gripper_duck_bbox_filled", False)) else ""
        tl.append(f"bbox={bbox_full[0]:.2f},{bbox_full[1]:.2f},{bbox_full[2]:.2f},{bbox_full[3]:.2f}"
                  f"{interp_flag}")
        samples.append({
            "subfolder": "rejected_phase_A_no_blue",
            "cam": "gripper",
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": bbox_full,
            "extra_bbox_norm": bbox_shrunk,
            "accepted": False,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_gripper_duck.png",
            "title_lines": tl,
        })

    # 4. kept_phase_A_interp — accepted via interp bbox
    mask = (
        (df["gripper_duck_reason"] == R_ACCEPTED)
        & df["gripper_duck_bbox_filled"]
        & df["phase"].isin(["grasp", "transport"])
    )
    for s in _pick_samples(df[mask], n_per_folder, rng):
        bbox_full = _eff_bbox(s)
        bbox_shrunk = _shrunk_box_from(*bbox_full)
        tl = _title_base("gripper", "duck", int(s["episode_index"]), int(s["frame_index"]),
                         float(s["gripper_duck_conf"]), s["phase"], "accepted_interp",
                         cam_mode=s.get("cam_mode", ""))
        tl.append(_bluegate_line(s))
        tl.append("INTERPOLATED bbox passed blue gate")
        samples.append({
            "subfolder": "kept_phase_A_interp",
            "cam": "gripper",
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": bbox_full,
            "extra_bbox_norm": bbox_shrunk,
            "accepted": True,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_gripper_duck.png",
            "title_lines": tl,
        })

    # 5. kept_phase_A_blue — accepted via real bbox
    mask = (
        (df["gripper_duck_reason"] == R_ACCEPTED)
        & ~df["gripper_duck_bbox_filled"]
        & df["phase"].isin(["grasp", "transport"])
        & df["gripper_duck_x1"].notna()
    )
    for s in _pick_samples(df[mask], n_per_folder, rng):
        bbox_full = (s["gripper_duck_x1"], s["gripper_duck_y1"],
                     s["gripper_duck_x2"], s["gripper_duck_y2"])
        bbox_shrunk = _shrunk_box_from(*bbox_full)
        tl = _title_base("gripper", "duck", int(s["episode_index"]), int(s["frame_index"]),
                         float(s["gripper_duck_conf"]), s["phase"], "accepted_blue",
                         cam_mode=s.get("cam_mode", ""))
        tl.append(_bluegate_line(s))
        samples.append({
            "subfolder": "kept_phase_A_blue",
            "cam": "gripper",
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": bbox_full,
            "extra_bbox_norm": bbox_shrunk,
            "accepted": True,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_gripper_duck.png",
            "title_lines": tl,
        })

    # 6. rejected_gripper_cup_jaw — Filter B geometric
    mask = df["gripper_cup_reason"] == R_CUP_JAW
    for s in _pick_samples(df[mask], n_per_folder, rng):
        bbox = (s["gripper_cup_x1"], s["gripper_cup_y1"],
                s["gripper_cup_x2"], s["gripper_cup_y2"])
        tl = _title_base("gripper", "cup", int(s["episode_index"]), int(s["frame_index"]),
                         float(s["gripper_cup_conf"]), s["phase"], R_CUP_JAW,
                         cam_mode=s.get("cam_mode", ""))
        tl.append(f"cx={s['gripper_cup_cx']:.2f}  cy={s['gripper_cup_cy']:.2f}  "
                  f"area={s['gripper_cup_area_norm']:.3f}")
        samples.append({
            "subfolder": "rejected_gripper_cup_jaw",
            "cam": "gripper",
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": bbox,
            "accepted": False,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_gripper_cup.png",
            "title_lines": tl,
        })

    # 7. accepted_gripper_cup
    mask = (df["gripper_cup_reason"] == R_ACCEPTED) & df["gripper_cup_x1"].notna()
    for s in _pick_samples(df[mask], n_per_folder, rng):
        bbox = (s["gripper_cup_x1"], s["gripper_cup_y1"],
                s["gripper_cup_x2"], s["gripper_cup_y2"])
        tl = _title_base("gripper", "cup", int(s["episode_index"]), int(s["frame_index"]),
                         float(s["gripper_cup_conf"]), s["phase"], R_ACCEPTED,
                         cam_mode=s.get("cam_mode", ""))
        tl.append(f"cx={s['gripper_cup_cx']:.2f}  cy={s['gripper_cup_cy']:.2f}  "
                  f"area={s['gripper_cup_area_norm']:.3f}")
        samples.append({
            "subfolder": "accepted_gripper_cup",
            "cam": "gripper",
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": bbox,
            "accepted": True,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_gripper_cup.png",
            "title_lines": tl,
        })

    # 8. rejected_lowconf_E — spread across cams/objs
    lowconf_rows: list[tuple[str, str, pd.Series]] = []
    for cam in CAMERAS:
        for obj in OBJECTS:
            reason_col = f"{cam}_{obj}_reason"
            sub = df[df[reason_col] == R_LOWCONF]
            for s in _pick_samples(sub, max(1, n_per_folder // (len(CAMERAS) * len(OBJECTS))), rng):
                lowconf_rows.append((cam, obj, s))
    rng.shuffle(lowconf_rows)
    for cam, obj, s in lowconf_rows[:n_per_folder]:
        bbox = (s[f"{cam}_{obj}_x1"], s[f"{cam}_{obj}_y1"],
                s[f"{cam}_{obj}_x2"], s[f"{cam}_{obj}_y2"])
        if any(pd.isna(v) for v in bbox):
            bbox = None
        samples.append({
            "subfolder": "rejected_lowconf_E",
            "cam": cam,
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": bbox,
            "accepted": False,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_{cam}_{obj}.png",
            "title_lines": _title_base(cam, obj, int(s["episode_index"]), int(s["frame_index"]),
                                       float(s[f"{cam}_{obj}_conf"]), s["phase"], R_LOWCONF,
                                       cam_mode=s.get("cam_mode", "")),
        })

    # 9. accepted_gripper_duck — sanity
    mask = (
        (df["gripper_duck_reason"] == R_ACCEPTED)
        & df["gripper_duck_x1_eff"].notna()
    )
    for s in _pick_samples(df[mask], n_per_folder, rng):
        bbox = (s["gripper_duck_x1_eff"], s["gripper_duck_y1_eff"],
                s["gripper_duck_x2_eff"], s["gripper_duck_y2_eff"])
        tl = _title_base("gripper", "duck", int(s["episode_index"]), int(s["frame_index"]),
                         float(s["gripper_duck_conf"]), s["phase"], R_ACCEPTED,
                         cam_mode=s.get("cam_mode", ""))
        if not pd.isna(s.get("gripper_duck_mean_r_shrunk", np.nan)):
            tl.append(_bluegate_line(s))
        interp_flag = " [INTERP]" if bool(s.get("gripper_duck_bbox_filled", False)) else ""
        tl.append(f"bbox={bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}{interp_flag}")
        samples.append({
            "subfolder": "accepted_gripper_duck",
            "cam": "gripper",
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": bbox,
            "accepted": True,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_gripper_duck.png",
            "title_lines": tl,
        })

    # 10. accepted_sampled
    all_accepted: list[tuple[str, str, pd.Series]] = []
    for cam in CAMERAS:
        for obj in OBJECTS:
            reason_col = f"{cam}_{obj}_reason"
            sub = df[(df[reason_col] == R_ACCEPTED) & (df[f"{cam}_{obj}_x1"].notna())]
            for s in _pick_samples(sub, max(1, n_per_folder // (len(CAMERAS) * len(OBJECTS))), rng):
                all_accepted.append((cam, obj, s))
    rng.shuffle(all_accepted)
    for cam, obj, s in all_accepted[:n_per_folder]:
        bbox = (s[f"{cam}_{obj}_x1"], s[f"{cam}_{obj}_y1"],
                s[f"{cam}_{obj}_x2"], s[f"{cam}_{obj}_y2"])
        samples.append({
            "subfolder": "accepted_sampled",
            "cam": cam,
            "ep": int(s["episode_index"]),
            "fi": int(s["frame_index"]),
            "bbox_norm": bbox,
            "accepted": True,
            "filename": f"ep{int(s['episode_index']):03d}_fr{int(s['frame_index']):05d}_{cam}_{obj}.png",
            "title_lines": _title_base(cam, obj, int(s["episode_index"]), int(s["frame_index"]),
                                       float(s[f"{cam}_{obj}_conf"]), s["phase"], R_ACCEPTED,
                                       cam_mode=s.get("cam_mode", "")),
        })

    print(f"[gallery] rendering {len(samples)} samples across "
          f"{len({s['subfolder'] for s in samples})} folders")
    renderer.render_samples(samples, gallery_root)


# ------------------------------------------------------------------
# Findings summary
# ------------------------------------------------------------------

def write_findings(
    df: pd.DataFrame,
    out_dir: Path,
    phases_ok: bool,
    blue_tune: dict,
    thr_br: dict,
    thr_bg: dict,
    measure_log: dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Findings — filter sanity check (v3)")
    lines.append("")
    lines.append(f"Phases computed cleanly: **{phases_ok}**")
    lines.append("")

    lines.append("## What changed between v2 and v3")
    lines.append("")
    lines.append("- **Filter A step 1** — NaN bbox is now split by teacher_conf:")
    lines.append(f"  - conf>{INTERP_CONF_GATE} -> try to interpolate (cx,cy,w,h) from neighbours "
                 f"in +/-{INTERP_WINDOW} frames, weighted by 1/distance; failure -> "
                 "`phase_gripper_duck_no_neighbor`.")
    lines.append(f"  - conf<={INTERP_CONF_GATE} -> still `phase_gripper_duck_occluded`.")
    lines.append("- **Filter A step 2** — blue gate now samples the INNER 50% area of the bbox "
                 f"(scale={SHRINK_RATIO:.3f}) to avoid red-cup contamination.")
    lines.append("- **Per-cluster thresholds** — measured on tentative-accepted frames (B/R>=1.25 "
                 "AND B/G>=1.25) in each lighting cluster, then thr = 5th percentile of the ratio.")
    lines.append("- **Filter B** — color/red-dominance rule dropped. Pure geometry: "
                 f"cx>={CUP_JAW_CX_MIN} AND cy>={CUP_JAW_CY_MIN} AND area_norm<={CUP_JAW_AREA_MAX} "
                 "-> `gripper_cup_jaw_region`.")
    lines.append("- **Reason codes** — added `phase_gripper_duck_no_neighbor`, `gripper_cup_jaw_region`; "
                 "removed `red_pixel_gripper_cam_cup_bottom`.")
    lines.append("")

    lines.append("## Per-cluster threshold decision")
    lines.append("")
    lines.append("| cluster | n | B/R p05 | B/R p50 | B/G p05 | B/G p50 | thr_br | thr_bg |")
    lines.append("|---------|----:|-------:|-------:|-------:|-------:|------:|------:|")
    for c in CLUSTER_NAMES:
        lg = measure_log.get(c, {})
        n = lg.get("n", 0)
        br05 = lg.get("br_p05")
        br50 = lg.get("br_median")
        bg05 = lg.get("bg_p05")
        bg50 = lg.get("bg_median")
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "-"
        lines.append(
            f"| {c} | {n} | {fmt(br05)} | {fmt(br50)} | {fmt(bg05)} | {fmt(bg50)} | "
            f"{thr_br[c]:.2f} | {thr_bg[c]:.2f} |"
        )
    lines.append("")
    dec = measure_log.get("_decision", {})
    lines.append(f"thr range B/R={dec.get('br_range', 0):.3f}  B/G={dec.get('bg_range', 0):.3f}  "
                 f"meaningful-delta={BLUE_RATIO_MEANINGFUL_DELTA} -> "
                 f"**apply_per_mode={dec.get('apply_per_mode', False)}**.")
    lines.append("")

    lines.append("## Per-(cam,obj) rejection table (v3)")
    lines.append("")
    lines.append("| cam | obj | accepted | lowconf | phase_occl | no_neighbor | no_blue | cup_jaw | no_det |")
    lines.append("|-----|-----|---------:|--------:|-----------:|------------:|--------:|--------:|-------:|")
    for cam in CAMERAS:
        for obj in OBJECTS:
            reason_col = f"{cam}_{obj}_reason"
            vc = df[reason_col].value_counts()
            total = len(df)
            lines.append(
                f"| {cam} | {obj} | "
                f"{100 * int(vc.get(R_ACCEPTED, 0)) / total:.1f}% | "
                f"{100 * int(vc.get(R_LOWCONF, 0)) / total:.1f}% | "
                f"{100 * int(vc.get(R_PHASE_OCCLUDED, 0)) / total:.1f}% | "
                f"{100 * int(vc.get(R_PHASE_NO_NEIGHBOR, 0)) / total:.1f}% | "
                f"{100 * int(vc.get(R_PHASE_NO_BLUE, 0)) / total:.1f}% | "
                f"{100 * int(vc.get(R_CUP_JAW, 0)) / total:.1f}% | "
                f"{100 * int(vc.get(R_NO_DETECTION, 0)) / total:.1f}% |"
            )
    lines.append("")

    lines.append("## Filter A sensitivity (single-threshold report)")
    lines.append(f"candidates={blue_tune['candidates_total']}  RGB decoded={blue_tune['bbox_with_rgb']}")
    lines.append("")
    lines.append("| threshold | kept | rejected |")
    lines.append("|----------:|----:|--------:|")
    for thr in BLUE_RATIO_THRESHOLDS_REPORT:
        kept = blue_tune[f"thr_{thr:.2f}_kept_blue"]
        rej = blue_tune[f"thr_{thr:.2f}_rejected_no_blue"]
        lines.append(f"| {thr} | {kept} | {rej} |")
    lines.append("")

    # Top episodes by gripper_duck rejection
    gd = df.copy()
    gd["gd_reject"] = (gd["gripper_duck_reason"] != R_ACCEPTED) & (gd["gripper_duck_conf"] > 0)
    per_ep = gd.groupby("episode_index").agg(
        total=("frame_index", "count"),
        rej=("gd_reject", "sum"),
    )
    per_ep["rej_pct"] = 100 * per_ep["rej"] / per_ep["total"]
    outliers = per_ep.sort_values("rej_pct", ascending=False).head(10)
    lines.append("## Top 10 episodes by gripper_duck rejection %")
    lines.append("")
    lines.append("| episode | total | rejected | % |")
    lines.append("|--------:|------:|---------:|---:|")
    for ep, row in outliers.iterrows():
        lines.append(f"| {ep} | {row['total']} | {int(row['rej'])} | {row['rej_pct']:.1f}% |")
    lines.append("")

    # Blue ratio hist (shrunk)
    duck_rgb = df[df["gripper_duck_mean_r_shrunk"].notna()]
    if len(duck_rgb):
        blue_ratio = duck_rgb["gripper_duck_mean_b_shrunk"] / duck_rgb[[
            "gripper_duck_mean_r_shrunk", "gripper_duck_mean_g_shrunk"
        ]].max(axis=1)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(blue_ratio, bins=80, color="steelblue")
        for thr in BLUE_RATIO_THRESHOLDS_REPORT:
            ls = "-" if abs(thr - BLUE_RATIO_FALLBACK) < 1e-6 else "--"
            ax.axvline(thr, color="red", linestyle=ls, label=f"thr={thr}")
        ax.set_xlabel("B / max(R, G)  [inner 50% bbox]")
        ax.set_ylabel("count")
        ax.set_title("gripper_duck SHRUNK-bbox blue-ratio")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "blue_ratio_hist_v3.png", dpi=300)
        plt.close(fig)
        lines.append("![blue_ratio_hist_v3](blue_ratio_hist_v3.png)")
        lines.append("")

    (out_dir / "findings_v3.md").write_text("\n".join(lines))
    print(f"[findings] wrote {out_dir / 'findings_v3.md'}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply A+B+E v3 filters to detection parquet.")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repo-id or path")
    parser.add_argument("--output", required=True, help="Cleaned parquet output path")
    parser.add_argument("--gallery-dir", required=True,
                        help="Where to write data_analysis/ outputs (stats, galleries)")
    parser.add_argument("--n-samples", type=int, default=24, help="Samples per gallery subfolder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-phases", action="store_true",
                        help="Skip Filter A (debug mode — all phases set to 'unknown')")
    args = parser.parse_args()

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    ds_path = resolve_dataset_path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gallery_root = Path(args.gallery_dir)
    gallery_root.mkdir(parents=True, exist_ok=True)

    # 1. Load detection parquet
    det_path = ds_path / "detection_results.parquet"
    print(f"[main] Loading {det_path}")
    ts = time.perf_counter()
    det_df = pd.read_parquet(det_path)
    print(f"[main] {len(det_df)} rows, {det_df['episode_index'].nunique()} episodes")
    timings["load_parquet"] = time.perf_counter() - ts

    info, episodes = load_dataset_meta(ds_path)
    fps = info["fps"]

    # 2. Phases
    ts = time.perf_counter()
    phases_ok = True
    if args.skip_phases:
        print("[main] --skip-phases set, filling phase=unknown")
        phase_df = det_df[["frame_index", "episode_index"]].copy()
        phase_df["phase"] = "unknown"
        phases_ok = False
    else:
        print("[main] Running phase detection...")
        try:
            phase_df = compute_phases(ds_path, fps=fps)
        except Exception as e:
            print(f"[main] phases failed: {e}. Falling back to phase=unknown.")
            phase_df = det_df[["frame_index", "episode_index"]].copy()
            phase_df["phase"] = "unknown"
            phases_ok = False
    timings["phases"] = time.perf_counter() - ts

    # 3. Load cluster modes
    modes_df = load_cluster_modes()
    print(f"[main] cluster_modes: {modes_df['cam_mode'].value_counts().to_dict()}")

    # 4. Interpolate bboxes (cheap, does not decode video)
    ts = time.perf_counter()
    print("[main] Interpolating NaN-bbox high-conf gripper_duck frames...")
    # We need phase merged in so interp happens on all gripper_duck rows; then resample RGB
    det_with_interp = interpolate_gripper_duck_bboxes(det_df.copy())
    filled_n = int(det_with_interp["gripper_duck_bbox_filled"].sum())
    nan_bbox_n = int(det_df["gripper_duck_x1"].isna().sum())
    print(f"[main] gripper_duck NaN bbox rows: {nan_bbox_n}; interp filled: {filled_n}")
    timings["interp"] = time.perf_counter() - ts

    # 5. Pixel sampling on shrunk inner-50% bbox (uses gripper_duck_x1_eff etc.)
    ts = time.perf_counter()
    print("[main] Sampling shrunk-bbox RGB for gripper_duck Filter A...")
    duck_stats = compute_gripper_duck_pixel_stats(
        det_with_interp, phase_df, ds_path, episodes, info,
    )
    timings["pixel_sampling"] = time.perf_counter() - ts

    # 6. Build cleaned DF (v8 trust flags + v9 top_duck armlock)
    ts = time.perf_counter()
    print("[main] Building cleaned dataframe (v8+v9)...")
    clean_df, blue_tune, thr_br, thr_bg, measure_log = build_cleaned_df(
        det_df, phase_df, modes_df, duck_stats,
    )
    timings["build_df"] = time.perf_counter() - ts

    # 6b. Apply trust-gated interpolation + stable re-pass + v10 cy shift.
    # This is what makes the output "final" — cx/cy/conf/bbox dense across every
    # frame, only interpolated between trust=1 endpoints, with v10 target
    # correction for left/right duck.
    ts = time.perf_counter()
    print("[main] Trust-gated interpolation + stable re-pass + v10 cy shift...")
    clean_df = apply_trust_gated_interp_and_v10(clean_df)
    timings["interp_and_v10"] = time.perf_counter() - ts

    # 7. Slim output parquet
    out_cols = ["frame_index", "episode_index", "phase", "cam_mode",
                "gripper_duck_bbox_filled"]
    for cam in CAMERAS:
        for obj in OBJECTS:
            out_cols += [
                f"{cam}_{obj}_cx", f"{cam}_{obj}_cy", f"{cam}_{obj}_conf",
                f"{cam}_{obj}_x1", f"{cam}_{obj}_y1",
                f"{cam}_{obj}_x2", f"{cam}_{obj}_y2",
                f"{cam}_{obj}_trust", f"{cam}_{obj}_reason",
            ]
    slim = clean_df[out_cols].copy()
    slim["frame_index"] = slim["frame_index"].astype(np.int64)
    slim["episode_index"] = slim["episode_index"].astype(np.int64)
    slim["gripper_duck_bbox_filled"] = slim["gripper_duck_bbox_filled"].astype(bool)
    for cam in CAMERAS:
        for obj in OBJECTS:
            for field in ["cx", "cy", "conf", "x1", "y1", "x2", "y2"]:
                slim[f"{cam}_{obj}_{field}"] = slim[f"{cam}_{obj}_{field}"].astype(np.float32)
            slim[f"{cam}_{obj}_trust"] = slim[f"{cam}_{obj}_trust"].astype(np.int8)
    slim.to_parquet(output_path, index=False)
    print(f"[main] Wrote final parquet -> {output_path}")

    # 8. Stats + galleries + findings
    data_dir = gallery_root

    ts = time.perf_counter()
    write_stats(clean_df, data_dir, blue_tune, thr_br, thr_bg, measure_log)
    timings["stats"] = time.perf_counter() - ts

    # Wipe entire old gallery/ dir so no stale v2 folders remain
    gallery_dir = data_dir / "gallery"
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    ts = time.perf_counter()
    renderer = GalleryRenderer(ds_path, episodes, info)
    build_galleries(clean_df, renderer, gallery_dir, n_per_folder=args.n_samples, seed=args.seed)
    timings["galleries"] = time.perf_counter() - ts

    ts = time.perf_counter()
    write_findings(clean_df, data_dir, phases_ok, blue_tune, thr_br, thr_bg, measure_log)
    timings["findings"] = time.perf_counter() - ts

    elapsed = time.perf_counter() - t0
    print("")
    print("=" * 52)
    print("[timing] Per-stage seconds:")
    for k, v in timings.items():
        print(f"  {k:20s} {v:7.1f}")
    print(f"[timing] {'TOTAL':20s} {elapsed:7.1f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
