"""Stage 1 — initial DINO detection cleaning.

A clean re-build. Reads the raw per-frame G-DINO parquet and applies four
filters. The output is the same frame space as the input, with per
(cam, obj) `trust ∈ {0, 1}` + `reason` columns. Raw bboxes pass through
untouched.

Short-circuit per (cam, obj, frame):
    1. No raw detection       → reason = no_detection
       (conf missing/≤0 OR bbox NaN OR bbox is (0,0,0,0))
    2. gripper_duck, phase ∈ {grasp, transport}:  blue-gate on inner-50% bbox
       fails → reason = phase_gripper_duck_no_blue
    3. gripper_cup:  cx≥0.74 AND cy≥0.81 AND bbox_area≤0.14
       → reason = gripper_cup_jaw_region
    4. top_duck:  y2 > 0.85  → reason = top_duck_armlock_y2
    5. conf < threshold  → reason = low_confidence
       thresholds: gripper_duck (grasp/transport) = 0.15, everywhere else = 0.02
    6. else  → reason = accepted, trust = 1

NO interpolation. NO rescue. NO v10 cy-shift. Per-(cam, obj) independent:
a frame with top_duck rejected can still have left_duck accepted.

Usage:
    python -m vbti.logic.detection.distill_stage1 \\
        --dataset eternalmay33/01_02_03_merged_may-sim \\
        --output /home/may33/.cache/vbti/detection_labels_stage1.parquet

    # Visually verify with the viewer:
    python -m vbti.logic.dataset.viewer eternalmay33/01_02_03_merged_may-sim \\
        --parq /home/may33/.cache/vbti/detection_labels_stage1.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vbti.logic.dataset import resolve_dataset_path
from vbti.logic.detection.process_dataset import load_dataset_meta
from vbti.logic.detection.distill.distill_filter import (
    CAMERAS,
    CONF_THRESH_DEFAULT,
    CONF_THRESH_GRIPPER_DUCK,
    CUP_JAW_AREA_MAX,
    CUP_JAW_CX_MIN,
    CUP_JAW_CY_MIN,
    OBJECTS,
    R_ACCEPTED,
    R_CUP_JAW,
    R_CUP_INSIDE_DUCK,
    R_LEFT_DUCK_FIXED_BLOB,
    R_LEFT_DUCK_TOP_STRIP,
    R_LOWCONF,
    R_NO_DETECTION,
    R_PHASE_NO_BLUE,
    R_RIGHT_DUCK_ARM_BASE,
    R_SIDE_DUCK_RELEASE,
    R_TOP_DUCK_ARMLOCK,
    R_TOP_DUCK_FIXED_BLOB,
    REASON_CATEGORIES,
    Y2_ARMLOCK_THRESHOLD,
    compute_gripper_duck_pixel_stats,
    compute_phases,
    load_cluster_modes,
    measure_cluster_thresholds,
)

# Fixed-location artifact thresholds
LEFT_DUCK_TOP_STRIP_Y2_MAX = 0.10      # left_duck y2<0.10 = stuck at top strip

# left_duck fixed-blob: tiny ghost detection at cx≈0.28, cy≈0.74 (seen on 87+ eps).
# Calibrated from ep 83 fr 480 as reference.
LEFT_DUCK_FIXED_BLOB_CX_MIN = 0.27
LEFT_DUCK_FIXED_BLOB_CX_MAX = 0.29
LEFT_DUCK_FIXED_BLOB_CY_MIN = 0.73
LEFT_DUCK_FIXED_BLOB_CY_MAX = 0.75
LEFT_DUCK_FIXED_BLOB_AREA_MAX = 0.006

# right_duck arm base: bbox hugs left edge (robot arm enters frame on the left).
# Calibrated from ep 82 fr 492: arm-base x2 ≈ 0.202, so rejection threshold is
# x1 < (x2_armbase - 50px_normalized) = 0.202 - 50/640 = 0.124.
RIGHT_DUCK_ARM_BASE_X1_MAX = 0.124

TOP_DUCK_FIXED_BLOB_CX_MIN = 0.34      # top cam has a pixel-perfect ghost
TOP_DUCK_FIXED_BLOB_CX_MAX = 0.40
TOP_DUCK_FIXED_BLOB_CY_MIN = 0.26
TOP_DUCK_FIXED_BLOB_CY_MAX = 0.32
TOP_DUCK_FIXED_BLOB_AREA_MAX = 0.008

# Release-phase side-duck distance threshold (normalized coords).
# During release, keep left/right duck only if its center is within this
# distance of the cup center AND the cup itself is trusted on the same cam.
# Motivation: during release the duck is placed onto the cup — legitimate
# detections are directly on/adjacent to the cup. Detections elsewhere are
# usually jaw/reflection artefacts.
RELEASE_DUCK_CUP_MAX_DIST = 0.20


# ------------------------------------------------------------------
# RGB cache
# ------------------------------------------------------------------
# The shrunk-bbox RGB sampling on gripper_duck is the slow part (~70s from
# video decode). It only depends on (frame image, bbox) — both invariant
# across filter-threshold changes. Cache to parquet so threshold iteration
# is near-instant.

def _load_rgb_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    print(f"[rgb-cache] loading {cache_path}")
    df = pd.read_parquet(cache_path)
    ep = df["episode_index"].to_numpy(np.int64)
    fr = df["frame_index"].to_numpy(np.int64)
    r = df["r"].to_numpy(np.float32)
    g = df["g"].to_numpy(np.float32)
    b = df["b"].to_numpy(np.float32)
    stats = {
        (int(ep[i]), int(fr[i])): {"r": float(r[i]), "g": float(g[i]), "b": float(b[i])}
        for i in range(len(df))
    }
    print(f"[rgb-cache] {len(stats)} entries loaded")
    return stats


def _save_rgb_cache(cache_path: Path, stats: dict) -> None:
    rows = [
        {"episode_index": int(ep), "frame_index": int(fr),
         "r": float(v["r"]), "g": float(v["g"]), "b": float(v["b"])}
        for (ep, fr), v in stats.items()
    ]
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[rgb-cache] saved {len(df)} entries -> {cache_path}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _raw_detection_mask(conf, x1, y1, x2, y2):
    """True where DINO actually emitted a real detection on this frame.

    Real = conf > 0 AND all 4 bbox coords finite AND bbox is not the
    all-zero sentinel (0,0,0,0) that the pipeline writes when teacher
    returned nothing.
    """
    conf_ok = np.isfinite(conf) & (conf > 0)
    bbox_finite = (
        np.isfinite(x1) & np.isfinite(y1)
        & np.isfinite(x2) & np.isfinite(y2)
    )
    bbox_zero = (
        bbox_finite
        & (x1 == 0) & (y1 == 0) & (x2 == 0) & (y2 == 0)
    )
    return conf_ok & bbox_finite & ~bbox_zero


# ------------------------------------------------------------------
# Main filter
# ------------------------------------------------------------------

def apply_stage1_filter(
    det_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    modes_df: pd.DataFrame,
    duck_stats: dict,
) -> pd.DataFrame:
    """Apply v9 + A + B + E to the raw detection dataframe.

    Returns a new df with per (cam, obj):
        cx, cy, conf, x1, y1, x2, y2  (raw, unchanged)
        trust  ∈ {0, 1}
        reason ∈ REASON_CATEGORIES
    Plus: phase, cam_mode columns merged in.
    """
    df = det_df.merge(phase_df, on=["frame_index", "episode_index"], how="left")
    df["phase"] = df["phase"].fillna("unknown").astype("category")

    df = df.merge(modes_df, on="episode_index", how="left")
    df["cam_mode"] = df["cam_mode"].fillna("realsense_v1").astype("category")

    # ------------------------------------------------------------------
    # Blue-gate prep (gripper_duck Filter A only)
    # ------------------------------------------------------------------
    # Attach shrunk-bbox RGB (if any) from duck_stats dict.
    if duck_stats:
        idx = pd.MultiIndex.from_tuples(
            list(duck_stats.keys()),
            names=["episode_index", "frame_index"],
        )
        vals = np.array(
            [[v["r"], v["g"], v["b"]] for v in duck_stats.values()],
            dtype=np.float32,
        )
        rgb_df = pd.DataFrame(
            {
                "gripper_duck_mean_r_shrunk": vals[:, 0],
                "gripper_duck_mean_g_shrunk": vals[:, 1],
                "gripper_duck_mean_b_shrunk": vals[:, 2],
            },
            index=idx,
        ).reset_index()
        df = df.merge(rgb_df, on=["episode_index", "frame_index"], how="left")
    else:
        df["gripper_duck_mean_r_shrunk"] = np.nan
        df["gripper_duck_mean_g_shrunk"] = np.nan
        df["gripper_duck_mean_b_shrunk"] = np.nan

    # Per-cluster blue-ratio thresholds (same measurement logic as v8).
    thr_br_per, thr_bg_per, measure_log = measure_cluster_thresholds(df)
    print(f"[stage1] thr_bg per-cluster: {thr_bg_per}")
    print(f"[stage1] decision: {measure_log['_decision']}")

    # Precompute blue_ok for every row (only meaningful where RGB present).
    # v11 stage1: tightened from `b/g >= 1.25` to `(b/g >= 1.45) AND (b-g >= 20)`.
    # The single-ratio rule misclassified dark near-grayscale regions with
    # slight blue cast as duck. The tight labeled set (12 accept, 8 reject)
    # shows clean gaps at B/G=[1.40, 1.53] and B-G=[16.8, 23.2]; requiring
    # BOTH thresholds puts decisions firmly inside those gaps with safety
    # margin. Cost: ~19% of prior-accepted rows become rejected.
    r = df["gripper_duck_mean_r_shrunk"].to_numpy()
    g = df["gripper_duck_mean_g_shrunk"].to_numpy()
    b = df["gripper_duck_mean_b_shrunk"].to_numpy()
    have_rgb = ~np.isnan(r) & ~np.isnan(g) & ~np.isnan(b)

    # Single-threshold blue-gate. Labeled set (11 accept / 8 reject) has a
    # clean gap at B/G ∈ [1.40, 1.53]; 1.45 is the midpoint. Softer than the
    # earlier AND rule that was dropping real duck frames.
    BLUE_GATE_BG_MIN = 1.45
    with np.errstate(invalid="ignore"):
        blue_ok = have_rgb & (b >= BLUE_GATE_BG_MIN * g)

    # ------------------------------------------------------------------
    # Per (cam, obj) filter loop — short-circuit first fail.
    # ------------------------------------------------------------------
    phase_arr = df["phase"].to_numpy()

    for cam in CAMERAS:
        for obj in OBJECTS:
            conf = df[f"{cam}_{obj}_conf"].to_numpy(dtype=np.float32)
            x1 = df[f"{cam}_{obj}_x1"].to_numpy(dtype=np.float32)
            y1 = df[f"{cam}_{obj}_y1"].to_numpy(dtype=np.float32)
            x2 = df[f"{cam}_{obj}_x2"].to_numpy(dtype=np.float32)
            y2 = df[f"{cam}_{obj}_y2"].to_numpy(dtype=np.float32)
            cx = df[f"{cam}_{obj}_cx"].to_numpy(dtype=np.float32)
            cy = df[f"{cam}_{obj}_cy"].to_numpy(dtype=np.float32)

            reason = np.full(len(df), R_ACCEPTED, dtype=object)

            # --- Step 1: no raw detection ---
            raw_ok = _raw_detection_mask(conf, x1, y1, x2, y2)
            reason[~raw_ok] = R_NO_DETECTION

            # --- Step 2: gripper_duck blue-gate (Filter A, stage-invariant) ---
            # v11: runs in ALL phases (not only grasp/transport). The jaw is
            # predicted as 'duck' in reach/release too, and color is the only
            # reliable signal to distinguish jaw (dark/black) from real duck.
            if cam == "gripper" and obj == "duck":
                active = (reason == R_ACCEPTED)
                # No RGB sample → can't verify blue → fail (conservative).
                fail_blue = active & (~blue_ok)
                reason[fail_blue] = R_PHASE_NO_BLUE

            # --- Step 3: gripper_cup jaw-region (Filter B) ---
            if cam == "gripper" and obj == "cup":
                active = (reason == R_ACCEPTED)
                w = x2 - x1
                h = y2 - y1
                area_norm = w * h  # bboxes are normalized → area ∈ [0, 1]
                with np.errstate(invalid="ignore"):
                    in_jaw = (
                        (cx >= CUP_JAW_CX_MIN)
                        & (cy >= CUP_JAW_CY_MIN)
                        & (area_norm <= CUP_JAW_AREA_MAX)
                    )
                reason[active & in_jaw] = R_CUP_JAW

            # --- Step 4: top_duck arm-lock (Filter v9) ---
            if cam == "top" and obj == "duck":
                active = (reason == R_ACCEPTED)
                armlock = np.isfinite(y2) & (y2 > Y2_ARMLOCK_THRESHOLD)
                reason[active & armlock] = R_TOP_DUCK_ARMLOCK

            # --- Step 4a: top_duck fixed-blob artifact ---
            # DINO repeatedly fires on a ~pixel-perfect fixed spot in the top
            # cam view. Empirical cluster at cx~0.365, cy~0.283, area~0.004.
            if cam == "top" and obj == "duck":
                active = (reason == R_ACCEPTED)
                cx_arr = df[f"{cam}_{obj}_cx"].to_numpy(dtype=np.float32)
                cy_arr = df[f"{cam}_{obj}_cy"].to_numpy(dtype=np.float32)
                area = (x2 - x1) * (y2 - y1)
                fixed_blob = (
                    np.isfinite(cx_arr) & np.isfinite(cy_arr) & np.isfinite(area)
                    & (cx_arr >= TOP_DUCK_FIXED_BLOB_CX_MIN)
                    & (cx_arr <= TOP_DUCK_FIXED_BLOB_CX_MAX)
                    & (cy_arr >= TOP_DUCK_FIXED_BLOB_CY_MIN)
                    & (cy_arr <= TOP_DUCK_FIXED_BLOB_CY_MAX)
                    & (area <= TOP_DUCK_FIXED_BLOB_AREA_MAX)
                )
                reason[active & fixed_blob] = R_TOP_DUCK_FIXED_BLOB

            # --- Step 4b: left_duck top-strip artifact ---
            # DINO repeatedly predicts a tiny duck in the top ~10% of the
            # left cam's frame (ghost detection on the wall/ceiling area).
            if cam == "left" and obj == "duck":
                active = (reason == R_ACCEPTED)
                top_strip = np.isfinite(y2) & (y2 < LEFT_DUCK_TOP_STRIP_Y2_MAX)
                reason[active & top_strip] = R_LEFT_DUCK_TOP_STRIP

            # --- Step 4b': left_duck fixed-blob artifact ---
            # Tight pixel-cluster ghost at cx≈0.28, cy≈0.74 on left cam.
            if cam == "left" and obj == "duck":
                active = (reason == R_ACCEPTED)
                cx_arr = df[f"{cam}_{obj}_cx"].to_numpy(dtype=np.float32)
                cy_arr = df[f"{cam}_{obj}_cy"].to_numpy(dtype=np.float32)
                area = (x2 - x1) * (y2 - y1)
                fixed_blob = (
                    np.isfinite(cx_arr) & np.isfinite(cy_arr) & np.isfinite(area)
                    & (cx_arr >= LEFT_DUCK_FIXED_BLOB_CX_MIN)
                    & (cx_arr <= LEFT_DUCK_FIXED_BLOB_CX_MAX)
                    & (cy_arr >= LEFT_DUCK_FIXED_BLOB_CY_MIN)
                    & (cy_arr <= LEFT_DUCK_FIXED_BLOB_CY_MAX)
                    & (area <= LEFT_DUCK_FIXED_BLOB_AREA_MAX)
                )
                reason[active & fixed_blob] = R_LEFT_DUCK_FIXED_BLOB

            # --- Step 4c: right_duck arm-base artifact ---
            # When duck isn't visible, DINO predicts a bbox hugging the
            # right cam's left edge — that's the arm base intruding into the
            # frame from the robot side. Calibrated from ep 82 fr 492.
            if cam == "right" and obj == "duck":
                active = (reason == R_ACCEPTED)
                arm_base = np.isfinite(x1) & (x1 < RIGHT_DUCK_ARM_BASE_X1_MAX)
                reason[active & arm_base] = R_RIGHT_DUCK_ARM_BASE

            # --- Step 5: conf floor (Filter E) ---
            if cam == "gripper" and obj == "duck":
                thresh = np.where(
                    phase_arr == "release",
                    CONF_THRESH_DEFAULT,
                    CONF_THRESH_GRIPPER_DUCK,
                ).astype(np.float32)
            else:
                thresh = np.full(len(df), CONF_THRESH_DEFAULT, dtype=np.float32)
            active = (reason == R_ACCEPTED)
            lowconf = active & (conf < thresh)
            reason[lowconf] = R_LOWCONF

            trust = (reason == R_ACCEPTED).astype(np.int8)
            df[f"{cam}_{obj}_trust"] = trust
            df[f"{cam}_{obj}_reason"] = pd.Categorical(
                reason, categories=REASON_CATEGORIES
            )

    # ------------------------------------------------------------------
    # Post-loop: cup-inside-duck (pregrasp/grasp only)
    # ------------------------------------------------------------------
    # During pregrasp/grasp, DINO sometimes emits a single bbox that it
    # classifies as BOTH duck and cup (often with identical confidence).
    # When cup bbox ≥80% inside duck bbox in this phase, reject the cup.
    # Stage 2 interpolation will fill the cup back in from neighbors.
    in_pg_or_grasp = np.isin(df["phase"].to_numpy(), ["pregrasp", "grasp"])
    for cam in CAMERAS:
        dx1 = df[f"{cam}_duck_x1"].to_numpy(dtype=np.float32)
        dy1 = df[f"{cam}_duck_y1"].to_numpy(dtype=np.float32)
        dx2 = df[f"{cam}_duck_x2"].to_numpy(dtype=np.float32)
        dy2 = df[f"{cam}_duck_y2"].to_numpy(dtype=np.float32)
        cx1 = df[f"{cam}_cup_x1"].to_numpy(dtype=np.float32)
        cy1 = df[f"{cam}_cup_y1"].to_numpy(dtype=np.float32)
        cx2 = df[f"{cam}_cup_x2"].to_numpy(dtype=np.float32)
        cy2 = df[f"{cam}_cup_y2"].to_numpy(dtype=np.float32)

        ix1 = np.maximum(dx1, cx1); iy1 = np.maximum(dy1, cy1)
        ix2 = np.minimum(dx2, cx2); iy2 = np.minimum(dy2, cy2)
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        cup_area = (cx2 - cx1) * (cy2 - cy1)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(cup_area > 0, inter / cup_area, 0.0)

        duck_trust = df[f"{cam}_duck_trust"].to_numpy().astype(bool)
        cup_trust = df[f"{cam}_cup_trust"].to_numpy().astype(bool)
        hit = duck_trust & cup_trust & in_pg_or_grasp & np.isfinite(frac) & (frac >= 0.80)

        # Reject cup
        new_cup_trust = np.where(hit, np.int8(0), df[f"{cam}_cup_trust"].to_numpy()).astype(np.int8)
        df[f"{cam}_cup_trust"] = new_cup_trust
        reason_arr = df[f"{cam}_cup_reason"].astype(object).to_numpy().copy()
        reason_arr[hit] = R_CUP_INSIDE_DUCK
        df[f"{cam}_cup_reason"] = pd.Categorical(reason_arr, categories=REASON_CATEGORIES)

    # ------------------------------------------------------------------
    # Post-loop: left/right duck, release phase — reject unless near cup
    # ------------------------------------------------------------------
    # Runs after the main loop because it needs both duck and cup trust
    # on the same camera. Only downgrades (trust=1 → 0), never promotes.
    phase_np = df["phase"].to_numpy()
    in_release = (phase_np == "release")
    for cam in ("left", "right"):
        duck_cx = df[f"{cam}_duck_cx"].to_numpy(dtype=np.float32)
        duck_cy = df[f"{cam}_duck_cy"].to_numpy(dtype=np.float32)
        cup_cx = df[f"{cam}_cup_cx"].to_numpy(dtype=np.float32)
        cup_cy = df[f"{cam}_cup_cy"].to_numpy(dtype=np.float32)
        cup_trust = df[f"{cam}_cup_trust"].to_numpy().astype(bool)
        duck_trust = df[f"{cam}_duck_trust"].to_numpy().astype(bool)

        with np.errstate(invalid="ignore"):
            dist = np.sqrt((duck_cx - cup_cx) ** 2 + (duck_cy - cup_cy) ** 2)

        far_or_no_cup = (~cup_trust) | (~np.isfinite(dist)) | (dist > RELEASE_DUCK_CUP_MAX_DIST)
        newly_rejected = duck_trust & in_release & far_or_no_cup

        new_trust = np.where(newly_rejected, np.int8(0), df[f"{cam}_duck_trust"].to_numpy()).astype(np.int8)
        df[f"{cam}_duck_trust"] = new_trust

        reason_arr = df[f"{cam}_duck_reason"].astype(object).to_numpy().copy()
        reason_arr[newly_rejected] = R_SIDE_DUCK_RELEASE
        df[f"{cam}_duck_reason"] = pd.Categorical(
            reason_arr, categories=REASON_CATEGORIES
        )

    # ------------------------------------------------------------------
    # Post-loop: left/right duck cy-shift (option A — cy only, bbox stays)
    # ------------------------------------------------------------------
    # Side cams often produce tall DINO bboxes (bill-up duck). The training
    # target cy should anchor on a square region at the bottom of the bbox
    # so the model learns the duck's base position. cy := y2 - w/2 when
    # h > w. Bbox coordinates are left untouched — this is a label-only
    # correction. Trust is unaffected.
    for cam in ("left", "right"):
        x1 = df[f"{cam}_duck_x1"].to_numpy(dtype=np.float32)
        x2 = df[f"{cam}_duck_x2"].to_numpy(dtype=np.float32)
        y1 = df[f"{cam}_duck_y1"].to_numpy(dtype=np.float32)
        y2 = df[f"{cam}_duck_y2"].to_numpy(dtype=np.float32)
        w = x2 - x1
        h = y2 - y1
        should_square = np.isfinite(w) & np.isfinite(h) & (h > w)
        new_cy = y2 - w / 2.0
        cy_cur = df[f"{cam}_duck_cy"].to_numpy(dtype=np.float32)
        df[f"{cam}_duck_cy"] = np.where(should_square, new_cy, cy_cur).astype(np.float32)

    return df


# ------------------------------------------------------------------
# Manual drop rules (for no_objects dataset etc.)
# ------------------------------------------------------------------

def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _apply_drop_rules(
    df: pd.DataFrame,
    drop_all_eps: list[int],
    drop_duck_eps: list[int],
    drop_cup_eps: list[int],
) -> pd.DataFrame:
    """Force trust=0 + NaN bbox/cx/cy for specified (episode, object) combos.

    Setting bbox to NaN ensures stage-2 interpolation cannot treat these rows
    as anchors and cannot fill values in from them.
    """
    if not (drop_all_eps or drop_duck_eps or drop_cup_eps):
        return df

    ep_arr = df["episode_index"].to_numpy()
    all_set = set(drop_all_eps)
    duck_set = set(drop_duck_eps) | all_set
    cup_set = set(drop_cup_eps) | all_set

    for cam in CAMERAS:
        for obj in OBJECTS:
            target_set = duck_set if obj == "duck" else cup_set
            if not target_set:
                continue
            mask = np.isin(ep_arr, list(target_set))
            if not mask.any():
                continue
            n = int(mask.sum())
            # Force trust=0
            trust = df[f"{cam}_{obj}_trust"].to_numpy().copy()
            trust[mask] = 0
            df[f"{cam}_{obj}_trust"] = trust.astype(np.int8)
            # Force reason=no_detection
            reason_arr = df[f"{cam}_{obj}_reason"].astype(object).to_numpy().copy()
            reason_arr[mask] = R_NO_DETECTION
            df[f"{cam}_{obj}_reason"] = pd.Categorical(reason_arr, categories=REASON_CATEGORIES)
            # Null bbox + cx/cy so stage-2 can't use them as anchors
            for f in ("cx", "cy", "conf", "x1", "y1", "x2", "y2"):
                col = f"{cam}_{obj}_{f}"
                vals = df[col].to_numpy().copy().astype(np.float32)
                vals[mask] = np.float32("nan")
                df[col] = vals
            print(f"[drop] {cam}_{obj}: nulled {n} rows from {len(target_set)} episode(s)")
    return df


# ------------------------------------------------------------------
# Output + summary
# ------------------------------------------------------------------

def _slim_output(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["frame_index", "episode_index", "phase", "cam_mode"]
    # Expose shrunk-bbox RGB for gripper_duck so blue-gate decisions are
    # introspectable from the parquet alone (no video decode needed).
    cols += [
        "gripper_duck_mean_r_shrunk",
        "gripper_duck_mean_g_shrunk",
        "gripper_duck_mean_b_shrunk",
    ]
    for cam in CAMERAS:
        for obj in OBJECTS:
            cols += [
                f"{cam}_{obj}_cx", f"{cam}_{obj}_cy", f"{cam}_{obj}_conf",
                f"{cam}_{obj}_x1", f"{cam}_{obj}_y1",
                f"{cam}_{obj}_x2", f"{cam}_{obj}_y2",
                f"{cam}_{obj}_trust", f"{cam}_{obj}_reason",
            ]
    slim = df[cols].copy()
    slim["frame_index"] = slim["frame_index"].astype(np.int64)
    slim["episode_index"] = slim["episode_index"].astype(np.int64)
    for rgb_col in ("gripper_duck_mean_r_shrunk",
                    "gripper_duck_mean_g_shrunk",
                    "gripper_duck_mean_b_shrunk"):
        slim[rgb_col] = slim[rgb_col].astype(np.float32)
    for cam in CAMERAS:
        for obj in OBJECTS:
            for field in ("cx", "cy", "conf", "x1", "y1", "x2", "y2"):
                slim[f"{cam}_{obj}_{field}"] = slim[f"{cam}_{obj}_{field}"].astype(
                    np.float32
                )
            slim[f"{cam}_{obj}_trust"] = slim[f"{cam}_{obj}_trust"].astype(np.int8)
    return slim


def _print_summary(slim: pd.DataFrame) -> None:
    n = len(slim)
    print(f"\n[stage1] summary  ({n} total rows)")
    print(f"{'cam_obj':<14} {'trust=1':>9} {'trust=1%':>9}   reason breakdown")
    for cam in CAMERAS:
        for obj in OBJECTS:
            key = f"{cam}_{obj}"
            t = int(slim[f"{key}_trust"].sum())
            counts = slim[f"{key}_reason"].value_counts()
            breakdown = " ".join(
                f"{k}={int(v)}" for k, v in counts.items() if int(v) > 0
            )
            print(f"{key:<14} {t:>9d} {100*t/n:>8.1f}%   {breakdown}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 — initial DINO detection cleaning (v9+A+B+E, no interp)."
    )
    parser.add_argument("--dataset", required=True,
                        help="LeRobot dataset repo-id or local path")
    parser.add_argument("--output", required=True,
                        help="Output parquet path "
                             "(e.g. detection_labels_stage1.parquet)")
    parser.add_argument("--skip-phases", action="store_true",
                        help="Debug: fill phase=unknown → disables Filter A")
    parser.add_argument("--rgb-cache", default=None,
                        help="Path to RGB cache parquet. Default: sibling of "
                             "--output with suffix .rgb_cache.parquet")
    parser.add_argument("--refresh-rgb", action="store_true",
                        help="Force rebuild of RGB cache (re-decode video).")
    parser.add_argument("--drop-all-eps", default="",
                        help="Comma-separated episodes: drop ALL detections (null bbox, trust=0).")
    parser.add_argument("--drop-duck-eps", default="",
                        help="Comma-separated episodes: drop duck detections only.")
    parser.add_argument("--drop-cup-eps", default="",
                        help="Comma-separated episodes: drop cup detections only.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    ds_path = resolve_dataset_path(args.dataset)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Raw detections
    ts = time.perf_counter()
    det_df = pd.read_parquet(ds_path / "detection_results.parquet")
    print(
        f"[stage1] raw detection parquet: {len(det_df)} rows, "
        f"{det_df['episode_index'].nunique()} episodes"
    )
    timings["load"] = time.perf_counter() - ts

    info, episodes = load_dataset_meta(ds_path)
    fps = info["fps"]

    # 2. Phases
    ts = time.perf_counter()
    if args.skip_phases:
        print("[stage1] --skip-phases → phase=unknown everywhere (disables A)")
        phase_df = det_df[["frame_index", "episode_index"]].copy()
        phase_df["phase"] = "unknown"
    else:
        phase_df = compute_phases(ds_path, fps=fps)
    timings["phases"] = time.perf_counter() - ts

    # 3. Camera-mode clusters
    modes_df = load_cluster_modes()
    print(f"[stage1] cam clusters: {modes_df['cam_mode'].value_counts().to_dict()}")

    # 4. Shrunk-bbox RGB for gripper_duck (Filter A).
    # compute_gripper_duck_pixel_stats expects "_eff" cols. With no interp,
    # these are just the raw cols — alias them.
    rgb_cache_path = (
        Path(args.rgb_cache) if args.rgb_cache
        else out.with_suffix(".rgb_cache.parquet")
    )
    ts = time.perf_counter()
    duck_stats = None
    if not args.refresh_rgb:
        duck_stats = _load_rgb_cache(rgb_cache_path)
    if duck_stats is None:
        print(f"[stage1] RGB cache miss → decoding gripper video")
        det_for_rgb = det_df.copy()
        for f in ("x1", "y1", "x2", "y2"):
            det_for_rgb[f"gripper_duck_{f}_eff"] = det_for_rgb[f"gripper_duck_{f}"]
        duck_stats = compute_gripper_duck_pixel_stats(
            det_for_rgb, phase_df, ds_path, episodes, info,
        )
        _save_rgb_cache(rgb_cache_path, duck_stats)
    timings["pixel_sample"] = time.perf_counter() - ts

    # 5. Apply filters
    ts = time.perf_counter()
    clean = apply_stage1_filter(det_df, phase_df, modes_df, duck_stats)
    timings["filter"] = time.perf_counter() - ts

    # 5b. Apply manual drop rules (for no_objects dataset etc.)
    drop_all = _parse_int_list(args.drop_all_eps)
    drop_duck = _parse_int_list(args.drop_duck_eps)
    drop_cup = _parse_int_list(args.drop_cup_eps)
    clean = _apply_drop_rules(clean, drop_all, drop_duck, drop_cup)

    # 6. Slim + save
    ts = time.perf_counter()
    slim = _slim_output(clean)
    slim.to_parquet(out, index=False)
    print(f"[stage1] wrote {out}")
    timings["save"] = time.perf_counter() - ts

    # 7. Summary
    _print_summary(slim)

    # Timings
    elapsed = time.perf_counter() - t0
    print("")
    print("=" * 52)
    for k, v in timings.items():
        print(f"[timing] {k:<14} {v:>7.1f}s")
    print(f"[timing] {'TOTAL':<14} {elapsed:>7.1f}s")
    print("=" * 52)


if __name__ == "__main__":
    main()
