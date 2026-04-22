"""Stage 2 — interpolation of filtered detection labels.

Reads the stage1 parquet and fills trust=0 rows by linearly interpolating
(cx, cy, x1, y1, x2, y2) from the nearest trust=1 anchors in the same
episode, subject to per-reason policies.

Per-reason policy:

  no_detection, low_confidence      → always attempt interp
  top_duck_armlock_y2               → always attempt interp
                                       (gripper occluding duck — recovers)
  phase_gripper_duck_no_blue,
  gripper_cup_jaw_region            → attempt interp ONLY IF at least one
                                       of the K nearest frames is trust=1
                                       (isolated reject vs. persistent).
  side_duck_release                 → never interp (stays rejected).

Interpolation constraints:
  * same episode only (no cross-episode interp)
  * gap to each bookend anchor ≤ MAX_GAP frames
  * both a before-anchor and an after-anchor must exist

Output schema = stage1 schema; interpolated rows:
    trust = 1
    reason = 'interpolated'
    cx, cy, x1..y2 = linearly interpolated values

NO post-interp re-filtering.  (will be added in a later stage after
visual verification)

Usage:
    python -m vbti.logic.detection.distill_stage2 \\
        --input /home/may33/.cache/vbti/detection_labels_stage1.parquet \\
        --output /home/may33/.cache/vbti/detection_labels_stage2.parquet

    # Visually verify:
    python -m vbti.logic.dataset.viewer eternalmay33/01_02_03_merged_may-sim \\
        --parq /home/may33/.cache/vbti/detection_labels_stage2.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vbti.logic.detection.distill.distill_filter import (
    CAMERAS,
    OBJECTS,
    R_CUP_JAW,
    R_INTERPOLATED,
    R_LEFT_DUCK_TOP_STRIP,
    R_LOWCONF,
    R_NO_DETECTION,
    R_PHASE_NO_BLUE,
    R_RIGHT_DUCK_ARM_BASE,
    R_SIDE_DUCK_RELEASE,
    R_TOP_DUCK_ARMLOCK,
    R_TOP_DUCK_FIXED_BLOB,
    REASON_CATEGORIES,
)

# Reason → policy mapping.
# NEVER interp: the detection is WRONG at this spot (fixed artifact or semantic
# reject), so inserting anything there would re-introduce bad labels.
REASONS_NEVER_INTERP = {R_SIDE_DUCK_RELEASE}
REASONS_CHECK_NEIGHBORS = {R_PHASE_NO_BLUE, R_CUP_JAW}
# Everything else (no_det, lowconf, armlock, cup_inside_duck) is "always interp if possible".


def _interp_one_cam_obj(
    df: pd.DataFrame,
    cam: str,
    obj: str,
    max_gap: int,
    neighbor_k: int,
    ep_slices: list[tuple[int, int]],
) -> dict:
    """Interpolate one (cam, obj) in place on `df`. Returns stats dict.

    `ep_slices` is a list of (start, end) row-index pairs (half-open) —
    one per episode, in df's current row order.
    """
    trust_col = f"{cam}_{obj}_trust"
    reason_col = f"{cam}_{obj}_reason"
    field_cols = {f: f"{cam}_{obj}_{f}" for f in ("cx", "cy", "x1", "y1", "x2", "y2")}

    trust = df[trust_col].to_numpy().astype(np.int8).copy()
    reason = df[reason_col].astype(object).to_numpy().copy()
    fields = {f: df[c].to_numpy().astype(np.float32).copy() for f, c in field_cols.items()}
    frame_idx_arr = df["frame_index"].to_numpy()

    stats = {
        "interpolated": 0,
        "skipped_never": 0,             # side_duck_release
        "skipped_persistent": 0,         # K-neighbors all trust=0
        "skipped_no_bookend": 0,         # no before- or after-anchor
        "skipped_gap_too_large": 0,
        "skipped_nan_anchor": 0,
    }

    for start, end in ep_slices:
        n = end - start
        if n < 2:
            continue

        ep_frames = frame_idx_arr[start:end]
        ep_trust = trust[start:end]
        ep_reason = reason[start:end]

        anchors = np.where(ep_trust == 1)[0]
        if len(anchors) == 0:
            # Whole episode has no anchors for this (cam, obj) — nothing to interp.
            continue

        anchor_frames = ep_frames[anchors]

        for i in range(n):
            if ep_trust[i] == 1:
                continue
            r = str(ep_reason[i])
            if r in REASONS_NEVER_INTERP:
                stats["skipped_never"] += 1
                continue

            if r in REASONS_CHECK_NEIGHBORS:
                # Find K nearest frames (excluding self) by |frame_diff|.
                dists = np.abs(ep_frames - ep_frames[i]).astype(np.int64)
                dists[i] = np.iinfo(np.int64).max
                k = min(neighbor_k, n - 1)
                k_nearest_idx = np.argpartition(dists, k - 1)[:k]
                if np.all(ep_trust[k_nearest_idx] == 0):
                    stats["skipped_persistent"] += 1
                    continue

            # Find bookend anchors by frame distance.
            cur_frame = ep_frames[i]
            pos = np.searchsorted(anchor_frames, cur_frame)
            if pos == 0 or pos == len(anchor_frames):
                stats["skipped_no_bookend"] += 1
                continue
            before_local = anchors[pos - 1]
            after_local = anchors[pos]
            gap_before = cur_frame - ep_frames[before_local]
            gap_after = ep_frames[after_local] - cur_frame
            if gap_before > max_gap or gap_after > max_gap:
                stats["skipped_gap_too_large"] += 1
                continue

            # Verify anchors have finite bbox/cx/cy (they should, since trust=1).
            gi_before = start + before_local
            gi_after = start + after_local
            gi = start + i
            any_nan = False
            for f in ("cx", "cy", "x1", "y1", "x2", "y2"):
                vb = fields[f][gi_before]
                va = fields[f][gi_after]
                if not (np.isfinite(vb) and np.isfinite(va)):
                    any_nan = True
                    break
            if any_nan:
                stats["skipped_nan_anchor"] += 1
                continue

            total = gap_before + gap_after
            t = gap_before / total

            for f in ("cx", "cy", "x1", "y1", "x2", "y2"):
                vb = fields[f][gi_before]
                va = fields[f][gi_after]
                fields[f][gi] = vb + t * (va - vb)

            trust[gi] = 1
            reason[gi] = R_INTERPOLATED
            stats["interpolated"] += 1

    # Write back.
    df[trust_col] = trust
    df[reason_col] = pd.Categorical(reason, categories=REASON_CATEGORIES)
    for f, c in field_cols.items():
        df[c] = fields[f].astype(np.float32)

    return stats


def apply_stage2(df: pd.DataFrame, max_gap: int, neighbor_k: int) -> pd.DataFrame:
    """Apply stage-2 interpolation across all (cam, obj) pairs."""
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # Precompute episode slice boundaries.
    ep_arr = df["episode_index"].to_numpy()
    if len(ep_arr) == 0:
        return df
    change_idx = np.where(np.diff(ep_arr) != 0)[0] + 1
    boundaries = np.concatenate(([0], change_idx, [len(df)]))
    ep_slices: list[tuple[int, int]] = [
        (int(boundaries[i]), int(boundaries[i + 1]))
        for i in range(len(boundaries) - 1)
    ]
    print(f"[stage2] {len(ep_slices)} episodes")

    for cam in CAMERAS:
        for obj in OBJECTS:
            t0 = time.perf_counter()
            stats = _interp_one_cam_obj(df, cam, obj, max_gap, neighbor_k, ep_slices)
            dt = time.perf_counter() - t0
            key = f"{cam}_{obj}"
            print(
                f"[stage2] {key:<14} interp={stats['interpolated']:>6}  "
                f"skip_never={stats['skipped_never']}  "
                f"skip_persistent={stats['skipped_persistent']}  "
                f"skip_no_bookend={stats['skipped_no_bookend']}  "
                f"skip_gap>{max_gap}={stats['skipped_gap_too_large']}  "
                f"({dt:.1f}s)"
            )

    return df


def _print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n[stage2] summary  ({n} total rows)")
    print(f"{'cam_obj':<14} {'trust=1':>9} {'trust=1%':>9}   reason breakdown")
    for cam in CAMERAS:
        for obj in OBJECTS:
            key = f"{cam}_{obj}"
            t = int(df[f"{key}_trust"].sum())
            counts = df[f"{key}_reason"].value_counts()
            breakdown = " ".join(
                f"{k}={int(v)}" for k, v in counts.items() if int(v) > 0
            )
            print(f"{key:<14} {t:>9d} {100*t/n:>8.1f}%   {breakdown}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 — linear interpolation between trust=1 anchors."
    )
    parser.add_argument("--input", required=True,
                        help="Stage1 parquet path")
    parser.add_argument("--output", required=True,
                        help="Output parquet path "
                             "(e.g. detection_labels_stage2.parquet)")
    parser.add_argument("--max-gap", type=int, default=30,
                        help="Max frame gap to each bookend anchor (default 30)")
    parser.add_argument("--neighbor-k", type=int, default=3,
                        help="K nearest frames for the no_blue/jaw_region "
                             "neighbor check (default 3)")
    args = parser.parse_args()

    t_total = time.perf_counter()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[stage2] loading {inp}")
    df = pd.read_parquet(inp)
    print(f"[stage2] {len(df)} rows, {df['episode_index'].nunique()} episodes")
    print(f"[stage2] params: max_gap={args.max_gap}  neighbor_k={args.neighbor_k}")

    df = apply_stage2(df, max_gap=args.max_gap, neighbor_k=args.neighbor_k)
    df.to_parquet(out, index=False)
    print(f"[stage2] wrote {out}")

    _print_summary(df)

    print(f"\n[stage2] total: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
