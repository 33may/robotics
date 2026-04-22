"""Augmented dataset viewer — plays episodes with detection overlays.

Usage:
    python -m vbti.logic.dataset.viewer eternalmay33/02_black_full_center_aug
    python -m vbti.logic.dataset.viewer eternalmay33/02_black_full_center_aug --episode 5

    # Overlay the filter output parquet (what actually gets used in training):
    python -m vbti.logic.dataset.viewer eternalmay33/01_02_03_merged_may-sim \\
        --parq /home/may33/.cache/vbti/detection_labels_final.parquet

Controls:
    SPACE     pause/resume
    RIGHT     step forward (paused)
    LEFT      step backward (paused)
    N / P     next / previous episode
    +/-       speed up / slow down
    T         (parq mode) toggle show-only-trusted
    R         (parq mode) toggle show rejected bboxes
    Q / ESC   quit

Parq mode colors:
    GREEN      trust=1, real detection
    YELLOW     trust=1, gripper_duck interpolated bbox
    RED        trust=0 with a geometric reject reason (armlock/jaw/no_blue/...)
    (nothing)  trust=0 with reason=no_detection (teacher saw nothing)
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vbti.logic.dataset import resolve_dataset_path

PHASE_NAMES = ["reach", "pregrasp", "grasp", "transport", "release"]
PHASE_COLORS_BGR = {
    "reach":    (0, 165, 255),
    "pregrasp": (60, 60, 255),
    "grasp":    (0, 200, 0),
    "transport":(255, 180, 0),
    "release":  (255, 0, 200),
}
CAMERA_GRID = ["top", "left", "right", "gripper"]  # 2x2 layout


def _parse_state_layout(state_names: list[str]) -> tuple[dict, slice | None]:
    """Build detection slices and phase slice from state names.

    Returns:
        det_slices: {"left": {"duck": (i, i+2), "cup": (j, j+2)}, ...}
        phase_slice: slice or None if no phase columns
    """
    name_to_idx = {n: i for i, n in enumerate(state_names)}

    det_slices: dict[str, dict[str, tuple[int, int]]] = {}
    for cam in ["left", "right", "top", "gripper"]:
        det_slices[cam] = {}
        for obj in ["duck", "cup"]:
            cx_name = f"{cam}_{obj}_cx"
            if cx_name in name_to_idx:
                idx = name_to_idx[cx_name]
                det_slices[cam][obj] = (idx, idx + 2)

    # Phase: look for phase_reach, phase_pregrasp, etc.
    phase_start = name_to_idx.get("phase_reach")
    phase_end = name_to_idx.get("phase_release")
    phase_sl = slice(phase_start, phase_end + 1) if phase_start is not None and phase_end is not None else None

    return det_slices, phase_sl


def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) float [0,1] -> (H,W,3) uint8 BGR."""
    rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_detection(frame: np.ndarray, cx: float, cy: float,
                    label: str, color: tuple, conf: float = -1):
    """Draw a detection circle + label on a frame."""
    h, w = frame.shape[:2]
    px, py = int(cx * w), int(cy * h)
    if px == 0 and py == 0:
        return
    r = max(8, min(30, int(conf * 60))) if conf > 0 else 12
    # filled circle with transparency
    overlay = frame.copy()
    cv2.circle(overlay, (px, py), r, color, -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.circle(frame, (px, py), r, color, 2)
    # label
    txt = f"{label} {conf:.2f}" if conf > 0 else label
    cv2.putText(frame, txt, (px + r + 4, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_phase_bar(frame: np.ndarray, phase_name: str, y: int = 30):
    """Draw phase name badge on top of frame."""
    color = PHASE_COLORS_BGR.get(phase_name, (200, 200, 200))
    cv2.putText(frame, phase_name.upper(), (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _build_info_bar(w: int, state: np.ndarray, ep_idx: int,
                    frame_idx: int, n_frames: int, speed: float,
                    phase_name: str = "") -> np.ndarray:
    """Build a bottom info bar showing state details."""
    bar_h = 60
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)

    phase = phase_name or "?"
    color = PHASE_COLORS_BGR.get(phase, (200, 200, 200))

    joints = state[:6]
    joint_str = " ".join(f"{v:6.1f}" for v in joints)

    line1 = f"ep {ep_idx}  frame {frame_idx}/{n_frames-1}  speed {speed:.1f}x"
    line2 = f"joints: {joint_str}"

    cv2.putText(bar, line1, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # phase badge on the right
    cv2.putText(bar, phase.upper(), (w - 160, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return bar


def load_conf_lookup(repo_id: str) -> dict[tuple[int, int], dict] | None:
    """Try to load detection confidence from original dataset."""
    # Strip _aug suffix to find original
    orig = repo_id.replace("_aug", "")
    try:
        ds_path = resolve_dataset_path(orig)
        det_path = ds_path / "detection_results.parquet"
        if not det_path.exists():
            return None
        df = pd.read_parquet(det_path)
        lookup = {}
        for _, row in df.iterrows():
            key = (int(row["episode_index"]), int(row["frame_index"]))
            lookup[key] = {
                f"{cam}_{obj}_conf": float(row[f"{cam}_{obj}_conf"])
                for cam in ["left", "right", "top"]
                for obj in ["duck", "cup"]
            }
        return lookup
    except Exception:
        return None


# ------------------------------------------------------------------
# Filter-parquet overlay support
# ------------------------------------------------------------------

# Short labels for the reason codes so they fit on screen next to the bbox.
_REASON_SHORT = {
    "accepted":                       "OK",
    "no_detection":                   "no_det",
    "low_confidence":                 "lowconf",
    "phase_gripper_duck_occluded":    "occl",
    "phase_gripper_duck_no_neighbor": "no_nbr",
    "phase_gripper_duck_no_blue":     "no_blue",
    "gripper_cup_jaw_region":         "jaw",
    "top_duck_armlock_y2":            "armlock",
    "side_duck_release":              "rel_far",
    "left_duck_top_strip":            "top_strip",
    "right_duck_arm_base":            "arm_base",
    "top_duck_fixed_blob":            "fx_blob",
    "interpolated":                   "INTERP",
}

# Per-object colors (BGR).
_OBJ_COLOR = {
    "duck": (200,  40, 200),   # purple
    "cup":  ( 40, 140, 255),   # orange
}

_PARQ_OBJ_COLS = ["cx", "cy", "conf", "x1", "y1", "x2", "y2", "trust", "reason"]


def load_parq_lookup(parq_path: str) -> dict[tuple[int, int], dict]:
    """Load filter-output parquet into {(ep, frame): row_dict}.

    Each row_dict has keys:
        "phase": str
        "gripper_duck_bbox_filled": bool
        "<cam>_<obj>_<field>": value  for each (cam, obj) and each of
            {cx, cy, conf, x1, y1, x2, y2, trust, reason}.
    """
    print(f"Loading parq: {parq_path}")
    df = pd.read_parquet(parq_path)
    print(f"  {len(df)} rows, {df['episode_index'].nunique()} episodes")

    # Build the minimal set of columns we need (keeps memory sane).
    keep = ["episode_index", "frame_index", "phase", "gripper_duck_bbox_filled"]
    for cam in ["left", "right", "top", "gripper"]:
        for obj in ["duck", "cup"]:
            for f in _PARQ_OBJ_COLS:
                keep.append(f"{cam}_{obj}_{f}")
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Cast reason categoricals to plain str for easy dict access.
    for cam in ["left", "right", "top", "gripper"]:
        for obj in ["duck", "cup"]:
            rc = f"{cam}_{obj}_reason"
            if rc in df.columns:
                df[rc] = df[rc].astype(str)

    lookup: dict[tuple[int, int], dict] = {}
    cols = df.columns.tolist()
    for vals in df.itertuples(index=False, name=None):
        d = dict(zip(cols, vals))
        key = (int(d["episode_index"]), int(d["frame_index"]))
        lookup[key] = d
    print(f"  lookup built: {len(lookup)} (ep, frame) keys")
    return lookup


def _parq_row_for(lookup, sample) -> dict | None:
    ep = int(sample["episode_index"].item())
    fr = int(sample["frame_index"].item())
    return lookup.get((ep, fr))


def _draw_parq_bbox(
    frame: np.ndarray,
    x1n: float, y1n: float, x2n: float, y2n: float,
    cx: float, cy: float,
    color: tuple,
    label: str,
    thickness: int = 2,
):
    """Draw bbox rectangle + cx/cy dot + text label. All coords are normalized [0,1]."""
    h, w = frame.shape[:2]
    ix1, iy1 = int(x1n * w), int(y1n * h)
    ix2, iy2 = int(x2n * w), int(y2n * h)
    # Clip to frame
    ix1 = max(0, min(w - 1, ix1))
    iy1 = max(0, min(h - 1, iy1))
    ix2 = max(0, min(w - 1, ix2))
    iy2 = max(0, min(h - 1, iy2))
    if ix2 > ix1 and iy2 > iy1:
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, thickness)
    # cx/cy dot
    if 0.0 < cx < 1.0 and 0.0 < cy < 1.0:
        px, py = int(cx * w), int(cy * h)
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.circle(frame, (px, py), 6, (0, 0, 0), 1)
    # label just above the top-left corner of bbox (or the dot if no bbox)
    tx = ix1 if ix2 > ix1 else int(cx * w)
    ty = max(12, iy1 - 4) if ix2 > ix1 else int(cy * h) - 8
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_parq_detections(
    frame: np.ndarray,
    cam: str,
    parq: dict,
    show_rejected: bool,
    show_only_trusted: bool,
):
    """Overlay parquet detections for one camera onto the frame."""
    gripper_filled = bool(parq.get("gripper_duck_bbox_filled", False))
    for obj in ["duck", "cup"]:
        trust = int(parq.get(f"{cam}_{obj}_trust", 0) or 0)
        reason = parq.get(f"{cam}_{obj}_reason", "no_detection") or "no_detection"
        conf = float(parq.get(f"{cam}_{obj}_conf", 0.0) or 0.0)
        cx = float(parq.get(f"{cam}_{obj}_cx", np.nan) or np.nan)
        cy = float(parq.get(f"{cam}_{obj}_cy", np.nan) or np.nan)
        x1 = float(parq.get(f"{cam}_{obj}_x1", np.nan) or np.nan)
        y1 = float(parq.get(f"{cam}_{obj}_y1", np.nan) or np.nan)
        x2 = float(parq.get(f"{cam}_{obj}_x2", np.nan) or np.nan)
        y2 = float(parq.get(f"{cam}_{obj}_y2", np.nan) or np.nan)

        # Per-object color (duck=purple, cup=orange). Trust state is conveyed
        # via line thickness (thick=accepted, thin=rejected) and label suffix.
        color = _OBJ_COLOR[obj]
        if trust == 1:
            interp = (reason == "interpolated"
                      or (cam == "gripper" and obj == "duck" and gripper_filled))
            thickness = 2
            tag = f"{obj[0].upper()} {conf:.2f}" + (" I" if interp else "")
        else:
            if show_only_trusted:
                continue
            if reason == "no_detection" or not show_rejected:
                continue
            thickness = 1
            short = _REASON_SHORT.get(reason, reason[:8])
            tag = f"{obj[0].upper()} REJ:{short}"

        # Skip if both bbox and dot are invalid.
        have_bbox = all(np.isfinite([x1, y1, x2, y2])) and (x2 > x1) and (y2 > y1)
        have_dot = np.isfinite(cx) and np.isfinite(cy) and (cx > 0 or cy > 0)
        if not have_bbox and not have_dot:
            continue

        if have_bbox:
            _draw_parq_bbox(frame, x1, y1, x2, y2, cx, cy, color, tag, thickness=thickness)
        else:
            # bbox missing — just draw the dot with label
            h, w = frame.shape[:2]
            px, py = int(cx * w), int(cy * h)
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.putText(frame, tag, (px + 8, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _load_episode_index(ds: LeRobotDataset) -> pd.DataFrame:
    """Load episode boundaries from meta parquets."""
    root = ds.meta.root
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    eps = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    return eps[["episode_index", "length", "dataset_from_index", "dataset_to_index"]]


def run_viewer(repo_id: str, start_episode: int = 0, parq_path: str | None = None):
    print(f"Loading {repo_id}...")
    ds = LeRobotDataset(repo_id)
    n_episodes = ds.meta.total_episodes
    print(f"  {n_episodes} episodes, {ds.meta.total_frames} frames")

    # Parse state layout from dataset metadata
    import json
    with open(ds.meta.root / "meta" / "info.json") as f:
        info = json.load(f)
    state_names = info["features"]["observation.state"]["names"]
    det_slices, phase_slice = _parse_state_layout(state_names)
    state_dim = len(state_names)
    print(f"  State: {state_dim}d, detections: {sum(len(v) for v in det_slices.values())} pairs, phase: {'yes' if phase_slice else 'no'}")

    print("Loading episode index...")
    ep_df = _load_episode_index(ds)

    parq_lookup: dict[tuple[int, int], dict] | None = None
    conf_lookup = None
    if parq_path:
        parq_lookup = load_parq_lookup(parq_path)
    else:
        print("Loading detection confidences...")
        conf_lookup = load_conf_lookup(repo_id)
        if conf_lookup:
            print(f"  {len(conf_lookup)} entries loaded")
        else:
            print("  not available (will show positions without confidence)")

    show_rejected = True      # toggle with R
    show_only_trusted = False # toggle with T

    ep_idx = start_episode
    paused = False
    speed = 1.0
    fps = ds.meta.fps

    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)

    while True:
        # Episode boundaries from metadata
        ep_row = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
        ep_start = int(ep_row["dataset_from_index"])
        n_frames = int(ep_row["length"])
        local_frame = 0

        print(f"\nEpisode {ep_idx} ({n_frames} frames)")

        while 0 <= local_frame < n_frames:
            global_idx = ep_start + local_frame
            sample = ds[global_idx]
            state = sample["observation.state"].numpy()

            # Get phase
            parq_row = _parq_row_for(parq_lookup, sample) if parq_lookup else None
            if parq_row is not None and parq_row.get("phase") is not None:
                phase_name = str(parq_row["phase"])
            elif phase_slice is not None:
                phase_oh = state[phase_slice]
                phase_idx_val = int(np.argmax(phase_oh))
                phase_name = PHASE_NAMES[phase_idx_val] if phase_oh.max() > 0 else "unknown"
            else:
                phase_name = ""

            # Get confidence values (legacy path — only used without --parq)
            conf = {}
            if conf_lookup:
                key = (int(sample["episode_index"].item()), int(sample["frame_index"].item()))
                conf = conf_lookup.get(key, {})

            # Build camera grid
            cam_frames = {}
            for cam in CAMERA_GRID:
                k = f"observation.images.{cam}"
                if k in sample:
                    cam_frames[cam] = _tensor_to_bgr(sample[k])
                else:
                    cam_frames[cam] = np.zeros((480, 640, 3), dtype=np.uint8)

            # Draw detections on each camera
            for cam in ["left", "right", "top", "gripper"]:
                if cam not in cam_frames:
                    continue
                frame = cam_frames[cam]

                if parq_row is not None:
                    # Parq mode: draw filter-aware overlays (bbox + trust color + reason).
                    _draw_parq_detections(
                        frame, cam, parq_row,
                        show_rejected=show_rejected,
                        show_only_trusted=show_only_trusted,
                    )
                else:
                    # Legacy path — state-based cx/cy dots.
                    if cam not in det_slices:
                        continue
                    for obj, color in [("duck", (255, 100, 0)), ("cup", (0, 0, 255))]:
                        if obj not in det_slices[cam]:
                            continue
                        s, _e = det_slices[cam][obj]
                        cx, cy = state[s], state[s + 1]
                        if cx == 0 and cy == 0:
                            continue
                        c = conf.get(f"{cam}_{obj}_conf", -1)
                        if 0 < c < 0.05:
                            continue
                        _draw_detection(frame, cx, cy, obj[0].upper(), color, c)

                # Phase badge per camera
                if phase_name:
                    _draw_phase_bar(frame, phase_name)

            # Add camera labels
            for cam, frame in cam_frames.items():
                cv2.putText(frame, cam, (frame.shape[1] - 80, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)

            # Assemble 2x2 grid
            top_row = np.hstack([cam_frames["top"], cam_frames["left"]])
            bot_row = np.hstack([cam_frames["right"], cam_frames["gripper"]])
            grid = np.vstack([top_row, bot_row])

            # Info bar
            info = _build_info_bar(grid.shape[1], state, ep_idx, local_frame, n_frames, speed, phase_name)
            display = np.vstack([grid, info])

            cv2.imshow("Dataset Viewer", display)

            delay = max(1, int(1000 / (fps * speed))) if not paused else 0
            key = cv2.waitKey(delay if not paused else 30) & 0xFF

            if key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return
            elif key == ord(" "):
                paused = not paused
            elif key == ord("n"):
                ep_idx = (ep_idx + 1) % n_episodes
                break
            elif key == ord("p"):
                ep_idx = (ep_idx - 1) % n_episodes
                break
            elif key == ord("+") or key == ord("="):
                speed = min(speed * 1.5, 10.0)
            elif key == ord("-"):
                speed = max(speed / 1.5, 0.1)
            elif key == ord("t") and parq_lookup is not None:
                show_only_trusted = not show_only_trusted
                print(f"  show_only_trusted = {show_only_trusted}")
            elif key == ord("r") and parq_lookup is not None:
                show_rejected = not show_rejected
                print(f"  show_rejected = {show_rejected}")
            elif key == 83 and paused:  # right arrow
                local_frame = min(local_frame + 1, n_frames - 1)
                continue
            elif key == 81 and paused:  # left arrow
                local_frame = max(local_frame - 1, 0)
                continue

            if not paused:
                local_frame += 1

        else:
            # Episode ended naturally — go to next
            ep_idx = (ep_idx + 1) % n_episodes

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Augmented dataset viewer")
    parser.add_argument("dataset", help="Augmented dataset repo_id")
    parser.add_argument("--episode", type=int, default=0, help="Start episode")
    parser.add_argument(
        "--parq", default=None,
        help="Path to filter-output parquet (e.g. detection_labels_final.parquet). "
             "When set, detection overlays come from the parquet (bboxes + trust + reason) "
             "instead of the dataset's state cx/cy.",
    )
    args = parser.parse_args()
    run_viewer(args.dataset, args.episode, parq_path=args.parq)


if __name__ == "__main__":
    main()
