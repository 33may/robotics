"""Backfill a LeRobot v3.0 dataset with a gripper-depth video feature.

Loads a source dataset, runs Depth Anything v2 (metric, indoor) on the gripper
camera frames, colorizes (turbo), and writes a NEW dataset that contains all
original features + ``observation.images.gripper_depth`` as a 3-channel uint8
H.264 video stream.

Critical caveat from exp02a (preview on may-sim gripper frames):
  DA-V2-Indoor outputs ~0.4–2.0 m for scenes that the real D405 will see at
  ~0.05–0.5 m. There is a large absolute-scale mismatch between
  estimated and real depth.

Implications for the ``--mode`` flag:
  fixed-clip       — preserves absolute distance information from the model's
                     POV. The DA-V2 distribution is covered by [0.30, 2.00]m
                     (default below). Real D405 will need its own clip
                     (likely [0.05, 0.5]m). The two sources will then look
                     visually different unless rescaled to a shared range.
  per-frame-norm   — per-frame min-max → [0,1] → turbo. Throws away absolute
                     scale, but estimated and real depth share the same
                     visual treatment. Recommended for the v016 first pass.

Usage:
    conda run -n lerobot python -m vbti.logic.depth.add_gripper_depth \\
        --src eternalmay33/01_02_03_merged_may-sim_detection \\
        --dst eternalmay33/01_02_03_merged_may-sim_detection_gripper-depth \\
        --gripper-key observation.images.gripper \\
        --depth-key observation.images.gripper_depth \\
        --mode per-frame-norm
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vbti.logic.depth.colorize import colorize_fixed_clip, colorize_per_frame_norm

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
# Canonical clip for v016+ datasets (real D405 only). Tuned 2026-04-30 to put
# the entire LUT span inside the gripper workspace so the duck/cup gets
# maximum color resolution at grasp. Anything farther than 0.20 m saturates to
# the red end of turbo — explicit "background / not relevant" for the policy.
DEFAULT_CLIP_MIN_M = 0.05
DEFAULT_CLIP_MAX_M = 0.20


# ── frame coercion ────────────────────────────────────────────────────────

def _to_uint8_hwc(img) -> np.ndarray:
    """Coerce torch.Tensor (CHW float/uint8) or numpy array to uint8 HWC."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


# ── episode-boundary scan ─────────────────────────────────────────────────

def _scan_episode_boundaries(src) -> list[int]:
    """Return ``[start_frame_for_ep_0, ..., start_for_ep_N, total_frames]``.

    Reads only the episode_index column from parquet — fast (no video decode).
    """
    src._ensure_hf_dataset_loaded()
    ep_col = src.hf_dataset["episode_index"]
    boundaries = [0]
    prev = int(ep_col[0])
    for i in range(1, len(ep_col)):
        cur = int(ep_col[i])
        if cur != prev:
            boundaries.append(i)
            prev = cur
    boundaries.append(len(ep_col))
    return boundaries


# ── main ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--src", required=True, help="Source repo_id or local path")
    ap.add_argument("--src-root", default=None, help="Local root override for src")
    ap.add_argument("--dst", required=True, help="Destination repo_id")
    ap.add_argument("--dst-root", default=None, help="Local root override for dst")
    ap.add_argument("--gripper-key", default="observation.images.gripper")
    ap.add_argument("--depth-key", default="observation.images.gripper_depth")
    ap.add_argument(
        "--mode",
        choices=["fixed-clip", "per-frame-norm"],
        default="per-frame-norm",
        help="per-frame-norm sidesteps the DA-V2/D405 scale mismatch (recommended for first pass)",
    )
    ap.add_argument("--clip-min-m", type=float, default=DEFAULT_CLIP_MIN_M)
    ap.add_argument("--clip-max-m", type=float, default=DEFAULT_CLIP_MAX_M)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--episodes",
        default=None,
        help="Slice 'A:B' (e.g. '0:10') or csv '1,3,5'. Default: all.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Wipe dst-root if it exists")
    args = ap.parse_args(argv)

    # Imports here so --help is fast and error messages stay clean
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from transformers import pipeline as hf_pipeline

    # ── load source ──
    print(f"[load] {args.src}")
    src = LeRobotDataset(args.src, root=args.src_root)
    print(src)

    if args.gripper_key not in src.features:
        sys.exit(
            f"gripper_key '{args.gripper_key}' not in src features:\n  "
            + "\n  ".join(src.features.keys())
        )
    g_ft = src.features[args.gripper_key]
    if g_ft["dtype"] not in ("image", "video"):
        sys.exit(f"gripper_key dtype must be image/video; got {g_ft['dtype']}")

    # ── build destination feature dict (skip default-features auto-added by create) ──
    DEFAULT_KEYS = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
    new_features = {k: dict(v) for k, v in src.features.items() if k not in DEFAULT_KEYS}
    new_features[args.depth_key] = {
        "dtype": "video",
        "shape": tuple(g_ft["shape"]),
        "names": list(g_ft.get("names", ["height", "width", "channels"])),
    }

    # ── prep destination root ──
    dst_root = Path(args.dst_root).expanduser().resolve() if args.dst_root else None
    if dst_root is not None and dst_root.exists():
        if args.overwrite:
            print(f"[wipe] {dst_root}")
            shutil.rmtree(dst_root)
        else:
            sys.exit(f"dst-root exists: {dst_root}  (pass --overwrite to wipe)")

    print(f"[create] {args.dst}")
    dst = LeRobotDataset.create(
        repo_id=args.dst,
        fps=src.fps,
        robot_type=src.meta.robot_type,
        features=new_features,
        root=str(dst_root) if dst_root else None,
    )

    # ── episode boundaries (fast, parquet-only) ──
    print("[scan] episode boundaries…")
    bnd = _scan_episode_boundaries(src)
    total_eps = len(bnd) - 1
    print(f"[scan] {total_eps} episodes, {bnd[-1]} total frames")

    # ── episode selection ──
    if args.episodes:
        if ":" in args.episodes:
            a, b = args.episodes.split(":")
            ep_idxs = list(range(int(a or 0), int(b or total_eps)))
        else:
            ep_idxs = [int(x) for x in args.episodes.split(",")]
    else:
        ep_idxs = list(range(total_eps))
    print(f"[run] processing {len(ep_idxs)} episodes")

    # ── load depth model ──
    print(f"[depth] {args.model} on {args.device}")
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model=args.model,
        device=args.device,
    )

    # ── iterate episodes ──
    img_keys = [k for k, ft in new_features.items() if ft["dtype"] in ("image", "video")]
    state_keys = [k for k in new_features if k not in img_keys]

    for ep in tqdm(ep_idxs, desc="episodes"):
        start, end = bnd[ep], bnd[ep + 1]
        for i in range(start, end):
            item = src[i]

            gripper_rgb = _to_uint8_hwc(item[args.gripper_key])  # (H, W, 3) uint8
            with torch.no_grad():
                res = depth_pipe(Image.fromarray(gripper_rgb))

            # transformers depth pipeline → {'predicted_depth': tensor, 'depth': PIL}
            if "predicted_depth" in res:
                pd = res["predicted_depth"]
                depth_m = (
                    pd.detach().cpu().numpy().astype(np.float32)
                    if torch.is_tensor(pd)
                    else np.asarray(pd, dtype=np.float32)
                )
                if depth_m.ndim == 3:
                    depth_m = depth_m[0]
            else:
                depth_m = np.asarray(res["depth"], dtype=np.float32)

            # resize predicted depth to the gripper RGB shape if needed
            gh, gw = gripper_rgb.shape[:2]
            if depth_m.shape != (gh, gw):
                depth_m = cv2.resize(depth_m, (gw, gh), interpolation=cv2.INTER_LINEAR)

            if args.mode == "fixed-clip":
                depth_rgb = colorize_fixed_clip(depth_m, args.clip_min_m, args.clip_max_m)
            else:
                depth_rgb = colorize_per_frame_norm(depth_m)

            # build new frame dict
            frame: dict = {"task": item["task"]}
            for k in state_keys:
                if k == "task":
                    continue
                v = item[k]
                if torch.is_tensor(v):
                    v = v.detach().cpu().numpy()
                frame[k] = v
            for k in img_keys:
                if k == args.depth_key:
                    continue
                frame[k] = _to_uint8_hwc(item[k])
            frame[args.depth_key] = depth_rgb

            dst.add_frame(frame)

        dst.save_episode()

    dst.finalize()
    print(f"\n[done] {dst.root}")


if __name__ == "__main__":
    main()
