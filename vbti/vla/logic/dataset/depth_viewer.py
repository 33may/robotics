"""Depth-aware dataset viewer.

Decodes uint16-packed-into-RGB depth frames (the v016+ canonical format) and
renders them as turbo-colorized depth panels. Supports two modes:

  ``raw``     — render a single depth PNG as turbo (no episode metadata needed,
                works on an in-progress recording).
  ``frame``   — render a single (episode, frame) as a 5-panel layout:
                top | left | right | gripper | gripper_depth (decoded).
                Requires a finalized dataset (parquet + videos written).
  ``replay``  — play through an episode with the same 5-panel layout.

Usage:
    # Single frame to PNG
    python -m vbti.logic.dataset.depth_viewer frame \\
        eternalmay33/16_gripper_depth --episode 0 --frame 50 \\
        --out /tmp/v016_frame.png

    # Single frame on screen
    python -m vbti.logic.dataset.depth_viewer frame \\
        eternalmay33/16_gripper_depth --episode 0 --frame 50

    # Replay a whole episode
    python -m vbti.logic.dataset.depth_viewer replay \\
        eternalmay33/16_gripper_depth --episode 0

Controls (replay):
    SPACE     pause/resume
    RIGHT     step forward
    LEFT      step backward
    N / P     next / previous episode
    +/-       faster / slower
    Q / ESC   quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vbti.logic.dataset import resolve_dataset_path
from vbti.logic.depth.colorize import colorize_fixed_clip, unpack_rgb_to_uint16

CAMERA_GRID = ["top", "left", "right", "gripper"]
DEFAULT_DEPTH_KEY = "observation.images.gripper_depth"
DEFAULT_DEPTH_SCALE_M = 1e-4  # D405 default
DEFAULT_CLIP_MIN_M = 0.05   # gripper-tip clearance (D405 near-clip ≈ 0.07; we go slightly tighter)
DEFAULT_CLIP_MAX_M = 0.20   # gripper workspace; >0.20m saturates to red — picked for max duck-grasp detail
PANEL_W, PANEL_H = 320, 240  # per-panel display size


def _tensor_to_rgb(x) -> np.ndarray:
    """LeRobot tensor or array → (H, W, 3) uint8 RGB."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def decode_depth_packed_rgb(
    rgb_packed: np.ndarray | torch.Tensor,
    depth_scale_m: float = DEFAULT_DEPTH_SCALE_M,
) -> np.ndarray:
    """uint8 (H, W, 3) packed RGB → float32 (H, W) depth in meters.

    Inverse of ``vbti.logic.depth.colorize.pack_uint16_to_rgb`` and of
    ``RealSenseCamera.read_latest_depth_packed``.
    """
    rgb = _tensor_to_rgb(rgb_packed)
    depth_u16 = unpack_rgb_to_uint16(rgb)
    return depth_u16.astype(np.float32) * depth_scale_m


def render_5panel(
    frames_rgb: dict[str, np.ndarray],
    depth_packed_rgb: np.ndarray,
    *,
    clip_min_m: float = DEFAULT_CLIP_MIN_M,
    clip_max_m: float = DEFAULT_CLIP_MAX_M,
    depth_scale_m: float = DEFAULT_DEPTH_SCALE_M,
    panel_w: int = PANEL_W,
    panel_h: int = PANEL_H,
    label_text: str | None = None,
) -> np.ndarray:
    """4 RGB cameras + 1 decoded turbo depth, side-by-side. Returns BGR uint8."""
    depth_m = decode_depth_packed_rgb(depth_packed_rgb, depth_scale_m=depth_scale_m)
    depth_rgb = colorize_fixed_clip(depth_m, clip_min_m, clip_max_m)  # (H, W, 3) RGB

    panels: list[np.ndarray] = []
    for name in CAMERA_GRID:
        img_rgb = frames_rgb.get(name)
        if img_rgb is None:
            img_rgb = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panels.append(img_rgb)
    panels.append(depth_rgb)
    titles = list(CAMERA_GRID) + ["gripper_depth (decoded)"]

    out_panels: list[np.ndarray] = []
    for img_rgb, title in zip(panels, titles):
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (panel_w, panel_h))
        cv2.rectangle(bgr, (0, 0), (panel_w - 1, 22), (0, 0, 0), -1)
        cv2.putText(bgr, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out_panels.append(bgr)

    grid = np.hstack(out_panels)
    if label_text:
        cv2.rectangle(grid, (0, panel_h), (grid.shape[1], panel_h + 22), (0, 0, 0), -1)
        cv2.putText(grid, label_text, (6, panel_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        grid = np.vstack([grid, np.zeros((22, grid.shape[1], 3), dtype=np.uint8)])
        # we accidentally added a row above — drop the extra row
        grid = grid[: panel_h + 22]
    return grid


# ── raw mode ───────────────────────────────────────────────────────────────

def cmd_raw(args: argparse.Namespace) -> None:
    """Read one packed-depth PNG straight from disk and render it.

    Works without a finalized LeRobotDataset — useful while recording is still
    in progress and parquet metadata hasn't been written yet.
    """
    png_path = Path(args.png_path).expanduser().resolve()
    if not png_path.is_file():
        sys.exit(f"PNG not found: {png_path}")
    rgb_packed = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if rgb_packed is None:
        sys.exit(f"failed to read {png_path}")
    if rgb_packed.ndim == 3 and rgb_packed.shape[2] == 3:
        rgb_packed = cv2.cvtColor(rgb_packed, cv2.COLOR_BGR2RGB)

    depth_m = decode_depth_packed_rgb(rgb_packed, depth_scale_m=args.depth_scale_m)
    valid = depth_m[depth_m > 0]
    if valid.size:
        p = np.percentile(valid, [5, 50, 95])
        stats = f"valid={valid.size/depth_m.size:.1%}  p5={p[0]:.3f}  p50={p[1]:.3f}  p95={p[2]:.3f} m"
    else:
        stats = "no valid depth pixels"

    decoded_rgb = colorize_fixed_clip(depth_m, args.clip_min_m, args.clip_max_m)
    raw_bgr = cv2.cvtColor(rgb_packed, cv2.COLOR_RGB2BGR)
    decoded_bgr = cv2.cvtColor(decoded_rgb, cv2.COLOR_RGB2BGR)

    h, w = raw_bgr.shape[:2]
    decoded_bgr = cv2.resize(decoded_bgr, (w, h))
    grid = np.hstack([raw_bgr, decoded_bgr])

    label = (
        f"{png_path.name}  clip=[{args.clip_min_m:.3f}, {args.clip_max_m:.3f}]m  "
        f"scale={args.depth_scale_m}  | {stats}"
    )
    bar = np.zeros((24, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    title_bar = np.zeros((22, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, "raw packed RGB", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(title_bar, "decoded turbo (depth)", (w + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    grid = np.vstack([title_bar, grid, bar])

    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), grid)
        print(f"[saved] {out}")
        print(f"[stats] {stats}")
    else:
        cv2.imshow("depth_viewer.raw", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── frame mode ─────────────────────────────────────────────────────────────

def _resolve_dataset(repo_id_or_path: str) -> LeRobotDataset:
    root = resolve_dataset_path(repo_id_or_path)
    return LeRobotDataset(repo_id_or_path, root=root if root else None)


def _episode_bounds(ds: LeRobotDataset) -> list[int]:
    ds._ensure_hf_dataset_loaded()
    ep_col = ds.hf_dataset["episode_index"]
    boundaries = [0]
    prev = int(ep_col[0])
    for i in range(1, len(ep_col)):
        cur = int(ep_col[i])
        if cur != prev:
            boundaries.append(i)
            prev = cur
    boundaries.append(len(ep_col))
    return boundaries


def cmd_frame(args: argparse.Namespace) -> None:
    ds = _resolve_dataset(args.dataset)
    bnd = _episode_bounds(ds)
    n_eps = len(bnd) - 1

    if args.episode < 0 or args.episode >= n_eps:
        sys.exit(f"--episode out of range: {args.episode} (have {n_eps} episodes)")
    ep_start, ep_end = bnd[args.episode], bnd[args.episode + 1]
    ep_len = ep_end - ep_start
    if args.frame < 0 or args.frame >= ep_len:
        sys.exit(f"--frame out of range: {args.frame} (episode {args.episode} has {ep_len} frames)")

    abs_idx = ep_start + args.frame
    item = ds[abs_idx]

    frames_rgb = {name: _tensor_to_rgb(item[f"observation.images.{name}"]) for name in CAMERA_GRID}
    depth_packed = item.get(args.depth_key)
    if depth_packed is None:
        sys.exit(f"depth key '{args.depth_key}' not in sample. Available: "
                 + ", ".join(k for k in item if k.startswith('observation.images.')))

    label = (
        f"{args.dataset}  ep={args.episode}  frame={args.frame}  "
        f"clip=[{args.clip_min_m:.3f}, {args.clip_max_m:.3f}]m  "
        f"scale={args.depth_scale_m}"
    )
    grid = render_5panel(
        frames_rgb,
        depth_packed,
        clip_min_m=args.clip_min_m,
        clip_max_m=args.clip_max_m,
        depth_scale_m=args.depth_scale_m,
        label_text=label,
    )

    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), grid)
        print(f"[saved] {out}")
    else:
        cv2.imshow("depth_viewer.frame", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── replay mode ────────────────────────────────────────────────────────────

def cmd_replay(args: argparse.Namespace) -> None:
    ds = _resolve_dataset(args.dataset)
    bnd = _episode_bounds(ds)
    n_eps = len(bnd) - 1

    ep = max(0, min(args.episode, n_eps - 1))
    paused = False
    fps = 30
    speed = 1.0
    frame_idx = 0

    while True:
        ep_start, ep_end = bnd[ep], bnd[ep + 1]
        ep_len = ep_end - ep_start
        frame_idx = max(0, min(frame_idx, ep_len - 1))

        abs_idx = ep_start + frame_idx
        item = ds[abs_idx]

        frames_rgb = {name: _tensor_to_rgb(item[f"observation.images.{name}"]) for name in CAMERA_GRID}
        depth_packed = item[args.depth_key]
        label = (
            f"{args.dataset}  ep={ep}/{n_eps - 1}  frame={frame_idx}/{ep_len - 1}  "
            f"speed={speed:.1f}x  {'PAUSED' if paused else 'PLAY'}  clip=[{args.clip_min_m:.3f},{args.clip_max_m:.3f}]m"
        )
        grid = render_5panel(
            frames_rgb,
            depth_packed,
            clip_min_m=args.clip_min_m,
            clip_max_m=args.clip_max_m,
            depth_scale_m=args.depth_scale_m,
            label_text=label,
        )
        cv2.imshow("depth_viewer.replay", grid)

        wait = 1 if paused else max(1, int(1000 / (fps * speed)))
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == 83 or key == ord("d"):  # right
            paused = True
            frame_idx += 1
        elif key == 81 or key == ord("a"):  # left
            paused = True
            frame_idx -= 1
        elif key in (ord("n"),):
            ep = (ep + 1) % n_eps
            frame_idx = 0
        elif key in (ord("p"),):
            ep = (ep - 1) % n_eps
            frame_idx = 0
        elif key == ord("+") or key == ord("="):
            speed = min(speed * 1.5, 8.0)
        elif key == ord("-") or key == ord("_"):
            speed = max(speed / 1.5, 0.1)
        else:
            if not paused:
                frame_idx += 1
                if frame_idx >= ep_len:
                    frame_idx = 0  # loop episode

    cv2.destroyAllWindows()


# ── main ──

def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("dataset", help="Repo ID or local path")
    common.add_argument("--depth-key", default=DEFAULT_DEPTH_KEY)
    common.add_argument("--depth-scale-m", type=float, default=DEFAULT_DEPTH_SCALE_M,
                        help="Hardware unit scale (D405 default 1e-4 m/unit)")
    common.add_argument("--clip-min-m", type=float, default=DEFAULT_CLIP_MIN_M)
    common.add_argument("--clip-max-m", type=float, default=DEFAULT_CLIP_MAX_M)

    praw = sub.add_parser("raw", help="Render a single depth PNG (no metadata needed)")
    praw.add_argument("png_path", help="Path to a packed-depth PNG file")
    praw.add_argument("--depth-scale-m", type=float, default=DEFAULT_DEPTH_SCALE_M)
    praw.add_argument("--clip-min-m", type=float, default=DEFAULT_CLIP_MIN_M)
    praw.add_argument("--clip-max-m", type=float, default=DEFAULT_CLIP_MAX_M)
    praw.add_argument("--out", default=None,
                      help="Save side-by-side PNG (raw packed | decoded turbo). Default: open a window")
    praw.set_defaults(fn=cmd_raw)

    pf = sub.add_parser("frame", parents=[common], help="Render single frame")
    pf.add_argument("--episode", type=int, default=0)
    pf.add_argument("--frame", type=int, default=0)
    pf.add_argument("--out", default=None,
                    help="Save PNG instead of opening a window")
    pf.set_defaults(fn=cmd_frame)

    pr = sub.add_parser("replay", parents=[common], help="Play through episode")
    pr.add_argument("--episode", type=int, default=0)
    pr.set_defaults(fn=cmd_replay)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
