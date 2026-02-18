#!/usr/bin/env python3
"""CLI for COLMAP frame extraction — select top-k sharpest frames per second and export."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vbti.image_viewer.backend import probe_video, init_work_dir, extract_frames, export_accepted


def main():
    parser = argparse.ArgumentParser(description="Extract top-k sharpest frames per second for COLMAP")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Output directory (default: <video>_colmap)")
    parser.add_argument("-k", type=int, default=2, help="Frames per second to keep (default: 2)")
    parser.add_argument("--fmt", choices=["jpg", "png"], default="jpg", help="Output format (default: jpg)")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality 70-100 (default: 90)")
    parser.add_argument("--threshold", type=float, default=0, help="Minimum sharpness score (default: 0, keep all)")
    args = parser.parse_args()

    video = args.video
    output = args.output or str(Path(video).with_suffix("")) + "_colmap"

    # Probe
    info = probe_video(video)
    print(f"Video: {info['width']}x{info['height']}, {info['fps']:.1f}fps, {info['duration_sec']}s, {info['total_frames']} frames")
    print(f"Extracting top-{args.k} per second...")

    # Extract
    work_dir = init_work_dir(video)

    def progress(current, total, desc):
        if total > 0:
            pct = int(current / total * 100)
            print(f"\r  {desc}... {pct}%", end="", flush=True)

    frames = extract_frames(video, work_dir, args.k, progress)
    print()

    # Filter by preselected + threshold
    accepted = {
        fd["frame_idx"]: True
        for fd in frames
        if fd["preselected"] and fd["score"] >= args.threshold
    }

    if args.threshold > 0:
        total_preselected = sum(1 for fd in frames if fd["preselected"])
        print(f"Threshold {args.threshold}: {len(accepted)}/{total_preselected} frames pass")

    # Export
    path, count = export_accepted(frames, accepted, output, args.fmt, args.quality, progress)
    print(f"\nExported {count} frames to {path}")


if __name__ == "__main__":
    main()
