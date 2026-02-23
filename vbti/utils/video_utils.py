import subprocess
import sys
from pathlib import Path

import cv2


def get_video_info(video_path: str) -> dict:
    """Read video metadata: fps, frame count, duration, resolution, filesize."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "path": str(video_path),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    info["filesize_mb"] = Path(video_path).stat().st_size / (1024 * 1024)

    cap.release()
    return info


def print_video_stats(video_path: str):
    """Pretty-print video statistics to terminal."""
    info = get_video_info(video_path)

    mins = int(info["duration_sec"] // 60)
    secs = info["duration_sec"] % 60

    print(f"Video:       {Path(info['path']).name}")
    print(f"Resolution:  {info['width']}x{info['height']}")
    print(f"FPS:         {info['fps']:.2f}")
    print(f"Frames:      {info['total_frames']}")
    print(f"Duration:    {mins}m {secs:.1f}s")
    print(f"File size:   {info['filesize_mb']:.1f} MB")


def calculate_frame_count(video_path: str, percentage: float) -> int:
    """Calculate absolute frame count from a percentage of total frames.

    Example: video has 1800 frames, percentage=0.7 -> returns 1260
    """
    if not 0 < percentage <= 1.0:
        raise ValueError(f"Percentage must be between 0 and 1, got {percentage}")

    info = get_video_info(video_path)
    count = int(info["total_frames"] * percentage)

    print(f"{info['total_frames']} total frames x {percentage:.0%} = {count} frames")
    return count


def fix_rotation(video_path: str, output_path: str = None) -> str:
    """Re-encode video with rotation metadata baked into pixels.

    Phone cameras store rotation as EXIF metadata, but OpenCV/PyAV
    ignore it and decode raw pixels. This re-encodes so the rotation
    is applied to the actual pixel data.

    Args:
        video_path: Input video (MOV, MP4, etc.)
        output_path: Output path. Defaults to <name>_fixed.mp4 next to input.

    Returns:
        Path to the fixed video.
    """
    video_path = str(Path(video_path).resolve())
    if output_path is None:
        p = Path(video_path)
        output_path = str(p.with_name(p.stem + "_fixed.mp4"))

    cmd = [
        "ffmpeg", "-i", video_path,
        "-c:v", "libx264", "-crf", "18",
        "-an",  # drop audio — not needed for frame extraction
        output_path,
    ]

    print(f"Fixing rotation: {Path(video_path).name} → {Path(output_path).name}")
    subprocess.run(cmd, check=True)
    print(f"Saved to {output_path}")
    return output_path


def extract_frames(
    video_path: str,
    output_dir: str,
    mode: str = "count",
    value: float = 200,
):
    """Extract sharp frames from video using sharp-frame-extractor.

    Modes:
        count:      extract `value` frames total (default 200)
        every:      extract one sharp frame every `value` seconds
        percentage: extract `value` fraction of total frames (0.0-1.0)
    """
    video_path = str(Path(video_path).resolve())
    output_dir = str(Path(output_dir).resolve())

    # Auto-fix rotation (phones store rotation as metadata, OpenCV ignores it)
    video_path = fix_rotation(video_path)

    if mode == "percentage":
        value = calculate_frame_count(video_path, value)
        mode = "count"

    cmd = [
        sys.executable, "-m", "sharp_frame_extractor",
        video_path,
        "-o", output_dir,
    ]

    if mode == "count":
        cmd += ["--count", str(int(value))]
    elif mode == "every":
        cmd += ["--every", str(value)]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'count', 'every', or 'percentage'")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Report results
    out_path = Path(output_dir) / Path(video_path).stem
    frames = sorted(out_path.glob("*.png"))
    print(f"\nExtracted {len(frames)} frames to {out_path}")
    return len(frames)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "stats": print_video_stats,
        "count": calculate_frame_count,
        "fix_rotation": fix_rotation,
        "extract": extract_frames,
    })
