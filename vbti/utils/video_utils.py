import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm import tqdm


def get_video_info(video_path: str) -> dict:
    """Read video metadata: fps, frame count, duration, resolution."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    info["frames_per_second"] = int(info["fps"])

    cap.release()
    return info


def score_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return score


def process_video(video_path, output_path, k_frames=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = []
    batch = []

    for frame_idx in tqdm(range(total_frames), desc="Scoring frames"):
        ret, frame = cap.read()
        if not ret:
            break

        score = score_frame(frame)
        batch.append((frame_idx, frame, score))

        # Once we've collected fps frames (= 1 second), pick top-k
        if len(batch) == fps:
            batch.sort(key=lambda x: x[2], reverse=True)
            selected_frames.extend(batch[:k_frames])
            batch = []

    # Handle leftover frames (last partial second)
    if batch:
        batch.sort(key=lambda x: x[2], reverse=True)
        selected_frames.extend(batch[:k_frames])

    cap.release()

    # Sort back into chronological order
    selected_frames.sort(key=lambda x: x[0])

    # Save selected frames to output folder
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame, score in selected_frames:
        filename = out_dir / f"frame_{idx:06d}_{score:.1f}.png"
        cv2.imwrite(str(filename), frame)

    print(f"Saved {len(selected_frames)} frames to {out_dir}")
    return selected_frames


def score_stats(image_folder):
    """Score all images in a folder and plot a histogram of sharpness scores."""
    folder = Path(image_folder)
    paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))

    scores = []
    for p in tqdm(paths, desc="Scoring images"):
        img = cv2.imread(str(p))
        scores.append((p.name, score_frame(img)))

    scores.sort(key=lambda x: x[1])

    names, vals = zip(*scores)
    vals = np.array(vals)

    print(f"Images: {len(vals)}")
    print(f"Min: {vals.min():.1f}  Max: {vals.max():.1f}  Mean: {vals.mean():.1f}  Std: {vals.std():.1f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(vals, bins=30, edgecolor="black")
    ax.axvline(vals.mean(), color="red", linestyle="--", label=f"mean={vals.mean():.1f}")
    ax.set_xlabel("Laplacian Variance (sharpness)")
    ax.set_ylabel("Count")
    ax.set_title(f"Sharpness distribution — {len(vals)} images")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return scores


video_path = "vbti/data/vbti_table/videos/IMG_0640.MOV"
output_path = "vbti/data/vbti_table/images/stable/"
k_frames = 2

info = get_video_info(video_path)

# print(info)

# selected_frames = process_video(video_path, output_path, k_frames)

# print(len(selected_frames))


score_stats("vbti/data/vbti_table/images/stable")