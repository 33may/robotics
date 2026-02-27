#!/usr/bin/env python3
"""HDF5 dataset utilities — inspect, view, and export episode data."""

import os
import re
import shutil
import subprocess
import tempfile
import h5py
import cv2
import numpy as np

# Non-camera obs keys to exclude when discovering cameras
_NON_CAM_KEYS = {"joint_pos", "joint_vel", "joint_pos_rel", "joint_vel_rel", "actions"}
_SENSOR_SUFFIX = {"rgb": "", "depth": "_depth", "seg": "_seg"}


def _natural_sort(names):
    """Sort demo_0, demo_1, ..., demo_10 in numeric order (not alphabetical)."""
    def key(s):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
    return sorted(names, key=key)


def _resolve_episode(f, episode):
    """Resolve episode index or name to the actual group name."""
    episodes = _natural_sort([k for k in f["data"].keys() if k.startswith("demo_")])
    if isinstance(episode, int):
        if episode < 0 or episode >= len(episodes):
            raise ValueError(f"Episode {episode} out of range [0, {len(episodes) - 1}]")
        return episodes[episode]
    if episode in f["data"]:
        return episode
    raise ValueError(f"Episode '{episode}' not found. Available: {episodes[:5]}...")


def _discover_cameras(obs_keys):
    """Return sorted list of camera names from obs keys."""
    return sorted(set(
        k.replace("_depth", "").replace("_seg", "")
        for k in obs_keys if k not in _NON_CAM_KEYS
    ))


def _sensor_key(camera, sensor):
    """Build the obs key suffix for a given sensor type."""
    if sensor not in _SENSOR_SUFFIX:
        raise ValueError(f"Unknown sensor '{sensor}'. Choose from: {list(_SENSOR_SUFFIX.keys())}")
    return f"{camera}{_SENSOR_SUFFIX[sensor]}"


def _prepare_frames(frames, sensor):
    """Convert raw HDF5 frames to BGR uint8 for display/encoding. Vectorized."""
    if sensor == "rgb":
        return frames[:, :, :, ::-1].copy()  # RGB→BGR via slice, no per-frame loop
    elif sensor == "depth":
        sq = frames.squeeze()  # (T, H, W, 1) → (T, H, W)
        clipped = np.clip(sq, 0.01, 2.0)
        norm = ((clipped - 0.01) / (2.0 - 0.01) * 255).astype(np.uint8)
        # applyColorMap is per-frame but fast (C++ internally)
        return np.stack([cv2.applyColorMap(f, cv2.COLORMAP_TURBO) for f in norm])
    elif sensor == "seg":
        if frames.shape[-1] == 4:
            return frames[:, :, :, :3][:, :, :, ::-1].copy()  # RGBA→BGR
        return frames
    return frames


def _load_grid(f, ep_name, cameras, sensors):
    """Load and arrange frames into a grid: cameras vertically, sensors horizontally."""
    rows = []

    for cam in cameras:
        cols = []
        for sensor in sensors:
            full_key = f"data/{ep_name}/obs/{_sensor_key(cam, sensor)}"
            if full_key not in f:
                print(f"  [SKIP] {cam}/{sensor} not found")
                continue
            cols.append(_prepare_frames(f[full_key][:], sensor))

        if cols:
            min_t = min(c.shape[0] for c in cols)
            rows.append(np.concatenate([c[:min_t] for c in cols], axis=2))

    if not rows:
        raise ValueError("No valid camera/sensor combinations found")

    min_t = min(r.shape[0] for r in rows)
    max_w = max(r.shape[2] for r in rows)
    padded = []
    for r in rows:
        r = r[:min_t]
        if r.shape[2] < max_w:
            pad = np.zeros((r.shape[0], r.shape[1], max_w - r.shape[2], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=2)
        padded.append(r)

    return np.concatenate(padded, axis=1)


# ─── HUD overlay ───────────────────────────────────────────────

_BAR_H = 32          # timeline bar height in pixels
_BAR_PAD = 8         # padding from edges
_BAR_COLOR = (80, 80, 80)
_PROGRESS_COLOR = (0, 180, 255)
_CURSOR_COLOR = (255, 255, 255)
_TEXT_COLOR = (220, 220, 220)


def _draw_hud(frame, idx, n_frames, fps, paused, label):
    """Draw timeline bar + frame info on bottom of frame. Returns new frame."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Semi-transparent bar background
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - _BAR_H), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    # Timeline track
    x0 = _BAR_PAD + 80
    x1 = w - _BAR_PAD - 90
    y_mid = h - _BAR_H // 2
    cv2.line(out, (x0, y_mid), (x1, y_mid), _BAR_COLOR, 2)

    # Progress fill
    if n_frames > 1:
        progress_x = x0 + int((x1 - x0) * idx / (n_frames - 1))
    else:
        progress_x = x0
    cv2.line(out, (x0, y_mid), (progress_x, y_mid), _PROGRESS_COLOR, 2)

    # Cursor dot
    cv2.circle(out, (progress_x, y_mid), 5, _CURSOR_COLOR, -1)

    # Frame counter (left)
    text = f"{idx}/{n_frames - 1}"
    cv2.putText(out, text, (_BAR_PAD, y_mid + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT_COLOR, 1)

    # Status (right)
    status = "PAUSED" if paused else f"{fps}fps"
    cv2.putText(out, status, (x1 + 8, y_mid + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 255) if paused else _TEXT_COLOR, 1)

    # Label (top-left, small)
    cv2.putText(out, label, (_BAR_PAD, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT_COLOR, 1)

    return out


def _timeline_click(event, x, y, flags, param):
    """Mouse callback for clicking on the timeline bar."""
    state = param
    h = state["h"]
    w = state["w"]

    # Only handle clicks in the bar area
    if y < h - _BAR_H:
        return

    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
        x0 = _BAR_PAD + 80
        x1 = w - _BAR_PAD - 90
        if x0 <= x <= x1:
            frac = (x - x0) / (x1 - x0)
            state["idx"] = int(frac * (state["n_frames"] - 1))
            state["paused"] = True


# ─── Player ────────────────────────────────────────────────────

def _play(frames, title, fps, save):
    """Video player with timeline, seeking, and HUD overlay."""
    n_frames = frames.shape[0]
    h, w = frames.shape[1], frames.shape[2]

    if save:
        # Write mp4v first, then transcode to H.264 (browser/Obsidian compatible)
        has_ffmpeg = shutil.which("ffmpeg") is not None
        tmp_path = save if not has_ffmpeg else tempfile.mktemp(suffix=".mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()

        if has_ffmpeg and tmp_path != save:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264",
                 "-crf", "18", "-preset", "fast", save],
                capture_output=True,
            )
            os.remove(tmp_path)

        print(f"  Saved {n_frames} frames → {save}")
        return

    win = "viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scale = min(1920 / w, 1080 / h, 1.0)
    cv2.resizeWindow(win, int(w * scale), int(h * scale))

    state = {"idx": 0, "n_frames": n_frames, "paused": True, "h": h, "w": w}
    cv2.setMouseCallback(win, _timeline_click, state)

    delay = max(1, int(1000 / fps))
    speed = 1  # frame step per tick

    print(f"\n  Controls: SPACE=play/pause  A/D=step  W/S=speed  R=restart  Q=quit")
    print(f"            Click timeline to seek. Arrow keys also work.\n")

    while True:
        display = _draw_hud(frames[state["idx"]], state["idx"], n_frames, fps * speed, state["paused"], title)
        cv2.imshow(win, display)

        key = cv2.waitKey(0 if state["paused"] else delay) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            state["paused"] = not state["paused"]
        elif key == ord("d") or key == 83:      # d / Right → next frame
            state["idx"] = min(state["idx"] + 1, n_frames - 1)
            state["paused"] = True
        elif key == ord("a") or key == 81:      # a / Left → prev frame
            state["idx"] = max(state["idx"] - 1, 0)
            state["paused"] = True
        elif key == ord("w") or key == 82:      # w / Up → faster
            speed = min(speed * 2, 16)
        elif key == ord("s") or key == 84:      # s / Down → slower
            speed = max(speed // 2, 1)
        elif key == ord("r"):
            state["idx"] = 0
        elif key == ord("["):                    # jump back 30 frames
            state["idx"] = max(state["idx"] - 30, 0)
        elif key == ord("]"):                    # jump forward 30 frames
            state["idx"] = min(state["idx"] + 30, n_frames - 1)
        else:
            if not state["paused"]:
                state["idx"] += speed
                if state["idx"] >= n_frames:
                    state["idx"] = n_frames - 1
                    state["paused"] = True       # pause at end, don't close

    cv2.destroyAllWindows()


# ─── Public CLI ────────────────────────────────────────────────

def info(dataset_path: str):
    """Print dataset structure: episodes, cameras, sensors, shapes.

    Args:
        dataset_path: Path to the HDF5 file.
    """
    with h5py.File(dataset_path, "r") as f:
        total = f["data"].attrs.get("total", "?")
        env_name = f["data"].attrs.get("env_name", "?")
        episodes = _natural_sort([k for k in f["data"].keys() if k.startswith("demo_")])

        print(f"\n  Dataset: {dataset_path}")
        print(f"  Task:    {env_name}")
        print(f"  Total frames: {total}")
        print(f"  Episodes: {len(episodes)}\n")

        for ep_name in episodes:
            ep = f[f"data/{ep_name}"]
            if "obs" not in ep:
                continue
            n = ep.attrs.get("num_samples", "?")
            success = ep.attrs.get("success", "?")

            print(f"  {ep_name}  ({n} frames, success={success})")
            for key in sorted(ep["obs"].keys()):
                ds = ep[f"obs/{key}"]
                print(f"    obs/{key:20s}  {str(ds.shape):20s}  {ds.dtype}")

            print()
            remaining = [e for e in episodes[1:] if "obs" in f[f"data/{e}"]]
            if remaining:
                lengths = [f["data"][e].attrs.get("num_samples", 0) for e in remaining]
                print(f"  ... {len(remaining)} more episodes ({min(lengths)}-{max(lengths)} frames each)")
            break


def view(dataset_path: str, episode: int = 0, camera: str = None,
         sensor: str = "rgb", fps: int = 30, save: str = None):
    """Play or save video from an HDF5 dataset episode.

    Args:
        dataset_path: Path to the HDF5 file.
        episode: Episode index (int) or name (str like 'demo_5').
        camera: Camera name, space-separated list, or 'all'. None lists available.
        sensor: Sensor type: 'rgb', 'depth', 'seg', space-separated list, or 'all'.
        fps: Playback framerate (default 30).
        save: If set, save to this mp4 path instead of playing in a window.

    Examples:
        view data.hdf5 0                               # list cameras
        view data.hdf5 0 cam_top                        # single camera rgb
        view data.hdf5 0 cam_top --sensor depth         # single camera depth
        view data.hdf5 0 all                            # all cameras, rgb
        view data.hdf5 0 all --sensor all               # full grid
        view data.hdf5 0 "cam_top cam_right"            # two cameras
        view data.hdf5 0 cam_top --sensor "rgb depth"   # one cam, two sensors
    """
    with h5py.File(dataset_path, "r") as f:
        ep_name = _resolve_episode(f, episode)
        ep = f[f"data/{ep_name}"]

        if "obs" not in ep:
            print(f"  [WARN] {ep_name} has no obs data (broken episode)")
            return

        obs_keys = list(ep["obs"].keys())
        all_cams = _discover_cameras(obs_keys)

        if camera is None:
            print(f"\n  {ep_name} — available cameras:")
            for c in all_cams:
                types = [s for s in _SENSOR_SUFFIX if _sensor_key(c, s) in obs_keys
                         or (s == "rgb" and c in obs_keys)]
                print(f"    {c:20s}  [{', '.join(types)}]")
            print(f"\n  Usage: view {dataset_path} {episode} <camera|all> [--sensor rgb|depth|seg|all]")
            return

        cameras = all_cams if camera == "all" else camera.split()
        sensors = list(_SENSOR_SUFFIX.keys()) if sensor == "all" else sensor.split()

        # Single camera + single sensor
        if len(cameras) == 1 and len(sensors) == 1:
            obs_key = _sensor_key(cameras[0], sensors[0])
            full_key = f"data/{ep_name}/obs/{obs_key}"
            if full_key not in f:
                print(f"  [ERROR] Key '{full_key}' not found.")
                print(f"  Available obs: {obs_keys}")
                return
            raw = f[full_key][:]
            print(f"\n  {ep_name}/{cameras[0]}/{sensors[0]}  shape={raw.shape}  dtype={raw.dtype}")
            print(f"  Loading...", end=" ", flush=True)
            frames = _prepare_frames(raw, sensors[0])
            print("done.")
            title = f"{ep_name} | {cameras[0]} | {sensors[0]} ({frames.shape[0]}f)"
            _play(frames, title, fps, save)
            return

        # Multi camera/sensor grid
        cam_str = "all" if camera == "all" else "+".join(cameras)
        sen_str = "all" if sensor == "all" else "+".join(sensors)
        print(f"\n  {ep_name} — grid: [{cam_str}] x [{sen_str}]")
        print(f"  Loading...", end=" ", flush=True)
        frames = _load_grid(f, ep_name, cameras, sensors)
        print(f"done. Grid: {frames.shape}")

    title = f"{ep_name} | {cam_str} x {sen_str} ({frames.shape[0]}f)"
    _play(frames, title, fps, save)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"info": info, "view": view})
