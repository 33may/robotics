"""Cosmos Transfer pipeline for augmenting Isaac Sim dataset.

Commands:
    # Step 1: Replay episode in Isaac Sim with depth+seg cameras, save frames
    python vbti/utils/cosmos_transfer.py capture \
        --dataset_file=./datasets/vbti_table_v1/vbti_dataset_v1.hdf5 \
        --episode=33

    # Step 2: Convert captured frames → mp4 videos + edge maps
    python vbti/utils/cosmos_transfer.py process --episode=33

    # Step 3: Generate Cosmos Transfer config JSON
    python vbti/utils/cosmos_transfer.py config \
        --episode=33 \
        --prompt="realistic kitchen, warm lighting" \
        --variant=kitchen \
        --vis_weight=0.7

    # Step 4: Run Cosmos Transfer inference
    python vbti/utils/cosmos_transfer.py transfer --episode=33 --variant=kitchen

    # Step 5: Reassemble augmented frames back into HDF5
    python vbti/utils/cosmos_transfer.py reassemble --episode=33 --variant=kitchen
"""

import json
from pathlib import Path

import cv2
import fire
import h5py
import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

BASE_DIR = Path("./datasets/vbti_table_v2_cosmos")
COSMOS_DIR = BASE_DIR / "cosmos"
DATASET_FILE = BASE_DIR / "raw.hdf5"
CAMERAS = ["side_cam", "table_cam", "wrist"]
MODALITIES = ["rgb", "depth", "depth_raw", "seg"]
FRAME_SIZE = (640, 480)  # W x H


def _ep_dir(episode: int) -> Path:
    return COSMOS_DIR / "captures" / f"episode_{episode:03d}"


def _processed_dir(episode: int) -> Path:
    return COSMOS_DIR / "processed" / f"episode_{episode:03d}"


def _read_frames(
    f: h5py.File, ep_name: str, camera: str, modality: str,
) -> np.ndarray:
    """Read frames for one camera+modality from an open HDF5 file.

    Returns:
        rgb:   (N, H, W, 3) uint8
        depth: (N, H, W) float32
        seg:   (N, H, W, 4) uint8 RGBA
    """
    key_suffix = {"rgb": "", "depth": "_depth", "seg": "_seg"}
    obs_key = f"data/{ep_name}/obs/{camera}{key_suffix[modality]}"
    frames = f[obs_key][:]
    if modality == "depth" and frames.ndim == 4 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    return frames


def extract(
    episode: int = 0,
    dataset_file: str = str(DATASET_FILE),
    depth_min: float = 0.01,
    depth_max: float = 2.0,
):
    """Extract all camera frames from an HDF5 episode and save to disk.

    Writes to: cosmos/captures/episode_000/<cam>/{rgb,depth,seg}/*.png
               cosmos/captures/episode_000/<cam>/depth_raw/*.npy

    This replaces the old 'capture' step — reads directly from HDF5
    instead of replaying in Isaac Sim.
    """
    with h5py.File(dataset_file, "r") as f:
        episodes = sorted(f["data"].keys())
        if episode >= len(episodes):
            raise ValueError(f"Episode {episode} not found (dataset has {len(episodes)})")
        ep_name = episodes[episode]

        ep_dir = _ep_dir(episode)
        print(f"\n=== Extract: episode {episode} ({ep_name}) → {ep_dir}")

        for cam in CAMERAS:
            # RGB — (N, H, W, 3) uint8 → PNG
            rgb = _read_frames(f, ep_name, cam, "rgb")
            out = ep_dir / cam / "rgb"
            out.mkdir(parents=True, exist_ok=True)
            for i in tqdm(range(len(rgb)), desc=f"  {cam}/rgb", unit="f"):
                cv2.imwrite(str(out / f"{i:05d}.png"), cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR))

            # Depth — (N, H, W) float32 → normalized PNG + raw NPY
            depth = _read_frames(f, ep_name, cam, "depth")
            out_png = ep_dir / cam / "depth"
            out_npy = ep_dir / cam / "depth_raw"
            out_png.mkdir(parents=True, exist_ok=True)
            out_npy.mkdir(parents=True, exist_ok=True)
            for i in tqdm(range(len(depth)), desc=f"  {cam}/depth", unit="f"):
                np.save(str(out_npy / f"{i:05d}.npy"), depth[i])
                clipped = np.clip(depth[i], depth_min, depth_max)
                norm = ((clipped - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                cv2.imwrite(str(out_png / f"{i:05d}.png"), norm)

            # Seg — (N, H, W, 4) uint8 RGBA → RGB PNG
            seg = _read_frames(f, ep_name, cam, "seg")
            out = ep_dir / cam / "seg"
            out.mkdir(parents=True, exist_ok=True)
            for i in tqdm(range(len(seg)), desc=f"  {cam}/seg", unit="f"):
                cv2.imwrite(str(out / f"{i:05d}.png"), cv2.cvtColor(seg[i, :, :, :3], cv2.COLOR_RGB2BGR))

    print(f"\nDone! → {ep_dir}")


# ──────────────────────────────────────────────
# Step 2: Process frames → mp4 + edges
# ──────────────────────────────────────────────

def process(episode: int = 0, fps: int = 30):
    """Encode extracted frames into mp4 videos and generate Canny edge maps.

    Reads from:  cosmos/captures/episode_000/<cam>/{rgb,depth}/*.png
    Writes to:   cosmos/processed/episode_000/<cam>/{rgb,depth,edge}.mp4
    """
    ep_dir = _ep_dir(episode)
    out_dir = _processed_dir(episode)

    for cam in CAMERAS:
        cam_capture = ep_dir / cam
        cam_out = out_dir / cam
        cam_out.mkdir(parents=True, exist_ok=True)

        # Encode rgb + depth + seg → mp4
        for modality in ["rgb", "depth", "seg"]:
            frames_dir = cam_capture / modality
            if not frames_dir.exists():
                print(f"  SKIP {cam}/{modality} — not found")
                continue

            frames = sorted(frames_dir.glob("*.png"))
            if not frames:
                print(f"  SKIP {cam}/{modality} — no frames")
                continue

            _encode_mp4(frames, cam_out / f"{modality}.mp4", fps)
            print(f"  {cam}/{modality}.mp4 — {len(frames)} frames")

        # Generate Canny edge maps from RGB → mp4
        rgb_dir = cam_capture / "rgb"
        if not rgb_dir.exists():
            continue

        edge_dir = cam_capture / "edge"
        edge_dir.mkdir(exist_ok=True)

        rgb_frames = sorted(rgb_dir.glob("*.png"))
        for frame_path in tqdm(rgb_frames, desc=f"  {cam}/edge", unit="f"):
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(img, threshold1=50, threshold2=150)
            cv2.imwrite(str(edge_dir / frame_path.name), edges)

        edge_frames = sorted(edge_dir.glob("*.png"))
        _encode_mp4(edge_frames, cam_out / "edge.mp4", fps)
        print(f"  {cam}/edge.mp4 — {len(edge_frames)} frames")

    print(f"\nProcessed → {out_dir}")


def _encode_mp4(frame_paths: list[Path], output_path: Path, fps: int):
    """Encode a list of PNG frames into an mp4 video."""
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for path in frame_paths:
        frame = cv2.imread(str(path))
        writer.write(frame)

    writer.release()


def _encode_mp4_from_arrays(
    frames: np.ndarray, output_path: Path, fps: int, desc: str = "",
):
    """Encode numpy array (N, H, W, 3) uint8 BGR directly to mp4."""
    n, h, w = frames.shape[:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for i in tqdm(range(n), desc=desc, unit="f", leave=False) if desc else range(n):
        writer.write(frames[i])
    writer.release()


# ──────────────────────────────────────────────
# Step 3: Generate Cosmos Transfer config
# ──────────────────────────────────────────────

def config(
    episode: int = 0,
    camera: str = "side_cam",
    prompt: str = "photorealistic scene, natural lighting",
    variant: str = "default",
    vis_weight: float = 0.7,
    depth_weight: float = 0.3,
    edge_weight: float = 0.3,
):
    """Generate a Cosmos Transfer config JSON.

    Writes to: cosmos/configs/episode_000_<camera>_<variant>.json
    """
    proc_dir = _processed_dir(episode)
    cam_dir = proc_dir / camera

    cfg = {
        "name": f"ep{episode:03d}_{camera}_{variant}",
        "prompt": prompt,
        "video_path": str(cam_dir / "rgb.mp4"),
        "guidance": 7,
        "vis": {"control_weight": vis_weight},
        "edge": {
            "control_path": str(cam_dir / "edge.mp4"),
            "control_weight": edge_weight,
        },
        "depth": {
            "control_path": str(cam_dir / "depth.mp4"),
            "control_weight": depth_weight,
        },
    }

    configs_dir = COSMOS_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"episode_{episode:03d}_{camera}_{variant}.json"

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Config → {config_path}")
    print(json.dumps(cfg, indent=2))
    return config_path


# ──────────────────────────────────────────────
# Step 4: Run Cosmos Transfer inference
# ──────────────────────────────────────────────

def transfer(episode: int = 33, variant: str = "default"):
    """Run Cosmos Transfer 2.5 on a prepared config.

    TODO: Wire up the actual Cosmos Transfer inference call.
    This depends on how you install it — either:
      a) cosmos-transfer2.5 CLI from the GitHub repo
      b) Python API from the cosmos package
      c) NVIDIA NGC container
    """
    raise NotImplementedError(
        "TODO: Wire up Cosmos Transfer 2.5 inference.\n"
        "Config files are in cosmos/configs/.\n"
        "Output should go to cosmos/output/episode_NNN/variant_NAME/\n"
    )


# ──────────────────────────────────────────────
# Step 5: Reassemble into HDF5
# ──────────────────────────────────────────────

def reassemble(
    episode: int = 33,
    variant: str = "default",
    dataset_file: str = str(DATASET_FILE),
):
    """Merge Cosmos Transfer output frames back into the HDF5 dataset.

    Reads Cosmos output video, decodes frames, and writes them as new
    camera observations in the dataset (or a new dataset file).

    TODO: Implement. Key decisions:
      - Write to a new HDF5 file (augmented.hdf5)?
      - Or append new episodes to the existing file?
      - Which cameras to replace (just side_cam? all three?)
    """
    raise NotImplementedError(
        "TODO: Implement reassembly.\n"
        "Read Cosmos output mp4 → decode frames → write to HDF5.\n"
        "Actions/states copied from original episode, images replaced.\n"
    )


# ──────────────────────────────────────────────
# Batch: prepare_cosmos — HDF5 → ready for Cosmos
# ──────────────────────────────────────────────

def prepare_cosmos(
    dataset_file: str = str(DATASET_FILE),
    output_dir: str = str(COSMOS_DIR / "cosmos_ready"),
    prompt: str = "photorealistic scene, natural lighting",
    cameras: str = "side_cam,table_cam,wrist",
    controls: str = "depth,edge,seg",
    guidance: int = 7,
    vis_weight: float = 0.7,
    depth_weight: float = 0.3,
    edge_weight: float = 0.3,
    depth_min: float = 0.01,
    depth_max: float = 2.0,
    fps: int = 30,
):
    """Convert an entire HDF5 dataset into Cosmos-ready videos + configs.

    Goes directly from HDF5 → mp4 (no intermediate PNGs).

    Output structure:
        output_dir/
          videos/episode_000/side_cam/{rgb,depth,edge}.mp4
          configs/episode_000_side_cam.json
          output/   (empty, for Cosmos results)

    Args:
        cameras: Comma-separated camera names.
        controls: Comma-separated control signals to generate.
                  Options: depth, edge, seg. RGB is always included.
    """
    out = Path(output_dir)
    if isinstance(cameras, (list, tuple)):
        cam_list = list(cameras)
    else:
        cam_list = [c.strip() for c in cameras.split(",")]
    if isinstance(controls, (list, tuple)):
        ctrl_list = list(controls)
    else:
        ctrl_list = [c.strip() for c in controls.split(",")]

    with h5py.File(dataset_file, "r") as f:
        episodes = sorted(f["data"].keys())
        num_eps = len(episodes)
        print(f"\n=== prepare_cosmos ===")
        print(f"Dataset: {dataset_file} ({num_eps} episodes)")
        print(f"Cameras: {cam_list}")
        print(f"Controls: {ctrl_list} (+ rgb always)")
        print(f"Output: {out}\n")

        for ep_idx, ep_name in tqdm(
            list(enumerate(episodes)), desc="Episodes", unit="ep"
        ):
            ep_data = f[f"data/{ep_name}"]
            if "obs" not in ep_data:
                tqdm.write(f"  SKIP episode {ep_idx} ({ep_name}) — no obs")
                continue

            for cam in tqdm(cam_list, desc=f"  ep {ep_idx:03d}", unit="cam", leave=False):
                cam_out = out / "videos" / f"episode_{ep_idx:03d}" / cam
                cam_out.mkdir(parents=True, exist_ok=True)
                tag = f"ep{ep_idx:03d}/{cam}"

                # --- RGB (always) ---
                rgb = _read_frames(f, ep_name, cam, "rgb")
                rgb_bgr = rgb[:, :, :, ::-1].copy()
                _encode_mp4_from_arrays(rgb_bgr, cam_out / "rgb.mp4", fps, desc=f"    {tag}/rgb")
                n_frames = len(rgb)

                # --- Depth ---
                if "depth" in ctrl_list:
                    depth = _read_frames(f, ep_name, cam, "depth")
                    clipped = np.clip(depth, depth_min, depth_max)
                    norm = ((clipped - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                    depth_bgr = np.stack([norm, norm, norm], axis=-1)
                    _encode_mp4_from_arrays(depth_bgr, cam_out / "depth.mp4", fps, desc=f"    {tag}/depth")

                # --- Edge (Canny from RGB) ---
                if "edge" in ctrl_list:
                    edge_frames = []
                    for i in tqdm(range(n_frames), desc=f"    {tag}/edge", unit="f", leave=False):
                        gray = cv2.cvtColor(rgb_bgr[i], cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
                        edge_frames.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
                    edge_arr = np.stack(edge_frames)
                    _encode_mp4_from_arrays(edge_arr, cam_out / "edge.mp4", fps)

                # --- Seg ---
                if "seg" in ctrl_list:
                    seg = _read_frames(f, ep_name, cam, "seg")
                    seg_bgr = seg[:, :, :, :3][:, :, :, ::-1].copy()
                    _encode_mp4_from_arrays(seg_bgr, cam_out / "seg.mp4", fps, desc=f"    {tag}/seg")

                # --- Config JSON ---
                cfg_dir = out / "configs"
                cfg_dir.mkdir(parents=True, exist_ok=True)

                cfg = {
                    "name": f"ep{ep_idx:03d}_{cam}",
                    "prompt": prompt,
                    "video_path": f"videos/episode_{ep_idx:03d}/{cam}/rgb.mp4",
                    "guidance": guidance,
                    "vis": {"control_weight": vis_weight},
                }
                if "edge" in ctrl_list:
                    cfg["edge"] = {
                        "control_path": f"videos/episode_{ep_idx:03d}/{cam}/edge.mp4",
                        "control_weight": edge_weight,
                    }
                if "depth" in ctrl_list:
                    cfg["depth"] = {
                        "control_path": f"videos/episode_{ep_idx:03d}/{cam}/depth.mp4",
                        "control_weight": depth_weight,
                    }

                cfg_path = cfg_dir / f"episode_{ep_idx:03d}_{cam}.json"
                with open(cfg_path, "w") as cf:
                    json.dump(cfg, cf, indent=2)

    # Create output dir for Cosmos results
    (out / "output").mkdir(parents=True, exist_ok=True)

    print(f"\n=== Done! {num_eps} episodes × {len(cam_list)} cameras ===")
    print(f"Ready at: {out}")
    print(f"  videos/  — input mp4s")
    print(f"  configs/ — Cosmos spec JSONs (relative paths)")
    print(f"  output/  — empty, for Cosmos results")


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    fire.Fire({
        "extract": extract,
        "process": process,
        "config": config,
        "transfer": transfer,
        "reassemble": reassemble,
        "prepare": prepare_cosmos,
    })
