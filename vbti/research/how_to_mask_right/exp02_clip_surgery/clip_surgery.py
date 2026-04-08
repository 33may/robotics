"""CLIP Surgery spatial attribution for SigLIP — compute + save only.

Computes per-patch cosine similarity between SigLIP patch tokens and text
embeddings, with CLIP Surgery redundancy correction. Saves raw data to .npz.
Plotting is in plot_surgery.py.

Architecture notes:
- SigLIP has NO CLS token and NO separate projection layers.
- Vision: 27 encoder layers -> post_layernorm -> head (attention pooling + MLP)
- Text: encoder -> final_layer_norm -> head (Linear 1152->1152)
- For patch-level analysis we use last_hidden_state (post layernorm, pre head).
  Text uses pooler_output (post head).
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import SiglipModel, SiglipProcessor
from PIL import Image
import torchvision.transforms.functional as TF

# ── Config ──────────────────────────────────────────────────────────────────

CACHE_DIR = Path("/home/may33/projects/ml_portfolio/robotics/vbti/research/how_to_mask_right/cached_frames")
OUT_DIR = Path("/home/may33/projects/ml_portfolio/robotics/vbti/research/how_to_mask_right/exp02_clip_surgery")
MODEL_ID = "google/siglip-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_PROMPTS = [
    "a rubber duck",
    "a red cup",
    "a robot gripper",
    "a wooden table",
    "Pick up the duck and place it in the cup",
]

CAMERAS = ["top", "left", "right", "gripper"]
N_SAMPLE_FRAMES = 5
GRID_SIZE = 27


# ── Helpers ─────────────────────────────────────────────────────────────────

_shard_cache = {}

def load_frame(frame_idx: int, manifest: dict) -> dict:
    if frame_idx in _shard_cache:
        return _shard_cache[frame_idx]
    for shard_i in range(manifest["n_shards"]):
        shard = torch.load(CACHE_DIR / f"shard_{shard_i:04d}.pt",
                           map_location="cpu", weights_only=True)
        if frame_idx in shard:
            _shard_cache[frame_idx] = shard[frame_idx]
            return shard[frame_idx]
    raise ValueError(f"Frame {frame_idx} not found")


def preprocess_image(img_tensor, processor):
    img_pil = TF.to_pil_image(img_tensor.clamp(0, 1))
    inputs = processor(images=img_pil, return_tensors="pt")
    return inputs["pixel_values"].to(DEVICE)


def clip_surgery(pixel_values, text_features, model):
    """Returns (surgery_map, raw_map) each [N_prompts, 27, 27]."""
    with torch.no_grad():
        vis_out = model.vision_model(pixel_values=pixel_values)
        patch_features = vis_out.last_hidden_state          # [1, 729, 1152]
        patch_features = F.normalize(patch_features, dim=-1)

    N = text_features.shape[0]

    # Raw cosine similarity
    raw_sim = torch.einsum("bpd,nd->bpn", patch_features, text_features)

    # CLIP Surgery: element-wise product, subtract mean across prompts
    feats = patch_features.unsqueeze(2) * text_features.unsqueeze(0).unsqueeze(0)
    redundant = feats.mean(dim=2, keepdim=True)
    surgery_sim = (feats - redundant).sum(dim=-1)

    raw_map = raw_sim[0].T.reshape(N, GRID_SIZE, GRID_SIZE).cpu().numpy()
    surgery_map = surgery_sim[0].T.reshape(N, GRID_SIZE, GRID_SIZE).cpu().numpy()
    return surgery_map, raw_map


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    with open(CACHE_DIR / "manifest.json") as f:
        manifest = json.load(f)

    frame_indices = manifest["frame_indices"]
    n_frames = len(frame_indices)
    sample_positions = np.linspace(0, n_frames - 1, N_SAMPLE_FRAMES, dtype=int)
    sampled = [frame_indices[i] for i in sample_positions]
    print(f"Sampled frames: {sampled}")

    # Load model
    print("Loading SigLIP...")
    model = SiglipModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = SiglipProcessor.from_pretrained(MODEL_ID)

    # Encode text
    text_inputs = processor(text=TEXT_PROMPTS, padding="max_length", return_tensors="pt")
    text_ids = text_inputs["input_ids"].to(DEVICE)
    text_mask = text_inputs.get("attention_mask")
    if text_mask is not None:
        text_mask = text_mask.to(DEVICE)

    with torch.no_grad():
        text_out = model.text_model(input_ids=text_ids, attention_mask=text_mask)
        text_features = F.normalize(text_out.pooler_output, dim=-1)

    print(f"Text features: {text_features.shape}")

    # Process frames
    npz_data = {}
    for fi, frame_idx in enumerate(sampled):
        print(f"  Frame {frame_idx} ({fi+1}/{N_SAMPLE_FRAMES})...")
        frame_data = load_frame(frame_idx, manifest)

        for cam in CAMERAS:
            img_tensor = frame_data[cam]
            pixel_values = preprocess_image(img_tensor, processor)
            surgery_map, raw_map = clip_surgery(pixel_values, text_features, model)

            npz_data[f"surgery_{frame_idx}_{cam}"] = surgery_map
            npz_data[f"raw_{frame_idx}_{cam}"] = raw_map

            # Save original image as numpy for plotting
            img_np = (img_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            npz_data[f"image_{frame_idx}_{cam}"] = img_np

    npz_data["frame_indices"] = np.array(sampled)
    npz_data["prompts"] = np.array(TEXT_PROMPTS)
    npz_data["cameras"] = np.array(CAMERAS)

    out_path = OUT_DIR / "similarity_maps.npz"
    np.savez_compressed(out_path, **npz_data)
    print(f"Saved to {out_path}")

    # Print summary
    print("\nMean positive surgery similarity [prompt x camera]:")
    header = f"{'Prompt':<45}" + "".join(f"{c:>10}" for c in CAMERAS)
    print(header)
    print("-" * len(header))
    for pi, prompt in enumerate(TEXT_PROMPTS):
        row = f"{prompt:<45}"
        for cam in CAMERAS:
            vals = []
            for frame_idx in sampled:
                hmap = npz_data[f"surgery_{frame_idx}_{cam}"][pi]
                vals.append(hmap[hmap > 0].mean() if (hmap > 0).any() else 0.0)
            row += f"{np.mean(vals):>10.4f}"
        print(row)


if __name__ == "__main__":
    main()
