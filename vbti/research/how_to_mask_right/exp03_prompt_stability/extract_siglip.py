"""Step 1: Extract raw SigLIP patch tokens for all captured images.

Runs frozen SigLIP vision encoder (no text, no VLM) on every capture.
Saves 256 patch tokens (576-d each) per image per camera.
This is the Stage 0 baseline — vision before any text influence.

Usage:
    python extract_siglip.py

Output:
    siglip_tokens.npz with keys:
        tokens_{dataset}_{index}_{camera}  →  (256, 576) float32
        datasets  →  list of dataset names
        indices_{dataset}  →  list of capture indices
        cameras  →  ["top", "left", "right", "gripper"]
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import SiglipVisionModel, SiglipImageProcessor
from PIL import Image
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────

EXP_DIR = Path("/home/may33/projects/ml_portfolio/robotics/vbti/research/how_to_mask_right/exp03_prompt_stability")
DATASETS = {
    "fixed_duck_free_red": EXP_DIR / "captures_fixed_duck_free_red",
    "free_duck_fixed_red": EXP_DIR / "captures_free_duck_fixed_red",
    "free_duck_free_red":  EXP_DIR / "captures_free_duck_free_red",
}
CAMERAS = ["top", "left", "right", "gripper"]
MODEL_ID = "google/siglip-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH = EXP_DIR / "siglip_tokens.npz"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading SigLIP vision encoder ({MODEL_ID})...")
    model = SiglipVisionModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = SiglipImageProcessor.from_pretrained(MODEL_ID)
    print(f"Loaded on {DEVICE}")

    npz_data = {}
    npz_data["cameras"] = np.array(CAMERAS)

    for ds_name, ds_path in DATASETS.items():
        # Find all capture indices
        meta_files = sorted(ds_path.glob("*_meta.json"))
        indices = [int(f.stem.split("_")[0]) for f in meta_files]
        npz_data[f"indices_{ds_name}"] = np.array(indices)
        print(f"\n{ds_name}: {len(indices)} captures")

        for idx in tqdm(indices, desc=f"  {ds_name}"):
            for cam in CAMERAS:
                img_path = ds_path / f"{idx:03d}_{cam}.png"
                if not img_path.exists():
                    print(f"    MISSING: {img_path.name}")
                    continue

                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    outputs = model(pixel_values=inputs["pixel_values"])
                    patch_tokens = outputs.last_hidden_state[0].cpu().numpy()  # (729, 1152)

                npz_data[f"tokens_{ds_name}_{idx}_{cam}"] = patch_tokens

    # Save
    np.savez_compressed(OUT_PATH, **npz_data)
    total_keys = sum(1 for k in npz_data if k.startswith("tokens_"))
    print(f"\nSaved {total_keys} token arrays to {OUT_PATH}")
    print(f"Each array: {patch_tokens.shape} (patches, hidden_dim)")


if __name__ == "__main__":
    main()
