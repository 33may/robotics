"""Step 2: Extract VLM-processed vision tokens for all captured images x prompts.

Runs the SmolVLA VLM prefix forward pass (SigLIP + connector + transformer self-attention)
with different text prompts to see how prompt text modulates vision representations.

The key difference from extract_siglip.py: these tokens have been through the VLM's
self-attention layers where vision and language tokens interact.

Usage:
    conda run -n lerobot python extract_vlm_tokens.py

Output:
    vlm_tokens.npz with keys:
        vlm_tokens_{prompt_idx}_{dataset}_{capture_idx}_{camera} -> (num_patches, hidden_dim) float32
        num_vision_patches -> int
        prompts -> array of prompt strings
        datasets -> array of dataset names
        indices_{dataset} -> array of capture indices
        cameras -> array of camera names
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "/home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v001/checkpoints/best"

EXP_DIR = Path("/home/may33/projects/ml_portfolio/robotics/vbti/research/how_to_mask_right/exp03_prompt_stability")
DATASETS = {
    "fixed_duck_free_red": EXP_DIR / "captures_fixed_duck_free_red",
    "free_duck_fixed_red": EXP_DIR / "captures_free_duck_fixed_red",
    "free_duck_free_red":  EXP_DIR / "captures_free_duck_free_red",
}
CAMERAS = ["top", "left", "right", "gripper"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH = EXP_DIR / "vlm_tokens.npz"

PROMPTS = [
    "pick up the duck and place it in the cup",
    "pick up the duck",
    "place it in the cup",
]

# ── Helpers ─────────────────────────────────────────────────────────────────


def load_policy(checkpoint_path):
    """Load SmolVLA policy from checkpoint."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print(f"Loading SmolVLA from {checkpoint_path}...")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(DEVICE)
    print(f"Loaded on {DEVICE}")
    return policy


def tokenize_prompt(prompt, tokenizer, max_length=48, device="cuda"):
    """Tokenize a single prompt string, adding trailing newline as SmolVLA expects."""
    text = prompt if prompt.endswith("\n") else f"{prompt}\n"
    tokens = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    lang_tokens = tokens["input_ids"].to(device)       # (1, max_length)
    lang_masks = tokens["attention_mask"].bool().to(device)  # (1, max_length)
    return lang_tokens, lang_masks


def load_image_tensor(img_path, device="cuda"):
    """Load PNG image as (1, 3, H, W) tensor in [0, 1] range."""
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    return img_tensor


def extract_vlm_vision_tokens(policy, camera_images, lang_tokens, lang_masks):
    """Run VLM prefix forward pass with all 4 cameras, return per-camera vision tokens.

    Args:
        camera_images: dict {camera_name: (1, 3, H, W) tensor in [0,1]}
        lang_tokens: (1, seq_len) tokenized prompt
        lang_masks: (1, seq_len) bool mask

    Returns:
        dict {camera_name: (num_vision_patches, hidden_dim) numpy array}
        num_vision_patches: int
    """
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks, resize_with_pad

    model = policy.model
    config = policy.config

    # image_features defines the camera order the model expects
    camera_order = list(config.image_features.keys())
    # e.g. ["observation.images.top", "observation.images.left", ...]

    images = []
    img_masks = []
    for feat_name in camera_order:
        cam_name = feat_name.split(".")[-1]  # "observation.images.top" -> "top"
        img_tensor = camera_images[cam_name]

        if config.resize_imgs_with_padding is not None:
            img = resize_with_pad(img_tensor, *config.resize_imgs_with_padding, pad_value=0)
        else:
            img = img_tensor
        img = img * 2.0 - 1.0  # [0,1] -> [-1,1] for SigLIP

        images.append(img)
        img_masks.append(torch.ones(1, dtype=torch.bool, device=DEVICE))

    # Dummy state
    state = torch.zeros(1, config.max_state_dim, dtype=torch.float32, device=DEVICE)

    # Run embed_prefix to get prefix embeddings
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )

    # Build attention masks and position IDs
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1

    # Run VLM forward — prefix only, no expert suffix needed
    # With inputs_embeds=[prefix_embs], expert layers are skipped (no inputs_embeds[1])
    hidden_states_list, _ = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs],
        use_cache=False,
        fill_kv_cache=True,
    )
    vlm_output = hidden_states_list[0]  # (1, prefix_length, hidden_dim)

    # Figure out vision patch positions per camera in the prefix sequence
    # Layout per camera: [special_tokens(if any) | vision_patches | end_token(if any)]
    # Then: [cam0_block | cam1_block | cam2_block | cam3_block | lang_tokens | state]

    # Get num patches per camera
    with torch.no_grad():
        test_emb = model.vlm_with_expert.embed_image(images[0])
        num_vision_patches = test_emb.shape[1]

    # Tokens per camera block
    special_start = 2 if model.add_image_special_tokens else 0  # image_start tokens
    special_end = 1 if model.add_image_special_tokens else 0    # image_end token
    tokens_per_camera = special_start + num_vision_patches + special_end

    # Extract vision tokens for each camera
    result = {}
    for cam_idx, feat_name in enumerate(camera_order):
        cam_name = feat_name.split(".")[-1]
        block_start = cam_idx * tokens_per_camera
        vision_start = block_start + special_start
        vision_end = vision_start + num_vision_patches
        result[cam_name] = vlm_output[0, vision_start:vision_end, :].float().cpu().numpy()

    return result, num_vision_patches


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    policy = load_policy(CHECKPOINT_PATH)
    model = policy.model
    config = policy.config

    # Load tokenizer from the VLM backbone
    tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)

    # Print model info
    print(f"\nModel config:")
    print(f"  VLM backbone: {config.vlm_model_name}")
    print(f"  resize_imgs_with_padding: {config.resize_imgs_with_padding}")
    print(f"  add_image_special_tokens: {model.add_image_special_tokens}")
    print(f"  prefix_length: {model.prefix_length}")
    print(f"  tokenizer_max_length: {config.tokenizer_max_length}")
    print(f"  max_state_dim: {config.max_state_dim}")
    print(f"  image_features: {list(config.image_features.keys())}")
    hidden_dim = model.vlm_with_expert.config.text_config.hidden_size
    print(f"  VLM hidden_size: {hidden_dim}")

    # Test run: 1 capture (all 4 cameras), 1 prompt to verify shapes
    print("\n--- Test run ---")
    first_ds = list(DATASETS.keys())[0]
    first_path = DATASETS[first_ds]
    meta_files = sorted(first_path.glob("*_meta.json"))
    first_idx = int(meta_files[0].stem.split("_")[0])

    test_images = {}
    for cam in CAMERAS:
        img_path = first_path / f"{first_idx:03d}_{cam}.png"
        test_images[cam] = load_image_tensor(img_path, DEVICE)
        print(f"  {cam}: {img_path.name} -> {test_images[cam].shape}")

    lang_tokens, lang_masks = tokenize_prompt(PROMPTS[0], tokenizer, config.tokenizer_max_length, DEVICE)
    print(f"  Lang tokens shape: {lang_tokens.shape}")
    print(f"  Active lang tokens: {lang_masks.sum().item()}")

    with torch.no_grad():
        vision_tokens, num_patches = extract_vlm_vision_tokens(
            policy, test_images, lang_tokens, lang_masks
        )
    for cam, vt in vision_tokens.items():
        print(f"  VLM tokens [{cam}]: {vt.shape}")
    print(f"  Num vision patches: {num_patches}")
    print("--- Test passed ---\n")

    # Tokenize all prompts upfront
    all_lang = []
    for prompt in PROMPTS:
        lt, lm = tokenize_prompt(prompt, tokenizer, config.tokenizer_max_length, DEVICE)
        all_lang.append((lt, lm))
        print(f"Prompt '{prompt}': {lm.sum().item()} active tokens")

    # Extract all tokens
    npz_data = {}
    npz_data["cameras"] = np.array(CAMERAS)
    npz_data["prompts"] = np.array(PROMPTS)
    npz_data["num_vision_patches"] = np.array(num_patches)
    npz_data["datasets"] = np.array(list(DATASETS.keys()))

    total_forward_passes = 0
    for ds_name, ds_path in DATASETS.items():
        meta_files = sorted(ds_path.glob("*_meta.json"))
        indices = [int(f.stem.split("_")[0]) for f in meta_files]
        npz_data[f"indices_{ds_name}"] = np.array(indices)
        print(f"\n{ds_name}: {len(indices)} captures")

        for idx in tqdm(indices, desc=f"  {ds_name}"):
            # Load all 4 cameras for this capture
            camera_images = {}
            skip = False
            for cam in CAMERAS:
                img_path = ds_path / f"{idx:03d}_{cam}.png"
                if not img_path.exists():
                    print(f"    MISSING: {img_path.name}, skipping capture {idx}")
                    skip = True
                    break
                camera_images[cam] = load_image_tensor(img_path, DEVICE)
            if skip:
                continue

            # One forward pass per prompt (all 4 cameras processed together)
            for prompt_idx, (lt, lm) in enumerate(all_lang):
                with torch.no_grad():
                    vt_dict, _ = extract_vlm_vision_tokens(policy, camera_images, lt, lm)

                for cam, vt in vt_dict.items():
                    key = f"vlm_tokens_{prompt_idx}_{ds_name}_{idx}_{cam}"
                    npz_data[key] = vt
                total_forward_passes += 1

    # Save
    np.savez_compressed(OUT_PATH, **npz_data)
    total_keys = sum(1 for k in npz_data if k.startswith("vlm_tokens_"))
    print(f"\nSaved {total_keys} token arrays to {OUT_PATH}")
    print(f"Each array: ({num_patches}, {hidden_dim}) (patches, hidden_dim)")
    print(f"Total forward passes: {total_forward_passes} (each processes all 4 cameras)")


if __name__ == "__main__":
    main()
