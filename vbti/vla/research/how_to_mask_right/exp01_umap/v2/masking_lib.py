"""Masking strategies and SigLIP embedding utilities for mask analysis."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FILL_MODES = ["black", "mean", "noise", "blur", "shuffle"]


# ── Fill ────────────────────────────────────────────────────────────────────

def _apply_fill(img: torch.Tensor, out: torch.Tensor, mask: torch.Tensor,
                fill: str, blur_kernel: int = 51):
    C = img.shape[0]
    if fill == "black":
        out[:, mask] = 0.0
    elif fill == "mean":
        out[:, mask] = img.mean()
    elif fill == "noise":
        n = mask.sum().item()
        out[:, mask] = torch.rand(C, n)
    elif fill == "blur":
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        padded = F.pad(img.unsqueeze(0), [k // 2] * 4, mode="reflect")
        blurred = F.avg_pool2d(padded, kernel_size=k, stride=1)[0]
        out[:, mask] = blurred[:, mask]
    elif fill == "shuffle":
        for c in range(C):
            vals = out[c, mask].clone()
            out[c, mask] = vals[torch.randperm(len(vals))]


# ── Mask shapes ─────────────────────────────────────────────────────────────

def _generate_mask_field(H: int, W: int, scale: float = 110.0,
                         threshold: float = 0.55) -> torch.Tensor:
    noise_h = max(int(H / scale), 2)
    noise_w = max(int(W / scale), 2)
    small_noise = torch.randn(1, 1, noise_h, noise_w)
    field = F.interpolate(small_noise, size=(H, W), mode="bicubic",
                          align_corners=False)[0, 0]
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field > threshold


def mask_gaussian_noise(img: torch.Tensor, scale: float = 110.0,
                        threshold: float = 0.55, fill: str = "black",
                        blur_kernel: int = 51) -> tuple[torch.Tensor, torch.Tensor]:
    C, H, W = img.shape
    binary_mask = _generate_mask_field(H, W, scale, threshold)
    out = img.clone()
    _apply_fill(img, out, binary_mask, fill, blur_kernel)
    return out, binary_mask.float()


def mask_rectangular_cutout(img: torch.Tensor, n_patches: int = 3,
                            patch_scale: tuple[float, float] = (0.05, 0.15),
                            fill: str = "black",
                            blur_kernel: int = 51) -> tuple[torch.Tensor, torch.Tensor]:
    C, H, W = img.shape
    out = img.clone()
    combined_mask = torch.zeros(H, W, dtype=torch.bool)
    for _ in range(n_patches):
        scale = torch.empty(1).uniform_(*patch_scale).item()
        area = H * W * scale
        aspect = torch.empty(1).uniform_(0.5, 2.0).item()
        ph = int(min((area * aspect) ** 0.5, H))
        pw = int(min((area / aspect) ** 0.5, W))
        y = torch.randint(0, max(H - ph, 1), (1,)).item()
        x = torch.randint(0, max(W - pw, 1), (1,)).item()
        combined_mask[y:y + ph, x:x + pw] = True
    _apply_fill(img, out, combined_mask, fill, blur_kernel)
    return out, combined_mask.float()


SHAPES = {
    "gaussian_noise": lambda img, fill, bk: mask_gaussian_noise(
        img, scale=100, threshold=0.6, fill=fill, blur_kernel=bk)[0],
    "rectangular_cutout": lambda img, fill, bk: mask_rectangular_cutout(
        img, n_patches=3, fill=fill, blur_kernel=bk)[0],
}


# ── SigLIP embedding ───────────────────────────────────────────────────────

def _img_to_pil(img: torch.Tensor) -> Image.Image:
    return Image.fromarray(
        (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))


def load_siglip(model_id: str = "google/siglip-so400m-patch14-384",
                device: torch.device | None = None):
    from transformers import SiglipVisionModel, SiglipImageProcessor
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiglipVisionModel.from_pretrained(model_id).to(device).eval()
    proc = SiglipImageProcessor.from_pretrained(model_id)
    print(f"SigLIP loaded on {device}  "
          f"(hidden_dim={model.config.hidden_size}, "
          f"params={sum(p.numel() for p in model.parameters()):,})")
    return model, proc, device


@torch.no_grad()
def get_embeddings_batch(imgs: list[torch.Tensor], model, processor, device,
                         batch_size: int = 64) -> np.ndarray:
    from tqdm import tqdm
    all_embs = []
    for i in tqdm(range(0, len(imgs), batch_size), desc="Embedding"):
        batch_pils = [_img_to_pil(img) for img in imgs[i:i + batch_size]]
        inputs = processor(images=batch_pils, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embs.append(embs)
    return np.concatenate(all_embs, axis=0)
