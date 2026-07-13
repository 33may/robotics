"""imaging.py — pure-numpy display helpers for panels (no imgui / GL dependency).

Kept dependency-free so the transforms are unit-testable in any env.
"""

from __future__ import annotations

import numpy as np


def fit_within(img_w: int, img_h: int, box_w: float, box_h: float) -> tuple[int, int]:
    """Largest (w, h) that keeps `img_w:img_h` aspect and fits inside the `box_w × box_h`.

    Scales UP or DOWN — the returned image fills as much of the box as possible while
    staying fully visible (no crop, no scroll). Used to size panel images to the live
    ImGui content region so two docked panels each show their whole image. A degenerate
    (≤0) box clamps to 1×1 so ImmVision never gets a zero dimension.
    """
    if img_w <= 0 or img_h <= 0 or box_w <= 0 or box_h <= 0:
        return (1, 1)
    scale = min(box_w / img_w, box_h / img_h)
    return (max(1, int(img_w * scale)), max(1, int(img_h * scale)))


def colorize_depth(depth: np.ndarray, near: float = 0.2, far: float = 5.0) -> np.ndarray:
    """Map a float depth image (metres) to a uint8 (H, W, 3) RGB jet colourization.

    Values are clipped to [near, far] then mapped by a standard jet ramp: near→blue,
    mid→green/yellow, far→red. Non-finite depth (invalid) renders black. Returns a
    contiguous uint8 array (immvision-ready).
    """
    d = depth.astype(np.float32)
    invalid = ~np.isfinite(d)
    # nan/inf → 0 before the arithmetic so the uint8 cast never sees a non-finite value.
    t = np.clip(np.nan_to_num((d - near) / max(far - near, 1e-6), nan=0.0), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[invalid] = 0
    return np.ascontiguousarray(rgb)
