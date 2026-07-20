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


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Stable bright RGB for a feature track id (golden-ratio hue walk — adjacent ids get
    far-apart hues, and a given id keeps its color for its whole tracked life)."""
    h = (track_id * 0.61803398875) % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    v, p, q, t = 255, 64, int(255 - 191 * f), int(64 + 191 * f)
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return rgb


def bake_feature_dots(rgb: np.ndarray, observations, radius: int = 4) -> np.ndarray:
    """Overlay cuVSLAM feature observations [(u, v, id), …] onto a COPY of `rgb`.

    The dev_app Localization panel's "what the robot sees" view (NVIDIA figure-3 style):
    square dots colored stably per track id. Pure numpy; observations outside the frame
    are skipped (defensive — the tracker emits sub-pixel u/v near borders).
    """
    img = np.ascontiguousarray(rgb).copy()
    h, w = img.shape[:2]
    for u, v, tid in observations:
        c, r = int(round(u)), int(round(v))
        if not (0 <= r < h and 0 <= c < w):
            continue
        r0, r1 = max(0, r - radius), min(h, r + radius + 1)
        c0, c1 = max(0, c - radius), min(w, c + radius + 1)
        img[r0:r1, c0:c1] = _id_color(tid)
    return img
