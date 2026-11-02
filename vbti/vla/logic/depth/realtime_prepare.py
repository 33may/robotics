"""Realtime equivalent of ``bake_packed_depth.py`` — single-frame uint16 → turbo RGB.

The dataset prep pipeline (``vbti.logic.depth.bake_packed_depth``) reads each
parquet's packed-PNG depth, unpacks to uint16, multiplies by ``depth_scale_m``
to get meters, clips to ``[clip_min_m, clip_max_m]``, normalizes, and applies
the turbo colormap → 3-channel uint8 RGB stored back as a PNG. That's what the
v018 SmolVLA policy was trained on.

At inference time we don't have packed-PNG bytes; we have a live uint16 depth
frame straight from the D405 (already aligned to the color image via
``rs.align``). This module skips the unpack step and runs the same final
clip→normalize→turbo step using the SAME ``colorize_fixed_clip`` function the
bake uses — guaranteed pixel-parity with training data.

Parity-critical knobs (must match the dataset's bake):
- ``depth_scale_m``: meters per uint16 unit. D405 default = ``1e-4``.
- ``clip_min_m`` / ``clip_max_m``: canonical workspace clip = ``[0.05, 0.20]`` m.
- LUT: ``cv2.COLORMAP_TURBO`` (handled inside ``colorize_fixed_clip``).

If the dataset was baked with non-default values, override here at call time.
"""
from __future__ import annotations

import numpy as np

from vbti.logic.depth.colorize import colorize_fixed_clip
# Single source of truth for the canonical clip + scale. Imported (not redefined)
# so any future tweak in ``depth_transform.py`` flows here automatically.
from vbti.logic.dataset.depth_transform import (
    DEFAULT_CLIP_MIN_M,
    DEFAULT_CLIP_MAX_M,
    DEFAULT_DEPTH_SCALE_M,
)


def depth_uint16_to_turbo_rgb(
    depth_u16: np.ndarray,
    depth_scale_m: float = DEFAULT_DEPTH_SCALE_M,
    clip_min_m:    float = DEFAULT_CLIP_MIN_M,
    clip_max_m:    float = DEFAULT_CLIP_MAX_M,
) -> np.ndarray:
    """Live D405 depth (uint16 hardware units) → turbo-RGB matching the bake.

    Args:
        depth_u16: ``(H, W)`` uint16. Hardware-units depth aligned to color.
            (Use ``rs.align(rs.stream.color)`` upstream of this call.)
        depth_scale_m: meters per uint16 unit. D405 default = ``1e-4``.
        clip_min_m, clip_max_m: same canonical clip as ``bake_packed_depth.py``.

    Returns:
        ``(H, W, 3)`` uint8 RGB — pixel-identical to what
        ``bake_packed_depth.py`` writes into the
        ``observation.images.gripper_depth`` PNG cells of the training dataset.

    Notes:
        - Pixels at distance < ``clip_min_m`` or > ``clip_max_m`` saturate to
          the LUT ends (deep-blue and dark-red respectively) — same as the bake.
        - Pixels with raw value 0 (D405's "no depth here" sentinel) become
          ``clip_min_m`` after clipping. If the policy needs an explicit
          "invalid" channel, it must be added in this module.
    """
    if depth_u16.dtype != np.uint16:
        depth_u16 = depth_u16.astype(np.uint16)
    depth_m = depth_u16.astype(np.float32) * depth_scale_m
    return colorize_fixed_clip(depth_m, clip_min_m, clip_max_m)
