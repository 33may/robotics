"""Per-sample DataLoader transform: decode packed-PNG depth → turbo RGB.

The dataset stores ``observation.images.gripper_depth`` as a uint16 metric depth
(0.0001 m / unit on the D405) packed into a 3-channel uint8 PNG:

    R = (depth >> 8) & 0xFF    # high byte
    G =  depth       & 0xFF    # low byte
    B =  depth // 256          # preview byte (viewer-only — ignored)

LeRobot's dataset reader returns this as a float32 tensor in ``[0, 1]`` with
shape ``(3, H, W)`` (no delta_timestamps) or ``(T, 3, H, W)`` (with delta_timestamps).

This module wraps a base dataset so that every ``__getitem__`` decodes the depth
back to metric, clips it to the canonical range, normalizes, and applies the
turbo colormap — producing a smooth RGB image whose color encodes distance,
suitable for SmolVLA's frozen SigLIP encoder.

Why per-sample on the worker (not baked into the dataset):
- Storage stays lossless uint16 → future v019+ can re-render with different
  clips without rebuilding the dataset.
- DataLoader workers parallelize the cost; for a 480×640 frame the LUT-based
  decode takes ~0.3 ms on CPU.

Why turbo specifically:
- 3-channel image stays on the natural-image manifold that SigLIP was trained on.
- Smooth, high-contrast across the full clip range.
- Drawback: not perceptually monotonic — model has to learn the colormap. If
  v018 underperforms vs v017 we can swap to viridis (monotonic-luminance) in
  v019 by changing the LUT only.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


# Canonical clip for v016+ datasets. Chosen 2026-04-30 to maximize duck/cup
# detail at grasp (close-range): the LUT span sits entirely inside the gripper
# workspace. Anything farther than 0.20 m saturates to the red end of turbo —
# that's an explicit "background / not relevant" signal for the policy.
DEFAULT_CLIP_MIN_M = 0.05
DEFAULT_CLIP_MAX_M = 0.20
# D405 default depth_scale (m / unit). Matches the value emitted by
# RealSenseCamera at connect time and used by ``add_gripper_depth`` /
# ``capture_gripper_sample``.
DEFAULT_DEPTH_SCALE_M = 1e-4


def _build_turbo_lut() -> torch.Tensor:
    """Return a (256, 3) uint8 RGB LUT for cv2.COLORMAP_TURBO."""
    grid = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(grid, cv2.COLORMAP_TURBO).reshape(256, 3)
    rgb = bgr[:, ::-1].copy()  # BGR → RGB
    return torch.from_numpy(rgb)  # (256, 3) uint8


def decode_packed_depth_to_turbo(
    x: torch.Tensor,
    clip_min: float = DEFAULT_CLIP_MIN_M,
    clip_max: float = DEFAULT_CLIP_MAX_M,
    scale: float = DEFAULT_DEPTH_SCALE_M,
    lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert packed-PNG depth tensor → turbo-colored RGB tensor.

    Args:
        x: float32 tensor in ``[0, 1]``, shape ``(..., 3, H, W)``. Channels are
           ``[R=high_byte/255, G=low_byte/255, B=preview]``.
        clip_min, clip_max: metric depth clip range (m).
        scale: meters-per-unit of the stored uint16 (D405 default = 1e-4).
        lut: optional pre-built (256, 3) uint8 LUT; built on the fly if None.

    Returns:
        float32 tensor in ``[0, 1]``, shape ``(..., 3, H, W)``.
        Pixels at distances < clip_min or > clip_max saturate to the LUT ends.
    """
    if lut is None:
        lut = _build_turbo_lut()
    lut = lut.to(x.device)

    # uint8 channels — round trip through float because that's what LeRobot
    # gives us; the int values are exact (PNG → uint8 → /255 → ×255 → round).
    xu8 = (x * 255.0).round().clamp_(0, 255).to(torch.int32)
    high = xu8[..., 0, :, :]
    low = xu8[..., 1, :, :]
    d_u16 = (high << 8) | low                        # (..., H, W) int32
    d_m = d_u16.to(torch.float32) * scale            # metric meters
    d_clipped = d_m.clamp(clip_min, clip_max)
    span = max(clip_max - clip_min, 1e-6)
    d_norm = (d_clipped - clip_min) / span           # [0, 1]
    d_idx = (d_norm * 255.0).round().clamp_(0, 255).to(torch.long)  # (..., H, W)

    # LUT lookup → (..., H, W, 3) uint8
    rgb = lut[d_idx]
    # → (..., 3, H, W) float32 in [0, 1]
    rgb = rgb.movedim(-1, -3).to(torch.float32) / 255.0
    return rgb


class GripperDepthDecoder(torch.utils.data.Dataset):
    """Wrap a base LeRobotDataset and decode packed-PNG depth in place."""

    def __init__(
        self,
        base: torch.utils.data.Dataset,
        depth_keys: list[str],
        clip_min: float = DEFAULT_CLIP_MIN_M,
        clip_max: float = DEFAULT_CLIP_MAX_M,
        scale: float = DEFAULT_DEPTH_SCALE_M,
    ):
        self.base = base
        self.depth_keys = list(depth_keys)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.scale = scale
        self._lut = _build_turbo_lut()

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        item = self.base[idx]
        for key in self.depth_keys:
            if key in item:
                item[key] = decode_packed_depth_to_turbo(
                    item[key],
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                    scale=self.scale,
                    lut=self._lut,
                )
        return item

    # Forward useful attributes so consumers don't notice the wrap.
    @property
    def fps(self):
        return getattr(self.base, "fps", None)

    @property
    def meta(self):
        return getattr(self.base, "meta", None)

    def __getattr__(self, name):
        # Called only when attribute not found on self — delegate to base.
        return getattr(self.__dict__["base"], name)
