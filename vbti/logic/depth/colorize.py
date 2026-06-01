"""Depth-map colorization + uint16↔RGB packing helpers. Pure numpy/cv2.

Two colorization modes (lossy, baked at write time):
- ``colorize_fixed_clip``: clip metric depth to [m_min, m_max], linearly map to [0,255], turbo.
  Preserves cross-frame absolute distance information.
- ``colorize_per_frame_norm``: per-frame min-max → [0,1] → turbo.
  Sidesteps absolute-scale mismatches (e.g. DA-V2 estimated depth vs real D405),
  but the policy can no longer learn absolute distance.

Lossless packing (v016+ canonical):
- ``pack_uint16_to_rgb`` / ``unpack_rgb_to_uint16``: pair to round-trip uint16
  depth through a 3-channel uint8 image. Mirrors the encoding done at capture
  time by ``RealSenseCamera.read_latest_depth_packed``. Use these in DataLoader
  transforms to recover metric depth from PNG-stored frames.
"""
from __future__ import annotations

import cv2
import numpy as np


def colorize_fixed_clip(
    depth: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Clip-then-turbo. ``depth`` is (H, W) float in meters.

    Returns (H, W, 3) uint8 RGB.
    """
    clipped = np.clip(depth, clip_min, clip_max)
    span = max(clip_max - clip_min, 1e-6)
    norm = (clipped - clip_min) / span
    u8 = (norm * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pack_uint16_to_rgb(depth_u16: np.ndarray) -> np.ndarray:
    """Pack uint16 depth into 3-channel uint8 RGB (lossless when stored as PNG).

    Encoding: ``R = (depth >> 8) & 0xFF``, ``G = depth & 0xFF``,
    ``B = clip(depth / 256, 0, 255)`` (preview byte, not metric — ignore on read).
    """
    d = depth_u16.astype(np.uint16)
    r = ((d >> 8) & 0xFF).astype(np.uint8)
    g = (d & 0xFF).astype(np.uint8)
    b = np.clip(d.astype(np.float32) / 256.0, 0.0, 255.0).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def unpack_rgb_to_uint16(rgb_u8: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_uint16_to_rgb`. ``rgb_u8`` is (H, W, 3) uint8.

    Returns ``(H, W) uint16`` — the original depth in hardware units (mm for D405).
    Only R and G are read; B is a viewer-only preview byte.
    """
    r = rgb_u8[..., 0].astype(np.uint16)
    g = rgb_u8[..., 1].astype(np.uint16)
    return (r << 8) | g


def colorize_per_frame_norm(depth: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalize → turbo.

    Use this when the absolute scale of ``depth`` is not trustworthy
    or differs between data sources (e.g. estimated vs. real D405).
    """
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max - d_min < 1e-6:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    norm = (depth - d_min) / (d_max - d_min)
    u8 = (norm * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
