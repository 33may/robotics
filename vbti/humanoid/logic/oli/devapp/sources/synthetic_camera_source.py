"""synthetic_camera_source.py — synthetic RGBD source for building/validating the panel.

Yields animated gradient frames for a set of named streams (default chest + head), so the
camera panel can be developed and screenshot-validated independently of the real Isaac
cameras (MAY-149). Swap it for the real source with no panel change.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .camera_source import CameraFrame, CameraIntrinsics

# Per-stream RGB tint so distinct streams are visually distinguishable in one screenshot.
_TINTS: Dict[str, tuple] = {
    "chest": (1.0, 1.0, 1.0),
    "head": (1.0, 0.6, 0.7),
}


class SyntheticCameraSource:
    """A CameraSource yielding animated gradient RGBD for a set of named streams."""

    def __init__(
        self,
        names: Sequence[str] = ("chest", "head"),
        width: int = 640,
        height: int = 360,
    ) -> None:
        self._names: List[str] = list(names)
        self._w, self._h = int(width), int(height)
        self._count: Dict[str, int] = {k: 0 for k in self._names}

        ys, xs = np.mgrid[0:self._h, 0:self._w]
        self._xs = (xs / max(self._w - 1, 1)).astype(np.float32)
        self._ys = (ys / max(self._h - 1, 1)).astype(np.float32)
        cx, cy = self._w / 2.0, self._h / 2.0
        radial = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
        self._radial = radial / radial.max()
        f = 0.9 * self._w
        self._intr = CameraIntrinsics(self._w, self._h, f, f, cx, cy)

    def stream_names(self) -> List[str]:
        return list(self._names)

    def read(self, name: str) -> Optional[CameraFrame]:
        if name not in self._count:
            return None
        i = self._count[name]
        self._count[name] = i + 1
        phase = i * 0.1

        r = self._xs
        g = self._ys
        b = 0.5 + 0.5 * np.sin(phase + self._xs * 6.0)
        rgb = np.stack([r, g, b], axis=-1)
        tint = _TINTS.get(name, (1.0, 1.0, 1.0))
        rgb = np.clip(rgb * np.asarray(tint, dtype=np.float32), 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)

        # Moving white square marker proves the stream is live across frames.
        sq = max(self._w // 16, 8)
        px = int((0.5 + 0.4 * np.sin(phase)) * (self._w - sq))
        py = int((0.5 + 0.4 * np.cos(phase * 0.7)) * (self._h - sq))
        rgb[py:py + sq, px:px + sq] = (255, 255, 255)

        depth = (0.3 + 3.0 * self._radial + 0.1 * np.sin(phase)).astype(np.float32)
        return CameraFrame(
            stamp_ns=i * 33_000_000,  # ~30 Hz
            name=name,
            rgb=np.ascontiguousarray(rgb),
            depth=np.ascontiguousarray(depth),
            intrinsics=self._intr,
        )

    def close(self) -> None:
        pass
