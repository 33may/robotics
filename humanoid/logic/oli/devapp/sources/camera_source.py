"""camera_source.py — the CameraSource protocol + the CameraFrame it yields.

Panels depend on THIS, never on Isaac or a socket, so swapping the concrete source
(TestCameraSource → IsaacCameraSource → RealCameraSource) needs zero panel change.

`CameraFrame` mirrors MAY-149's forthcoming `contracts.CameraFrame`
(stamp, name, rgb, depth, intrinsics). It is defined locally as a stand-in until that
contract lands in `logic/oli/contracts.py`; when it does, this module re-exports it and
the shape is unchanged. Intrinsics only (no extrinsics) — camera pose is derived brain
side by FK, identical sim vs real (see agent memory `oli-perception-camera-design`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for one camera (pixels)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraFrame:
    """One RGBD frame from one camera.

    rgb:   (H, W, 3) uint8, channel order RGB
    depth: (H, W) float32, metres (planar Z; non-finite = invalid)
    """

    stamp_ns: int
    name: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: Optional[CameraIntrinsics] = None


@runtime_checkable
class CameraSource(Protocol):
    """Non-blocking, latest-wins provider of camera frames.

    `read()` must never block the UI thread: a socket-backed source buffers frames on its
    own reader thread and returns the newest (or None if none has arrived yet).
    """

    def stream_names(self) -> List[str]:
        """Names of the streams this source exposes (e.g. ["chest", "head"])."""
        ...

    def read(self, name: str) -> Optional[CameraFrame]:
        """Latest frame for `name`, or None if unavailable. Non-blocking."""
        ...

    def close(self) -> None:
        """Release any resources (sockets, threads)."""
        ...
