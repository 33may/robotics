"""recorder.py — motion-agnostic coverage-drive sink (MAY-173 locdev T2).

Persists what was RENDERED: RGB frames + each camera's actual world pose (read
from the camera prim by the caller — NOT derived base ∘ mount, so a later
walk-motion mimic that perturbs camera prims keeps recordings truthful with zero
changes here). The neutral on-disk layout is the durable artifact; rosbag
synthesis for `create_map_offline.py` happens later inside the Isaac ROS
container and is re-runnable from this dump.

    <out>/rig.json                     rig dict, verbatim (intrinsics, mounts, baseline)
    <out>/frames/<cam>/<stamp>.png     19-digit zero-padded ns → lexical sort = time sort
    <out>/poses.jsonl                  one row per frame; cam="base" rows carry x,y,yaw

Pure numpy + PIL + stdlib; runs inside the isaac-env world process but never
imports isaacsim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


class DriveRecorder:
    """Write-once recorder for one coverage drive."""

    def __init__(self, out_dir: Path | str, rig: Dict[str, Any]) -> None:
        self._root = Path(out_dir)
        (self._root / "frames").mkdir(parents=True, exist_ok=True)
        (self._root / "rig.json").write_text(json.dumps(rig, indent=2))
        self._poses = open(self._root / "poses.jsonl", "w", encoding="utf-8")
        self.frames_written = 0

    def add_frame(
        self,
        cam: str,
        stamp_ns: int,
        rgb: np.ndarray,
        T_world_cam: np.ndarray,
        depth_m: np.ndarray | None = None,
    ) -> None:
        """Persist one rendered frame + the camera's world pose at that stamp.
        Optional depth (float32 meters) is stored as uint16 millimeter PNG —
        cuVSLAM's own unit convention (depth_scale_factor=1000)."""
        cam_dir = self._root / "frames" / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        rel = f"frames/{cam}/{int(stamp_ns):019d}.png"
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(self._root / rel)
        depth_rel = None
        if depth_m is not None:
            depth_dir = self._root / "frames" / f"{cam}_depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_rel = f"frames/{cam}_depth/{int(stamp_ns):019d}.png"
            mm = np.clip(np.asarray(depth_m, dtype=np.float64) * 1000.0, 0, 65535)
            Image.fromarray(mm.astype(np.uint16)).save(self._root / depth_rel)
        self._write_row(
            cam=cam,
            stamp_ns=int(stamp_ns),
            file=rel,
            depth_file=depth_rel,
            T_world_cam=np.asarray(T_world_cam, dtype=float).tolist(),
        )
        self.frames_written += 1

    def add_base_pose(self, stamp_ns: int, *, x: float, y: float, yaw: float) -> None:
        """Persist the base GT pose (cuVSLAM warm-start + bake pose hints)."""
        self._write_row(
            cam="base", stamp_ns=int(stamp_ns), file=None,
            x=float(x), y=float(y), yaw=float(yaw),
        )

    def _write_row(self, **row: Any) -> None:
        self._poses.write(json.dumps(row) + "\n")

    def close(self) -> None:
        if not self._poses.closed:
            self._poses.close()
