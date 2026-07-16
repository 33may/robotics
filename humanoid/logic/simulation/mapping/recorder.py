"""recorder.py — motion-agnostic coverage-drive sink (MAY-173 locdev T2 + slam-demo-loop 1.2).

Persists what was RENDERED: RGB frames + each camera's actual world pose (read
from the camera prim by the caller — NOT derived base ∘ mount, so a later
walk-motion mimic that perturbs camera prims keeps recordings truthful with zero
changes here). The neutral on-disk layout is the durable artifact; rosbag
synthesis for `create_map_offline.py` happens later inside the Isaac ROS
container and is re-runnable from this dump.

    <out>/rig.json                       rig dict, verbatim (intrinsics, mounts, baseline)
    <out>/frames/<cam>/<stamp>.png|.jpg  19-digit zero-padded ns → lexical sort = time sort
    <out>/poses.jsonl                    one row per frame; cam="base" rows carry x,y,yaw

Two capture-speed levers (MAY-173 slam-demo-loop 1.2 — live teleop recording):

  codec="jpeg" (q95)   RGB as JPEG — ~15× faster encode, ~3.8× smaller; PROVEN
                       cuVSLAM-safe on the cell benchmark (2026-07-16: ep0
                       0.023→0.029 m, ep1 0.016→0.015 m ATE, zero jumps). Depth
                       ALWAYS stays uint16-mm PNG (lossless by requirement).
  async_write=True     add_frame() hands the (copied) arrays to a bounded queue
                       drained by encoder worker threads — the caller's loop
                       never blocks on an encoder. Backpressure: a full queue
                       BLOCKS the producer (never drops a frame); close()
                       drains everything before returning.

Rows (poses.jsonl / imu.jsonl) are flushed per line: a crash loses at most the
frame in flight, never the jsonl tail. Pure numpy + PIL + stdlib; runs inside
the isaac-env world process (or the brain-side recorder) but never imports
isaacsim.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

_CODECS = {"png": ".png", "jpeg": ".jpg"}


class DriveRecorder:
    """Write-once recorder for one coverage drive."""

    def __init__(
        self,
        out_dir: Path | str,
        rig: Dict[str, Any],
        *,
        codec: str = "png",
        quality: int = 95,
        async_write: bool = False,
        workers: int = 2,
        queue_frames: int = 64,
    ) -> None:
        if codec not in _CODECS:
            raise ValueError(f"unknown codec {codec!r} — one of {sorted(_CODECS)}")
        self._codec = codec
        self._ext = _CODECS[codec]
        self._quality = int(quality)
        self._root = Path(out_dir)
        (self._root / "frames").mkdir(parents=True, exist_ok=True)
        (self._root / "rig.json").write_text(json.dumps(rig, indent=2))
        self._poses = open(self._root / "poses.jsonl", "w", encoding="utf-8")
        self._imu = None
        self._lock = threading.Lock()   # rows + counters (workers share them)
        self._frames_written = 0
        self._q: Optional[queue.Queue] = None
        self._workers: list = []
        if async_write:
            self._q = queue.Queue(maxsize=max(1, int(queue_frames)))
            for _ in range(max(1, int(workers))):
                t = threading.Thread(target=self._worker, daemon=True)
                t.start()
                self._workers.append(t)

    @property
    def frames_written(self) -> int:
        with self._lock:
            return self._frames_written

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
        cuVSLAM's own unit convention (depth_scale_factor=1000).

        Async mode: arrays are COPIED here (the caller may reuse its render
        buffers) and encoded on a worker; a full queue blocks (never drops)."""
        if self._q is not None:
            item = (
                cam, int(stamp_ns),
                np.array(rgb, dtype=np.uint8, copy=True),
                np.array(T_world_cam, dtype=float, copy=True),
                None if depth_m is None else np.array(depth_m, dtype=np.float64, copy=True),
            )
            self._q.put(item)  # bounded → backpressure, not loss
            return
        self._persist_frame(cam, int(stamp_ns), rgb, T_world_cam, depth_m)

    def add_base_pose(self, stamp_ns: int, *, x: float, y: float, yaw: float) -> None:
        """Persist the base GT pose (cuVSLAM warm-start + bake pose hints)."""
        self._write_row(
            cam="base", stamp_ns=int(stamp_ns), file=None,
            x=float(x), y=float(y), yaw=float(yaw),
        )

    def add_imu(self, stamp_ns: int, *, acc, gyro) -> None:
        """Persist one IMU sample (body-frame m/s² + rad/s) to imu.jsonl —
        the raw stream rosbag_synth turns into sensor_msgs/Imu for cuVSLAM
        inertial mode. File is created lazily: no IMU calls → no file."""
        row = json.dumps({
            "stamp_ns": int(stamp_ns),
            "acc": [float(v) for v in acc],
            "gyro": [float(v) for v in gyro],
        })
        with self._lock:
            if self._imu is None:
                self._imu = open(self._root / "imu.jsonl", "w", encoding="utf-8")
            self._imu.write(row + "\n")
            self._imu.flush()

    # ── internals ────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        assert self._q is not None
        q = self._q
        while True:
            item = q.get()
            if item is None:  # sentinel → this worker is done
                q.task_done()
                return
            try:
                self._persist_frame(*item)
            finally:
                q.task_done()

    def _persist_frame(self, cam, stamp_ns, rgb, T_world_cam, depth_m) -> None:
        cam_dir = self._root / "frames" / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        rel = f"frames/{cam}/{stamp_ns:019d}{self._ext}"
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        if self._codec == "jpeg":
            img.save(self._root / rel, quality=self._quality)
        else:
            img.save(self._root / rel)
        depth_rel = None
        if depth_m is not None:
            depth_dir = self._root / "frames" / f"{cam}_depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_rel = f"frames/{cam}_depth/{stamp_ns:019d}.png"  # ALWAYS lossless PNG
            mm = np.clip(np.asarray(depth_m, dtype=np.float64) * 1000.0, 0, 65535)
            Image.fromarray(mm.astype(np.uint16)).save(self._root / depth_rel)
        self._write_row(
            cam=cam,
            stamp_ns=stamp_ns,
            file=rel,
            depth_file=depth_rel,
            T_world_cam=np.asarray(T_world_cam, dtype=float).tolist(),
        )
        with self._lock:
            self._frames_written += 1

    def _write_row(self, **row: Any) -> None:
        line = json.dumps(row) + "\n"
        with self._lock:
            self._poses.write(line)
            self._poses.flush()  # crash loses at most the frame in flight

    def close(self) -> None:
        if self._q is not None:
            for _ in self._workers:
                self._q.put(None)
            for t in self._workers:
                t.join()
            self._q = None
            self._workers = []
        with self._lock:
            if not self._poses.closed:
                self._poses.close()
            if self._imu is not None and not self._imu.closed:
                self._imu.close()
