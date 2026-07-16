"""session.py — the recorder process's testable heart (MAY-173 slam-demo-loop 1.4).

Turns live-channel reads into a bake-grade coverage dump. The caller (recorder_main)
polls the camera mailboxes and the debug-pose channel and feeds everything here;
the session owns dedupe, stamp↔pose joining, FK rows, and gap accounting, writing
through an ASYNC `DriveRecorder` (the feed path never blocks on an encoder).

Row discipline (what `rosbag_synth` requires):
  - samples are grouped BY STAMP → the base row is emitted once per frame stamp,
    pose = nearest GT sample (the debug-pose stream runs ~10× faster than frames)
  - cam rows carry `T_world_cam` from FK — consistent with the base row by
    construction, which is all `recover_static_mount` needs
  - frames arriving before any GT pose are counted (`skipped_no_pose`), never
    written with a made-up pose

The camera channel is latest-wins (lossy): the session cannot recover a dropped
frame, so it REPORTS — `gaps` (cadence-break events on the distinct-stamp
timeline) and `missed_frames` (estimated frames lost) — for the panel and the
acceptance audit to judge.

Brain-pure: contracts + fk + mapping.recorder. No sockets here (testable).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np

from humanoid.logic.oli.camera_mounts import CAMERAS, STEREO_CAMERAS
from humanoid.logic.oli.contracts import CameraFrame
from humanoid.logic.oli.recording.fk import cam_world, rig_dict
from humanoid.logic.simulation.mapping.recorder import DriveRecorder

_MOUNTS = {m.name: m for m in (*CAMERAS, *STEREO_CAMERAS)}
_STAMP_MEMORY = 64          # recent base stamps kept for cross-stream dedupe
_POSE_BUFFER = 512          # recent GT samples kept for nearest-stamp joining


class RecordingSession:
    """One live recording: feed frames + GT poses, get a coverage dump."""

    def __init__(
        self,
        out_dir: Path | str,
        *,
        camera_res: Tuple[int, int],
        streams: Iterable[str],
        codec: str = "jpeg",
        quality: int = 95,
        workers: int = 2,
        queue_frames: int = 64,
    ) -> None:
        self._rec = DriveRecorder(
            out_dir, rig_dict(tuple(camera_res)),
            codec=codec, quality=quality,
            async_write=True, workers=workers, queue_frames=queue_frames,
        )
        self._streams = list(streams)
        self._poses: Deque[Tuple[int, float, float, float]] = deque(maxlen=_POSE_BUFFER)
        self._last_stamp: Dict[str, int] = {}          # per-stream dedupe watermark
        self._base_emitted: Deque[int] = deque(maxlen=_STAMP_MEMORY)
        self._stamps: list = []                        # distinct base stamps, in order
        self._per_stream: Dict[str, int] = {s: 0 for s in self._streams}
        self._frames = 0
        self._skipped_no_pose = 0
        self._stopped = False

    # ── feeding ──────────────────────────────────────────────────────────────

    def feed_base(self, stamp_ns: int, *, x: float, y: float, yaw: float) -> None:
        """Buffer one GT base sample (debug-pose channel). Rows are emitted on
        FRAME stamps, not here — the synth groups samples by stamp."""
        self._poses.append((int(stamp_ns), float(x), float(y), float(yaw)))

    def feed_frame(self, frame: CameraFrame) -> bool:
        """Persist one camera frame if it is NEW for its stream. Returns True if
        the frame was submitted to the writer."""
        name, stamp = frame.name, int(frame.stamp_ns)
        if self._last_stamp.get(name, -1) >= stamp:
            return False                               # mailbox re-delivery / stale
        pose = self._nearest_pose(stamp)
        if pose is None:
            self._skipped_no_pose += 1
            self._last_stamp[name] = stamp             # don't recount the same frame
            return False
        _, x, y, yaw = pose
        if stamp not in self._base_emitted:            # once per stamp, across streams
            self._rec.add_base_pose(stamp, x=x, y=y, yaw=yaw)
            self._base_emitted.append(stamp)
            self._stamps.append(stamp)
        mount = _MOUNTS[name]
        self._rec.add_frame(
            name, stamp, frame.rgb, cam_world(x, y, yaw, mount),
            depth_m=frame.depth,
        )
        self._last_stamp[name] = stamp
        self._per_stream[name] = self._per_stream.get(name, 0) + 1
        self._frames += 1
        return True

    # ── reporting ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        gaps, missed = self._gap_stats()
        return {
            "frames": self._frames,
            "stamps": len(self._stamps),
            "per_stream": dict(self._per_stream),
            "gaps": gaps,
            "missed_frames": missed,
            "skipped_no_pose": self._skipped_no_pose,
            "last_stamp_ns": self._stamps[-1] if self._stamps else None,
            "queued": self._rec.frames_written,        # persisted so far (async lag)
        }

    def stop(self) -> dict:
        """Drain the writer and close the dump. Idempotent. Returns final stats."""
        if not self._stopped:
            self._rec.close()
            self._stopped = True
        return self.stats()

    # ── internals ────────────────────────────────────────────────────────────

    def _nearest_pose(self, stamp_ns: int) -> Optional[Tuple[int, float, float, float]]:
        if not self._poses:
            return None
        return min(self._poses, key=lambda p: abs(p[0] - stamp_ns))

    def _gap_stats(self) -> Tuple[int, int]:
        """Cadence breaks on the distinct-stamp timeline: a delta > 1.5× the median
        counts one gap event and ~delta/median - 1 missed frames."""
        if len(self._stamps) < 3:
            return 0, 0
        deltas = np.diff(np.asarray(self._stamps, dtype=np.int64))
        med = float(np.median(deltas))
        if med <= 0:
            return 0, 0
        big = deltas[deltas > 1.5 * med]
        gaps = int(big.size)
        missed = int(sum(int(round(d / med)) - 1 for d in big))
        return gaps, missed
