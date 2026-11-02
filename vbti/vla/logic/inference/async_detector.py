"""Async student-model detector for real-time inference.

A single worker thread runs detection in the background.  The policy loop
submits the latest camera frame (overwriting any still-pending frame for
that camera — we only care about the freshest), and reads back whatever
completed detection is most recent.  Staleness is surfaced via
``frame_idx`` / ``age`` so the caller can decide how to handle it
(extrapolate, hold, mark confidence=0, etc.).

Scheduling across cameras is a weighted round-robin — `priority=3` for a
camera means it gets 3 slots in every cycle, so the gripper camera can
be updated faster than the world-fixed side cameras.

Design notes:
* The worker holds a 1-slot pending buffer per camera.  Submitting a
  frame while another is still pending just overwrites the previous one;
  we never queue up stale work.
* Detection runs on the GPU and serialises against the policy — this
  class doesn't try to be clever about that, it just measures real
  latency so downstream code can model the distribution.
* No cross-thread numpy arrays are mutated after submit/get; the frame
  passed in is kept by reference until the worker is done with it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vbti.logic.detection.detect import StudentDetector


@dataclass
class DetectionResult:
    cam: str
    frame_idx: int         # episode-local frame index the detection was run on
    submitted_t: float     # wall-clock when caller handed us the frame
    started_t: float       # wall-clock when inference actually began
    completed_t: float     # wall-clock when inference finished
    queue_wait: float      # started - submitted (how long the frame waited)
    detect_time: float     # completed - started (pure inference latency)
    result: dict           # {obj: {found, center_norm, bbox, confidence}}


class AsyncDetector:
    def __init__(
        self,
        cameras: list[str],
        detector: StudentDetector | None = None,
        priority: dict[str, int] | None = None,
        warmup: bool = True,
        warmup_shape: tuple[int, int, int] = (480, 640, 3),
    ):
        self.cameras = list(cameras)
        self.detector = detector if detector is not None else StudentDetector()
        self.priority = {c: 1 for c in self.cameras}
        if priority:
            self.priority.update(priority)

        # Build the round-robin schedule once up-front.
        self._schedule: list[str] = []
        for cam in self.cameras:
            self._schedule.extend([cam] * self.priority.get(cam, 1))
        if not self._schedule:
            raise ValueError("AsyncDetector needs at least one camera")

        self._lock = threading.Lock()
        # 1-slot pending frame per camera: {cam: (frame, frame_idx, submitted_t)}
        self._pending: dict[str, tuple[np.ndarray, int, float]] = {}
        self._latest: dict[str, DetectionResult] = {}
        self._history: list[DetectionResult] = []  # append-only, for post-hoc analysis

        self._running = True
        self._wake = threading.Event()

        if warmup:
            self._warmup(warmup_shape)

        self._worker = threading.Thread(
            target=self._run, daemon=True, name="AsyncDetector"
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, cam: str, frame: np.ndarray, frame_idx: int) -> None:
        """Submit a frame. Any pending frame for this camera is overwritten."""
        if cam not in self.priority:
            raise ValueError(f"Unknown camera: {cam!r}")
        with self._lock:
            self._pending[cam] = (frame, frame_idx, time.perf_counter())
        self._wake.set()

    def get_latest(self, cam: str) -> Optional[DetectionResult]:
        with self._lock:
            return self._latest.get(cam)

    def get_all_latest(self) -> dict[str, DetectionResult]:
        with self._lock:
            return dict(self._latest)

    def history(self) -> list[DetectionResult]:
        with self._lock:
            return list(self._history)

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        self._wake.set()
        self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Worker internals
    # ------------------------------------------------------------------

    def _warmup(self, shape: tuple[int, int, int]) -> None:
        dummy = np.zeros(shape, dtype=np.uint8)
        for cam in self.cameras:
            for _ in range(2):
                self.detector.detect(dummy, cam)

    def _run(self) -> None:
        schedule = self._schedule
        si = 0
        idle_sleep = 0.001

        while self._running:
            cam = schedule[si % len(schedule)]
            si += 1

            with self._lock:
                work = self._pending.pop(cam, None)

            if work is None:
                # Nothing for this camera; advance one slot but don't spin hot.
                # Check if ANY camera has pending — if not, wait on the event.
                with self._lock:
                    has_any = bool(self._pending)
                if not has_any:
                    self._wake.wait(timeout=0.05)
                    self._wake.clear()
                else:
                    time.sleep(idle_sleep)
                continue

            frame, frame_idx, submitted_t = work

            started_t = time.perf_counter()
            result = self.detector.detect(frame, cam)
            completed_t = time.perf_counter()

            det_result = DetectionResult(
                cam=cam,
                frame_idx=frame_idx,
                submitted_t=submitted_t,
                started_t=started_t,
                completed_t=completed_t,
                queue_wait=started_t - submitted_t,
                detect_time=completed_t - started_t,
                result=result,
            )
            with self._lock:
                self._latest[cam] = det_result
                self._history.append(det_result)
