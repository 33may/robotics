"""Rig implementations behind the rig contract: q/move/capture/frame/close."""
import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class FakeRig:
    """Fake rig for testing: stoppable interpolating moves, synthetic captures."""

    def __init__(self, q0, stop_event=None, dt=0.01, speed=3.0):
        self._q = np.asarray(q0, dtype=float)
        self.stop_event = stop_event or threading.Event()
        self.dt, self.speed = dt, speed
        from inspection.tests.synth import synth_capture
        depth, intr, scale, T_bc, _ = synth_capture()
        self._depth, self._T_bc = depth, T_bc
        self.intr, self.depth_scale = intr, scale
        self.moves = []

    def q(self):
        """Return current joint configuration."""
        return self._q.copy()

    def move(self, path):
        """Interpolate through path, stop on stop_event.

        Returns: {"waypoints_done", "s", "final_err_deg", "stopped"}
        """
        self.moves.append(len(path))
        t0, done = time.perf_counter(), 0
        for q_goal in [np.asarray(q, float) for q in path[1:]]:
            while True:
                if self.stop_event.is_set():
                    return {"waypoints_done": done, "s": time.perf_counter() - t0,
                            "final_err_deg": 0.0, "stopped": True}
                delta = q_goal - self._q
                dist = np.abs(delta).max()
                if dist <= self.speed * self.dt:
                    self._q = q_goal.copy()
                    break
                self._q = self._q + delta / dist * self.speed * self.dt
                time.sleep(self.dt)
            done += 1
        return {"waypoints_done": done, "s": time.perf_counter() - t0,
                "final_err_deg": 0.0, "stopped": False}

    def frame(self):
        """Return live 64x64x3 uint8 gradient rgb."""
        rgb = np.zeros((64, 64, 3), np.uint8)
        rgb[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        return rgb

    def capture(self, pose_id):
        """Capture synthetic observation at current pose.

        Returns: {"dir", "rgb", "depth_raw", "T_base_cam", "q"}
        """
        return {"dir": f"(fake {pose_id:03d})", "rgb": self.frame(),
                "depth_raw": self._depth, "T_base_cam": self._T_bc,
                "q": self._q.copy()}

    def close(self):
        """Close rig (no-op for fake)."""
        pass


class CameraWorker(threading.Thread):
    """Daemon thread that grabs frames and publishes rgb, with freshness tracking."""

    def __init__(self, grab, publish=None, hz=10.0):
        """Initialize.

        Args:
            grab: callable() -> dict | None (keys at least "rgb")
            publish: optional callable(rgb) to call each tick
            hz: frame rate
        """
        super().__init__(daemon=True)
        self._grab = grab
        self._publish = publish
        self._hz = hz
        self._should_stop = threading.Event()
        self._cond = threading.Condition()
        self._count = 0
        self._latest = None

    def run(self):
        """Loop: grab, update count/latest, publish (in try/except), sleep."""
        while not self._should_stop.is_set():
            try:
                b = self._grab()
            except Exception as e:
                log.exception(f"Camera grab failed: {e}")
                time.sleep(1.0 / self._hz)
                continue
            if b is not None:
                with self._cond:
                    self._count += 1
                    self._latest = b
                    self._cond.notify_all()
                if self._publish:
                    try:
                        self._publish(b["rgb"])
                    except Exception as e:
                        log.exception(f"Camera publish callback failed: {e}")
            time.sleep(1.0 / self._hz)

    def latest(self):
        """Return (count, latest_bundle) under lock."""
        with self._cond:
            return self._count, self._latest

    def fresh_bundle(self, min_new=3, timeout=3.0):
        """Return first bundle at least min_new grabs after call.

        Raises RuntimeError on timeout.
        """
        with self._cond:
            target = self._count + min_new
            if not self._cond.wait_for(lambda: self._count >= target, timeout=timeout):
                raise RuntimeError("camera produced no fresh frames")
            return self._latest

    def stop(self):
        """Signal stop and wait for thread to exit with timeout."""
        self._should_stop.set()
        self.join(timeout=2.0)


class PoseStreamer(threading.Thread):
    """Daemon thread that publishes pose q_fn() per tick while .active is set."""

    def __init__(self, q_fn, publish_pose, hz=30.0):
        """Initialize.

        Args:
            q_fn: callable() -> np.ndarray (joint config)
            publish_pose: callable(q)
            hz: publish rate
        """
        super().__init__(daemon=True)
        self._q_fn = q_fn
        self._publish_pose = publish_pose
        self._hz = hz
        self.active = threading.Event()
        self._should_stop = threading.Event()

    def run(self):
        """Loop: if active.is_set(), publish q_fn() (in try/except), sleep."""
        while not self._should_stop.is_set():
            if self.active.is_set():
                try:
                    self._publish_pose(self._q_fn())
                except Exception as e:
                    log.exception(f"Pose publish callback failed: {e}")
            time.sleep(1.0 / self._hz)

    def stop(self):
        """Signal stop and wait for thread to exit with timeout."""
        self._should_stop.set()
        self.join(timeout=2.0)
