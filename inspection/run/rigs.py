"""Rig implementations behind the rig contract: q/move/capture/frame/close."""
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from inspection.motion.ik import UR5eIK

log = logging.getLogger(__name__)

ROBOT_IP = "192.168.2.50"


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


class RealRig:
    """UR5e + wrist D405. Camera pipe is owned by a CameraWorker; recv is
    shared with the executor's safety polls behind _recv_lock."""

    def __init__(self, world, stop_event, outdir, ip=ROBOT_IP):
        from inspection.motion.execute import UR5eArm
        from inspection.perception.camera import (
            WRIST_SERIAL, grab_aligned, open_camera, session_metadata,
            t_flange_cam)
        self.world, self.stop_event = world, stop_event
        self.outdir = Path(outdir)
        self.arm = UR5eArm(ip)
        self._recv_lock = threading.Lock()
        # wrap the arm's recv reads: same lock for q() and safety polls
        arm_safety = self.arm._safety_ok
        self.arm._safety_ok = lambda: self._locked(arm_safety)
        try:
            self.pipe, profile, self.align, self.depth_scale = open_camera()
            meta = session_metadata(profile, self.depth_scale, WRIST_SERIAL)
            self.outdir.mkdir(parents=True, exist_ok=True)
            (self.outdir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
            self.intr = meta["intrinsics"]["ir_left"]
            self._T_fc = t_flange_cam()
            self._ik = UR5eIK()
            self._grab = lambda: grab_aligned(self.pipe, self.align)
            self.camera = None
        except Exception:
            if hasattr(self, "pipe"):
                try: self.pipe.stop()
                except Exception: pass
            self.arm.close()
            raise

    def _locked(self, fn):
        with self._recv_lock:
            return fn()

    def q(self):
        with self._recv_lock:
            return self.arm.q()

    def start_camera(self, pub, hz=10.0):
        self.camera = CameraWorker(self._grab, publish=pub.publish_frame, hz=hz)
        self.camera.start()

    def move(self, path):
        return self.arm.execute(path, world=self.world, stop_event=self.stop_event)

    def capture(self, pose_id):
        from inspection.perception.capture import save_bundle
        bundle = self.camera.fresh_bundle(min_new=3) if self.camera \
            else self._grab()
        q = self.q()
        T_bf = self._ik.fk(q)
        pose = {"joints_rad": q, "T_base_flange": T_bf,
                "T_base_cam": T_bf @ self._T_fc}
        d = save_bundle(self.outdir, pose_id, bundle, pose)
        return {"dir": str(d), "rgb": bundle["rgb"],
                "depth_raw": bundle["depth_raw"],
                "T_base_cam": pose["T_base_cam"], "q": q}

    def close(self):
        if self.camera: self.camera.stop()
        try: self.pipe.stop()
        except Exception: pass
        self.arm.close()


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
