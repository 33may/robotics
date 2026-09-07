"""Rig implementations behind the rig contract: q/move/capture/frame/close.

`capture(pose_id, out_dir)` writes BYTES ONLY, into the step directory the
run's `RunWriter` allocated. Nothing here writes a record any more: the
session identity (`RealRig.session`) is handed to the writer by the
composition root, and the per-capture facts go into `steps/NNN/step.json`.
"""
import logging
import threading
import time
from pathlib import Path

import numpy as np

from inspection.cell.geometry import is_half_turn, rotate180
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
        # The synthetic camera has one imager, so the colour view and the
        # depth view coincide; on the D405 they differ by 2.6 mm (measured).
        # Same key, so the settle leg needs no branch for the fake.
        self.intr_color = intr
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

    def capture(self, pose_id, out_dir=None):
        """Capture synthetic observation at current pose.

        `out_dir` is the step directory the run's writer handed out. With one,
        the bytes are written exactly as the real rig writes them (rotation
        policy included) so the mock exercises every reader keyed on capture
        files — view images, chain overlays, the image tier. Without one
        (unit tests), the capture is memory-only and reports a dir that says
        so. No meta.json either way: the record is the writer's job.

        Returns the rig capture contract: {"dir", "rgb", "pose_id", "t",
        "depth_raw", "depth_aligned", "T_base_cam", "q", "rgb_rotation_deg"}.
        """
        # rgb must be frame-sized here, not the 64x64 preview `frame()`
        # returns: the settle leg segments it against `depth_aligned`, and a
        # mask has to match the depth it is applied to.
        h, w = self._depth.shape
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:, :, 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        rot = 180 if is_half_turn(self._T_bc) else 0
        d = f"(fake {pose_id:03d})"
        if out_dir is not None:
            import cv2
            real = Path(out_dir)
            real.mkdir(parents=True, exist_ok=True)
            stored = rotate180(rgb) if rot else rgb
            cv2.imwrite(str(real / "rgb.png"),
                        cv2.cvtColor(stored, cv2.COLOR_RGB2BGR))
            np.save(real / "depth_aligned.npy",
                    rotate180(self._depth) if rot else self._depth)
            d = str(real)
        return {"dir": d, "rgb": rgb, "pose_id": pose_id, "t": time.time(),
                "depth_raw": self._depth, "depth_aligned": self._depth,
                "T_base_cam": self._T_bc, "q": self._q.copy(),
                "rgb_rotation_deg": rot}

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
        self._grab_fails = 0        # consecutive; drives the log rate limit

    def run(self):
        """Loop: grab, update count/latest, publish (in try/except), sleep."""
        while not self._should_stop.is_set():
            try:
                b = self._grab()
            except Exception:
                # Rate-limited: a camera that has come unplugged fails every
                # tick, and 10 tracebacks a second bury the rest of the run's
                # log. First failure gets the traceback (that is the one that
                # says what broke); after that, one counted line per ~5 s.
                self._grab_fails += 1
                if self._grab_fails == 1:
                    log.exception("camera grab failed")
                elif self._grab_fails % 50 == 0:
                    log.warning("camera grab still failing (%d consecutive)",
                                self._grab_fails)
                time.sleep(1.0 / self._hz)
                continue
            if self._grab_fails:
                log.info("camera grab recovered after %d failures", self._grab_fails)
                self._grab_fails = 0
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
        if self.is_alive():
            # Only reachable if grab() is wedged inside the driver: the pipe is
            # about to be stopped under a thread still reading it, and that is
            # worth a line in the shutdown log rather than a silent return.
            log.warning("CameraWorker did not exit within 2.0 s "
                        "(grab() blocked?) — closing the pipe anyway")


class RealRig:
    """UR5e + wrist D405. Camera pipe is owned by a CameraWorker; recv is
    shared with the executor's q()/safety polls behind _recv_lock — every
    RTDEReceiveInterface read (RealRig.q(), execute()'s internal q()/
    _safety_ok()) goes through the same lock, one acquisition each."""

    def __init__(self, world, stop_event, outdir, ip=ROBOT_IP):
        from inspection.motion.execute import UR5eArm
        from inspection.perception.camera import (
            WRIST_SERIAL, grab_aligned, open_camera, session_metadata,
            t_flange_cam)
        from inspection.record.schema import SessionRecord
        self.world, self.stop_event = world, stop_event
        self.outdir = Path(outdir)
        self.arm = UR5eArm(ip)
        self._recv_lock = threading.Lock()
        # wrap the arm's recv reads: same lock for q() and safety polls, so
        # execute()'s own self.q()/self._safety_ok() calls (start-tolerance
        # check, final-error calc, safety polls) serialize against
        # RealRig.q() (PoseStreamer, capture()) on the same RTDEReceiveInterface
        arm_safety = self.arm._safety_ok
        self.arm._safety_ok = lambda: self._locked(arm_safety)
        arm_q = self.arm.q
        self.arm.q = lambda: self._locked(arm_q)
        try:
            self.pipe, profile, self.align, self.depth_scale = open_camera()
            meta = session_metadata(profile, self.depth_scale, WRIST_SERIAL)
            #: Camera identity for the record. Built here — this is where the
            #: camera is opened and the only place the numbers exist — but
            #: NOT written here: the composition root hands it to the run's
            #: writer (`RunWriter.write_session`), which owns every JSON in a
            #: run directory.
            self.session = SessionRecord.model_validate(meta)
            self.intr = meta["intrinsics"]["ir_left"]
            # `depth_aligned` is warped into the COLOUR viewport, which is the
            # frame the rgb mask is computed in — so the masked lift uses this
            # pair and the mask needs no warping at all.
            self.intr_color = meta["intrinsics"]["color"]
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
        # self.arm.q is already the locked wrapper installed in __init__ —
        # don't re-acquire here, _recv_lock is a plain (non-reentrant) Lock
        return self.arm.q()

    def start_camera(self, pub, hz=10.0):
        self.camera = CameraWorker(self._grab, publish=pub.publish_frame, hz=hz)
        self.camera.start()

    def move(self, path):
        return self.arm.execute(path, world=self.world, stop_event=self.stop_event)

    def capture(self, pose_id, out_dir=None):
        """Grab a bundle and write its bytes into `out_dir` (the step dir).

        Falls back to `{outdir}/{pose_id:03d}` only for a rig driven outside
        a recorded run; no meta.json is written either way.
        """
        from inspection.perception.capture import save_bundle
        bundle = self.camera.fresh_bundle(min_new=3) if self.camera \
            else self._grab()
        q = self.q()
        T_bf = self._ik.fk(q)
        pose = {"joints_rad": q, "T_base_flange": T_bf,
                "T_base_cam": T_bf @ self._T_fc}
        d = save_bundle(self.outdir, pose_id, bundle, pose,
                        write_meta=False, dest=out_dir)
        return {"dir": str(d), "rgb": bundle["rgb"], "pose_id": pose_id,
                "t": bundle["timestamp"],
                "depth_raw": bundle["depth_raw"],
                # RAW orientation, like everything else returned here:
                # `save_bundle` rotates only what it writes to disk.
                "depth_aligned": bundle["depth_aligned"],
                "T_base_cam": pose["T_base_cam"], "q": q,
                # What `save_bundle` just did to rgb.png/depth_aligned.npy —
                # same rule, one decision, so the record cannot disagree with
                # the pixels.
                "rgb_rotation_deg": 180 if is_half_turn(pose["T_base_cam"]) else 0}

    def close(self):
        if self.camera: self.camera.stop()
        try: self.pipe.stop()
        except Exception: pass
        self.arm.close()


class PoseStreamer(threading.Thread):
    """Daemon thread that publishes pose q_fn() per tick while .active is set.

    `quiesce()` is the other half of the pose-producer handoff: the publisher
    caches pinocchio geom_data, and `Supervisor._recenter()` calls
    `world.set_object()`, which resizes the geometry model. A publish that
    overlaps that resize is an out-of-bounds write in C++, not an exception.
    Clearing `active` alone does not prevent the overlap — a tick already
    inside `_publish_pose` keeps running — so the check-and-mark below is
    atomic under `_cond`, and `quiesce()` returns only once `active` is clear
    AND no publish is in flight.
    """

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
        self._cond = threading.Condition()
        self._publishing = False

    def run(self):
        """Loop: if active.is_set(), publish q_fn() (in try/except), sleep."""
        while not self._should_stop.is_set():
            # Deciding to publish and announcing it happen under one lock, so
            # a quiesce() that observes "not publishing" cannot be overtaken
            # by a tick that had already passed the `active` check.
            with self._cond:
                publishing = self._publishing = self.active.is_set()
            if publishing:
                try:
                    self._publish_pose(self._q_fn())
                except Exception as e:
                    log.exception(f"Pose publish callback failed: {e}")
                finally:
                    with self._cond:
                        self._publishing = False
                        self._cond.notify_all()
            time.sleep(1.0 / self._hz)

    def quiesce(self, timeout=1.0):
        """Block until no publish is in flight and `active` is clear.

        Returns True if that state was reached, False on timeout (the caller
        logs and proceeds — this is a safety interlock, not a gate that may
        wedge a run with a robot in it). The waiter is woken by every publish
        completing, so the predicate is re-tested at the streamer's own rate.
        """
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._publishing and not self.active.is_set(),
                timeout)

    def stop(self):
        """Signal stop and wait for thread to exit with timeout."""
        self._should_stop.set()
        self.join(timeout=2.0)
        if self.is_alive():
            log.warning("PoseStreamer did not exit within 2.0 s "
                        "(publish blocked?)")
