"""TDD for the in-brain localization host (reason/localization/host.py) — locbench D6.

The brain owns its senses: the host consumes the camera stream on a SIDE thread, assembles
frame-paced `LocalizationIn` bundles (newest frames + latest in-process obs/intent), runs
`module.step()` there, and publishes the latest `LocalizationOut` for telemetry / the
Stage-2 Localizer adapter. Guarantees under test:

  - the control-loop surface (`on_tick`, `latest`, `request_*`) never blocks on a slow module;
  - a slow module means SKIPPED frames (latest-wins), never a queue, never a stall;
  - lifecycle commands (start/stop) execute on the host thread — a slow `start()` cannot
    stall the control loop either;
  - a raising/contract-breaking module marks the episode `crashed`, host survives;
  - warm start: each `start` builds a FRESH module instance (no state leaks across episodes).

`brain` env (threads + fakes, no sockets).
"""

import threading
import time
from typing import List, Optional

import numpy as np
import pytest

from humanoid.logic.oli.contracts import CameraFrame, CameraIntrinsics, Intent, Mode, Observation
from humanoid.logic.oli.reason.localization import (
    LocalizationOut,
    LocalizationSetup,
    LocalizationStatus,
    RobotPose,
)
from humanoid.logic.oli.reason.localization.host import LocalizationHost

pytestmark = pytest.mark.brain

MS = 1_000_000


def _obs(stamp_ns):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.zeros(3, dtype=np.float32), gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
    )


def _frame(stamp_ns, name="head"):
    return CameraFrame(
        stamp_ns=stamp_ns, name=name,
        rgb=np.zeros((3, 4, 3), dtype=np.uint8), depth=np.ones((3, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=4, height=3, fx=2.0, fy=2.0, cx=2.0, cy=1.5),
    )


_SETUP = LocalizationSetup(map_dir="/tmp/nomap")


class FakeFrames:
    """Latest-wins mailbox shaped like CameraStreamReader."""

    def __init__(self):
        self._latest = {}
        self._lock = threading.Lock()
        self.published = 0

    def push(self, frame):
        with self._lock:
            self._latest[frame.name] = frame
            self.published += 1

    def read(self, name):
        with self._lock:
            return self._latest.get(name)

    def stream_names(self):
        with self._lock:
            return list(self._latest)


class FakeModule:
    """Programmable module: optional per-step delay, optional exception, GIL-friendly."""

    instances = 0

    def __init__(self, step_delay=0.0, raise_at=None):
        FakeModule.instances += 1
        self.step_delay = step_delay
        self.raise_at = raise_at
        self.loc_ins: List = []
        self.started: Optional[LocalizationSetup] = None
        self.stopped = False

    def start(self, setup):
        self.started = setup

    def step(self, loc_in):
        self.loc_ins.append(loc_in)
        if self.raise_at is not None and len(self.loc_ins) >= self.raise_at:
            raise RuntimeError("segv-ish")
        if self.step_delay:
            time.sleep(self.step_delay)
        return LocalizationOut(
            stamp_ns=loc_in.stamp_ns,
            pose=RobotPose(loc_in.stamp_ns, 1.0, 2.0, 0.1),
            status=LocalizationStatus.TRACKING,
        )

    def stop(self):
        self.stopped = True


def _wait(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.002)
    return False


@pytest.fixture()
def rig():
    frames = FakeFrames()
    made: List[FakeModule] = []

    def factory(**kw):
        def f():
            m = FakeModule(**kw)
            made.append(m)
            return m
        return f

    hosts: List[LocalizationHost] = []

    def build(**kw):
        h = LocalizationHost(factory(**kw), frames)
        h.start()
        hosts.append(h)
        return h

    yield frames, build, made
    for h in hosts:
        h.close()


def test_frame_paced_step_with_latest_obs_and_intent(rig):
    frames, build, made = rig
    host = build()
    host.request_start(_SETUP)
    assert _wait(lambda: host.state == "running")

    host.on_tick(_obs(90), Intent(mode=Mode.WALK, v_x=0.3))
    host.on_tick(_obs(100), Intent(mode=Mode.WALK, v_x=0.4))
    frames.push(_frame(105))
    assert _wait(lambda: host.latest() is not None)

    out = host.latest()
    assert out.pose == RobotPose(105, 1.0, 2.0, 0.1)
    loc_in = made[0].loc_ins[0]
    assert loc_in.stamp_ns == 105                    # frame-paced
    assert loc_in.observation.stamp_ns == 100        # latest obs rode along
    assert loc_in.intent.v_x == pytest.approx(0.4)   # latest intent rode along


def test_no_obs_yet_means_no_step(rig):
    frames, build, made = rig
    host = build()
    host.request_start(_SETUP)
    assert _wait(lambda: host.state == "running")
    frames.push(_frame(10))
    time.sleep(0.1)
    assert made[0].loc_ins == []      # LocalizationIn requires an Observation — skip, no crash


def test_slow_module_skips_frames_and_never_blocks_the_control_side(rig):
    frames, build, made = rig
    host = build(step_delay=0.05)     # 20 Hz module vs ~100 Hz frames
    host.request_start(_SETUP)
    assert _wait(lambda: host.state == "running")
    host.on_tick(_obs(1), None)

    t_control = []
    for i in range(30):
        frames.push(_frame((i + 1) * 10 * MS))
        t0 = time.monotonic()
        host.on_tick(_obs((i + 1) * 10 * MS), None)   # the control loop's call
        host.latest()
        t_control.append(time.monotonic() - t0)
        time.sleep(0.01)
    assert _wait(lambda: len(made[0].loc_ins) >= 2)

    assert max(t_control) < 0.02, "control-side call blocked on the slow module"
    assert len(made[0].loc_ins) < frames.published   # frames were SKIPPED, not queued
    stamps = [li.stamp_ns for li in made[0].loc_ins]
    assert stamps == sorted(stamps)                  # and what ran stayed monotonic


def test_equal_stamp_multi_stream_frames_all_reach_the_module(rig):
    """A shared stamp watermark must not starve the stream written LAST.

    The real race (cuvslam run 20260714-130101): the World publishes chest@t, the host
    polls in the gap and steps a chest-only bundle, then head@t lands with the SAME
    stamp — `head.stamp_ns > last_frame_stamp` is False forever after, and the module
    never sees another head frame. Watermarks must be per-stream.
    """
    frames, build, made = rig
    host = build()
    host.request_start(_SETUP)
    assert _wait(lambda: host.state == "running")
    host.on_tick(_obs(1), None)

    for tick in range(1, 4):
        t = tick * 10 * MS
        frames.push(_frame(t, name="chest"))          # chest written first...
        steps_before = host.steps
        assert _wait(lambda: host.steps > steps_before)   # ...host consumed chest-only bundle
        frames.push(_frame(t, name="head"))           # ...head lands late, EQUAL stamp
        assert _wait(lambda: any(
            "head" in li.frames and li.frames["head"].stamp_ns == t
            for li in made[0].loc_ins)), f"head@{t} starved by the shared watermark"


def test_crashing_module_marks_crashed_and_host_survives(rig):
    frames, build, made = rig
    host = build(raise_at=2)
    host.request_start(_SETUP)
    assert _wait(lambda: host.state == "running")
    host.on_tick(_obs(1), None)
    frames.push(_frame(10 * MS))
    assert _wait(lambda: made[0].loc_ins)
    frames.push(_frame(20 * MS))
    assert _wait(lambda: host.state == "crashed")
    assert "segv-ish" in host.last_error
    assert made[0].stopped                     # teardown attempted
    # host thread is alive and accepts the next episode
    host.request_start(_SETUP)
    assert _wait(lambda: host.state == "running")
    assert len(made) == 2                      # FRESH module instance per start


def test_stop_clears_latest_and_returns_to_idle(rig):
    frames, build, made = rig
    host = build()
    host.request_start(_SETUP)
    assert _wait(lambda: host.state == "running")
    host.on_tick(_obs(1), None)
    frames.push(_frame(10 * MS))
    assert _wait(lambda: host.latest() is not None)
    host.request_stop()
    assert _wait(lambda: host.state == "idle")
    assert made[0].stopped
    assert host.latest() is None               # last episode's pose must not leak into the next


def test_slow_start_does_not_block_control_side(rig):
    frames, build, made = rig

    class SlowStart(FakeModule):
        def start(self, setup):
            time.sleep(0.3)
            super().start(setup)

    host = LocalizationHost(lambda: SlowStart(), frames)
    host.start()
    try:
        t0 = time.monotonic()
        host.request_start(_SETUP)             # must only ENQUEUE
        assert time.monotonic() - t0 < 0.05
        assert host.state in ("starting", "idle")
        assert _wait(lambda: host.state == "running")
    finally:
        host.close()


def test_contract_breaking_output_crashes_the_episode(rig):
    frames, build, made = rig

    class BadOut(FakeModule):
        def step(self, loc_in):
            super().step(loc_in)
            return "not a LocalizationOut"

    host = LocalizationHost(lambda: BadOut(), frames)
    host.start()
    try:
        host.request_start(_SETUP)
        assert _wait(lambda: host.state == "running")
        host.on_tick(_obs(1), None)
        frames.push(_frame(10 * MS))
        assert _wait(lambda: host.state == "crashed")
        assert "LocalizationOut" in host.last_error
    finally:
        host.close()


# ── HostLocalizer: the Stage-2 seam (dormant in Stage 1) ─────────────────────


def test_host_localizer_unwraps_the_latest_pose(rig):
    from humanoid.logic.oli.reason.localization.host import HostLocalizer

    frames, build, made = rig
    host = build()
    loc = HostLocalizer(host)
    assert loc.estimate(_obs(1)) is None          # nothing yet → Nav holds

    host.request_start(_SETUP)
    assert _wait(lambda: host.state == "running")
    host.on_tick(_obs(1), None)
    frames.push(_frame(10 * MS))
    assert _wait(lambda: host.latest() is not None)
    pose = loc.estimate(_obs(2))
    assert pose == RobotPose(10 * MS, 1.0, 2.0, 0.1)


def test_host_localizer_returns_none_on_lost(rig):
    from humanoid.logic.oli.reason.localization.host import HostLocalizer

    class LostModule(FakeModule):
        def step(self, loc_in):
            super().step(loc_in)
            return LocalizationOut(stamp_ns=loc_in.stamp_ns, pose=None,
                                   status=LocalizationStatus.LOST)

    frames, _, _ = rig
    host = LocalizationHost(lambda: LostModule(), frames)
    host.start()
    try:
        host.request_start(_SETUP)
        assert _wait(lambda: host.state == "running")
        host.on_tick(_obs(1), None)
        frames.push(_frame(10 * MS))
        assert _wait(lambda: host.steps >= 1)
        assert HostLocalizer(host).estimate(_obs(2)) is None   # LOST → hold, never a stale pose
    finally:
        host.close()
