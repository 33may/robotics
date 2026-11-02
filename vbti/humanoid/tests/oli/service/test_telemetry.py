"""TDD for the W5 telemetry channel (oli/service/telemetry.py).

Two layers: the transport (`TelemetryServer` = brain-side fire-and-forget sender,
`TelemetryClient` = reader that binds and drains latest-wins — the debug_pose pattern with a
JSON payload) and the assembly (`TelemetryPublisher` — the callable that sits in the
Orchestrator's `recorder` slot and turns each tick's (obs, policy_in) + Nav's observable
state + the localization host's latest verdict into one published `TelemetrySnapshot`).
Real loopback UDS, stdlib+numpy only → `brain` env.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import pytest

from humanoid.logic.oli import Intent, Mode, Observation, PolicyIn
from humanoid.logic.oli.reason.localization import (
    LocalizationOut,
    LocalizationStatus,
    RobotPose,
)
from humanoid.logic.oli.reason.nav import GoalCoordinate
from humanoid.logic.oli.service.protocol import TelemetrySnapshot
from humanoid.logic.oli.service.telemetry import (
    TelemetryClient,
    TelemetryPublisher,
    TelemetryServer,
)

pytestmark = pytest.mark.brain

N = 31


def _obs(stamp):
    return Observation(
        stamp_ns=stamp, q=np.zeros(N), dq=np.zeros(N), tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _policy_in(stamp, vx=0.3, wz=0.1):
    return PolicyIn(observation=_obs(stamp),
                    intent=Intent(mode=Mode.WALK, v_x=vx, v_y=0.0, w_z=wz))


# ── transport (real loopback UDS) ────────────────────────────────────────────


def _drain_until(client: TelemetryClient, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = client.latest()
        if snap is not None:
            return snap
        time.sleep(0.005)
    raise AssertionError("no telemetry arrived within timeout")


def test_publish_reaches_client(tmp_path):
    path = str(tmp_path / "telemetry.sock")
    client = TelemetryClient(path)
    server = TelemetryServer(path)
    snap = TelemetrySnapshot(stamp_ns=42, pose=(1.0, 2.0, 0.5),
                             path=[(1.0, 2.0), (2.0, 3.0)],
                             goal=GoalCoordinate(2.0, 3.0, None),
                             intent=(0.3, 0.0, 0.1), loop_hz=99.0)
    server.publish(snap)
    assert _drain_until(client) == snap
    server.close()
    client.close()


def test_latest_wins(tmp_path):
    path = str(tmp_path / "telemetry.sock")
    client = TelemetryClient(path)
    server = TelemetryServer(path)
    for stamp in (1, 2, 3):
        server.publish(TelemetrySnapshot(stamp_ns=stamp))
    assert _drain_until(client).stamp_ns == 3
    server.close()
    client.close()


def test_publish_without_reader_is_silent(tmp_path):
    # Telemetry is best-effort by design: the brain loop must never stall or crash
    # because nobody is listening.
    server = TelemetryServer(str(tmp_path / "nobody-home.sock"))
    server.publish(TelemetrySnapshot(stamp_ns=1))
    server.close()


def test_client_skips_malformed_datagram(tmp_path):
    path = str(tmp_path / "telemetry.sock")
    client = TelemetryClient(path)
    server = TelemetryServer(path)
    server._sock.sendto(b"not json at all", path)
    server.publish(TelemetrySnapshot(stamp_ns=7))
    assert _drain_until(client).stamp_ns == 7
    server.close()
    client.close()


# ── publisher (the recorder-slot assembly) ───────────────────────────────────


class FakeNav:
    """Nav's observable surface: path/goal/last_pose read-only properties."""

    def __init__(self) -> None:
        self.path: Optional[List[Tuple[float, float]]] = [(0.0, 0.0), (1.0, 1.0)]
        self.goal: Optional[GoalCoordinate] = GoalCoordinate(1.0, 1.0, None)
        self.last_pose: Optional[RobotPose] = RobotPose(10, 0.5, 0.5, 0.25)


class CaptureServer:
    def __init__(self) -> None:
        self.published: List[TelemetrySnapshot] = []

    def publish(self, snap: TelemetrySnapshot) -> None:
        self.published.append(snap)


def test_publisher_assembles_the_tick():
    nav = FakeNav()
    server = CaptureServer()
    est = LocalizationOut(stamp_ns=9, pose=RobotPose(9, 0.4, 0.6, 0.2),
                          status=LocalizationStatus.TRACKING)
    pub = TelemetryPublisher(server, nav, est_source=lambda: est)
    pub(_obs(100), _policy_in(100, vx=0.3, wz=0.1), None, None)

    assert len(server.published) == 1
    snap = server.published[0]
    assert snap.stamp_ns == 100
    assert snap.pose == (0.5, 0.5, 0.25)
    assert snap.path == [(0.0, 0.0), (1.0, 1.0)]
    assert snap.goal == GoalCoordinate(1.0, 1.0, None)
    assert snap.est == est
    assert snap.intent == (0.3, 0.0, 0.1)


def test_publisher_handles_the_empty_brain():
    # Boot state: no goal, no path, no pose yet, no localization host attached.
    nav = FakeNav()
    nav.path = None
    nav.goal = None
    nav.last_pose = None
    server = CaptureServer()
    pub = TelemetryPublisher(server, nav)  # est_source defaults to "no host"
    pub(_obs(5), _policy_in(5), None, None)
    snap = server.published[0]
    assert snap.pose is None and snap.path is None and snap.goal is None
    assert snap.est is None


def test_publisher_measures_loop_rate():
    nav = FakeNav()
    server = CaptureServer()
    pub = TelemetryPublisher(server, nav)
    pub(_obs(1), _policy_in(1), None, None)
    assert server.published[0].loop_hz is None  # one sample = no rate yet
    for stamp in range(2, 12):
        time.sleep(0.002)
        pub(_obs(stamp), _policy_in(stamp), None, None)
    hz = server.published[-1].loop_hz
    assert hz is not None and 0.0 < hz < 500.0  # wall-clock rate, sane bounds
