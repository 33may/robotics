"""TDD for the ServiceHost (oli/service/host.py) — the one-object service assembly.

`brain_main --service` needs exactly one thing to bolt the seam onto the brain loop: a host
that owns both channels. `poll()` is called every loop iteration (drives W4 goals into Nav);
`recorder` is the Orchestrator's recorder-slot callable (drives W5 telemetry out). This test
runs the FULL client→brain→client loop over real UDS with a fake Nav. `brain` env.
"""

from typing import List, Optional, Tuple

import numpy as np
import pytest

from humanoid.logic.oli import Intent, Mode, Observation, PolicyIn
from humanoid.logic.oli.reason.localization import RobotPose
from humanoid.logic.oli.reason.nav import GoalCoordinate
from humanoid.logic.oli.service import GoalChannelClient, TelemetryClient
from humanoid.logic.oli.service.host import ServiceHost

pytestmark = pytest.mark.brain

N = 31


def _obs(stamp):
    return Observation(
        stamp_ns=stamp, q=np.zeros(N), dq=np.zeros(N), tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _policy_in(stamp):
    return PolicyIn(observation=_obs(stamp),
                    intent=Intent(mode=Mode.WALK, v_x=0.2, v_y=0.0, w_z=0.0))


class FakeNav:
    """Both Nav surfaces the host touches: the goal seam + the observable surface."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Optional[GoalCoordinate]]] = []
        self._goal: Optional[GoalCoordinate] = None
        self.last_pose: Optional[RobotPose] = RobotPose(1, 0.5, 0.5, 0.0)

    def set_goal(self, goal: GoalCoordinate) -> None:
        self.calls.append(("set", goal))
        self._goal = goal

    def clear_goal(self) -> None:
        self.calls.append(("clear", None))
        self._goal = None

    @property
    def goal(self) -> Optional[GoalCoordinate]:
        return self._goal

    @property
    def path(self) -> Optional[List[Tuple[float, float]]]:
        return [(0.5, 0.5), (1.0, 1.0)] if self._goal is not None else None


def test_full_loop_goal_in_telemetry_out(tmp_path):
    goal_sock = str(tmp_path / "goal.sock")
    tel_sock = str(tmp_path / "telemetry.sock")

    nav = FakeNav()
    tel_client = TelemetryClient(tel_sock)
    host = ServiceHost(nav, goal_socket=goal_sock, telemetry_socket=tel_sock)
    goal_client = GoalChannelClient(goal_sock)

    # W4: the evaluator sends a goal; one brain-loop poll lands it in Nav.
    goal_client.send_goal(2.0, 3.0)
    host.poll()
    assert nav.calls == [("set", GoalCoordinate(2.0, 3.0, None))]

    # W5: one brain tick through the recorder slot; the evaluator reads a coherent snapshot.
    host.recorder(_obs(100), _policy_in(100), None, None)
    snap = tel_client.latest()
    assert snap is not None
    assert snap.stamp_ns == 100
    assert snap.goal == GoalCoordinate(2.0, 3.0, None)   # the goal we just sent, echoed back
    assert snap.path == [(0.5, 0.5), (1.0, 1.0)]
    assert snap.pose == (0.5, 0.5, 0.0)
    assert snap.intent == (0.2, 0.0, 0.0)
    assert snap.est is None                              # no localization host attached (§5)

    goal_client.close()
    host.close()
    tel_client.close()


def test_close_releases_the_socket_paths(tmp_path):
    goal_sock = str(tmp_path / "goal.sock")
    nav = FakeNav()
    host = ServiceHost(nav, goal_socket=goal_sock,
                       telemetry_socket=str(tmp_path / "t.sock"))
    host.close()
    # A second host must be able to bind the same path immediately (clean teardown).
    host2 = ServiceHost(nav, goal_socket=goal_sock,
                        telemetry_socket=str(tmp_path / "t.sock"))
    host2.close()


# ── full shadow loop: ServiceHost + LocalizationHost (§5.3) ──────────────────


def test_shadow_loop_lifecycle_over_the_wire(tmp_path):
    import time

    from humanoid.logic.oli.contracts import CameraFrame, CameraIntrinsics
    from humanoid.logic.oli.reason.localization import (
        LocalizationOut,
        LocalizationStatus,
        RobotPose,
    )
    from humanoid.logic.oli.reason.localization.host import LocalizationHost
    from humanoid.logic.oli.service.loc_ctrl import LocCtrlClient

    class Frames:  # latest-wins mailbox shaped like CameraStreamReader
        def __init__(self):
            self.f = None

        def push(self, fr):
            self.f = fr

        def read(self, name):
            return self.f

        def stream_names(self):
            return ["head"] if self.f is not None else []

    class Echo:   # module echoing the warm-start pose (template-style)
        def start(self, setup):
            self.hint = setup.initial_pose

        def step(self, loc_in):
            return LocalizationOut(stamp_ns=loc_in.stamp_ns,
                                   pose=RobotPose(loc_in.stamp_ns, self.hint.x, self.hint.y,
                                                  self.hint.yaw),
                                   status=LocalizationStatus.TRACKING)

        def stop(self):
            pass

    def _wait(cond, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.005)
        return False

    frames = Frames()
    loc_host = LocalizationHost(lambda: Echo(), frames)
    loc_host.start()
    nav = FakeNav()
    tel_client = TelemetryClient(str(tmp_path / "t.sock"))
    host = ServiceHost(nav, goal_socket=str(tmp_path / "g.sock"),
                       telemetry_socket=str(tmp_path / "t.sock"),
                       loc_host=loc_host, loc_ctrl_socket=str(tmp_path / "c.sock"))
    ctrl = LocCtrlClient(str(tmp_path / "c.sock"))

    def tick(stamp):
        host.poll()
        host.recorder(_obs(stamp), _policy_in(stamp), None, None)

    # idle before start
    tick(1)
    assert tel_client.latest().loc_state == "idle"

    # start over the wire → running
    ctrl.send_start(map_dir="/tmp/m", initial_pose=(1.0, 2.0, 0.5))
    assert _wait(lambda: (tick(2), tel_client.latest().loc_state == "running")[1])

    # a frame arrives → shadow est appears on telemetry
    frames.push(CameraFrame(
        stamp_ns=50_000_000, name="head",
        rgb=np.zeros((3, 4, 3), dtype=np.uint8), depth=np.ones((3, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=4, height=3, fx=2.0, fy=2.0, cx=2.0, cy=1.5)))
    assert _wait(lambda: (tick(3), tel_client.latest().est is not None)[1])
    est = tel_client.latest().est
    assert est.pose == RobotPose(50_000_000, 1.0, 2.0, 0.5)   # the warm-start echo

    # stop over the wire → idle, est cleared
    ctrl.send_stop()
    assert _wait(lambda: (tick(4), tel_client.latest().loc_state == "idle")[1])
    assert tel_client.latest().est is None

    ctrl.close()
    host.close()
    tel_client.close()
