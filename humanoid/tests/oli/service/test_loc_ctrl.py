"""TDD for the localization-control channel (oli/service/loc_ctrl.py) — locbench D6/5.3.

The evaluator drives the in-brain host's per-episode lifecycle over a wire: `start(Setup)`
(map_dir + warm-start pose + calibration as JSON) and `stop`. Commands are datagrams into a
brain-side server that ENQUEUES onto the host (a heavy `start()` runs on the host thread,
never the control loop); progress is observed via the telemetry `loc_state` field, not a
reply channel — same one-way latest-wins philosophy as the rest of the seam. `brain` env.
"""

import time
from typing import List

import pytest

from humanoid.logic.oli.reason.localization import LocalizationSetup, RobotPose
from humanoid.logic.oli.service.loc_ctrl import LocCtrlClient, LocCtrlServer
from humanoid.logic.oli.service.protocol import TelemetrySnapshot, decode_telemetry, encode_telemetry

pytestmark = pytest.mark.brain


class FakeHost:
    def __init__(self):
        self.calls: List = []

    def request_start(self, setup: LocalizationSetup) -> None:
        self.calls.append(("start", setup))

    def request_stop(self) -> None:
        self.calls.append(("stop", None))


@pytest.fixture()
def channel(tmp_path):
    path = str(tmp_path / "loc-ctrl.sock")
    host = FakeHost()
    server = LocCtrlServer(path, host)
    client = LocCtrlClient(path)
    yield server, client, host
    client.close()
    server.close()


def _poll_until(server, host, n, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(host.calls) < n:
        server.poll()
        time.sleep(0.005)
    return host.calls


def test_start_carries_the_full_setup(channel):
    server, client, host = channel
    client.send_start(map_dir="/maps/rtab-v1", initial_pose=(1.0, 2.0, 0.5),
                      calibration={"debug_pose_socket": "/tmp/p.sock", "depth": "f32"})
    calls = _poll_until(server, host, 1)
    op, setup = calls[0]
    assert op == "start"
    assert isinstance(setup, LocalizationSetup)
    assert str(setup.map_dir) == "/maps/rtab-v1"
    assert setup.initial_pose == RobotPose(0, 1.0, 2.0, 0.5)
    assert setup.calibration == {"debug_pose_socket": "/tmp/p.sock", "depth": "f32"}


def test_start_without_pose_or_calibration(channel):
    server, client, host = channel
    client.send_start(map_dir="/maps/x")
    calls = _poll_until(server, host, 1)
    setup = calls[0][1]
    assert setup.initial_pose is None and setup.calibration == {}


def test_stop(channel):
    server, client, host = channel
    client.send_stop()
    assert _poll_until(server, host, 1) == [("stop", None)]


def test_malformed_datagram_dropped(channel):
    server, client, host = channel
    client._sock.sendto(b"{not json", client._path)
    client.send_stop()
    assert _poll_until(server, host, 1) == [("stop", None)]


def test_commands_apply_in_order_not_latest_wins(channel):
    # Lifecycle is NOT latest-wins: stop-then-start must both reach the host, in order
    # (unlike goals, where only the newest matters).
    server, client, host = channel
    client.send_stop()
    client.send_start(map_dir="/maps/x")
    calls = _poll_until(server, host, 2)
    assert [c[0] for c in calls] == ["stop", "start"]


# ── telemetry grows loc_state/loc_error (additive) ───────────────────────────


def test_telemetry_carries_loc_state():
    snap = TelemetrySnapshot(stamp_ns=1, loc_state="crashed", loc_error="step() failed: boom")
    out = decode_telemetry(encode_telemetry(snap))
    assert out.loc_state == "crashed" and out.loc_error == "step() failed: boom"


def test_telemetry_loc_fields_default_none():
    out = decode_telemetry(encode_telemetry(TelemetrySnapshot(stamp_ns=1)))
    assert out.loc_state is None and out.loc_error is None
