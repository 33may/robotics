"""TDD for the W4 goal channel (oli/service/goal_channel.py).

Brain-side `GoalChannelServer` binds a UDS datagram path and, on `poll()`, drains the backlog
and applies ONLY the newest valid command to Nav (`set_goal`/`clear_goal`) — latest-wins, one
planner-cache clear per poll no matter how deep the backlog. Malformed datagrams are dropped.
Client-side `GoalChannelClient` is the thin sender any goal source uses (evaluator scripts,
dev_app's click later). Real loopback UDS, stdlib only → `brain` env.
"""

from typing import List, Optional, Tuple

import pytest

from humanoid.logic.oli.reason.nav import GoalCoordinate
from humanoid.logic.oli.service.goal_channel import GoalChannelClient, GoalChannelServer

pytestmark = pytest.mark.brain


class FakeNav:
    """Records the goal-seam calls Nav would receive."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Optional[GoalCoordinate]]] = []

    def set_goal(self, goal: GoalCoordinate) -> None:
        self.calls.append(("set", goal))

    def clear_goal(self) -> None:
        self.calls.append(("clear", None))


@pytest.fixture()
def channel(tmp_path):
    path = str(tmp_path / "goal.sock")
    nav = FakeNav()
    server = GoalChannelServer(path, nav)
    client = GoalChannelClient(path)
    yield server, client, nav
    client.close()
    server.close()


def test_set_goal_reaches_nav(channel):
    server, client, nav = channel
    client.send_goal(2.0, 3.0, 0.5)
    assert server.poll() == GoalCoordinate(2.0, 3.0, 0.5)
    assert nav.calls == [("set", GoalCoordinate(2.0, 3.0, 0.5))]


def test_clear_goal_reaches_nav(channel):
    server, client, nav = channel
    client.clear_goal()
    server.poll()
    assert nav.calls == [("clear", None)]


def test_latest_wins_one_nav_call_per_poll(channel):
    server, client, nav = channel
    client.send_goal(1.0, 1.0)
    client.send_goal(2.0, 2.0)
    client.send_goal(3.0, 3.0, 1.5)
    assert server.poll() == GoalCoordinate(3.0, 3.0, 1.5)
    # A 3-deep backlog must produce exactly ONE set_goal (each call clears the
    # planner's path cache — replaying the backlog would thrash it).
    assert nav.calls == [("set", GoalCoordinate(3.0, 3.0, 1.5))]


def test_malformed_datagrams_are_dropped(channel):
    server, client, nav = channel
    client.send_goal(1.0, 1.0)
    client._sock.sendto(b"garbage", client._path)          # wrong size
    client._sock.sendto(b"\x2a" + b"\x00" * 32, client._path)  # right size, unknown op
    assert server.poll() == GoalCoordinate(1.0, 1.0, None)
    assert nav.calls == [("set", GoalCoordinate(1.0, 1.0, None))]


def test_poll_with_no_traffic_touches_nothing(channel):
    server, client, nav = channel
    assert server.poll() is None
    assert nav.calls == []


def test_yawless_goal_roundtrips_as_none(channel):
    server, client, nav = channel
    client.send_goal(4.0, 5.0)
    server.poll()
    assert nav.calls == [("set", GoalCoordinate(4.0, 5.0, None))]


def test_server_rebinds_over_stale_socket_file(tmp_path):
    # A crashed prior run leaves the socket file behind — a new server must bind anyway.
    path = str(tmp_path / "goal.sock")
    nav1, nav2 = FakeNav(), FakeNav()
    s1 = GoalChannelServer(path, nav1)
    s1._sock.close()  # crash: fd gone, file left on disk
    s2 = GoalChannelServer(path, nav2)
    client = GoalChannelClient(path)
    client.send_goal(1.0, 2.0)
    assert s2.poll() == GoalCoordinate(1.0, 2.0, None)
    client.close()
    s2.close()


def test_send_without_server_raises(tmp_path):
    # Goals are commands, not telemetry: a missing brain must be LOUD, not silent.
    client = GoalChannelClient(str(tmp_path / "nobody-home.sock"))
    with pytest.raises(OSError):
        client.send_goal(1.0, 1.0)
    client.close()
