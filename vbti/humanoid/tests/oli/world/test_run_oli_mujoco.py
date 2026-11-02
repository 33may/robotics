"""Unit TDD for the MuJoCo launcher backend's command routing.

The single entrypoint is now `logic/oli/launcher.py`; the mujoco-specific argv/stage logic
lives in `logic/oli/launch/backends/mujoco.py` (imported here as `r`). It is pure stdlib
orchestration (no limxsdk import), so it runs in the `brain` env. We test the real logic:
which flags route to the sim / edge / brain `conda run` argv lists, and that all three
stream live (no buffering to EOF). (Cross-backend dispatch lives in test_launcher.py.)
"""

import argparse

import pytest

from humanoid.logic.oli.launch.backends import mujoco as r

pytestmark = pytest.mark.brain


def _args(**over):
    base = dict(
        socket="/tmp/s.sock", log="/tmp/l.log", boot_timeout=120.0, ip="127.0.0.1",
        rate=1000.0, watchdog_ms=200.0, sim_delay=3.0,
        mode="forward", vx=None, vy=None, wz=None, joy_port=9001,
        spawn_app=False, walk_after=None, duration=0.0,
        sim_env="limx", brain_env="brain",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_all_children_stream_live():
    for argv in (r.sim_argv(_args()), r.edge_argv(_args()), r.brain_argv(_args()),
                 r.bridge_argv(_args())):
        assert argv[:2] == ["conda", "run"]
        assert "--no-capture-output" in argv
        assert "-u" in argv[argv.index("python"):]


def test_bridge_runs_in_limx_forwarding_to_brain_joyport():
    argv = r.bridge_argv(_args(joy_port=9001))
    assert argv[3:5] == ["-n", "limx"]
    assert str(r.BRIDGE_ENTRY) in argv
    assert argv[argv.index("--joy-port") + 1] == "9001"
    assert argv[argv.index("--robot-ip") + 1] == "127.0.0.1"


def test_joystick_argv_is_the_vendor_binary():
    argv = r.joystick_argv(_args())
    assert argv == [str(r.JOYSTICK_BIN)]
    assert argv[0].endswith("robot-joystick")


def test_walk_brain_never_spawns_an_app():
    # the orchestrator owns the vendor pad now; the brain must NOT spawn an app itself
    assert "--spawn-app" not in r.brain_argv(_args(mode="walk", spawn_app=True))


def test_sim_runs_in_limx_no_deploy_no_policy_flags():
    argv = r.sim_argv(_args())
    assert argv[3:5] == ["-n", "limx"]
    assert str(r.SIM_ENTRY) in argv
    # sim gets the robot ip; locomotion/socket are not its business
    assert "127.0.0.1" in argv
    assert "--vx" not in argv and "--socket" not in argv


def test_edge_runs_in_limx_with_socket_and_bus_ip():
    argv = r.edge_argv(_args(rate=500.0, watchdog_ms=150.0))
    assert argv[3:5] == ["-n", "limx"]
    assert str(r.EDGE_ENTRY) in argv
    assert argv[argv.index("--socket") + 1] == "/tmp/s.sock"
    assert argv[argv.index("--robot-ip") + 1] == "127.0.0.1"
    assert argv[argv.index("--rate") + 1] == "500.0"
    assert argv[argv.index("--watchdog-ms") + 1] == "150.0"


def test_brain_runs_in_brain_env_with_locomotion():
    argv = r.brain_argv(_args(mode="forward", vx=0.3))
    assert argv[3:5] == ["-n", "brain"]
    assert str(r.BRAIN_ENTRY) in argv
    assert argv[argv.index("--mode") + 1] == "walk"  # forward → walk policy, fixed cmd
    assert argv[argv.index("--joystick") + 1] == "fixed"
    assert argv[argv.index("--vx") + 1] == "0.3"


def test_mode_presets_select_brain_behavior():
    stand = r.brain_argv(_args(mode="stand"))
    assert stand[stand.index("--mode") + 1] == "stand"
    assert stand[stand.index("--joystick") + 1] == "fixed"
    assert "--vx" not in stand

    walk = r.brain_argv(_args(mode="walk"))
    assert walk[walk.index("--mode") + 1] == "walk"
    assert walk[walk.index("--joystick") + 1] == "socket"
    assert "--joy-port" in walk

    forward = r.brain_argv(_args(mode="forward"))  # no explicit --vx
    assert forward[forward.index("--vx") + 1] == str(r._FORWARD_VX)


def test_socket_shared_sim_has_none():
    a = _args()
    assert r.edge_argv(a)[r.edge_argv(a).index("--socket") + 1] == "/tmp/s.sock"
    b = r.brain_argv(a)
    assert b[b.index("--socket") + 1] == "/tmp/s.sock"


def test_duration_routed_to_brain_only_when_set():
    assert "--duration" not in r.brain_argv(_args(duration=0.0))
    b = r.brain_argv(_args(duration=15.0))
    assert b[b.index("--duration") + 1] == "15.0"


def test_walk_after_and_joyport_optional():
    assert "--walk-after" not in r.brain_argv(_args(walk_after=None))
    assert r.brain_argv(_args(walk_after=3.0))[
        r.brain_argv(_args(walk_after=3.0)).index("--walk-after") + 1] == "3.0"
    assert "--joy-port" not in r.brain_argv(_args(mode="forward"))
    assert "--joy-port" in r.brain_argv(_args(mode="walk"))
