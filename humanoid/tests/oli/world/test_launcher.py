"""Unit TDD for the single-entrypoint launcher's backend dispatch + supervisor gating.

`logic/oli/launcher.py` is the ONE command; `--sim` picks a backend plugin, `--mode` picks
the brain behavior. This file tests the wiring the backends themselves don't: two-phase arg
parsing (backend-specific flags stay isolated), the registry, the ordered Stage plan each
backend hands the Supervisor, and the `--dry-run` / reserved-backend paths. Pure stdlib —
runs in the `brain` env, spawns nothing.
"""

import pytest

from humanoid.logic.oli import launcher
from humanoid.logic.oli.launch.backends import REGISTRY, isaac, mujoco, real
from humanoid.logic.oli.launch.supervisor import Stage, Supervisor

pytestmark = pytest.mark.brain


def test_registry_exposes_the_three_worlds():
    assert set(REGISTRY) == {"isaac", "mujoco", "real"}
    assert REGISTRY["isaac"] is isaac and REGISTRY["mujoco"] is mujoco


def test_default_sim_is_isaac():
    args, backend = launcher.build_args(["--mode", "forward"])
    assert args.sim == "isaac" and backend is isaac


def test_sim_selects_backend_and_common_flags_survive():
    args, backend = launcher.build_args(["--sim", "mujoco", "--mode", "walk", "--vx", "0.2"])
    assert backend is mujoco
    assert args.mode == "walk" and args.vx == 0.2
    assert args.socket == "/tmp/oli-world.sock"      # common default


def test_two_phase_parse_isolates_backend_flags():
    # a backend-specific flag reaches args only for the backend that declares it
    a_isaac, _ = launcher.build_args(["--sim", "isaac", "--height-kp", "25"])
    assert a_isaac.height_kp == 25.0
    a_mj, _ = launcher.build_args(["--sim", "mujoco", "--rate", "500"])
    assert a_mj.rate == 500.0
    # …and the WRONG backend rejects it (mujoco has no --height-kp)
    with pytest.raises(SystemExit):
        launcher.build_args(["--sim", "mujoco", "--height-kp", "25"])


def test_unknown_sim_rejected():
    with pytest.raises(SystemExit):
        launcher.build_args(["--sim", "gazebo"])


def test_isaac_fused_glide_dev_app_plan():
    """The headline single command: glide World + dev-app brain, socket-steered."""
    args, backend = launcher.build_args(["--sim", "isaac", "--mode", "glide", "--dev-app"])
    stages = backend.stages(args)
    assert [s.name for s in stages] == ["world", "brain"]
    world, brain = stages
    assert str(isaac._GLIDE_WORLD_ENTRY) in world.argv
    assert world.serving_marker == "serving on" and world.core
    assert brain.argv[brain.argv.index("-m") + 1] == "humanoid.logic.oli.devapp"
    assert brain.argv[brain.argv.index("--mode") + 1] == "glide"
    assert brain.argv[brain.argv.index("--joystick") + 1] == "socket"


def test_fused_glide_cameras_dev_app_the_headline_command():
    """`--sim isaac --mode glide --dev-app --cameras` = drive + live RGBD, one command."""
    args, backend = launcher.build_args(
        ["--sim", "isaac", "--mode", "glide", "--dev-app", "--cameras"])
    assert args.cameras is True
    world, brain = backend.stages(args)
    assert str(isaac._GLIDE_WORLD_ENTRY) in world.argv and "--cameras" in world.argv
    assert "--camera-socket" in brain.argv        # dev app builds IsaacCameraSource → live feed


def test_glide_scale_flag_threads_to_dev_app_brain():
    """`--glide-scale` sets the dev-app GlideAction speed multiplier; default 5.0."""
    # explicit value reaches the brain argv
    args, backend = launcher.build_args(
        ["--sim", "isaac", "--mode", "glide", "--dev-app", "--glide-scale", "4.0"])
    assert args.glide_scale == 4.0
    _, brain = backend.stages(args)
    assert brain.argv[brain.argv.index("--glide-scale") + 1] == "4.0"
    # default is the tuned demo feel (full-stick 1.75 m/s)
    args_def, backend_def = launcher.build_args(
        ["--sim", "isaac", "--mode", "glide", "--dev-app"])
    assert args_def.glide_scale == 5.0
    _, brain_def = backend_def.stages(args_def)
    assert brain_def.argv[brain_def.argv.index("--glide-scale") + 1] == "5.0"


def test_camera_every_flag_threads_to_glide_world():
    """`--camera-every` sets the glide World camera cadence; default 32 (~30 Hz @ 1 kHz)."""
    args, backend = launcher.build_args(
        ["--sim", "isaac", "--mode", "glide", "--dev-app", "--cameras",
         "--camera-every", "16"])
    assert args.camera_every == 16
    world, _ = backend.stages(args)
    assert world.argv[world.argv.index("--camera-every") + 1] == "16"
    # default cadence present when cameras are on
    args_def, backend_def = launcher.build_args(
        ["--sim", "isaac", "--mode", "glide", "--dev-app", "--cameras"])
    assert args_def.camera_every == 32
    world_def, _ = backend_def.stages(args_def)
    assert world_def.argv[world_def.argv.index("--camera-every") + 1] == "32"


def test_mujoco_walk_plan_has_core_bus_plus_pad_extras():
    args, backend = launcher.build_args(["--sim", "mujoco", "--mode", "walk"])
    stages = backend.stages(args)
    names = [s.name for s in stages]
    assert names[:3] == ["sim", "edge", "brain"]          # the 3 core bus procs, in order
    assert all(s.core for s in stages[:3])
    assert stages[1].boot_delay == args.sim_delay          # edge waits for MuJoCo+ELF
    assert "joy-bridge" in names                            # walk → vendor pad + bridge
    bridge = next(s for s in stages if s.name == "joy-bridge")
    assert bridge.core is False                             # closing the pad ≠ killing the sim
    if mujoco.JOYSTICK_BIN.exists():
        assert "joystick" in names


def test_mujoco_rejects_isaac_only_modes():
    for extra in (["--mode", "glide"], ["--dev-app"]):
        args, backend = launcher.build_args(["--sim", "mujoco", *extra])
        with pytest.raises(SystemExit):
            backend.stages(args)


def test_real_backend_reserved_but_raises():
    args, backend = launcher.build_args(["--sim", "real"])
    assert backend is real
    with pytest.raises(SystemExit):
        backend.stages(args)


def test_dry_run_prints_plan_and_spawns_nothing():
    # main() through the supervisor dry-run path returns 0 without launching a process
    assert launcher.main(["--sim", "isaac", "--mode", "forward", "--dry-run"]) == 0


def test_supervisor_dry_run_returns_zero_for_arbitrary_stages():
    from pathlib import Path
    sup = Supervisor(log_path="/tmp/should-not-be-written.log", boot_timeout=1.0)
    stages = [Stage("a", ["echo", "hi"], cwd=Path("/tmp"), serving_marker="up"),
              Stage("b", ["echo", "bye"], cwd=Path("/tmp"), core=False)]
    assert sup.run(stages, dry_run=True) == 0
    assert not Path("/tmp/should-not-be-written.log").exists()
