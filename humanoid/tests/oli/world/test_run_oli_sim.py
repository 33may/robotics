"""Unit TDD for the ISAAC launcher backend's command routing.

The single entrypoint is now `logic/oli/launcher.py`; the isaac-specific argv/stage logic
lives in `logic/oli/launch/backends/isaac.py` (imported here as `r`). It is pure stdlib
orchestration (no isaacsim import), so it runs in the `brain` env. We test the part with
real logic: which CLI flags get routed to the World process vs the brain process, and that
`--mode glide` swaps in the glide World. (Cross-backend dispatch lives in test_launcher.py.)
"""

import argparse

import pytest

from humanoid.logic.oli.launch.backends import isaac as r

pytestmark = pytest.mark.brain


def _args(**over):
    base = dict(
        socket="/tmp/s.sock", log="/tmp/l.log", boot_timeout=240.0,
        headless=False, pace="lockstep", decimation=10, spawn_height=1.1,
        render_every=16, pin_root=False,
        control="implicit", armature=False, solver_vel_iters=0, solver_pos_iters=0,
        ground_friction=1.0, ground_restitution=0.0, ankle_effort=42.0, ankle_vel=13.6,
        ankle_kp_scale=1.0, ankle_roll_scale=1.0, waist_kp_scale=1.0, ankle_parallel=False,
        mode="forward", vx=None, vy=None, wz=None, joy_port=9001,
        spawn_app=False, walk_after=None, duration=0.0, dev_app=False,
        hold_kp=80.0, hold_kd=4.0, height_kp=20.0, foot_clearance=0.10,
        lin_accel=10.0, yaw_accel=20.0, glide_scale=5.0,
        world_env="isaac", brain_env="brain", scene="none",
        cameras=False, camera_socket="/tmp/oli-world-frames.sock", camera_res=[1280, 720],
        camera_every=32,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_conda_run_streams_live_output():
    """Both children must use --no-capture-output + python -u, else logs buffer to EOF."""
    for argv in (r.world_argv(_args()), r.brain_argv(_args())):
        assert "--no-capture-output" in argv
        assert argv[:2] == ["conda", "run"]
        # `python -u` (unbuffered) so the tee sees lines as they're printed
        assert "-u" in argv[argv.index("python"):]


def test_world_gets_sim_flags_not_locomotion():
    argv = r.world_argv(_args(headless=True, pace="free", decimation=8))
    assert argv[3:5] == ["-n", "isaac"]
    assert str(r._WORLD_ENTRY) in argv
    assert "--headless" in argv
    assert argv[argv.index("--pace") + 1] == "free"
    assert argv[argv.index("--decimation") + 1] == "8"
    # locomotion command is the brain's business, never the World's
    assert "--vx" not in argv


def test_brain_gets_locomotion_not_sim_flags():
    argv = r.brain_argv(_args(mode="forward", vx=0.3))
    assert argv[3:5] == ["-n", "brain"]
    assert str(r._BRAIN_ENTRY) in argv
    assert argv[argv.index("--vx") + 1] == "0.3"
    assert argv[argv.index("--mode") + 1] == "walk"  # forward → walk policy, fixed cmd
    assert argv[argv.index("--joystick") + 1] == "fixed"
    # sim-only knobs never leak to the brain
    assert "--headless" not in argv
    assert "--spawn-height" not in argv


def test_dev_app_routes_brain_to_devapp_module():
    """--dev-app runs the windowed dev app as the brain (same socket + locomotion flags)."""
    default = r.brain_argv(_args(mode="forward", vx=0.3))
    assert str(r._BRAIN_ENTRY) in default and "-m" not in default  # headless brain by default

    dev = r.brain_argv(_args(mode="forward", vx=0.3, dev_app=True))
    assert dev[dev.index("-m") + 1] == "humanoid.logic.oli.devapp"
    assert str(r._BRAIN_ENTRY) not in dev
    # still the brain env, same socket, and the locomotion command is unchanged
    assert dev[3:5] == ["-n", "brain"]
    assert dev[dev.index("--socket") + 1] == "/tmp/s.sock"
    assert dev[dev.index("--vx") + 1] == "0.3"
    assert dev[dev.index("--mode") + 1] == "walk"


def test_dev_app_walk_does_not_spawn_app():
    """In the dev app the joystick app is launched from the Teleop panel, not by the launcher."""
    dev = r.brain_argv(_args(mode="walk", spawn_app=True, dev_app=True))
    assert "--spawn-app" not in dev
    headless = r.brain_argv(_args(mode="walk", spawn_app=True, dev_app=False))
    assert "--spawn-app" in headless


def test_glide_world_routes_to_glide_world_main():
    """--mode glide boots the glide World with glide tuning, not the walk ankle flags."""
    w = r.world_argv(_args(mode="glide", height_kp=25.0, foot_clearance=0.02))
    assert str(r._GLIDE_WORLD_ENTRY) in w
    assert str(r._WORLD_ENTRY) not in w
    assert w[w.index("--height-kp") + 1] == "25.0"
    assert w[w.index("--foot-clearance") + 1] == "0.02"
    assert "--hold-kp" in w and "--yaw-accel" in w
    # walk-only actuator knobs must not leak into the glide World
    assert "--ankle-effort" not in w and "--control" not in w


def test_glide_brain_forces_glide_mode():
    """--mode glide → brain --mode glide; fixed cmd with --vx, else socket (operator)."""
    scripted = r.brain_argv(_args(mode="glide", vx=0.4))
    assert scripted[scripted.index("--mode") + 1] == "glide"
    assert scripted[scripted.index("--joystick") + 1] == "fixed"
    assert scripted[scripted.index("--vx") + 1] == "0.4"

    operator = r.brain_argv(_args(mode="glide"))  # no --vx → socket-steered
    assert operator[operator.index("--joystick") + 1] == "socket"

    # --vx WITH --dev-app → fixed auto-glide (watch it drive, no joystick needed)
    auto = r.brain_argv(_args(mode="glide", dev_app=True, vx=0.3))
    assert auto[auto.index("--joystick") + 1] == "fixed"
    assert auto[auto.index("--vx") + 1] == "0.3"


def test_glide_dev_app_fused_command():
    """--mode glide + --dev-app → glide World + dev app brain in glide mode on socket joystick."""
    a = _args(mode="glide", dev_app=True)
    w, b = r.world_argv(a), r.brain_argv(a)
    assert str(r._GLIDE_WORLD_ENTRY) in w
    assert b[b.index("-m") + 1] == "humanoid.logic.oli.devapp"
    assert b[b.index("--mode") + 1] == "glide"
    assert b[b.index("--joystick") + 1] == "socket"  # dev app drives via Teleop panel


def test_cameras_stream_rgbd_and_feed_the_dev_app():
    """--cameras adds the World camera flags AND hands the dev app the frame socket (live RGBD)."""
    a = _args(mode="glide", dev_app=True, cameras=True)
    w, b = r.world_argv(a), r.brain_argv(a)
    assert "--cameras" in w
    assert w[w.index("--camera-socket") + 1] == "/tmp/oli-world-frames.sock"
    assert w[w.index("--camera-res") + 1] == "1280" and w[w.index("--camera-res") + 2] == "720"
    # the dev app brain gets the socket so it builds IsaacCameraSource; a headless brain doesn't
    assert b[b.index("--camera-socket") + 1] == "/tmp/oli-world-frames.sock"


def test_cameras_off_by_default():
    a = _args()
    assert "--cameras" not in r.world_argv(a)
    assert "--camera-socket" not in r.brain_argv(a)


def test_camera_socket_only_to_dev_app_brain():
    """A headless brain has no display → no camera socket even with --cameras; the World still streams."""
    assert "--camera-socket" not in r.brain_argv(_args(cameras=True, dev_app=False))
    assert "--camera-socket" in r.brain_argv(_args(cameras=True, dev_app=True))


def test_cameras_available_on_the_walk_world_too():
    w = r.world_argv(_args(mode="walk", cameras=True))
    assert "--cameras" in w and "--camera-socket" in w


def test_stages_are_world_then_brain():
    """The isaac boot plan is exactly [world (gated), brain]."""
    st = r.stages(_args(mode="forward"))
    assert [s.name for s in st] == ["world", "brain"]
    assert st[0].serving_marker == "serving on" and st[0].wait_for_path == "/tmp/s.sock"
    assert all(s.core for s in st)
    # the glide plan swaps the World entry but keeps the same two-stage shape
    g = r.stages(_args(mode="glide"))
    assert str(r._GLIDE_WORLD_ENTRY) in g[0].argv


def test_mode_presets_select_brain_behavior():
    stand = r.brain_argv(_args(mode="stand"))
    assert stand[stand.index("--mode") + 1] == "stand"
    assert stand[stand.index("--joystick") + 1] == "fixed"
    assert "--vx" not in stand  # analytic hold has no locomotion command

    walk = r.brain_argv(_args(mode="walk"))
    assert walk[walk.index("--mode") + 1] == "walk"
    assert walk[walk.index("--joystick") + 1] == "socket"  # operator-steered
    assert "--joy-port" in walk

    forward = r.brain_argv(_args(mode="forward"))  # no explicit --vx
    assert forward[forward.index("--vx") + 1] == str(r._FORWARD_VX)  # default fwd speed


def test_duration_routed_to_both():
    a = _args(duration=15.0)
    w, b = r.world_argv(a), r.brain_argv(a)
    assert w[w.index("--socket") + 1] == "/tmp/s.sock"
    assert b[b.index("--socket") + 1] == "/tmp/s.sock"
    assert w[w.index("--duration") + 1] == "15.0"
    assert b[b.index("--duration") + 1] == "15.0"


def test_duration_omitted_when_zero():
    a = _args(duration=0.0)
    assert "--duration" not in r.world_argv(a)
    assert "--duration" not in r.brain_argv(a)


def test_armature_opt_in_default_off():
    # default: armature OFF (TRON1 trains armature=0) → no --armature flag emitted
    assert "--armature" not in r.world_argv(_args())
    # --armature opts in for the A/B run
    assert "on" in r.world_argv(_args(armature=True))


def test_control_defaults_implicit_world_only():
    argv = r.world_argv(_args())
    assert argv[argv.index("--control") + 1] == "implicit"  # smoother actuator by default
    assert "explicit" in r.world_argv(_args(control="explicit"))  # opt-in TRON1 repro
    # the control law is a World/actuator concern — never the brain's
    assert "--control" not in r.brain_argv(_args())


def test_ground_contact_params_routed_to_world():
    argv = r.world_argv(_args(ground_friction=1.0, ground_restitution=0.0))
    assert argv[argv.index("--ground-friction") + 1] == "1.0"
    assert argv[argv.index("--ground-restitution") + 1] == "0.0"
    # contact model is World physics, never the brain's
    assert "--ground-friction" not in r.brain_argv(_args())


def test_solver_iters_routed_to_world_only_when_set():
    assert "--solver-vel-iters" not in r.world_argv(_args())
    argv = r.world_argv(_args(solver_vel_iters=4, solver_pos_iters=4))
    assert argv[argv.index("--solver-vel-iters") + 1] == "4"
    assert argv[argv.index("--solver-pos-iters") + 1] == "4"
    # physics-fidelity knobs never leak to the brain
    assert "--solver-vel-iters" not in r.brain_argv(_args(solver_vel_iters=4))


def test_walk_after_optional():
    assert "--walk-after" not in r.brain_argv(_args(walk_after=None))
    argv = r.brain_argv(_args(walk_after=3.0))
    assert argv[argv.index("--walk-after") + 1] == "3.0"


def test_joy_port_only_for_operator_walk():
    assert "--joy-port" not in r.brain_argv(_args(mode="forward"))
    assert "--joy-port" not in r.brain_argv(_args(mode="stand"))
    assert "--joy-port" in r.brain_argv(_args(mode="walk"))


def test_parallel_joint_gain_scales_routed_to_world_only():
    # achilles-driven joints (ankle pitch/roll, waist) need a joint-space stiffness recovery;
    # the scales are a World/actuator concern, never the brain's.
    argv = r.world_argv(_args(ankle_kp_scale=3.0, ankle_roll_scale=1.0, waist_kp_scale=1.0))
    assert argv[argv.index("--ankle-kp-scale") + 1] == "3.0"
    assert argv[argv.index("--ankle-roll-scale") + 1] == "1.0"
    assert argv[argv.index("--waist-kp-scale") + 1] == "1.0"
    for flag in ("--ankle-kp-scale", "--ankle-roll-scale", "--waist-kp-scale"):
        assert flag not in r.brain_argv(_args())


def test_ankle_parallel_opt_in_world_only():
    # faithful dual-motor achilles emulation is a World/actuator concern; default off.
    assert "--ankle-parallel" not in r.world_argv(_args())
    assert "--ankle-parallel" in r.world_argv(_args(ankle_parallel=True))
    assert "--ankle-parallel" not in r.brain_argv(_args(ankle_parallel=True))


# ── --service: the goal-driven brain via the single entrypoint (locbench §2) ──────


def _service_args(**over):
    base = dict(mode="glide", service=True, dev_app=False,
                debug_pose="/tmp/oli-world-pose.sock",
                map="assets/envs/warehouse_nvidia/nav_maps/v1")
    base.update(over)
    return _args(**base)


def test_service_brain_boots_brain_main_with_the_seam():
    cmd = r.brain_argv(_service_args())
    s = " ".join(cmd)
    assert "brain_main.py" in s and "--service" in s
    assert "--mode glide" in s
    assert "--debug-pose /tmp/oli-world-pose.sock" in s
    assert "--map-dir" in s and "nav_maps/v1" in s
    assert "--glide-scale 5.0" in s          # caps pre-divided in build_nav; product = tuned speed
    assert "--joystick" not in s             # the goal channel is the only steering input


def test_service_map_dir_is_absolutized():
    cmd = r.brain_argv(_service_args())
    map_dir = cmd[cmd.index("--map-dir") + 1]
    assert map_dir.startswith("/")           # brain subprocess runs cwd=repo root


def test_service_conflicts_with_dev_app():
    with pytest.raises(ValueError, match="dev-app"):
        r.brain_argv(_service_args(dev_app=True))


def test_service_requires_glide_mode():
    with pytest.raises(ValueError, match="glide"):
        r.brain_argv(_service_args(mode="walk"))


def test_service_requires_debug_pose_and_map():
    with pytest.raises(ValueError, match="--debug-pose"):
        r.brain_argv(_service_args(debug_pose=None))
    with pytest.raises(ValueError, match="--map"):
        r.brain_argv(_service_args(map=None))


def test_service_shadow_routes_candidate_and_frames():
    cmd = r.brain_argv(_service_args(shadow="reference", cameras=True,
                                     camera_socket="/tmp/oli-world-frames.sock"))
    s = " ".join(cmd)
    assert "--shadow reference" in s
    assert "--camera-socket /tmp/oli-world-frames.sock" in s


def test_shadow_without_cameras_rejected():
    with pytest.raises(ValueError, match="cameras"):
        r.brain_argv(_service_args(shadow="reference", cameras=False))
