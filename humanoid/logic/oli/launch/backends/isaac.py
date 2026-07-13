"""isaac.py — launcher backend for the Isaac sim World (MAY-147 / MAY-172).

Boots two processes: the Isaac World (`sim_world_main.py`, or `glide_world_main.py` when
`--mode glide`) in the isaac env, then the brain in the brain env — the headless
`brain_main.py`, or the windowed dev app (`-m humanoid.logic.oli.devapp`) with `--dev-app`.

The `world_argv` / `brain_argv` builders are pure (unit-tested without spawning). Glide
used to be a separate `--glide` flag on `run_oli_sim`; it is now just `--mode glide`, so
the same command shape covers stand/walk/forward/glide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..supervisor import Stage

NAME = "isaac"

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]                       # …/launch/backends/isaac.py → repo root
_ISAAC_DIR = _REPO_ROOT / "humanoid" / "logic" / "simulation" / "isaacsim"
_WORLD_ENTRY = _ISAAC_DIR / "sim_world_main.py"
_GLIDE_WORLD_ENTRY = _ISAAC_DIR / "glide_world_main.py"
_BRAIN_ENTRY = _REPO_ROOT / "humanoid" / "logic" / "oli" / "brain_main.py"
_DEVAPP_MODULE = "humanoid.logic.oli.devapp"

#: default forward speed (m/s) for --mode forward without an explicit --vx
_FORWARD_VX = 0.3

#: substring the World prints right before it binds the socket and waits for the brain
_SERVING_MARKER = "serving on"


def _camera_world_flags(a: argparse.Namespace) -> list[str]:
    """--cameras → the World camera flags (shared by the walk + glide World argvs)."""
    if not getattr(a, "cameras", False):
        return []
    return ["--cameras", "--camera-socket", a.camera_socket,
            "--camera-res", str(a.camera_res[0]), str(a.camera_res[1])]


# ── CLI (backend-specific flags; the launcher adds the common brain/mode flags) ──────

def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--headless", action="store_true",
                    help="run Isaac without a viewport window (default: windowed UI)")
    ap.add_argument("--pace", choices=["lockstep", "free"], default="lockstep")
    ap.add_argument("--decimation", type=int, default=10)
    ap.add_argument("--spawn-height", type=float, default=1.1)
    ap.add_argument("--render-every", type=int, default=2,
                    help="render 1 in N physics ticks (lower = smoother viewport)")
    ap.add_argument("--pin-root", action="store_true")
    ap.add_argument("--control", choices=["explicit", "implicit"], default="implicit",
                    help="implicit = PhysX drive (smoother, default); explicit = legged_gym "
                         "per-substep torque PD (faithful to TRON1 but jitters at armature=0)")
    ap.add_argument("--armature", action="store_true",
                    help="inject MJCF rotor inertia (default off; TRON1 trains armature=0)")
    ap.add_argument("--solver-vel-iters", type=int, default=0)
    ap.add_argument("--solver-pos-iters", type=int, default=0)
    ap.add_argument("--ground-friction", type=float, default=1.0)
    ap.add_argument("--ground-restitution", type=float, default=0.0)
    ap.add_argument("--ankle-effort", type=float, default=42.0)
    ap.add_argument("--ankle-vel", type=float, default=13.6)
    ap.add_argument("--ankle-kp-scale", type=float, default=1.0)
    ap.add_argument("--ankle-roll-scale", type=float, default=1.0)
    ap.add_argument("--waist-kp-scale", type=float, default=1.0)
    ap.add_argument("--ankle-parallel", action="store_true")
    ap.add_argument("--spawn-app", action="store_true",
                    help="walk/glide (non-dev-app): also launch the joystick app (pygame)")
    # glide (MAY-172) tuning — only consumed when --mode glide
    ap.add_argument("--hold-kp", type=float, default=80.0, help="glide: leg-hold PD kp")
    ap.add_argument("--hold-kd", type=float, default=4.0, help="glide: leg-hold PD kd")
    ap.add_argument("--height-kp", type=float, default=20.0, help="glide: base height-hold P")
    ap.add_argument("--foot-clearance", type=float, default=0.10,
                    help="glide: hover height above floor [m] — 0.10 lifts the feet clear so "
                         "they don't drag/collide at speed (walls still block the body)")
    ap.add_argument("--lin-accel", type=float, default=10.0, help="glide: linear accel clamp (snappy ramp ~0.25 s to top)")
    ap.add_argument("--yaw-accel", type=float, default=20.0, help="glide: yaw accel clamp")
    ap.add_argument("--glide-scale", type=float, default=5.0,
                    help="glide (dev-app): stick→speed multiplier; 5.0 = full-stick 2.5 m/s fwd "
                         "+ 2.5 rad/s turn")
    ap.add_argument("--world-env", default="isaac")
    # scene (MAY-171): fixed ground-truth world referenced into the glide World for camera
    # render + SLAM. Defaults to 'none' (bare ground plane) while the scene is being built;
    # pass --scene /path/to/world.usd once it's ready.
    ap.add_argument("--scene", default="none",
                    help="USD world for --mode glide (MAY-171 ground-truth scene); "
                         "default 'none' = bare ground plane")
    # cameras (MAY-149): stream Oli's baked D435i RGBD; the dev app renders it live
    ap.add_argument("--cameras", action="store_true",
                    help="stream Oli's baked D435i RGBD on a frame channel (the dev app shows "
                         "it live in the Cameras panel)")
    ap.add_argument("--camera-socket", default="/tmp/oli-world-frames.sock",
                    help="AF_UNIX SOCK_STREAM path for the camera frame channel")
    ap.add_argument("--camera-res", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"),
                    help="camera resolution W H (D435i native 1280×720)")
    ap.add_argument("--camera-every", type=int, default=32,
                    help="glide: publish a frame every N physics ticks (1 kHz) — 32 ≈ 30 Hz "
                         "(D435i-faithful); 16 ≈ 60 Hz; lower = more frequent but heavier")
    # nav debug overlay (MAY-173/175): the glide World streams its ground-truth base pose on a
    # side channel; the dev-app Nav Map draws it over the baked occupancy artifact.
    ap.add_argument("--debug-pose", nargs="?", const="/tmp/oli-world-pose.sock", default=None,
                    help="glide: stream Oli's ground-truth base pose and overlay it on the dev-app "
                         "Nav Map. Bare flag → /tmp/oli-world-pose.sock; NOT the invariance spine.")
    ap.add_argument("--map", default=None,
                    help="baked occupancy artifact dir (occupancy.npy + occupancy.json) → the "
                         "dev-app Nav Map panel (bake with nav/occupancy_io.py)")


# ── pure command builders ────────────────────────────────────────────────────────

def _glide_world_argv(a: argparse.Namespace) -> list[str]:
    """conda-run argv for the GLIDE Isaac World — free base, velocity-driven (MAY-172)."""
    py = ["conda", "run", "--no-capture-output", "-n", a.world_env, "python", "-u",
          str(_GLIDE_WORLD_ENTRY),
          "--socket", a.socket,
          "--decimation", str(a.decimation),
          "--spawn-height", str(a.spawn_height),
          "--render-every", str(a.render_every),
          "--hold-kp", str(a.hold_kp), "--hold-kd", str(a.hold_kd),
          "--height-kp", str(a.height_kp), "--foot-clearance", str(a.foot_clearance),
          "--lin-accel", str(a.lin_accel), "--yaw-accel", str(a.yaw_accel),
          "--ground-friction", str(a.ground_friction),
          "--ground-restitution", str(a.ground_restitution)]
    if getattr(a, "scene", "none") and a.scene.lower() != "none":
        # Resolve here (launcher CWD = invocation dir) so the World — spawned with cwd=repo
        # root — still finds a path the user gave relative to humanoid/.
        py += ["--scene", str(Path(a.scene).resolve())]
    py += _camera_world_flags(a)
    if getattr(a, "cameras", False):
        py += ["--camera-every", str(a.camera_every)]   # glide World honors the cadence knob
    if getattr(a, "debug_pose", None):
        py += ["--debug-pose", a.debug_pose]            # stream ground-truth base pose (nav overlay)
    if a.headless:
        py.append("--headless")
    if a.duration:
        py += ["--duration", str(a.duration)]
    return py


def world_argv(a: argparse.Namespace) -> list[str]:
    """conda-run argv for the Isaac World (walk World, or the glide World for --mode glide)."""
    if a.mode == "glide":
        return _glide_world_argv(a)
    py = ["conda", "run", "--no-capture-output", "-n", a.world_env, "python", "-u",
          str(_WORLD_ENTRY),
          "--socket", a.socket,
          "--pace", a.pace,
          "--decimation", str(a.decimation),
          "--spawn-height", str(a.spawn_height),
          "--render-every", str(a.render_every)]
    if a.headless:
        py.append("--headless")
    if a.pin_root:
        py.append("--pin-root")
    py += ["--control", a.control]
    if a.armature:
        py += ["--armature", "on"]
    if a.solver_vel_iters:
        py += ["--solver-vel-iters", str(a.solver_vel_iters)]
    if a.solver_pos_iters:
        py += ["--solver-pos-iters", str(a.solver_pos_iters)]
    py += ["--ground-friction", str(a.ground_friction),
           "--ground-restitution", str(a.ground_restitution),
           "--ankle-effort", str(a.ankle_effort), "--ankle-vel", str(a.ankle_vel),
           "--ankle-kp-scale", str(a.ankle_kp_scale),
           "--ankle-roll-scale", str(a.ankle_roll_scale),
           "--waist-kp-scale", str(a.waist_kp_scale)]
    if a.ankle_parallel:
        py.append("--ankle-parallel")
    py += _camera_world_flags(a)
    if a.duration:
        py += ["--duration", str(a.duration)]
    return py


def brain_argv(a: argparse.Namespace) -> list[str]:
    """conda-run argv for the brain, derived from --mode.

      stand   → analytic StandPolicy, hold the crouch (no locomotion)
      walk    → walk policy steered LIVE by the socket joystick (operator drives)
      forward → walk policy with a FIXED command (constant forward walk, no operator)
      glide   → GlideAction forwards the velocity Intent (MAY-172)

    `--dev-app` routes the brain to the windowed dev app (same Orchestrator + a UI); it
    attaches to the same World socket and launches the joystick from its Teleop panel.
    """
    entry = ["-m", _DEVAPP_MODULE] if a.dev_app else [str(_BRAIN_ENTRY)]
    py = ["conda", "run", "--no-capture-output", "-n", a.brain_env, "python", "-u",
          *entry, "--socket", a.socket]
    if a.mode == "glide":
        py += ["--mode", "glide"]
        if a.dev_app:
            # only the dev-app brain honors --glide-scale (headless brain_main doesn't take it)
            py += ["--glide-scale", str(a.glide_scale)]
        if a.vx is not None:
            # fixed auto-glide (works with --dev-app too: watch it drive, no joystick needed)
            py += ["--joystick", "fixed",
                   "--vx", str(a.vx), "--vy", str(a.vy or 0.0), "--wz", str(a.wz or 0.0)]
        else:  # operator-steered (dev app Teleop panel, or brain_main --spawn-app)
            py += ["--joystick", "socket", "--joy-port", str(a.joy_port)]
            if a.spawn_app and not a.dev_app:
                py.append("--spawn-app")
    elif a.mode == "stand":
        py += ["--mode", "stand", "--joystick", "fixed"]
    elif a.mode == "walk":
        py += ["--mode", "walk", "--joystick", "socket", "--joy-port", str(a.joy_port)]
        if a.spawn_app and not a.dev_app:
            py.append("--spawn-app")
    else:  # forward
        vx = a.vx if a.vx is not None else _FORWARD_VX
        py += ["--mode", "walk", "--joystick", "fixed",
               "--vx", str(vx), "--vy", str(a.vy or 0.0), "--wz", str(a.wz or 0.0)]
    # the dev app consumes the camera frame channel (headless brain_main has no display)
    if getattr(a, "cameras", False) and a.dev_app:
        py += ["--camera-socket", a.camera_socket]
    # the dev app draws the Nav Map: the ground-truth pose stream + the baked occupancy artifact
    if a.dev_app:
        if getattr(a, "debug_pose", None):
            py += ["--debug-pose", a.debug_pose]
        if getattr(a, "map", None):
            # Same reason as --scene: absolutize against the launcher CWD so the brain
            # subprocess (cwd=repo root) resolves the artifact dir correctly.
            py += ["--map", str(Path(a.map).resolve())]
    if a.walk_after is not None:
        py += ["--walk-after", str(a.walk_after)]
    if a.duration:
        py += ["--duration", str(a.duration)]
    return py


# ── the ordered boot plan ──────────────────────────────────────────────────────────

def stages(a: argparse.Namespace) -> list[Stage]:
    return [
        Stage("world", world_argv(a), cwd=_REPO_ROOT,
              serving_marker=_SERVING_MARKER, wait_for_path=a.socket, core=True),
        Stage("brain", brain_argv(a), cwd=_REPO_ROOT, core=True),
    ]
