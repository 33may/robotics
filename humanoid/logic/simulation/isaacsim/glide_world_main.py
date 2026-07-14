"""glide_world_main.py — the glide World process (MAY-172): Isaac + Oli + kinematic glide.

Runs in the `isaac` env. The demo-time stand-in for the LimX-gated walk: instead of a walk
policy driving the legs and PhysX producing base motion, the brain sends a `GLIDE_CMD`
(body-frame velocity) and this World glides Oli's base. The base is **velocity-driven**
(not a true kinematic body) so the body is BLOCKED by world collision geometry (walls, once
MAY-171's mesh lands), with a P height-hold + an upright lock keeping Oli standing. Legs are
held in the standing pose (a visual step is a later pass). Same Comm socket + Observation
flow as the walk World, so the brain is byte-identical across modes — swapping in the real
walk means launching `sim_world_main.py` instead of this, with the joystick unchanged.

    # terminal 1 (World):
    conda run -n isaac python humanoid/logic/simulation/isaacsim/glide_world_main.py
    # terminal 2 (brain, glide mode):
    conda run -n brain python humanoid/logic/oli/brain_main.py --mode glide --joystick socket
    conda run -n isaac python -m humanoid.logic.oli.reason.teleoperation.joystick.app
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Pure imports (no isaacsim): the walk crouch (drift-guarded against walk_param) + the model.
from humanoid.logic.simulation.isaacsim.sim_world_main import HOME_POSE_PR  # noqa: E402
from humanoid.logic.oli.glide import GlideModel  # noqa: E402
from humanoid.logic.oli.comm.debug_pose import DebugPoseServer  # noqa: E402  (pure stdlib)


def _yaw_from_quat_wxyz(q) -> float:
    """Heading (yaw about world-Z) from a (w, x, y, z) quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_pitch_from_quat_wxyz(q) -> tuple:
    """Roll (about body-X) and pitch (about body-Y) from a (w, x, y, z) quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/oli-world.sock")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--render-every", type=int, default=2,
                    help="render 1 in N physics ticks (lower = smoother viewport)")
    ap.add_argument("--watchdog-ms", type=float, default=500.0)
    ap.add_argument("--decimation", type=int, default=10,
                    help="physics ticks per glide command (matches the brain's 10 ms pacing)")
    ap.add_argument("--spawn-height", type=float, default=1.1,
                    help="base z at spawn (settles to the stand height below)")
    ap.add_argument("--settle-steps", type=int, default=200,
                    help="pre-serve physics steps holding the crouch so feet plant + z settles")
    ap.add_argument("--settle-kp", type=float, default=150.0)
    ap.add_argument("--settle-kd", type=float, default=5.0)
    ap.add_argument("--hold-kp", type=float, default=80.0,
                    help="leg/arm PD stiffness holding the stand pose during glide")
    ap.add_argument("--hold-kd", type=float, default=4.0)
    ap.add_argument("--lin-accel", type=float, default=10.0,
                    help="glide linear accel limit [m/s²] — how fast a stick step ramps (~0.25 s to top)")
    ap.add_argument("--yaw-accel", type=float, default=20.0,
                    help="glide yaw accel limit [rad/s²]")
    ap.add_argument("--height-kp", type=float, default=20.0,
                    help="P gain [1/s] on the base-height hold (vz = kp·(z0 − z))")
    ap.add_argument("--foot-clearance", type=float, default=0.10,
                    help="glide at settled-stand-height + this (hover; >0 lifts feet off the "
                         "floor to avoid foot-ground drag while the base translates)")
    ap.add_argument("--ground-friction", type=float, default=1.0)
    ap.add_argument("--ground-restitution", type=float, default=0.0)
    ap.add_argument("--cameras", action="store_true",
                    help="attach Oli's baked D435i cameras and stream RGBD on a separate frame "
                         "channel (off by default so glide pays no render cost)")
    ap.add_argument("--camera-socket", default="/tmp/oli-world-frames.sock",
                    help="AF_UNIX SOCK_STREAM path for the camera frame channel (separate from --socket)")
    ap.add_argument("--camera-res", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"),
                    help="camera resolution W H (D435i native 1280×720)")
    ap.add_argument("--camera-every", type=int, default=32,
                    help="publish cameras every N physics ticks (~30 Hz at 1 kHz) — DECOUPLED "
                         "from the viewport render rate so 720p cameras don't stall the loop")
    ap.add_argument("--scene", default=None,
                    help="USD world referenced as the fixed ground-truth scene (MAY-171) — "
                         "visual reference for Oli's cameras / SLAM. The default ground plane "
                         "is still added for physics; pass 'none' (or omit) to skip.")
    ap.add_argument("--debug-pose", default=None,
                    help="AF_UNIX SOCK_DGRAM path to stream Oli's ground-truth base pose "
                         "(stamp, x, y, yaw) for nav debug/eval. NOT the invariance spine — the "
                         "real robot has no such signal; off by default.")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="wall seconds to run (0 = until the window closes)")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})

    import numpy as np  # noqa: E402
    from isaacsim.core.api import World  # noqa: E402

    from humanoid.logic.simulation.isaacsim.oli import NUM_JOINTS, Oli  # noqa: E402
    from humanoid.logic.simulation.isaacsim.sim_comm import SimComm, SimCommError  # noqa: E402
    from humanoid.logic.oli.comm.camera_publisher import CameraPublisher  # noqa: E402

    physics_dt = 1.0 / 1000.0
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane(
        static_friction=args.ground_friction,
        dynamic_friction=args.ground_friction,
        restitution=args.ground_restitution,
    )
    # Fixed ground-truth world (MAY-171): reference a USD scene for Oli's cameras + SLAM to
    # render against. The default ground plane above stays as the guaranteed z=0 physics floor;
    # a missing file just warns and boots bare (keeps the repo portable without the download).
    if args.scene and args.scene.lower() != "none":
        from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
        if Path(args.scene).exists():
            add_reference_to_stage(usd_path=args.scene, prim_path="/World/Scene")
            print(f"[glide-world] scene loaded: {args.scene}", flush=True)
        else:
            print(f"[glide-world] scene NOT found → bare ground plane: {args.scene}", flush=True)
    # Free base (NOT pinned): the velocity driver moves the root and PhysX resolves contacts.
    oli = Oli(world, pin_root=False, spawn_pose=(0.0, 0.0, args.spawn_height),
              cameras=args.cameras, camera_resolution=tuple(args.camera_res))
    simcomm = SimComm(oli, socket_path=args.socket)

    # Debug/eval ground-truth pose channel (opt-in, NOT the spine): a separate SOCK_DGRAM stream of
    # the base (x, y, yaw) so the nav brain's DebugPoseLocalizer can plan on perfect coords before a
    # real localizer exists. Best-effort; production never launches it (the real robot has no GT pose).
    dbg_pose = DebugPoseServer(args.debug_pose) if args.debug_pose else None
    if dbg_pose is not None:
        print(f"[glide-world] debug pose ON → {args.debug_pose} (ground-truth base x,y,yaw)", flush=True)

    # Camera frame channel (opt-in): a SEPARATE SOCK_STREAM server shipping RGBD on the render
    # sub-tick, never blocking the glide control loop (latest-wins per camera). Off by default.
    pub = None
    if args.cameras:
        pub = CameraPublisher(oli, socket_path=args.camera_socket, every=args.camera_every)
        print(f"[glide-world] cameras ON ({args.camera_res[0]}×{args.camera_res[1]}) → "
              f"frame channel {args.camera_socket} (streams: {oli.camera_names})", flush=True)

    home_isaac = simcomm.pr_to_isaac_vector(np.asarray(HOME_POSE_PR, dtype=np.float32))
    zeros = np.zeros(NUM_JOINTS, dtype=np.float32)
    oli.set_joint_state(home_isaac)

    # Settle under physics so feet plant and the base finds its natural stand height.
    if args.settle_steps > 0:
        kp = np.full(NUM_JOINTS, args.settle_kp, dtype=np.float32)
        kd = np.full(NUM_JOINTS, args.settle_kd, dtype=np.float32)
        for _ in range(args.settle_steps):
            oli.apply_isaac(home_isaac, zeros, zeros, kp, kd)
            world.step(render=False)
    stand_z = float(oli.base_world_position()[2])
    glide_z = stand_z + args.foot_clearance
    print(f"[glide-world] settled stand height z={stand_z:.3f}; glide z0={glide_z:.3f} "
          f"(clearance {args.foot_clearance:.3f})", flush=True)

    # Zero ONLY the lateral lean (roll) at the glide height; KEEP the settled pitch and yaw.
    # The base carries a small forward pelvis tilt that compensates for Oli's reclined home
    # crouch, so forcing pitch=0 tips the torso BACK — keep it. This removes the sideways tilt
    # (Oli's under-actuated ankle/waist roll, a known Isaac fidelity gap) without the lean-back;
    # the loop's rate-lock then holds this orientation.
    _p = oli.base_world_position()
    _q = oli.base_world_quat_wxyz()
    _, _pitch0 = _roll_pitch_from_quat_wxyz(_q)
    _yaw0 = _yaw_from_quat_wxyz(_q)
    _cz, _sz = math.cos(_yaw0 / 2.0), math.sin(_yaw0 / 2.0)
    _cp, _sp = math.cos(_pitch0 / 2.0), math.sin(_pitch0 / 2.0)
    oli.set_base_pose(position=(float(_p[0]), float(_p[1]), glide_z),
                      quat_wxyz=(_cp * _cz, -_sp * _sz, _sp * _cz, _cp * _sz))

    # Seed the model at Oli's actual pose so its bookkeeping matches the world.
    pos0 = oli.base_world_position()
    yaw0 = _yaw_from_quat_wxyz(oli.base_world_quat_wxyz())
    model = GlideModel(lin_accel=args.lin_accel, yaw_accel=args.yaw_accel,
                       x=float(pos0[0]), y=float(pos0[1]), yaw=yaw0)

    hold_kp = np.full(NUM_JOINTS, args.hold_kp, dtype=np.float32)
    hold_kd = np.full(NUM_JOINTS, args.hold_kd, dtype=np.float32)
    cmd_dt = args.decimation * physics_dt

    def _drive_base_substep() -> None:
        """One physics substep of the velocity-driven glide: command the root's spatial
        velocity from the model (body→world twist + P height-hold + upright lock) and hold
        the legs/arms in the stand pose, then let PhysX integrate + resolve contacts."""
        pos = oli.base_world_position()
        yaw = _yaw_from_quat_wxyz(oli.base_world_quat_wxyz())
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        vx_w = model.vx * cos_y - model.vy * sin_y
        vy_w = model.vx * sin_y + model.vy * cos_y
        vz = args.height_kp * (glide_z - float(pos[2]))         # P height hold
        # Drive translation + yaw only. The base roll/pitch are PINNED upright after the step
        # (see the loop) rather than chased via angular velocity — that pinning is drift-free
        # and never fights the planted feet (no float/spin).
        oli.set_base_velocity(linear=(vx_w, vy_w, vz), angular=(0.0, 0.0, model.wz))
        oli.apply_isaac(home_isaac, zeros, zeros, hold_kp, hold_kd)

    # Warm the camera render products before serving: Isaac's annotators need a few render
    # ticks to populate before get_rgba/depth return valid buffers (the publisher tolerates a
    # not-ready camera, but warming here makes RGBD flow from the first loop tick).
    if pub is not None:
        for _ in range(8):
            world.step(render=True)
        print("[glide-world] cameras warmed (8 render ticks)", flush=True)

    print(f"[glide-world] serving on {args.socket}; waiting for brain (--mode glide)...",
          flush=True)
    simcomm.serve()
    print("[glide-world] brain connected. gliding.", flush=True)

    watchdog_s = args.watchdog_ms / 1000.0
    tick = 0                    # counts world.step() calls — the render/camera CADENCE gate
    n_cmds = 0
    loop_start = time.monotonic()

    def _sim_ns() -> int:
        """The D8 stamp clock = TRUE simulated seconds. NOT tick·physics_dt: world.step()
        advances rendering_dt (~16.7 ms) on render ticks and physics_dt (1 ms) otherwise, so
        `tick` is not a uniform-dt clock — stamping tick·1e6 ran the clock ~8× slow under
        --render-every 2 (MAY-173, 2026-07-13). world.current_time is Isaac's real sim time."""
        return int(world.current_time * 1e9)

    def _timed_out() -> bool:
        return bool(args.duration) and (time.monotonic() - loop_start) > args.duration

    try:
        simcomm.publish(_sim_ns())
        while app.is_running() and not _timed_out():
            cmd = simcomm.receive_glide_blocking(timeout=watchdog_s)
            if cmd is None:  # brain silent → decay the glide to rest, keep holding station
                model.step(0.0, 0.0, 0.0, cmd_dt)
            else:
                model.step(cmd.v_x, cmd.v_y, cmd.w_z, cmd_dt)
                n_cmds += 1
            for _ in range(args.decimation):
                _drive_base_substep()
                render = tick % args.render_every == 0
                world.step(render=render)
                # Pin base orientation upright: roll=0, natural pitch (_cp/_sp from the settle),
                # current yaw. Position stays PhysX-integrated (walls still block); drift-free,
                # no fight with the planted feet.
                _pp = oli.base_world_position()
                _yy = _yaw_from_quat_wxyz(oli.base_world_quat_wxyz())
                _czz, _szz = math.cos(_yy / 2.0), math.sin(_yy / 2.0)
                oli.set_base_pose(
                    position=(float(_pp[0]), float(_pp[1]), float(_pp[2])),
                    quat_wxyz=(_cp * _czz, -_sp * _szz, _sp * _czz, _cp * _szz))
                if render and pub is not None:  # camera pixels only refresh on render steps
                    pub.publish(tick, _sim_ns())   # `tick` = cadence gate; stamp = real sim time
                tick += 1
            simcomm.publish(_sim_ns())
            if dbg_pose is not None:
                _pdp = oli.base_world_position()
                _ydp = _yaw_from_quat_wxyz(oli.base_world_quat_wxyz())
                dbg_pose.publish(_sim_ns(), float(_pdp[0]), float(_pdp[1]), _ydp)
            if cmd is not None and n_cmds % 25 == 0:
                p = oli.base_world_position()
                print(f"[glide-world] t={world.current_time:5.2f}s "
                      f"base=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"vx={model.vx:+.2f} vy={model.vy:+.2f} wz={model.wz:+.2f}", flush=True)
    except KeyboardInterrupt:
        print("\n[glide-world] stopping.", flush=True)
    except SimCommError as e:
        print(f"[glide-world] brain disconnected: {e}; stopping.", flush=True)
    finally:
        if pub is not None:
            pub.close()
        if dbg_pose is not None:
            dbg_pose.close()
        simcomm.close()
        app.close()


if __name__ == "__main__":
    main()
