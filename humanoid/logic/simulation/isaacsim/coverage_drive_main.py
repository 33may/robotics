"""coverage_drive_main.py — scripted coverage drive → map-bake recording (MAY-173 T2).

Standalone Isaac process, NO brain/sockets: the deployment planner expands a
committed route's sparse targets into a dense path (routes.plan_route), a
WaypointFollower chases it on GT pose through the glide base driver, and a
DriveRecorder dumps stereo RGB + head RGBD + camera world poses at --fps.
GT is allowed here by design — the bake is offline tooling, not the runtime path.

The three seams are swappable on purpose (reusable collection flow):
  motion source  = WaypointFollower   (walk-mimic can replace/augment later)
  camera poses   = read from the RENDERED prims, not base ∘ mount (mimic-proof)
  sink           = DriveRecorder      (neutral dump; rosbag synth in the container)

    conda run -n isaac python humanoid/logic/simulation/isaacsim/coverage_drive_main.py \
        --out /tmp/coverage_drive --headless [--duration 40]
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

_HUMANOID = Path(__file__).resolve().parents[3]

# Pure imports (no isaacsim) — same pattern as glide_world_main.
from humanoid.logic.simulation.isaacsim.sim_world_main import HOME_POSE_PR  # noqa: E402
from humanoid.logic.simulation.isaacsim.glide_world_main import (  # noqa: E402
    _roll_pitch_from_quat_wxyz,
    _yaw_from_quat_wxyz,
)
from humanoid.logic.oli.glide import GlideModel  # noqa: E402
from humanoid.logic.oli.camera_mounts import (  # noqa: E402
    CAMERAS,
    D435I_STEREO_BASELINE_M,
    HEAD_CAM,
    STEREO_CAMERAS,
    rgb_intrinsics,
)
from humanoid.logic.oli.reason.mapping.occupancy_io import load_occupancy  # noqa: E402
from humanoid.logic.simulation.mapping.cell_coverage import build_coverage_route  # noqa: E402
from humanoid.logic.simulation.mapping.recorder import DriveRecorder  # noqa: E402
from humanoid.logic.simulation.mapping.routes import plan_route  # noqa: E402
from humanoid.logic.simulation.mapping.waypoint_follower import WaypointFollower  # noqa: E402

_DEF_ROUTE = _HUMANOID / "logic/simulation/mapping/routes/warehouse_coverage.yaml"
_DEF_MAP = _HUMANOID / "assets/envs/warehouse_nvidia/nav_maps/v1"
_DEF_SCENE = (_HUMANOID / "assets/envs/warehouse_nvidia/Isaac/Environments/"
              "Simple_Warehouse/full_warehouse.usd")


def _rig_dict(res) -> dict:
    """rig.json payload: everything the bake-side converter needs to build
    camera_info + extrinsics without importing this repo."""
    def mount_entry(m):
        intr = rgb_intrinsics(width=res[0], height=res[1], hfov_deg=m.hfov_deg)
        return {
            "parent_link": m.parent_link,
            "pos_base": [float(v) for v in m.pos_base],
            "pitch_down_deg": float(m.pitch_down_deg),
            "intrinsics": {"width": intr.width, "height": intr.height,
                           "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
        }
    return {
        "camera_axes": "usd (-Z view, +Y up); convert to ROS optical (+Z view) at bag synth",
        "stamp": "sim time ns (world.current_time)",
        "baseline_m": D435I_STEREO_BASELINE_M,
        "stereo_pair": ["head_left", "head_right"],
        "rgbd": ["head"],
        "cameras": {m.name: mount_entry(m) for m in (HEAD_CAM, *STEREO_CAMERAS)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", type=Path, default=_DEF_ROUTE)
    ap.add_argument("--map-dir", type=Path, default=_DEF_MAP)
    ap.add_argument("--scene", type=Path, default=_DEF_SCENE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0, help="capture rate (cuVGL floor: 5 Hz)")
    ap.add_argument("--camera-res", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
    ap.add_argument("--duration", type=float, default=0.0,
                    help="wall-clock cap in seconds (0 = drive the whole route)")
    ap.add_argument("--spawn-height", type=float, default=1.1)
    ap.add_argument("--settle-steps", type=int, default=200)
    ap.add_argument("--arrive-radius", type=float, default=0.6)
    ap.add_argument("--decimation", type=int, default=10,
                    help="physics ticks per follower command (10 ms at 1 kHz physics)")
    args = ap.parse_args()

    # Coverage spec → cell-grid targets → dense deployment-planner path (fails
    # fast, before Isaac boots). build_coverage_route is deterministic per spec seed.
    grid = load_occupancy(str(args.map_dir))
    cov, route = build_coverage_route(args.route, grid)
    path = plan_route(route, grid, spacing_m=1.0)
    total_m = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))
    n_cells = sum(1 for pts in cov.cells.values() if pts)
    print(f"[coverage] spec '{route.name}': {len(route.waypoints)} targets over "
          f"{n_cells} cells → {len(path)} pts, {total_m:.0f} m, "
          f"~{total_m / route.speed / 60:.1f} sim-min", flush=True)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})

    import numpy as np  # noqa: E402
    import omni.usd  # noqa: E402
    from isaacsim.core.api import World  # noqa: E402
    from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
    from pxr import Usd, UsdGeom  # noqa: E402

    from humanoid.logic.simulation.isaacsim.oli import NUM_JOINTS, Oli  # noqa: E402
    from humanoid.logic.simulation.isaacsim.sim_comm import SimComm  # noqa: E402

    # 1 kHz physics — MUST match glide_world_main exactly. The settled base pitch is a
    # capture off the settle TRANSIENT (glide pins it at 200 steps × 1/1000 = 0.2 s, torso
    # still upright → level head). At 1/200 the same 200 steps run 1.0 s, the crouch relaxes
    # ~13° further back, and the head ends up staring at the ceiling (2026-07-15 bug). Keep
    # the drive's base behaviour byte-identical to the demo runtime it must localize against.
    physics_dt = 1.0 / 1000.0
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    if args.scene.exists():
        add_reference_to_stage(usd_path=str(args.scene), prim_path="/World/Scene")
        print(f"[coverage] scene loaded: {args.scene}", flush=True)
    else:
        print(f"[coverage] scene NOT found → bare plane (bake data useless!): "
              f"{args.scene}", flush=True)

    oli = Oli(world, pin_root=False, spawn_pose=(path[0][0], path[0][1], args.spawn_height),
              cameras=True, stereo_cameras=True, camera_resolution=tuple(args.camera_res))
    # SimComm only for the PR→Isaac joint permutation of the home crouch (no socket served).
    simcomm = SimComm(oli, socket_path=str(args.out) + ".sock")
    home_isaac = simcomm.pr_to_isaac_vector(np.asarray(HOME_POSE_PR, dtype=np.float32))
    zeros = np.zeros(NUM_JOINTS, dtype=np.float32)
    oli.set_joint_state(home_isaac)

    kp = np.full(NUM_JOINTS, 150.0, dtype=np.float32)
    kd = np.full(NUM_JOINTS, 5.0, dtype=np.float32)
    for _ in range(args.settle_steps):
        oli.apply_isaac(home_isaac, zeros, zeros, kp, kd)
        world.step(render=False)
    stand_z = float(oli.base_world_position()[2])
    glide_z = stand_z + 0.10
    _, pitch0 = _roll_pitch_from_quat_wxyz(oli.base_world_quat_wxyz())
    _cp, _sp = math.cos(pitch0 / 2.0), math.sin(pitch0 / 2.0)

    # Face the first leg at the route start.
    yaw0 = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    _cz, _sz = math.cos(yaw0 / 2.0), math.sin(yaw0 / 2.0)
    oli.set_base_pose(position=(path[0][0], path[0][1], glide_z),
                      quat_wxyz=(_cp * _cz, -_sp * _sz, _sp * _cz, _cp * _sz))

    model = GlideModel(lin_accel=10.0, yaw_accel=20.0, x=path[0][0], y=path[0][1], yaw=yaw0)
    follower = WaypointFollower(path, speed=route.speed, arrive_radius=args.arrive_radius,
                                loop=route.loop)
    hold_kp = np.full(NUM_JOINTS, 80.0, dtype=np.float32)
    hold_kd = np.full(NUM_JOINTS, 4.0, dtype=np.float32)
    cmd_dt = args.decimation * physics_dt

    # Camera warmup (isaac-camera-first-render-not-ready: annotators need render ticks).
    for _ in range(8):
        world.step(render=True)

    rec = DriveRecorder(args.out, {**_rig_dict(args.camera_res),
                                   "route": route.name, "scene": str(args.scene)})
    stage = omni.usd.get_context().get_stage()

    def _cam_world(prim_name: str, parent_link: str) -> np.ndarray:
        m = UsdGeom.Xformable(
            stage.GetPrimAtPath(f"/World/Oli/{parent_link}/{prim_name}_camera")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(m).T  # Gf row-vector → column-vector convention

    def _sim_ns() -> int:
        return int(world.current_time * 1e9)

    capture_period_ns = int(1e9 / args.fps)
    next_capture = _sim_ns()
    head_mount = next(m for m in CAMERAS if m.name == "head")
    t_start = time.monotonic()
    captures = 0

    print(f"[coverage] driving ({'headless' if args.headless else 'gui'}), "
          f"recording to {args.out} @ {args.fps:.0f} Hz", flush=True)
    try:
        while app.is_running() and not follower.done:
            if args.duration and (time.monotonic() - t_start) > args.duration:
                print("[coverage] duration cap hit — stopping early", flush=True)
                break
            pos = oli.base_world_position()
            yaw = _yaw_from_quat_wxyz(oli.base_world_quat_wxyz())
            vx, vy, wz = follower.command(float(pos[0]), float(pos[1]), yaw)
            model.step(vx, vy, wz, cmd_dt)

            capture = _sim_ns() >= next_capture
            for i in range(args.decimation):
                p = oli.base_world_position()
                yw = _yaw_from_quat_wxyz(oli.base_world_quat_wxyz())
                cy, sy = math.cos(yw), math.sin(yw)
                vz = 20.0 * (glide_z - float(p[2]))
                oli.set_base_velocity(
                    linear=(model.vx * cy - model.vy * sy,
                            model.vx * sy + model.vy * cy, vz),
                    angular=(0.0, 0.0, model.wz))
                oli.apply_isaac(home_isaac, zeros, zeros, hold_kp, hold_kd)
                render = capture and i == args.decimation - 1
                world.step(render=render)
                # upright pin (roll 0, settled pitch, current yaw) — glide fidelity rule
                pp = oli.base_world_position()
                yy = _yaw_from_quat_wxyz(oli.base_world_quat_wxyz())
                czz, szz = math.cos(yy / 2.0), math.sin(yy / 2.0)
                oli.set_base_pose(position=(float(pp[0]), float(pp[1]), float(pp[2])),
                                  quat_wxyz=(_cp * czz, -_sp * szz, _sp * czz, _cp * szz))

            if capture:
                stamp = _sim_ns()
                for m in STEREO_CAMERAS:
                    rec.add_frame(m.name, stamp, oli.read_camera_rgb(m.name),
                                  _cam_world(m.name, m.parent_link))
                rgb, depth = oli.read_camera_rgbd("head")
                rec.add_frame("head", stamp, rgb,
                              _cam_world("head", head_mount.parent_link), depth_m=depth)
                bp = oli.base_world_position()
                rec.add_base_pose(stamp, x=float(bp[0]), y=float(bp[1]),
                                  yaw=_yaw_from_quat_wxyz(oli.base_world_quat_wxyz()))
                captures += 1
                next_capture += capture_period_ns
                if captures % 50 == 0:
                    print(f"[coverage] t={world.current_time:6.1f}s wp {follower.index}/"
                          f"{len(path)} base=({bp[0]:+.1f},{bp[1]:+.1f}) "
                          f"frames={rec.frames_written}", flush=True)
    except KeyboardInterrupt:
        print("\n[coverage] interrupted.", flush=True)
    finally:
        rec.close()
        done = follower.done
        print(f"[coverage] {'ROUTE COMPLETE' if done else 'PARTIAL'} — "
              f"{captures} captures, {rec.frames_written} frames → {args.out}", flush=True)
        app.close()
    sys.exit(0 if done or args.duration else 1)


if __name__ == "__main__":
    main()
