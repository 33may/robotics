"""brain_main.py — launch the deployment-invariant brain against a World.

Runs in the `brain` env. Connects to the World server (SimComm) over UDS, then drives
the Orchestrator loop (read→reason→act→write, stamp-paced). Joystick source is selectable:

  --joystick fixed   a constant CLI command (--vx/--vy/--wz); --walk-after scripts the demo
  --joystick socket  live axes from the joystick app (UDP JoyPacket on --joy-port)

    # scripted demo, no operator:
    conda run -n brain python humanoid/logic/oli/brain_main.py --walk-after 3 --vx 0.3
    # live keyboard joystick (run the app in any env that has pygame, e.g. isaac):
    conda run -n brain python humanoid/logic/oli/brain_main.py --joystick socket
    conda run -n isaac python -m humanoid.logic.oli.reason.teleoperation.joystick.app

--service boots the isolated goal-driven brain instead (locbench design.md D5): Nav as the
reason (no joystick), the W4/W5 service seam attached — any client (the locbench evaluator,
dev_app later) sends `GoalCoordinate`s over the goal socket and reads pose/path/goal/est
telemetry back. Requires --mode glide, --debug-pose (the GT pose channel Nav drives on in
Stage 1) and --map-dir (the baked occupancy map):

    conda run -n brain python humanoid/logic/oli/brain_main.py --mode glide --service \\
        --debug-pose /tmp/oli-debug-pose.sock --map-dir <scene's baked map dir>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Make the `humanoid` namespace package importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.action.policy_runner import PolicyRunner  # noqa: E402
from humanoid.logic.oli.comm.client import BrainComm, CommClosedError  # noqa: E402
from humanoid.logic.oli.contracts import Mode  # noqa: E402
from humanoid.logic.oli.glide import GlideAction  # noqa: E402
from humanoid.logic.oli.reason.teleoperation.joystick import (  # noqa: E402
    FixedJoystick,
    SocketJoystickSource,
    Teleop,
)
from humanoid.logic.oli.runtime import Orchestrator  # noqa: E402


def _spawn_app(python: str, host: str, port: int) -> subprocess.Popen:
    """Launch the joystick app as a child process (needs pygame in `python`'s env)."""
    cmd = [python, "-m", "humanoid.logic.oli.reason.teleoperation.joystick.app",
           "--host", host, "--port", str(port)]
    print(f"[brain] spawning joystick app: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=str(_REPO_ROOT))


def _run_service(args) -> None:
    """The isolated goal-driven brain: Nav reason + the W4/W5 service seam, no joystick."""
    from humanoid.logic.oli.comm.debug_pose import DebugPoseClient
    from humanoid.logic.oli.reason.localization import DebugPoseLocalizer
    from humanoid.logic.oli.reason.mapping import StaticMapping
    from humanoid.logic.oli.reason.nav import build_nav
    from humanoid.logic.oli.service import ServiceHost

    pose_client = DebugPoseClient(args.debug_pose)
    nav = build_nav(
        StaticMapping(args.map_dir),
        DebugPoseLocalizer(pose_client),
        speed_scale=args.glide_scale,
    )
    host = ServiceHost(nav, goal_socket=args.goal_socket,
                       telemetry_socket=args.telemetry_socket)
    comm = BrainComm(socket_path=args.socket)
    # Nav is the reason; joystick=None, so the orchestrator hands Nav's optional second
    # parameter (camera_frame) a None — the goal channel is the only steering input.
    orch = Orchestrator(comm, nav, GlideAction(speed_scale=args.glide_scale),
                        joystick=None, recorder=host.recorder)

    print(f"[brain] service mode: goals on {args.goal_socket}, "
          f"telemetry on {args.telemetry_socket}", flush=True)
    print(f"[brain] connecting to {args.socket}...", flush=True)
    comm.connect()
    print("[brain] connected — running.", flush=True)
    t0 = time.monotonic()
    try:
        while True:
            if args.duration and (time.monotonic() - t0) > args.duration:
                print("[brain] duration reached; stopping.", flush=True)
                break
            host.poll()  # W4: latest goal command → Nav (non-blocking, latest-wins)
            if orch.step_once() is None:
                time.sleep(0.0005)  # nothing to do this iteration; yield
    except KeyboardInterrupt:
        print("\n[brain] stopping.", flush=True)
    except CommClosedError:
        print("[brain] World closed the connection; stopping.", flush=True)
    finally:
        comm.close()
        host.close()
        pose_client.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/oli-world.sock")
    ap.add_argument("--mode", choices=["stand", "walk", "glide"], default="stand",
                    help="glide = kinematic locomotion (MAY-172): forward the joystick "
                         "velocity, World glides the base (no walk policy)")
    ap.add_argument("--service", action="store_true",
                    help="boot the goal-driven brain with the W4/W5 service seam "
                         "(requires --mode glide, --debug-pose, --map-dir)")
    ap.add_argument("--goal-socket", default="/tmp/oli-goal.sock",
                    help="W4: UDS path this brain binds for GoalCoordinate commands")
    ap.add_argument("--telemetry-socket", default="/tmp/oli-telemetry.sock",
                    help="W5: UDS path this brain publishes TelemetrySnapshots to")
    ap.add_argument("--debug-pose", default=None,
                    help="UDS path of the World's GT pose channel (Nav's Stage-1 localizer)")
    ap.add_argument("--map-dir", default=None,
                    help="baked occupancy map dir (occupancy.npy + occupancy.json)")
    ap.add_argument("--glide-scale", type=float, default=1.0,
                    help="GlideAction velocity multiplier; Nav caps are pre-divided by it")
    ap.add_argument("--joystick", choices=["fixed", "socket"], default="fixed",
                    help="fixed = constant CLI cmd; socket = live app over UDP")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--joy-host", default="127.0.0.1")
    ap.add_argument("--joy-port", type=int, default=SocketJoystickSource.DEFAULT_PORT)
    ap.add_argument("--spawn-app", action="store_true",
                    help="also launch the joystick app as a subprocess (needs pygame)")
    ap.add_argument("--app-python", default=sys.executable,
                    help="interpreter to run the joystick app (default: this one)")
    ap.add_argument("--walk-after", type=float, default=None,
                    help="start in STAND, switch to WALK after N seconds (demo)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="wall seconds to run (0 = until Ctrl-C)")
    args = ap.parse_args()

    if args.service:
        if args.mode != "glide":
            ap.error("--service requires --mode glide (Stage 1 drives the glide path)")
        if not args.debug_pose or not args.map_dir:
            ap.error("--service requires --debug-pose and --map-dir")
        _run_service(args)
        return

    start_mode = Mode.STAND if (args.walk_after is not None or args.mode == "stand") else Mode.WALK
    teleop = Teleop(mode=start_mode)
    comm = BrainComm(socket_path=args.socket)
    # Glide swaps ONLY the Action: forward the velocity Intent instead of running the walk
    # ONNX. Same Teleop, same Comm, same Orchestrator loop; the World integrates the glide.
    action = GlideAction() if args.mode == "glide" else PolicyRunner()

    app_proc = None
    if args.joystick == "socket":
        joystick = SocketJoystickSource(host=args.joy_host, port=args.joy_port)
        print(f"[brain] joystick: UDP socket on {args.joy_host}:{joystick.port}", flush=True)
        if args.spawn_app:
            app_proc = _spawn_app(args.app_python, args.joy_host, joystick.port)
    else:
        joystick = FixedJoystick(args.vx, args.vy, args.wz)
    orch = Orchestrator(comm, teleop, action, joystick=joystick)

    print(f"[brain] connecting to {args.socket} (start mode={teleop.mode.name})...", flush=True)
    comm.connect()
    print("[brain] connected — running.", flush=True)

    t0 = time.monotonic()
    switched = False
    try:
        while True:
            if args.duration and (time.monotonic() - t0) > args.duration:
                print("[brain] duration reached; stopping.", flush=True)
                break
            if (args.walk_after is not None and not switched
                    and (time.monotonic() - t0) > args.walk_after):
                teleop.set_mode(Mode.WALK)
                switched = True
                print("[brain] STAND -> WALK", flush=True)
            if orch.step_once() is None:
                time.sleep(0.0005)  # nothing to do this iteration; yield
    except KeyboardInterrupt:
        print("\n[brain] stopping.", flush=True)
    except CommClosedError:
        print("[brain] World closed the connection; stopping.", flush=True)
    finally:
        comm.close()
        if isinstance(joystick, SocketJoystickSource):
            joystick.close()
        if app_proc is not None:
            app_proc.terminate()


if __name__ == "__main__":
    main()
