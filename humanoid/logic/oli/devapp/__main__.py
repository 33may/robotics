"""__main__.py — launch the Oli robot-brain dev app.

    # UI only (synthetic cameras, no World):
    /home/may33/miniconda3/envs/brain/bin/python -m humanoid.logic.oli.devapp

    # attach to a running World and drive the walk policy (app IS the brain):
    /home/may33/miniconda3/envs/brain/bin/python -m humanoid.logic.oli.devapp \\
        --socket /tmp/oli-world.sock --mode walk --joystick fixed --vx 0.3

    # headless screenshot (self-validation, no monitor):
    xvfb-run -a -s "-screen 0 1600x1000x24" \\
        /home/may33/miniconda3/envs/brain/bin/python -m humanoid.logic.oli.devapp \\
        --screenshot /tmp/app.png

Usually you do not run this directly — `run_oli_sim.py --dev-app` boots the Isaac World and
this app together from one command. The app does NOT own the World; it attaches to whatever
already serves `--socket`. Brain flags mirror `brain_main.py` so `run_oli_sim` drives both
the same way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the `humanoid` namespace package importable when run as a module or directly.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.devapp.app import run  # noqa: E402
from humanoid.logic.oli.devapp.panels.camera_panel import CameraPanel  # noqa: E402
from humanoid.logic.oli.devapp.panels.state_panel import StatePanel  # noqa: E402
from humanoid.logic.oli.devapp.panels.teleop_panel import TeleopPanel  # noqa: E402
from humanoid.logic.oli.devapp.registry import PanelRegistry  # noqa: E402
from humanoid.logic.oli.devapp.sources.synthetic_camera_source import (  # noqa: E402
    SyntheticCameraSource,
)
from humanoid.logic.oli.devapp.state import AppState  # noqa: E402


def build_registry(args: argparse.Namespace) -> PanelRegistry:
    """Compose the app's panels. Add a panel = one register() line here."""
    reg = PanelRegistry()
    # LIVE Isaac RGBD when a camera frame channel is given (MAY-149), else synthetic —
    # same CameraPanel either way (it depends only on the CameraSource protocol).
    if getattr(args, "camera_socket", None):
        from humanoid.logic.oli.devapp.sources.isaac_camera_source import IsaacCameraSource
        camera_source = IsaacCameraSource(args.camera_socket)
    else:
        camera_source = SyntheticCameraSource()
    reg.register(CameraPanel(camera_source))
    reg.register(TeleopPanel(host=args.joy_host, port=args.joy_port))
    reg.register(StatePanel())
    if getattr(args, "camera_socket", None):
        # Live World attached → offer mapping-capture control (MAY-173 slam-demo-loop):
        # spawns/monitors the standalone recorder process; see panels/record_panel.py.
        from humanoid.logic.oli.devapp.panels.record_panel import RecordPanel
        reg.register(RecordPanel(camera_socket=args.camera_socket))
    if getattr(args, "map", None):
        from humanoid.logic.oli.devapp.panels.map_panel import MapPanel
        reg.register(MapPanel(args.map))
    if getattr(args, "localizer", None):
        # Localization cockpit (slam-demo-loop): feature view + accuracy + re-hint.
        from humanoid.logic.oli.devapp.panels.loc_panel import LocPanel
        reg.register(LocPanel(
            camera_source,
            candidate=args.localizer,
            map_name=Path(args.loc_map).parent.name + "/" + Path(args.loc_map).name,
        ))
    return reg


def main() -> None:
    ap = argparse.ArgumentParser(description="Oli robot-brain dev app")
    # attach to a World (omit → UI-only). Brain flags mirror brain_main.py.
    ap.add_argument("--socket", default=None,
                    help="World UDS socket to attach the brain to (omit = UI only)")
    ap.add_argument("--mode", choices=["stand", "walk", "glide"], default="walk")
    ap.add_argument("--glide-scale", type=float, default=1.0,
                    help="glide: stick→speed multiplier (launcher passes 3.5 = full-stick 1.75 m/s)")
    ap.add_argument("--joystick", choices=["fixed", "socket"], default="fixed")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--joy-host", default="127.0.0.1")
    ap.add_argument("--joy-port", type=int, default=9001)
    ap.add_argument("--camera-socket", default=None,
                    help="World camera frame channel (SOCK_STREAM); omit → synthetic cameras")
    ap.add_argument("--map", default=None,
                    help="baked occupancy artifact dir (occupancy.npy + occupancy.json) → show the "
                         "Nav Map panel")
    ap.add_argument("--debug-pose", default=None,
                    help="World debug-pose SOCK_DGRAM path → stream Oli's ground-truth pose onto "
                         "the map (pairs with glide_world_main --debug-pose)")
    ap.add_argument("--localizer", default=None, metavar="NAME",
                    help="Nav steers on this localization realization's ESTIMATE (slam-demo-loop "
                         "D7); GT demotes to the map ghost + known-start hint. Requires "
                         "--debug-pose, --loc-map, --camera-socket. Run in bench-<name> env.")
    ap.add_argument("--loc-map", default=None, metavar="DIR",
                    help="the --localizer candidate's baked map dir (e.g. <bake>/pycuvslam_map)")
    ap.add_argument("--walk-after", type=float, default=None)
    ap.add_argument("--duration", type=float, default=0.0)
    # self-validation
    ap.add_argument("--screenshot", default=None,
                    help="render a few frames, save this PNG, then exit")
    ap.add_argument("--frames", type=int, default=20,
                    help="frames to render before the screenshot (default 20)")
    args = ap.parse_args()

    if args.localizer and not (args.debug_pose and args.loc_map and args.camera_socket
                               and args.map and args.socket):
        ap.error("--localizer requires --socket, --debug-pose (hint+ghost), --loc-map, "
                 "--camera-socket and --map (the Nav grid)")

    state = AppState()
    reg = build_registry(args)

    link = None
    if args.socket:
        # Lazy import: only load the brain (and its ONNX policy) when actually attaching.
        from humanoid.logic.oli.devapp.brain_link import BrainLink
        link = BrainLink(
            state,
            socket=args.socket,
            mode=args.mode,
            glide_scale=args.glide_scale,
            joystick=args.joystick,
            vx=args.vx, vy=args.vy, wz=args.wz,
            joy_host=args.joy_host, joy_port=args.joy_port,
            walk_after=args.walk_after, duration=args.duration,
            debug_pose=args.debug_pose, map_dir=args.map,
            localizer=args.localizer, loc_map=args.loc_map,
            camera_socket=args.camera_socket,
        )
        link.start()

    try:
        run(reg, state=state, screenshot=args.screenshot, n_frames=args.frames)
    finally:
        if link is not None:
            link.stop()


if __name__ == "__main__":
    main()
