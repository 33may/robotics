"""launcher.py — ONE command to boot Oli against ANY world.

    p logic/oli/launcher.py --sim isaac  --mode glide --dev-app     # drive Oli, live cameras
    p logic/oli/launcher.py --sim isaac  --mode forward --vx 0.3    # constant forward walk
    p logic/oli/launcher.py --sim mujoco --mode walk               # vendor pad live-drive
    p logic/oli/launcher.py --sim isaac  --mode walk --dry-run     # print the boot plan only

`--sim` picks the world backend (isaac | mujoco | real); `--mode` picks the brain behavior
(stand | walk | forward | glide). The chosen backend describes an ordered list of child
processes; the generic `Supervisor` boots them, tees every log into one live stream, and
tears the stack down on Ctrl-C or any core process exiting. No 2-3 terminal dance.

Runs with any plain Python 3 (stdlib only — it imports NO isaacsim/limxsdk/contracts; it
just orchestrates `conda run` subprocesses). Backends are plugins under `launch/backends/`;
adding a world is one module + one registry line — the launcher itself never changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Runnable BOTH as a bare script path (`p logic/oli/launcher.py …`) and as a module
# (`python -m humanoid.logic.oli.launcher`). As a script there is no package context for
# relative imports, so we put the repo root on sys.path and import absolutely (a no-op
# when already importable via -m).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.launch.backends import REGISTRY  # noqa: E402
from humanoid.logic.oli.launch.supervisor import Supervisor  # noqa: E402


def _add_common_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared by every backend (transport + brain behavior)."""
    ap.add_argument("--socket", default="/tmp/oli-world.sock",
                    help="UDS the World serves and the brain connects to")
    ap.add_argument("--log", default="/tmp/oli-launch.log",
                    help="shared log file for the merged, timestamped multi-process stream")
    ap.add_argument("--boot-timeout", type=float, default=240.0,
                    help="seconds to wait for the World/edge to serve before giving up")
    ap.add_argument("--mode", choices=["stand", "walk", "forward", "glide"], default="forward",
                    help="brain behavior: stand=analytic hold; walk=joystick-steered; "
                         "forward=constant fwd walk (default); glide=velocity-driven (isaac)")
    ap.add_argument("--vx", type=float, default=None,
                    help="forward/glide: override fwd speed (default 0.3 m/s)")
    ap.add_argument("--vy", type=float, default=None)
    ap.add_argument("--wz", type=float, default=None)
    ap.add_argument("--joy-port", type=int, default=9001)
    ap.add_argument("--walk-after", type=float, default=None,
                    help="start STAND, switch to WALK after N s (scripted demo)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="wall seconds for the run (0 = until the window closes)")
    ap.add_argument("--dev-app", action="store_true",
                    help="boot the windowed dev app as the brain (cameras + teleop + live "
                         "state) instead of the headless brain — isaac only for now")
    ap.add_argument("--brain-env", default="brain")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved boot plan (stages + argv) and exit — spawns nothing")


def build_args(argv=None):
    """Two-phase parse: resolve --sim first, then let the chosen backend add its flags.

    Returns (parsed_args, backend_module). Kept separate from main() so tests can assert
    the resolved argv/stages without spawning anything.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sim", choices=sorted(REGISTRY), default="isaac")
    known, _ = pre.parse_known_args(argv)
    backend = REGISTRY[known.sim]

    ap = argparse.ArgumentParser(
        description="Boot Oli against any world (isaac/mujoco/real) from one command.")
    ap.add_argument("--sim", choices=sorted(REGISTRY), default="isaac",
                    help="world backend to launch (default: isaac)")
    _add_common_args(ap)
    backend.add_args(ap)
    return ap.parse_args(argv), backend


def main(argv=None) -> int:
    args, backend = build_args(argv)
    stages = backend.stages(args)              # backend may SystemExit on unsupported combos
    reap = getattr(backend, "reap", None)
    sup = Supervisor(log_path=args.log, boot_timeout=args.boot_timeout, reap=reap)
    return sup.run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
