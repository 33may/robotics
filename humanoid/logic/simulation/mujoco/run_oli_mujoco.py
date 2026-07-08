"""run_oli_mujoco.py — SHIM. The MuJoCo launcher folded into the unified single entrypoint.

The one command to boot Oli against any world is now `logic/oli/launcher.py`:

    p logic/oli/launcher.py --sim mujoco --mode walk               # vendor pad live-drive
    p logic/oli/launcher.py --sim mujoco --mode forward --vx 0.3   # stand→walk, no joystick

This file is kept for muscle memory: it forwards everything to the launcher with
`--sim mujoco`. Every mujoco flag is unchanged; the old no-op `--spawn-app` is dropped. The
mujoco argv/stage logic (sim→edge→brain + vendor pad/bridge + orphan reaping) now lives in
`logic/oli/launch/backends/mujoco.py`; the shared supervisor in `logic/oli/launch/supervisor.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.launcher import main as _launch  # noqa: E402


def _translate(argv: list[str]) -> list[str]:
    """Old run_oli_mujoco flags → launcher flags (`--spawn-app` was a deprecated no-op)."""
    out = ["--sim", "mujoco"]
    for a in argv:
        if a == "--spawn-app":
            continue
        out.append(a)
    return out


if __name__ == "__main__":
    print("[run_oli_mujoco] shim → launcher.py --sim mujoco (run it directly next time)",
          file=sys.stderr)
    raise SystemExit(_launch(_translate(sys.argv[1:])))
