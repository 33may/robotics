"""run_oli_sim.py — SHIM. The Isaac launcher folded into the unified single entrypoint.

The one command to boot Oli against any world is now `logic/oli/launcher.py`:

    p logic/oli/launcher.py --sim isaac --mode glide --dev-app     # was: run_oli_sim --glide --dev-app
    p logic/oli/launcher.py --sim isaac --mode forward --vx 0.3    # was: run_oli_sim --vx 0.3

This file is kept for muscle memory: it forwards everything to the launcher with
`--sim isaac`. Every isaac flag is unchanged; the only rename is the old `--glide` flag →
`--mode glide`, which this shim translates for you. The isaac argv/stage logic now lives in
`logic/oli/launch/backends/isaac.py`; the shared supervisor in `logic/oli/launch/supervisor.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.launcher import main as _launch  # noqa: E402


def _translate(argv: list[str]) -> list[str]:
    """Old run_oli_sim flags → launcher flags (only `--glide` was renamed)."""
    out = ["--sim", "isaac"]
    for a in argv:
        if a == "--glide":
            out += ["--mode", "glide"]
        else:
            out.append(a)
    return out


if __name__ == "__main__":
    print("[run_oli_sim] shim → launcher.py --sim isaac (run it directly next time)",
          file=sys.stderr)
    raise SystemExit(_launch(_translate(sys.argv[1:])))
