"""real.py — reserved backend for the physical Oli (deploy edge).

The interface exists so `--sim real` is a first-class choice and the launcher's help is
complete, but the deploy edge that serves the brain socket from the real robot is not
wired yet. When it lands, fill in `add_args` (robot IP, safety limits, …) and `stages`
(the deploy edge + brain), and everything downstream — supervisor, logging, teardown —
works unchanged.
"""

from __future__ import annotations

import argparse

from ..supervisor import Stage

NAME = "real"


def add_args(ap: argparse.ArgumentParser) -> None:
    # No flags yet — reserved. (Robot IP / safety limits land with the deploy edge.)
    pass


def stages(a: argparse.Namespace) -> list[Stage]:
    raise SystemExit(
        "--sim real is reserved but not wired yet (the deploy edge that serves the brain "
        "from the physical robot is pending). Use --sim isaac or --sim mujoco.")
