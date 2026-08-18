# inspection — language-driven UR5e inspection

## Purpose
VLM decides how to inspect an object; this package perceives, models the
workcell, picks viewpoints, solves motion, and orchestrates the loop.
Pure Python, no ROS. Stack: pinocchio+coal, pyroboplan, EAIK, ur_rtde,
open3d, meshcat.

## Files
Pipeline shape: perceive (`perception/`) → model world (`cell/`) → pick
view (`view/`) → solve motion (`motion/`) → orchestrate (`run/`).
`calib/` holds registration scripts and data. Top-level `ik.py` and
`safety.py` are SO-101 legacy, kept only until `motion/` replaces them.

## Contracts & decisions
- Self-contained: new code never imports `vbti.*`. When touching a file
  that still has a vbti import, migrate the dependency into this package.
- Dependency direction: run → view/motion → cell ← perception.
  `cell/` imports nothing from siblings. Nothing imports from `run/`.

## Does NOT belong here
- ROS, launch files, message definitions.
- Anything that grows a new `vbti.*` dependency.
