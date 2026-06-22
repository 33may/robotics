---
name: limx-orphan-subprocess
description: LimX main.py spawns its CLI ability via os.system, which orphans to PID 1 on parent crash and keeps the SDK HTTP port bound — blocks next launch with EADDRINUSE
metadata:
  type: project
---

`humanoid-rl-deploy-python/main.py` launches `python3 -m limxsdk.ability.cli load` via `os.system(...)`. When `main.py` crashes (or is killed without proper teardown), that child gets reparented to PID 1 and keeps its HTTP server bound. Next launch fails with `[Errno 98] Address already in use` → cascade into the SDK segfault pattern.

**Why:** `os.system` does not register the child with the parent's process group, and `main.py` has no `SIGTERM`/`atexit` handler that cleans up its spawned children.

**How to apply:** Our launcher (`humanoid/logic/simulation/mujoco/simulator.py`) runs a `/proc`-based orphan reaper at startup and shutdown. If anyone re-implements bring-up, the reaper is mandatory. If we ever debug a "fresh launch hits EADDRINUSE" → check for orphaned `limxsdk.ability.cli` processes from the previous crash.
