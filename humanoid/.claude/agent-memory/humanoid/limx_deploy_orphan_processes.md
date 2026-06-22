---
name: limx-deploy-orphan-processes
description: humanoid-rl-deploy-python main.py spawns ability cli via os.system; if main.py SIGSEGVs the child becomes init-reparented orphan and holds the SDK HTTP port, breaking the next run with "Address already in use"
metadata:
  type: project
---

`humanoid-rl-deploy-python/main.py` launches the ability framework as a subprocess via `os.system("python3 -m limxsdk.ability.cli load ...")`. The framework binds an HTTP control port. If `main.py` then SIGSEGVs (we hit this when the `click` dep was missing, and again sporadically), the ability subprocess gets reparented to PID 1 and keeps the port.

The next launch then fails with `Ability - ERROR - Failed to start HTTP server: [Errno 98] Address already in use` — and `main.py` segfaults again, likely from interacting with a half-initialised framework.

**Why:** Process-group `SIGTERM` from our launcher (`os.killpg(setsid)`) misses the orphan because it's no longer in the group after reparenting.

**How to apply:** The MuJoCo launcher (`humanoid/logic/simulation/mujoco/simulator.py`) reaps orphans at both startup and shutdown via `reap_orphans()` — scans `/proc` for cmdlines matching `humanoid-mujoco-sim/simulator.py`, `humanoid-rl-deploy-python/main.py`, `limxsdk.ability.cli`, `robot-joystick/robot-joystick`, `prebuild/kinematic_projection` and SIGTERM/SIGKILLs them. If a similar bring-up is built elsewhere (Isaac side, real-robot side), copy that pattern.

Manual cleanup if needed: `pkill -f 'limxsdk.ability.cli|humanoid-rl-deploy-python|humanoid-mujoco-sim'`

Related: [[limxsdk-undeclared-deps]], [[vendor-humanoid-rl-deploy-python]].
