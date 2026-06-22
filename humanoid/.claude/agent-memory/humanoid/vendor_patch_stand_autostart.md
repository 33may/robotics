---
name: vendor-patch-stand-autostart
description: humanoid-rl-deploy-python controllers.yaml stand.autostart flipped to true so deploy publishes RobotCmd immediately on bring-up; closes the free-fall gap
metadata:
  type: project
---

`humanoid-rl-deploy-python/controllers/{HU_D04_01,HU_D03_03}/controllers.yaml` ships `abilities.stand.autostart: false`. With our launcher (`humanoid/logic/simulation/mujoco/simulator.py`), that leaves a ~2 second window after sim bring-up where no `RobotCmd` is on the bus — the MuJoCo PD law sees `Kp = Kd = 0` and Oli free-falls before standing.

Patched to `autostart: true` on both files 2026-06-19. Stand controller now starts publishing the instant the deploy framework establishes its bus connection — Oli stands within ~tens of ms of sim being up. Joystick switching between modes still works unchanged.

**Why:** First user-visible bring-up was the free-fall + snap, which looks broken. Documented under § 10 of `humanoid/docs/vendor/humanoid-rl-deploy-python.md`.

**How to apply:** If the deploy repo is re-cloned, re-flip `autostart` to `true` for the `stand` ability in both YAMLs. Long-term consider forking the repo and pointing our docs there.

Related: [[vendor-humanoid-rl-deploy-python]], [[limxsdk-undeclared-deps]], [[limx-deploy-orphan-processes]].
