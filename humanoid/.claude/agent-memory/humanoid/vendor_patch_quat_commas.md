---
name: vendor-patch-quat-commas
description: humanoid-description HU_D04_01.xml ships invalid comma-separated quat attrs on hand_manip bodies; MuJoCo rejects, patch in place
metadata:
  type: project
---

`humanoid-description` submodule (pin `63eaa67`) ships `HU_D04_description/xml/HU_D04_01.xml` with `quat="0.707107, 0, 0.707107, 0"` on the `left_hand_manip` and `right_hand_manip` bodies (lines 328, 383). MuJoCo's XML parser splits `quat` on whitespace and rejects commas with `ValueError: XML Error: bad format in attribute 'quat'`. Patched in place 2026-06-19 to `quat="0.707107 0 0.707107 0"`.

**Why:** First boot of `simulator.py` fails with this error. The patch is recorded in `humanoid/docs/vendor/humanoid-mujoco-sim.md` § 10 as a vendor patch.

**How to apply:** If the submodule is ever reset (e.g. `git submodule update --force`), grep for `quat="[^"]*,'` in the description XMLs and re-apply space substitution. Long-term consider forking `humanoid-description` and pointing the submodule at our fork.

Related: [[vendor-humanoid-mujoco-sim]] for full vendor context.
