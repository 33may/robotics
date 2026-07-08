---
name: asset-usd-symlink-vendor
description: humanoid/assets/oli is a symlink INTO the vendor submodule; the USD asset space is untracked/regenerable and (per Anton 2026-07-01) the vendor tree is treated as our own working repo.
metadata:
  type: project
---

`humanoid/assets/oli` → symlink → `vendor/humanoid-mujoco-sim/humanoid-description/HU_D04_description`.
So `assets/oli/usd/` **is** the vendor submodule's `usd/` dir — there is NO separate project
copy. The USD files there (incl. `HU_D04_01_sensor.usd`, all `HU_D04_01_rl*.usd`) are
**untracked** in the vendor submodule and are generated/scratch build artifacts:
`build_rl_usd.py` and `build_camera_usd.py` both write into this space.

**Decision (Anton, 2026-07-01):** treat the vendor tree as **our own working repo** for now
— it's fine to bake generated assets (e.g. the MAY-149 cameras) into the vendor USDs via the
symlink. No separate project copy needed. The parent repo's ` m vendor/humanoid-mujoco-sim`
dirty marker is expected (untracked generated USDs), not a problem.

**Why it's safe:** the *tracked build script is the source of truth* (design.md D3). The USD
binary is disposable — `python logic/simulation/isaacsim/build_camera_usd.py` (and `--rl`)
regenerates the cameras after any submodule reset. A pristine backup of the original sensor
layer also exists at `/home/may33/vendor_backups/2026-06-22/…/HU_D04_01_sensor.usd`.

**How to apply:** don't waste effort trying to keep vendor USDs "pristine" or making project
copies; write generated assets straight into the symlinked space and rely on the build script.
The isaac loader path `oli.py` DEFAULT_USD = `assets/oli/usd/HU_D04_01.usd` resolves through
this symlink. Related: [[oli_perception_camera_design]], [[vendor_humanoid_mujoco_sim]],
[[vendor-submodule-conversion]], [[feedback_no_delete_without_guidance]].
