---
name: vendor-submodule-conversion
description: How to convert an embedded vendor clone to a proper submodule without losing local patches, and the current submodule state of the robotics repo.
metadata:
  type: project
---

When a vendor under `humanoid/vendor/` (or any vendor path) exists as an embedded `git clone` with no `.gitmodules` entry, convert it to a proper submodule by patch-extract → re-clone via `git submodule add` → re-apply.

**Why:** This came up 2026-06-22 with `humanoid/vendor/humanoid-rl-deploy-python`. The directory was an embedded clone (`.git/` inside) with local `controllers.yaml` edits (`autostart: true`) plus `__pycache__` noise. Naively `git add`-ing it makes a bare gitlink with no `.gitmodules` registration — that's the pattern the older vendors (`humanoid-mujoco-sim`, `robosuite_src`, `SO-ARM101`) use, which means `git clone --recursive` of the parent can't fetch them. The proper-submodule path keeps patches AND makes the parent cloneable elsewhere.

**How to apply:**

1. Audit before touching. Inside the embedded clone:
   - `git remote get-url origin` (note exact URL form — SSH vs HTTPS)
   - `git rev-parse HEAD` (the upstream commit to pin to)
   - `git status -s` + `git ls-files --others --exclude-standard` (separate real edits from noise)
2. Extract real edits as a patch:
   `git diff <real-files> > /tmp/<name>.patch`
3. Move the embedded clone aside (don't delete — it's your last backup of any local-only refs/stashes):
   `mv humanoid/vendor/<name> /tmp/<name>-pre-conversion`
4. Add as proper submodule from the parent repo:
   `git submodule add <exact-origin-url> humanoid/vendor/<name>`
5. Pin to the same upstream commit:
   `cd humanoid/vendor/<name> && git checkout <sha>`
6. Re-apply local patch from the parent repo:
   `git apply --directory=humanoid/vendor/<name> /tmp/<name>.patch`
7. Verify with `git diff` inside the submodule — diff should match step 2.

**Current submodule state in `33may/robotics` (2026-06-22):**

| Path | URL | Pinned | `.gitmodules` entry | `ignore` | Local patches |
|---|---|---|---|---|---|
| `humanoid/vendor/humanoid-rl-deploy-python` | `https://github.com/limxdynamics/humanoid-rl-deploy-python.git` | `6d8771c` | yes (first proper one) | none (so yaml patches stay visible) | `controllers/HU_D{03_03,04_01}/controllers.yaml` autostart |
| `humanoid/vendor/humanoid-mujoco-sim` | `git@github.com:limxdynamics/humanoid-mujoco-sim.git` | `59424de` | yes | `untracked` | `simulator.py` `_cmd_received` gate + `mj_forward` init |
| `old/diffusion/robosuite_src` | `https://github.com/ARISE-Initiative/robosuite.git` | `a14951c` (local-only commit!) | yes | `untracked` | none, just untracked noise |
| `robots/SO-ARM101` | `git@github.com:TheRobotStudio/SO-ARM100.git` | `608122e` | yes | `untracked` | none, just untracked USD/py WIP |

**Critical gotcha:** Vendor patches live in **submodule working trees only**, not in any pushed commit on the vendor origins. A fresh `git clone --recursive 33may/robotics` on another machine gets clean upstream submodules — no patches. This is what [[tasks/may-146-decide-vendor-patch-fork-strategy]] needs to resolve (fork vs `.patch` files in parent). Backups at `/home/may33/vendor_backups/2026-06-22/` (3.0 GB, full incl. `.git`) are the only durable copy until then.

**Why `ignore = untracked` on three of them:** suppresses the constant ` m` noise from logs/caches/WIP files inside the submodule, while still letting real `modified content` show up — so future WIP edits to e.g. `simulator.py` won't go silent. The `humanoid-rl-deploy-python` one is deliberately left without `ignore` because the yaml patches *are* modified content and need to stay visible.

Related: [[oli_main_software_tarball]], [[vendor_humanoid_mujoco_sim]].
