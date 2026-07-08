---
name: feedback-single-entrypoint-no-multiterminal
description: Anton wants ONE command to boot the whole Oli stack (World/sim + brain) — never a 2-3 terminal dance; the single entrypoint is a `launcher` selecting backend + mode.
metadata:
  type: feedback
---

Anton has said **repeatedly** (and was frustrated it wasn't captured): he never wants to
boot Oli by hand across 2 or 3 terminals. There must be **one command** that spawns the whole
stack — the World/sim in its env AND the brain in the brain env — with ordered boot, merged
logs, clean teardown. "I want to have single entrypoint!!"

The decision (2026-07-02): a **single standalone `launcher`** — NOT the dev app owning the
boot. Flags select **what backend to launch** (`mujoco` | `isaac` | `real`) and the **mode**
(stand/walk/glide/forward; mode is also controllable inside the app). It reuses/rebuilds from
the existing `run_oli_sim.py` (isaac) + `run_oli_mujoco.py` (mujoco), which are already the
SAME process-supervisor skeleton (`_spawn`/`_pump`-tee/`_shutdown`/wait-for-`serving`/teardown)
differing only in the ordered process list. Easy version now; a proper OpenSpec spec later.

**Why:** the env split (isaac/limx World vs py3.11 brain) means "single entrypoint" always =
one command spawning N processes — the only question is who supervises. Anton's answer: a
dedicated launcher, so run_oli_sim/run_oli_mujoco collapse into one backend-selectable command.

**How to apply:** never hand Anton a multi-terminal recipe for the sim stack. If a flow needs
World + brain (+ edge + joystick), wire it into the launcher and give him `p .../launcher.py
--sim <backend> --mode <mode> [--dev-app]`. Relates to [[devapp_build_and_validation]],
[[mujoco_world_via_limx_comm_edge]], [[may172_glide_wiring]].

**BUILT 2026-07-02** — the launcher exists at `logic/oli/launcher.py`; full architecture in
[[launcher-single-entrypoint]]. `run_oli_sim`/`run_oli_mujoco` are now shims for it.
