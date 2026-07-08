---
name: launcher-single-entrypoint
description: The unified Oli launcher at logic/oli/launcher.py — ONE command (`--sim` backend + `--mode`) boots the whole stack; generic Supervisor + per-world backend plugins; run_oli_sim/run_oli_mujoco are now shims.
metadata:
  type: reference
---

Built 2026-07-02 (MAY-150 side). Anton's hard requirement: [[feedback_single_entrypoint_no_multiterminal]]
— never a 2-3 terminal dance. `run_oli_sim.py` and `run_oli_mujoco.py` were the *same*
process-supervisor twice, differing only in the ordered process list; that duplication is
now extracted once and the two collapse into one backend-selectable command.

**The command:**
```
p logic/oli/launcher.py --sim isaac  --mode glide --dev-app    # drive Oli, live cameras (headline)
p logic/oli/launcher.py --sim isaac  --mode forward --vx 0.3   # constant fwd walk
p logic/oli/launcher.py --sim mujoco --mode walk              # vendor pad live-drive
p logic/oli/launcher.py --sim isaac  --mode walk --dry-run    # print the boot plan, spawn nothing
```
`--sim {isaac,mujoco,real}` picks the world; `--mode {stand,walk,forward,glide}` picks the
brain behavior. Glide is no longer a separate `--glide` flag — it's `--mode glide` (isaac-only).

**Architecture (`logic/oli/launch/`):**
- `supervisor.py` — `Supervisor` + `Stage`. The ONE shared copy of spawn / tee-to-merged-log /
  SIGINT→TERM→KILL-per-group teardown / serving-marker gating / socket-file wait / supervise-
  core-until-exit. Backend-agnostic (stdlib only, no isaacsim/limxsdk).
- `backend.py` — the `Backend` contract (a module exposing `NAME`, `add_args(ap)`,
  `stages(a)->list[Stage]`, optional `reap(label)`).
- `backends/{isaac,mujoco,real}.py` — one plugin per world; `backends/__init__.py` holds
  `REGISTRY = {m.NAME: m}`. **Adding a world = one module + one registry line**; the launcher
  and supervisor never change. `real` is reserved (its `stages()` raises "not wired yet").
- `launcher.py` — thin CLI: two-phase parse (resolve `--sim`, then the backend adds its own
  flags → backend flags stay isolated), dispatch, `--dry-run`. Self-bootstraps `sys.path` so
  it runs BOTH as `p logic/oli/launcher.py …` (bare script) and `python -m humanoid.logic.oli.launcher`.

`Stage = (name, argv, cwd, serving_marker, wait_for_path, env, core, boot_delay)`. isaac plan =
[world(gate "serving on"), brain]; mujoco plan = [sim, edge(3s settle, gate "serving the brain"),
brain, +walk: joy-bridge & vendor pad as non-core extras]; mujoco keeps its orphan `reap`.

**Shims:** `run_oli_sim.py` / `run_oli_mujoco.py` now forward to the launcher with `--sim
isaac`/`--sim mujoco` (translate old `--glide`→`--mode glide`; drop mujoco's deprecated no-op
`--spawn-app`) and print a one-line "run launcher directly next time" note. Non-destructive.

**Tests:** `tests/oli/world/test_run_oli_sim.py` (isaac backend argv/stages), `…/test_run_oli_mujoco.py`
(mujoco backend), `…/test_launcher.py` (registry + two-phase parse isolation + fused-plan +
dry-run + real-raises). All `@pytest.mark.brain`, pure (no spawn). 66 green with the devapp suite.
Relates to [[devapp_build_and_validation]], [[mujoco_world_via_limx_comm_edge]], [[may172_glide_wiring]].
