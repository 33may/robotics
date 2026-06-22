---
name: isaac-oli-smoke-loader
description: Isaac Sim already has a working HU_D04_01 USD loader at humanoid/logic/simulation/isaacsim/load_oli.py with imported asset under humanoid/assets/oli/usd/.
metadata:
  type: project
---

The Isaac side of the humanoid project is **not** at zero — there is already a working smoke-loader and the USD assets are imported into the project tree.

**Imported asset (PR / serial-URDF space):**
- `humanoid/assets/oli/usd/HU_D04_01.usd` — the asset `load_oli.py` references at `/World/Oli`.
- Source bundle (layered base/physics/sensor/robot variants) lives in vendor at `humanoid/vendor/humanoid-mujoco-sim/humanoid-description/HU_D04_description/usd/`.

**Smoke loader:** `humanoid/logic/simulation/isaacsim/load_oli.py`
- Boots Isaac Sim with the **full Kit** experience (`isaacsim.exp.full.kit`) so the editor UI is available — Stage panel, Property panel, Robotics/Physics/Sensors menus.
- Loads USD via `add_reference_to_stage` at `/World/Oli`, then lifts the base to `SPAWN_HEIGHT_M = 1.05 m` (before `world.reset()`, because `fixRootLink` pins at current pose at reset).
- Walks `Usd.PrimRange` to find the `ArticulationRootAPI` prim and applies `physxArticulation:fixRootLink = True` via the raw attribute (version-tolerant across Isaac Sim releases).
- After reset, builds a `SingleArticulation`, prints the DOF index → name table (this is the **Isaac-side joint order** — distinct from the SDK PR order, will need a permutation map when wiring `limxsdk`).
- Runs `world.step(render=True)` until the window closes. No joints driven yet — arms/legs hang limp under default soft drives (`k_p≈1.7, k_d≈0.017`).

**Why:** This is the right baseline to extend for MAY-147 (wire limxsdk loop in Isaac). Do not re-import the USD or write a new loader.

**How to apply:**
- For MAY-147, extend `load_oli.py` (or fork it into `oli_sim_peer.py` next to it) by adding the limxsdk sim-peer construction, `subscribeRobotCmdForSim` handler, and `publishRobotState`/`publishImuData` at the physics tick.
- Build the **Isaac DOF order ↔ SDK PR-order** permutation table the first time using the printed table from `load_oli.py` — the SDK PR order is documented in `humanoid/docs/vendor/humanoid-rl-deploy-python.md § 11`.
- The SDK `publishRobotCmd` interface explicitly states it operates in **serial URDF space** with the low-level system handling parallel-mechanism conversion transparently (`oli-corpus://sdk-guide#5.1.6?part=1`). So if our USD is the serial-URDF model (which the vendor's USD bundle is), we do NOT need to implement PR↔AB conversion on the Isaac side — apply commands directly to the articulation joints by name.

Related: [[vendor-humanoid-mujoco-sim]], [[reference-oli-corpus-mcp]], [[isaac-sim-over-lab]].
