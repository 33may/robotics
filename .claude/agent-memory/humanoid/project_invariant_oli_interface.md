---
name: project-invariant-oli-interface
description: MAY-147 architecture — a deployment-invariant Oli brain (Reason+Action) drives sim and real unchanged; the brain runs in a dedicated py3.11 env (NOT py3.8 — that's only the deferred RealComm edge); World is an independent process in sim AND real; three canonical-PR contracts.
metadata:
  type: project
---

**Decision (2026-06-22, MAY-147 design):** The target architecture is a single deployment-invariant `Oli` interface (in future a `Humanoid` base class). The same code/logic drives the robot in **Sim** and in **Real**; the realization is selected by a flag, not by different call sites. Two contracts bound the system:

- **World ↔ Action Interface** — obs in / actions out toward the body (sim articulation or real robot).
- **Reason ↔ Action Interface** — the reasoning layer ↔ the action layer.

**RESOLVED (2026-06-23, design `may-147-oli-deployment-interface`):** the brain (Reasoning + Action, including the ONNX policy runner) runs in a dedicated **py3.11 `brain` env — NOT py3.8**. The py3.8 `limxsdk` ABI lives ONLY in the deferred **RealComm** edge, which is part of the **Communication** layer, not the brain. So the seams are: **World** (independent process — sim = Isaac + Oli + `SimComm` server, py3.11; real = robot + `RealComm`, py3.8) ⇄ **Communication** (the only world-aware layer; brain-side `BrainComm` client) ⇄ **Reasoning** (Teleop → PolicyIn) ⇄ **Action** (PolicyRunner/ONNX). Three canonical-PR contracts: `Observation` (World→Reason), `PolicyIn` (Reason→Action), `PolicyOut` (Action→World). The brain is the connection client; the World is the always-on server.

**Why:** Anton wants reasoning/action modules built once and deployed unchanged to sim and real — his words: "the main idea is to have the single Oli robot interface … where it could be easily deployed in Sim and in Real using the same code and logic" and "the same interface that is invariant." The Py3.8(SDK)↔Py3.11(Isaac) ABI split, plus the kinematic_projection bus-relay blocker, pushed the design toward process/contract boundaries instead of one shared runtime.

**How to apply:** Preserve the invariant boundary — the brain imports NEITHER `isaacsim` NOR `limxsdk` (enforced as a test); all world-specifics (joint order, mechanism, units) live in Communication. Don't bake sim-only/real-only assumptions into Reason/Action; the only thing that swaps between sim and real is the Communication realization. The policy runner is brain-side (py3.11), not the py3.8 SDK process — and per the **2026-07-02 pivot** ([[arch-dataflow-bus-and-policyrunner]]) it now runs as **its own py3.11 process (`PolicyRunner`), a sibling to Reason over an inner dataflow bus**, no longer bundled in one monolithic brain process. That pivot supersedes the "one brain process / direct calls" shape below where they conflict; the invariance boundary, the no-isaacsim/no-limxsdk test, and the 3 canonical contracts all still hold.

Related: [[arch-dataflow-bus-and-policyrunner]] (2026-07-02 process-model evolution), [[limx_kinematic_projection_bus_relay]], [[isaac_pd_implicit_drive]], [[walk_policy_obs_builder_fidelity]], [[feedback_tests_in_repo_tdd]], [[limx-sdk-role-gating]], [[humanoid-summer-2026-plan]]. Spec home: OpenSpec change `may-147-oli-deployment-interface` (supersedes `may-147-isaac-limx-sdk-bridge`).
