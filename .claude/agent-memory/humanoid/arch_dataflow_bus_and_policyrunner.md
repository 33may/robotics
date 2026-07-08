---
name: arch-dataflow-bus-and-policyrunner
description: 2026-07-02 architecture pivot (session = source of truth) — Robot container = Comm + Brain(Reason) + PolicyRunner + dev_app as nodes/processes over a brokerless inner dataflow bus; PolicyRunner is its own process owning the action buffer + two clocks; supersedes June's monolithic-brain / direct-calls model.
metadata:
  type: project
---

**Decision (2026-07-02, "treat this session as source of truth" — Anton confirmed):** Move from the June "monolithic py3.11 brain, direct calls" model to a **node/dataflow architecture (Path 2)**. This session supersedes, where they conflict, the earlier decisions that (a) the policy runner lives inside the one brain process and (b) internals talk via direct calls.

**Shape.** Two top-level blocks: **World container** (Isaac / MuJoCo / Real world processes) and **Robot container**. The Robot container = separate processes/nodes on an **inner dataflow bus**:
- **Comm** — world edge ONLY, pure translator (world-specific prep: reorder / units / PR-space). Holds NO temporal state. RealComm is the py3.8 `limxsdk` edge → forced-separate process anyway.
- **Brain (Reason)** — SLAM / 3D recon / planning; lazy/bursty; emits **intent** (= the existing `PolicyIn` contract, the "WHAT").
- **PolicyRunner** — its OWN process, a *sibling* to Brain (NOT nested inside it). Owns the **action buffer + two clocks**: slow clock = ONNX inference refills the buffer; fast clock = drains the buffer at control rate R, emitting canonical actions (= `PolicyOut`, the "HOW"). The fast drain is driven by a **tick the world/Comm publishes** (sim-step in sim, wall-timer on real) — so rate R (a robot property → invariant) is honored while the timing *source* stays world-specific.
- **dev_app** — observability frontend over the WHOLE Robot container: watches every bus edge / module artifact / processing result. It's the UI of the container, not a separate architectural peer; enabled by the bus being traceable.

**Two protocol classes (session vocabulary):** `WorldProtocols` (outer, world edge — the invariant spine `Observation`-in / `PolicyOut`-out, PLUS explicitly-flagged **auxiliary** interfaces like glide that bypass the policy path) and `BrainProtocols` (inner, fully fixed/invariant — `PolicyIn` + inner topics). The June 3-contract set (`Observation` / `PolicyIn` / `PolicyOut`) are instances of these. Auxiliary (non-invariant) interfaces MUST be clearly flagged, never mixed into the spine.

**Why:** Anton's stated goal this session — "a debuggable, traceable, scalable data system… topics and dataflows consumed by different nodes… parallel isolated systems." PolicyRunner-as-own-process buys crash-isolation (a crashy SLAM burst can't starve the 50Hz drain) + lifecycle independence + escapes the Python GIL between reasoning and the control loop. The shared-memory bus makes the process split near-free and turns "in-brain vs separate" into a deploy-config flip, not a rewrite.

**How to apply:** Comm stays a stateless translator — NEVER put the action buffer or the control-rate loop there. Reason emits intent, never joint actions. PolicyRunner owns all action-execution timing. Internal traffic rides the inner bus (brokerless — see [[feedback-comm-is-world-edge-not-hub]]), never through Comm. Invariance test still holds: Brain/PolicyRunner import neither `isaacsim` nor `limxsdk`.

**Open:** which bus — see [[middleware-inner-comm-research]] (dora-rs is front-runner). Intent granularity (`PolicyIn` semantics — "walk at v=0.3" vs "go to waypoint") is deferred; brain internals are an explicitly-open section.

Related: [[project-invariant-oli-interface]], [[middleware-inner-comm-research]], [[may172_glide_wiring]], [[project_dev_app_stack]], [[agentic-dev-foundation-initiative]].
