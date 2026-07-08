---
name: feedback-comm-is-world-edge-not-hub
description: "Communication" is the robot↔world edge ONLY, not a central hub brokering messages between internal modules; internals (Reason, Action, Orchestrator) talk directly.
metadata:
  type: feedback
---

**Rule:** A "Communication" / IO-boundary layer sits at the edge between the system and the external world ONLY — it is not a central hub that brokers messages between internal modules. Internals talk to each other directly.

**Why:** On MAY-147 I drew the Communication interface as the middleman wiring all components together. Anton corrected it: *"communication interface is only layer between the robot and the world, and internals communicate between each other."* This became design decision D4 — Communication is the only world-aware layer (joint reorder, mechanism, units), and Reason→Action is a single direct seam (no separate Command broker in the middle).

**How to apply:** When diagramming or building a layered runtime, keep the boundary adapter at the boundary. Don't route internal contracts through it. The brain's internal contracts (PolicyIn) cross directly between modules; only Observation/PolicyOut cross the Communication edge to the world.

**Update (2026-07-02):** internal modules now talk over a **brokerless inner dataflow bus** (dora/zenoh peer-mode — see [[arch-dataflow-bus-and-policyrunner]]), not raw direct function calls. The rule still holds: that bus is decentralized point-to-point (no central hub), and Comm still carries ONLY world-edge traffic (Observation/PolicyOut), never internal contracts. "Talk directly" evolved to "talk over the brokerless bus," not "route through Comm."

Relates to [[project-invariant-oli-interface]], [[arch-dataflow-bus-and-policyrunner]].
