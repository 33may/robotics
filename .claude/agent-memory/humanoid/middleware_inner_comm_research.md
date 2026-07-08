---
name: middleware-inner-comm-research
description: 2026-07-02 research verdict on the inner-comm bus — MROS is closed/wheel-gated (matching buys nothing), full ROS2 not justified for us, observability is orthogonal (MCAP/rerun); bus candidates are dora-rs (front-runner) vs standalone Zenoh+MCAP vs custom in-process; final pick pending.
metadata:
  type: reference
---

**Context:** picking the inner dataflow bus for the Robot container (Path 2, see [[arch-dataflow-bus-and-policyrunner]]). Two research fan-outs, 2026-07-02.

**Verdicts:**
- **MROS is irrelevant to our bus choice.** LimX's MROS is a closed, ROS2-lineage fork, wheel-gated behind `limxsdk` and role-gated; the public web has zero docs on it and `kinematic_projection` routing isn't surfaced in Python. We can't speak it and gain nothing by matching it — the SDK is the boundary regardless of our internal bus. (Reinforces [[limx-sdk-role-gating]], [[research_ros2_bridger_alternative]].)
- **Full ROS2 is not justified** for a single-machine Python sim-first project: no official Fedora binaries (RHEL-only), `rclpy` is a second-class perf citizen (no Python shared-memory zero-copy path), heavy install that fights our conda envs.
- **Observability is ORTHOGONAL to the bus.** Best-in-class tracing = **MCAP** logs + **rerun** (2026 robot-learning viz standard) / Foxglove, bolt-on to ANY transport. Pick the bus for dataflow reasons; get debuggability regardless.

**Bus candidates (pick pending):**
- **dora-rs — front-runner.** Declarative YAML dataflow graph; native Python nodes (not bridged); zero-copy shared-mem local → auto-Zenoh when a node moves to another machine (same graph ⇒ "in-proc now, real robot later" is a config change, not a rewrite). Native OpenTelemetry + record/replay-with-node-substitution. LeRobot-adjacent (`dora-lerobot`, first-party `dora-rerun`; Thomas Wolf endorsed). RISK: young — v0.5 line, 1.0 RC, ~3.8k stars, thin ecosystem, some fault-tolerance features partial.
- **Standalone Zenoh + MCAP** — lean, mature transport, pip-clean on Fedora, peer→router growth path; more DIY (you wire the node graph + observability yourself). Note dora uses Zenoh under the hood for remote transport.
- **Custom in-process asyncio + pydantic bus** — simplest, zero runtime dep, but single-process only + you build all observability; only right if single-process is permanent (pair with rerun). `aiopubsub` is the closest off-the-shelf base.

**How to apply:** when we pick, prefer dora unless its maturity bites; either way log to MCAP + view in rerun from day one. Reserve full ROS2 / `rmw_zenoh` for the day we're genuinely multi-machine. Full sourced findings in the 2026-07-02 session log.
