# ADR-0003: PolicyRunner is its own process

- **Status:** accepted
- **Date:** 2026-07-06

## Context

Policy inference (ONNX / VLA action chunks) and heavy bursty reasoning (SLAM, 3D reconstruction) in one process cause GIL contention and crash coupling; a ~50 Hz control drain must not be starved.

## Decision

`PolicyRunner` is a **separate process** (sibling to Reason), owning the action buffer + two clocks: a slow clock runs inference to refill the buffer, a fast clock drains it at control rate R off the world tick. Supersedes the June decision that placed the policy runner inside the single brain process.

## Why

Crash-isolation + lifecycle independence + escaping the Python GIL between reasoning and the control loop. Marginal cost is ≈ 0 given the shared-memory dataflow bus (ADR-0002).

## Alternatives rejected

- **In-brain (one process)** — GIL contention and shared-crash coupling.
- **Buffer + drain in Comm** — overloads the World edge and breaks the invariance of execution timing (rate R is a robot property, not a world one).

## Consequences

`PolicyIn` = the WHAT, `PolicyOut` = the HOW; glide bypasses PolicyRunner entirely. Separate-vs-colocated deployment becomes a Launcher config flip, not a rewrite.
