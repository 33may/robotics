# ADR-0002: Inner comm is a brokerless dataflow bus

- **Status:** accepted (specific bus implementation TBD)
- **Date:** 2026-07-06

## Context

The Robot-container modules (Reason, PolicyRunner, Comm, dev_app) run as separate processes and must exchange data. We want it debuggable, traceable, and scalable. The earlier (June) design was a monolithic brain with direct calls.

## Decision

Internal communication is a **brokerless node/topic dataflow bus** (the BrainProtocols). The specific bus — dora-rs vs Zenoh+MCAP vs a custom in-process/socket bus — is **not yet chosen**. Observability (MCAP logs + rerun/Foxglove) is treated as orthogonal to the bus.

## Why

A node/dataflow model gives parallel isolated nodes plus per-edge traceability; brokerless keeps `Comm` the World edge only, not an internal hub.

## Alternatives rejected

- **Full ROS2** — Fedora/rclpy tax, and no interop benefit: LimX's MROS is a closed, wheel-gated fork, so matching it buys nothing.
- **Monolithic brain + direct calls** — doesn't scale and isn't traceable.

## Consequences

PolicyRunner-as-its-own-process (ADR-0003) becomes near-free. The bus pick stays open (architecture §11). We log to MCAP and view in rerun from day one, independent of the bus choice.
