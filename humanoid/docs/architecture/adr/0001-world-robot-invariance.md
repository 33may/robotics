# ADR-0001: World/Robot invariance

- **Status:** accepted
- **Date:** 2026-07-06

## Context

The Oli brain must deploy unchanged across Isaac, MuJoCo, and the real robot — which differ in joint order, units, mechanism, and even Python ABI (isaacsim 3.11 vs limxsdk 3.8).

## Decision

The Robot (Reason + PolicyRunner) is world-invariant: it consumes a fixed input set and emits a fixed output set, and imports **neither** `isaacsim` **nor** `limxsdk`. All world-specific translation lives in `Comm`.

## Why

Build the intelligence once, deploy it everywhere. The ABI split plus joint/mechanism differences force a hard process/contract boundary rather than one shared runtime.

## Alternatives rejected

- **One shared runtime** — impossible: the limxsdk py3.8 ABI and Isaac py3.11 cannot coexist in one interpreter.
- **Per-world brain branches** — drifts over time and defeats the point of invariance.

## Consequences

Enforced by a test (`brain`/`PolicyRunner` import nothing world-specific) + the env split + the `brain` pytest marker. Adding a World becomes "write a `Comm` adapter," not "touch the brain." All complexity concentrates in `Comm`.
