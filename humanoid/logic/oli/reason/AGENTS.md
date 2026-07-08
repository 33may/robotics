# reason/ — the Brain (world-invariant decision layer)

`Reason` is the world-invariant decision layer — perception + planning (SLAM, 3D reconstruction, motion planning, teleoperation). It decides **what** to do and emits `intent` (`PolicyIn`).

## Rules

- **Invariance boundary (hard):** import **neither** `isaacsim` **nor** `limxsdk`, directly or transitively. This is the deployment-invariant core — the `brain` pytest marker guards it.
- **Emit intent, never joint commands.** Reason produces `intent` / `PolicyIn` (the WHAT); turning it into joint actions is `PolicyRunner`'s job.
- **Consume invariant inputs only** — `observation`, `camera_frame` off the bus. Never read world-specific structs.
- **Lazy / bursty** — think at your own pace; do not run a control-rate loop here.
- Internal submodule structure is still open — shape it as the reasoning stack is built.

See `docs/architecture/architecture.md` §6–§7.
