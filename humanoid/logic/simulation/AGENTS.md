# simulation/ — World implementations

Each subdir is a **World** (`isaacsim/`, `mujoco/`, `real/`) — the body + physics. A World **applies** commands and **reports** state; it holds no brain logic.

## Rules

- **Satisfy the WorldProtocols spine (§4):** report `STATE_IMU` (`Observation`) + `CAMERA_FRAME`; accept `CMD` (`PolicyOut`); integrate `GLIDE_CMD` when in glide mode.
- **World-specifics are allowed here** — this is the World side of the boundary. Everything crossing to the Robot must be in canonical spine form (Comm does the Robot-side translation).
- **Apply + report only** — no policy, no decision-making, no action buffering.
- `isaacsim/` is py3.11 (direct `SimComm`); `mujoco/` + `real/` go through the py3.8 `limxsdk` edge.
- `walkmatch/` is a sim-to-sim fidelity harness, not a World.

See `docs/architecture/architecture.md` §4–§5.
