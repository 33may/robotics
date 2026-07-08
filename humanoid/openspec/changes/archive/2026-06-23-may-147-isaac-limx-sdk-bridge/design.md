## Context

`isaac-limx-sdk-bridge` is the plumbing that lets the Isaac Sim viewport stand in for a real Oli on the LimX MROS bus. Without it, the project's primary 3D environment cannot exchange packets with any of LimX's policy stack or with our own RL code that uses the SDK.

The central constraint is settled and proven:

- The `limxsdk-4.0.1-py3-none-any.whl` shipped in `humanoid/vendor/humanoid-mujoco-sim/limxsdk-lowlevel/python3/amd64/` contains `_robot.so` plus its own bundled `libpython3.8.so.1.0`. The "py3-none-any" wheel tag is misleading; the binary is hard-locked to the CPython 3.8 ABI. Reverify with `unzip -l limxsdk-4.0.1-py3-none-any.whl | grep libpython` if needed.
- Isaac Sim 5.x requires CPython 3.11 (4.x = 3.10). No Isaac Sim version supports 3.8. No LimX-published wheel supports 3.10/3.11. PyPI carries no `limxsdk` at all.
- Subinterpreters / dual-libpython loading would conflict at link time and produce undefined symbol resolution. Not viable.
- Reimplementing MROS in pure Python is multi-week reverse-engineering (transport not documented; protocol lives inside `_robot.so`).

A parallel external ask to LimX for proper 3.10/3.11 wheels is in flight (separate track); a background research subagent is currently mapping LimX's GitHub footprint to see if they have already solved this internally for TRON1 or any other robot. The bridge designed here assumes the worst case (no wheel forthcoming) and is structured to be deletable if LimX delivers.

The wire contract on the MROS bus side is fully captured — empirically by MAY-145 (`humanoid/docs/vendor/humanoid-rl-deploy-python.md` § 11) and structurally by the canonical headers (`oli-corpus://limxsdk#datatypes.h`). The two agree byte-for-byte. § 11 is the source of truth for any disagreement, since it documents what actually transits the bus.

- **`RobotState` — 5 fields, all sized to 31 motors.** `stamp` (uint64 ns), `tau` / `q` / `dq` (31 × float32 each, PR-order), `motor_names` (31 × string, PR-order). Empirical publish rate from the MuJoCo reference: **~884.5 Hz** (MJCF nominal 1 kHz; the deficit is viewer + Python loop overhead).
- **`ImuData` — 4 fields, fixed-size C arrays.** Struct field order is `stamp` (uint64 ns), `acc[3]` (float, m/s², body frame), `gyro[3]` (float, rad/s, body frame), `quat[4]` (float, **`(w, x, y, z)` convention** — first-w). Standing pose empirical: `w ≈ 0.998`. Same publish loop as `RobotState`, ~884.5 Hz.
- **`RobotCmd` — 9 fields, vector fields sized to 31 motors.** `stamp` (uint64 ns), `mode` (31 × uint8 — control-law selector, `0` = torque-position hybrid PD-FF, all shipped controllers publish 0), `q` / `dq` / `tau` / `Kp` / `Kd` (31 × float32 each, PR-order), `motor_names` (31 × string, PR-order, echoed back), `parallel_solve_required` (31 × bool, defaults `true`, no shipped controller modifies it). Empirical publish rate ~945.6 Hz — decoupled from state's 884.5 Hz (cmd and state are independent clocks).
- **Bus role gating:** sim peers MUST construct `Robot(type, True)` and use `subscribeRobotCmdForSim` / `publishRobotStateForSim` / `publishImuDataForSim`. Cross-role subscriptions silently deliver zero. See `.claude/agent-memory/humanoid/limx-sdk-role-gating.md`.
- **Cmd ↔ state rate asymmetry:** the deploy-side controller publishes RobotCmd at ~945 Hz while the sim peer publishes RobotState at ~885 Hz. The bridge MUST treat them as independent streams — never lock-step them.

LimX's own MuJoCo bridge (`humanoid-mujoco-sim/simulator.py`, 237 LOC) is the reference implementation. It:

- Constructs `Robot(RobotType.Humanoid, True)`, calls `init(robot_ip)`.
- Sets `MROS_IP_LIST` from the first three octets of `robot_ip`.
- Subscribes `subscribeRobotCmdForSim(callback)`; the callback stores the latest cmd.
- Per MuJoCo step: applies `ctrl[i] = Kp[i]*(q_d[i]-q[i]) + Kd[i]*(dq_d[i]-dq[i]) + tau_ff[i]`, reads new `q`/`dq`/`tau` from `mj_data.sensordata`, publishes `RobotState` and `ImuData`.
- Ignores `RobotCmd.mode` entirely — actuator law is always PD-with-feedforward.
- Runs at `1 / model.opt.timestep` Hz = 1000 Hz for HU_D04_01.

The Isaac side should mimic this behavior. The only structural difference is that we have two processes instead of one.

## Workflows the design must serve

The change ships two cooperating pieces — `Oli` (the reusable Isaac component) and `OliBridge` (the MROS bus plumbing). They are designed so each can be used independently. The matrix below is the load-bearing constraint behind every decision in this doc:

| Workflow | `Oli` instance | `OliBridge` | Sidecar process | n_envs | Notes |
|---|---|---|---|---|---|
| **MAY-147** — deploy-python (damping/stand/walk) drives Isaac | one | `OliBridge.spawn_sidecar(...)` or `OliBridge.connect(...)` | 1 | 1 | The bridge use case — this is what we test against in v1 |
| **RL training** (future, IsaacLab pattern) | uses `ArticulationView` for n_envs vectorization, not single `Oli` instances | none | 0 | 1 → 4096 | Policy + physics co-located; SDK never imported. Matches `tron1-rl-isaaclab` pattern |
| **ONNX policy eval in Isaac** | one (`bridge=None`) | none | 0 | 1 | `oli.apply_cmd(q_d=..., kp=..., kd=...)` is the escape hatch |
| **Recon / nav / SLAM** apps with Oli in scene | one (`bridge=` optional) | optional — only if app scripts Oli from a Py 3.8 controller | 0 or 1 | 1 | Most apps don't need the bridge — they drive Oli kinematically or via their own policy |
| **Real-robot deploy** (LimX's stack, unchanged) | n/a | n/a | n/a | n/a | The bridge has no role on the real robot; the real robot IS the SDK peer |

**The bridge is therefore optional plumbing for the deploy-time-controllers-vs-Isaac workflow specifically.** `Oli` is the reusable surface everywhere; `OliBridge` is one possible cmd source plugged into it.

This matters because it tells us:

- `Oli` MUST work fine with `bridge=None` — no IPC, no sidecar, just USD + articulation + IMU.
- `Oli` MUST expose a `apply_cmd(...)` method (or equivalent low-level API) that takes PR-space targets and gain vectors directly, bypassing the bridge — this is what RL/ONNX/teleop call.
- The PD-with-feedforward law lives **inside `Oli.tick()`**, not inside the bridge. The bridge just delivers cmds.
- Single-Isaac-process is the only n_envs context for v1; RL vectorization is future work that uses different Isaac primitives (`ArticulationView` instead of `SingleArticulation`).
- The pathological "n parallel deploy-python rollouts" case (n sidecars on different MROS partitions) is theoretically possible but explicitly out of scope.

## Goals / Non-Goals

**Goals**

- Isaac Sim publishes `RobotState` + `ImuData` on the MROS bus at ≥850 Hz, indistinguishable from `humanoid-mujoco-sim` to any subscriber.
- Isaac Sim consumes `RobotCmd` and applies PD-with-feedforward to its articulation, mirroring LimX's own bridge behavior.
- LimX's `humanoid-rl-deploy-python` damping controller connects to Isaac at `127.0.0.1` and the loop closes without packet drops or shape errors.
- Zero-cmd → Oli holds its rest pose; small position step → Oli moves the expected joint by the expected amount.
- IPC adds < 2 ms one-way latency under normal load (target < 200 µs; design buffer ×10).
- The bridge is composed of small, self-contained modules so we can delete it cleanly if LimX ships a 3.11 wheel.
- `Oli` is importable into any Isaac host app with one line; `OliBridge.spawn_sidecar(...)` brings up the full stack with one context-manager block.
- Smoke-test demo (`load_oli.py`) is ≤ 30 lines of substantive code.

**Non-Goals**

- Solving the upstream Python ABI problem — that's an external ask to LimX, not part of this proposal.
- Reimplementing the MROS transport protocol — we use LimX's wheel as-is inside the sidecar.
- Implementing `kinematic_projection` (the parallel-ankle / waist coordinate translator). HU_D04_01's USD is the serial kinematic chain, which is what the SDK exposes anyway. Parallel-ankle physics in Isaac is a separate ticket.
- Implementing PR↔AB joint conversion. The firmware does that via the `parallel_solve_required` flag; on our serial-USD sim, PR ordering is the final actuator ordering.
- Auto-reconnect, retry logic, or any resilience beyond clean EOF handling. Failure modes are blunt and visible by design in v1.
- Vectorized n_envs (`OliVec` / `ArticulationView`) — RL training surface comes in a later change.
- Multi-bridge / n parallel sidecars — single sidecar per Isaac process is the v1 contract.
- A generic LimX bridge — scoped to HU_D04_01. Other robots may reuse the protocol but their joint counts and IMU layouts must be re-verified.
- Authoritative latency benchmarks — informal `perf_counter` instrumentation is enough for v1.

## Decisions

### D1. Two-process architecture (sidecar + driver)

The only viable shape. Sidecar runs in `limx` conda env (Py 3.8.18); driver runs in `isaac` conda env (Py 3.11.14). No subinterpreters, no FFI hacks, no protocol reimplementation.

Sidecar responsibilities:
- Construct `Robot(RobotType.Humanoid, True)` and call `robot.init(<ip>)`.
- Set `MROS_IP_LIST` env var from the first three octets of `<ip>` before constructing `Robot` (matching `humanoid-mujoco-sim/simulator.py` behavior).
- Open the IPC server endpoint; accept exactly one client connection (the `OliBridge` in the Isaac process).
- On each `subscribeRobotCmdForSim` callback, frame the cmd and write it to the client.
- On each frame received from the client, decode and call `publishRobotStateForSim` + `publishImuDataForSim`.
- Has no notion of physics or time stepping. It is a thin relay.

Isaac-side responsibilities split across two layers:

- **`Oli` class** (`humanoid/logic/simulation/isaacsim/oli.py`, Py 3.11) — owns USD loading, root pinning, articulation init, IMU sensor at `base_link`, PD-with-feedforward law, Isaac↔PR joint permutation. Provides `tick()` (one physics-tick worth of state read + cmd apply), `apply_cmd(...)` (low-level escape hatch for non-bridge workflows), and `read_state()`. Accepts an optional `bridge` argument; if `None`, `tick()` is a no-op for the cmd-receive path and `read_state` is still usable.
- **`OliBridge` class** (`bridge/__init__.py`, Py 3.11) — owns the IPC client socket and (optionally) the sidecar subprocess. Two factory constructors: `OliBridge.spawn_sidecar(ip=..., socket=...)` (context manager that starts the sidecar) and `OliBridge.connect(socket=...)` (attach to an existing one). Methods: `send_state_imu(...)`, `poll_cmd() -> Optional[RobotCmdView]`, `close()`. Plugs into `Oli(world, bridge=...)`.
- **`load_oli.py`** — smoke-test demo (~20–30 lines) that composes `Oli` + `OliBridge.spawn_sidecar` and runs the render+physics loop. NOT load-bearing; replaceable by any host app.

### D2. IPC channel: AF_UNIX SEQPACKET socket

Default choice. Single Unix-domain socket at `/tmp/limx-isaac-bridge.sock` (override via env var).

| Criterion | UDS SEQPACKET | UDS STREAM | POSIX shared memory | ZeroMQ PUB/SUB |
|---|---|---|---|---|
| Latency (loopback, 1 KB) | ~50 µs | ~50 µs | ~5 µs | ~150 µs |
| Cross-Python-version safety | ✓ stdlib | ✓ stdlib | ✓ stdlib | needs `pyzmq` both envs |
| Preserves message boundaries | ✓ native | requires framing | requires seq counter | ✓ |
| Debuggability | `nc -U` works | `nc -U` works | requires custom dumper | `zmq_proxy` |
| Code complexity | low | low | medium | low-medium |
| Cross-machine extension | no | no | no | ✓ |

SEQPACKET wins on simplicity-per-correctness for v1. Standard library, message boundaries preserved (no manual framing), trivial to inspect. If jitter at 1 kHz proves a problem we will swap to POSIX shm (`mmap` + atomic seq counter — both stdlib in both Python versions). Protocol module is structured so the transport is replaceable.

Cross-machine support is not needed — sidecar and Isaac always live on the same workstation.

Linux supports `SOCK_SEQPACKET` for AF_UNIX since kernel 2.6.4 (2004). Fedora 42 trivially supports it. macOS does not — flagged as a non-portability but out of scope (we don't develop on macOS).

### D3. Three message types

| Direction | Type | Wire payload (after header) | Sent when |
|---|---|---|---|
| Driver → Sidecar | `HELLO` | DOF count (uint32) + 31 × 32-byte NUL-padded joint name strings (Isaac DOF order) | Once at startup |
| Driver → Sidecar | `STATE_IMU` | seq + stamp + 31×{q,dq,tau} (PR-order, float32) + IMU(acc[3], gyro[3], quat[4] — struct order, `(w,x,y,z)`) | Every physics tick |
| Sidecar → Driver | `CMD` | seq + stamp + 31×{mode (u8), q, dq, tau, Kp, Kd (f32 each), parallel_solve_required (u8)} (PR-order) | On each `subscribeRobotCmdForSim` callback |

Common 8-byte header: `type` (uint16) + `payload_len` (uint16) + `seq` (uint32). On SEQPACKET, payload_len is redundant with the message boundary but is kept for forward compatibility with a future STREAM transport.

**Intentional deviation from canonical `RobotCmd`:** the canonical struct (`oli-corpus://limxsdk#datatypes.h`) has a 9th field `motor_names` (31 × string) that controllers echo back. Our IPC drops it — the driver already has the permutation map from `HELLO`, and `motor_names` is invariant per session. Sidecar receives `motor_names` from `subscribeRobotCmdForSim` and verifies set-equality once against the handshake list (cheap sanity check), then strips it from the IPC frame.

No `BYE` / `HEARTBEAT` / `ACK` messages in v1. UDS EOF detection handles both directions of clean shutdown. Future iteration if needed.

### D4. Fixed-size, little-endian struct payloads in PR-space order

All payload joint arrays are exactly 31 entries (Oli's PR joint count). Joint ordering on the wire is the SDK's PR ordering as observed empirically and documented in `humanoid/docs/vendor/humanoid-rl-deploy-python.md` § 11 — the 31-row table from index 0 (`left_hip_pitch_joint`) to index 30 (`right_wrist_roll_joint`). **Note**: indices 15 and 16 are `head_yaw_joint` and `head_pitch_joint` respectively (yaw before pitch — the on-robot `head_config.yaml` and the live wire probe agree; the `sdk_joint_order` MCP tool currently reports them swapped due to a stale `walk_param.yaml` extraction. Trust the probe).

Sizes (verified by `struct.calcsize` in `bridge/protocol.py`, 2026-06-22):

| Message | Total wire size | Calc |
|---|---|---|
| `HELLO` | 8 + 996 = **1004 B** | header + payload (4 dof_count + 31×32 names = 996) |
| `CMD` | 8 + 690 = **698 B** | header + payload (8 stamp + 31×(1+4+4+4+4+4+1) = 690) |
| `STATE_IMU` | 8 + 420 = **428 B** | header + payload (8 stamp + 31×3×4 q/dq/tau + 12+12+16 acc/gyro/quat = 420) |

`struct.pack`/`unpack` byte layouts are identical between CPython 3.8 and 3.11 — verified by `_research/test_protocol_cross_version.py` (Phase 3): canned `HELLO`/`CMD`/`STATE_IMU` packed in both envs produce identical sha256 digests.

Single source of truth: `bridge/protocol.py`. Both processes import it; in v1 we keep two copies in sync via a `make sync-protocol` recipe that diffs the file from the Isaac env's copy. (Both envs share the same filesystem path, so simple symlinks via a checked-in `bridge/protocol.py` work — no copies needed.)

### D5. Host app drives the cadence via `oli.tick()`; cmd is latched

The host application owns the physics + render loop and calls `oli.tick()` once per physics step. `Oli` is a **pull-model** component — it does not run its own thread, does not own the world, does not call `world.step()` itself. This keeps everything single-threaded and deterministic; the host always knows when `tick()` ran relative to physics.

Target cadence is **1 kHz** (`world = World(physics_dt=1/1000)`); realistic empirical rate is **~880 Hz** — the LimX MuJoCo reference falls from nominal 1000 Hz to ~884.5 Hz on desktop due to viewer + Python overhead (`humanoid-rl-deploy-python.md` § 11). Isaac is likely to land in the same regime; design budgets assume ≥850 Hz sustained.

Inside `Oli.tick()` (with a bridge attached):

1. Read articulation `q`, `dq`, `tau` from `SingleArticulation.get_joint_positions/velocities/get_measured_joint_efforts`.
2. Read IMU from the dedicated `IMUSensor` prim (D8).
3. Permute Isaac DOF order → PR order using the cached `isaac_to_pr` index.
4. Call `bridge.send_state_imu(...)` (which packs + non-blocking `sendmsg`).
5. Call `bridge.poll_cmd()` — non-blocking; drain all pending CMD frames, keep only the latest; update the cached cmd.
6. **Realize the cached PD cmd via the PhysX implicit drive** (see "PD realization" below).

The host then calls `world.step(render=...)` separately. `Oli.tick()` does NOT step or render.

With `bridge=None`, steps 1–5 are skipped; `tick()` just re-applies the cached cmd (step 6) against the latest articulation state. Apps that drive Oli directly (RL, ONNX, teleop) call `oli.apply_cmd(q_d, dq_d, tau_ff, kp, kd)` to update the cache.

#### PD realization: PhysX implicit drive, NOT explicit `set_joint_efforts` (verified 2026-06-22)

The PD-with-feedforward law `τ = Kp(q_d − q) + Kd(dq_d − dq) + τ_ff` is realized by pushing `Kp`/`Kd` into the joint **drive gains** (`articulation_view.set_gains`) and commanding **position + velocity targets** (`set_joint_position_targets` / `set_joint_velocity_targets`), letting PhysX integrate the PD term implicitly. The feedforward `τ_ff` is added separately via `set_joint_efforts` (additive on top of the drive).

**Why not compute `τ` ourselves and call `set_joint_efforts`?** That applies the cmd's `Kd` as an *explicit, one-step-lagged* external force. The deploy controllers' gains (`Kd≈17`) are tuned for implicit integration (real motor firmware + MuJoCo's implicit damping). Applied explicitly in Isaac, the discrete stability bound `Kd·dt/I ≲ 2` is violated for the light links (knee `izz≈0.003` → `17·0.001/0.003 ≈ 5.6`), producing a sustained velocity limit-cycle (~±1.8 rad/s ringing — measured). The implicit drive integrates the *identical* formula stably, exactly as the real motor controller and MuJoCo do, with **no gain retuning** (the gains come from LimX over the wire — we can't retune them). This is also how IsaacLab itself implements actuators.

Empirical confirmation (`_research/smoke_oli_nobridge.py`, `bridge=None`): explicit effort rang at ±1.8 rad/s indefinitely; implicit drive decayed peak velocity 12.5 → 0.54 rad/s (23×) and held the spawn pose to 0.087 rad. Tick latency p50=115µs, p99=204µs.

**Gains are re-written only on change.** `set_gains` is called only when the incoming cmd's `Kp`/`Kd` differ from the last-written values (the walk/stand controllers send constant gains, so this is rare). Position/velocity targets + `τ_ff` are written every tick.

**Steady-state droop is expected.** With finite `Kp` and no gravity feedforward, a held joint settles at `q ≈ q_d − τ_gravity/Kp`. The shipped controllers compensate via `τ_ff` and tuned per-joint gains; faithful holding/tracking against gravity is MAY-148's concern, not the bridge's.

Latching policy: most recent cmd wins — no queue, no interpolation. This matches `humanoid-mujoco-sim/simulator.py` exactly. Empirically, cmd arrives at ~945 Hz while state publishes at ~885 Hz — the two clocks are independent on the MROS bus and `Oli` must NOT lock them together. Cold start: all-zero cmd; Oli sags to its rest pose with the soft default drive gains baked into the USD.

Render decimation is the **host's** responsibility, not `Oli`'s — the smoke demo defaults to `render=(tick % 20 == 0)` matching LimX. Apps with heavier scenes can choose differently.

### D6. Handshake & lifecycle

```
sidecar startup:
  read CLI: ip, socket_path
  set MROS_IP_LIST = ip.rsplit('.', 1)[0] + '.0/24'   (or similar — match humanoid-mujoco-sim)
  Robot = limxsdk.robot.Robot(RobotType.Humanoid, True)
  Robot.init(ip)
  Robot.subscribeRobotCmdForSim(on_cmd)
  socket = socket.socket(AF_UNIX, SOCK_SEQPACKET)
  socket.bind(socket_path)
  socket.listen(1)
  client, _ = socket.accept()         # blocks
  hello = recv(client)                # blocks until HELLO arrives
  build pr_to_isaac, isaac_to_pr permutation indices from hello.dof_names
  ack the hello (or close on dimension mismatch)
  enter main relay loop

driver startup (inside load_oli.py, after world.reset()):
  client = socket.socket(AF_UNIX, SOCK_SEQPACKET)
  client.connect(socket_path)
  send HELLO with oli.dof_names (Isaac order, 31 entries)
  recv ack (or exit on close)
  client.setblocking(False)
  enter render+physics loop
```

Failure semantics:

- Sidecar fails to `Robot.init`: log + exit non-zero; launcher catches and tears down driver.
- Sidecar socket bind fails (path in use): unlink old socket + retry once; on second failure exit.
- Driver fails to connect within 10 s: exit; launcher tears down sidecar.
- Either side observes EOF on the socket: log + exit cleanly.
- DOF count mismatch in HELLO: sidecar logs the mismatch with both sides' lists and closes the connection.
- Driver crashes: sidecar's recv returns EOF; sidecar calls `Robot.deinit` (if available, else just exits) and unlinks the socket.

No reconnection logic, no signal forwarding beyond the launcher's SIGINT → SIGTERM cascade.

### D7. Joint order: PR-canonical wire, permutation lives in driver

The MROS bus and the SDK speak PR space — 31 joints ordered as captured empirically in `humanoid/docs/vendor/humanoid-rl-deploy-python.md` § 11 (idx 0 = `left_hip_pitch_joint` through idx 30 = `right_wrist_roll_joint`). Every byte that goes across the IPC is in this PR order. The driver — and only the driver — owns the permutation between PR and Isaac's DOF order.

At handshake the driver sends Isaac's DOF name list (31 strings, Isaac order). The sidecar's perspective of joint order is fixed at compile time (PR), so it just acks the hello after a quick sanity check (count + name set equality). Building the permutation table is a driver-side concern.

**Known corpus-tool disagreement at the head joints**: indices 15 and 16 are `head_yaw_joint` and `head_pitch_joint` respectively per the wire probe and the on-robot `oli-corpus://oli-main-2.2.12#install/etc/upper_body/head/D04_01/head_config.yaml`. The `sdk_joint_order` MCP tool currently reports them swapped because its extraction reads a stale `walk_param.yaml` annotation. **Trust the probe.** The driver's permutation table must be built against § 11's 31-row table, not against `sdk_joint_order`. Filed as a corpus bug (see Open Questions).

Why this design: the wire stays canonical against the documented MROS protocol, so future MROS sniffers / packet captures can be diffed against `humanoid-mujoco-sim` outputs byte-for-byte. The driver — which is the only side that has to consult Isaac's DOF ordering — does the conversion locally and never exposes Isaac-order packets to the bus.

This pattern is also useful if Isaac's USD re-imports change the DOF ordering down the line — only the driver's permutation table changes; no protocol or sidecar change.

### D8. IMU source in Isaac

The HU_D04_01 URDF has **no dedicated IMU link** — there is no `imu_link`, no `chest_link`, nothing tagged as an IMU mount (verified by walking the link list via `links("HU_D04_01")`). On the real robot the IMU is the `hi13_imu_driver` (`oli-corpus://oli-main-2.2.12#install/share/hi13_imu_driver/package.xml`) and is physically mounted somewhere in the torso, but the URDF doesn't model the mount as a separate link.

The SDK guide § 4.1 (`oli-corpus://sdk-guide#4.1`) is explicit: **`base_link` is the canonical robot base coordinate frame** for HU-series robots. The MuJoCo bridge publishes IMU data referenced to that base frame. We mirror this in Isaac: attach `omni.isaac.sensor.IMUSensor` at `base_link` (the pelvis link, parent of the hip joints), reading at the physics dt.

If the on-robot `imu_offset_pitch` / `imu_offset_roll` values are nonzero (default config in `oli-corpus://oli-main-2.2.12#install/etc/mission_engine/imuoffset.yaml` is `0.0` / `0.0`), apply them to the sensor's local orientation. For v1 we hard-code zero offsets and document the lookup path for later tuning.

**Quaternion convention.** Both LimX (`oli-corpus://limxsdk#datatypes.h` — `float quat[4] // (w, x, y, z)`) and Isaac native use **`(w, x, y, z)` first-w ordering**. No reordering needed in the packer. Empirically confirmed in § 11: standing pose has `w ≈ 0.998`. (An earlier version of this design claimed LimX used `(x, y, z, w)` — that was wrong, corrected 2026-06-22.)

**ImuData struct field order on the wire**: `stamp` (uint64), `acc[3]` (float32), `gyro[3]` (float32), `quat[4]` (float32). The packer MUST follow this order, not (quat, gyro, accel) — this is the order the canonical struct lays out in memory and what the sidecar receives from `subscribeImuData`.

IMU sample rate: matches physics dt (target 1 kHz, expected ~880 Hz desktop). No oversampling, no smoothing — Isaac's sensor is already filtered.

### D9. `kinematic_projection` is out of scope

LimX's MuJoCo bridge spawns the `kinematic_projection` ELF binary as a subprocess to translate between the serial-equivalent URDF kinematics that the SDK exposes and the parallel-Achilles MJCF that MuJoCo simulates. The on-robot equivalent lives **inside the low-level motion control system** rather than as a separate ELF, per `oli-corpus://sdk-guide#5.1.6?part=1` — that's why the same `publishRobotCmd` packet works for both sim and real robots. The HU_D04_01 USD shipped with the asset bundle mirrors the URDF (serial), so Isaac is already in the same space the SDK speaks — no projection needed.

The parallel ankle linkage configs that the MuJoCo path consumes are at `oli-corpus://oli-main-2.2.12#install/etc/kinematic_projection/HU_D04_01/twisted_left_ankle_model.yaml` (and the right-side equivalent). They are NOT needed for Isaac in this change.

Consequence: Isaac with this bridge will simulate **serial ankle kinematics**, not parallel. For early-stage control tests (damping, stand) this is correct and matches what the policy expects on the wire. For high-fidelity walking and contact-rich behaviors that depend on the parallel linkage's effective mechanical advantage, a separate ticket will introduce a parallel-USD variant + Isaac-side projection. Documented as a known limitation in `bridge/README.md`.

### D10. `parallel_solve_required` is out of scope

The per-joint `parallel_solve_required: vector<bool>` field in `RobotCmd` (default `true` per the struct constructor in `oli-corpus://limxsdk#datatypes.h`) tells the firmware to convert PR-space commands to AB-actuator targets before applying. Empirically confirmed in § 11: no shipped controller (`damping`, `stand`, `walk`, `mimic`) modifies the default. The driver MUST therefore handle PR-space commands directly. In our serial-USD Isaac sim, PR space *is* the actuator space, so the conversion is the identity — we accept the flag in the cmd, assert it is `true` for all 31 joints (warn-and-continue if not), and otherwise ignore it. Documented in `bridge/README.md`.

### D11. Conda env strategy

| Process | Conda env | Python | Key deps |
|---|---|---|---|
| Sidecar (`bridge/sidecar.py`) | `limx` | 3.8.18 | `limxsdk-4.0.1` (already installed via wheel from `humanoid-mujoco-sim`) |
| Host app (e.g. `load_oli.py`, future RL/recon/nav apps) | `isaac` | 3.11.14 | `isaacsim==5.0.0`, `pxr` |

`OliBridge.spawn_sidecar(...)` resolves the sidecar interpreter path via env var with a sane default:
- `LIMX_BRIDGE_SIDECAR_PY` (default `/home/may33/miniconda3/envs/limx/bin/python`)

Host apps run in whatever Isaac-capable env they choose; `OliBridge` itself is stdlib-only.

### D12. File layout

```
humanoid/logic/simulation/isaacsim/
├── oli.py                  # Oli class — the reusable component (D14)
├── load_oli.py             # smoke demo composing Oli + OliBridge (~20–30 LOC)
├── bridge/
│   ├── __init__.py         # OliBridge class with two constructors (D15)
│   ├── protocol.py         # struct formats + framing helpers (D3, D4)
│   ├── sidecar.py          # Py 3.8 sidecar (D1)
│   └── README.md           # how to run, workflows mapping, known limitations
└── README.md               # overview, links to bridge/README.md
```

No `bridge/launcher.py` as a separate module — its role is absorbed into `OliBridge.spawn_sidecar(...)` (the context manager that owns the sidecar subprocess). The smoke demo's `__main__` block is the closest thing to a "launcher script."

Apps that want the bridge import it explicitly:
```python
from humanoid.logic.simulation.isaacsim import Oli
from humanoid.logic.simulation.isaacsim.bridge import OliBridge
```

Apps that don't (RL, ONNX, kinematic-only) import only `Oli` — no `bridge` symbols leak into their dependency graph.

### D13. Versioning the IPC protocol

The 8-byte header reserves bits in the `type` field as a version tag: `type = (version << 14) | type_code`. Version is `0` in v1. Any future incompatible change bumps it, and the sidecar must close the connection on version mismatch with a clear error.

This is cheap insurance and costs no bytes; we will not regret adding it.

### D14. `Oli` class API

`Oli` is the reusable Isaac component. Any host app (smoke demo, RL trainer, recon, nav, SLAM) imports it the same way. Sketch:

```python
from humanoid.logic.simulation.isaacsim import Oli
from humanoid.logic.simulation.isaacsim.bridge import OliBridge

class Oli:
    def __init__(
        self,
        world: World,
        *,
        prim_path: str = "/World/Oli",
        spawn_pose: tuple[float, float, float] = (0.0, 0.0, 1.05),
        pin_root: bool = True,
        variant: Literal["bare", "gripper", "hand"] = "bare",
        bridge: OliBridge | None = None,
    ) -> None: ...

    @property
    def dof_names(self) -> list[str]: ...
    @property
    def num_dof(self) -> int: ...

    def tick(self) -> None:
        """One physics-tick worth of work. If bridge is attached:
        read state+IMU, send STATE_IMU, drain CMD, apply PD law.
        If bridge is None: no-op except for whatever the cached cmd
        (set via apply_cmd) requires."""

    def apply_cmd(
        self,
        q_d: np.ndarray,
        dq_d: np.ndarray | None = None,
        tau_ff: np.ndarray | None = None,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        """Low-level escape hatch — set the cached cmd directly.
        All arrays in PR order, length 31. Missing kwargs are kept
        at their previous value (or zero on first call)."""

    def read_state(self) -> dict:
        """Return {stamp, q, dq, tau, motor_names} in PR order — same
        shape as RobotState. Useful for apps that need state without
        engaging the bridge."""

    def read_imu(self) -> dict:
        """Return {stamp, acc, gyro, quat_wxyz}. Same shape as ImuData."""
```

Key contracts:

- **No global state.** Multiple `Oli` instances in one world (different `prim_path`) work if the host wants them, but the v1 IMU prim is namespaced per instance and the bridge is single-tenant — two `Oli(bridge=...)` instances in one process would race on the same UDS socket. Document; don't enforce.
- **Pull-model only.** `Oli` does not own threads or callbacks. The host calls `tick()`.
- **PR-space is the canonical interface.** All array I/O on `Oli` (apply_cmd inputs, read_state outputs) is in PR order. Internal Isaac DOF permutation is hidden.
- **No physics or render side effects.** `Oli.tick()` calls `articulation.set_joint_efforts(...)` but never `world.step()` or `SIM_APP.update()` — those are the host's concern.

### D15. `OliBridge` class API

`OliBridge` owns the IPC client socket and (when spawned) the sidecar subprocess. Two named constructors signal intent:

```python
from humanoid.logic.simulation.isaacsim.bridge import OliBridge

# Use case A: one-line bring-up, sidecar lifetime tied to context manager
with OliBridge.spawn_sidecar(ip="127.0.0.1") as bridge:
    oli = Oli(world, bridge=bridge)
    ...

# Use case B: attach to a sidecar started externally (e.g., systemd, tmux, dev shell)
bridge = OliBridge.connect(socket="/tmp/limx-isaac-bridge.sock")
oli = Oli(world, bridge=bridge)
...
bridge.close()
```

Sketch:

```python
class OliBridge:
    @classmethod
    def spawn_sidecar(
        cls,
        ip: str = "127.0.0.1",
        socket: str = "/tmp/limx-isaac-bridge.sock",
        sidecar_py: str | None = None,  # env var fallback
    ) -> "OliBridge": ...

    @classmethod
    def connect(
        cls,
        socket: str = "/tmp/limx-isaac-bridge.sock",
        timeout: float = 10.0,
    ) -> "OliBridge": ...

    # Context manager — only for spawn_sidecar; .connect() requires explicit close
    def __enter__(self) -> "OliBridge": ...
    def __exit__(self, *exc) -> None: ...

    # Used by Oli.tick() — not host-facing in normal flow
    def handshake(self, dof_names: list[str]) -> None: ...
    def send_state_imu(self, seq: int, stamp_ns: int, q, dq, tau, acc, gyro, quat) -> None: ...
    def poll_cmd(self) -> RobotCmdView | None: ...  # most recent only, None if empty

    def close(self) -> None: ...  # idempotent
```

Key contracts:

- **`spawn_sidecar` owns the subprocess; `connect` does not.** `__exit__` from a `spawn_sidecar`-created bridge SIGINTs the sidecar; `.close()` on a `connect`-created bridge only closes the socket.
- **Handshake is exchanged inside `Oli.__init__` when a bridge is passed.** The host app never calls `bridge.handshake(...)` directly — but it can, and `Oli` checks idempotently.
- **`poll_cmd()` drains the socket on every call.** Returns only the most recent message; intermediate frames are dropped per D5.
- **Sidecar stdout/stderr** is multiplexed into the host process's stdout/stderr with `[sidecar] ` prefix when spawned via `spawn_sidecar`.

## Open Questions

### OQ1. Sidecar location: `humanoid/logic/...` vs `humanoid/vendor/...`?

**Lean**: `humanoid/logic/simulation/isaacsim/bridge/sidecar.py`. It is our code calling the vendor wheel; it is not the wheel itself. Living next to the driver also keeps `protocol.py` as a single file imported by both. Defer to first-pass file creation; revisit if it stops feeling right.

### OQ2. IMU prim location in `HU_D04_01.usd` — *resolved 2026-06-22*

The URDF has no `imu_link` / `chest_link` / IMU-tagged prim (verified via `links("HU_D04_01")` against the corpus). The `_sensor.usd` USD layer may still ship a sensor prim, but no link-side anchor exists for it. **Resolution**: per D8, attach `omni.isaac.sensor.IMUSensor` at `base_link` — the canonical robot base frame per `oli-corpus://sdk-guide#4.1`. If the `_sensor.usd` layer turns out to already define an IMU prim at boot, prefer that prim and skip the attach step; otherwise create one. Task 2.5 captures the audit.

### OQ3. PR joint count vs Isaac DOF count

PR space is documented as 31 joints (MAY-145 § 11). Isaac's DOF count from the existing `load_oli.py` smoke run needs verification — `[load_oli] Total DOFs: …` line in the stdout will tell us. The two **must** be 31. If Isaac reports a different count, the USD has extra mimic / fixed / mimic-driven joints that need filtering in the permutation table.

### OQ4. Render rate

LimX's MuJoCo bridge syncs the viewer every 20 steps (50 Hz at 1 kHz physics). Isaac's render is more expensive than MuJoCo's passive viewer; we may need to drop to every 33 or 50 steps (30 / 20 Hz). **Action**: profile in tasks; default to every 20 and adjust.

### OQ5. Single sidecar or per-deploy-controller sidecar?

`humanoid-rl-deploy-python` uses an ability subprocess pattern — the joystick dispatcher spawns a controller process per mode (damping, stand, walk, etc.). All of them connect to the same bus IP, so a single sidecar serves all of them (the bus itself does the fan-out). **Lean**: single sidecar. No action needed; documenting for clarity.

### OQ6. Launcher: subprocess vs systemd vs tmux

`subprocess.Popen` is the v1 default — pure Python, no external deps, SIGINT works. tmux gives nice manual debugging (separate panes per process) but adds a runtime dep. **Lean**: subprocess with stdout/stderr prefixing in v1; document tmux as an optional dev convenience in `bridge/README.md`.

### OQ7. Background-research outcome — *resolved 2026-06-22*

Two background subagents surveyed LimX's GitHub footprint. Findings (full reports at `.claude/agent-memory/humanoid/research_limx_isaac_integration.md` and `research_ros2_bridger_alternative.md`):

1. **No Py 3.10/3.11 wheel exists anywhere** — PyPI, all 25 LimX repos, no CI evidence. Binding generator is raw CPython C API (not pybind11), so a community rebuild is non-trivial.
2. **`tron1-rl-isaaclab` does NOT import `limxsdk`** — it's offline training that exports ONNX; deployment is a separate Py 3.8 process. LimX has no public precedent for live Isaac ↔ SDK loops.
3. **`ros2-bridger` is NOT a viable alternative** — it has no factories for the aggregate `RobotState*`/`RobotCmd*` topics our sim peer needs (only per-joint `JointState`/`JointCmd`), no role gating, and the Py-3.11 conflict re-emerges if used with Isaac. Even LimX's own `humanoid-rl-deploy-ros2` links `limxsdk` directly instead of using the bridge.

**Conclusion**: the custom Py 3.8 sidecar is the only viable architecture. Proposal continues unchanged. External ask to LimX for proper 3.10/3.11 wheels remains in flight as a separate track (the bridge stays deletable per the Reversibility section).

### OQ8. Corpus `sdk_joint_order` tool reports stale head-joint order

The MCP tool `sdk_joint_order("HU_D04_01")` returns `head_pitch_joint, head_yaw_joint` at indices 15/16. The on-robot config (`oli-corpus://oli-main-2.2.12#install/etc/upper_body/head/D04_01/head_config.yaml`) and the MAY-145 wire probe both report `head_yaw_joint, head_pitch_joint`. The tool's extractor reads `walk_param.yaml` annotations which appear to be stale relative to the runtime config. **Action**: file a corpus bug as a follow-up ticket; in the meantime, the design and the driver's permutation table take their truth from § 11.

## Risks / Trade-offs

- **[Risk] IPC jitter at 1 kHz UDS exceeds budget.** Cmd → actuator latency might gain a tail at high system load. **Mitigation:** instrument both ends with `perf_counter` from day one; histogram the round-trip in a smoke test. If p99 exceeds 1 ms, swap to POSIX shm — the protocol module is designed to be transport-swappable.
- **[Risk] Sidecar's MROS env vars don't match LimX's expectations.** `MROS_IP_LIST` format is undocumented; we copy LimX's reference behavior verbatim. **Mitigation:** test against deploy-python on first integration; if traffic doesn't flow, sniff their MuJoCo sim's env via `cat /proc/<pid>/environ` and copy exactly.
- **[Risk] Joint name permutation wrong on first attempt and Oli moves the wrong joints under cmd.** **Mitigation:** smoke-test by hand-injecting a single-joint position step from a CLI stub before connecting deploy-python. If the wrong joint moves, the permutation is wrong and easy to fix.
- **[Risk] IMU values from Isaac don't match what the standing controller's ONNX expects.** The controller was trained on MuJoCo's IMU output. Isaac's sensor may use different units (rad/s² vs g, m/s² vs g) or different gravity-frame conventions. **Mitigation:** capture a static-Oli IMU sample from MuJoCo via the probe; compare against Isaac's sample with identical Oli pose; document deltas.
- **[Risk] Background research returns "LimX has a 3.11 wheel" and this whole change is wasted.** **Mitigation:** structure for deletion (D-decisions); ship the parallel external ask today regardless. Wasted-effort risk is small (bridge is ≈ 400–600 LOC).
- **[Risk] `kinematic_projection` matters for stand controller more than expected.** The HU_D04_01 ankle linkage might be load-bearing for stand stability. **Mitigation:** measure first — run the stand controller and observe. If it falls over, document as expected and gate further controllers behind the parallel-USD ticket.
- **[Risk] `parallel_solve_required` ignored is wrong for some controller.** A future LimX policy might rely on it being honored. **Mitigation:** log and count cmds with the flag set; flag in stand-controller smoke tests as a follow-up.
- **[Risk] Conda env paths drift / users have different miniconda installs.** Hard-coded `/home/may33/miniconda3/envs/limx/bin/python` will break for any other developer. **Mitigation:** env vars (`LIMX_BRIDGE_SIDECAR_PY`, `LIMX_BRIDGE_DRIVER_PY`) override the constants; document in `bridge/README.md`.

## Migration Plan

This is a green-field capability — there is no prior bridge to migrate from. The build order follows the dependency graph (`protocol → sidecar → Oli → OliBridge → load_oli.py demo`):

1. Scaffold `humanoid/logic/simulation/isaacsim/bridge/` with empty modules + `oli.py` at the parent level.
2. Implement `bridge/protocol.py` first (no other deps). Write a cross-version unit test that imports it from both `limx` and `isaac` envs and confirms a canned 100-byte pack produces byte-identical output.
3. Implement `bridge/sidecar.py` with a stub that prints each incoming cmd to stdout. Run in `limx` env; verify against a real `humanoid-rl-deploy-python` damping controller.
4. Implement `Oli` class in `oli.py` — USD load, root pin, articulation init, IMU sensor at `base_link`, permutation table, PD law, `tick()`, `apply_cmd()`, `read_state()` / `read_imu()`. Test it with `bridge=None` against a hand-injected `apply_cmd(...)` smoke.
5. Implement `OliBridge` class in `bridge/__init__.py`: both `spawn_sidecar` and `connect` constructors, context-manager protocol, `send_state_imu` / `poll_cmd` / `handshake`.
6. Rewrite `load_oli.py` as the smoke demo — `with OliBridge.spawn_sidecar(ip=...) as bridge: Oli(world, bridge=bridge); loop: oli.tick(); world.step()`. Target ≤30 LOC of substantive code.
7. End-to-end test: `load_oli.py` → deploy-python damping → Oli holds pose.
8. Stand controller smoke: `load_oli.py` → deploy-python stand → Oli stands or fails-informatively.
9. Document in `bridge/README.md` (workflows mapping, API reference, known limitations) and add cross-references in the two existing vendor docs.

No rollback plan needed beyond `git revert` — the change is local code with no shared resources.

## Reversibility for the LimX-ships-3.11-wheel scenario

If LimX delivers a Python 3.11 wheel for `limxsdk` during the lifetime of this change:

1. Install the new wheel in the `isaac` env directly.
2. **`Oli` stays unchanged.** Its public API (`tick`, `apply_cmd`, `read_state`, `read_imu`) is bridge-agnostic.
3. Add a new `OliSdkDirect` class to `bridge/__init__.py` (or replace `OliBridge`) that wraps `Robot(RobotType.Humanoid, True)` directly in-process — same `send_state_imu` / `poll_cmd` interface, no IPC, no subprocess.
4. Delete `bridge/sidecar.py` and `bridge/protocol.py` (or keep around for cross-machine future use).
5. The OpenSpec capability can be archived as superseded or amended.

Estimated migration effort: half a working day — only `OliBridge` (~150 LOC) needs replacement; everything else (PD law, permutation, IMU sensor, host apps) is untouched.
