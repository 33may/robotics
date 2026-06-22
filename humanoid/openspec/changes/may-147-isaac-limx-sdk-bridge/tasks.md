## 1. OpenSpec scaffolding

- [x] 1.1 `openspec new change may-147-isaac-limx-sdk-bridge`
- [x] 1.2 Author `proposal.md`
- [x] 1.3 Author `design.md`
- [x] 1.4 Author `tasks.md` (this file)
- [x] 1.5 Author `specs/isaac-limx-sdk-bridge/spec.md`
- [x] 1.6 `openspec validate may-147-isaac-limx-sdk-bridge` — reported valid 2026-06-22

## 2. Environment + asset prechecks (D11, OQ2, OQ3) — *done 2026-06-22*

- [x] 2.1 `limx` env confirmed: Py 3.8.18, `limxsdk` importable, `Robot`/`RobotType.Humanoid`/`datatypes` all resolve
- [x] 2.2 `isaac` env confirmed: Py 3.11.14, `isaacsim` 5.0.0 importable
- [x] 2.3 Isaac DOF order captured via `_research/audit_isaac_oli.py` → `_research/isaac_dof_dump.txt` (headless Isaac boot, 31 DOFs)
- [x] 2.4 Compared to PR §11: count = 31 ✓, set equality ✓, permutation table computed + verified in `_research/joint_name_audit.md`. Head joint order = yaw-then-pitch (matches PR canonical; the corpus `sdk_joint_order` MCP tool is the only outlier — confirms OQ8)
- [x] 2.5 IMU prim audit via the same audit script → `_research/imu_prim_audit.{txt,md}`. **No IMU prim ships in HU_D04_01.usd.** `Oli.__init__` must attach `omni.isaac.sensor.IMUSensor` at `base_link` at runtime (D8 confirmed; OQ2 fully resolved)
- [x] 2.6 `MROS_IP_LIST` form captured from `humanoid-mujoco-sim/simulator.py:226`: literal `f"{a}.{b}.{c}.x"` (last octet replaced with `x`). For `127.0.0.1` → `MROS_IP_LIST=127.0.0.x`

## 3. Protocol layer (Spec R-protocol) — *done 2026-06-22*

- [x] 3.1 Created `humanoid/logic/simulation/isaacsim/bridge/__init__.py`
- [x] 3.2 Implemented `bridge/protocol.py`: `MsgType` enum (`HELLO=0`, `CMD=1`, `STATE_IMU=2`), `PROTOCOL_VERSION=0`, struct formats per design D3+D4
- [x] 3.3 Implemented `pack_hello(seq, dof_names) -> bytes` and `unpack_hello(buf) -> (seq, dof_count, dof_names)`
- [x] 3.4 Implemented `pack_cmd(seq, stamp_ns, mode, q, dq, tau, kp, kd, parallel_solve_required) -> bytes` and `unpack_cmd(buf)`
- [x] 3.5 Implemented `pack_state_imu(seq, stamp_ns, q, dq, tau, acc, gyro, quat_wxyz) -> bytes` and `unpack_state_imu(buf)` — note IMU field order is `acc, gyro, quat` (struct order), quaternion is `(w, x, y, z)` per D8
- [x] 3.6 Cross-version pack identity test (`_research/test_protocol_cross_version.py`): canned HELLO/CMD/STATE_IMU produce identical sha256 in Py 3.8.18 and Py 3.11.14
- [x] 3.7 Round-trip test: HELLO/CMD/STATE_IMU pack→unpack lossless (uint64/uint32 bit-exact; floats within 1e-6); version mismatch + unknown type_code both correctly raise `ProtocolError`
- [x] 3.8 Module docstring documents header layout + payload sizes (HELLO=1004 B, CMD=698 B, STATE_IMU=428 B — corrected from earlier design hand-math)

## 4. Sidecar — Py 3.8 process (Spec R-sidecar, D1, D6) — *done 2026-06-22*

- [x] 4.1 Implemented `bridge/sidecar.py` with argparse: `--ip` (default `127.0.0.1`), `--socket` (default `/tmp/limx-isaac-bridge.sock`), `--debug`
- [x] 4.2 Set `MROS_IP_LIST` from `--ip` matching `humanoid-mujoco-sim/simulator.py:226` (`f"{a}.{b}.{c}.x"`)
- [x] 4.3 Construct `Robot(RobotType.Humanoid, True)`; call `robot.init(ip)` — verified returns `True` for `127.0.0.1`
- [x] 4.4 Open AF_UNIX SOCK_SEQPACKET socket; bind to `--socket`; listen(1); accept one client
- [x] 4.5 Receive `HELLO`; verify `dof_count == 31` and joint-name set-equality against PR canonical; close + exit non-zero on mismatch
- [x] 4.6 Ack `HELLO` with header-only frame (`pack_header(HELLO, payload_len=0, seq=0)`)
- [x] 4.7 Register `subscribeRobotCmdForSim(callback)`; callback runs in SDK thread; packs cmd + non-blocking `send` under a lock; drop+count on `BlockingIOError`
- [x] 4.8 Main loop: `selectors`-based read; unpack `STATE_IMU`; call `publishImuDataForSim(imu)` then `publishRobotStateForSim(state)` (matches MuJoCo reference publish order)
- [x] 4.9 EOF on socket → log + clean exit (unlink socket file in `finally`)
- [x] 4.10 SIGTERM/SIGINT handlers flip a running flag; main loop exits at next select wakeup
- [x] 4.11 Smoke verified via `_research/fake_driver_smoke.py`: sidecar accepts connection, HELLO ok, decodes 50 STATE_IMU frames (`state_pub=50`), exits cleanly on driver EOF, unlinks socket file. Spec R1/R2/R4/R10/R11 scenarios green. *Deferred to Phase 7 integration*: live deploy-python damping verification.

## 5. `Oli` class — Py 3.11 (Spec R-oli, D5, D7, D14) — *done 2026-06-22*

- [x] 5.1 Created `humanoid/logic/simulation/isaacsim/oli.py` with `Oli`: `__init__`, props `dof_names`/`num_dof`/`base_link_path`, methods `tick`/`apply_cmd`/`read_state`/`read_imu`/`tick_latency_stats`
- [x] 5.2 `__init__` flow: `add_reference_to_stage` (variant-selectable USD), spawn translate before reset, `fixRootLink=True` if `pin_root`, `world.reset()`, `SingleArticulation.initialize()`
- [x] 5.3 IMU sensor attached at `base_link` via `isaacsim.sensors.physics.IMUSensor` (task 2.5 confirmed no IMU prim ships in the USD). Quat confirmed `(w,x,y,z)` native — no reorder
- [x] 5.4 Built `pr_to_isaac` / `isaac_to_pr` index arrays at runtime from PR §11 vs `oli.dof_names`; asserts `num_dof == 31` + name set-equality
- [x] 5.5 `_CachedCmd` zero-initialized (q_d/dq_d/tau_ff/Kp/Kd float32, mode/parallel_solve int)
- [x] 5.6 `apply_cmd(q_d=..., dq_d=None, ...)` updates only passed kwargs; accepts numpy or sequence; asserts shape (31,)
- [x] 5.7 `tick()`: (bridge) read state+IMU → permute Isaac→PR → `send_state_imu` → drain `poll_cmd` → realize cmd. **PD realized via PhysX implicit drive** (`set_gains` on Kp/Kd change + position/velocity targets + additive `tau_ff` effort), NOT explicit `set_joint_efforts(tau)` — the latter rings unstably (D5 PD-realization)
- [x] 5.8 `read_state()` → `{stamp, q, dq, tau, motor_names}` PR-ordered
- [x] 5.9 `read_imu()` → `{stamp, acc, gyro, quat_wxyz}`
- [x] 5.10 Rolling p50/p99 tick-latency via `tick_latency_stats()` — measured p50=115µs, p99=204µs (10× under the 2ms budget)
- [x] 5.11 Standalone smoke (`_research/smoke_oli_nobridge.py`, `bridge=None`): Test 1 stable hold (velocity decays 23×, no ring), Test 2 joint step (commanded joint is biggest mover — permutation correct), Test 3 zero-cmd gravity sag. ALL PASS. Found + fixed the explicit-effort instability → implicit drive (D5)

## 6. `OliBridge` class — Py 3.11 (Spec R-bridge, D6, D11, D15) — *done 2026-06-22*

- [x] 6.1 `OliBridge` in `bridge/__init__.py` with both factory constructors (`spawn_sidecar`, `connect`)
- [x] 6.2 `connect(socket, timeout)`: AF_UNIX SOCK_SEQPACKET client, retry to timeout @100ms backoff, raises `ConnectionError`
- [x] 6.3 `spawn_sidecar(ip, socket, sidecar_py, debug)`: resolves interp (kwarg → `LIMX_BRIDGE_SIDECAR_PY` → default), `Popen`, daemon thread forwards stdout/stderr with `[sidecar]` prefix, waits for socket file (or surfaces early-exit code), then `connect`
- [x] 6.4 `handshake(dof_names)`: send HELLO, recv ack, raises `BridgeProtocolError` on EOF/bad-ack/version mismatch; sets socket non-blocking after; idempotent
- [x] 6.5 `send_state_imu(...)`: pack + `send`; drop on `BlockingIOError`; raises `BridgeClosedError` if sidecar gone
- [x] 6.6 `poll_cmd()`: non-blocking recv; returns decoded CMD dict or None; EOF → `BridgeClosedError`; malformed → drop+log
- [x] 6.7 `__enter__`/`__exit__` + `_kill` escalation (SIGINT → 5s → SIGTERM → 2s → SIGKILL)
- [x] 6.8 `close()`: idempotent; closes socket, tears down owned subprocess, unlinks socket if left over
- [x] 6.9 Smoke (`spawn_sidecar`, cross-env): `_research/smoke_oli_bridge.py` run in isaac env (Py 3.11) spawns sidecar in limx env (Py 3.8); handshake OK, 100 STATE_IMU → `state_pub=100`, clean SIGINT shutdown + socket unlinked. **Proves the cross-env two-process path end-to-end.**
- [ ] 6.10 Smoke (`connect` path): *deferred to Phase 7* — exercised naturally when load_oli.py uses an externally-started sidecar

## 7. End-to-end smoke tests against deploy-python (Spec R-end-to-end)

- [ ] 7.0 Rewrite `load_oli.py` as smoke demo (≤30 LOC substantive): `with OliBridge.spawn_sidecar(ip=...) as bridge: oli = Oli(world, bridge=bridge); while SIM_APP.is_running(): oli.tick(); world.step(render=(tick % 20 == 0))`. Keep `--no-bridge` flag for kinematics-only loader.
- [ ] 7.1 Run `python load_oli.py --ip 127.0.0.1`; in another shell, start `humanoid-rl-deploy-python` damping controller (`ROBOT_TYPE=HU_D04_01 python main.py 127.0.0.1`); observe Oli holds pose without drift for ≥30 s
- [ ] 7.2 Capture `Oli.tick()` latency histogram during 7.1; confirm p99 < 1 ms (target) or document if above
- [ ] 7.3 Run `load_oli.py` + deploy-python stand controller (joystick L1+Y or `cli switch '' 'stand'`); observe Oli reaches stand pose, holds (or fails informatively — log what we see)
- [ ] 7.4 Optional: run `load_oli.py` + deploy-python walk controller (R1+X); observe walking behavior, document fidelity gaps vs MuJoCo (likely large due to D9 serial-ankle limitation)
- [ ] 7.5 Verify `subscribeRobotState` from a third Py 3.8 client (probe_contract.py policy role) returns Isaac's RobotState (proving Isaac is observable on the bus from any deploy-side peer)
- [ ] 7.6 Smoke `OliBridge.connect` path: run sidecar manually (`python -m humanoid.logic.simulation.isaacsim.bridge.sidecar --ip 127.0.0.1`); in another shell run a host app that uses `OliBridge.connect(...)`; confirm same end-to-end behavior

## 8. Documentation

- [ ] 8.1 Author `humanoid/logic/simulation/isaacsim/bridge/README.md`:
  - quickstart (`with OliBridge.spawn_sidecar(...) as bridge: ...`)
  - architecture diagram (sidecar ↔ OliBridge ↔ Oli ↔ articulation ↔ MROS bus)
  - workflows mapping (which workflows use bridge vs not — copy from design Workflows section)
  - API reference for `Oli` and `OliBridge`
  - known limitations: serial ankle (D9), `parallel_solve_required` ignored (D10), `RobotCmd.mode` ignored (D5/§5 humanoid-mujoco-sim)
- [ ] 8.2 Update `humanoid/docs/vendor/humanoid-mujoco-sim.md`: add a `## 11. Isaac counterpart` section linking to `bridge/README.md`
- [ ] 8.3 Update `humanoid/docs/vendor/humanoid-rl-deploy-python.md`: add a `## 12. Running against Isaac` section
- [ ] 8.4 Update `humanoid/logic/simulation/isaacsim/README.md` (create if absent): overview + workflows mapping + pointer to `bridge/README.md`

## 9. Validation

- [ ] 9.1 `openspec validate may-147-isaac-limx-sdk-bridge` — must report valid
- [ ] 9.2 Walk every `#### Scenario:` in `specs/isaac-limx-sdk-bridge/spec.md`; record pass/fail in `_research/spec_walk.md`
- [ ] 9.3 Save memories: `oli_component_api.md`, `oli_bridge_api.md`, `protocol_v0.md`, `joint_permutation_pr_isaac.md`
- [ ] 9.4 Update daily note with summary + tick-latency findings
- [ ] 9.5 Linear MAY-147 → Done; archive change (`openspec archive may-147-isaac-limx-sdk-bridge` — do not run, note readiness)

## 10. Follow-up work (out of scope here; capture as new tickets)

- [ ] 10.1 Open Linear ticket: "Parallel-ankle USD + Isaac kinematic projection" (consumer of D9 deferral)
- [ ] 10.2 Open Linear ticket: "Evaluate ros2-bridger as alternative bridge architecture" — populate from background research report once landed (OQ7)
- [ ] 10.3 Open Linear ticket: "Ask LimX for CPython 3.10/3.11 wheels" — draft external message; track response
