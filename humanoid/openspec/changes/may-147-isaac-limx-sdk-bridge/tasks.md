## 1. OpenSpec scaffolding

- [x] 1.1 `openspec new change may-147-isaac-limx-sdk-bridge`
- [x] 1.2 Author `proposal.md`
- [x] 1.3 Author `design.md`
- [x] 1.4 Author `tasks.md` (this file)
- [x] 1.5 Author `specs/isaac-limx-sdk-bridge/spec.md`
- [x] 1.6 `openspec validate may-147-isaac-limx-sdk-bridge` — reported valid 2026-06-22

## 2. Environment + asset prechecks (D11, OQ2, OQ3)

- [ ] 2.1 Confirm `limx` conda env has `limxsdk` 4.0.1 installed: `/home/may33/miniconda3/envs/limx/bin/python -c "import limxsdk; print(limxsdk.__version__)"`
- [ ] 2.2 Confirm `isaac` conda env has `isaacsim` 5.0.0 installed and importable from a fresh shell
- [ ] 2.3 Run `load_oli.py` once and capture printed DOF list to `_research/isaac_dof_dump.txt`
- [ ] 2.4 Compare against MAY-145 § 11 PR joint order; confirm count == 31; record any name mismatches in `_research/joint_name_audit.md`
- [ ] 2.5 Enumerate IMU sensor prims under `/World/Oli`: open USD in Isaac, list prims with `omni.isaac.sensor.IMUSensor` schema. Record findings in `_research/imu_prim_audit.md` (OQ2)
- [ ] 2.6 Inspect `MROS_IP_LIST` value used by `humanoid-mujoco-sim/simulator.py`; record canonical form for sidecar to copy verbatim

## 3. Protocol layer (Spec R-protocol)

- [ ] 3.1 Create `humanoid/logic/simulation/isaacsim/bridge/__init__.py`
- [ ] 3.2 Implement `bridge/protocol.py`: message-type constants (`HELLO=0`, `CMD=1`, `STATE_IMU=2`), version=`0`, struct formats per design D3+D4
- [ ] 3.3 Implement `pack_hello(dof_names) -> bytes` and `unpack_hello(buf) -> (dof_count, dof_names)`
- [ ] 3.4 Implement `pack_cmd(seq, stamp, mode, q, dq, tau, kp, kd, parallel_solve_required) -> bytes` and `unpack_cmd(buf)`
- [ ] 3.5 Implement `pack_state_imu(seq, stamp, q, dq, tau, quat_xyzw, gyro, accel) -> bytes` and `unpack_state_imu(buf)`
- [ ] 3.6 Cross-version pack identity test: produce a canned 100-byte packet in both `limx` (Py 3.8) and `isaac` (Py 3.11) envs; assert sha256 match
- [ ] 3.7 Round-trip test: pack → unpack returns identical values within float32 tolerance
- [ ] 3.8 Document the byte layout in `protocol.py` module docstring with an ASCII offset table

## 4. Sidecar — Py 3.8 process (Spec R-sidecar, D1, D6)

- [ ] 4.1 Implement `bridge/sidecar.py` skeleton with argparse: `--ip` (default `127.0.0.1`), `--socket` (default `/tmp/limx-isaac-bridge.sock`)
- [ ] 4.2 Set `MROS_IP_LIST` env var from `--ip` (form copied from § 2.6 audit)
- [ ] 4.3 Construct `Robot(RobotType.Humanoid, True)`; call `robot.init(ip)`
- [ ] 4.4 Open AF_UNIX SOCK_SEQPACKET socket; bind to `--socket`; listen(1); accept one client
- [ ] 4.5 Receive `HELLO`; verify `dof_count == 31`; build set-equality check against expected PR joint names; close + exit non-zero on mismatch
- [ ] 4.6 Ack `HELLO` (sidecar sends a zero-payload HELLO back so the driver knows it's good to start)
- [ ] 4.7 Register `subscribeRobotCmdForSim(callback)`; callback packs the cmd and writes it to the socket (non-blocking; drop on full buffer with a counter log)
- [ ] 4.8 Main loop: blocking `recv()` on the socket for `STATE_IMU`; unpack; call `robot.publishRobotStateForSim(state)` + `robot.publishImuDataForSim(imu)`
- [ ] 4.9 EOF on socket → log + clean exit (unlink socket file; call `robot.deinit()` if exposed)
- [ ] 4.10 Catch `SIGTERM` from launcher → clean exit
- [ ] 4.11 Smoke 4.10: run sidecar standalone, run `humanoid-rl-deploy-python` damping controller; observe `subscribeRobotCmdForSim` callback fires and cmd bytes arrive in the socket buffer (no Isaac yet; sidecar prints recv'd cmd to stderr in debug mode)

## 5. `Oli` class — Py 3.11 (Spec R-oli, D5, D7, D14)

- [ ] 5.1 Create `humanoid/logic/simulation/isaacsim/oli.py` with `Oli` skeleton: `__init__`, properties `dof_names` / `num_dof`, methods `tick`, `apply_cmd`, `read_state`, `read_imu`
- [ ] 5.2 `__init__` flow: `add_reference_to_stage(HU_D04_01.usd)` (variant-selectable), set spawn translate before reset, set `physxArticulation:fixRootLink = True` if `pin_root`, `world.reset()`, init `SingleArticulation`
- [ ] 5.3 IMU sensor setup at `base_link` (D8): prefer existing IMU prim in `_sensor.usd` layer if present (per task 2.5 audit); otherwise attach `omni.isaac.sensor.IMUSensor`
- [ ] 5.4 Build `pr_to_isaac` and `isaac_to_pr` numpy index arrays from PR canonical joint list (§ 11) against `oli.dof_names`; assert `num_dof == 31` and set-equality of names
- [ ] 5.5 Initialize `cached_cmd` with zero arrays (`q_d`, `dq_d`, `tau_ff`, `Kp`, `Kd` — all 31, float32)
- [ ] 5.6 Implement `apply_cmd(q_d=..., dq_d=None, ...)`: update only the kwargs that were passed; PR-space arrays only; assert length 31
- [ ] 5.7 Implement `tick()` with optional bridge:
  - read `q`, `dq`, `tau` from articulation (verify exact Isaac API — likely `get_joint_positions()` etc.)
  - read IMU sample (`acc`, `gyro`, `quat_wxyz`)
  - if bridge attached: permute Isaac→PR, `bridge.send_state_imu(...)`, `bridge.poll_cmd()` → update `cached_cmd` if non-None
  - compute `tau_apply_pr = Kp*(q_d − q_pr) + Kd*(dq_d − dq_pr) + tau_ff`
  - permute PR→Isaac; `articulation.set_joint_efforts(tau_apply_isaac)`
- [ ] 5.8 Implement `read_state()` returning `{stamp, q, dq, tau, motor_names}` PR-ordered
- [ ] 5.9 Implement `read_imu()` returning `{stamp, acc, gyro, quat_wxyz}`
- [ ] 5.10 Instrument tick with `time.perf_counter`: maintain rolling p50/p99 histograms for the host to query
- [ ] 5.11 Standalone smoke (`bridge=None`): instantiate `Oli`, run a host loop that calls `oli.apply_cmd(...)` with a hand-crafted PR-space target moving one joint by 0.1 rad; observe expected motion

## 6. `OliBridge` class — Py 3.11 (Spec R-bridge, D6, D11, D15)

- [ ] 6.1 Create `bridge/__init__.py` with `OliBridge` skeleton + the two factory constructors (`spawn_sidecar`, `connect`)
- [ ] 6.2 Implement `OliBridge.connect(socket, timeout)`: open AF_UNIX SOCK_SEQPACKET client; connect with retry up to `timeout`, 100ms backoff; on failure log + raise `ConnectionError`
- [ ] 6.3 Implement `OliBridge.spawn_sidecar(ip, socket, sidecar_py)`: resolve sidecar interpreter from kwarg → env var (`LIMX_BRIDGE_SIDECAR_PY`) → hardcoded default; `subprocess.Popen` the sidecar; pipe stdout/stderr through a thread that prefixes `[sidecar] ` and forwards; wait for socket file to appear with timeout; then call `connect`
- [ ] 6.4 Implement `handshake(dof_names)`: send `HELLO`; receive ack; on EOF or version mismatch raise `BridgeProtocolError`
- [ ] 6.5 Implement `send_state_imu(seq, stamp_ns, q, dq, tau, acc, gyro, quat)` — non-blocking `sendmsg`; drop on full buffer with a counter log
- [ ] 6.6 Implement `poll_cmd()` — drain socket non-blocking, return last decoded `CMD` or `None`; intermediate frames silently dropped per D5
- [ ] 6.7 Implement `__enter__`/`__exit__` (only meaningful for `spawn_sidecar`-created instances): `__exit__` SIGINTs the sidecar, waits 5s, escalates to SIGTERM then SIGKILL, then `close()`
- [ ] 6.8 Implement `close()`: idempotent; close socket, unlink socket file if we created it, join the stdout-forwarder thread
- [ ] 6.9 Smoke (`spawn_sidecar` path): `with OliBridge.spawn_sidecar(ip="127.0.0.1") as bridge: ...` from a `__main__` test in `bridge/__init__.py`; observe sidecar starts, socket appears, handshake succeeds with stub `dof_names`, clean shutdown on context exit
- [ ] 6.10 Smoke (`connect` path): manually run sidecar in one shell; in another, `OliBridge.connect(...)` succeeds and the socket round-trips a HELLO+ack

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
