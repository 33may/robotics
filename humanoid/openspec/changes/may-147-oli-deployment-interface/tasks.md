## 1. OpenSpec scaffolding & predecessor archive

- [x] 1.1 Author `tasks.md` (this file)
- [x] 1.2 `openspec validate may-147-oli-deployment-interface --strict` — must report valid
- [x] 1.3 Archive predecessor `may-147-isaac-limx-sdk-bridge` (CONCLUSION.md already written)

## 2. Brain conda env (D2)

- [x] 2.1 Create `brain` conda env (Py 3.11.15: onnxruntime 1.27.0, scipy 1.17.1, numpy 2.4.6)
- [x] 2.2 Load walk `policy.onnx` in the env; confirmed input `obs[1,510]` / output `actions[1,31]`, zero-obs forward finite
- [x] 2.3 Confirmed `brain` env has neither `isaacsim` nor `limxsdk` (find_spec → False both)

## 3. Contracts — `logic/oli/contracts.py` (Spec: three canonical-PR contracts, D3)

- [x] 3.1 `PR_ORDER` (31) relocated into `contracts.py`; `Mode` IntEnum {STAND, WALK} (extensible selector)
- [x] 3.2 `Observation` frozen dataclass: stamp_ns(sim-time), q/dq/tau[31], acc/gyro[3], quat_wxyz[4] — PR order, unscaled, f32-coerced
- [x] 3.3 `Intent` (mode + v_x,v_y,w_z expected-input, defaulted) + `PolicyIn` (observation + intent)
- [x] 3.4 `PolicyOut` frozen dataclass: q_des, dq_des, tau_ff, kp, kd [31] + per-joint motor mode — resolved RobotCmd
- [x] 3.5 Contracts carry no world-order/scaling/history; `__post_init__` shape-asserts; verified pure (no isaacsim/limxsdk in import graph)

## 4. Communication — `logic/oli/comm/` (Spec: Comm is the only world-aware adapter, D4, D10, D11)

- [x] 4.1 Reframe `bridge/protocol.py` → `comm/protocol.py` (pure wire moved verbatim, docstring re-homed; regression-locked, 7 tests). bridge/ orphaned → delete in §8
- [x] 4.1b `comm/codec.py` — pure contract↔wire (Observation↔STATE_IMU, PolicyOut↔CMD); TDD red→green, 4 round-trip tests
- [x] 4.2 `comm/base.py`: `Comm` ABC — `connect()`, `read_observation()` (latest-wins), `write_policy_out()`, `close()`
- [x] 4.3 `comm/client.py`: `BrainComm` client — connect to World server, handshake, send PolicyOut, drain-to-newest Observation
- [x] 4.4 `isaacsim/sim_comm.py` `SimComm` SERVER: bind/listen/accept, PR↔Isaac permutation, read body (Isaac)→Observation (PR), apply PolicyOut (PR)→body (Isaac). Import-pure (body injected). Loopback integration test: 4 cases, real UDS, no mocks
- [x] 4.5 No control/policy logic in Comm — only reorder/convert/apply (verified: codec + permutation only, World owns stamp per D8)

## 5. Slim `oli.py` → pure articulation (D4)

- [ ] 5.1 Strip in-class bridge I/O: remove `BridgeLike`, `_bridge`, send/poll branch in `tick()`
- [ ] 5.2 Relocate PR↔Isaac permutation out of `oli.py` into `SimComm`; `oli.py` exposes Isaac-order state + Isaac-order apply
- [ ] 5.3 `oli.py` API: `read_state_isaac()`, `read_imu()`, `apply_cmd_isaac(q_d,dq_d,tau_ff,kp,kd)` (implicit PD drive retained), `step`/`tick` is host's job
- [ ] 5.4 Standalone smoke (no comm) still green: stable hold, joint step, gravity sag

## 6. Reasoning — `logic/oli/reason/teleop.py` (Spec: Reasoning is the single world-processor, D5)

- [x] 6.1 `JoystickAdapter`: axes → `(v_x, v_y, w_z)` (joy[1]→v_x, joy[0]→v_y, joy[3]→w_z; clip to max_vx/vy/vz). Faithful to `walk_controller._update_commands_from_joy`. Mode is operator-held (deploy switches via ability CLI, not stick)
- [x] 6.2 `Teleop`: `(Observation, joystick axes) → PolicyIn` directly (no Reason→Command seam); held/settable mode. 5 tests

## 7. Action / PolicyRunner — `logic/oli/action/policy_runner.py` ★ THE CRUX (Spec: Action owns all policy logic, D6, D8)

- [x] 7.1 Cross-check vs LimX TRON repos: confirmed standard LimX deploy pattern (RL_TYPE=isaaclab, ONNX, history/scales/proj-gravity). TRON1 is a different robot; authoritative = our `walk_controller.py`
- [x] 7.2 `WalkPolicy` loads `walk_param.yaml`: action_scale, kp, kd, default_angle, user_torque_limit, obs_scales, decimation (spot-checked in tests)
- [x] 7.3 obs[102] encoder `encode_walk_obs`, verified component-by-component vs `walk_controller.compute_observation` (4 tests, incl. term layout)
- [x] 7.4 projected_gravity: quat wxyz→xyzw, `Rᵀ·[0,0,-1]` — proven == deploy's euler('zyx') roundtrip to 1e-5
- [x] 7.5 5-deep history ring (newest-first) → obs[510]; first-obs replicate ×5; `last_actions` memory in the runner (=clamped, aliasing fidelity)
- [x] 7.6 ONNX session (CPUExecutionProvider, intra/inter=1); run; clip actions to ±clip_actions
- [x] 7.7 Resolution to PolicyOut: per-joint torque-limit clamp on live q/dq, `q_des = a·action_scale + default_angle`, dq_des=0, tau_ff=0, attach kp/kd
- [x] 7.8 `StandPolicy`: analytic ramp (captured spawn pose → stand_pos over 2 s, stamp-paced), stiff stand gains, no ONNX. `PolicyRunner` dispatches by mode + re-seeds on switch (cold-start mitigation). 6 tests
- [x] 7.9 Unit checks: encoder matches `walk_controller` reference (4 tests) + history 510 newest-first replicate-×5 (covers obs[510])

## 8. Orchestrator + Sim World main (Spec: async run cycle, freeze-until-cmd, watchdog; D1, D7, D8, D9)

- [ ] 8.1 `logic/oli/runtime.py` `Orchestrator`: `read → reason → act → write` loop; pace by world-stamp Δ≥10 ms; latest-wins; owns logging/recording of every contract; watchdog
- [ ] 8.2 `logic/oli/brain_main.py`: wire `comm.client` + `Teleop` + `PolicyRunner` + `Orchestrator`; runs in `brain` env
- [ ] 8.3 `isaacsim/sim_world_main.py` (from `load_oli.py`): Isaac boot, `World`+`Oli`+`SimComm` server, 1 kHz loop, **freeze-until-first-PolicyOut**, watchdog fail-safe damping
- [ ] 8.4 `RealComm` stub (interface only; from `sidecar.py`, role flipped sim-peer→real-robot) — deferred, must not preclude

## 9. Walk milestone (Spec: walk milestone acceptance)

- [ ] 9.1 Launch Sim World + brain (two processes); STAND mode holds the pose
- [ ] 9.2 Command forward velocity in WALK → Oli transitions to gait and translates forward in the viewport
- [ ] 9.3 Confirm the brain process imported neither `isaacsim` nor `limxsdk` (module-graph assert)

## 10. Validation, memory, daily

- [ ] 10.1 `openspec validate may-147-oli-deployment-interface --strict`; walk every `#### Scenario:` → record pass/fail
- [ ] 10.2 Save memories: contracts shape, comm topology (World=server/brain=client), policy_runner obs layout, brain env, slimmed-oli API
- [ ] 10.3 Update humanoid daily note with milestone + findings (show draft first)
- [ ] 10.4 Linear MAY-147 status update; note archive readiness
