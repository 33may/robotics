## ADDED Requirements

### Requirement: Deployment-invariant brain

The Reasoning and Action layers (the "brain") SHALL be authored once and run unchanged against any World realization. The brain SHALL import neither `isaacsim` nor `limxsdk`, nor reference any world-native joint order or mechanism — all world-specific knowledge lives in Communication. The same brain process, env, and code SHALL drive the Isaac simulation and the physical robot; only the Communication realization differs.

#### Scenario: Brain imports no world SDK

- **WHEN** the brain package (`humanoid/logic/oli/{contracts,comm/client,reason,action,runtime}`) is imported in the `brain` env
- **THEN** the import succeeds with neither `isaacsim` nor `limxsdk` in the module dependency graph
- **AND** no symbol in the brain references an Isaac DOF index or a limxsdk type

#### Scenario: Same brain code targets sim and real

- **WHEN** the operator launches the brain against `SimComm` and later against `RealComm`
- **THEN** the Reasoning and Action source files are byte-identical between the two runs
- **AND** the only differing wiring is which Communication realization the Orchestrator is composed with

### Requirement: World is an independent free-running process

The World SHALL run as its own process with its own 1 kHz physics/control loop, independent of the brain. It SHALL be the authoritative source of body state and SHALL NOT wait for the brain to step. The World SHALL act as the connection server; the brain SHALL connect to it as a client (mirroring the real robot, which is the endpoint a controller connects into).

#### Scenario: World runs its own loop

- **WHEN** the brain process is absent or stalled
- **THEN** the World process continues running its physics loop and emitting Observations
- **AND** the World never blocks waiting on the brain

#### Scenario: World is the server, brain is the client

- **WHEN** the system starts
- **THEN** the World process opens the connection endpoint and listens
- **AND** the brain process connects to it and performs the handshake

### Requirement: Three canonical-PR invariant contracts

The brain↔world boundary SHALL be expressed as exactly three contracts in canonical PR space (31 joints, `wxyz` quaternion): `Observation` (stamp, q/dq/tau, imu acc/gyro/quat) flowing World→brain; `PolicyIn` (an Observation plus intent: mode and body-frame velocities) flowing Reason→Action; and `PolicyOut` (q_des/dq_des/tau_ff/kp/kd/mode) flowing Action→World. Contracts SHALL carry no world-native ordering, no scaling, and no policy history.

#### Scenario: Observation is a raw physical snapshot

- **WHEN** the World emits an Observation
- **THEN** its q/dq/tau arrays are length 31 in PR order, unscaled
- **AND** it carries no encoded obs vector, no history, and no last-actions

#### Scenario: PolicyOut is a resolved PD command

- **WHEN** the Action layer emits a PolicyOut
- **THEN** it contains resolved per-joint q_des plus Kp/Kd in PR order (a RobotCmd shape)
- **AND** the World can apply it without knowing any policy scale, default angle, or gain

### Requirement: Communication is the only world-aware adapter

Communication SHALL be the sole layer that knows world-native types and joint order. It SHALL convert between canonical-PR contracts and the world representation: `SimComm` performs the PR↔Isaac DOF permutation, reads the articulation into an Observation, and applies a PolicyOut to the articulation via the implicit PD drive; `RealComm` (deferred) performs PR↔limxsdk-order and the parallel↔serial (AB↔PR) mechanism conversion. No control or policy logic SHALL live in Communication.

#### Scenario: SimComm owns the permutation

- **WHEN** SimComm reads the Isaac articulation (native DOF order) to build an Observation
- **THEN** it permutes the values into PR order before they cross the contract
- **AND** the brain never receives Isaac-ordered data

#### Scenario: Communication carries no policy logic

- **WHEN** Communication processes a PolicyOut
- **THEN** it only reorders/converts and applies; it does not scale actions, add default angles, or choose gains

### Requirement: Reasoning is the single world-processor

Reasoning SHALL be the only producer of `PolicyIn`. It SHALL take an `Observation` plus an external operator signal (the joystick) and emit `PolicyIn` directly — there is no separate Reason→Action command seam. The foundation realization SHALL be Teleoperation with a JoystickAdapter that maps joystick axes to `{mode, v_x, v_y, w_z}`.

#### Scenario: Reasoning combines world and joystick into PolicyIn

- **WHEN** Reasoning receives an Observation and the joystick reports forward deflection
- **THEN** it emits a PolicyIn containing that Observation plus an intent with v_x > 0 and mode = WALK
- **AND** Reasoning performs no policy obs-vector encoding

### Requirement: Action owns all policy logic and cross-step memory

The Action/PolicyRunner layer SHALL contain all policy-specific logic: obs-vector encoding (per-component scales, quaternion `wxyz→xyzw` reorder, projected-gravity), the policy's cross-step memory (the proprioceptive history ring and last-actions), the ONNX session, and resolution of raw actions to a PD command (`action·scale + default_angle`, gains, torque-limit clamp). It SHALL expose a STAND mode (analytic ramp to the stand pose) and a WALK mode (the walk ONNX). It SHALL NOT build the unused gait observation terms (commented out of the shipped walk obs).

#### Scenario: PolicyRunner builds the walk observation and runs ONNX

- **WHEN** the PolicyRunner is in WALK mode and receives a PolicyIn
- **THEN** it encodes a 102-dim observation (ang_vel·0.25, projected_gravity, commands, (q−default)·1.0, dq·0.05, last_actions), pushes it into a 5-deep history forming a 510-vector, and runs the walk ONNX
- **AND** it resolves the 31 outputs to q_des = action·action_scale + default_angle and attaches the configured Kp/Kd

#### Scenario: Cross-step memory lives in the runner, not the contracts

- **WHEN** two consecutive policy steps occur
- **THEN** the runner advances its own history ring and updates last_actions internally
- **AND** neither Observation, PolicyIn, nor PolicyOut carries history or last_actions

### Requirement: Asynchronous run cycle with world-stamp pacing

The brain and World SHALL communicate asynchronously with latest-wins semantics in both directions: the brain SHALL drain to the newest Observation, and the World SHALL hold and re-apply the most recent PolicyOut every physics tick. The brain SHALL trigger a policy step paced by the World's stamp delta (one step per ≥10 ms of simulation time, the trained decimation), NOT by brain wall-clock, so the policy's effective timestep is correct at any real-time factor.

#### Scenario: Policy steps are paced by simulation time

- **WHEN** the World runs slower than real time (RTF < 1)
- **THEN** the brain still triggers exactly one policy step per ~10 ms of advancing Observation stamp
- **AND** the 5-deep history continues to span ~50 ms of simulation time

#### Scenario: Dropped packets are harmless

- **WHEN** an Observation or a PolicyOut frame is dropped on the wire
- **THEN** the brain proceeds with the next newest Observation and the World keeps applying the last PolicyOut
- **AND** no retransmission or blocking occurs

### Requirement: Freeze-until-command startup safety

The World SHALL emit Observations of the static spawn pose but SHALL NOT step physics until it has received the first PolicyOut. This SHALL prevent a free-base Oli from collapsing during brain/ONNX bring-up while still letting the brain bootstrap off the frozen-pose Observation.

#### Scenario: World holds until the first command

- **WHEN** the World has spawned Oli but no PolicyOut has yet arrived
- **THEN** it publishes Observations of the static pose
- **AND** it does not call world.step(), so Oli does not move or fall
- **AND** on receipt of the first PolicyOut it begins stepping

### Requirement: Stale-command watchdog

The World SHALL run a watchdog that, if no fresh PolicyOut arrives within a configured timeout while stepping, transitions to a fail-safe (damping) state rather than indefinitely holding a stale command. A held command that is safe at standstill is unsafe mid-stride.

#### Scenario: Watchdog trips on brain stall

- **WHEN** the World is stepping and receives no PolicyOut for longer than the watchdog timeout
- **THEN** it switches the articulation to a damping/hold fail-safe
- **AND** it logs the stale-command event

### Requirement: Walk milestone acceptance

The system SHALL demonstrate Oli standing and then walking in the Isaac viewport, driven by the LimX walk ONNX through this interface, steered by a joystick command via Teleop. The brain SHALL run in the dedicated `brain` env as a separate process from the Isaac World.

#### Scenario: Oli walks in Isaac on the walk ONNX

- **WHEN** the Sim World process (Isaac + Oli + SimComm) and the brain process are both running and the operator commands forward velocity in WALK mode
- **THEN** Oli transitions from stance into a walking gait and translates forward in the viewport
- **AND** the brain process has neither imported `isaacsim` nor `limxsdk`
