## ADDED Requirements

### Requirement: Bridge presents Isaac Sim as a sim peer on the MROS bus

The bridge SHALL make Isaac Sim externally indistinguishable from `humanoid-mujoco-sim` to any SDK consumer on the MROS bus. Any subscriber that previously connected to the MuJoCo simulator MUST be able to connect to Isaac via the same `Robot(type)` + `init("127.0.0.1")` pattern with no code changes, and receive byte-compatible `RobotState`, `ImuData`, and `RobotCmd` packets.

#### Scenario: A deploy-python damping controller connects to the bridge as if it were the MuJoCo sim

- **WHEN** the bridge is running and a separate Py 3.8 process executes `humanoid-rl-deploy-python` with `ROBOT_TYPE=HU_D04_01` and `robot_ip=127.0.0.1`
- **THEN** the deploy-python's `subscribeRobotState` callback fires with packets whose `motor_names` are in PR order and whose `q`/`dq`/`tau` have length 31
- **AND** the deploy-python's RobotCmd publish reaches the sidecar's `subscribeRobotCmdForSim` callback within one MROS round-trip

#### Scenario: The sidecar is constructed in sim-peer role

- **WHEN** the sidecar process starts
- **THEN** it constructs `Robot(RobotType.Humanoid, True)` (the `True` argument explicitly puts the SDK in sim-peer role)
- **AND** it registers `subscribeRobotCmdForSim` (NOT `subscribeRobotCmd`)
- **AND** it never registers `subscribeRobotState` or `subscribeImuData` (those are policy-peer subscriptions)

### Requirement: Bridge is a two-process system spanning two Python versions

The bridge SHALL consist of two cooperating processes — a sidecar in the `limx` conda env (CPython 3.8.18) that owns the `limxsdk.Robot` connection, and a host process in the `isaac` conda env (CPython 3.11.14) that owns Isaac Sim physics and instantiates `Oli` + `OliBridge`. The two processes SHALL communicate exclusively over a local IPC channel; neither process imports modules that exist only in the other process's environment.

#### Scenario: Sidecar runs only in the limx env

- **WHEN** the sidecar process is started
- **THEN** its Python interpreter resolves to a CPython 3.8 binary in a conda env where `import limxsdk` succeeds
- **AND** the process does not import any module from the `isaac` env (no `isaacsim`, no `pxr`)

#### Scenario: Host process runs only in the isaac env

- **WHEN** the host process (e.g., `load_oli.py`) is started
- **THEN** its Python interpreter resolves to a CPython 3.11 binary in a conda env where `import isaacsim` succeeds
- **AND** the process does not `import limxsdk` (which would fail with an ABI error)

### Requirement: `Oli` is a reusable Isaac component

The `Oli` class SHALL be importable into any Isaac-based host application with a single import line, configurable with a small set of constructor kwargs, and operable in two modes — with a bridge (`Oli(world, bridge=bridge)`) and without (`Oli(world, bridge=None)`). When `bridge=None`, `Oli` SHALL still load USD, initialize the articulation, attach the IMU sensor, and accept low-level commands via `apply_cmd(...)`; it SHALL NOT attempt any IPC.

#### Scenario: Bridge-less Oli accepts direct cmds via apply_cmd

- **WHEN** a host instantiates `Oli(world, bridge=None)` and calls `oli.apply_cmd(q_d=<31-element PR array>, kp=<...>, kd=<...>)` followed by `oli.tick()` and `world.step()`
- **THEN** Oli's articulation accelerates toward the target without raising any exception about a missing bridge
- **AND** no AF_UNIX socket is opened by the host process

#### Scenario: Oli is importable from a host app with no bridge dependency

- **WHEN** a host module does `from humanoid.logic.simulation.isaacsim import Oli` but NOT `from ...bridge import OliBridge`
- **THEN** the import succeeds without loading any `bridge/` submodule
- **AND** instantiating `Oli(world, bridge=None)` works with no `OliBridge` or `subprocess` import in the module's dependency graph

#### Scenario: Same Oli class powers both smoke demo and a custom host app

- **WHEN** `load_oli.py` (smoke demo) and a hypothetical `recon_app.py` (custom scene) both `from humanoid.logic.simulation.isaacsim import Oli`
- **THEN** both apps see the same class
- **AND** both apps can construct it against their own `World` and `prim_path` arguments without modifying `oli.py`

### Requirement: `OliBridge` supports two constructors (spawn vs connect)

The `OliBridge` class SHALL provide two named factory constructors that signal subprocess ownership intent. `OliBridge.spawn_sidecar(...)` SHALL start a Py 3.8 sidecar subprocess and SHALL own its lifecycle (terminate on `__exit__` / `close`). `OliBridge.connect(...)` SHALL attach to an already-running sidecar socket and SHALL NOT terminate any subprocess on close.

#### Scenario: spawn_sidecar context manager owns the subprocess

- **WHEN** a host enters `with OliBridge.spawn_sidecar(ip="127.0.0.1") as bridge: ...`
- **THEN** a child sidecar process exists on entry
- **AND** within 5 s of exiting the `with` block the child process has exited
- **AND** the UDS socket file has been unlinked

#### Scenario: connect attaches to an externally-managed sidecar

- **WHEN** a sidecar has been started externally (e.g., from a separate shell with `python -m humanoid.logic.simulation.isaacsim.bridge.sidecar --ip ...`)
- **AND** a host calls `OliBridge.connect(socket="/tmp/limx-isaac-bridge.sock")`
- **THEN** the connection succeeds within the configured timeout
- **AND** calling `.close()` on the resulting bridge closes the socket but does NOT terminate the external sidecar process

### Requirement: IPC protocol is fixed-size, versioned, and Py3.8/3.11 byte-identical

The IPC protocol SHALL define exactly three message types (`HELLO`, `CMD`, `STATE_IMU`) plus a common 8-byte header carrying `version`, `type_code`, `payload_len`, and `seq`. All payload fields SHALL use explicit `struct.pack` little-endian fixed-width formats so that the same Python source, run in CPython 3.8 and CPython 3.11, produces byte-identical output for any input.

#### Scenario: Cross-version pack identity holds

- **WHEN** an identical Python `pack_cmd(...)` call is executed in the `limx` env (Py 3.8.18) and in the `isaac` env (Py 3.11.14) with the same inputs
- **THEN** the resulting `bytes` objects have identical sha256 digests

#### Scenario: Round-trip is lossless within float32 tolerance

- **WHEN** a `STATE_IMU` payload is packed with arbitrary float values and then unpacked
- **THEN** every numeric field round-trips within `1e-6` relative tolerance
- **AND** the `stamp` field round-trips exactly (uint64 is bit-exact)
- **AND** the `seq` field round-trips exactly (uint32 is bit-exact)

#### Scenario: Protocol version mismatch is rejected cleanly

- **WHEN** the sidecar receives a `HELLO` whose header carries a version different from the sidecar's compiled-in version
- **THEN** the sidecar logs the version mismatch with both numeric values
- **AND** the sidecar closes the connection without entering the relay loop
- **AND** the driver observes EOF and exits non-zero

### Requirement: Joint order on the wire is canonical PR ordering

Every joint-indexed payload that crosses the IPC SHALL be in PR-space order — the 31-entry order documented in `humanoid/docs/vendor/humanoid-rl-deploy-python.md` § 11, indices 0..30 from `left_hip_pitch_joint` to `right_wrist_roll_joint`. Indices 15 and 16 SHALL be `head_yaw_joint` and `head_pitch_joint` respectively (yaw before pitch — the on-robot config and live wire probe agreement is the source of truth). The driver, and only the driver, SHALL perform the permutation between PR order and Isaac's DOF order. The sidecar SHALL NOT permute joint payloads.

#### Scenario: Driver builds permutation from handshake

- **WHEN** the driver sends `HELLO` with `oli.dof_names` (Isaac DOF order, 31 entries)
- **THEN** the driver builds and caches two permutation index arrays: `pr_to_isaac[31]` and `isaac_to_pr[31]`
- **AND** every outgoing `STATE_IMU` permutes Isaac-DOF arrays through `isaac_to_pr` before packing
- **AND** every incoming `CMD` permutes through `pr_to_isaac` after unpacking

#### Scenario: Sidecar refuses dimension mismatch at handshake

- **WHEN** the driver sends `HELLO` with a `dof_count` other than 31
- **THEN** the sidecar logs the mismatch (expected 31, received `<n>`) and closes the connection
- **AND** the sidecar exits cleanly

#### Scenario: Head joint order on the wire is yaw-then-pitch

- **WHEN** the sidecar publishes any `RobotState` derived from `STATE_IMU` to the MROS bus
- **THEN** the published `motor_names[15] == "head_yaw_joint"` and `motor_names[16] == "head_pitch_joint"`

### Requirement: `Oli.tick()` applies PD-with-feedforward law each physics tick

`Oli.tick()` SHALL compute joint torques per the law `τ_apply[i] = Kp[i]·(q_d[i] − q[i]) + Kd[i]·(dq_d[i] − dq[i]) + τ_ff[i]`, where `q_d`, `dq_d`, `Kp`, `Kd`, `τ_ff` come from the cached cmd (last received via the bridge, OR last set via `apply_cmd(...)`, whichever is more recent). `Oli.tick()` SHALL apply the computed torques to the articulation via Isaac's effort API. `Oli` SHALL NOT interpret `RobotCmd.mode` (matching LimX's MuJoCo reference behavior). `Oli.tick()` SHALL NOT call `world.step()` or render — the host owns those.

#### Scenario: Zero-cmd produces no actuation drift

- **WHEN** `Oli` has received no `CMD` since startup and no `apply_cmd(...)` call has been made (cold cache: `q_d = dq_d = τ_ff = Kp = Kd = 0`)
- **AND** the host calls `oli.tick()` followed by `world.step()` repeatedly for 5 s
- **THEN** Oli holds the pinned spawn pose without joint drift over the 5 s window

#### Scenario: Hand-injected single-joint position step moves the expected joint

- **WHEN** the host calls `oli.apply_cmd(q_d=<zeros except idx 3 = 0.1>, kp=<zeros except idx 3 = 50.0>, kd=<zeros except idx 3 = 1.0>)` where idx 3 is `left_knee_joint` in PR order, then runs `tick()` + `step()` for 500 ms
- **THEN** `oli.read_state()["q"][3]` reaches `0.1 ± 0.01 rad`
- **AND** no other PR-indexed joint moves more than `0.01 rad` from its starting position

#### Scenario: RobotCmd.mode is recorded but does not change the actuator law

- **WHEN** a `CMD` arrives with `mode[i] = 2` (position control) for some `i`
- **THEN** `Oli` logs the non-zero mode value (warn-level, rate-limited) but still applies the PD-with-feedforward law
- **AND** no shipped LimX controller observably misbehaves (they all publish `mode = 0`)

### Requirement: RobotState and ImuData publish at sustained rate matching the MuJoCo reference

The bridge SHALL publish `RobotState` and `ImuData` on the MROS bus at a sustained rate of at least **850 Hz** over any 5-second observation window once steady state is reached. The MuJoCo reference achieves ~884.5 Hz on equivalent desktop hardware; Isaac SHOULD land within ±5% of that.

#### Scenario: Steady-state publish rate exceeds 850 Hz

- **WHEN** the bridge has been running with the driver applying zero cmd for at least 5 s
- **AND** a third-party process subscribes to `RobotState` and counts callbacks over the next 5 s
- **THEN** the count divided by elapsed seconds is at least 850

#### Scenario: ImuData and RobotState are published in the same tick

- **WHEN** the driver completes one physics tick
- **THEN** exactly one `RobotState` and exactly one `ImuData` packet are published on the bus by the sidecar
- **AND** their `stamp` fields are equal (both derived from the same tick's `time.time_ns()` on the driver side)

### Requirement: `Oli` latches the most recent cmd

`Oli` SHALL maintain a single cached cmd (`q_d`, `dq_d`, `τ_ff`, `Kp`, `Kd`, `mode`, `parallel_solve_required`). On each `tick()` call, when a bridge is attached, `Oli` SHALL non-blocking-drain all pending `CMD` frames from `bridge.poll_cmd()`, retaining only the most recent. `apply_cmd(...)` SHALL update the same cache directly without going through the bridge. `Oli` SHALL NOT queue, interpolate, or extrapolate cmds.

#### Scenario: Newer cmd replaces older cmd

- **WHEN** two `CMD` frames with sequence numbers `seq=N` and `seq=N+1` arrive between two consecutive `Oli.tick()` calls
- **THEN** the next `tick()`'s actuator law uses the values from `seq=N+1` and the values from `seq=N` are discarded

#### Scenario: Cmd-publish gap does not stop physics

- **WHEN** no `CMD` has arrived from the bridge for 100 ms
- **AND** the host continues calling `oli.tick()` + `world.step()`
- **THEN** Oli continues applying the most recent cached cmd
- **AND** no error is logged for the gap

#### Scenario: apply_cmd overrides the bridge cache

- **WHEN** a bridge is attached AND the host calls `oli.apply_cmd(q_d=X, kp=Y)` between two `tick()` calls AND no new `CMD` arrives from the bridge in between
- **THEN** the next `tick()` uses `X` and `Y` for `q_d` and `Kp`
- **AND** the previously bridge-cached `dq_d`, `τ_ff`, `Kd`, `mode`, `parallel_solve_required` remain in effect (apply_cmd only updates kwargs that were passed)

### Requirement: parallel_solve_required is accepted and asserted, not honored

The bridge SHALL accept the `parallel_solve_required` field from incoming `RobotCmd` packets and SHALL warn (without crashing) if any element is `false`. The bridge SHALL apply commands in PR space directly because the Isaac USD models serial-equivalent kinematics. The bridge SHALL NOT implement PR↔AB conversion in this change.

#### Scenario: All-true parallel_solve_required is silently accepted

- **WHEN** a `CMD` arrives with `parallel_solve_required[i] == true` for all `i` in `0..30`
- **THEN** no warning is logged about the flag

#### Scenario: A false element produces a warning, not a crash

- **WHEN** a `CMD` arrives with `parallel_solve_required[i] == false` for at least one `i`
- **THEN** a single warn-level log entry names the affected indices
- **AND** the driver continues applying the PD-with-feedforward law as if the flag were true

### Requirement: motor_names is omitted from the IPC frame

The IPC `CMD` message SHALL NOT carry the `RobotCmd.motor_names` field, even though the canonical `limxsdk::RobotCmd` struct includes it. The sidecar SHALL verify once at first cmd that the SDK-reported `motor_names` matches its compiled-in PR order (set-equality + count); subsequent cmds skip the check.

#### Scenario: Sidecar verifies motor_names exactly once

- **WHEN** the sidecar's `subscribeRobotCmdForSim` callback fires for the first time after startup
- **THEN** the sidecar logs at info level whether the cmd's `motor_names` matches the compiled PR order (set + count)
- **AND** the sidecar packs and forwards the cmd to the driver without the `motor_names` field
- **AND** subsequent cmd callbacks skip the verification but still strip `motor_names` from the IPC frame

### Requirement: IMU sensor is attached at base_link

The driver SHALL source IMU data (acc, gyro, quat) from a sensor anchored at the URDF link `base_link`, which is the canonical robot base frame per `oli-corpus://sdk-guide#4.1`. The quaternion SHALL be packed in `(w, x, y, z)` first-w order, matching both Isaac's native convention and the LimX `ImuData` struct documentation.

#### Scenario: IMU sample is in base_link frame at rest

- **WHEN** Oli is held at rest (pinned root, no joint torque applied) and the bridge is running
- **THEN** the sidecar publishes `ImuData` whose `quat` is approximately `[1.0, 0, 0, 0]` (within `0.05` Euclidean distance) — confirming first-w convention and base_link frame
- **AND** `acc` is approximately `[0, 0, 9.81]` m/s² (within `0.5`) — confirming body-frame gravity
- **AND** `gyro` is approximately `[0, 0, 0]` rad/s (within `0.05`)

### Requirement: Handshake establishes the session; EOF terminates it

The sidecar SHALL accept exactly one driver connection per run. The session SHALL begin with a `HELLO` exchange and SHALL terminate cleanly on either side observing EOF on the IPC socket. The bridge SHALL NOT implement auto-reconnect, retry, or heartbeat in v1.

#### Scenario: Successful handshake transitions to relay loop

- **WHEN** the sidecar accepts a driver connection and receives a well-formed `HELLO` with `dof_count == 31`
- **THEN** the sidecar sends a `HELLO` ack (empty payload) back to the driver
- **AND** the sidecar transitions into the cmd/state relay loop

#### Scenario: Driver EOF triggers clean sidecar shutdown

- **WHEN** the driver process exits or closes its IPC socket
- **THEN** the sidecar's next IPC read returns EOF
- **AND** the sidecar logs the EOF, calls `Robot.deinit()` if available, unlinks the socket file, and exits with code 0

#### Scenario: Sidecar crash triggers driver shutdown

- **WHEN** the sidecar process exits or closes its IPC socket
- **THEN** the driver's next IPC read returns EOF
- **AND** the driver closes the Isaac Sim app cleanly and exits with code 0

### Requirement: `OliBridge.spawn_sidecar` brings the stack up in one block

`OliBridge.spawn_sidecar(ip=..., socket=...)` SHALL act as a context manager that starts the Py 3.8 sidecar subprocess, multiplexes its stdout/stderr into the host process's stdout/stderr with a `[sidecar] ` prefix, completes the HELLO handshake when an `Oli` is constructed against it, and tears the sidecar down cleanly on `__exit__` (SIGINT → 5 s grace → SIGTERM → SIGKILL). The host process SHALL be able to bring up the full bridge with a single `with` block — no separate launcher script, no manual process management.

A smoke-demo script `load_oli.py` SHALL exist that demonstrates the one-block bring-up pattern in ≤30 lines of substantive code.

#### Scenario: One-block bring-up via context manager

- **WHEN** a host process executes `with OliBridge.spawn_sidecar(ip="127.0.0.1") as bridge: oli = Oli(world, bridge=bridge); ...`
- **THEN** a child sidecar process exists by the time the `with` body starts
- **AND** the `Oli` constructor completes the HELLO handshake within 30 s
- **AND** the host's combined stdout contains `[sidecar] ` prefixed log lines from the sidecar

#### Scenario: Context exit terminates the sidecar

- **WHEN** the `with` block exits (normal completion, exception, or KeyboardInterrupt)
- **THEN** within 5 s the sidecar process has exited
- **AND** the IPC socket file at the configured path has been unlinked

#### Scenario: Failed sidecar startup raises in the host

- **WHEN** the sidecar fails to construct `Robot(RobotType.Humanoid, True)` (e.g., `limxsdk` unavailable) and exits non-zero
- **THEN** `OliBridge.spawn_sidecar(...)` raises an exception with the sidecar's last stderr lines included in the message
- **AND** the host's `with` block never enters

#### Scenario: Sidecar crash mid-session raises on next bridge call

- **WHEN** the bridge is connected and operating, and the sidecar process unexpectedly exits
- **THEN** the next `bridge.send_state_imu(...)` or `bridge.poll_cmd()` call raises a `BridgeClosedError`
- **AND** the host can catch the exception, finalize Isaac state, and exit cleanly

### Requirement: Bridge is structurally deletable; `Oli` survives unchanged

If LimX publishes a CPython 3.10/3.11 wheel of `limxsdk` during the lifetime of this change, the bridge SHALL be removable by replacing the `OliBridge` class with a direct in-process SDK adapter that exposes the same `handshake`/`send_state_imu`/`poll_cmd` surface. The `Oli` class, its PD-actuation logic, the joint-permutation table, the IMU sourcing, and every host application using `Oli` SHALL remain unchanged after such migration.

#### Scenario: PD law and permutation logic live in `Oli`, not in the bridge

- **WHEN** `oli.py` is reviewed
- **THEN** the PD-with-feedforward law and the PR↔Isaac permutation table are implemented inside `Oli`, not inside any module under `bridge/`
- **AND** `Oli` does NOT import the sidecar module or the protocol module at top level
- **AND** the bridge interface used by `Oli` is structural (duck-typed `send_state_imu` / `poll_cmd` / `handshake` methods) so a non-IPC implementation can substitute without changing `Oli`'s code

### Requirement: kinematic_projection is out of scope and the limitation is documented

The bridge SHALL simulate **serial-equivalent ankle kinematics** as authored in the HU_D04_01 USD. The bridge SHALL NOT spawn `kinematic_projection` or any equivalent parallel-mechanism translator. The `bridge/README.md` SHALL document this limitation and link to the canonical config locations for future parallel-USD work.

#### Scenario: README documents the serial-ankle limitation

- **WHEN** `humanoid/logic/simulation/isaacsim/bridge/README.md` is read
- **THEN** it contains an explicit "Known limitations" section
- **AND** that section names "Serial-ankle kinematics only (parallel Achilles linkage not modeled)"
- **AND** it links to `oli-corpus://oli-main-2.2.12#install/etc/kinematic_projection/HU_D04_01/twisted_left_ankle_model.yaml` and the right-side equivalent
