# humanoid-mujoco-sim — vendor index

> Vendor repo: `humanoid/vendor/humanoid-mujoco-sim/` — cloned 2026-06-18 from `git@github.com:limxdynamics/humanoid-mujoco-sim`. Submodule pins recorded in § 2.

## 1. What it is

LimX's official MuJoCo simulation harness for their humanoid line. Combines three concerns in one repo:

- robot asset library (`humanoid-description` submodule) — URDF / MJCF / USD / meshes per model
- low-level SDK (`limxsdk-lowlevel` submodule) — C++ headers + prebuilt Python wheel
- joystick host (`robot-joystick` submodule) — controller pairing binary
- a 237-LOC `simulator.py` glue script and a `kinematic_projection` runtime that bridges the SDK to MuJoCo

## 2. Layout map

```
humanoid-mujoco-sim/
├── simulator.py              # MuJoCo ↔ SDK bridge — § 5
├── prebuild/
│   ├── kinematic_projection  # 9.4MB ELF binary — § 6
│   └── etc/kinematic_projection/<ROBOT>/
│       ├── config.yaml
│       ├── twisted_left_ankle_model.yaml
│       ├── twisted_right_ankle_model.yaml
│       └── waist_model.yaml
├── doc/simulator.gif         # demo recording
├── humanoid-description/     # SUBMODULE — assets, per model — § 4
│   ├── HU_D03_description/
│   └── HU_D04_description/
│       ├── meshes/HU_D04_01/   (~93 STL files)
│       ├── urdf/               HU_D04_01.urdf, _with_gripper, _with_hand, .srdf
│       ├── xml/                HU_D04_01.xml  (MJCF)
│       ├── usd/                HU_D04_01.usd (+ _with_gripper, _with_hand),
│       │                       configuration/*_{base,physics,sensor,robot}.usd
│       └── world/empty_world.world
├── limxsdk-lowlevel/         # SUBMODULE — low-level SDK — § 7
│   ├── include/limxsdk/      C++ headers: humanoid.h, datatypes.h, apibase.h, …
│   ├── lib/                  prebuilt C++ libs
│   ├── python3/              limxsdk-4.0.1-py3-none-any.whl  (amd64, aarch64, win)
│   │                         wheel ships libpython3.8.so.1.0 (Python 3.8 ABI)
│   └── examples/             ability/, api/
└── robot-joystick/           # SUBMODULE — joystick host
```

Submodule commits at clone time:

| Submodule | Commit |
|---|---|
| humanoid-description | `63eaa67b…` |
| limxsdk-lowlevel | `17a4b25d…` |
| robot-joystick | `30f69a9b…` |

## 3. Robot asset variants

HU_D04 description ships three end-effector variants:

| Variant | USD | URDF |
|---|---|---|
| Bare (no EE) | `HU_D04_01.usd` | `HU_D04_01.urdf` |
| Gripper EE | `HU_D04_01_with_gripper.usd` | `HU_D04_01_with_gripper.urdf` |
| Dexterous hand EE | `HU_D04_01_with_hand.usd` | `HU_D04_01_with_hand.urdf` |
| Semantic description | — | `HU_D04_01.srdf` |

USD layering (same pattern for all variants):

```
<variant>.usd → configuration/<variant>_sensor.usd
              → configuration/<variant>_physics.usd
              → configuration/<variant>_base.usd
```

By convention: `_base` = geometry, `_physics` = articulation/inertia/collision, `_sensor` = sensors/cameras/IMU. Layer contents not yet inspected; flatten-to-ASCII pass pending.

A separate `_robot.usd` exists in the `configuration/` folder for each variant.

## 4. Asset topology — MJCF vs URDF/USD

The MJCF (`HU_D04_01.xml`) defines `left_A_achilles_link`, `left_A_achilles_rod_link`, `left_B_achilles_link`, `left_B_achilles_rod_link` and the right-side equivalents — the **physical Achilles parallelogram** that drives the ankle.

The URDF for the same robot has these collapsed; it presents a serial kinematic chain.

So the same robot is described two different ways inside this repo:

| File | Kinematic view | Used by |
|---|---|---|
| MJCF (`xml/`) | parallel (Achilles rods present) | MuJoCo physics |
| URDF (`urdf/`) | serial-equivalent | most ROS / planner / SDK tooling |
| USD (`usd/`) | not yet verified — assumed to match URDF | Isaac Sim / IsaacLab |

Joint count, joint names, and DOF order can differ between MJCF and URDF as a result.

## 5. `simulator.py` — what it does

237 LOC. Single class `SimulatorMujoco` plus an `if __name__ == '__main__':` driver.

Driver:

1. Reads `ROBOT_TYPE` env var (e.g. `HU_D04_01`); exits if missing.
2. Decides `floating_base = robot_type.startswith(('DA_', 'UB_'))`. For HU_D04 this is `False`.
3. Picks robot IP (default `127.0.0.1`, override via CLI arg). Sets `MROS_IP_LIST` from the first three octets.
4. Constructs `Robot(RobotType.Humanoid, True)` and `robot.init(robot_ip)`.
5. Loads MJCF at `humanoid-description/<main_type>_description/xml/<robot_type>.xml`.
6. Starts `kinematic_projection` as a subprocess (§ 6).
7. Instantiates `SimulatorMujoco` and calls `.run()`.

`SimulatorMujoco`:

- Loads MJCF with `mujoco.MjModel.from_xml_path`, builds `MjData`, opens a passive viewer.
- Counts actuators (`nu`) and joints (`njnt`), records joint names.
- Allocates `RobotCmd` (mode, q, dq, tau, Kp, Kd) and `RobotState` (q, dq, tau, motor_names) and `ImuData`.
- Subscribes to incoming `RobotCmd` via `robot.subscribeRobotCmdForSim(callback)`.
- In `run()`, per step:
  - `mujoco.mj_step(model, data)`
  - If `floating_base=False`: reads IMU from `sensordata[0..9]` (4 quat + 3 gyro + 3 acc), publishes via `robot.publishImuDataForSim(imu)`. Joint sensor indices then start at offset 10.
  - If `floating_base=True`: joint sensor indices start at offset 0; no IMU publish.
  - For each actuator: reads `q[i]`, `dq[i]` from `sensordata`, reads `tau[i] = data.ctrl[i]`, then writes:
    ```
    ctrl[i] = Kp[i]*(q_d[i] - q[i]) + Kd[i]*(dq_d[i] - dq[i]) + tau_ff[i]
    ```
  - Publishes `RobotState` via `robot.publishRobotStateForSim(state)`.
  - Viewer sync every 20 steps.
- Loop rate = `1 / model.opt.timestep`. HU_D04's MJCF declares `<option timestep="0.001" />` → 1000 Hz.

`RobotCmd.mode` (per-joint 0/1/2 flag) is present in the packet but `simulator.py` ignores it — the actuator law is always PD-with-feedforward.

## 6. `kinematic_projection` — what it does

A 9.4 MB ELF binary (Linux x86_64). Spawned as a subprocess by `simulator.py` (`run_kinematic_projection`) with two env vars:

- `MROS_ETC_PATH = <repo>/prebuild/etc`
- `MROS_LOG_LEVEL = "0"`

It uses MROS (LimX's middleware) as its IO surface and reads its per-robot config from `prebuild/etc/kinematic_projection/<ROBOT_TYPE>/`:

| File | Apparent role (from filename) |
|---|---|
| `config.yaml` | top-level config |
| `twisted_left_ankle_model.yaml` | left Achilles linkage model |
| `twisted_right_ankle_model.yaml` | right Achilles linkage model |
| `waist_model.yaml` | waist mechanism model |

Source is not shipped. From context (the configs describe the parallel mechanisms, and the SDK exposes a serial joint view while MuJoCo simulates parallel links), it functions as a coordinate translator between serial joint commands/state and the parallel-mechanism joint commands/state that MuJoCo physics actually uses.

Robots with config entries under `prebuild/etc/kinematic_projection/`: `DA_D03_01..03`, `HU_D03_01..03`, `HU_D04_01`, `UB_D03_01..02`.

## 7. `limxsdk-lowlevel`

C++ source surface (`include/limxsdk/`):

- `apibase.h`, `datatypes.h`, `macros.h`
- per-robot-family headers: `humanoid.h`, `pointfoot.h`, `tron2.h`, `wheellegged.h`
- `ability/` subsystem: `ability_manager.h`, `base_ability.h`, `plugin_loader.h`, `plugin_registry.h`, `rate.h`, `robot_data.h`, `yaml_config_parser.h`

Python wheel surface (`limxsdk-4.0.1-py3-none-any.whl`, contents):

```
limxsdk/
├── ability/         ability_manager, base_ability, cli, config, registry, utils
├── datatypes/       DiagnosticValue, ImuData, LightEffect, RobotCmd, RobotState,
│                    SensorJoy, TerrainData
├── robot/           Joystick, Rate, Robot, RobotType, _robot.so, libpython3.8.so.1.0
└── (dist-info)
```

Wheel platform variants under `python3/`: `amd64/`, `aarch64/`, `win/`. The Linux wheels embed `libpython3.8.so.1.0` — Python 3.8 ABI.

Examples shipped under `python3/examples/`: `ability/`, `api/`.

## 8. `robot-joystick`

Two executables (`robot-joystick`, `robot-joystick.exe`) plus a `doc/` folder. Used for joystick pairing / host-side handling. Not Python; not imported elsewhere in this repo.

## 9. Top-level support files

| File | Content |
|---|---|
| `README.md` / `README_cn.md` | Setup walkthrough: install wheel, set `ROBOT_TYPE`, run `simulator.py`. |
| `LICENSE` | LimX repo license. |
| `.gitmodules` | Three submodules listed above. |
| `doc/simulator.gif` | Demo recording. |

## 10. Local vendor patches

Submodules are pinned but we keep small local patches applied on top. Re-apply if the submodule is ever reset.

### `humanoid-description` @ `63eaa67`

- **`HU_D04_description/xml/HU_D04_01.xml` lines 328, 383** — `quat` attributes on `left_hand_manip` / `right_hand_manip` bodies use comma separators (`quat="0.707107, 0, 0.707107, 0"`). MuJoCo's XML parser splits `quat` on whitespace only, so commas fail to load with `ValueError: XML Error: bad format in attribute 'quat'`. Fix: replace commas with spaces → `quat="0.707107 0 0.707107 0"`. Applied 2026-06-19. Both bodies are zero-mass IK/grasp marker spheres.

### `humanoid-mujoco-sim/simulator.py` (parent repo, not a submodule)

- **`SimulatorMujoco.__init__`, `robotCmdCallback`, `run()`** — gate physics integration on the first `RobotCmd`. Vendor `run()` calls `mj_step` unconditionally from frame 0; with our launcher's bring-up order (sim → deploy → autostart stand) there's a ~1 s window before any cmd publishes, during which `Kp = Kd = 0` and Oli free-falls. Patch adds a `self._cmd_received` flag (False initially, flipped True in the cmd callback). In `run()`, both the `mj_step` call and the per-actuator `ctrl[i] = …` assignment are skipped while `_cmd_received` is False. A single `mj_forward` is added at loop start so the first `RobotState` publish reflects the real MJCF rest pose (not stale zeros) — important because `StandController.on_start()` seeds `init_joint_angles` from that first state. Applied 2026-06-19.
