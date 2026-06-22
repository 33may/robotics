# humanoid-rl-deploy-python — vendor index

> Vendor repo: `humanoid/vendor/humanoid-rl-deploy-python/` — cloned 2026-06-19 from `git@github.com:limxdynamics/humanoid-rl-deploy-python`. No submodules. ~36 MB on disk (mostly ONNX weights).

## 1. What it is

LimX's official deployment-side stack for trained RL policies on their humanoid line. Pairs with [[humanoid-mujoco-sim]] — the sim publishes `RobotState`/`ImuData` and subscribes to `RobotCmd`; this repo publishes `RobotCmd` driven by ONNX policies. Same exact ROS-less wire protocol works for the MuJoCo sim and the real robot — only `robot_ip` changes.

What ships:

- a 73-LOC `main.py` that's only a **joystick → mode switch dispatcher** — controllers themselves are loaded by the SDK's ability framework as a subprocess
- four ability classes per robot model (`damping`, `stand`, `mimic`, `walk`) — see § 4
- ONNX policy weights + per-controller YAML configs (kp/kd, action_scale, normalization, sizes)
- one technical doc (parallel joint PR/AB mapping)

What is **not** in this repo:

- the ability framework itself (lives in `limxsdk.ability.*` — see [[humanoid-mujoco-sim#7]])
- training code or sim env — only deployment-side inference
- a walk policy for HU_D03_03 (only HU_D04_01)

## 2. Layout map

```
humanoid-rl-deploy-python/
├── main.py                       # joystick → ability switcher, § 3
├── README.md / README_cn.md
├── LICENSE
├── doc/
│   ├── parallel_joint_mapping_en.md   # PR ↔ AB joint space, § 6
│   ├── parallel_joint_mapping_cn.md
│   ├── humanoid_hw.png, ip.png, robot-joystick.png
│   └── simulator.gif
└── controllers/
    ├── HU_D03_03/
    │   ├── controllers.yaml      # ability registry, § 4
    │   ├── damping_controller/   damping_controller.py + joint_params.yaml
    │   ├── stand_controller/     stand_controller.py + joint_params.yaml
    │   └── mimic_controller/     mimic_controller.py + mimic_param.yaml
    │                             + policy/default/{policy,lin_encoder,priv_encoder}.onnx
    └── HU_D04_01/
        ├── controllers.yaml
        ├── damping_controller/   (as above)
        ├── stand_controller/     (as above)
        ├── mimic_controller/     (as above, with ONNX weights)
        └── walk_controller/      walk_controller.py + walk_param.yaml
                                  + policy/default/policy.onnx        ← user-replaceable
```

Note: `__pycache__/` directories already committed contain `cpython-38.pyc` files — concrete confirmation that the SDK + controllers run on **Python 3.8** (matches `libpython3.8.so.1.0` embedded in the SDK wheel).

## 3. `main.py` — what it does

73 LOC. No control loop, no policy code. Pure dispatcher:

1. Reads `ROBOT_TYPE` env var; exits if missing.
2. `Robot(RobotType.Humanoid)` + `robot.init(robot_ip)` where `robot_ip` defaults to `127.0.0.1`, overridable via `sys.argv[1]`.
3. Exports `ROBOT_IP` env var for downstream controllers.
4. `robot.subscribeSensorJoy(callback)` — joystick events fire mode switches via `os.system("python3 -m limxsdk.ability.cli switch '<from>' '<to>'")`.
5. `os.system("python3 -m limxsdk.ability.cli load --config controllers/<ROBOT_TYPE>/controllers.yaml")` — blocking, runs the ability framework in a subprocess.

Joystick button map (axes/buttons indices match SDL2 / LimX virtual joystick):

| Combo | Buttons | Action |
|---|---|---|
| L1 + Y | 4 + 3 | switch → `stand` |
| L1 + B | 4 + 1 | switch → `mimic` |
| R1 + X | 7 + 2 | switch → `walk` |
| L1 + A | 4 + 0 | switch → `damping` |
| L1 + X | 4 + 2 | exit (switch all → '') |

Notes:

- The two-process design (`main.py` for joystick + a separate ability subprocess for the control loop) means `Ctrl+C` on `main.py` does **not** stop the controller — you have to also kill the ability subprocess, or send the exit combo.
- Mode switching via `os.system` shells out on every button event — there's no internal IPC. Adequate but coarse.

## 4. Controllers — common shape

All four controllers inherit `limxsdk.ability.base_ability.BaseAbility` and register via the `@register_ability("<type>/controller")` decorator. The ability framework drives this lifecycle:

```
initialize(config) → on_start() → on_main()  [blocks on Rate.sleep loop] → on_stop()
```

Inside `on_main`, every controller follows the same wire pattern:

- `self.get_robot_state()` → 31-DOF `RobotState` (`q`, `dq`, `motor_names`, …)
- `self.get_imu_data()` → `ImuData` (`quat` as **(w,x,y,z)** from SDK; controllers rotate to `(x,y,z,w)` for scipy)
- compute target `q_d`, `Kp`, `Kd` (and optionally `dq`, `tau_ff`)
- `self.robot.publishRobotCmd(self.robot_cmd)` at the controller's `update_rate` (1000 Hz for all four)

`update_rate` is set in `controllers.yaml::abilities.<name>.config.update_rate` — all four set to **1000 Hz** for HU_D04_01. The sim publishes IMU/state at the MJCF timestep (1000 Hz for HU_D04 — matches).

Decimation: the two ONNX-driven controllers (`walk`, `mimic`) run **policy inference every 10 control ticks** (`decimation: 10`), so policy at 100 Hz, PD interpolation at 1000 Hz.

### 4.1 `controllers.yaml`

Schema (per `HU_D04_01/controllers.yaml`):

```yaml
robot_ip: "127.0.0.1"          # ROBOT_IP env overrides
robot_type: "Humanoid"
abilities:
  stand:
    type: "stand/controller"   # must match @register_ability arg
    script_path: "stand_controller/stand_controller.py"
    autostart: false           # framework will not run until 'cli switch' targets it
    config: { update_rate: 1000 }
  mimic:   { type: "mimic/controller",   script_path: "mimic_controller/mimic_controller.py",     config: { update_rate: 1000 } }
  walk:    { type: "walk/controller",    script_path: "walk_controller/walk_controller.py",       config: { update_rate: 1000 } }
  damping: { type: "damping/controller", script_path: "damping_controller/damping_controller.py", config: { update_rate: 1000 } }
```

Same file for HU_D03_03 minus the `walk` entry (no policy shipped).

### 4.2 Controller summary

| Controller | DOF | Policy(s) | Action computation | Reset behavior |
|---|---|---|---|---|
| `damping` | 31 | none | `q_d = 0`, `Kp = 0`, `Kd ≈ 2–10` | publishes constant zero target — joints go limp with damping only |
| `stand` | 31 | none | linear 2000-step interpolation `q_init → stand_pos` (≈2 s at 1 kHz) | reads current `q` as `init_joint_angles` on `on_start`; stiff `Kp` 10–800 |
| `mimic` | 31 | `policy.onnx` (510+3+3+32 → 31) + `lin_encoder.onnx` (510→3) + `priv_encoder.onnx` (510→32) | encoders + policy on flattened proprio history; gated by `motion_phase = motion_iter/motion_frames` (764 frames for HU_D04) | `motion_iter` resets to 0 each `on_start` |
| `walk` | 31 | `policy.onnx` (~510 → 31) | proprio history → policy → clamp by per-joint torque-limit + action_scale; joystick `axes[1,0,3]` → `commands = (vx, vy, ωz)` clipped to `±0.5` | history buffer rebuilt on first observation |

### 4.3 PD law applied by every controller

`walk_controller.on_main` (representative):

```python
pos_des = action[i] * action_scale[i] + default_angle[i]
robot_cmd.q[i]  = pos_des
robot_cmd.dq[i] = 0
robot_cmd.tau[i] = 0
robot_cmd.Kp[i] = kp[i]
robot_cmd.Kd[i] = kd[i]
```

So the **deployment side sets the PD gains and target**, and the simulator (or motor firmware on real robot) closes the loop with:

```
τ_motor = Kp (q_d − q) + Kd (dq_d − dq) + τ_ff   # exactly the law in simulator.py § 5
```

### 4.4 Observation vector — walk

Built per tick (every `decimation`) by `compute_observation`:

```
base_ang_vel * ang_vel(0.25)           [3]    # IMU gyro
projected_gravity                       [3]    # gravity rotated into body frame via IMU quat
commands                                [3]    # vx, vy, ωz from joystick
(q - default_angle) * dof_pos(1.0)     [31]
dq * dof_vel(0.05)                     [31]
last_actions                            [31]
                                       ────
                                       102    # observations_size
```

Then flattened into a 5-step rolling history → **510-d policy input**. Gait observations (`gait_freq`, `gait_offset`, height, sin/cos of phase) are computed by `compute_gait_observation` **but commented out** of the obs vector (look at line ~193). The policy currently runs purely on proprio history without gait conditioning.

### 4.5 Observation vector — mimic

Same shape minus `commands`, plus a 1-d `motion_phase`:

```
base_ang_vel * 0.25          [3]
projected_gravity             [3]
q * 1.0                      [31]
dq * 0.05                    [31]
last_actions                 [31]
motion_phase                  [1]      # clamped to [0, 1]
                             ────
                            100        # mimic_param.yaml/observations_size
```

5-step history → 500-d. Mimic policy concatenates: `[obs, command_filtered(3), lin_enc_out(3), priv_enc_out(32)]` — but **`command_filtered` is hardcoded to zeros**, so command channels are dead.

## 5. Joint order (31 DOF, PR space)

Defined implicitly by every `walk_param.yaml` / `stand/joint_params.yaml` array order. Confirmed by `doc/parallel_joint_mapping_en.md` § "Deployment Parameter Configuration":

```
 0-  5 : Left leg   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
 6- 11 : Right leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
12- 14 : Waist      (waist_yaw, waist_roll, waist_pitch)
15- 16 : Head       (head_yaw, head_pitch)        ← reversed from vendor doc; matches probe (§ 11)
17- 23 : Left arm   (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                     wrist_yaw, wrist_pitch, wrist_roll)
24- 30 : Right arm  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                     wrist_yaw, wrist_pitch, wrist_roll)
```

This is the **PR-space** order — what the policy outputs, what the SDK accepts via `publishRobotCmd` in default PR mode. The MJCF's actuator order is **AB-space** (different names: `A_achilles_joint`, `waist_A_joint`, …) — SDK converts internally. See § 6.

## 6. PR ↔ AB joint space (parallel mechanism)

LimX humanoids use parallel four-bar linkages for ankle (Achilles) and waist. Two coexisting naming conventions:

- **PR (Pitch-Roll) space** — semantic joint names (`left_ankle_pitch_joint`, `waist_roll_joint`). Used by RL policies + this deploy code.
- **AB (A-B Motor) space** — physical motor names (`left_A_achilles_joint`, `waist_A_joint`). Used by MJCF actuators + real hardware.

SDK does PR→AB conversion automatically when a per-joint flag is enabled (default). For AB mode the user must do parallel kinematics themselves. Full table:

| PR (policy) | AB (motors) | MJCF actuator |
|---|---|---|
| `left_ankle_pitch_joint` | `left_A_achilles_joint` | `ankle_A_left` |
| `left_ankle_roll_joint` | `left_B_achilles_joint` | `ankle_B_left` |
| `right_ankle_pitch_joint` | `right_A_achilles_joint` | `ankle_A_right` |
| `right_ankle_roll_joint` | `right_B_achilles_joint` | `ankle_B_right` |
| `waist_roll_joint` | `waist_A_joint` | `body_waist_A` |
| `waist_pitch_joint` | `waist_B_joint` | `body_waist_B` |

`waist_yaw_joint` is a standalone single-motor joint — not parallel.

The MuJoCo side encodes the parallelogram with `<equality><connect>` constraints between the rod endpoints (see `doc/parallel_joint_mapping_en.md` § "MJCF XML Structure"). The `kinematic_projection` ELF binary in [[humanoid-mujoco-sim#6]] is presumably what does the PR↔AB translation on the sim side — both directions need it (state from MJCF AB → PR for policy obs, cmd from PR → AB for actuators).

## 7. Per-controller gain reference (HU_D04_01)

Quick eyeball comparison — stand uses much stiffer gains than walk, as expected:

|  | Hip pitch | Knee | Ankle pitch | Waist yaw | Shoulder pitch |
|---|---|---|---|---|---|
| `stand_kp` | 580 | 660 | 400 | 800 | 80 |
| `walk_kp`  | 139.41 | 139.41 | 93.65 | 93.65 | 87.51 |
| `damping_kp` | 0 | 0 | 0 | 0 | 0 |
| `damping_kd` | 10 | 10 | 4 | 6 | 10 |

`stand` reads current `q` on entry and ramps to `stand_pos` over 2 s — so a hard transition from a flopped pose to standing is intentional. If the robot has fallen over, the stiff `Kp` will yank it; README warns to hit MuJoCo "Reset" first.

## 8. Dependencies (Python wheels needed by this repo)

Reading imports across `main.py` + controllers:

| Package | Used by | Notes |
|---|---|---|
| `limxsdk` | all | 4.0.1 wheel from [[humanoid-mujoco-sim]] — Py 3.8 ABI locks everything else |
| `onnxruntime` | walk, mimic | CPU only (`CPUExecutionProvider` hard-coded, intra/inter op threads = 1) |
| `numpy` | walk, mimic | – |
| `scipy.spatial.transform.Rotation` | walk, mimic | quaternion ↔ Euler |
| `pyyaml` | all | YAML config loading |
| `mujoco` | (sim only, not here) | required for the companion repo's `simulator.py` |

No torch, no CUDA, no ROS. ONNX inference is single-threaded CPU. Whole control loop is feasible on any modest x86_64 box.

## 9. Open questions / things to verify when running

- **MROS bring-up:** does `simulator.py` + `main.py` work over loopback (`127.0.0.1`) without anything else, or does MROS need a separate broker? The SDK README claims "ROS-less", but `kinematic_projection` reads `MROS_ETC_PATH` and the SDK seems to use a pub/sub bus. Verify by running both and watching `lsof`/`ss` ports.
- **Joystick subscription with no joystick:** what does `robot.subscribeSensorJoy` do if no `SensorJoy` is ever published? `main.py` will block forever waiting for the ability subprocess; we may need a tiny stub that publishes a synthetic `SensorJoy` to drive the mode switches.
- **`cli switch` semantics:** does it kill+restart the ability subprocess, or hot-swap within the same process? Affects how cleanly `walk → damping` lands.
- **HU_D03_03 walk:** confirmed missing; if we ever want HU_D03 walking, we'd need to train + drop our own `policy.onnx` into `controllers/HU_D04_01/walk_controller/policy/default/`. README explicitly invites this.
- **PR↔AB flag location:** doc says it's a per-joint flag set on `publishRobotCmd`, but the controllers never touch it explicitly — they only set `q`, `Kp`, `Kd`. Either it's a global SDK default, or `RobotCmd.mode[i]` (which controllers set to 0) is that flag. Worth tracing into `_robot.so`.

## 10. Local vendor patches

Repo is a plain clone (no submodule), so edits live in the working tree until we either upstream or fork. Re-apply if the repo is ever re-cloned.

### `controllers/HU_D04_01/controllers.yaml`, `controllers/HU_D03_03/controllers.yaml`

- **`abilities.stand.autostart`: `false` → `true`** (applied 2026-06-19). Default ships `autostart: false`, which means after `cli load` no ability runs until you fire `cli switch '' stand`. In our launcher (`humanoid/logic/simulation/mujoco/simulator.py`) the switch round-trip plus deploy-startup latency leaves a ~2 s window where no `RobotCmd` is published and the MuJoCo PD law sees `Kp = Kd = 0`, so Oli free-falls before standing. Setting `autostart: true` makes the deploy framework start the stand controller the instant it connects to the bus — Oli stands within tens of ms of sim being up. The joystick still works for switching between modes after bring-up.

## 11. Wire contract observed (2026-06-19, HU_D04_01, sim loopback)

Captured via `humanoid/logic/simulation/mujoco/probe_contract.py` while the launcher was running with the stand ability active. All figures from `127.0.0.1` MROS loopback against the patched `simulator.py` (mj_step gated on first cmd, 1 ms MJCF timestep).

### `RobotState` (sim → policy)

| Field | Shape | Notes |
|---|---|---|
| `q` | 31 | joint positions, radians |
| `dq` | 31 | joint velocities, rad/s |
| `tau` | 31 | joint torques (sim reads from `data.ctrl[i]`) |
| `motor_names` | 31 strings | **PR space** — see table below |
| `stamp` | int (ns) | `time.time_ns()` from sim |

Observed publish rate: **~885 Hz** (MJCF nominal 1000 Hz; ~11.5 % real-time deficit caused by viewer + Python loop overhead on desktop. Real robot expected to hit nominal). Same packet over `subscribeRobotState` and `subscribeRobotStateForSim` — confirmed.

`motor_names` ordering (confirms § 5 of this doc):

```
 0  left_hip_pitch_joint        16  head_pitch_joint
 1  left_hip_roll_joint         17  left_shoulder_pitch_joint
 2  left_hip_yaw_joint          18  left_shoulder_roll_joint
 3  left_knee_joint             19  left_shoulder_yaw_joint
 4  left_ankle_pitch_joint      20  left_elbow_joint
 5  left_ankle_roll_joint       21  left_wrist_yaw_joint
 6  right_hip_pitch_joint       22  left_wrist_pitch_joint
 7  right_hip_roll_joint        23  left_wrist_roll_joint
 8  right_hip_yaw_joint         24  right_shoulder_pitch_joint
 9  right_knee_joint            25  right_shoulder_roll_joint
10  right_ankle_pitch_joint     26  right_shoulder_yaw_joint
11  right_ankle_roll_joint      27  right_elbow_joint
12  waist_yaw_joint             28  right_wrist_yaw_joint
13  waist_roll_joint            29  right_wrist_pitch_joint
14  waist_pitch_joint           30  right_wrist_roll_joint
15  head_yaw_joint
```

Note: index 15/16 are `head_yaw` then `head_pitch` — **reversed from § 5's "head_pitch, head_yaw"**. § 5 was based on doc text; this probe is empirical. Trust the probe.

### `ImuData` (sim → policy)

| Field | Shape | Notes |
|---|---|---|
| `quat` | 4 floats | **(w, x, y, z)** convention — sample: `[0.998, -0.0009, -0.012, 0.060]` (Oli standing, slight forward lean) |
| `gyro` | 3 floats | rad/s, body frame |
| `acc` | 3 floats | m/s², body frame — gravity dominant: `[~0, ~0, ~9.4]` |
| `stamp` | int (ns) | `time.time_ns()` |

Observed publish rate: **~885 Hz** (same as RobotState — published in the same `simulator.py` loop iteration).

Quaternion convention: SDK returns `(w, x, y, z)`. Controllers (`walk`, `mimic`) then rotate to `(x, y, z, w)` for scipy via `quat = quat[1:] + quat[:1]`. Empirically the standing IMU sample matches identity-ish quaternion in (w,x,y,z) form (w ≈ 1.0).

### `RobotCmd` (policy → sim)

Pending re-run with `subscribeRobotCmdForSim` (initial probe used `subscribeRobotCmd` — policy-side subscription, sees 0 samples by design). Expected shape per `walk_controller.py`:

| Field | Shape | Notes |
|---|---|---|
| `q` | 31 floats | target joint positions, PR space, radians |
| `dq` | 31 floats | typically all-zero from controllers |
| `tau` | 31 floats | feed-forward torques, typically all-zero |
| `Kp`, `Kd` | 31 floats each | per-joint gains; non-zero only for active ability |
| `mode` | 31 ints | per-joint flag for PR↔AB conversion; **defaults to 0** in all shipped controllers |
| `motor_names` | 31 strings | echoed back from `RobotState` by controllers |

### Mode flag — empirical answer

Walk + stand + mimic all run successfully with `RobotCmd.mode[i] = 0` for every joint. Combined with the fact that controllers operate in PR space (semantic joint names) and the SDK's auto-conversion routes targets to AB-space MJCF actuators correctly, **`mode = 0` = PR mode (SDK converts)**. This contradicts no explicit doc, but was previously a hypothesis — now confirmed.

### `kinematic_projection`

Spawned as a subprocess by `simulator.py`. Without it, ankle/waist parallel mechanism conversion fails. Required for sim; presumably required on real robot too (where it lives inside the robot firmware rather than as a separate ELF).
