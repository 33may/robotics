# LimX ros2-bridger — viability as MAY-147 architecture (2026-06-22)

## TL;DR

**No — ros2-bridger does not replace the custom Py 3.8 sidecar for MAY-147.** Three independent blockers:

1. The bridge only registers converters for a small per-joint subset of MROS topics (`IMUData`, `JointState`, `JointCmd`, `JointCmdLimx`, `JointCmdNew`) — it does **not** ship factories for the `RobotState*` / `RobotCmd*` / `RobotCmdForSim` / `SensorJoy` / `DiagnosticValue` types that `limxsdk` actually pubs and subs to as the sim peer. The MROS-side type-name strings appear in the binary but no `controller_msgs__msg__RobotState*__factories.cpp` is compiled in, so those topics will never bridge.
2. The bridge has **no `is_sim` role flag** anywhere in its config. limxsdk role-gating (sim peer vs policy peer) lives inside the SDK at `Robot()` construction time — bridging at the topic layer flattens that asymmetry and can't represent "I am the sim peer." Best case our Isaac node would join the bus as a generic third peer with no role membership.
3. Isaac Sim 5.0 needs ROS2 built **against Python 3.11**, but ros2-bridger ships binary `.so`s linked against the **stock Humble/Jazzy/Foxy** Python (3.10 / 3.12 / 3.8). Mixing Isaac's Py 3.11 rclpy with the bridge's stock-distro rclcpp side via DDS works in principle, but it forces us to install full ROS2 Humble system-wide AND rebuild a Py-3.11 workspace for Isaac's side — meaningfully more setup than the current sidecar.

The custom sidecar plan stays the right call. ros2-bridger is targeted at a different use case (running LimX's own ROS2 controller stack against the real robot or MuJoCo).

## ros2-bridger — what it actually is

- **Repo**: https://github.com/limxdynamics/ros2-bridger, single squashed commit `f8b8e45 1.0.0.20260531175317`, last push 2026-05-31.
- **Distribution model**: pre-built binary install trees committed via Git LFS, **not** source. 401 MB LFS payload. No `CMakeLists.txt`, no source files, no message `.msg` definitions outside the compiled `install/<pkg>/share/*/msg/` artifacts. The bridge's source URL `/home/limx/workspace/mros/src/3rd/mrosdds/...` is leaked through compiled-in strings but the source itself is private.
- **Targets**: 6 install trees — `{amd64, aarch64} × {foxy, humble, jazzy}`. Each is a ready-to-source colcon workspace (`install/setup.bash` style).
- **Workflow**: `source /opt/ros/<distro>/setup.bash && source <arch>/<distro>/install/setup.bash && ros2 launch mrosbridger mrosbridger.launch.py`. Single ELF executable `mrosbridger/bin/mrosbridger` (73 MB), plus `controller_msgs`, `hand_msgs`, `std_msgs`, `tron2_manipulation`, `upper_body` message packages.
- **Architecture**: it's a port of the canonical ROS1↔ROS2 `dynamic_bridge.cpp` pattern. Topics are discovered at runtime via DDS introspection, then bridged where both a MROS publisher and a ROS subscriber (or vice-versa) exist for a known type-name pair. The classic `created mros-to-ros bridge for topic '%s' with MROS type '%s' and ROS type '%s'` log line is in the binary.
- **Launch args**: `bridge_mros2ros` (default `true`), `bridge_ros2mros` (default **`false`**), plus include/exclude topic glob params. Default config bridges MROS→ROS only — you must explicitly enable the reverse direction to publish into MROS from ROS.
- **MROS transport**: bundled `mrosdds` (FastDDS fork by LimX). The `MROS_IP_LIST` env var picks which peer(s) to discover, same as in `limxsdk`.

### MROS ↔ ROS2 topic surface — converter coverage

Verified by extracting `_GLOBAL__sub_I__ZN11mrosbridger*_factories.cpp` symbol names and the `<pkg>__msg__<Type>__factories.cpp` strings from the binary. Only types with a compiled-in factory get auto-bridged.

| MROS-side type LimX uses | Where it shows up | Factory compiled in? | Notes |
|---|---|---|---|
| `controller_msgs/IMUData` | `subscribeImuData` / `publishImuData` analogue | **yes** | Bridges to `controller_msgs/msg/IMUData` (custom, not `sensor_msgs/Imu`). |
| `controller_msgs/JointState` | per-joint state | **yes** | Custom `controller_msgs/msg/JointState` with `q,v,vd,tau,names`. |
| `controller_msgs/JointCmd` | per-joint cmd | **yes** | `q,v,tau,kp,kd,mode,names`. |
| `controller_msgs/JointCmdLimx` | per-joint cmd (Limx variant w/ parallel solver) | **yes** | adds `parallel_solver_mode`. |
| `controller_msgs/JointCmdNew` | per-joint cmd (newer variant) | **yes** | adds `parallel_solve_required` bool array. |
| `controller_msgs/RobotState*` (PointFoot/Wheel/Biped/Humanoid/UpperBody/Teleoperation/Vis/WheelSingleTest) | what `subscribeRobotState` delivers in limxsdk | **NO** | Type-name strings present in the binary but **no `RobotState*__factories.cpp`** is compiled in. Will not bridge. |
| `controller_msgs/RobotCmd*` (PointFoot/Wheel/Biped/Humanoid/UpperBody/Teleoperation) | what `publishRobotCmd` sends in limxsdk | **NO** | Same — type names appear (likely via included headers) but no converter is registered. Will not bridge. |
| `RobotCmdForSim` (sim-peer-only cmd flow) | `subscribeRobotCmdForSim` | **NO** | Type name does not appear in the binary at all. |
| `SensorJoy`, `DiagnosticValue`, `LightEffect`, `TerrainData` | gamepad / diag / lights / terrain | **partial** | Only `diagnostic_msgs/DiagnosticValue` (LimX's, not std) has a factory; the others (`SensorJoy`, `LightEffect`, `TerrainData`) do not. |
| `hand_msgs/HandCmd`, `HandState`, `HandMsg` | hand control | yes | Not relevant to humanoid locomotion sim peer. |
| `tron2_manipulation/arm_pose`, `arm_status` | manipulation | yes | Not relevant. |
| `upper_body/arm_servo`, `arm_status`, `servoJ`, `waist_cmd` | upper body | yes | Not relevant. |
| `sensor_msgs/Imu`, `nav_msgs/Odometry`, `tf2_msgs/TFMessage`, all std_msgs, etc. | standard ROS | yes (full) | Generic ROS pipes, not LimX-specific. |

**The relevant gap**: `RobotState`/`RobotCmd`/`RobotCmdForSim` are the high-level aggregate messages that `limxsdk::Humanoid::publishRobotState`, `publishRobotCmd`, `subscribeRobotCmdForSim` work with. These are exactly what `humanoid-mujoco-sim` and `humanoid-rl-deploy-python` use to talk. ros2-bridger does **not** expose them.

Inference about LimX's intent: the bridge is built to wire MROS into a **ros2_control** style stack (per-joint state/cmd in standard formats), not to replicate the limxsdk-level aggregate topics. So even the topics that *do* bridge wouldn't slot cleanly into the sim-peer contract; we'd be bridging joint-level cmd, not the `RobotCmd` aggregate the deploy-python controllers send.

### Role-gating handling

ros2-bridger has **no concept of `is_sim`**. The role split is enforced inside `limxsdk` at `Robot()` construction (see `limx-sdk-role-gating.md`) — it's not a property of MROS topics, it's a property of the peer participant. A bridge node necessarily creates its own MROS DDS participant when it joins the bus, and it has no LimX API surface to declare itself a sim peer. Likely consequence: the bridge participant is just another generic node and the sim/policy partition simply doesn't apply to it.

For us this means even if the bridge *did* expose `RobotCmd`/`RobotCmdForSim`, we couldn't use it to take the place of the sim peer — limxsdk would still need a sim-role participant somewhere on the bus to be the canonical state publisher.

## ROS2 + Isaac Sim 5.0 + Py 3.11 compatibility

Isaac Sim 5.0/5.1 is **Python 3.11 only**. ROS2 distros and their default Python:

| Distro | Ubuntu | Default Py | Matches Isaac 5.0? |
|---|---|---|---|
| Foxy | 20.04 | 3.8 | No |
| Humble | 22.04 | 3.10 | No |
| Jazzy | 24.04 | 3.12 | No |

NVIDIA's docs are explicit: to use `rclpy` and any custom messages from inside Isaac Sim, the entire ROS2 workspace must be **rebuilt from source against Python 3.11**. They ship Dockerfiles in `IsaacSim-ros_workspaces` that do exactly this for Humble and Jazzy. The error mode otherwise is `No module named rclpy._rclpy_pybind11` — the compiled `.so` lives under `.../python3.10/site-packages/rclpy/` but Isaac is looking under `.../python3.11/site-packages/rclpy/`.

For Isaac Sim ↔ ROS2 communication, NVIDIA's three workflows:
1. **Default ROS2 system install** (Py 3.10/3.12) + Isaac's internal Py 3.11 ROS libs — works for std/sensor/geometry/nav msgs only, because DDS handles transport regardless of Py version; custom messages fail.
2. **Internal libs only** — Isaac runs with bundled `librclcpp`/`librclpy` at Py 3.11. No system ROS, but no custom messages either.
3. **Custom packages** — rebuild ROS2 + custom packages from source against Py 3.11 in a separate workspace.

For ros2-bridger specifically: the bridge ships pre-built against stock Humble/Jazzy/Foxy Python. Its message packages (`controller_msgs`, `hand_msgs`, etc.) include `.so`s like `libcontroller_msgs__rosidl_typesupport_introspection_cpp.so` and Python type-support libs that are Py-version-locked. We would need to:

- Install full ROS2 Humble on Ubuntu 22.04 (or Jazzy on 24.04) for the bridge to run.
- Run the bridge as a system-Python ROS node (separate process — that's fine, DDS transports).
- **Rebuild `controller_msgs` against Py 3.11** in a separate workspace if we want Isaac's `rclpy` to import them. Otherwise we can only use built-in std/sensor types from inside Isaac.

That last step alone is comparable in cost to writing the sidecar — and we'd still be bridging the *wrong* messages (per-joint, not aggregate).

## How LimX themselves use it

The single LimX repo that depends on ROS2 for humanoid is `humanoid-rl-deploy-ros2` (last commit 2025-09-05 → 2026-05-22 per prior research). Inspection of its tree shows:

- It **does not depend on `mrosbridger`**. Its `CMakeLists.txt` and `package.xml` link only `limxsdk` (added as a Git submodule under `src/limxsdk-lowlevel/`).
- `robot_hw/src/HardwareBase.cpp` is the canonical pattern:
  ```cpp
  HardwareBase::HardwareBase(limxsdk::ApiBase* robot) : robot_(robot) {
    robot_->subscribeRobotState([this](const limxsdk::RobotStateConstPtr& msg) { ... });
    robot_->subscribeImuData([this](const limxsdk::ImuDataConstPtr& msg) { ... });
    robot_->publishRobotCmd(robotCmd_);
  }
  ```
  i.e. they go straight through limxsdk C++ from inside a ROS2 node, **bypassing the bridge entirely**.
- The sim launch (`humanoid_hw_sim.launch.py`) constructs `humanoid_hw_node` with `arguments=["127.0.0.1"]` — that's the `MROS_IP_LIST` style address passed to `limxsdk::Humanoid::getInstance()`. limxsdk talks to MuJoCo sim directly over MROS.

So LimX's own ROS2 deploy doesn't use ros2-bridger. The bridge appears to be an auxiliary tool for **observation / debugging** (echoing MROS into `ros2 topic echo`, recording with `ros2 bag`), and possibly for integrating LimX robots into third-party ROS2 stacks that expect ros2_control-style per-joint topics. It is not the production interop layer.

## Architecture comparison

| Dimension | Custom Py 3.8 sidecar (current MAY-147 design) | ros2-bridger alternative |
|---|---|---|
| Lines of glue we write | ~600 LOC (sidecar + protocol + driver mods + launcher) | ~600+ LOC anyway — we'd need a Py-3.11 build of `controller_msgs`, an Isaac-side `rclpy` adapter, plus glue to translate per-joint `JointCmd` ↔ aggregate `RobotCmd` ourselves |
| External runtime deps added | none (stdlib + existing `limx` Py 3.8 conda env) | full ROS2 Humble or Jazzy system install (~2 GB), the 401 MB ros2-bridger LFS payload, plus a Py-3.11 rebuilt ROS2 workspace for Isaac's side |
| Processes at runtime | sidecar (Py 3.8) + Isaac (Py 3.11) + deploy-python (Py 3.8) = 3 | ros2-bridger (system Py) + Isaac (Py 3.11) + deploy-python (Py 3.8) + **still need a sim-role limxsdk peer somewhere** because bridge can't be one = 4+ |
| Py 3.8 process eliminated? | no (sidecar owns limxsdk) | **no** (deploy-python is still Py 3.8; and we still need a Py 3.8 sim-role peer to publish RobotState because the bridge has no sim role) |
| Latency added | ~50–100 µs (UDS round-trip) | ~100–500 µs typical DDS pubsub, plus an extra hop through MROS↔DDS conversion |
| Role gating preserved (sim vs policy) | yes — we construct sidecar as `Robot(Humanoid, True)` explicitly | **no** — bridge has no `is_sim` knob; bus role asymmetry can't be expressed at the bridge layer |
| Covers full sim-peer contract (RobotState/ImuData/RobotCmdForSim) | yes | **no** — ros2-bridger lacks factories for `RobotState*` and `RobotCmd*ForSim` aggregates |
| Cross-machine deployment | hard (UDS is local) | easy (DDS is native LAN) — but this isn't a MAY-147 requirement |
| Debuggability | `nc -U` the socket, hand-rolled protocol | `ros2 topic echo` (only for the bridged subset) |
| Matches LimX precedent | yes (mirrors `humanoid-mujoco-sim` pattern) | no — no LimX repo uses the bridge as the sim path |
| Required new investigation | none — design.md already drafted | (a) reverse-engineer the MROS topic-name LimX uses for each aggregate; (b) figure out whether `controller_msgs/RobotState*` can be added to the bridge without source access; (c) Py-3.11 rebuild of `controller_msgs` |

The decisive row is **"Py 3.8 process eliminated?"** — the central question. The answer is **no** under the ros2-bridger architecture, because:

1. The bridge has no sim-role knob — somebody still has to be the sim-role limxsdk peer on the bus to publish `RobotState`/`ImuData`. That somebody has to be a Py 3.8 process (or a C++ process), the same as today.
2. Even if we made Isaac the sim-role peer via a Py 3.8 sidecar, we wouldn't need the bridge — the sidecar already provides the bus access.
3. Even if we used the bridge instead of the sidecar, the bridge doesn't expose `RobotCmdForSim`, so Isaac couldn't receive policy commands from `humanoid-rl-deploy-python` through it.

The alleged benefit (no Py 3.8) does not materialize. The downside (extra ROS2 infra, more processes, lost role gating, lost coverage of the aggregate topics) is real.

## Recommendation for MAY-147

**A. Stick with the custom sidecar.** The ros2-bridger alternative does not solve the Py 3.8 ABI problem — it just moves it sideways while losing the role-gating contract and the aggregate-topic coverage. The sidecar plan in `humanoid/openspec/changes/may-147-isaac-limx-sdk-bridge/design.md` remains the right architecture.

Where ros2-bridger may still be useful, separately:

- As a **debugging adjunct**: run the bridge alongside whatever sim setup we end up with, and use `ros2 topic echo /controller_msgs/JointState` to sanity-check per-joint signals from outside the limxsdk process. Zero impact on the sim-peer wiring.
- If we later want **third-party ROS2 stacks** (Nav2, MoveIt2, etc.) to consume sim-peer signals, the bridge gives us a way without writing a custom translator — but that's a future consideration, not MAY-147.

## Open questions

- **Are there topic-name conventions the bridge expects?** Bridging is type-driven not topic-name-driven, so `/robot_state` and `/joint_state` could mean anything. If we ever care about the bridge as a debug adjunct we should check what topic name the LimX sim actually publishes its `JointState` under.
- **Could we extend ros2-bridger to cover `RobotState*` / `RobotCmd*ForSim`?** Only if LimX open-sources or hands us the source. The bridge binary at 73 MB plus the `mrosdds` private fork makes a clean-room re-implementation expensive. Not worth chasing for MAY-147.
- **Does Isaac Sim 5.0 work with a Py-3.11-rebuilt Humble at all?** NVIDIA claims yes via the Dockerfile recipe, but I haven't verified locally. Marginal relevance to the recommendation.
- **What about a hybrid where the sidecar uses MROS only for sim-side state and ros2-bridger pipes diagnostics?** Possible, but adds complexity for marginal gain. Defer until there's a concrete reason.

## Sources

- ros2-bridger repo (cloned to `/tmp/limx-research/ros2-bridger/`): https://github.com/limxdynamics/ros2-bridger
- humanoid-rl-deploy-ros2 repo (cloned to `/tmp/limx-research/humanoid-rl-deploy-ros2/`): https://github.com/limxdynamics/humanoid-rl-deploy-ros2
- Isaac Sim 5.1 ROS2 Installation: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html
- ros2/rclpy #1194 — Py 3.10 vs 3.11 mismatch on Ubuntu 22.04: https://github.com/ros2/rclpy/issues/1194
- Prior research: `/home/may33/projects/ml_portfolio/robotics/.claude/agent-memory/humanoid/research_limx_isaac_integration.md`
- Role gating reference: `/home/may33/projects/ml_portfolio/robotics/.claude/agent-memory/humanoid/limx-sdk-role-gating.md`
