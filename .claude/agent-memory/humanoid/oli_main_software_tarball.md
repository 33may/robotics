---
name: oli-main-software-tarball
description: Contents and structure of LimX Oli Main Software v2.2.12 EDU tarball — what's indexable for the corpus and what isn't
metadata:
  type: reference
---

LimX ships the on-robot stack as a wrapper tarball: outer `robot-hu-r-2.2.12.20260508181921.tar.gz` contains an inner `.tar.gz` (1.5 GB) + md5sum. The inner archive is a **colcon ROS2 install workspace** with **148 packages** and ~7000 entries, single top-level dir `install/`.

**Where to download**: `https://limx.cn/en/products/oli/download` → "Oli Main Software (EDU Ed.)" button. JS-gated (Next.js click handler, presigned URL) — `WebFetch` cannot see the URL. Must download via browser.

**Top-level layout** (`install/...`):
- `share/<pkg>/` — 148 ROS package install dirs (`package.xml`, `launch/`, `config/`)
- `mbl/include/<lib>/` — C++ headers for control + teleop libs (gold for AI-readable API)
- `mbl/bin/`, `bin/` — ~700 ELF binaries (controllers, mros nodes, teleop)
- `mbl/lib/`, `lib/` — 120 .so shared libs
- `oli/` — `local_cosa-arm` binary + startup wav
- `python/` — `auto_run.py`, `post_analyze/`, `result_processors/`
- `docker/`, `etc/` — runtime configs
- `run_on_local.sh`, `run_on_robot.sh`, `setup.bash` — entry points

**Package families** (in `install/share/`):
- **MROS** (LimX's modular ROS clone, ~120 pkgs): `mrosagent`, `mrosnode`, `mrostopic`, `mroslaunch`, `mrosurdf`, `mrosbag`, `mrosjoy`, `mrosrobot_state_publisher`, etc. — wraps standard ROS2 stack.
- **Robot descriptions**: `HU_D04_description` (humanoid), `HU_L01/L02/N01_description`, `UB_D04_description`, `DA_D04_description`.
- **Control stack**: `robot_controllers`, `robot_kinematics`, `robot_model`, `robot_planner`, `robot_data`, `robot_state_detection`, `robot_utility`, `robot_trajectory`, `joint_analyzer`, `joint_calibration`, `monitor_safety`, `fall_detection`.
- **Teleop**: `tele_operation`, `teleop_kinematics`, `teleop_planner`, `teleop_planner2`, `teleop_extrapolator`, `mocopi_teleop`, `noitom_mocap_node`, `udcap_glove_teleop`, `glove_retarget`.
- **Perception / SLAM**: `fast_lio`, `fast_lio_localization_sc_qn`, `fast_lio_sam_sc_qn`, `eskf`, `state_estimation`, `realsense_mros`, `livox_ros_driver2`, `livox_sdk2`, `hi13_imu_driver`, `nano_gicp`, `quatro`, `teaserpp`, `octomap_msgs`.
- **Nav + behaviour**: `behaviortree_cpp`, `mission_engine`, `move_base`, `global_planner`, `teb_local_planner`, `sentry_nav`, `sentry_decision`, `waypoint_cycle_sender`, `mroscostmap2d`, `map_server`, `pcd2pgm`, `rotate_recovery`, `clear_costmap_recovery`.
- **Streaming**: `camera_publisher` (WebRTC), `livekit_rust_stream`, `quest_server`, `signaling`, `rtsp_cam_node`, `usb_cam_node`, `mroswebvideo`, `mroswebsockets`.
- **Their own MCP server**: `oli_mcp_server` (LimX ships a Model-Context-Protocol server on-robot — separate from our `oli-corpus-mcp`).

**File-type breakdown** (non-dir, total ~6983):
- **Indexable**: 360 `.yaml`, 189 `.xml` (mostly ROS launch), 148 `package.xml`, 83 `.py`, 79 `.h`/`.hpp`, 44 `.urdf`, 18 `.json`, 6 `.srdf`, 8 `.launch`, 6 `.world`.
- **Boilerplate (skip)**: 1403 `.sh`, 1369 `.dsv`, 929 `.ps1`, 305 `.zsh`, 305 `.bash`, 305 `.cmake` — colcon-generated env activation hooks.
- **Opaque (skip)**: 711 binaries no-ext, 120 `.so`, 152 `.onnx`, 136 `.rknn` (Rockchip NPU weights), 215 `.STL`, 23 `.bin`, 19 `.wav`.

**Critical**: **zero `.msg` and zero `.srv` files** in the install tree. ROS message IDL is stripped — only compiled types in `.so`. This means **topic names and wire schemas are NOT recoverable from the tarball** without dlopen-ing the libs or sniffing the bus. Reinforces that [[probe-contract-py]] stays on the roadmap for wire-level discovery.

**Highest-leverage ingest targets** for [[reference_oli-corpus-mcp]]:
1. `install/mbl/include/**/*.{h,hpp}` — control + teleop C++ API (no fetch, just walk)
2. `install/share/HU_D04_description/**/*.{urdf,srdf,xml}` — joint/link tables for the Oli variant we run
3. `install/share/<pkg>/launch/*.xml` — node startup, topic remaps, params (closest thing to "what runs on the robot")
4. `install/share/<pkg>/config/*.yaml` — controller/planner/EKF tuning
5. `install/share/<pkg>/package.xml` — package dependency graph

**Vendor location** (when extracted): suggest `humanoid/vendor/oli-main-software-2.2.12/` to match existing vendor convention. Pin version in folder name so upgrades don't silently overwrite.
