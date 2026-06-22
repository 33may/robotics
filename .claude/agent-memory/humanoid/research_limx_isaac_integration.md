# LimX Dynamics — Isaac integration & Python version research (2026-06-22)

## TL;DR
`tron1-rl-isaaclab` **does NOT import `limxsdk` anywhere** — it is a thin fork of `isaac-sim/IsaacLabExtensionTemplate` that only trains a policy in Isaac Lab (Python 3.10, Isaac Sim 4.5.0) and exports it to ONNX/JIT via `play.py`. The Py 3.8 ABI lock is then "solved" downstream by `tron1-rl-deploy-python` (and the ROS / ROS2 deploy variants), which runs in **a separate Python 3.8 process** that loads the exported ONNX with `onnxruntime` and bridges to hardware through `limxsdk`. LimX has no public Python 3.10/3.11 wheel anywhere — not on PyPI, not in any repo, not in any CI workflow. The training/deploy split is a hard process boundary, not a single-Python solution.

## tron1-rl-isaaclab — full dissection

- **Repo URL**: https://github.com/limxdynamics/tron1-rl-isaaclab (fork of `isaac-sim/IsaacLabExtensionTemplate`, single squashed commit `28ae509 3.0.17.20250901122933`, last push 2025-09-01)
- **Python version**: `>=3.10` (`exts/bipedal_locomotion/setup.py:63`, `pyproject.toml:72`, classifier `Programming Language :: Python :: 3.10`)
- **Isaac Lab / Isaac Sim version**: `Isaac Sim :: 4.5.0` (setup.py classifier). Deps in `extension.toml`: `isaaclab`, `isaaclab_assets`, `isaaclab_tasks` — no version pin.
- **Uses limxsdk**: **NO.** Verified by recursive grep across the entire repo for `limxsdk`, `import limx`, `from limx`, `RobotCmd`, `RobotState`, `subscribeRobot`, `publishRobot`. **Zero hits.** The only match for the string `limx` is the module name `limx_pointfoot_env_cfg` in `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py` — a config file naming convention, not an SDK import.
- **What it actually does**: Defines RL environments (`limx_pointfoot_env_cfg`, `limx_wheelfoot_env_cfg`, `limx_solefoot_env_cfg`) for the three TRON1 variants (PF / WF / SF). Trains via `scripts/rsl_rl/train.py`. `scripts/rsl_rl/play.py` calls `export_policy_as_jit(...)` and `export_mlp_as_onnx(...)` to produce `policy.onnx` + `encoder.onnx` artifacts.
- **Bridge to deployment**: The ONNX file is the artifact boundary. Whoever owns the robot loads it in `tron1-rl-deploy-python/controllers/PointfootController.py` (or `Wheelfoot…` / `Solefoot…`), where:
  - `import onnxruntime as ort` runs the policy
  - `import limxsdk.robot.Robot as Robot` talks to hardware/MuJoCo over MROS
  - An `RL_TYPE` env var (`isaacgym` | `isaaclab`) selects joint-order conversion at the boundary (`if self.rl_type == "isaaclab": ... swap positions ...`)
- **No commits/issues discussing the version split**: only one issue across the org (`tron1-mujoco-sim#1` "macos .so error" — unrelated). Zero `gh search code` hits for `python3.10` / `python3.11` in any limxsdk-relevant repo; the only Py 3.10 hits are inside `robot-visualization` (compiled ROS2 Humble artifacts).

## All LimX public repos (table)

| Repo | Purpose | Lang | Py / runtime | Uses limxsdk? | Last commit |
|---|---|---|---|---|---|
| **tron1-rl-isaaclab** | RL training in Isaac Lab (TRON1 PF/WF/SF) | Python | Py 3.10, Isaac Sim 4.5.0 | **No** — training only, exports ONNX | 2025-09-01 (fork) |
| **tron1-rl-isaacgym** | RL training in Isaac Gym (`legged_gym` fork) | Python | Isaac Gym (Py 3.8 native) | **No** — training only, exports ONNX | 2025-07-18 (fork) |
| tron1-rl-deploy-python | Deploy ONNX policy to robot/sim | Python | **Py 3.8** (limxsdk wheel) | **Yes** — wheel 4.0.1 (Py 3.8) | 2025-09-05 |
| tron1-rl-deploy-ros | Deploy via ROS1 Noetic (Ubuntu 20.04) | C++ | ROS1 Noetic | Yes — C++ lib | 2025-09-05 |
| tron1-rl-deploy-ros2 | Deploy via ROS2 Humble (Ubuntu 22.04) | C++ | ROS2 Humble | Yes — C++ lib | 2025-09-05 |
| tron1-rl-deploy-arm | Manipulation deploy (ROS Noetic) | C++ | ROS1 Noetic | Yes — C++ lib | 2025-07-25 |
| tron1-mujoco-sim | MuJoCo sim w/ limxsdk pub/sub | Python | Py 3.8 | Yes — wheel 4.0.1 (submodule) | 2025-09-17 |
| tron1-gazebo-ros / tron1-gazebo-ros2 | Gazebo sim for TRON1 | C++ | ROS1 / ROS2 | Yes — C++ lib | 2025-09-15 |
| tron1-ss | TRON1 something (state-sync? skill-server?) | C++ | — | likely | 2025-10-13 |
| tron1-agent | TRON1 voice/LLM agent (langchain, edge_tts, whisper) | Python | inferred Py 3.10+ (transformers 4.51) | Not from imports inspected | 2025-08-13 |
| **humanoid-mujoco-sim** | MuJoCo sim for humanoid (HU_D03/D04) | Python | **Py 3.8** (README says "3.8 or higher" — misleading; wheel is hard 3.8) | Yes — wheel 4.0.1 (submodule) | 2026-03-03 |
| humanoid-rl-deploy-python | Humanoid deploy | Python | Py 3.8 | Yes — wheel | 2026-03-30 |
| humanoid-rl-deploy-cpp / -ros / -ros2 | Humanoid deploy C++/ROS variants | C++/Shell | — | Yes — C++ lib | 2026-03-03 → 2026-05-22 |
| humanoid-description | URDF/MJCF for HU_D03, HU_D04 | CMake | n/a | No | 2025-11-03 |
| **limxsdk-lowlevel** | SDK distribution (binary only) | C++ | C++ headers + Py 3.8 wheel | n/a | 2026-03-03 |
| robot-description | Pointfoot URDFs/MJCF | CMake | n/a | No | 2025-09-05 |
| robot-visualization | Pre-compiled ROS2 Humble artifacts (mrosbridger, msgs) | C | Py 3.10 (Humble bundle) | n/a | 2026-02-26 |
| robot-joystick | Joystick binary | — | n/a | No | 2025-07-14 |
| ros1-bridger / ros2-bridger | Pre-built MROS↔ROS1/ROS2 bridges (binary) | C / Common Lisp | system Python via ROS | Indirectly (bridges MROS topics) | 2026-05-31 |
| gradmotion-cli | Stub repo (LICENSE only) | — | n/a | — | 2026-03-20 |
| limxdynamics | Org profile / placeholder | — | — | — | 2025-09-10 |

## Newer limxsdk wheels?

- **PyPI presence**: NONE. `https://pypi.org/pypi/limxsdk/json` → 404 "Not Found".
- **Multi-Python wheels in the repo**: NONE. Three wheels exist in `limxsdk-lowlevel/python3/`:
  - `amd64/limxsdk-4.0.1-py3-none-any.whl` → bundles `libpython3.8.so.1.0` (3.8.10), `_robot.so` links to `libpython3.8.so.1.0`
  - `aarch64/limxsdk-4.0.1-py3-none-any.whl` → same Py 3.8 bundling, ARM build
  - `win/limxsdk-3.4.2-py3-none-any.whl` → older version, Windows
- **CI evidence**: NONE. `limxsdk-lowlevel` has no `.github/workflows/`, no `setup.py` for Python (only a CMake C++ build), single squashed commit on `master`, no tags/branches. The `python3/` directory ships pre-built wheels; the C/C++ source that backs `_robot.so` is **not public**.
- **Binding generator**: Direct CPython C API (symbols `PyInit__robot`, `PyModule_Create2`, `PyModule_GetState`). No `pybind11`, `nanobind`, `SWIG`, or `Boost.Python` strings in `_robot.so`. This means a Python 3.10/3.11 rebuild would require either (a) LimX recompiling against newer libpython, or (b) re-running the binding generator they internally use (likely an in-house IDL/codegen pattern, given the direct C API style).
- **Wheel `Requires-Dist`**: `onnxruntime, pyyaml, numpy<1.26.4,>1.21.0, pygame, scipy, pandas, mujoco>3.2.2`. The `numpy<1.26.4` ceiling is another soft Py 3.8 signal (numpy 1.26.x is the last line supporting Py 3.8).
- **METADATA tag**: `py3-none-any` (misleading: file extension says universal pure-Python; reality is a CPython 3.8-only binary).

## Implications for MAY-147

- **LimX has not solved the bridge problem publicly.** Their `tron1-rl-isaaclab` runs Py 3.10 because IsaacLab needs it, and it deliberately stays SDK-free. The Py 3.8 ABI lock isn't crossed — it's **side-stepped via ONNX as the IPC boundary**. Training process (Py 3.10/3.11) and deploy process (Py 3.8) never share an interpreter.
- **The "RL_TYPE" pattern is a useful precedent.** `tron1-rl-deploy-python/controllers/*Controller.py` switches joint ordering between `isaacgym` and `isaaclab` conventions at the SDK boundary. If we adopt the same split, we get a clean place to do convention translation (joint order, sign conventions, observation packing) at the process boundary rather than inside the policy.
- **Two viable architectures for our stack**:
  1. **ONNX boundary (LimX-native pattern)**: train in IsaacLab (Py 3.10/3.11) → export ONNX → deploy from a Py 3.8 process that imports `limxsdk` + `onnxruntime`. Minimal risk, matches LimX's own deploy code, easy to copy `humanoid-rl-deploy-python` patterns.
  2. **IPC bridge (custom)**: keep training-time inference and SDK in separate processes, communicate via shared memory / ZMQ / gRPC. Necessary only if we want closed-loop training-from-real-data or if ONNX export doesn't cover our policy class (e.g., diffusion / VLA outputs that don't ONNX cleanly).
- **Don't expect a Python 3.11 wheel from LimX soon.** No PyPI presence, no public source, no CI, no Py 3.10/3.11 branch in any of their 25 repos. Even their newest ROS2 Humble bridge (`ros2-bridger`, last push 2026-05-31) is a pre-compiled binary that talks MROS — it does not expose a Py 3.10/3.11 import path to `limxsdk`. A direct ask to LimX is the only path to a newer wheel.
- **MROS, not limxsdk, is the actual cross-language interop layer.** `ros2-bridger` (and the older `ros1-bridger`) translate MROS topics to ROS2/ROS1 topics. If we run ROS2 Humble, we could in principle bypass `limxsdk` Python entirely and subscribe to MROS-bridged ROS2 topics from any Python version. This is a third architectural option worth mentioning.

## Open questions

- **Is there a private/customer-only newer `limxsdk` wheel?** The wheel filename pattern `py3-none-any` and the bundled libpython suggest LimX's build system can in principle produce per-Python wheels; they just don't ship them. Worth asking directly: "do you have a Python 3.10 or 3.11 build of `limxsdk` for sim use?"
- **What is `tron1-ss`?** C++ repo last touched 2025-10-13, no description, no README inspected. Could be skill-server, state-sync, or a sim-side helper. Worth a one-line check if it ever becomes relevant.
- **How does `tron1-agent` import the robot?** It uses transformers 4.51 / `onnxruntime` / langchain — likely Py 3.10+. Did not grep for limxsdk import. If it does import it on Py 3.10, that would contradict everything above — should verify before relying on the conclusion.
- **What does MROS look like in Python beyond `limxsdk`?** The `MROS_IP_LIST` / `MROS_ETC_PATH` env vars suggest MROS has a stable wire protocol. Is there a pure-Python MROS client anywhere in their stack, or only the compiled `_robot.so`? If the wire protocol is documented, we could write our own Py 3.11 client and skip LimX's wheel entirely.
- **Does any LimX customer publicly ship a Py 3.10+ rebuild?** Worth a `gh search code "import limxsdk" -L python` across all of GitHub (not just `limxdynamics` org) to see if anyone outside the company has patched a newer build.
