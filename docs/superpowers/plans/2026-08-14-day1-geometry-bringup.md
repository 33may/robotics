# Day 1 — Geometry Bring-up on the SO-ARM101

Date: 2026-08-14
Spec: `docs/superpowers/specs/2026-08-12-ai-inspection-project-definition.md`

## Goal for today

Get the SO-ARM101 moving under script, capturing full frames, and answer the single
biggest technical risk in the project: **does depth work on glazed ceramic?**

The SO-ARM101 is the development platform for the whole geometry loop. Build it here,
prove it works, port to the UR5e. It is a camera positioner, not a kinematics
reference — its 5 DOF and short reach mean many viewsphere cells will be unreachable,
which is the arm, not a bug.

## Machine state, verified 2026-08-14

| | |
|---|---|
| Venv | `~/projects/robotics/.venv` — `fire`, `numpy 2.5.2`, `opencv-contrib-python 4.14.0.94`, `pyrealsense2 2.58.3`, `PyYAML`, `tqdm`, `termcolor`, editable `vbti` |
| Missing | no torch, no pinocchio, no open3d, no scservo_sdk, no lerobot |
| Cameras | **3× D405 connected** on USB bus 6 |
| Arm | **Not connected** — no `/dev/ttyACM*` |
| GPU | RTX 5090, 32 GB, driver 580.178.04. No CUDA toolkit (fine — wheels don't need `nvcc`) |
| Servo control | `vbti/logic/servos/` drives Feetech motors directly via `scservo_sdk`. **No LeRobot needed.** |
| URDF | None in repo. Must be sourced. |

`vbti/logic/reconstruct/` is the *previous* approach — COLMAP, Gaussian splats, Isaac
asset generation. Offline photogrammetry, not live depth. New code does not go there,
though `clean_mesh.py` has Open3D outlier removal and OBB fitting worth borrowing.

## Tasks

### T0 — Commit the spec first

It has never been committed and changed substantially on 2026-08-13. Get a clean base
before writing code.

### T1 — Arm bring-up

Plug in the SO-ARM101, `pip install scservo_sdk`, confirm the port, read joint angles
back, command a single pose.

**Done when:** a script prints live joint angles and moves the arm to a commanded
configuration and back to rest.

Start from `vbti/logic/servos/scan_all.py` and `rest.py`. Note `factory_reset_motors.py`
hardcodes `/dev/ttyACM1` — verify the actual port rather than assuming.

### T2 — Capture rig

One function: settle, dwell ~100 ms, then grab and save **RGB + raw IR stereo pair +
native depth + joint angles + timestamp**.

**Saving the raw IR pair is the important part.** It lets native depth and
Fast-FoundationStereo be compared later on byte-identical frames, so the comparison is
exact regardless of how the camera moved.

**Done when:** N poses produce N complete frame bundles on disk with joint angles
attached.

Never capture while moving — up to 5 cm of displacement in the base frame on a moving
arm, with 10 mm spikes tracking acceleration.

### T3 — Depth quality on glazed ceramic ← **the day's real objective**

Put a glossy mug on the table. Capture from several distances and angles. Look at where
native D405 depth has holes and how much of the object survives.

**Why this matters most:** the spec assumes native depth collapses on glossy surfaces —
the D405 has no IR projector, so it is passive stereo and needs texture. If that
assumption is wrong, Fast-FoundationStereo drops from a hard dependency to an
optimisation, and the riskiest install in the plan becomes optional. If it is right, we
know the install is mandatory before spending days elsewhere.

**Done when:** there is a number — what fraction of object pixels return valid depth at
~15 cm, ~25 cm, ~35 cm — and a saved frame set to re-run through learned stereo later.

D405 notes: min-Z 7 cm, optimal at 848×480, use the Medium Density preset (High Accuracy
deletes most points on low-texture surfaces), depth accuracy ±2% at 50 cm.

### T4 — Forward kinematics

Source an SO-ARM101 URDF — check the SO-ARM100/101 upstream repo and LeRobot's robot
descriptions. Verify link lengths against the physical arm rather than trusting it.
Joint angles → flange pose.

**Done when:** FK output moves sensibly as the arm moves, checked against a ruler at two
or three configurations.

### T5 — Hand-eye calibration (stretch)

ChArUco board, N arm poses, `cv2.calibrateHandEye`. This makes `T_base_cam` known, and
everything geometric depends on it.

Doing it on the SO-ARM101 first is a rehearsal for the UR5e, not a detour — same board,
same solver.

**Done when:** residual is reported. Expect 0.4–0.6 mm on a good eye-in-hand ChArUco
calibration; above 1 mm the calibration has failed.

`cv2.aruco.CharucoDetector`, `calibrateHandEye`, `calibrateRobotWorldHandEye` and
`CALIB_HAND_EYE_PARK` are all verified present in the installed OpenCV 4.14.0.94.
**Do not upgrade to OpenCV 5.x — it does not expose `calibrateHandEye`.**

## In parallel, zero effort

**Ask the colleague who drove the UR5e for the factory DH calibration file.** Longest
lead item in the project, one message. Without it UR states end-effector positions are
off "in the magnitude of centimetres", which dwarfs every algorithmic choice downstream.
The extraction tool `ur_calibration` is a ROS 2 Jazzy binary and the pure-Python route is
unresolved — so the cheapest path by far is that the file already exists on disk
somewhere.

## Realistic cut line

T0–T3 is a solid day and delivers the risk answer. T4–T5 will likely slip to tomorrow.
If time is short, **T3 is the one to protect** — it is the only task today that can
change the design.

## Where the code goes

New module, e.g. `vbti/logic/inspection/`, alongside `cameras`, `depth`, `detection`.
Not in `reconstruct/`.

## Deliberately not today

- Fast-FoundationStereo install (torch ≥2.7 + cu128, flash-attn for sm_120) — T3 tells us
  how urgent it is
- Viewsphere, IK, collision filter — needs FK and hand-eye first
- Anything involving a VLM
- Isaac, and any simulator

## Context for a fresh session

Read the spec first. Then these memory leaves:

- `project/geometry-is-depth-first-not-silhouette`
- `project/geometry-module-two-envelopes`
- `project/viewsphere-cell-addressing-hvrd`
- `project/development-approach-kinematic-sandbox`
- `hardware/d405-passive-stereo-no-ir-projector`
- `hardware/ur5e-d405-hand-eye-calibration-recipe`
- `manipulation/geometry-refinement-multiview-accumulation`

Note `project/development-approach-kinematic-sandbox` says the SO-ARM101 is out of the
project. **That was reversed on 2026-08-14** — it is the development platform for the
geometry loop, as a camera positioner. The leaf needs updating.
