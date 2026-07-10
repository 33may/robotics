# Localization for Oli — Research & Experiments

> Decision document for the architect. Staged: (1) literature, (2) code & repos, (3) our closed-loop experiments, (4) recommendation + plan. Evidence and a clear lean; the architect decides. Every unverified assumption is tagged `[VERIFY]`. Where the scout dossiers and the adversarial verifier disagreed, **the verifier wins** and the correction is noted inline.

---

## 0. TL;DR

1. **Two-layer stack, not one localizer.** A fast in-brain **proprioceptive odometry** layer (IMU + leg-FK, InEKF-style) carries the pose between fixes; a **heavy out-of-process visual relocalizer** against the pre-baked map emits occasional SE(2) fixes over the bus. The seam (`Localizer.estimate`) stays a thin pose-reader.
2. **Now-track pick: RTAB-Map** (BSD-3, native occupancy-grid output that mirrors our baked costmap, true saved-`.db` localization mode, runs fully out-of-process — even **ROS-free** via `rtabmap-reprocess`). **Runner-up / GPU path: PyCuVSLAM** (save/localize IS in the Python API — verifier correction; NVIDIA-hardware-only license, Oli qualifies).
3. **Our experiments set the budget** (real, unmodified nav stack, closed loop): steady-state error must stay **< ~0.1–0.2 m position, < ~10° yaw, and — the binding spec — < ~0.1 m systematic bias**. Any few-cm RGBD relocalizer sits comfortably inside this.
4. **Rate is cheap *if* odometry exists.** Without odometry the relocalizer must run **≥ 1 Hz** (hold-last-fix cliffs hard past 0.5 Hz → 58% collisions). With odometry between fixes, **0.2 Hz relocalization still gives 100% success.** This is the measured justification for the out-of-process + fast-odometry architecture.
5. **Build the odometry layer first.** It converts a hard real-time relocalizer requirement into a soft/bursty one, and the < 0.1 m bias spec is a calibration/extrinsics job (moving-camera FK time-sync), not an algorithm choice.

**Sensor flag (load-bearing):** no lidar; D435i RGBD (head + waist/chest, camera on a *moving* link → extrinsics from FK), body IMU, joint encoders. This favors appearance/RGBD-native relocalizers over 2D-scan MCL, and makes leg-FK odometry the differentiated humanoid ingredient. If a rear/side camera or a real 2D lidar is ever added, the ranking shifts toward vanilla AMCL. `[VERIFY]`

---

## 1. Problem & constraints

**The seam.** `logic/oli/reason/nav/localizer.py` defines `Localizer.estimate(observation, camera_frame) -> Optional[RobotPose]` (SE(2): x, y, yaw in a fixed "map frame"). `GroundTruthLocalizer` / `DebugPoseLocalizer` already run the full downstream loop on perfect pose read from a fenced debug side-channel. The task is a **swap**, not a rebuild: the invariant nav stack (OccupancyGrid costmap → 8-connected A* → holonomic pure-pursuit → body-frame vx,vy,wz) is done. The seam docstring itself already names cuVSLAM / RTAB-Map as the intended day-2 backends.

**Invariance boundary (hard rule).** `brain` imports **neither** `isaacsim` **nor** `limxsdk`, directly or transitively (enforced by the `brain` pytest marker + env split). Therefore any heavy/CUDA/ROS localizer runs as a **separate process** emitting SE(2) over a bus; the Localizer becomes a thin pose-reader — exactly the shape `DebugPoseLocalizer` already models. **"Can this run out-of-process and just emit an SE(2) pose?"** is a primary evaluation axis.

**Sensors.** D435i RGBD ×2 (head + waist/chest), body IMU, joint encoders (leg-FK proprioceptive odometry). **No lidar.** Camera is on a moving link → camera→base extrinsics come from FK, time-synced to `observation`. `Observation` carries IMU (acc/gyro/quat) + joint q/dq + measured torque τ, but **no explicit foot contact / force-torque sensor** — contact must be *inferred* (τ+FK or a small learned estimator). This narrows humanoid state estimators (§2.C).

**Environments.** brain = Py3.11, isaac = Py3.11 (Isaac Sim), limx = Py3.8. Dev GPU: RTX 4070 Ti SUPER 16 GB. Real Oli may use an onboard Jetson later.

**Scope (architect-decided).** Sim-first (Isaac Sim) now, real Oli later — shortlist only methods with a credible real path. **Known pre-built map now** (we bake a 2D occupancy grid offline from the scene USD); live reconstruction / online SLAM is a **secondary later track**. Not using Gaussian Splatting.

---

## 2. Stage 1 — Literature review (the open landscape)

Five method families. For each: what it is, humanoid fit, the failure mode that matters, and whether it answers the *known-map* question or only *odometry*.

### 2.A Geometric SLAM with map persistence (RGBD graph/feature SLAM)

The mature turnkey known-map relocalizers. **RTAB-Map** — RGBD graph SLAM with appearance-based (bag-of-words) loop closure, an explicit saved-`.db` **localization mode**, and — uniquely — native **2D occupancy-grid** output that matches our offline-baked costmap regime by construction ([RTAB-Map site](http://introlab.github.io/rtabmap/); [arXiv 2403.06341](https://arxiv.org/pdf/2403.06341)). **stella_vslam / OpenVSLAM** — feature-based, cleanest `--map-db-in` + `--disable-mapping` pure-localization API, BSD-2, CPU-only, but **no IMU** and sparse map only ([README](https://github.com/stella-cv/stella_vslam/blob/main/README.md)). **ORB-SLAM3** — highest classical accuracy via Atlas + DBoW2, but RGBD saved-map reload is a **documented, still-open defect** needing a C++ `ComputeBoW()` patch ([issue #515](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/515)).

**Humanoid evidence (load-bearing).** The SURENA-V real-humanoid RGBD study ([arXiv 2401.02816](https://arxiv.org/abs/2401.02816)) measured ATE ORB-SLAM3 **0.107** < RTAB-Map **0.164** < OpenVSLAM **0.185** — but RTAB-Map was the **most robust** (only one to hold tracking in feature-poor scenes) and the only one with native occupancy-grid output. So ORB-SLAM3 wins raw accuracy, RTAB-Map wins usable robustness + map artifact.

### 2.B GPU visual-inertial odometry/SLAM (NVIDIA cuVSLAM)

**cuVSLAM / PyCuVSLAM** — CUDA VO + within-session SLAM, RGBD ("mono-depth") + stereo-inertial + multi-camera, sub-4 ms stereo-inertial on Jetson Orin, official Isaac Sim tutorial → best sim→real continuity ([arXiv 2506.04359](https://arxiv.org/html/2506.04359v2); [tutorial](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/tutorial_isaac_sim.html)). **Verifier correction to the original brief:** saved-map **save/localize IS exposed in the standalone Python API** (`tracker.save_map` / `tracker.localize_in_map` / `SlamLocalizationSettings`, verified in `examples/kitti/track_kitti_slam.py`), and the license is **NOT non-commercial** — it is the NVIDIA Community License, commercially usable but **NVIDIA-hardware-only** (Oli qualifies). Real caveats that remain: relocalization needs a **pose hint** (no native global/kidnapped-robot recovery — pair with cuVGL for hint-free cold start), its map is cuVSLAM's own landmark DB (needs one-time alignment to our occupancy grid), and there is **no Python 3.11 wheel** (irrelevant — runs out-of-process on 3.10/3.12 in a container).

### 2.C Legged/humanoid proprioceptive state estimation (the odometry layer)

Contact-aided **InEKF** / **DRIFT** (BSD-3, modular, CPU-real-time) / **SEROW** (humanoid-native, estimates base + CoM) / **Cerberus** (VILO, <1% drift) / learned **Legolas / AutoOdom** (sim-to-real, AutoOdom validated on a real Booster T1 humanoid, IMU+encoders only). Key theory: proprioception is **provably yaw+position-unobservable** ([Hartley 2020, IJRR](https://journals.sagepub.com/doi/full/10.1177/0278364919894385)) — it **drifts** and can *never* be the sole map-frame localizer. Its role is the **fast, gait-robust motion prior** that stabilizes the visual relocalizer and carries the pose between bursty fixes. **Contact-detection crux:** every classical estimator here assumes a contact signal Oli lacks natively; infer it from τ+FK or a small learned estimator (DRIFT ships one). `[VERIFY]` biped flat-foot vs point-contact adaptation cost.

**Why this matters for a biped (2024-26 evidence):** foot-impact shock injects high-frequency noise that can make *tightly-coupled* visual-inertial modes **degrade** on legged robots ([quadruped eval, arXiv 2606.19067](https://arxiv.org/html/2606.19067)); ORB-SLAM3 inertial modes hit catastrophic failure under fast rotation. Implication: IMU coupling is **not a free win** — visual-only reloc (RTAB-Map RGBD-only, stella_vslam) may be *more* robust for the demo, and leg-FK odometry (not IMU-only VIO) is the differentiated humanoid path.

### 2.D 2D probabilistic localization on the occupancy grid (MCL / AMCL)

The classic answer, and the **only** family that natively consumes our exact baked 2D grid: a particle filter scoring a range scan against the map, emitting SE(2) directly ([Nav2 AMCL](https://docs.nav2.org/configuration/packages/configuring-amcl.html); humanoid precedent [Hornung et al. IJHR](https://www.arminhornung.de/Research/pub/hornung14ijhr.pdf)). **But** the no-lidar constraint bites: a forward D435i flattens to a narrow ~55–90° virtual scan (`depthimage_to_laserscan` / `pointcloud_to_laserscan`), which is under-constrained for global localization and symmetric geometry, and a swaying/pitching torso tilts the virtual scan plane (needs per-frame IMU-gravity-leveling + FK reprojection). Workable as a cheap first smoke test on the existing grid, but the sensor geometry makes it a weak *primary*. Modern engines: **Beluga AMCL** (Apache-2.0, ROS-agnostic C++17 core) and **MRPT pf-localization** (BSD-3, ROS-independent core, best structural fit) — see §3 caveats.

### 2.E Learned relocalization & place recognition

**Scene Coordinate Regression (ACE / GLACE)** — trains a scene into a ~4–9 MB MLP in minutes from posed RGB (which Isaac renders for free), regresses metric 6-DoF pose via PnP → excellent instant global-init seed ([ACE](https://nianticlabs.github.io/ace/); [GLACE](https://arxiv.org/abs/2406.04340)). **hloc** (retrieval → SuperPoint → LightGlue → PnP) — battle-tested geometry-first accuracy anchor, Apache-2.0 ([hloc](https://github.com/cvg/Hierarchical-Localization)). **Reloc3r / MASt3R** — scene-agnostic feed-forward, best hedge against sim-to-real and per-scene rebaking. Weak spot across the family: **sim-to-real appearance gap** (SCR hardest hit; needs domain randomization) and per-scene retraining. Best role: a **global-init / kidnapped-robot seed** feeding cuVSLAM/RTAB-Map's tracker, not the primary control-rate loop. **Absolute Pose Regression (PoseNet)** — skip; low accuracy, effectively retrieval in disguise.

### Family summary

| Family | Answers known-map? | Emits SE(2) OOP? | Humanoid fit | Main failure mode |
|---|---|---|---|---|
| A. RGBD graph/feature SLAM (RTAB-Map, stella, ORB3) | **Yes** (saved DB) | Yes | Strong (SURENA-V validated) | Feature-poor walls; ORB3 RGBD reload defect |
| B. GPU VIO/SLAM (cuVSLAM) | Yes (+ pose hint) | Yes (Python) | Strong (RealSense+Jetson native) | No global reloc; own map frame; foot-impact IMU |
| C. Proprioceptive (DRIFT/SEROW/InEKF) | **No** (drifts) | Yes | Highest (biped-native) | Unbounded drift; needs contact inference |
| D. 2D MCL/AMCL (Beluga/MRPT) | Yes (our grid!) | Yes | Weak w/o lidar | Narrow-FOV virtual scan; tilting scan plane |
| E. Learned reloc (ACE/hloc/Reloc3r) | Yes (baked scene) | Yes | Moderate (global-init) | Sim-to-real gap; per-scene retrain |

---

## 3. Stage 2 — Code & repos

Adversarially verified rows marked **✅ verified**; unverified marked ⚠️. Where verifier corrected the scout, the corrected value is shown and flagged.

| System | Family | License | Python / CUDA | ROS? | OOP SE(2)? | Map save/load | Maturity / activity | Our-env fit | Effort | Relevance | Verified |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **RTAB-Map** | A RGBD graph SLAM | **BSD-3** | C++ core, no pip SLAM API; **no CUDA** for our path | Optional (also **ROS-free** `rtabmap-reprocess`) | **Yes** | **Native `.db` loc-mode + 2D occupancy grid** | Very mature, v0.23.8 (2026-07-05) | Good OOP; brain integrates over bus | Low–med | **0.88** | **✅ verified** |
| **PyCuVSLAM / cuVSLAM** | B GPU VIO/SLAM | **NVIDIA Community** (commercial-OK, **NV-HW-only** — *corrected*) | Py3.10/3.12 wheels, **no 3.11**; **CUDA 12/13 req** | No (standalone Python) | **Yes** (`world_from_rig` SE(3)→SE(2)) | **Yes** `save_map`/`localize_in_map` (*corrected: IS exposed*) | Vendor-active, v16.0 (2026-06-02) | Partial: 3.10/3.12 side-env + Ubuntu container | Med | **0.88** | **✅ verified** |
| **stella_vslam** | A feature SLAM | **BSD-2** | C++, no official Py; **no CUDA** | Optional (separate wrapper) | Yes (SE(3), world→cam — invert) | Native `.msg` + `--disable-mapping` | Active, v0.6.0 (2025-01) | CPU, build-from-source/Docker | Med | **0.72** | **✅ verified** |
| **ORB-SLAM3** | A feature-VI SLAM | **GPL-3.0** | `orbslam3-python` cp38–cp312 wheels; **no CUDA** | No | Yes | ⚠️ **Atlas save/load = open defect** (RGBD reload needs C++ patch; not in Py API) | Stale upstream (v1.0 2021) | Good (CPU, wheels) | Med–high | **0.68** | **✅ verified** |
| **GeoFlow-SLAM** | A+C legged RGBD-inertial-leg | **GPL-3.0** | C++, no Py bindings; **no CUDA** | Optional | Yes (SE(3)→SE(2)) | Inherited Atlas (`SaveAtlasToFile` cfg-key; `SaveAtlas` *private* — *corrected*) | Thin research drop (10 commits) | OOP C++ process | Med–high | **0.68** | **✅ verified** |
| **Beluga AMCL** | D particle MCL | **Apache-2.0** | C++17 header core, no Py; **no CUDA** | Core ROS-free; node needs ROS2 | Yes | ⚠️ pgm/yaml + reloc services are **ROS-node only**; core has no file loader (*corrected*) | Very active, 2.1.1 (2026-04) | Good; Route A = more glue than dossier implied | Med | **0.82** | **✅ verified** |
| **MRPT pf-localization** | D particle MCL | **BSD-3** | `pymrpt` (no wheel; .deb ABI-locked to system Py) | No (ROS-independent core) | Yes | Excellent (`.gridmap.gz`, ROS-yaml ingest) | Very active, 3.1.2 (2026-07-07 — *dates corrected*) | Medium; 3.11 → source build or bridge | Med–high | **0.62** | **✅ verified** |
| **DRIFT (UMich-CURLY)** | C InEKF odometry | **BSD-3** | C++, no Py bindings; **no CUDA** (core) | No | Yes (drifting odom) | **No** (odometry only) | Field-tested, dormant since 2024-02 | Good CPU; needs biped adaptation | Med–high | **0.50** | **✅ verified** |
| **SEROW** | C humanoid EKF | **GPL-3.0** | **pybind11** (build-from-source); **no CUDA** | Optional | Yes (base SE(3)) | **No** (no map/reloc) | Active, 2026-04 | Moderate (Pinocchio build) | Med–high | **0.35** | **✅ verified** |
| **Kimera-VIO / Kimera2** | C VIO+RPGO | **BSD-2** | C++, no Py; **no CUDA** | Optional | Yes | ⚠️ **No known-map reload** (open request) | Mature, maintenance-mode | Neutral (own binary) | High | **0.35** | ✅ verified |
| **maplab 2.0** | offline VI-map + reloc | **Apache-2.0** | C++, no Py; no CUDA | **ROS1 required** (EOL) | Yes (ROVIOLI LOC) | **Yes** (durable multi-session) | Dormant since mid-2023 | Poor (EOL ROS1, no RGBD depth) | High | **0.40** | ⚠️ unverified |
| **DPVO / DPV-SLAM** | E deep VO/SLAM | **MIT** | Py3.10 + CUDA (hard) | No | Yes (SE(3)) | **No** (VO / within-session) | Stable, 2024-10 | Separate CUDA env | High | **0.20** | ✅ verified |
| **ACE / GLACE / scrstudio** | E scene-coord regression | ACE patent-pending; GLACE/scrstudio OSS | Py + CUDA | No | Yes (PnP → SE(2)) | Baked MLP per scene | Active (CVPR'23–'25) | Good sim-first; GPU inference | Med | **0.55** | ⚠️ unverified |
| **hloc (+LightGlue)** | E structure matching | **Apache-2.0** (LightGlue) | Py + CUDA | No | Yes | SfM map from GT poses | Very active | Good; heavy per-query | Med | **0.55** | ⚠️ unverified |
| **nav2_amcl** | D reference AMCL | **LGPL-2.1** | C++, no Py; no CUDA | **ROS2 required** | Yes | Loads baked grid | Active | Ref spec only; needs depth→scan | Med | **0.35** | ✅ verified |
| **Cartographer (pure_localization)** | D CSM localization | **Apache-2.0** | C++, no Py; no CUDA | **ROS2 node** | Yes | `.pbstream` load | ROS2 port active; core frozen | Heavy; lidar-first, no RGBD | High | **0.28** | ✅ verified |
| **MegaParticles** | D GPU Stein PF | **none (no code)** | — | — | — | Needs 3D cloud map | **Paper only, no repo** | Not adoptable | Very high | **0.15** | ✅ verified |

**Key verifier corrections folded into the table above:** (1) PyCuVSLAM license is NV-hardware-only *not* non-commercial, and map save/localize *is* in the Python API. (2) RTAB-Map has a fully **ROS-free** localization path (`rtabmap-reprocess`), so the invariance boundary holds even without ROS2. (3) Beluga's map-load + relocalization services live in the **ROS node**, not the ROS-agnostic core — the pure-C++ "Route A" is more glue than the dossier implied. (4) ORB-SLAM3 / GeoFlow Atlas save/load is config-key-driven (and `SaveAtlas` is private in GeoFlow); ORB3 RGBD known-map reload is an **open defect**, not a feature. (5) MRPT is further along than dated (3.1.2, 2026-07-07) and its pymrpt binding-completeness wiki page is stale (bindings are pybind11 and near-complete).

---

## 4. Experiments — our closed-loop nav-tolerance measurements

We built a headless, deterministic SE(2) kinematic harness that drives the **real, unmodified** Oli nav stack (`OccupancyGrid.inflate` → `plan_path` A* → `PurePursuit.command` → body twist) and measures how much localization error and how low a relocalization rate the loop tolerates. Each tick: `est = true_pose + injected noise`; plan+steer on `est`; integrate the *true* pose by the commanded twist; judge success/collision on the *true* pose vs the raw grid. Three baked maps (empty, corridor+doorway, obstacle field), 24 trials/cell, fixed seed `20260710`, ~18 s runtime, imports neither isaacsim nor limxsdk. **Reproduced fresh by a separate verifier — matched `results.json` cell-for-cell at 24 trials.**

- Harness: `/home/may33/projects/ml_portfolio/robotics/humanoid/docs/research/localization/experiments/harness.py`
- Results: `.../experiments/results/results.json`, `results.csv`
- Plots: `.../experiments/results/e1_accuracy.png`, `.../experiments/results/e2_reloc_rate.png`

### E1 — Accuracy budget

| Noise | Magnitude | Success | Collision | Note |
|---|---|---|---|---|
| Pos Gaussian σ | 0.05 m | 1.00 | 0 | harmless |
| Pos Gaussian σ | 0.1 m | 1.00 | 0 | harmless |
| Pos Gaussian σ | 0.2 m | 0.958 | 0.042 | collisions creep in |
| Pos Gaussian σ | 0.4 m | 0.917 | 0.083 | still clears 0.9 |
| Yaw Gaussian σ | 10° | 1.00 | 0 | robust |
| Yaw Gaussian σ | 20° | 0.833 | 0 | all timeouts |
| **Pos constant BIAS** | 0.05 m | 1.00 | 0 | fine |
| **Pos constant BIAS** | 0.1 m | 0.875 | 0 | edge |
| **Pos constant BIAS** | **0.2 m** | **0.417** | 0 | **collapses** |

**Verdict:** position error is cheap; heading is robust to ~10°; **constant bias is the binding constraint** — a persistent offset steers the robot consistently wrong where zero-mean noise averages out. Actionable steady-state target: **< ~0.1–0.2 m position, < ~10° yaw, < ~0.1 m systematic bias.** (Baseline at σ=0 is 0.958, not 1.0 — a hand-rolled-pursuit artifact with no path-recovery, documented; the budget is stated relative to that baseline.)

### E2 — Relocalization rate & does-it-need-odometry

| Reloc period | Rate | Between-fix = HOLD last | Between-fix = DEAD-RECKON |
|---|---|---|---|
| 1 tick | 10 Hz | 1.00 | 1.00 |
| 5 ticks | 2 Hz | 1.00 | 1.00 |
| 10 ticks | 1 Hz | 0.958 | 1.00 |
| 20 ticks | 0.5 Hz | **0.208** (58% collision) | 1.00 |
| 50 ticks | 0.2 Hz | **0.083** (92% collision) | **1.00** (~0.24 m) |

**Verdict:** hold-last-fix survives to **~1 Hz** then falls off a cliff; dead-reckon (perfect-odometry proxy) holds **100% to 0.2 Hz**. So **a slow relocalizer is fine only if paired with fast odometry.** This is the measured endorsement of the two-layer, out-of-process architecture: heavy CUDA/ROS relocalizer emitting occasional fixes + fast in-brain IMU+FK odometry between them.

**Reproduction verdict:** ✅ reproduced (fresh run, base conda Py3.13, real invariant nav stack, no isaacsim/limxsdk touched; 24-trial run bit-for-bit vs `results.json`, 12-trial run same shape).

**Caveats (read the numbers correctly):** pure kinematic model (perfect actuation — real humanoid tracking error is unmodeled, so these are an **upper bound**; real budgets tighter); DEAD-RECKON is a **zero-drift** proxy (real IMU+FK drifts, so the usable reloc period sits between the hold and dead-reckon curves — but the *takeaway* is drift-direction-independent); no latency or dynamic obstacles modeled; the harness injects abstract SE(2) error so conclusions are **sensor-agnostic** (the no-lidar/RGBD assumption is not exercised here).

---

## 5. Stage 3 — Recommendation + integration plan

### The recommendation (grounded in survey + budget)

Fill the seam with a **two-layer localizer**:

1. **Fast odometry layer (build first).** A proprioceptive InEKF-style estimator (IMU + leg-FK, contact inferred from τ+FK) running at control-adjacent rate. **DRIFT** (BSD-3, CPU-real-time, modular, ships a learned contact estimator) is the prototype pick; **SEROW** is the humanoid-native reference. This is what E2 proves the loop needs: it turns the relocalizer's hard ≥1 Hz requirement into a soft 0.2 Hz one. It **cannot** be the whole answer (provably drifts).
2. **Known-map relocalizer (the visual fix).** **RTAB-Map** is the now-track pick: BSD-3, native occupancy-grid output that agrees with our baked costmap by construction, true saved-`.db` localization mode, and it runs fully out-of-process — either as a ROS2 node bridged to the bus or **ROS-free** via `rtabmap-reprocess` against a saved `.db`, so the invariance boundary holds without ROS in the brain. **PyCuVSLAM** is the runner-up / GPU upgrade (fastest+most accurate, Isaac Sim tutorial, Jetson-native) once a Jetson is in the loop — pair it with **cuVGL** for hint-free cold start since it needs a pose hint.

Both meet few-cm accuracy → comfortably inside the E1 envelope. The real spec is the **< 0.1 m systematic bias**, which is a **moving-camera FK-extrinsics calibration** job, not an algorithm choice.

### Mapping onto the seam

| Concern | Decision |
|---|---|
| **Env / process** | Relocalizer runs OUT-OF-PROCESS (isaac/ROS2 container or a 3.10/3.12 CUDA side-env for PyCuVSLAM). Odometry layer: in-brain-adjacent process or brain-side if pure-Python. Brain stays 3.11, pure. |
| **Node vs in-brain** | Neither localizer imports into the brain. The seam `Localizer.estimate` reads the fused SE(2) pose off the bus (the `DebugPoseLocalizer` shape). |
| **Frame anchoring** | RTAB-Map: bake our 2D costmap FROM RTAB-Map's occupancy grid, or co-register `.db` map frame to the USD-baked grid (one-time rigid alignment). cuVSLAM/stella: one-time SE(3) rigid alignment of the SLAM map frame to the grid. |
| **Moving-camera FK** | Stream time-synced camera→base FK transforms over the bus alongside `camera_frame.stamp_ns`; the relocalizer composes camera-in-map with FK to get base SE(2). **This is where the 0.1 m bias budget is won or lost.** `[VERIFY]` FK/extrinsic timing jitter at the head during dynamic walking — may argue for the **chest/waist camera** (less gait-coupled) as primary SLAM sensor. |
| **Pose-bus contract** | Minimal message: `(stamp_ns, x, y, yaw, quality/covariance)`. `[VERIFY]` sim-time (D8 clock) propagation to the external localizer. Reloc events emit **discrete jumps** (REP-105) — the pose-reader/fuser must absorb them before the pure-pursuit planner (the odometry layer naturally smooths this). |

### Task breakdown (now-track)

1. **Odometry layer first** — prototype DRIFT out-of-process on Isaac IMU+joint+contact GT; validate drift vs `GroundTruthLocalizer`. Infer contact from τ+FK. `[VERIFY]` biped flat-foot adaptation.
2. **Bake the RTAB-Map map in sim** — drive the scene with rendered D435i RGBD, save `.db`; reconcile its occupancy grid with the USD-baked costmap (co-register or re-bake).
3. **Stand up the relocalizer process** — RTAB-Map in localization mode (`Mem/IncrementalMemory=false`, `Reg/Strategy=0` vision-only), bridge `/localization_pose` → bus SE(2). Feed one fused odom stream (built-in RGBD VO, or leg-FK+IMU via a `robot_localization`-style EKF).
4. **Fuse + smooth** — combine bursty visual fixes with the fast odometry into one SE(2) stream that absorbs REP-105 jumps; wire into the seam as a pose-reader.
5. **Close the loop in sim** — run the full nav stack on the fused pose; measure estimated-vs-truth against the GT baseline; confirm it lands inside the E1/E2 budget.
6. **Bias hunt** — calibrate moving-camera FK extrinsics until systematic bias < 0.1 m (E1's binding spec).

### Live-reconstruction later-track (secondary)

Same systems flip to build-map mode with loop closure: **ORB-SLAM3 Atlas** (multi-session reference), **RTAB-Map long-term memory**, **cuVSLAM + nvblox** (live TSDF/ESDF → online occupancy/costmap so nav can drop the pre-baked grid). Deep VO (**DPV-SLAM**) is the low-memory single-GPU option and the bipedal-head-shake robustness hedge. **GeoFlow-SLAM** is the differentiated humanoid-robustness upgrade (RGBD-inertial + leg-FK fusion, Unitree-G1-tested) if visual-only reloc proves gait-fragile. Learned **ACE/GLACE** or **Reloc3r** provide global-init / kidnapped-robot recovery seeds. Hold all of these until the RTAB-Map + odometry baseline is proven.

---

## 6. Open questions & `[VERIFY]` assumptions

- `[VERIFY]` **Does the demo need global localization / kidnapped-robot recovery, or only tracking from a known start?** If a start pose is always available, skip full global reloc → lean on prior-seeded RTAB-Map/cuVSLAM + odometry, defer ACE/hloc/cuVGL.
- `[VERIFY]` **Contact inference:** is τ+FK contact good enough, or is a learned contact estimator (DRIFT's, deep-contact-estimator) needed? This picks the filter vs learned-odometry track.
- `[VERIFY]` **Moving-camera FK timing/jitter at the head during walking** — does it inject > 0.1 m bias? If so, prefer the chest/waist camera as primary SLAM sensor.
- `[VERIFY]` **Map artifact agreement:** does the baked grid's obstacle set match what a chest/head depth slice sees at scan height (glass, thin legs, furniture)?
- `[VERIFY]` **IMU coupling on a biped** — is tightly-coupled VIO a net loss under foot-impact shock? Consider RGBD-only reloc + loose-coupled leg-FK.
- `[VERIFY]` **Sim-to-real for appearance-based maps** — how photorealistic is the current Isaac scene render? If low, geometry-based (RTAB-Map RGBD, registration) transfers cleanly; appearance/feature/SCR maps may need re-mapping on real Oli or domain randomization.
- `[VERIFY]` **Sim-time propagation** to the external localizer over the bus (D8 clock).
- `[VERIFY]` **Sensor assumption itself:** adding a rear/side camera or a 2D lidar flips the ranking toward vanilla AMCL — is the suite frozen?
- `[VERIFY]` **Jetson budget:** RTAB-Map (CPU) and cuVSLAM (GPU, Jetson-native) both fit; hloc SfM (>30 GB maps) / MASt3R may not — confirm onboard SWaP before shortlisting heavy learned methods.

---

## 7. Sources

**Systems & repos**
- [RTAB-Map (introlab/rtabmap)](https://github.com/introlab/rtabmap) · [RTAB-Map site](http://introlab.github.io/rtabmap/) · [arXiv 2403.06341](https://arxiv.org/pdf/2403.06341)
- [PyCuVSLAM / cuVSLAM (nvidia-isaac)](https://github.com/nvidia-isaac/cuVSLAM) · [KITTI save/localize example](https://github.com/nvidia-isaac/cuVSLAM/blob/main/examples/kitti/track_kitti_slam.py) · [cuVSLAM arXiv 2506.04359](https://arxiv.org/html/2506.04359v2) · [Isaac Sim tutorial](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/tutorial_isaac_sim.html)
- [stella_vslam](https://github.com/stella-cv/stella_vslam) · [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) · [ORB3 RGBD reload defect #515](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/515) · [GeoFlow-SLAM](https://github.com/HorizonRobotics/GeoFlowSlam) ([arXiv 2503.14247](https://arxiv.org/abs/2503.14247))
- [Beluga AMCL](https://github.com/Ekumen-OS/beluga) · [MRPT](https://github.com/MRPT/mrpt) · [mrpt_pf_localization](https://docs.ros.org/en/jazzy/p/mrpt_pf_localization/) · [nav2_amcl](https://docs.nav2.org/configuration/packages/configuring-amcl.html)
- [DRIFT (UMich-CURLY)](https://github.com/UMich-CURLY/drift) ([arXiv 2311.04320](https://arxiv.org/html/2311.04320v2)) · [SEROW](https://github.com/mrsp/serow) · [Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO) ([Kimera2 arXiv 2401.06323](https://arxiv.org/abs/2401.06323)) · [maplab 2.0](https://github.com/ethz-asl/maplab)
- [DPVO / DPV-SLAM](https://github.com/princeton-vl/DPVO) ([ECCV 2024 arXiv 2408.01654](https://arxiv.org/abs/2408.01654)) · [ACE](https://nianticlabs.github.io/ace/) · [GLACE](https://arxiv.org/abs/2406.04340) · [hloc](https://github.com/cvg/Hierarchical-Localization) · [Reloc3r](https://github.com/ffrivera0/reloc3r) · [MASt3R](https://arxiv.org/pdf/2406.09756)

**Humanoid / legged evidence**
- [Comparative RGB-D SLAM on SURENA-V humanoid (arXiv 2401.02816)](https://arxiv.org/abs/2401.02816)
- [Sensor Configuration Matters: foot-impact degrades tightly-coupled VIO on quadrupeds (arXiv 2606.19067)](https://arxiv.org/html/2606.19067)
- [Contact-aided invariant EKF (Hartley, IJRR 2020)](https://journals.sagepub.com/doi/full/10.1177/0278364919894385) · [MCL for humanoid navigation (Hornung, IJHR 2014)](https://www.arminhornung.de/Research/pub/hornung14ijhr.pdf) · [AutoOdom (arXiv 2511.18857)](https://arxiv.org/abs/2511.18857)

**Front-ends / theory**
- [KLD-Sampling (Fox, IJRR 2003)](https://journals.sagepub.com/doi/10.1177/0278364903022012001) · [depthimage_to_laserscan](https://github.com/ros-perception/depthimage_to_laserscan/tree/ros2) · [pointcloud_to_laserscan](https://github.com/ros-perception/pointcloud_to_laserscan) · [Nav2 lidar-free vision nav](https://docs.nav2.org/tutorials/docs/using_isaac_perceptor.html)

**Our experiments** — harness + results + plots under `/home/may33/projects/ml_portfolio/robotics/humanoid/docs/research/localization/experiments/` (reproduced fresh, deterministic seed 20260710).
