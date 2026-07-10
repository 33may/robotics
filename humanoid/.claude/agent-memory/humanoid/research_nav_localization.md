---
name: research-nav-localization
description: Localization research outcome (2026-07-10) — two-layer stack, RTAB-Map now / cuVSLAM later; full report at docs/research/localization.md
metadata:
  type: project
---

Open-landscape survey + our own closed-loop experiments for the `Localizer` seam completed
2026-07-10 (ultracode workflow, 37 agents). Full decision doc: `docs/research/localization.md`
(branch `33may/nav-localization-research`). Experiment budget in [[experiment-nav-pose-tolerance]].

**DECISION 2026-07-10 (Anton): DROP the proprioceptive odometry layer (C) for the now-track.**
Now-track = a **single out-of-process visual SLAM box** (RTAB-Map) that provides BOTH its own visual/RGBD
odometry (smooth frame-to-frame tracking) AND known-map relocalization; the seam stays a thin
pose-reader. **Why:** (1) leg-FK odometry is a real-robot scaling concern, defer it; (2) **Oli glides now,
doesn't step** — holonomic smooth base motion means there's no gait/foot-contact to derive leg-FK odometry
from, AND no foot-impact shock, so the biped-VIO-degradation risk (report §2.C) evaporates and visual
odometry is actually *easier* now. The E2 "dead-reckon carries the pose between bursty fixes" finding still
holds — it's just sourced from the SLAM's **visual odometry**, not legs. **Accepted trade-off:** we lose the
drift safety-net if visual tracking is lost (feature-poor wall / motion blur); that's exactly what a
proprioceptive layer buys back when we scale to the real *stepping* robot later.

**Original research lean (superseded by the decision above for the now-track):** a two-layer localizer —
1. a fast **in-brain proprioceptive odometry** layer (InEKF-style; DRIFT — IMU + leg-FK) — **deferred**, and
2. a heavy **out-of-process known-map visual relocalizer** emitting bursty SE(2) fixes over the bus;
   the seam stays a thin pose-reader (`DebugPoseLocalizer` shape).

- **Now-track pick: RTAB-Map** (BSD-3; native 2D occupancy-grid output that mirrors our USD-baked
  costmap by construction; true saved-`.db` localization mode; runs **ROS-free** via
  `rtabmap-reprocess`, so invariance holds without ROS in the brain).
- **Runner-up / GPU + real-Jetson path: cuVSLAM / PyCuVSLAM.** Verifier-corrected two brief errors:
  `save_map`/`localize_in_map` **IS** in the standalone Python API, and the license is
  NVIDIA-hardware-only (commercially usable — Oli qualifies), **not** non-commercial. Caveats: needs a
  pose hint (no native kidnapped-robot recovery — pair with cuVGL), own map frame (one-time align to
  the baked grid), no Py3.11 wheel (irrelevant — runs OOP on 3.10/3.12).

**Why this shape:** no lidar + RGBD + leg-FK favors appearance/RGBD reloc over 2D AMCL; the invariance
boundary forces the heavy localizer out-of-process; the E2 experiment proves odometry converts a hard
real-time reloc requirement into a soft bursty one.

**How to apply:** if/when Anton greenlights, follow the report's §5 plan (odometry layer first). Open
decisions still his: (a) does the demo need global/kidnapped recovery or only tracking-from-known-start
(gates ACE/hloc/cuVGL), (b) is the sensor suite frozen (a rear/side cam or real 2D lidar flips the
ranking toward vanilla AMCL). Links: [[nav-brain-localizer-seam]], [[reconstruct-pipeline-milo]].
