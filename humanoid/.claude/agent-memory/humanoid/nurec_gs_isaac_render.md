---
name: nurec-gs-isaac-render
description: Isaac TiledCamera would NOT render a Gaussian-splat/NuRec scene when tried ~Feb 2026 on the 5090 box; maybe IsaacLab#4951 Blackwell bug; re-test on Isaac Sim 6.0
metadata:
  type: project
---

**#1 risk gating the GS half of the humanoid SLAM PoC (demo ~17-18 Jul 2026): can Isaac actually render a robot camera against a Gaussian-splat (NuRec) scene?**

**Confirmed by Anton:** ~5 months ago (≈Feb 2026) the team tried it and Isaac's **TiledCamera would not render the GS** — real, not a red herring.

**Hardware topology (load-bearing):** two machines — the **working machine** = RTX **4070 Ti SUPER** (Ada, sm_89); **"the box"** = RTX **5090** (Blackwell, sm_120). Heavy Isaac / GS render runs on the box, so the failing test was on the **5090**. (Global memory's "5090 upgrade claim was wrong" refers to the *working machine*; the 5090 is the separate box.)

**Why this reframes the risk:** because the render ran on a 5090/Blackwell card, the subagent's attribution to **IsaacLab issue #4951 (Blackwell / sm_120 tiled-render bug)** is now *plausible and consistent*, not contradictory. Whether current **Isaac Sim 6.0** (GA Jun 2026, ships NuRec natively) fixes it is UNKNOWN.

**How to apply:** Before betting the GS demo on live camera rendering, stand up a NuRec ParticleField in Isaac Sim 6.0 **on the box (5090)** and point a robot RGBD camera at it. Regardless of outcome, one hard fact holds: **splats are visual-only — depth/LiDAR/collision can't see them**, so depth + collision must come from the co-registered MILo mesh ([[reconstruct-pipeline-milo]]). Update this note once re-tested.
