---
name: nav-brain-localizer-seam
description: Oli nav brain (reason/nav/) is invariant + complete; GroundTruthLocalizer is the deliberate seam where real SLAM swaps in for the PoC
metadata:
  type: project
---

The navigation brain (`logic/oli/reason/nav/`, shipped 2026-07-09, MAY-173/175) is a fully
**world-invariant** localize→plan→follow stack: `OccupancyGrid` costmap → 8-connected A* →
PurePursuit → body-frame (vx, vy, wz). It is a **drop-in velocity source that replaces Teleop** —
the downstream glide path (Intent→GlideAction→GLIDE_CMD) is unchanged, so Nav just becomes the joystick.

**The seam:** a `Localizer` interface with a `GroundTruthLocalizer` stand-in. This is the single
swap-point — real relocalization (RTAB-Map loc-mode / cuVSLAM) drops in *here* with no other change.
The GT localizer runs the whole nav loop end-to-end on perfect pose *today*, so the PoC's remaining
work is wiring + a localizer swap, not a rebuild.

**Strategy that produced it:** build the risk-free invariant half first (A*/pursuit/costmap are
commodity, testable offline — 41 tests, brain suite 284 green), leaving only the world-coupled half
(Isaac render, real loc) as the open feasibility question. The world half is the 10-07 front.

Related: [[reconstruct-pipeline-milo]], [[nurec-gs-isaac-render]]; backbone decision in ADR-0002
([[architecture-brain-backbone-ros-vs-diy]]).
