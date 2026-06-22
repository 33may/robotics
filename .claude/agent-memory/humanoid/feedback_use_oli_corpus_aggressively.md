---
name: feedback-use-oli-corpus-aggressively
description: Query the oli-corpus MCP server as a routine second opinion on Oli/LimX claims — it's often authoritative and almost always faster than reconstructing from memory or grepping the wheel.
metadata:
  type: feedback
---

When writing about Oli, LimX SDK, MROS, or HU_D03/D04 hardware, **make a habit of asking the oli-corpus MCP server** — even for claims I think I already know. It's another opinion, often authoritative, and the structured tools added 2026-06-22 make it nearly free to consult.

**Why:** Anton wants me to lean on the corpus more. We've already established the docs themselves aren't infallible (LimX's own `humanoid-mujoco-sim/README.md` says "Python 3.8 or higher is recommended" — wrong, the wheel is hard-locked), so this isn't "the corpus is always right." It's that I tend to lean on session memory and paraphrased prior work when the corpus would be a fast, structured, cite-able cross-check. A second opinion catches more drift than confidence.

**How to apply:**

- **Structured tools first** (cheaper and more precise than search):
  - `sdk_joint_order(robot_id)` → canonical PR-order joint list from on-robot `walk_param.yaml` + URDF cross-check
  - `joints(robot_id, ...)` / `links(robot_id, ...)` → URDF/MJCF metadata
  - `find_symbol("RobotCmd", kind="struct")` etc. → C++ struct body from `limxsdk-lowlevel/include`
  - `pkg_info` / `nodes` / `topics` / `robots` → ROS-side metadata
  - `raw_file(uri)` → fetch any indexed file verbatim
- **Then `search`** for narrative claims (control modes, MROS conventions, IMU placement, env vars).
- **Cite** with `oli-corpus://<doc_id>#<section>` URIs when the claim ends up in a design, spec, vendor doc, or memory entry — citation > paraphrase.
- When the corpus disagrees with something I "remember" — investigate. Usually I'm wrong; occasionally the docs are. Both outcomes are useful.
- Querying the corpus is essentially free; not querying it before writing an authoritative claim is the costlier default.

Related: [[feedback-check-memory-and-corpus-first]], [[reference-oli-corpus-mcp]], [[reference-oli-corpus-structured-tools]].
