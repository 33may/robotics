---
name: limx-sdk-role-gating
description: LimX SDK pub/sub is role-gated — RobotCmd packets only deliver to sim-role peers, not policy-role subscribers
metadata:
  type: project
---

`Robot(RobotType.Humanoid)` vs `Robot(RobotType.Humanoid, True)` route to different topic delivery sets. A passive observer cannot just subscribe to everything — the role determines what packets reach you.

Concrete case: `subscribeRobotCmd` from a policy-role peer receives 0 `RobotCmd` samples. Those only flow to sim-role peers via `subscribeRobotCmdForSim`. Discovered while trying to capture the wire contract in MAY-142.

**Why:** SDK design assumes a strict deploy↔sim topology. Sim peer listens for cmds, policy peer publishes them — they don't see each other's outbound traffic.

**How to apply:** To passively capture any LimX topic, declare a role that *receives* it. For `RobotCmd` shape capture → use `Robot(type, True)`. For `RobotState`/`ImuData` → policy-role is fine.
