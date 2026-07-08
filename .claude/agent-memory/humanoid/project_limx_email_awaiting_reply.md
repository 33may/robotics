---
name: project-limx-email-awaiting-reply
description: 2026-07-01 Anton emailed LimX Dynamics for the HU-D04-01 Isaac asset + training-time serial-ankle/waist actuator config; reliable dynamic Isaac walk is blocked on their reply.
metadata:
  type: project
---

On **2026-07-01** Anton sent messages to **LimX Dynamics** requesting: (1) the official
HU-D04-01 (Oli) Isaac Sim / IsaacLab asset + the articulation/actuator config they use, and
(2) the training-time serial-PR-ankle/waist params (kp/kd, armature/reflected inertia,
effort limit, drive mode, sim dt/decimation) + how they make the serial ankle dynamically
equivalent to the parallel achilles deploy. **Reliable dynamic Isaac walk is blocked on
their reply.**

**Why:** The 2026-07-01 last-manual-attempt proved dynamic Isaac walk is a first-step
lateral contact/body-fidelity gap that only LimX's HU-D04 training config can close (proven
from 5 independent angles — see [[isaac-walk-physics-fidelity]]). The full email is drafted
at the top of `humanoid/daily/01-07-2026.md`.

**How to apply:** Before reopening Isaac dynamic-walk work, check whether LimX has replied.
If they have, their HU-D04 IsaacLab contact/foot-collision/actuator config is the missing
fidelity piece — reconcile against it. Until then, dynamic walk stays in MuJoCo (works) and
the demo locomotion is kinematic glide ([[may172_glide_wiring]]).
