---
name: may172-glide-scope-defer-fit
description: MAY-172 scoped to a reliable kinematic glide; the MuJoCo joystick→base-velocity dynamics FIT is deferred — SLAM doesn't need velocity fidelity.
metadata:
  type: project
---

**Decision (Anton, 2026-07-01):** MAY-172 is scoped down to making the **kinematic
glide work reliably**. The MuJoCo-fitted joystick→base-velocity dynamics model is
**deferred**, not built this pass. Use a simple accel / turn-rate-limited integrator
(commanded velocity → smoothed base velocity → integrate pose) instead.

**Why:** SLAM (MAY-173) consumes camera streams + ground-truth pose. Because the base
is kinematic, the ground-truth trajectory is known *exactly* for free — so how faithfully
the commanded velocity matches the real walk's command→velocity response is invisible to
SLAM. The MuJoCo fit only earns its keep for demo believability (Oli moves like Oli, not
a floating cam) and real-walk drop-in parity (nav commands transfer to the real gait) —
neither on the SLAM critical path. Anton questioned the fit's value; the reasoning held.

**How to apply:** Build the glide + a placeholder motion model now; both plug into the
*same* seam so the fitted model (or the real walk) can drop in later without changing
callers. Revisit the fit only when polishing the demo or wiring real-robot nav. A later
**camera perturbation** (bob/sway mimicking a walking robot) is the intended way to make
SLAM more realistic — separate from the velocity fit, and cheaper.

Related: [[project-joystick-teleop-architecture]], [[project-invariant-oli-interface]],
[[mujoco-world-via-limx-comm-edge]], [[isaac-oli-smoke-loader]].
