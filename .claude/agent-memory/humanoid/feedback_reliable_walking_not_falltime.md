---
name: feedback-reliable-walking-not-falltime
description: Don't treat fall-time differences (held 1s vs 2.5s) as progress — the bar is a sustained repeating gait; sub-second timing noise in a marginally-stable system is luck, not signal.
metadata:
  type: feedback
---

When debugging the Isaac walk, do NOT report "it held upright longer" (1.0 s vs 2.5 s)
as progress. In a marginally-unstable system those fall-time differences are **chaotic
noise / luck**, not evidence a change helped.

**Why:** stated 2026-06-25 after I called an armature run "progress" because base-z held
0.88 for 1 s vs cratering by 1 s without it — Anton: "either he walked 1s or 2s or 1.5s,
that is just luck, this is not reliable walking so it doesn't matter and not a good
indicator of progress." The robot never produced a sustained gait under any physics-knob
combo; all runs share the same failure mode (brief hold → lunge → topple), only the
fall-time varies.

**How to apply:** the success metric is a **sustained, repeating gait** (clean cyclic
steps, base-z stable for many seconds), not seconds-until-fall. If a change doesn't
change the *failure mode*, treat it as null. Stop trial-and-error tweaking of Isaac
physics params (armature/friction/restitution/solver) read off fall-time; instead diff
our pipeline against the reference that DOES walk (the MuJoCo deploy running the same
walk ONNX) to localize where obs/action/dynamics diverge. See
[[isaac_walk_physics_fidelity]].
