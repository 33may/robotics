# motion — motion solving (MAY-184)

## Purpose
Get the arm from viewpoint to viewpoint without hitting anything, and
without ugly motion. Replaces top-level `ik.py` and `safety.py` (SO-101 legacy).

## Files
- `ik.py` — `UR5eIK`: EAIK all-branches analytic IK on NOMINAL DH, flange
  frame. Frame adapters verified vs pinocchio (<1e-9). ALWAYS filters the
  `is_LS` pseudo-solutions (they carry cm-level error). Calibrated-DH
  Newton-refine hook pending calibration extraction (MAY-181).
- `direct.py` — tier 1: validated straight joint move + endpoint branch
  filter. No search — either the exact moveJ is provably clear or we escalate.
  `--demo`: sample hover targets, execute all, freeze red at impact.
- `smooth.py` — post-processing, planner-agnostic. THIS is what fixes path
  quality: `partial_shortcut` straightens ONE joint at a time. Full-DOF
  shortcutting (the classic algorithm, and pyroboplan's) can never undo a
  single wrapped joint — every all-joint chord across the wrap collides, so
  the wrap survives every iteration. Measured on the same planner:
  median detour 1.84 → 1.15, worst 8.97 → 3.50.
- `rrt.py` — tier 2: OMPL RRT-Connect, best-of-N within a time budget,
  `smooth.polish`, then a path-quality GATE. Goal is a SET (`ob.GoalStates`
  over every free IK branch), so the search picks the cheapest branch instead
  of committing to the nearest-by-L1 one. Escalates 0.2 s → 0.6 s → refuse.
- `plan.py` — THE ladder: `plan_viewpoint()` = branches → tier 1 → tier 2.
  Every returned path is validator-approved; None = honest refusal, with a
  `reason` distinguishing "no free IK branch" from "reachable but only badly".
  `--demo`: orbit a fake cup, targets colored by winning tier.
- `bench.py` — planner comparison, kept because the answer is CELL-DEPENDENT.
  Re-run when cell geometry, tool, or robot changes. `--compare` pits the
  asymptotically-optimal planners against the shipping tier 2.
- `execute.py` — the ONLY module that commands robot motion. `UR5eArm.execute`
  re-validates the path against the live world (stale path = refused path),
  requires current q == path[0] (<1.1 deg), sequential blocking moveJ blends
  off, safety-mode check between waypoints, stopJ on anomaly. Pendant must be
  in Remote Control mode. `preflight` is read-only; `bringup` = dashboard
  power on + brake release; `demo` = validated ±5 deg wrist_3 wiggle.

## Contracts & decisions
- TWO tiers over ONE boolean world. Tier 1 wins ~75% of moves at ~25 ms;
  tier 2 takes the rest at ~400 ms; anything else is an honest refusal.
- **Escalate on QUALITY, then refuse.** The gate (detour ≤ 2.5×) turns an
  unbounded risk ("the planner might emit something wild") into a bounded cost
  ("spend a bit more, then look from somewhere else"). A gate with no
  escalation is just a refusal machine; escalation with no gate never stops.
- **OMPL, not pyroboplan** (2026-08-19). pyroboplan takes ONE global
  `collision_distance_padding`, hardcoded at every call site. We need TWO
  margins (env 20 mm, self 5 mm — the UR5e's own design clearances at park are
  17–19 mm, so a 20 mm blanket margin reports the parked robot in
  self-collision). Expressing two margins through one number forced a second
  inflated "planning world", and the two worlds disagreed exactly at the
  boundary — which is where viewpoints live, since looking closely at a cup on
  a table means hugging the table. That seam is why `planning_world()`,
  `_sync_objects()`, `_escape_hug()` and the bridge ladder existed. OMPL takes
  a validity CALLBACK, so the planner explores exactly the set we accept: one
  world, seam and workarounds deleted.
- **Best-of-N, not an asymptotically optimal planner.** Benchmarked, 15 hard
  cases, equal budget (see bench.py):
      RRTC best-of-N 0.2s  15/15 solved, 0 hangs, detour med 1.13
      AORRTC         0.2s   7/15 solved, 2 hangs, detour med 1.03
  AORRTC (RA-L Dec 2025) makes tighter paths but solves 40–60% of queries and
  HANGS on up to 27% — hang rate GROWS with budget. It ignores its own
  termination condition (1 s budget observed running 9.5 min): OMPL's timed
  condition only works if the planner polls it, and a C++ `solve()` holding
  the GIL cannot be preempted from Python. Only a forked child + kill bounds
  it. Reliability beat the last few percent of quality.
- **OMPL's global RNG cannot be re-seeded** once sampling starts, so runs are
  not bit-reproducible. Best-of-N depends on exactly that variation; expect
  ±1 case of run-to-run variance in bench results.
- Never moveL between viewpoints: the chord passes through the object.
- cuRobo ruled out. VAMP ruled out for now — ships precompiled UR5, not UR5e
  (different link lengths); revisit only if rebuilt from our URDF.
- Execution via ur_rtde `moveJ`, blends off (`execute.py`). Conservative
  defaults: 0.25 rad/s, 0.5 rad/s², speed slider 25%.

## Removed
- `retract.py` (radial retract via an outer shell) — deleted 2026-08-19,
  measured **0/11** hard cases. Structural, not tunable: the r=0.50 m safe
  shell around a cup sitting 16 cm above a table is mostly INSIDE the table
  and behind the robot's own column, so "the shell is empty space" is false
  here; and its transit leg was still a joint-space lerp between configs on
  different IK branches. In git history if standoff geometry ever changes.

## Known / not done
- Tier 2 costs ~400 ms wall against a 0.2 s budget: the pruner and final gate
  validation sit outside the budget. Only fires on ~10% of moves, so left
  unoptimised.

## Does NOT belong here
- World modeling (cell/), viewpoint choice (view/), camera IO
  (perception/), loop orchestration (run/).
