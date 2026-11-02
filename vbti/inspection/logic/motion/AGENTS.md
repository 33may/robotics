# motion — motion solving and execution

## Purpose
Get the arm from viewpoint to viewpoint without hitting anything and without
ugly motion, then execute the validated path on the UR5e.

## Files
- `ik.py`: `UR5eIK`: EAIK all-branches analytic IK on NOMINAL DH, flange
  frame. Frame adapters verified vs pinocchio (<1e-9). ALWAYS filters EAIK's
  `is_LS` pseudo-solutions (cm-level error). Newton refine on the calibrated
  DH is a hook only (factory calibration not extracted).
- `direct.py`: tier 1: validated straight joint move + endpoint branch
  filter. No search: either the exact moveJ is provably clear or we escalate.
- `rrt.py`: tier 2: OMPL RRT-Connect, best-of-N within a time budget,
  `smooth.polish`, then a path-quality GATE (detour ≤ 2.5×). Goal is a SET
  (`ob.GoalStates` over every free IK branch). Escalates 0.2 s → 0.6 s → refuse.
- `smooth.py`: planner-agnostic post-processing. `partial_shortcut`
  straightens ONE joint at a time: full-DOF shortcutting cannot undo a single
  wrapped joint (every all-joint chord across the wrap collides).
- `plan.py`: the ladder: `plan_viewpoint()` = IK branches → tier 1 → tier 2.
  Returns `(path, report)`; `path=None` is an honest refusal and
  `report["reason"]` says "no free branch" vs "reachable only badly".
  `q_goal=` lands on an exact config (used by the transaction's hop).
- `transaction.py`: `plan_transaction`: a move between two ray addresses is
  three legs (`Legs(out, hop, slide_in, world)`): slide-out on the current ray,
  hop via the ladder (the only leg where the wrist may flip roll), slide-in on
  the target ray. Pinned linear search, two-pass (tier 1 sweep, then tier 2 on
  `MAX_LADDER_TRIES` candidates), last resort a plain ladder plan to the view.
  Design: `docs/blueprints/ray-views.md` §Move transaction.
- `execute.py`: the ONLY module that commands robot motion (`UR5eArm`;
  CLIs `preflight` read-only, `bringup` power on + brake release, `demo`
  ±5 deg wrist_3 wiggle).
- `bench.py`: planner comparison (`--compare` vs AORRTC/BIT*/SORRT*).
  Re-run when cell geometry, tool or robot changes: the answer is cell-dependent.
- `rtde_soak.py`: RTDE stream-death hunt (scenarios layered onto a monitored
  receive stream). Never commands motion; run with the app stopped.
- `wrist_watch.py`: live per-joint readout vs `logic/run/survey_pose.json`, for
  freedriving back to a plannable pose.

## Contracts & decisions
- TWO tiers over ONE boolean world (`logic/cell/world.py`). Tier 1 wins most moves
  at ~25 ms; tier 2 takes the rest at ~400 ms; anything else is a refusal.
- **Escalate on QUALITY, then refuse.** The detour gate bounds the risk of a
  wild path; escalation bounds the refusal rate.
- **OMPL, not pyroboplan.** We need two margins (env 20 mm, self 5 mm);
  pyroboplan has one global padding, which forced a second inflated planning
  world that disagreed with the validator exactly at the table boundary where
  viewpoints live. OMPL takes a validity CALLBACK: one world.
- **Best-of-N RRTC, not an asymptotically optimal planner.** On 15 hard cases
  RRTC best-of-N solved 15/15, AORRTC 7/15 with hangs (ignores its own
  termination; a C++ `solve()` holding the GIL cannot be preempted from
  Python, only a forked child + kill bounds it). `bench.py` forks for this reason.
- OMPL's global RNG cannot be re-seeded once sampling starts: runs are not
  bit-reproducible; expect ±1 case variance in bench results.
- Never moveL between viewpoints: the chord passes through the object.
- Slides are one IK branch: per-joint step ≤ `SLIDE_BRANCH_MAX` (pi/4).
- The legs carry the world they were validated against; execution
  re-validates against it.
- cuRobo ruled out. VAMP ships a UR5 (not UR5e) model; revisit only if
  rebuilt from our URDF.

## Execution (`execute.py`)
- `UR5eArm.execute(path, world, stop_event)` refuses (raises, no motion) on
  start mismatch > `START_TOL_RAD` (~1.1 deg), world refusal, or safety not
  NORMAL/RUNNING. The path runs as ONE async path-form moveJ blended at
  `BLEND_M` (5 mm, zero at short segments and the endpoint). Safety polled
  every 50 ms; `stop_event` → `stopJ`, report `{"stopped": True}` (normal
  outcome); `waypoints_done` undercounts by design.
- Pendant must be in Remote Control mode for motion; reads work in any mode.
- **The speed slider is code-authoritative.** `UR5eArm.__init__` writes
  `SPEED_SLIDER` on every connect, overwriting the pendant. Change the
  constant, not the pendant. Effective top speed = `SPEED_RAD_S` ×
  `SPEED_SLIDER` (0.25 rad/s × 1.0 now, ~14 deg/s vs a 50 deg/s pendant cap).
  Speed does not affect collision safety (validation is geometric); it costs
  stopping distance, so raise it in steps.
- The receive interface subscribes only to `RECV_VARIABLES` at 125 Hz: the
  default full subscription at 500 Hz overflows the socket buffer in ~0.2 s
  of scheduling stall and the controller closes the stream (`asio.misc:2 End
  of file`). Freshness is checked (`stale_window`); reconnects back off 2 s.

## Removed
- `retract.py` (radial retract via an outer shell): 0/11 hard cases. The safe
  shell around an object on a table is mostly inside the table and behind the
  robot column. In git history if standoff geometry changes.

## Known / not done
- Tier 2 costs ~400 ms wall against a 0.2 s budget (pruner and final gate sit
  outside the budget). Left unoptimised.

## Does NOT belong here
- World modeling (`logic/cell/`), viewpoint choice (`logic/view/`), camera IO
  (`logic/perception/`), loop orchestration (`logic/run/`).
