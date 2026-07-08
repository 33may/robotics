---
name: isaac-walk-physics-fidelity
description: Oli's Isaac walk failure. STANDING solved (dead-IMU fix → derive IMU from articulation root state). FORWARD WALK unsolved — serial ankle too soft fore-aft (pitch ×3 holds ~2s) but a LATERAL single-support instability remains that no ankle gain fixes. 2026-07-01 LAST MANUAL ATTEMPT: explicit-torque actuator model AND Jacobian-coupled ankle BOTH tested, BOTH worse than pitch ×3; achilles Jacobian coupling measured ≈0 (collapses to a diagonal already tested). VERDICT: reliable dynamic Isaac walk is LimX-gated; use kinematic-glide for the demo, dynamic walk in MuJoCo.
metadata:
  type: project
---

## ⛔ FINAL VERDICT (2026-07-01): both strongest structural leads fail — reliable Isaac dynamic walk is LimX-gated

Last MANUAL attempt (Anton's directive: pursue the 2 strongest structural leads, judge by
measurement, STOP if neither clears the 10 s bar). Ran 4 Isaac configs + 2 MuJoCo measurements
(`logic/simulation/walkmatch/NOTEBOOK.md` rows 15-17, F10-F12). **Trusted current baseline
re-confirmed:** `--control implicit --ankle-kp-scale 3 --ankle-effort 80`, vx=0.1 → upright ~2.0 s,
lateral fall ~2.25 s (reproduces 2026-06-25).

- **Lead A — explicit-torque actuator model (match LimX EFFORT-mode per-substep PD): DEAD.**
  `--control explicit --ankle-kp-scale 3` fell 0.25 s (bare serial ankle rings, F1); + `--armature on`
  (ankle rotor inertia 0.1845) stood cleanly at 0.5 s but toppled LATERALLY by 1.0 s. Both WORSE than
  implicit's 2.25 s. Isaac's implicit PhysX drive solves PD semi-implicitly → extra numerical damping
  that helps our marginal serial ankle; explicit removes it. Matching the training actuator did NOT help.
- **Lead B — Jacobian-coupled ankle (`K = J^T diag(kp) J`): premise FALSIFIED by measurement.**
  Measured the achilles Jacobian in MuJoCo (`walkmatch/ankle_jacobian.py`): off-diagonal coupling ≈0
  at home AND across the whole ankle range (only singular at roll=±0.437 = joint limit = a fall state).
  So the "coupled 2×2 PD" collapses to a DIAGONAL: pitch ×2.30, roll ×2.04. Tested it (#17): fell 1.25 s
  — WORSE than pitch ×3 (the roll ×2.04 hurts the lateral axis, matching F9). No coupling to add; the
  diagonal it reduces to is already worse than what we have. A coupled-matrix flag would be dead code.

**Root cause is now TWO layers:** (1) serial ankle fore-aft softness → forward runaway, FIXABLE via
`--ankle-kp-scale 3` (upright ~2 s). (2) Underneath, a LATERAL single-support instability that NO ankle
gain fixes (gain scaling, explicit actuator, Jacobian coupling all hit the same ~2 s wall). The SAME
brain walks in MuJoCo (real parallel ankle + full body/contact fidelity) → layer (2) is a serial-model
+ Isaac body/contact fidelity gap, not a controller-gain problem.

**Obs-fidelity ruled out for the lateral axis too (F13, `/tmp/latcmp.py`):** compared Isaac-vs-MuJoCo
roll axis (base roll, ankle-roll q/qdes, gyro_x) over the first 2 s. Base roll is small & comparable in
BOTH through ~1.5 s and early ankle-roll COMMANDS match → the policy gets good lateral obs; NOT a
hidden IMU/frame bug. The tip is abrupt at the FIRST swing step (~1.75–2.0 s): MuJoCo holds ±1.5°,
Isaac snaps −8°→−28°. So both the actuator/gain side AND the obs side are exonerated — the residual is
purely first-step single-support body/contact fidelity. Two independent confirmations of the gate.

**Recommendation (Anton's call):** (1) treat reliable dynamic Isaac walk as LimX-gated — when LimX
replies, ask for their IsaacLab/IsaacGym HU_D04 walk training config (contact params, foot collision
approximation, lateral regularization) = the missing fidelity. (2) For the reasoning demo, use
root-driven KINEMATIC GLIDE in Isaac (no dynamic balance); keep dynamic walk in MuJoCo. (3) Ship
`--ankle-kp-scale 3` as the documented best serial-ankle partial (~2 s upright); STOP gain tuning.
**Do not retry: explicit actuator, ankle roll/waist boosts, Jacobian coupling, big ankle armature,
MJCF closed-loop import** (all measured dead). The unexplored frontier (bigger than a gain, for a
LimX-informed effort): free A/B joints + software kinematic constraint, and a foot-collision audit
during single support. Related: [[walkmatch_actuator_id_harness]].

**UPDATE — free-A/B path BUILT + TESTED + measured (2026-07-01):** Anton chose to pursue the
free-A/B option. Implemented `--ankle-parallel` (opt-in): faithful dual-motor achilles emulation
on the serial ankle — motor-space PD (per-motor ±42 clip) via the linkage Jacobian Jᵀ + reflected
rotor inertia diag(Jᵀdiag(I)J)≈[0.424,0.376] as ankle armature; forces explicit control. New pure
module `logic/simulation/isaacsim/ankle_parallel.py` + 10 committed brain-env tests
(`tests/oli/world/test_ankle_parallel.py`); wiring in oli.py (`configure_parallel_ankle` +
apply_torque_isaac override) + sim_world_main `--ankle-parallel` + run_oli_sim passthrough.
**Result (NOTEBOOK F14): 1.5 s** — best explicit variant (holds solid to 1.0 s) but STILL < the
implicit pitch×3 baseline (2.25 s). Clean reason (no free lunch): the faithful ankle NEEDS the
reflected inertia to make explicit PD stable, but reflected inertia DEstabilizes the implicit drive
(C1: 0.75 s) — so faithful-ankle forces EXPLICIT body control, which is less stable than implicit in
Isaac. A hybrid would inherit C1's instability. So the ankle is now faithfully solved and STILL the
first-step lateral CONTACT gap dominates (F13). 10 s in Isaac is gated on body/contact fidelity (foot
collision audit needs the GUI — headless can't traverse colliders — or LimX's training config), NOT
the ankle. `--ankle-parallel` kept as the most-faithful ankle model to pair with that future fix.

## ✅ FORWARD-WALK ROOT CAUSE (2026-06-25): serial ankle back-drives — policy expects the PARALLEL achilles ankle

After the IMU fix (below) Oli STANDS solid but forward-walk still topples ~0.78 s. Full
sim-to-sim system-ID this session (tools in `logic/simulation/walkmatch/`) nailed it:

- **Actuator-ID (pinned-base, gravity-off step response):** Isaac LEG joints (hip/knee,
  direct-drive) track a step IDENTICALLY to MuJoCo (RMS 0.0004 rad, explicit+armature). So the
  leg actuator is faithful — NOT the problem. (The bare serial ANKLE is numerically unstable
  under the deploy's explicit gains: kd·dt/I≈2.4>2 — this is the "explicit → all joints jitter"
  Anton saw; the ankle goes unstable first and shakes the chain.)
- **Closed-loop trace (Isaac vs MuJoCo forward walk, same brain):** early commands are similar
  in both, but Isaac's body lurches 0.35 m forward (12× too far) while MuJoCo walks calmly.
  Foot trace: feet DON'T slip; the body tips forward, then takes runaway steps (0.27→1.0 m).
- **The smoking gun — ankle q vs qdes:** Isaac ankle pitch is **back-driven to −0.32 while
  commanded +0.21** (0.53 rad error); MuJoCo holds −0.17. The ankle sags into plantarflexion
  under load → planted foot pitches the body forward → runaway → fall. Effort cap (42→80)
  barely changed it (track err 0.239→0.232) → it's COMPLIANCE, not torque saturation.
- **Why:** the ankle (PR idx 4,5,10,11) and waist (13,14) are achilles **parallel** joints
  (config.yaml `limited_joint_indices:[4,5,10,11,13,14]`). The policy trains/deploys against
  the parallel ankle = **TWO motors per axis**. Our shipped USD has a single SERIAL ankle, so
  applying the per-motor gains directly makes it ~2× too soft on fore-aft → it sags.
  (Measured the achilles gear ratio in MuJoCo: r≈0.93 pitch / 0.99 roll → linkage is ~1:1, the
  authority comes from the dual motors, not a gear amplification. My initial ×8 guess was wrong.)

**Partial fix (sim-side, env-invariant, NO policy change):** scale the serial ankle pitch
kp+kd ×3 — `sim_world_main --ankle-kp-scale 3 --ankle-roll-scale 1 --waist-kp-scale 1
--ankle-effort 80` (implicit drive, armature off). Eliminates the fore-aft runaway → Oli holds
ROCK-SOLID 2.0 s at vx=0.1 (was 0.78 s; pg=(0,0,−1), gyro≈0). NOT a full walk: a lateral
step-failure remains (the first real step at ~1.85 s tips it; ankle roll saturates at its joint
limit ±0.436 trying to recover), and it's speed-sensitive (vx=0.3 → fore-aft runaway 0.64 s).

**Dead ends this session (don't retry):** boosting ankle ROLL or WAIST gain (×2/×4/×8) — all
WORSE than ×1 (lateral wants the natural compliance); reflecting rotor inertia ×g² onto the
stiffened ankle (armature 1.48) — fell at spawn (implicit drive holds high kp fine at
armature 0); ankle effort cap alone (42 or 80) — it's compliance not torque.

**Why gain-tuning can't finish it:** a serial joint + linear gains can't reproduce the
achilles' nonlinear kinematics + dual-motor authority. MuJoCo (real parallel ankle) walks
reliably with the SAME brain. Defaults UNCHANGED (all gain scales default 1.0; the fix is
opt-in via flags). Tools: `walkmatch/{spec,actuator_id_isaac,actuator_id_mujoco,compare}.py`,
env-gated `OLI_FOOT_TRACE` in sim_world_main, `/tmp/gear.py`. See
`logic/simulation/walkmatch/NOTEBOOK.md`.

## ❌ DEAD END (2026-07-01): direct MJCF-import of the achilles closed loop DIVERGES in PhysX

Re-ran the parallel-ankle path (Anton asked to retry). The stock Isaac MJCF importer DOES
build the closed loop — imports the HU_D04_01 MJCF as a 55-DOF articulation with the 6 `<connect>`
constraints as `/World/Oli/loop_joints/*` PhysicsSphericalJoints (4 ankle + 2 waist) and the rod
ball-joints as 3-DOF spherical joints. Harness fix needed: the importer stamps ArticulationRootAPI
on BOTH the MuJoCo `worldBody` wrapper (non-rigid → breaks the tensor view) AND `base_link` —
strip the worldBody one, root at `.../base_link/base_link`.

BUT the loop **diverges catastrophically on the FIRST physics steps, even gravity-OFF with 255
solver iters**: max|dq| goes 2548 → 1.9e5 → 3.9e6 → 2.6e10 → 4.5e17 → NaN in 5 steps. Worst DOF
is always a rod ball-joint. Cause: the achilles rods are LIGHT bodies; PhysX imports the soft
MuJoCo `<connect solref="0.001 1">` as a RIGID articulation loop joint, so the tiny initial
geometric mismatch → enormous corrective acceleration → positive feedback → blow-up. MuJoCo's
soft constraint absorbs this; PhysX's rigid loop joint amplifies it. More solver iters make it
WORSE, not better. This is a PhysX reduced-coordinate-articulation limitation (loops bolted on
as stiff maximal-coord constraints), NOT a tuning issue. **Don't retry the stock-importer loop.**
Spike: `logic/simulation/isaacsim/achilles_spike.py` (fixed-base loop-stability; `--free` to
stand), diagnostics `/tmp/spike{3,4,5}.py`.

**Remaining faithful options (Anton's call):** (1) ANALYTIC linkage emulation — keep the serial
ankle, compute the achilles Jacobian (geometry in `twisted_left_ankle_model.yaml`) and apply the
COUPLED joint torque (adds the pitch↔roll coupling + nonlinearity that per-axis gain-scaling
misses). (2) Drive A/B as free joints + impose the PR↔AB kinematic constraint in software each
step (no physical rods → no rigid-loop blow-up). (3) Soft/compliant loop joints (convert the 6
PhysicsSphericalJoints to compliant D6 drives) — unproven, may still be stiff. (4) Ship the ×3
serial partial + do locomotion in MuJoCo. Gain-scaling ×3 (holds 2.0 s) remains the best
shippable Isaac state today.

---

## ✅ ROOT CAUSE (2026-06-25): the IMU was dead — policy flew blind

`Oli.read_imu()` used Isaac's `IMUSensor.get_current_frame()`. That sensor does **NOT
update inside our manual `world.step()` loop** (it needs Isaac's sensor/omnigraph tick we
don't run) — it returns **identity orientation `[1,0,0,0]` and zero gyro forever**.
Proven by `logic/simulation/isaacsim/imu_probe.py`: under real gravity the base tilts to
**54°** (`get_world_pose`) while the IMU still reads upright. So the brain always got
`projected_gravity=[0,0,-1]` and `ang_vel=0` → the walk policy (a balance feedback loop)
never saw itself tipping → walked open-loop and toppled. This is **timing/physics
invariant**, which is exactly why armature, friction, restitution, solver iters, and the
implicit/explicit actuator ALL changed the fall-time but never the failure mode — every
one of those was a red herring; the sensor feeding the controller was unplugged.

MuJoCo reference (`logic/simulation/mujoco/imu_probe.py`, the path that walks) is perfect:
`Body_Quat`/`Body_Gyro` track every orientation/spin, projected_gravity == analytic.

**Fix:** derive the IMU obs from the articulation ROOT STATE, not the sensor (what
MuJoCo/legged_gym do): `quat = get_world_pose()` (probe-confirmed correct), body-frame
`gyro = quat_rotate_inverse(quat, get_angular_velocity())`, acc = gravity reaction
(unused by walk). Implemented in `Oli.read_imu` + `_quat_rotate_inverse`. Probes
(`imu_probe.py` in both sims) and `Oli.set_base_pose/set_base_velocity` are the harness.

Lesson: when a trained policy "walks a bit then falls, a bit off, immune to every dynamics
change", suspect the OBSERVATION VALUES the sim feeds it (esp. IMU frame/liveness), not the
dynamics. Verify obs against ground truth (get_world_pose) and a reference sim early.

**After the fix (2026-06-25):**
- **STAND (vx=0) now ROCK SOLID** — base holds (-0.05,+0.03,+0.90) for the full 15 s,
  pg=[0,0,-1], gyro≈0. Before the fix it toppled in ~1 s. Balance is SOLVED. Confirmed
  with both implicit and explicit+armature actuators standing fine. Obs trace
  (instrumented in `_log_state`): pg starts [0,0,-1] and tracks tilt; gyro sane rad/s.
- **FORWARD (vx>0) still over-drives** — vx=0.3 sprints ~2.3 m/s while near-upright ~0.4 s
  then pitches over; vx=0.1 drifts sideways & falls ~0.9 s. Fails identically with implicit,
  explicit+armature, and implicit+armature → NOT the actuator, NOT armature.

**Brain is 100% CORRECT — confirmed (2026-06-25):** Anton verified our exact brain WALKS
RELIABLY in the MuJoCo World (`p logic/simulation/mujoco/run_oli_mujoco.py --mode walk
--spawn-app`). So policy + obs + action mapping are all correct; the entire remaining
forward-walk failure is **Isaac sim fidelity during the dynamic single-support step**
(standing = static double support works; stepping = needs heel→toe foot roll). IMPORTANT:
all earlier "armature/friction/actuator = noise" verdicts were made with the DEAD IMU and
are VOID — re-test dynamics levers now that the policy is sighted, judged by real walking.

**MJCF reference (what MuJoCo walks on), to match in Isaac:** total mass 54.05 kg; joint
damping = **0.01** on every joint; frictionloss 0; armature legs 0.14125 / achilles+waist
0.18455 / arms 0.08867 / head+wrist 0.01532 (NOTE: MJCF serial ankle_pitch/roll armature
is **0** — the 0.18455 is on the achilles A joints; our ARMATURE_PR may mis-assign ankle).
Foot = flat **box sole** `left_foot` (0.13×0.047×0.01 m) + 3 contact spheres
heel/center/tip, friction [1.0, 0.3, 0.3], condim 3.

**USD audit:** the Isaac USD ALREADY HAS the same foot contact structure — `contact_foot_
heel/center/tip_{L,R}` bodies fixed to ankle_roll_link + the foot collider (refs
`/colliders/...` in the 25 MB base). So foot geometry isn't missing. Remaining suspects:
collision **approximation** (convex-hull-of-mesh vs MuJoCo's exact box → wrong contact
shape/roll), PhysX contact/solver params, joint damping (match 0.01). USD layer chain:
`HU_D04_01.usd → configuration/{sensor→physics(13KB, collision defs)→base(25MB, meshes)}`.

**Probes/tools built:** `imu_probe.py` (both sims), `Oli.set_base_pose/set_base_velocity/
base_world_quat_wxyz`, obs logging in `sim_world_main._log_state` (pg+gyro). `--control
explicit|implicit` (implicit default, smoother), `--armature` opt-in.

Next: audit Isaac foot collision approximation/contact params vs the MJCF box; set joint
damping 0.01; re-test walking (windowed). Deeper fallback: get the HU_D04 IsaacLab/IsaacGym
training env config to replicate exactly.

**Systematic-debugging ledger (2026-06-25, post-IMU-fix, policy SIGHTED):**
- mass: Isaac 52.9 kg vs MJCF 54.05 — ~2% off, NOT the issue.
- actuator (implicit / explicit+armature / implicit+armature): all dive forward → NOT it.
- armature inject: doesn't fix forward walk → NOT it.
- obs (pg + gyro): proven correct → NOT it.
- **TOOLING WALL:** the robot's colliders do NOT appear in a headless `world.stage`
  traversal (`total colliders: 1` = only the ground) — yet it stands, so collision IS
  active in PhysX. The foot collision is loaded from the base USD's `/colliders` scope as
  a payload PhysX consumes but USD-Traverse doesn't expose. So the foot-collider SHAPE
  audit can't be done by USD traversal headless. Use the Isaac **GUI** (open the USD,
  toggle collision viz, watch the foot during a step) or the PhysX collision-shape API.
- **LEADING hypothesis (not yet confirmed):** foot can't roll heel→toe through single
  support (standing=double-support works; stepping fails). Confirm visually in the GUI.

Audit tooling: `logic/simulation/isaacsim/model_audit.py` (mass/armature/colliders),
`logic/simulation/mujoco/imu_probe.py` printed the MJCF reference (foot box + 3 contacts,
damping 0.01).

**POLICY TRAINED IN ISAAC ON A SERIAL ANKLE — confirmed (2026-06-25).** LimX's own doc
(`humanoid-rl-deploy-python/doc/parallel_joint_mapping_en.md`): *"Training environments in
IsaacLab/IsaacGym use PR space joint definitions (e.g. ankle_pitch_joint); trained policies
output PR-space actions."* And there's a dedicated **`HU_D04_01_rl.urdf`** (the RL training
asset, in `oli-main-software/install/etc/HU_D04_description/urdf/`) — **serial ankle, ZERO
achilles**, ankle `effort=42, velocity=13.6`, knee `effort=140, velocity=5`. So:
- The achilles A/B linkage is **deploy-only** (real robot + MuJoCo). The policy NEVER saw
  it. → the achilles-loop-in-Isaac idea is the WRONG direction (would be *less* faithful).
- Our serial-ankle USD is the right STRUCTURE. The real reference is **their IsaacLab
  serial-ankle setup**, not MuJoCo (which is the deploy model).
- The ankle whip is a serial-ankle **actuator-param** gap (armature/effort/velocity), not
  a missing linkage.
- Tried rebuilding from **`HU_D04_01_rl.urdf`** (`build_rl_usd.py` → `HU_D04_01_rl.usd`,
  31 DOF, names match PR_ORDER). **RESULT: WORSE — it can't even STAND** (vx=0: settles at
  z=0.899 then pitches ~90° backward, down by t=0.25 s), where the **shipped USD stands
  rock-solid 15 s**. Cause: Isaac's stock URDF importer **mangles the canted hip-pitch axis**
  (`0 0.906 -0.422`; it warns "Joint Axis not body aligned… reorienting bodies"). LimX's
  shipped USD handles that correctly. **CONCLUSION: the shipped `HU_D04_01.usd` is the
  CORRECT/most-faithful Isaac model** (it's effectively what LimX trained on — it stands).
  Reverted `Oli.DEFAULT_USD` to it. The URDF-reimport path is a dead end (don't retry).
- So the forward-walk whip is a sim-fidelity issue on the CORRECT model, not a wrong-model
  problem. Next concrete lever: the shipped-USD ankle effort/velocity limits vs the training
  URDF (`effort=42, velocity=13.6`) — if they differ, set the training values on the ankle
  (runtime `set_max_efforts` + velocity limit) and retest; that's principled + keeps the
  env-invariant model. Else: the harness (match ankle response to MuJoCo).

**Achilles-loop spike result (`achilles_spike.py`):** Isaac's MJCF importer DOES build the
closed loop — creates `/World/Oli/loop_joints/..._achilles_rod_link` as PhysicsSphericalJoint
(4 loop joints, both ankles). So PhysX *can* hold the loop. But per the above, we don't want
it. (Spike's stability check errored on wrong articulation root `worldBody` — never got the
verdict; moot now.)

---

## (historical investigation that led here)

The walk ONNX makes Oli topple at ~1.0–2.5 s in our Isaac World (zero cmd or forward),
no matter the spawn height, loop pacing, or render rate. The brain side is faithful
(kp/kd, action_scale, default_angle, decimation, effort clamp all match
walk_param.yaml / policy_runner). The divergence is **World-side physics fidelity**.

**Latency falsified (2026-06-24):** lock-step pacing (World blocks per brain cmd, steps
exactly decimation=10 ticks → ticks/cmd=10.0, zero sensor→actuator latency) STILL
topples ~1.0–1.5 s — *faster* than windowed free-run (which walked ~1.5 m first). So
timing is NOT the dominant cause. The fall is dynamics, not loop structure.

**Armature=0 — CONFIRMED a real lever (2026-06-25):** the USD ships joint armature
(rotor inertia) = 0 on all 31 joints; the policy was tuned in IsaacLab against real
rotor inertia. A stiff drive (legs kp≈139) on a massless rotor at 1 kHz buzzes into
divergence — timing-independent, why lock-step didn't help. **Injecting the MJCF armature
measurably helped:** base-z held 0.88 (fully upright) through t=1.0 s vs cratering to
0.39 by t=1.0 s without it. It no longer buzz-diverges. BUT it still topples at ~1.0–1.5 s
— now a *balance/traction* failure (x drifts backward, no commanded forward progress),
not a numerical one. So armature is necessary but not sufficient; keep it ON.

Authoritative armature values = HU_D04_01 MJCF
(`vendor/humanoid-mujoco-sim/humanoid-description/HU_D04_description/xml/HU_D04_01.xml`):

| group | armature (kg·m²) |
|---|---|
| hip ×3 + knee | 0.14125 |
| ankle pitch/roll (= serial form of achilles A/B) | 0.1845504 |
| waist yaw/roll/pitch | 0.1845504 |
| shoulder ×3 + elbow | 0.0886706 |
| head + wrist ×3 | 0.0153218 |

Injected at runtime (no USD edit) like gains: `articulation_view.set_armatures((1,31))`,
Isaac order. In-repo as `ARMATURE_PR` (built by joint name) in `sim_world_main.py`,
permuted PR→Isaac by SimComm, applied via `Oli.set_armature` before the crouch settle.
Default ON; `--no-armature` for A/B. Drift-guarded by `tests/oli/world/test_armature.py`.

**Ground contact was wrong (found 2026-06-25):** Isaac's stock `add_default_ground_plane`
is `static_friction=0.5, dynamic_friction=0.5, restitution=0.8` — a low-grip, **bouncy
(0.8)** floor. IsaacLab trains on **1.0 / 0.0**. The trampoline floor is a prime suspect
for the post-armature ~1 s balance loss (feet slip + bounce on footstrike). Now set via
`sim_world_main` args `--ground-friction` (default 1.0) and `--ground-restitution`
(default 0.0); A/B back with `--ground-friction 0.5 --ground-restitution 0.8`. NOTE: this
sets the GROUND material; foot collider material may still combine — if partial, set the
foot material too. (`set_friction_coefficients` on the articulation is JOINT friction,
NOT contact.)

**Solver iters — DON'T lower position iters.** Velocity iters = 1 (ours) vs 4 (IsaacLab):
bump via `--solver-vel-iters 4`. Position iters are already 32 (> IsaacLab's 4) — leaving
them is fine; **lowering to 4 regressed badly** (softer contacts → lunge-and-fall, base-z
0.19 at t=0.5 s). So only ever bump vel iters; never pass `--solver-pos-iters 4`.

Effort clamp 140/42/19 N·m present in USD drive (✅). IsaacLab config: dt=0.005 (200 Hz),
decimation=4, q_des = default_pos + 0.25·action, ImplicitActuatorCfg (✅ implicit matches).

Debug order (one variable at a time, all on top of armature=ON):
1. armature only → held 1 s, still falls (done).
2. + ground friction 1.0 / restitution 0.0 (now the default) → under test.
3. + solver **vel** iters 4 (NOT pos).
4. + foot collider material 1.0 if ground alone insufficient.
Run via `run_oli_sim.py` (armature + ground 1.0/0.0 are the defaults now).

**TRON1 IsaacGym comparison + explicit-torque actuator (2026-06-25):** cloned LimX's
`tron1-rl-isaacgym` (legged_gym; the official training stack). Findings:
- LimX trains with **explicit torque** control: `default_dof_drive_mode = EFFORT`, sim PD
  gains 0, `τ = kp(action·action_scale + default − q) − kd·q̇` recomputed EVERY physics
  substep, clipped to torque_limits, pushed as raw effort. We used the implicit PhysX
  drive — the one structural actuator difference.
- TRON1 config: `action_scale 0.25`, `control_type "P"`, `sim.dt 0.0025 × decimation 8`
  → 50 Hz; **`asset.armature = 0.0`** (so armature was never the lever — matches Anton's
  "fall-time is noise"). Our HU_D04 deploy `walk_param.yaml`: decimation 10, per-joint
  action_scale (legs 0.2511 / ankle-waist 0.1121 / head-wrist 0.3141 / arm 0.12), obs
  scales ang_vel 0.25 / dof_pos 1.0 / dof_vel 0.05, obs 102×5=510.
- **Brain verified faithful** to the deploy line-by-line: newest-first history, obs order/
  scales, per-joint action_scale, `q_des=action·scale+default`, last-action aliasing.

**Implemented explicit per-substep torque PD** (`Oli.set_effort_mode/set_command_isaac/
apply_torque_isaac`, `pd_torque`, `WorldComm.set_command`; `--control explicit|implicit`,
default explicit; armature default OFF). First cut **blew up** (base flew 8 m in 0.5 s) —
the missing piece was legged_gym's **per-substep clip**; added clip to the USD effort
limits (`get_max_efforts`, confirmed sane 19–140 N·m). With clip it's STABLE but **still
topples ~0.5 s** (z 0.90→0.17). So reproducing the TRON1 actuator did NOT fix walking.

**Net (2026-06-25): eliminated latency, armature, ground friction/restitution, solver
iters, AND the actuator.** Brain is faithful. Remaining unverified link = the OBSERVATION
VALUES the World feeds the policy (IMU frame: projected_gravity / gyro convention, joint
q/dq sign). Next: instrument the settled-crouch obs — projected_gravity must be ≈[0,0,−1]
upright; if not, the IMU frame is the bug. This supersedes the implicit-drive note in
[[isaac-pd-implicit-drive]] (explicit IS what training uses; it just isn't sufficient).

Related: [[isaac-pd-implicit-drive]], [[walk_policy_obs_builder_fidelity]],
[[isaac-oli-stand-spawn-height]], [[vendor_humanoid_mujoco_sim]],
[[feedback_reliable_walking_not_falltime]], [[project_invariant_oli_interface]].
