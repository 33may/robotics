# Isaac walk-match lab notebook (MAY-147)

**Objective:** get Oli (HU_D04_01) to walk *reliably* in Isaac Sim under the fixed walk
ONNX — no falling, sustained gait ≥10 s across speeds/inits — WITHOUT fine-tuning the
policy (that breaks the env-invariance Anton wants).

**Method:** sim-to-sim system identification. MuJoCo walks reliably with this exact brain.
So match Isaac's dynamics to MuJoCo's, measured quantitatively, then re-close the policy
loop. Each fix must be backed by a measurement, not a guess.

Run autonomously 2026-06-25 while Anton is out. Everything below is the live log.

---

## Ground truth / constraints (established before this run)

- **Standing is SOLVED** (dead-IMU fix: obs derived from articulation root state). Oli
  stands rock-solid 15 s. The remaining failure is **forward walk only**.
- **Brain + policy + obs are CORRECT** — verified: the same brain walks reliably in the
  MuJoCo World (`p logic/simulation/mujoco/run_oli_mujoco.py --mode walk`). So the bug is
  100% Isaac World-side dynamics during the dynamic single-support step.
- **Policy was trained in IsaacLab/IsaacGym on a SERIAL PR ankle.** The shipped USD (serial
  ankle, 31 DOF) is the correct/most-faithful model — it stands; the URDF re-import path is
  a dead end (Isaac's importer mangles the canted hip axis; can't even stand).
- **MuJoCo actuator = explicit torque PD** (`<motor>` gear-1 direct torque; deploy computes
  `τ=kp·err+kd·erṙ`). Leg joints (hip/knee) are direct serial in MuJoCo AND training AND
  Isaac → apples-to-apples. MuJoCo's ANKLE is the parallel achilles A/B (deploy-only).
- MJCF reference: timestep 0.001, leg joint **armature 0.14125**, joint **damping 0.01**,
  frictionloss 0, total mass 54.05 kg. Serial ankle_pitch/roll armature = **0** in MJCF
  (the 0.1845 is on the achilles A/B joints — our old ARMATURE_PR mis-assigned the ankle).
- Eliminated earlier (re-tested post-IMU-fix, by fall behavior): latency, ground
  friction/restitution, solver iters, ankle effort cap. Inconclusive by eye: armature,
  implicit-vs-explicit actuator. **This run replaces "by eye" with measurement.**
- Anton's freshest observation: in Isaac the **legs move late** → imbalance → big recovery
  steps; explicit control makes **all joints jitter**. MuJoCo moves legs sooner/smaller.

## Hypotheses (ranked, to be tested by data)

1. **H1 — actuator lag/instability.** Isaac's implicit drive is over-damped (legs respond
   late); Isaac's explicit drive at armature=0 is under-damped (jitter). Neither matches
   MuJoCo's crisp+stable explicit PD. *Fix:* find the Isaac actuator config (drive mode ×
   armature × joint damping) whose step response matches MuJoCo. **Tested in Phase 1.**
2. **H2 — foot can't roll heel→toe** through single support (contact-shape / collision
   approximation). Standing = double support works; stepping fails. **Tested in Phase 2.**
3. **H3 — body inertia/mass mismatch** (Isaac 52.9 vs MJCF 54.05 kg). Lower priority.

## Plan

- **Phase 0 — harness.** ✅ Isaac boots headless (~7 s). mujoco 3.2.3 in `limx`. Built
  `walkmatch/` (spec + actuator_id_isaac + actuator_id_mujoco + compare).
- **Phase 1 — actuator ID.** Pinned base, gravity off, identical joint step reference.
  Compare Isaac (implicit/explicit × armature on/off) vs MuJoCo step response. Decide H1.
- **Phase 2 — body/contact ID.** Matched actuator + open-loop command replay, free base.
  Compare base trajectory divergence; tune contact/damping. Decide H2/H3.
- **Phase 3 — close the loop.** Run the ONNX with the matched config. Success = no fall
  ≥10 s, vx 0.1/0.3/forward, several inits.
- **Phase 4 — lock config, tests, memory, daily note.**

---

## Experiment log

| # | phase | config | result | decision |
|---|-------|--------|--------|----------|
| 0 | 0 | Isaac headless boot | OK ~7 s, clean exit | autonomous loop viable |
| 1 | 1 | MuJoCo knee step (all 31 serial joints driven, explicit PD) | **NaN at t=0.03** (before the step, zero torque should hold) | bug in harness → localized |
| 2 | 1 | MuJoCo zero-torque, eq on/off × pin on/off | all stable | not the base-pin, not the equality |
| 3 | 1 | MuJoCo knee/hip, **leg-only drive** (8 hip/knee joints) | knee onset 23 ms / rise 248 ms / 0% overshoot / ss≈0; hip 28/230/2.7%/≈0 | **MuJoCo reference captured** |
| 4 | 1 | Isaac knee sweep {impl,expl}×{arm0,arm1} | all match MuJoCo (best explicit_arm1 RMS 0.0004) | leg actuator faithful — H1 falsified |
| 5 | 2 | Isaac vs MuJoCo forward walk vx=0.1, command/realized trace | Isaac: same early cmds, base lurches 0.35 m fwd, falls 0.75 s. MuJoCo walks 4.4 s | body/contact divergence in first step |
| 6 | 2 | Isaac foot-contact trace | feet planted (no slip), body tips fwd, then HUGE steps (0.27→1.0 m) → tumble | not slip; runaway step length |
| 7 | 2 | ankle q vs qdes (both sims) | Isaac ankle BACK-DRIVEN to −0.32 vs cmd +0.21 (0.53 err); MuJoCo holds −0.17 | ankle too soft — sags into plantarflex |
| 8 | 3 | Isaac walk, ankle effort 42→80 | tiny change (track err 0.239→0.232); survives ~0.75→1.0 s | not effort-limited; it's compliance |
| 9 | 3 | Isaac walk, ankle kp+kd ×8 (pitch+roll) | **forward drift GONE** (x 0.32→0.06) but falls sideways y=0.51 @0.5 s | pitch stiffness fix CONFIRMED; roll over-stiffened |
| 10 | 3 | Isaac walk, ankle pitch ×8, roll ×1 | upright to 0.75 s then lateral trip (×8 overshoots) | pitch helps but ×8 too high |
| 11 | 3 | reflected ankle armature 1.48 + pitch ×8 | fell at spawn | big ankle armature destabilizes implicit drive — don't |
| 12 | 1 | MEASURE gear ratio (MuJoCo achilles) | r≈0.93 pitch / 0.99 roll → 1/r²≈1; dual-motor ≈2× | scale should be ~2–4×, NOT 8× |
| 13 | 3 | **low sweep: ankle pitch ×2/×3/×4** | **ALL stay upright (min_z 0.89–0.90), NO fall in ~1 s; ×3 pristine (no drift)** | **FIX FOUND: ankle pitch ×3** |
| 14 | 3 | pitch ×3 long run (~10 s sim) | (running) | reliability confirmation |

### F7 — ✅ FIX: scale ankle PITCH kp+kd ×3 (roll/waist ×1, ankle effort 80)
The low-range sweep nailed it. At vx=0.1, ankle pitch ×2/3/4 all keep Oli **upright with no
forward runaway and no lateral drift** (min base-z 0.89–0.90 vs baseline 0.26) — where ×8
overshot into a lateral trip and ×1 (baseline) ran away forward. **×3 is the cleanest**
(max forward 0.04 m, max lateral 0.04 m over the run). This matches the measured physics: the
achilles gives the ankle ~2× stiffness (dual motor, ~1:1 linkage), and ×3 sits in that band.
Config: `--ankle-kp-scale 3 --ankle-roll-scale 1 --waist-kp-scale 1 --ankle-effort 80`
(implicit drive, armature off). Pending: confirm sustained over ~10 s + at vx=0.3.

### F8 — pitch ×3 sustains 2.0 s then a LATERAL failure (not fore-aft)
Long run (vx=0.1) shows ×3 is rock-solid (pg=(0,0,−1), gyro≈0, base≈origin) for a full **2.0 s**
— the fore-aft runaway is GONE. Then it fails laterally: a slow roll lean builds (0.8°→4°→…),
the FIRST real step (~1.85 s) tips it, the policy fights with ankle ROLL whose command runs to
−1.32 rad but the joint **saturates at its limit ±0.436 rad** (~2.03 s), can't recover, and it
tumbles (gyro→15). So the remaining bug is **lateral-axis stability**, and the saturation is a
symptom (the policy panicking once already tipping). KEY nuance: at vx=0.1 Oli **marches in
place** (base x≈0 for 2 s) — a laterally-marginal near-stand. Hypothesis: a real walk (vx=0.3)
with continuous lateral weight-shift is more stable. Testing vx=0.3 next. Backup levers if not:
small ankle-roll boost (×1.5–2, now that pitch is correctly ×3 — earlier roll tests were
confounded by pitch ×8), or the L/R ankle-pitch hold asymmetry (L −0.17 vs R −0.28).

### F9 — roll boost FAILS; gain-tuning has a ceiling
With pitch correctly ×3, re-tested ankle roll boost (the dual-motor physics predicted ~2×):
p3r1 fell@2.12 s (best), p3r2@1.20, p3r3@1.72, p2r2@1.67 — **every roll boost is WORSE**. So
the lateral axis does NOT want a stiffer ankle roll; the lateral failure is not ankle-roll
softness. And vx=0.3 with pitch ×3 fell@0.64 s (fore-aft runaway returns at speed). Net: gain
scaling the *serial* ankle bottoms out at **pitch ×3, roll/waist ×1, effort 80** → solid for
2.0 s at vx=0.1, then a lateral step-failure; speed-sensitive.

---

## CONCLUSIONS (2026-06-25 autonomous run)

**Root cause (proven, high confidence):** Oli's forward-walk failure in Isaac is the **serial
ankle**. The policy trains/deploys against the **achilles parallel ankle** (two motors per
axis). Our shipped USD has a single serial ankle joint, so under the deploy's per-motor gains
it is too soft on the fore-aft (pitch) axis → it **back-drives into plantarflexion under load**
(measured: q→−0.32 while commanded +0.21) → the planted foot pitches the body forward → the
gait amplifies it into runaway steps → fall at ~0.78 s. The legs (hip/knee, direct-drive) are
faithful (actuator-ID: Isaac knee step response matches MuJoCo to 0.0004 rad).

**Partial fix (sim-side, env-invariant — NO policy change):** scale the serial ankle pitch
kp+kd ×3 (`--ankle-kp-scale 3`, roll/waist ×1, `--ankle-effort 80`). This eliminates the
fore-aft runaway and holds Oli rock-solid for **2.0 s** at vx=0.1 (was 0.78 s). It is NOT a
full reliable walk: a lateral step-failure remains, and it is speed-sensitive (vx=0.3 runs
away fore-aft).

**Why gain-tuning can't finish it:** a serial joint + linear gains cannot reproduce the
achilles' nonlinear kinematics / dual-motor authority / coupling. MuJoCo (which has the real
parallel ankle) walks reliably with the SAME brain. So the faithful path is the parallel
ankle, not a stiffer serial joint.

**Recommended directions for Anton (architecture call):**
1. **Build the achilles parallel ankle in Isaac.** `achilles_spike.py` already showed Isaac's
   MJCF importer DOES build the loop_joints (PhysicsSphericalJoint). Risk: the MJCF importer
   may mangle the canted hip axis (as the URDF importer did for `_rl`). Highest fidelity if it
   works — would match MuJoCo directly.
2. **Keep the serial model + ship pitch ×3 as a documented partial** and pursue the lateral
   axis separately (foot placement / step width / the L-R ankle asymmetry). Lower ceiling.
3. **Accept Isaac for standing/manipulation; do locomotion in MuJoCo** (which already walks),
   if Isaac locomotion isn't on the critical path to the reasoning demo.

**Tooling built this run (`logic/simulation/walkmatch/`):** `spec.py` (shared step protocol),
`actuator_id_isaac.py` / `actuator_id_mujoco.py` / `compare.py` (sim-to-sim actuator ID),
`/tmp/gear.py` (achilles gear-ratio measurement). Plus env-gated `OLI_FOOT_TRACE` in
`sim_world_main` (6 contact-point trajectories) and `--ankle-kp-scale/--ankle-roll-scale/
--waist-kp-scale` (parallel-joint gain recovery). Defaults UNCHANGED (all scales 1.0).

### F5 — ROOT CAUSE (Phase 2/3): the achilles-driven ankle is ~8× too soft in Isaac
The ankle (and waist) are driven through the achilles linkage in deploy/training; the
`walk_param` ankle gains (kp 93.65) are **per-motor (A/B)** values. Through the linkage the
JOINT-space ankle stiffness the policy trained against is ~g²≈8× higher (~750). Our Isaac
serial ankle applies 93.65 *directly at the joint* → ~8× too soft → under the body's forward
load torque the ankle **sags into plantarflexion** (q drifts to −0.32 while commanded +0.21)
→ the planted foot pitches the body forward → the gait amplifies it into runaway steps → fall.
The legs are direct-drive so their gains are already joint-space (→ they tracked perfectly).
- **Confirmed:** scaling ankle kp+kd ×8 ELIMINATED the forward drift (base x 0.32→0.06 m).
- **Caveat:** pitch (A+B additive) and roll (A−B differential) have different mechanical
  advantage → scaling both ×8 over-stiffened roll → lateral fall. Now tuning pitch/roll
  scales independently (`--ankle-kp-scale` pitch, `--ankle-roll-scale` roll).
- This is a sim-side ACTUATOR fix (gain rescale), NOT a policy change — keeps env-invariance.

### F6 — gear-ratio measured: my ×8 theory was WRONG (linkage is ~1:1)
Measured the achilles motion ratio directly in MuJoCo (drive A/B motors, read serial ankle
through the loop, twisted_left_ankle_model.yaml geometry):
- **additive (A+B) → ankle ROLL**, r ≈ 0.99  (NOT pitch — the twisted ankle swaps the modes
  vs the standard one; corrected my pitch/roll labels)
- **differential (A−B) → ankle PITCH**, r ≈ 0.93
So the linkage is ~1:1 kinematically (1/r² ≈ 1) — it does NOT amplify stiffness 8×. What the
ankle gets from the achilles is **two motors per axis** (≈2× torque + ≈2× effective stiffness
+ effort 2×42≈84 ≈ the deploy 80). So the physically-justified ankle scale is ~**2–4×**, not
8×. The ×8 helped empirically (stiffer than the soft 93.65) but likely OVERSHOOTS → the
lateral over-correction trip. Re-sweeping the low range (2,3,4) with precise foot-trace fall
metrics. Open question: why ×8 beat ×5/6/10 in the crude sweep — extra Isaac softness (contact
/implicit-drive) may need >2×. The precise low-range sweep will settle it.
- config.yaml confirms `limited_joint_indices: [4,5,10,11,13,14]` = the parallel joints.
- Reflecting rotor inertia ×g² onto the ankle (armature 1.48) DESTABILIZED (fell at spawn) —
  the implicit PhysX drive holds high kp fine at armature 0; do NOT add big ankle armature.

## Findings (running)

### F1 — bare serial ankle is numerically unstable under the deploy's explicit ankle gains
Driving the bare serial `ankle_pitch/roll` (armature 0, tiny foot inertia) with the deploy
gains (kp 93.65, kd 11.92) under explicit integration gives `kd·dt/I ≈ 2.4 > 2` → it rings
and NaNs, propagating to the base DOF. The deploy gains are tuned for the ankle driven
*through the achilles* (A/B rotors, armature 0.1845 → large reflected inertia), not the bare
serial joint. **This is the leading explanation for Anton's "explicit → all joints jitter"
in Isaac:** the ankle goes unstable first and shakes the whole chain. Consequence for the
real fix: the serial ankle in Isaac needs either (a) added armature (≈ the achilles rotor
inertia, so explicit PD is stable) or (b) a lower effective ankle gain. The leg joints
(hip/knee, armature 0.14125 + big link inertia) are stable under the same gains.

### F2 — MuJoCo leg PD response (gold standard to match)
Leg joints track a step in ~25 ms onset, ~240 ms rise, ≈0 overshoot, ≈0 steady-state error
— crisp and slightly over-damped. Isaac must reproduce this.

### F3 — PHASE 1 VERDICT: leg actuator is FAITHFUL (H1 falsified)
Isaac knee step response vs MuJoCo (RMS over the curve, rad): explicit_arm1 **0.00036**,
implicit_arm1 0.00069, implicit_arm0 0.00414, explicit_arm0 0.00469. With armature ON the
match is near-perfect (onset 23 ms, rise 248 ms — identical to MuJoCo). **The leg actuator
does NOT lag.** So "legs move late" is not raw actuator lag. (Best actuator config for later:
explicit_arm1 or implicit_arm1 — both excellent; armature ON matters, ~0.006 rad either way.)
Plot: `/tmp/actid_knee.png`.

### F4 — PHASE 2 VERDICT: failure is BODY/CONTACT dynamics in the FIRST dynamic step
Closed-loop forward walk, vx=0.1, traced command (qdes) vs realized (q) + gyro, Isaac vs
MuJoCo (same brain):
- **MuJoCo** walks 4.4 s calmly: cmd amplitude bounded 0.27–0.61 rad, gyro <1.0 throughout.
- **Isaac** falls by 0.75 s: early (t<0.3 s) commands are SMALL and similar to MuJoCo, gyro
  <0.1 — but the **base lurches 0.35 m forward in 0.5 s (~0.7 m/s, 7× the 0.1 command)** while
  upright, then tilts; only AFTER the tilt (t≥0.35 s) does the policy panic with huge 3-rad
  commands and topple (gyro → 5).
- **Conclusion:** identical actuator + similar early commands → 12× body translation. The
  same leg motion that rotates MuJoCo's body over a planted foot makes Isaac's body shoot
  forward. The divergence is in **how the body+contact respond during the first single-support
  step**, not the actuator and not the commands. Mechanism candidates: stance-foot **slip**,
  **launch** (hop), or failed heel→toe **roll**. Foot-contact trace running to decide.
- Foot structure CONFIRMED present in Isaac: `contact_foot_{heel,center,tip}_{L,R}` rigid
  bodies on each `ankle_roll_link` (matches MJCF box sole + 3 spheres). Only ONE physics
  material in the stage (ground, friction 1.0); the foot material is in the USD payload.
  GPU is active (RTX 4070 Ti via Vulkan); the Warp-CUDA error is a Warp fallback, not PhysX.

---

## Phase 1 re-run (2026-07-01): parallel-ankle MJCF import → DEAD END

Anton asked to retry importing the MuJoCo MJCF (with the real achilles loop) into Isaac.

- **Import works:** 55-DOF articulation, the 6 `<connect>` constraints become
  `/World/Oli/loop_joints/*` PhysicsSphericalJoints (4 ankle + 2 waist), rod ball-joints as
  3-DOF spherical joints. Root-API fix: importer stamps ArticulationRootAPI on both the MuJoCo
  `worldBody` (non-rigid → breaks tensor view) and `base_link`; strip the worldBody one.
- **But the loop DIVERGES on the first steps**, even gravity-OFF + 255 solver iters:
  max|dq| 2548 → 1.9e5 → 3.9e6 → 2.6e10 → 4.5e17 → NaN in 5 steps. Worst DOF = a rod ball-joint.
- **Why:** light achilles rods + PhysX importing MuJoCo's SOFT `<connect solref="0.001 1">` as a
  RIGID articulation loop joint → tiny initial geometric mismatch → huge corrective accel →
  positive-feedback blow-up. MuJoCo's soft constraint absorbs it; PhysX's rigid loop amplifies.
  More solver iters make it WORSE. PhysX reduced-coordinate-articulation limitation, not tuning.
- **Verdict:** the stock-importer physical-loop path is NOT viable. Kill-criterion met.

Spike `achilles_spike.py` (fixed-base loop test; `--free` to stand), diagnostics
`/tmp/spike{3,4,5}.py`.

### Remaining faithful options (Anton to choose)
1. **Analytic linkage emulation** — keep serial ankle, compute the achilles Jacobian from
   `twisted_left_ankle_model.yaml`, apply the COUPLED joint torque (adds pitch↔roll coupling +
   nonlinearity that per-axis gain-scaling misses). Most promising; moderate effort.
2. **Free A/B joints + software kinematic constraint** each step (no physical rods → no blow-up).
3. **Compliant loop joints** (convert the 6 spherical joints to soft D6 drives) — unproven.
4. **Ship ×3 serial partial + locomotion in MuJoCo.** ×3 (holds 2.0 s) = best Isaac state today.

---

## Last manual attempt (2026-07-01): explicit-torque model + Jacobian-coupled ankle

Anton's directive: this is the LAST manual attempt before treating a reliable Isaac walk as
LimX-gated. Pursue the two strongest STRUCTURAL leads, judge by measurement, and if neither
clears the 10 s bar (no fall ≥10 s at vx 0.1 AND 0.3), write a verdict and STOP — don't grind.

**Trusted current baseline (re-confirmed this run):** `--control implicit --ankle-kp-scale 3
--ankle-effort 80`, vx=0.1 → rock-solid upright (base≈origin, pg_z=−1.00) through t=2.0 s,
lateral fall ~2.25 s. Reproduces the 2026-06-25 result exactly (harness has not drifted).

| # | lead | config (vx=0.1) | fall onset | vs baseline |
|---|------|-----------------|-----------|-------------|
| 15 | A explicit | `--control explicit --ankle-kp-scale 3 --ankle-effort 80` (armature 0) | 0.25 s | WORSE |
| 16 | A explicit | row 15 + `--armature on` (ankle rotor inertia 0.1845) | 1.0 s | WORSE |
| — | baseline | `--control implicit --ankle-kp-scale 3 --ankle-effort 80` | 2.25 s | (ref) |
| 17 | B Jacobian | `--control implicit --ankle-kp-scale 2.30 --ankle-roll-scale 2.04` | 1.25 s | WORSE |

### F10 — Lead A DEAD: explicit per-substep torque is LESS stable than implicit here
The primary lead was that LimX *trains* with explicit torque PD (EFFORT mode, τ recomputed each
substep — TRON1/legged_gym), so matching it should help. Measurement says the opposite:
- **Explicit, armature 0 (#15):** falls at t=0.25 s (essentially at spawn). The bare serial
  ankle rings under explicit integration (F1: kd·dt/I≈2.4>2) and shakes the whole chain. The
  effort clip (19..140 N·m) prevents NaN but not the topple.
- **Explicit + `--armature on` (#16):** MJCF rotor inertia (ankle 0.1845) damps the ring — it
  now STANDS cleanly at t=0.5 s (base-z 0.91, pg_z −1.00) — but lurches LATERALLY (y→−1.08) and
  topples by t=1.0 s. Still worse than implicit's 2.25 s.
- **Why:** Isaac's implicit PhysX drive solves the PD semi-implicitly, adding numerical damping
  that helps our marginal serial ankle survive longer. Explicit's per-substep torque has no such
  cushion and STILL hits the lateral wall. Reproducing the training actuator did NOT help.

### F11 — Lead B measurement: the achilles Jacobian coupling is ZERO (collapses to diagonal)
The secondary lead was that the TWISTED ankle (A/B rods at different heights,
`twisted_*_ankle_model.yaml`) creates a real pitch↔roll coupling that the scalar ×3 hack
misses — apply the coupled 2×2 `K = J^T diag(kp) J`. **Direct measurement falsifies the premise.**
Measured `G = ∂(pitch,roll)/∂(A,B)` in MuJoCo (real closed loop, `/tmp/jac.py`), computed
`J=G^-1` and `K = kp·J^T J`:
- At the home ankle pose: off-diagonal **coupling = 1.5e-4 ≈ 0**; K is diagonal **×2.30 pitch /
  ×2.04 roll**. (Right ankle mirrors left — the "twist" only flips the sign convention.)
- Swept the full operating range (pitch ±0.47, roll ±0.44, `/tmp/jac_sweep.py`): coupling stays
  |·|<0.27 and ≈0 near the operating point; K_pitch ×2.24–2.70, K_roll ×1.86–2.32. The ONLY
  large off-diagonal (−7, +2) is at roll=±0.437 = the joint LIMIT, where the linkage is singular
  — a fall state (cf. F8), not a control regime.
- **Consequence:** the "principled coupled 2×2 PD" is mathematically a DIAGONAL scaling (pitch
  ×2.3, roll ×2.0), already expressible via `--ankle-kp-scale`/`--ankle-roll-scale`. A coupled-
  matrix flag would be provably dead code. The honest test of lead B is therefore exp #17 (the
  measured diagonal). NOTE: F9 already found roll boosts (×1.5–2, atop pitch ×3) make the lateral
  axis WORSE — so #17 (roll ×2.04) is expected to fail laterally too; #17 confirms rigorously.

### F12 — measured: #17 fell at 1.25 s (WORSE than baseline), exactly as F9 predicted
The measured Jacobian diagonal (pitch ×2.30, roll ×2.04) falls at t=1.25 s — earlier than the
pitch ×3 / roll ×1 baseline (2.25 s). The roll ×2.04 component degrades the lateral axis, matching
F9. Since the coupled 2×2 PD ≡ this diagonal (F11), lead B is dead at the root: there is no
coupling for it to add, and its diagonal is worse than what we already have.

---

## ⛔ FINAL VERDICT (2026-07-01) — reliable Isaac walk is LimX-gated; STOP manual gain/actuator tuning

Both of the strongest remaining STRUCTURAL leads were tested to a decisive measurement-backed
conclusion this session. **Neither cracks the 10 s bar; both are WORSE than the pitch ×3 baseline.**

| lead | idea | result | why it's dead |
|------|------|--------|---------------|
| A — explicit-torque model | match LimX's EFFORT-mode per-substep PD (TRON1) | fell 0.25 s (bare) / 1.0 s (+armature) | implicit drive's semi-implicit damping helps our marginal serial ankle; explicit removes it and still fails laterally (F10) |
| B — Jacobian-coupled ankle | apply `K=J^T diag(kp) J` coupled 2×2 PD | coupling measured ≈0 → collapses to diagonal ×2.3/×2.0 → fell 1.25 s (F11, F12) | the achilles linkage has NO pitch↔roll coupling to exploit; the diagonal it reduces to is worse than pitch ×3 |

**Root cause, now doubly-established:** the failure has TWO layers. (1) The serial ankle is too
soft fore-aft → back-drives into plantarflexion → forward runaway. This IS fixable sim-side:
`--ankle-kp-scale 3` kills the runaway and holds Oli upright ~2.0 s. (2) Underneath it is a
LATERAL single-support instability that is NOT ankle-gain-addressable (every roll/waist/coupled
boost makes it worse; F9, B1). The SAME brain walks reliably in MuJoCo, which has the real
parallel ankle + full body/contact fidelity — so layer (2) is a serial-model + Isaac
body/contact fidelity gap during the dynamic single-support step, not a controller-gain problem.
Gain and actuator-model tuning of the serial ankle have a hard ceiling at ~2 s; we have hit it
from three independent directions (gain scaling, explicit actuator, Jacobian coupling).

**Recommendation (Anton's architecture call):**
1. **Treat a reliable *dynamic* Isaac walk as LimX-gated.** The faithful path is LimX's own
   IsaacLab/IsaacGym HU_D04 training env (their serial-ankle setup + their contact/body params),
   which we don't have. Reconcile with LimX's reply when it lands; ask specifically for their
   walk training config (contact params, foot collision approximation, any lateral/ankle
   regularization) — that is the missing fidelity, not a gain.
2. **For the reasoning-demo locomotion in Isaac, use root-driven KINEMATIC GLIDE** (script the
   base trajectory + a canned/looped leg motion, no dynamic balance) — decouples the demo from
   the unsolved dynamic-balance problem. Keep the dynamic walk in MuJoCo (which walks) for any
   acting/locomotion research on the critical path.
3. **Ship `--ankle-kp-scale 3` as the documented best serial-ankle partial** (upright ~2 s at
   vx=0.1) for standing/manipulation + short scripted steps; do NOT invest more in gain tuning.

**Not pursued (flagged for LimX / future, NOT grinding now):** the lateral single-support
instability is the real frontier. Candidates if Isaac dynamic walk becomes critical-path:
free A/B joints + software kinematic constraint (real dual-motor dynamics, no rigid-loop blow-up),
and a foot-collision-approximation audit vs the MJCF box during single support (F4's original
H2, never fully resolved because the ankle back-drive dominated first). Both are bigger than a
gain flag and belong to a LimX-informed effort.

**Tools built/kept this run:** `walkmatch/ankle_jacobian.py` (measure the achilles Jacobian +
effective coupled stiffness in MuJoCo — reusable for any "is there ankle coupling" question).
All Isaac fixes remain OPT-IN flags; World defaults UNCHANGED.

### F13 — lateral OBS-fidelity check: NOT an obs bug; it's the first single-support step
Ran the F4 command-vs-realized comparison but on the ROLL/LATERAL axis: Isaac (implicit ×3, falls
2.25 s) vs MuJoCo (walks), same brain, vx=0.1, from the `OLI_TRACE` q/dq/qdes/gyro/quat taps
(`/tmp/latcmp.py`). Result:
- **Base roll is small and comparable in BOTH sims through t≈1.5 s** (Isaac stays within −0.7°,
  MuJoCo within +0.9°) and the **early ankle-roll COMMANDS match** (~+0.11 rad L in both). → the
  policy receives good lateral obs early; this is NOT a hidden obs/frame bug like the dead-IMU was.
- The divergence is ABRUPT at the **first swing step (~1.75–2.0 s)**: MuJoCo holds roll within ±1.5°;
  Isaac snaps to −8° (t=2.0) then −28° (t=2.25) and tips. Only early tell: Isaac's ankle-roll q
  DRIFTS (+0.06/+0.07 rad by 1.5 s under the compliant serial joint) while MuJoCo's stays flat (~0)
  — but F9 proved stiffening roll overshoots, so there is no gain sweet spot.
- **Conclusion:** the lateral failure is a first-step single-support DYNAMICS/CONTACT gap, confirmed
  independently from the observation side. This closes the loop: gain/actuator tuning AND obs
  fidelity both exonerated → the residual is body/contact fidelity, i.e. LimX-gated. (Tool: `latcmp.py`.)

---

## Post-verdict: Anton chose the FREE-A/B path (2026-07-01) — built, tested, run

After the verdict Anton chose to pursue option (2): emulate the real dual-motor achilles on the
serial ankle in SOFTWARE (free A/B + kinematic constraint, no rods → no PhysX loop blowup). Built
it faithfully as opt-in `--ankle-parallel` (module `isaacsim/ankle_parallel.py`, 10 committed
brain-env tests): each ankle driven by motor-space PD (per-motor ±42 clip) mapped through the
linkage Jacobian Jᵀ, + reflected rotor inertia diag(Jᵀdiag(I)J)≈[0.424,0.376] as ankle armature;
forces explicit control (the law is per-substep torque). The per-motor clip BEFORE Jᵀ is the
faithful shared-authority nonlinearity (pure pitch→motors differential ~90 N·m; pure roll→additive
~85 N·m) that scalar scaling can't express.

| # | config (vx=0.1) | fall onset | vs baseline (2.25 s) |
|---|-----------------|-----------|----------------------|
| C1 | implicit + `--armature` + Jacobian stiffness ×2.3/×2.04 | 0.75 s | WORSE |
| C2 | explicit + `--armature` + Jacobian stiffness ×2.3/×2.04 | 1.0 s | WORSE |
| D1 | **`--ankle-parallel`** (faithful: motor-space PD + Jᵀ + reflected inertia) | 1.5 s | WORSE |

### F14 — free-A/B faithful emulation = 1.5 s: best explicit variant, still < implicit baseline
The faithful dual-motor ankle holds Oli solid (base≈origin, pg_z=−1.00) through t=1.0 s — better
than every other explicit/parallel variant (A2/C2 fell ~1.0 s) — then the first-step lateral tip
takes it at ~1.5 s. It does NOT beat the plain implicit pitch×3 baseline (2.25 s), for a clean
structural reason: the faithful emulation NEEDS the reflected ankle inertia to make its explicit
PD stable, but adding reflected inertia to the IMPLICIT drive destabilizes it (C1: 0.75 s). So
faithful-ankle forces EXPLICIT body control, and explicit body is less stable than implicit in
Isaac (implicit's semi-implicit damping). No free lunch:
- implicit body + stiff ankle (no reflected inertia) = baseline 2.25 s  ← best
- explicit body + faithful ankle (reflected inertia) = D1 1.5 s
- implicit body + faithful ankle (reflected inertia) = C1-like 0.75 s (inertia destabilizes implicit)
A hybrid (implicit body + explicit-feedforward ankle) would inherit C1's instability → not worth
building. **The ankle is now faithfully solved; the residual first-step LATERAL CONTACT gap (F13)
dominates regardless — 10 s in Isaac remains gated on body/contact fidelity, not the ankle.**
`--ankle-parallel` is committed as the most-faithful ankle model (opt-in; defaults unchanged) — the
right serial-ankle actuator to pair with a future contact-fidelity fix or LimX's training config.
