# Round 6 — Verification: `menu-crafting` (the intelligent viewsphere)

**Role:** adversarial verifier. **Verdict: needs-work.**
**Part under test:** closed-form per-primitive derivation (face → centroid + normal at
derived standoff; cylinder → orbit stations spaced by incidence tolerance + caps);
every item pre-verified IK+collision before the LLM sees it; two-stage selection;
always-listed stop; explicit abstain/fallback item; anti-repeat; reference frame in
every prompt; menu diffs announced between rounds; fallback tier = equal-area
Tammes/HEALPix sphere gated by the calibrated fit-score→viewpoint-error mapping; LLM
never sees primitives, coordinates or knobs.
**Sources checked:** round-4 feedback (binding), all three round-5 interpretations,
Branch 3/4/5/6 + benchmarking syntheses, framing doc.

---

## 1. Grounding audit — invented in round 5, or from the reviewed literature?

Element by element, verified against the round-4 texts (line references are to the
accumulate files):

| Element | Grounded? | Evidence |
|---|---|---|
| Face → centroid + normal at standoff, closed-form | **Yes** | Mosbach Eq. 6 (`C = S(u_c,v_c) + n·d`, one evaluation, no optimization — structuring §87, §145); Wang T-Mech 2024 pipeline structure (numbers correctly refused in round 5); Scott's OSZ lineage |
| Cylinder → caps + orbit stations from an incidence functional | **Yes, with a hole (§2.1)** | Mosbach Fig. 6d: cylinder under `E_IA = 60°, d = 20` → 48 candidates in rings → 22 selected (view-set-selection §388–389, §1255) |
| Pre-verified reachability before the menu exists | **Yes** | GraspView's filter-first pattern; ObjView 45.4/128 average reachable (range 26–74); NBV-Net 0.988→0.472 collapse when the action space shrinks under a discrete selector |
| Two-stage selection | **Yes, extrapolated (§2.5)** | MOKA hierarchy ablation 0.6–1.0 → 0.0–0.2 (vlm-interfaces §222); architect endorsed in feedback |
| Always-listed stop | **Yes** | MapGPT's hard-coded `"A. stop"` item (vlm-interfaces §711, §1077) |
| Anti-repeat | **Yes** | Think3D "Do not request the same angle as before" (vlm-interfaces §1078) |
| Explicit abstain item | **Yes** | Frontier abstention accuracy 16.7–33.3% unless first-class (benchmarking §1127) |
| Frame stated in every prompt | **Yes, extrapolated** | RoboSpatial frame-implicit collapse, 6/8 models ≤0.1% (vlm-interfaces §279). Source is image-VQA, not text-only menus — but the mitigation is free, so acceptable |
| Flagged-not-silent infeasibility | **Yes (as a gap)** | Branch 6 Q11: four patterns in the wild, "none is ablated against any other, anywhere" (§626); Branch 5: "prefer flagged over silently removed… no literature cover" (§857–860) |
| **Menu diffs announced between rounds** | **No — round-5 invention** | Both Anton §1.2 and academia admit the literature has neither this nor its alternative. Honestly flagged as invented — but see §2.4: it is the only invented mechanism *with no experiment attached* |
| Tammes/HEALPix fallback (not naive {h,v}) | **Yes** | Branch 5's "single most valuable methodological implication" (§877–882); benchmarking's Tammes-baseline finding |
| Gate = calibrated fit-score→viewpoint-error, not raw residuals | **Yes (absence-grounded)** | Branch 3: "No calibration of any fit score against downstream task failure. Zero." (§250, §582); SHREC cone degeneracy — L2 parameter error 1e11 while point-to-surface residual looks fine (§844–845) |
| LLM never sees primitives/coordinates/knobs | **Respected** | Internal address `(plane, face_3, {standoff, roll})` explicitly internal (Anton §1.6); angle-style *labels* are sanctioned by the feedback as tentative and A/B-tested in Part F — they are relative view descriptors, not pose knobs |

Bottom line: this is the best-grounded part of the whole design — the
geometry-proposes / model-disposes-by-label pattern is a six-of-six consensus in
Branch 6, and every hygiene item has a named precedent. One mechanism (diff
announcements) is invented; it is honestly flagged but unfalsified (§2.4).

## 2. Where it breaks or leaks

### 2.1 The orbit-spacing parameter has no valid source at build time (most serious)

Part D spaces cylinder orbit stations "by the incidence tolerance." Which one? Two
different tolerances are conflated across the three interpretations:

- the **depth-side scan-admissibility gate** (provisional 60°, academia E8 / industry
  E11) — a fitting-quality parameter, inherited from reconstruction-era planners;
- the **RGB-side legibility band** for evidence views — measured only by E6 (Tier-0
  code-reading vs incidence), which is sequenced under Part H / demo-v2, i.e.
  effectively *last*.

Evidence-menu density must come from the RGB band — and the architect's own
refuse-list says so explicitly: *"Any incidence-knee value from the 45–75° band (all
wrong sensor class — and mostly irrelevant now that evidence is RGB)."* Yet the only
precedent cited for orbit spacing (Mosbach's 60° functional) is exactly such a value.
So in week 2 the menu factory is parameterized by a number the design has formally
refused to trust, and the experiment that would supply the real number runs weeks
later. E6's own disproof condition ("if reading is flat out to 40–50°, menu density
relaxes") concedes the dependency without fixing the ordering. **Fix:** pull E6
forward (it is a Tier-0 afternoon experiment on existing captures — nothing blocks
it), name the tolerance's source in Part D, and mark the shipped value provisional
until E6 lands.

### 2.2 "Pre-verified" promises more than static IK+collision delivers

Branch 4's own open-problem list, item 7: *"`g_kin ≥ 1` (an IK solution exists) does
not imply the executed branch is collision-free, and no paper engages with this"*
(view-set-selection §1118–1120). The academia interpretation then resolves "roll and
IK branch at execution" — i.e. the exact unverified degree of freedom is bound *after*
the menu promised feasibility. A pre-verified item that fails at execution (no
collision-free path, wrong IK branch) reintroduces precisely the mid-episode
menu-shape change this part exists to eliminate — NBV-Net's collapse in miniature,
inside our own contract. At 8–20 items, a motion-plan query per item in the kinematic
sandbox is affordable. **Fix:** verification = IK + collision + motion-plan from the
current pose (or at minimum from a canonical home pose), IK branch pinned at
verification time; and a defined failure path when execution still fails (item
flagged unavailable + announced — the diff mechanism can carry it). Log
execution-failure rate of pre-verified items as a standing metric; the industry E5
reachability audit counts construction survivors, not execution successes.

### 2.3 The verification world is incomplete by construction, and Part D doesn't say so

Two omissions in the gate stack as this part states it:

- **Collision set on partial clouds.** At round k the object model is partial;
  checking collisions against fitted primitives alone passes poses that collide with
  unseen geometry. The loop part imports Pan-style conservative union, but Part D's
  verification contract never states what it checks against. **Fix:** state it —
  inflated union(accumulated cloud, fitted primitives, prior), per the loop's own
  fallible-prior rule.
- **Occlusion/visibility is missing from this part's checklist.** Academia (§2.3) and
  industry (Part C) both include an analytic/mesh-free occlusion check; Anton's Part D
  lists only IK + collision. A reachable item whose target surface is occluded by a
  neighbouring object delivers a bad evidence image the blind LLM cannot anticipate.
  Latent for single-box demo v2; bites on any multi-object table. Analytic
  primitive-level occlusion is cheap and precedented — keep it in the stack.

### 2.4 The one invented mechanism has no falsification path

Menu-diff announcements are the only element with zero literature cover in either
direction (Branch 6 Q11: presentation of infeasibility never ablated; Branch 5:
"no literature cover — every corpus precedent assumes a non-blind consumer"). The
plan measures *mechanical* churn (E3: identity flip rate) but never measures the
LLM's behaviour under churn presentation. The nearest precedent cuts both ways:
SayPlan's textual-diagnostic pattern bought executability 13.3→86.6% but cost up to
5 replanning rounds, with 2/15 tasks never converging. Announced diffs add tokens
and distraction each round; they could plausibly hurt. As it stands the design's
sole invention would be shipped confirmed-by-assumption — exactly what this pass is
supposed to catch. **Fix:** one cheap Tier-1 arm on episodes with forced churn:
{announced diffs, frozen menu after round k, silent change} × grounded outcome.
Piggybacks on E3's churn machinery and E7's replay harness.

### 2.5 Two-stage: right experiment, mis-specified construct at the edges

- **Plane degeneracy.** A face yields exactly one derived view; stage 2 has content
  only for cylinders and the fallback tier. For box-like objects, two-stage collapses
  into "LLM picks the surface, geometry supplies the view" — which is the industry
  S1 architecture. E9 run on boxes would compare two nearly identical conditions and
  report a null that says nothing about MOKA transfer. **Fix:** E9's stimuli must
  include cylinders/compounds where stage 2 genuinely has >1 option, and the writeup
  should state that for plane-dominated scenes the menu design and S1 coincide (this
  is actually a point *for* the design's robustness — say it, don't hide it).
- **Fallback interaction undefined.** Tammes/HEALPix views have no surface anchor, so
  stage 1 ("which surface, plain words") is undefined for fallback items. When the
  gate trips per-region (academia's version), the menu mixes anchored and unanchored
  items with no stated presentation. **Fix:** design it — e.g. fallback surfaces as a
  single stage-1 item ("overview scan — shape not yet trusted") with sphere views as
  its stage-2 leaf; whatever the choice, it must exist before E10 scores abstention.
- Counter-evidence is honestly carried: Branch 6 Q12 (flat dump competitive at our
  scale) predicts E9 may fire its drop condition. That is fine — the drop condition
  is real and stated.

### 2.6 Smaller leaks

- **Anti-repeat vs the loop.** After a refit, the "same" named view may have a
  materially better pose. A blanket Think3D-style anti-repeat blocks a legitimate
  re-capture. Carve-out needed: repeat allowed when the item is flagged as updated
  (the diff mechanism again).
- **Cold start unstated.** Round 0 has no fit; Part D presupposes primitives. The
  gate arguably covers it (no fit ⇒ no trust ⇒ uniform tier), but the bootstrap menu
  should be explicit, not derivable.
- **E4 generalization assumed.** The score→error calibration is built on calibration
  objects + the synthetic corruption triad; deployment is novel objects/materials.
  The holdout discipline exists in Part E — state that it applies to the *gate*
  specifically (gate thresholds frozen before holdout contact), and report the
  mapping per primitive type, since SHREC says cylinders/cones are the weak classes.

## 3. Constraint & scope compliance

- **Incremental partial-cloud fitting:** respected — per-round rebuild, gate on fit
  trust, fallback tier. (With §2.3's caveat on the collision world.)
- **Real sensor defects:** respected via E4's corruption triad + slim depth check;
  no metrology creep detected.
- **Kinematic reachability:** respected in intent; §2.2 is the gap between intent
  and contract.
- **Discrete LLM menu selection / geometry never exposed:** fully respected. Angle
  labels are within the feedback's explicit sanction (tentative, A/B-tested); no
  primitive tuples, no knobs, internal addresses stay internal.
- **ADAM regression:** none. All rules are per-primitive, closed-form; industry's
  Part F names the no-per-object-scripts guard explicitly. The framing doc's
  non-goals expressly permit primitive-level deterministic rules.
- **No VLA / RL / defect detection:** clean.
- **Benchmark-as-auxiliary:** respected — menu experiments ride Tier 0/1 replay;
  no benchmark sprawl inside this part.

## 4. Would it survive the architect's standard?

- **Falsification attempt:** E4 (flat mapping kills score gating), E9 (drop
  condition), E10 (abstention scored, not assumed) are genuine disproof designs.
  Not falsifiable as planned: diff announcements (§2.4), orbit-spacing tolerance
  (§2.1 — untested until E6, which is sequenced last), and the "pre-verified ⇒
  executable" promise (§2.2 — never audited).
- **Rendered evidence:** the part's outputs are menus + captured evidence images;
  E10/E12 produce them. Adequate.
- **Concrete usage story:** demo v2 exercises every element except the fallback tier
  and the diff mechanism under churn — both need a forced-failure placement
  (code face down / shape ambiguous) in the demo protocol, which academia's E9
  already specifies. Adopt that.

## 5. Verdict

**needs-work.** The core is the best-precedented piece of the whole architecture and
none of the issues above invalidates it. But as specified, the part ships one
parameter with a source its own refuse-list rejects (orbit spacing), one promise its
own literature says static checks can't keep (pre-verified executability), one
invented mechanism with no test (diff announcements), and two undefined interface
edges (fallback-tier presentation, cold start). All are repairable inside the
existing experiment plan at near-zero added cost — mostly by re-sequencing E6,
tightening the verification contract, and adding one Tier-1 churn arm.
