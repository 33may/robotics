# Round 6 — Verification: `view-crafting` (internal recipe of the intelligent viewsphere)

**Part under test:** quality by construction (Mosbach route b): face → normal through
centroid at FOV/Min-Z-derived standoff; cylinder → caps + orbit stations with K emergent
from the incidence tolerance; edges/corners out of scope v1. Constraint stack: hard
incidence gate (60° provisional, E8), IK+collision filtering composed with
defer-to-next-best (Yu & Gupta), action = (position, approach direction) with roll/IK
branch resolved at execution. Every entry carries a full pose internally; the LLM sees
none of it.

**Role:** adversarial verifier. **Verdict: needs-work.** The skeleton is genuinely
grounded in the round-3/4 corpus — nothing here was invented in round 5 — and it
respects the architect's scope (no VLA/RL, no defect detection, geometry never exposed,
per-primitive rules not per-object scripts). But the part as specified has one
calibration error that touches the founding complaint, two internal coherence defects,
one missing constraint that produces a direct demo-v2 failure mode, and no experiment
that falsifies the construction itself. All are fixable without re-architecting.

---

## 1. Provenance audit — grounded, with two citation defects

Checked claim by claim against the branch syntheses:

| Claim | Source | Verdict |
|---|---|---|
| Quality by construction, route (b) | Branch 4 §2.1(b): Mosbach Eq. 6 places every candidate on the surface normal; Scott 2009 OSZ; Peuzin-Jubert's "coverage optimization" category name | **Grounded** |
| Face → centroid + normal | Branch 4 §3.1 items 1–5 (metrology accessibility-cone chain for the principle; Roberts & Marshall for the rule; Gospodnetić Table 4's 2–4× inadmissibility cut for the engineering argument) | **Grounded** — with Branch 4's own caveat kept: the CMM chain covers the *principle*, nobody names "one view per face" |
| Cylinder rings emergent from incidence tolerance | Branch 4 §3.1 item 6: Mosbach Fig. 6d, 48 candidates as mantle rings + cap views at t=60°, reduced to 22 | **Grounded as pattern** — but see §2.1: this is a *reconstruction-coverage* precedent, and Mosbach's 60° is tagged "design choice" in Branch 1's own knee table, not a measured number |
| Standoff from FOV/Min-Z | Branch 1: SEE Alg 2 density↔standoff↔FOV closed form; D405 Min-Z 100 mm @ 1280×720; invalid-band `B·F/Z` | **Grounded** — and coherent: SEE's formula is fronto-parallel (no cos θ), which is *valid here precisely because* candidates are orthogonal by construction. State that; if the formula is ever reused for oblique coverage credit it is wrong |
| 60° provisional gate | Branch 1 §2.2: knee contested 45–75°, majority report ~60° (Soudarissanane, Lichti 65°, Johnson, Mehdi-Souzani — the latter two secondhand, unread) | **Grounded but quarantined** — see §2.4 |
| Filter + defer-to-next-best compose (Yu & Gupta) | Branch 4 §4.1 | **Miscited.** Yu & Gupta's implemented composition is rungs **(iii)+(iv)** — candidates drawn only from the roadmap's reachable component, plus next-best-target fallback. The (i)-flavoured filter + exact-query composition is **Spensieri 2025**. The design (i+iv) is still defensible, but cite Spensieri + Wang's ranked-retry for it, not Yu & Gupta |
| (position, approach direction), roll free | Branch 4 §4.4: Zacharias's 5D capability level; SCVP/MA-SCVP 5-DoF pose ("sixth DOF has almost no effect"); Wang 2026's roll-free `C_obs` | **Grounded** — but see §2.2/§2.3 for what round 5 did not price |
| Edges/corners out of scope v1 | Branch 5 §2.3: "either they get an address… or they are declared out of the action space. No precedent tells us which. This is ours to decide and to defend" | **Grounded** — this is exactly the explicit recorded decision Branch 5 demanded, and it matches the architect's open question #2 |

So: not invented in round 5. The two defects are the Yu & Gupta rung mislabel and the
Fig. 6d precedent being quietly upgraded from "coverage at ≤60°" to "evidence menu
sizing" — which is the substantive issue below.

---

## 2. Where it breaks

### 2.1 The K-sizing tolerance is calibrated by the wrong experiment — this touches the founding complaint

The part has **one** incidence knob, and its only calibration experiment is E8 (depth
fill/noise vs incidence on a plate — a *scanning-side, depth-side* measurement). But
under the architect's QA reframe, that knob does two jobs that now belong to two
different constructs:

- **τ_scan** — gates which surface points count as covered-at-quality for *fitting*.
  Depth-side. E8 is the right calibrator. This is what Mosbach's 60° functional and the
  whole 45–75° knee literature are actually about.
- **τ_evidence** — sizes K (orbit-station spacing) and defines "orthogonal enough" for
  the *evidence image*. RGB-side. Per the feedback ("once we are in a good view we care
  only about the RGB image"), its calibrator is the Tier-0 legibility-vs-incidence
  measurement (anton.md E6; industry E12's GSD floor), which the architect himself
  names as the thing that could tighten or relax menu density.

As specified, K emerges from the E8-calibrated tolerance. Run the arithmetic: orbit
stations spaced so every mantle point is within the 60° gate means stations every
~120° — K=3 — and the *worst-case evidence image of a mantle region is at ~60°
incidence*. A 60°-off-normal evidence image is precisely the grazing-evidence failure
Albert flagged twice and the project exists to eliminate. Mosbach's Fig. 6d cannot
license this: his is a reconstruction pipeline where ≤60° coverage is the goal; ours is
an evidence pipeline where the menu must contain a *near-orthogonal* view of any
inspectable region. Same formula shape, different construct, different number.

**Fix (cheap):** two named tolerances. τ_scan from E8 gates coverage credit and
scanning admissibility. τ_evidence from Tier-0 legibility-vs-angle sizes K and is the
pre-registered "orthogonal enough" band (academia §4 already promises that band — but
sources it "from E1/E8", which repeats the conflation; it must come from the RGB
measurement). If E6 shows reading accuracy flat to 40–50°, K relaxes honestly; if it
shows a 15° cliff, K grows and the design learns that early.

### 2.2 "Full pose internally" vs "roll/IK branch resolved at execution" vs "pre-verified" — pick two

As written, the part contradicts itself: an entry cannot both carry a full pose and
have roll resolved at execution; and if exact IK+collision verification runs at
menu-build time on one configuration while execution re-resolves roll/branch, the
executed configuration is *not the verified one*. A 6-DoF UR5e has up to 8 IK branches
per pose and a free roll circle; branches and rolls differ in collision status
(Branch 4 Q11 flags exactly this as the corpus's unengaged
execution-failure-after-commitment risk — Glorieux never says which IK solution gets
used). This silently re-imports the action-space integrity failure the design is built
to prevent (ObjView-Bench's NBV-Net collapse is the documented endpoint).

**Fix (cheap):** verify a **witness configuration** (specific roll + IK branch) per
menu entry at construction time; cache it; execution uses the witness or re-verifies
before commitment. The action stays (position, approach direction) — the witness is an
existence proof carried alongside, not a change of representation.

### 2.3 Free roll quietly breaks "well-framed" and the standoff math

Two concrete mechanisms, neither priced by any round-5 interpretation:

1. **Framing.** The sensor is rectangular. Whether an elongated box face fits in frame
   at the FOV-derived standoff depends on roll. If roll is resolved by IK convenience,
   the standoff computed for the favourable orientation does not guarantee framing.
   Either constrain roll for elongated faces at build time (fold into the witness), or
   compute standoff from the roll-invariant inscribed footprint and pay the GSD cost —
   a decision, not a detail.
2. **Invalid-depth band.** The `B·F/Z` band sits at one fixed image edge (~10% of
   width at 10 cm) — *which scene region loses depth is a function of roll*. Free roll
   means the planner cannot promise which part of the captured cloud is depth-valid.
   Matters for the loop's own accumulation, not the RGB evidence; still worth one line
   in the witness policy.

### 2.4 The 60° default sits on the architect's refuse-list — keep it quarantined

anton.md §4, verbatim refusal: "Any incidence-knee value from the 45–75° band (all
wrong sensor class — and mostly irrelevant now that evidence is RGB)." Adopting 60° as
a provisional default pending a half-day E8 is defensible engineering — but the part
as specified wires the provisional number into menu construction (K, coverage credit)
from day one. Discipline required: nothing derived from the 60° number ships or gets
reported before E8 replaces it, and the number is recorded as provisional in the
pre-registration. With §2.1's fix this mostly dissolves (the evidence knob stops
depending on it at all).

### 2.5 Defer-to-next-best is a scanning mechanism transplanted into an evidence job

Rung (iv) — Yu & Gupta's `getWhereToScan()` — defers to the next-best *target* when
the wanted target is unreachable. That serves the scanning phase. In the evidence
phase the target region is fixed by the question ("the code face"); deferring to a
different region does not answer it. Branch 4 §4.5 names this exact hole as a
first-class gap the field never filled: "the design decision — best-available view per
region, plus an explicit unreachable-region report — is ours to make and defend."

The dangerous sub-case is not the unreachable face (academia's abstention item covers
face-down) but the **reachable-but-degraded** one: all within-band stations of a region
blocked, one 50°-off-normal station reachable. If that station appears as a regular
menu item, the system serves a grazing evidence image while looking successful — the
original complaint, reproduced inside the fix. The menu must carry the quality band on
the fallback item ("best remaining view of face 3: ~50° off-normal") or route to
abstention, and the grounded scoring must count within-band separately (the
pre-registered tolerance band already gives the threshold — once §2.1 sources it
correctly).

### 2.6 The standoff derivation is missing its ceiling — direct demo-v2 failure mode

"FOV/Min-Z-derived standoff" supplies **lower bounds only** (face fits in frame;
camera past Min-Z). The QA construct imposes an **upper bound**: pixels-per-code —
industry's E12 exists precisely to back-propagate the GSD legibility floor into this
recipe, and the part as specified does not consume it. Worst case: large face, small
code — the FOV-fit lower bound exceeds the GSD ceiling, the single per-face centroid
view frames perfectly and reads nothing, and v1 has **no closer-look action** in the
action space. That is a demo-v2 failure with no recourse in the menu.

**Fix:** standoff = clamp(FOV-fit, Min-Z+margin, GSD ceiling); when the interval is
empty, the recipe emits a *deterministic ROI close-up* (same normal, centered on the
VLM-reported code region instead of the centroid) as a second entry for that face.
Still closed-form, still per-primitive-class, still not ADAM.

### 2.7 No experiment falsifies the construction itself

E8 falsifies the gate; E1/E2 falsify the upstream fit; E4 falsifies the headline. But
the recipe's own guarantees — the things "by construction" is doing load-bearing work
for — are never audited: (a) *achieved* incidence of executed captures vs planned
(the ~4 cm fused-frame offset plus execution pose error mean planned-orthogonal ≠
achieved-orthogonal; scoring must be against locally-refit raw-cloud normals, the
circularity guard industry §5 already prescribes); (b) framing verified on real
captured images, not assumed from the math; (c) worst-case evidence angle across the
mantle for the chosen K; (d) per-region menu completeness (every inspectable region
has ≥1 within-band entry or an explicit disclosure). Without this, "quality by
construction" is an assertion, and the part fails the architect's standard
(falsification attempt + rendered evidence). A half-day **construction audit** — run
the factory on fitted primitives of the real demo objects, execute or render each
entry, report the four numbers above with images — closes it and produces exactly the
rendered-evidence artifact the standard demands.

---

## 3. What survives scrutiny untouched

- **Not an ADAM regression.** Rules are per-primitive-*class*, closed-form, no
  per-object scripts; explicitly licensed by the framing doc's non-goals.
- **Scope-clean.** No VLA/RL/defect detection; primitives, poses, and knobs stay
  internal; the LLM sees only the finished menu — consistent with the feedback's
  intelligent-viewsphere correction and Branch 6's six-of-six consensus pattern.
- **Loop-compatible.** Construction is closed-form per round on the current fit;
  nothing assumes a complete survey; cost is negligible next to the fitting.
- **Reachability first-class.** Exact IK+collision before the LLM sees an item, with
  flagged-not-silent removal — the right response to ObjView's 45.4/128 base rate,
  modulo the witness defect (§2.2).
- **Edges/corners exclusion** is the explicit, recorded, revisitable decision Branch 5
  demanded; it blocks neither demo v2 nor the headline claim (near-primitive stratum
  is separately reported with abstention scored).
- **Minor, note only:** long cylinders need multiple rings along the axis (axial FOV
  footprint) — Mosbach's precedent had them; "orbit stations + caps" reads as one
  ring. Under-specified, trivially fixed.

## 4. Verdict

**needs-work.** Grounded, scope-clean, buildable — and carrying one construct-level
calibration error (evidence menu sized by a depth-side gate), two coherence defects
(witness configuration; roll vs framing), one missing constraint with a demo-shaped
failure mode (GSD ceiling / close-up action), a transplanted fallback that can
silently serve the exact failure the project exists to kill, and no falsification of
its own central promise. Every fix is days, not weeks; none changes the architecture.
