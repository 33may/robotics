# Round 6 — Adversarial verification: `identity-persistence` (stable view identity across refits)

**Part under review:** address primitives/views by geometric content (normal in table+gravity
frame + area + position), never detector index; deterministic detectors preferred; RIoU +
Hungarian matching layer across rounds; E3 = the re-scan/refit repeatability protocol
(identity flip rate, parameter dispersion, menu churn per round). Disproof condition:
frequent flips force per-round anonymous menus, "a materially worse interface."

**Verdict: NEEDS-WORK.** The problem is real, the mechanism is grounded in round 4 (not
invented in round 5), and E3 is a genuine measurement nobody in the read corpus supplies.
But as specified the part cannot be *falsified*: there is no acceptance bar for "flips are
common," no measured link from flip rate to LLM harm, the matcher's hard cases (growing
partial faces, growing primitive count, uncalibrated match thresholds) are unspecified, and
the flip-rate ground truth is left to the matcher itself — the same score-grades-itself
circularity the branch flagged for residual confidence. One novelty claim overreaches the
corpus. All fixable inside the build month.

---

## 1. Grounding audit — round-4 provenance, checked line by line

The part survives the "invented in round 5?" test cleanly. Every element traces to round 4:

| Claim in the part | Round-4 source | Status |
|---|---|---|
| CGAL Eff. RANSAC randomized, **no seed parameter in the public API** | Branch 3 §3.3 tooling table (verified accessor-by-accessor) + §7 RQ3 | grounded, primary-doc-verified |
| Address by geometric content, never extraction index | Branch 3 §7 RQ3 design consequence 1 — "hard design constraint, derived from a documented library property, not a hunch" | grounded, verbatim |
| Deterministic detectors preferred (CGAL region growing, RSPD) | Branch 3 §7 RQ3 consequence 2 | grounded — **but see §4.7: the caveat attached there was dropped in round 5** |
| RIoU + Hungarian matching layer | Branch 3 §4.1 (SPFN Eq. 1) — explicitly flagged as *our extrapolation* from a training-time device to inference | grounded, extrapolation honestly declared |
| No re-scan repeatability experiment exists | Branch 3 §3.2 absence 4, §6 absence 4 ("zero, across all 20 documents") | grounded **within the corpus** — see §4.5 |
| Menu instability is dangerous for a blind LLM | Branch 5 §5.2 ("deletion is the dangerous option… no literature cover for the alternative") | grounded as a *flag*, explicitly uncovered by evidence |
| NBV-Net 0.988 → 0.472 as the catastrophe precedent | Branch 2 F5 (`objview-bench` whole→quarter shift) | grounded — but it is an **analogy**, see §4.1 |
| E3 metric set (count stability / parameter dispersion / flip rate) | Branch 3 §8.2 item 3, handed to Branch 2 in round 4 | grounded, verbatim |

Round 5 added exactly two things: menu churn per round as a metric (a natural extension once
the loop premise landed in the round-4 feedback), and the disproof condition. The disproof
condition is where the trouble is.

## 2. Constraint audit — passes

- **Incremental partial-cloud fitting:** this part exists *because* of that constraint; the
  loop premise (feedback §view-set-selection) is its founding fact. Pass.
- **Geometry never exposed to the LLM:** "internal addresses stay internal — the menu shows
  stable human-readable identities" (anton Part C). Pass, provided the stable labels stay in
  plain words ("top face"), which all three interpretations say.
- **ADAM regression:** content addressing is primitive-generic (normal bucket + area +
  position); no per-object recipes anywhere. Pass. Note industry §1.4.3 correctly observes
  the demo does **not** need cross-episode identity — the code face is re-found by VLM
  content each episode — so the box-moved-and-rotated robustness claim does not secretly
  depend on this part working across placements.
- **No VLA/RL/defect detection; benchmark auxiliary:** E3 is a component protocol, mostly
  runnable in replay/sandbox (fitting operates on fetched depth; no robot time beyond
  captures that Tier 1 needs anyway). Pass.

## 3. Does E3 prove or disprove? — the central weakness

**As specified, E3 describes; it does not falsify.** Three gaps:

1. **No pre-registered bar.** "If identity flips are common" — common relative to what?
   Without a numeric acceptance threshold tied to something downstream, any measured flip
   rate can be narrated as tolerable. The architect's own standard elsewhere (E4's gate with
   measured false-accept/false-reject; the pre-stated losing condition for E8) is exactly
   what E3 lacks.
2. **The harm link is asserted, never measured.** "The agent's context is garbage" and
   "anonymous menus are a materially worse interface" both rest on Branch 5's flag that
   itself says *"there is no literature cover for the alternative."* The one number invoked
   (NBV-Net 0.988→0.472) is a different failure class: a trained CNN emitting a
   single index into a shrunken action space and **terminating the episode** on
   infeasibility. An LLM facing a renamed-but-explained menu is not that mechanism; nothing
   read measures what it is. The part's disproof arm therefore compares against an
   **untested cost**.
3. **Counter-evidence the interpretations don't confront:** in the VLN systems the corpus
   itself leans on (MapGPT et al.), the action list changes at *every step* by construction
   and agents cope — with menu diffs announced, which anton §1.2 already plans. So the
   fallback ("per-round anonymous menus") may be mildly worse, not materially worse. Nobody
   knows, and E3 as designed will not find out.

**The fix is cheap and turns E3 into a real falsifier:** inject synthetic identity flips at
controlled rates (0%, 5%, 20%, 50%) into Tier-1 replay episodes — rename/renumber menu
entries mid-episode, with and without announced diffs — and measure grounded-QA degradation.
That yields the flip-tolerance curve; pre-register the bar as "measured flip rate must sit
below the knee of that curve." Then flip rate (E3) vs tolerance (new arm) is a genuine
pass/fail, and the "anonymous menus" cost stops being an assumption — it is the 100%-flip
cell of the same sweep.

## 4. Break attempts on the mechanism itself

### 4.1 The requirement is broader than the load-bearing need
The orchestrator's real cross-round dependency is narrow: the **task-relevant referent**
("the face the VLM said has codes") must stay resolvable round-to-round. A flipped name on
an irrelevant back face costs approximately nothing; a flipped target is recoverable by
nearest-centroid re-anchoring (the target has a known position from the previous round — a
one-line re-grounding step none of the three interpretations includes). "Context is garbage"
overstates a degradation into a catastrophe. E3 should report **target-referent flip rate**
separately from (and above) global flip rate, and the re-anchor step should be in the design
so a single flip is a recoverable event, not a poisoned episode.

### 4.2 Content addresses are non-stationary by construction — the matcher spec is silent exactly where the loop bites
Two of the three address components (**area, position of the observed patch**) grow/drift
every round *because scanning more is the point*: round 1 sees half the top face (area
≈0.5A, centroid pulled toward the seen half), round 3 sees all of it. The rounds where
stability matters most (early, sparse) are the rounds with the largest drift. Unspecified:
- the match-cost function's tolerance to monotone area growth and centroid drift;
- **unbalanced assignment** — primitive count is *supposed* to grow every round; Hungarian
  needs dummy rows/gating costs or it will force-rename when a new face appears;
- the match-acceptance threshold — an uncalibrated constant, i.e. exactly the pattern
  Branch 3 C8 documents ("classical methods substitute global constant tuning for training
  data").
E3 will produce different flip rates for every choice of these; they must be stated (and the
threshold swept) or the protocol is not reproducible even by us.

### 4.3 RIoU-over-point-membership quietly assumes stable point indices
SPFN's Relaxed-IoU is computed over point membership. Across rounds the cloud is re-fused
and (almost certainly) voxel-downsampled — indices are not stable, so membership IoU is not
directly computable without a nearest-neighbour correspondence step or a voxel-hash
membership proxy. That is buildable, but the alternative everyone will actually reach for is
parameter-space matching (normal angle + centroid distance + area ratio) — which is the
version with the uncalibrated thresholds of §4.2. Name which one is being built.

### 4.4 Flip ground truth: the matcher must not grade itself
"Identity flip rate **after matching**" needs correspondence ground truth that is not the
matcher's own output — otherwise a confidently wrong match is invisible, the same structural
blindness Branch 3 §1.3 established for residual scores. For a static object in a fixed
placement this is easy and should be stated: reference decomposition from the dense Tier-1
capture (or CAD for the calibration subset), each per-round fit registered to it
independently. No interpretation specifies this; without it E3's headline number is
unfalsifiable in the philosophical sense too.

### 4.5 Novelty overclaim — and unread adjacent prior art that likely contains the matching layer
"No re-scan repeatability protocol has **ever been published**" is corpus-bounded: 20
documents in one branch, with the branch's own sections E and F unread. The *protocol* claim
can stand hedged ("absent from the primitive-fitting/inspection corpus we read"). But the
**mechanism** claim ("a matching layer we write, because the literature has neither") likely
overreaches: plane-landmark **data association in plane-based SLAM** (infinite-plane SLAM,
point-plane SLAM lineages) does cross-scan plane identity under partial observation as
routine machinery — geometric-content matching of plane parameters across incrementally
grown maps, with the growing-observation problem of §4.2 already handled. That literature
was never in the map (the branch searched primitive *fitting*, not SLAM association). A
half-day prior-art search before writing a bespoke matcher is the falsification attempt this
part owes itself; at best it hands us tested cost functions and gating heuristics, at worst
it costs half a day and the novelty claim gets properly scoped.

### 4.6 Scope inconsistency across the three interpretations
Industry §1.4.3 scopes identity **within-episode**, scene-anchored, explicitly *no promise
across episodes* (and shows the demo doesn't need one). Academia's E2 measures flip rate
across **re-placements** — cross-episode identity, the thing industry disclaimed. Anton's E3
says "N times per object and across rounds within a session," which straddles both. One
scoped sentence is needed: within-session identity is the contract E3 gates; cross-placement
dispersion is a reported-but-unpromised statistic. Otherwise E3's headline number pools a
promised property with an unpromised one and can fail on the half that was never claimed.

### 4.7 Two compressions from round 4 that lost load-bearing caveats
- **Determinism:** Branch 3 said determinism on identical input is "necessary but **not
  sufficient** — it says nothing about stability across two different scans." In the loop
  the input differs every round *by construction*, so deterministic detectors buy debugging
  reproducibility, not cross-round identity. Fine as a preference; it must not appear in
  E3's results as an identity mechanism. The real ablation is matching layer (and
  hysteresis) on/off — industry's E3 has this, anton's E3 does not.
- **Hysteresis:** industry adds it (label persists unless the fit moves beyond a threshold);
  anton's Part C — the binding methodology — omits it. Matching without hysteresis renames
  on any jitter that crosses a bucket boundary (a ~45°-tilted face oscillating between
  "top" and "side" wording). Since the LLM sees coarse word-labels, label-level hysteresis
  is nearly free and probably does most of the stabilizing work. It belongs in the solution
  spec, ablated in E3.

### 4.8 Smaller defects
- **Frame anchor:** "table+gravity frame" — if the table plane is re-fit per round from the
  same defective cloud, every normal-bucket address inherits a correlated jitter mode and
  E3 cannot attribute flips to the matcher vs the anchor. Anchor to the **robot base frame**
  (table calibrated in the cell, gravity known); the fitted table plane then becomes one
  more primitive, not the coordinate system.
- **Churn decomposition:** "menu churn per round" conflates desirable churn (new geometry →
  new views, announced as diffs — anton §1.2 *wants* this) with pathological churn. Report
  it decomposed: {adds from new geometry, drops from reachability, identity flips,
  jitter-relabels}. An undecomposed churn number gates nothing.
- **Expected outcome honesty:** on boxes/cylinders with a deterministic detector over an
  accumulating (superset) cloud plus coarse word-labels, the likely E3 result is a very low
  flip rate — i.e. the experiment will probably *confirm*. That is fine (it still produces
  the unpublished number), but it is one more reason the injected-flip harm sweep of §3 is
  the half of the experiment that can actually surprise us.

## 5. What would make this part survive the architect's standard

1. **Pre-register the bar:** flip-tolerance curve from injected flips on Tier-1 replay
   (0/5/20/50/100% cells, diffs announced vs silent); acceptance = measured E3 flip rate
   below the curve's knee; the 100% cell *is* the "anonymous menu" cost, measured not assumed.
2. **Report target-referent flip rate above global flip rate; add the nearest-centroid
   re-anchor as the standing recovery step.**
3. **Specify the matcher:** parameter-space vs membership-proxy; unbalanced-assignment
   handling; thresholds stated and swept; hysteresis in the spec; ablation = {matching,
   hysteresis} on/off (adopt industry's E3 form).
4. **Independent correspondence GT** from the dense capture / CAD calibration subset.
5. **Scope sentence:** within-session identity promised and gated; cross-placement reported,
   unpromised.
6. **Half-day prior-art pass on plane-SLAM data association** before writing the matcher;
   re-scope the novelty claim to the protocol, hedged to the read corpus.
7. **Anchor addresses in the robot base frame; decompose churn.**

None of this changes the architecture. It changes whether the eventual "identity is stable"
sentence is evidence or taste.
