# Round 6 — Adversarial verification: `menu-interface` ("what the text-only LLM sees")

**Part under test:** two-stage decision (which surface → which view; MOKA hierarchy
ablation 0.6–1.0 → 0.0–0.2), flat leaf menus of ≤~8 entries, stable content-derived
view names, explicit stop action, anti-repeat, explicit frame statement, explicit
unavailable-view disclosure (motivated by NBV-Net's 0.988 → 0.472 collapse), label
content settled by E3 not taste, menu order as a permuted variable.

**Inputs verified against:** `round4_accumulate/feedback.md` (binding),
`round4_accumulate/vlm-spatial-interfaces.md`, `structuring-the-viewpoint-space.md`,
`view-set-selection-under-constraints.md`, `benchmarking-view-planning.md`, all three
round-5 interpretations, and the framing doc.

**Verdict: NEEDS-WORK.** The skeleton of the part is the best-grounded thing in the
whole round-5 design — almost every element traces to a verified number or a deployed
precedent, and the two elements the architect left open (label content, two-stage
value) each have a real experiment with a real disproof condition. But the part
contains one invented mechanism wearing borrowed evidence (unavailable-view
disclosure), one prescribed gate from Branch 6 that all three interpretations
silently dropped (the instruction-following pre-check), and one experiment (E3)
whose declared power cannot resolve its own expected effect size. None of this is
fatal; all of it is fixable inside the month.

---

## 1. Provenance audit — grounded in the reading, or invented in round 5?

| Element | Provenance | Status |
|---|---|---|
| Selection over emission; finished menu, no geometry in the prompt | SoM +60.7 (within-paper ablation); 6/6 systems in Branch 6 §1.7; feedback.md's own "intelligent viewsphere" correction | **Grounded.** Also the architect's explicit ruling, not a round-5 addition. |
| Two-stage decision (surface → view) | MOKA Table IX, 0.6–1.0 → 0.0–0.2 (Branch 6 §1.8, verified against primary text); endorsed in feedback.md ("I like the idea... nothing more to add") | **Grounded, transfer untested** — see issue 3. |
| Flat leaf menu (no hierarchy at menu level) | Structured-Interfaces small-graph tie; SayPlan scale ablation (failures were token-limit only); MOKA's leaf is flat at ~25–40 options | **Grounded** (Q12: "at our scale, dump it flat"). |
| ≤~8 entries | How-Many-Tools K≈2.2 vs K=5 (−6.0 pp, 3 seeds — one of only 4 variance-reporting sources); deployed range 3–10 (Think3D 8, PIVOT 10, Explore 3); Branch 5 §3.3: 8–20 addresses/object, two-stage brings per-turn count down | **Grounded as a range; the "8" itself is convention.** Think3D never swept menu size (flagged in Branch 6 §2 as never-ablated). Acceptable as a default; see issue 6 for the stage-1 loophole. |
| Stable content-derived view names | CGAL seedless nondeterminism (Branch 3, verified); no re-scan repeatability protocol exists anywhere (confirmed absence); SayPlan names-over-indices (**demonstrated, never ablated** — Branch 6 Q4 item 5) | **Half grounded.** That stability is *achievable* is our own E2/E3 measurement (correct — no literature exists). That stability *matters to the decider* is assumed, never measured by anyone including us — see issue 4. |
| Explicit stop action | MapGPT's always-listed `"A. stop"` | **Grounded** (precedented, unablated, free — correctly treated as hygiene). |
| Anti-repeat instruction | Think3D "do not request the same angle" | **Grounded as precedent**, but transplanted from a static-scene setting — see issue 7. |
| Explicit frame statement in every prompt | RoboSpatial frame-implicit collapse to ~0% for 6/8 models (ablated: 8–41% → ~0–20%) | **Grounded.** Strongest cheap item in the part. Caveat carried correctly in round 5 (no restatement intervention exists anywhere). |
| Explicit unavailable-view disclosure | Branch 6 Q11: four observed patterns, **none ablated against any other, anywhere**; Branch 5 §5.2: "no literature cover for the alternative"; Branch 6 gap list #7: "no comparative study of infeasible-option presentation" | **Invented in round 4/5 — and admitted to be** ("this one is ours to design"). The problem is the evidence dressing — see issue 1. |
| Abstain/fallback as a first-class menu item | 16.7–33.3% abstention accuracy for frontier models (`view-level-evidence-benchmark`, Branch 2 Q16) — verified present in the synthesis | **Grounded as motivation**; E10 (anton) / E9-adversarial-placements (academic) actually scores it. Good. |
| Label content settled by E3 | feedback.md verbatim ("evaluate the different options as swappable components and select the best by numbers"); genuine field gap (nobody has ablated labels for a camera-pose menu; textual az/el never measured — Branch 6 gaps #5) | **Grounded and correctly owned.** The A/B fills a confirmed absence. Power problem — see issue 2. |
| Menu order permuted, never a formatting choice | Pezeshkpour & Hruschka: 13.1–73.4 pts **oracle** bound, 0–8 realized, no count correlation | **Grounded and correctly cited** — all three interpretations label the 73 as an oracle bound. Verified: no misquote. |

**Net:** one element invented (disclosure), one element half-invented (value of stable
names), everything else traces cleanly. No round-5 fabrication found; the round-5
docs are in places *more* careful than the task summary that describes them.

---

## 2. Issues (ordered by severity)

### Issue 1 — The disclosure design borrows NBV-Net's number for a failure mode the design already precludes, and no experiment tests disclosure itself

The equation "silent removal + blind decider = NBV-Net's measured collapse" does not
survive inspection of the source. ObjView-Bench's NBV-Net failure (0.988 → 0.472,
executed views 9.9 → 1.2) is a policy that **emits an index into a space that shrank
and terminates the episode on an infeasible prediction** (view-set synthesis §4.1:
"outputs a single discrete index and terminates the whole run if that index is
infeasible"). In our design every menu item is pre-verified (IK + collision) before
the LLM sees it — the LLM *mechanically cannot* pick an infeasible view, so the
NBV-Net termination mode cannot occur by construction. The measured collapse
motivates pre-verification (which we do), not disclosure.

What silent removal actually costs a blind decider is epistemic and **unmeasured by
anyone**: the agent cannot know a surface is uninspectable, may keep hunting for it
(MapGPT's blind arm: indecision, +5.8 OSR / −5.8 SPL), and its coverage claims go
wrong. The syntheses say this correctly ("closest documented analogue", "the
measured catastrophe at the end of that road" — rhetoric, flagged as such); the
part's summary hardens the analogy into an equation.

And then: **no experiment in any of the three round-5 methodologies ablates
disclosure vs silent removal.** E10/E9 score whether the agent *takes* the fallback
item; nothing tests whether the disclosure text changes behaviour at all — it could
plausibly be neutral, or even net-negative on the open-model tier (Branch 6 §1.1:
added interface machinery hurts below-threshold models; disclosure text is added
machinery). This is doubly annoying because the field's own gap list (#7) says this
exact comparison has never been run, we are building both variants anyway, and a
Tier-1 replay arm costs one serializer flag. As it stands, the one element of the
part with zero literature cover is also the one element with zero planned test.
That fails the architect's falsification standard.

### Issue 2 — E3's declared power cannot resolve its own expected effect

All three interpretations carry both statements: expected label-scheme deltas are
small (SCAFFOLD: ≤4.6 pts among schemes, +5.7–10.3 only for any-vs-none) *and* the
honest sizing note "at n≲100 episodes we resolve 15–20-point differences, not 5."
Put together: **E3 as specified is a catastrophe detector, not a ranker.** It can
find a scheme that collapses (bare indices with no description, frame-implicit); it
cannot rank angle labels vs directional words unless n scales well past 100 — which
Tier 0/Tier 1 replay does permit (a "move" is a fetch; items are cheap) but no doc
states the required n or the minimum detectable effect for E3 specifically.
Anton's pre-declared tie-rule ("if all variants tie within noise, the
cheapest-to-generate serializer wins by default") saves the experiment from being a
fishing expedition — that rule is good and should be kept — but the part currently
promises "E3 settles it" while budgeting an instrument that will, with high
probability, return "tie". Say which one E3 is, and size it accordingly.

### Issue 3 — MOKA transfer is treated as the expected outcome when the analogy has a named confound

Grounding is real (largest ablated effect in the branch, verified). But MOKA's flat
condition was "annotate everything at once, ask one VQA question" — **25–40
simultaneous visual marks on one image**. Our flat alternative is a text menu of
~8–20 items, under the visual-clutter regime's threshold, in a different modality.
MOKA's hierarchy also decomposes *task* structure (what to do), not surface→view.
The effect could shrink to nothing at our scale for reasons that have nothing to do
with whether hierarchy helps us. The round-5 docs handle the outcome correctly
(E7/E9 with a real drop-the-turn disproof condition — this is the right structure),
but the framing ("backed by the largest measured effect", "predicts a large
effect") oversells transfer probability. If E7 comes back null, that is the
*expected* result of the confound, not a surprise. Cheap fix: state the confound in
the E7 write-up and treat two-stage as default-on pending the ablation, which is
what the build order does anyway.

### Issue 4 — The value of stable naming to the decider is assumed; only its feasibility is measured

E2/E3(identity) measure whether content-derived addresses *can* be kept stable
across refits — genuinely novel, correctly designed, with a disproof condition. But
the claim that churn *hurts the LLM* (the reason the identity layer is load-bearing)
rests entirely on the Branch 5 §5.2 flag ("exactly the instability a language model
handles badly" — an inference, no citation) plus the NBV-Net analogy from issue 1.
No arm anywhere compares the same episodes with stable names vs per-round anonymous
renumbering. If identity flips turn out common (E2's disproof fires), the fallback
is described as "a materially worse design" — on what number? A one-flag Tier-1
churn arm (deliberately renumber between rounds, measure grounded-QA delta) prices
the thing the whole Part C engineering exists to prevent, and doubles as the
justification for the identity layer's build cost.

### Issue 5 — Branch 6's prescribed instruction-following pre-check was dropped by all three interpretations

Branch 6 §1.1's consequence, verbatim: "a cheap instruction-following pre-check must
gate the whole design." Grep of all three round-5 files: the pre-check appears in
none of them (model-tier stratification in E3 survives; the gate does not). The part
stacks two-stage + stop + anti-repeat + frame statement + disclosure + menu diffs
into one composite prompt. Each item is individually cheap; the *composite* is
exactly the "added machinery" that §1.1 shows is net-negative below the capability
threshold, on six independent results — and the deployable-open tier is an explicit
E3 arm. The pre-check is an afternoon of work and was already prescribed by the
reading. Restore it.

### Issue 6 — The ≤8 bound governs the leaf; nothing bounds stage 1

Two-stage keeps *views-per-surface* small by construction (orbit stations from the
incidence tolerance ≈ 6, faces fewer). The stage-1 menu — which surface — is bounded
only by object complexity: a compound of three primitives can exceed 8 listable
surfaces, and no doc says what happens then (truncate? group by primitive? scroll?).
For the month's object class this probably never binds (Branch 5: 8–20 addresses per
*object*), so this is a footnote — but the part claims a menu-size discipline it
only enforces on one of the two turns. One sentence fixes it.

### Issue 7 — Anti-repeat conflicts with the loop's own legitimate recaptures

Think3D's anti-repeat comes from a static-scene setting where re-requesting an angle
is always waste. In our incremental loop, re-taking "the same" view after the fit
matures is *correct* behaviour (early fit aims the scan; evidence capture waits for
the maturity gate — industry §1.4.1). A blanket "never repeat" instruction plus
stable view names could block the recapture the fit-maturity design requires. Needs
one rule: anti-repeat applies within fit-maturity state, or repeats are allowed when
the menu diff announces the underlying fit changed.

### Minor (citation hygiene, no design impact)

- The "30 pp on hard cases" figure attached to K≈2-vs-K=5 belongs to How-Many-Tools'
  **embedding-scorer** arm on the medium bucket (80.1 vs 50.0), not the LLM chooser
  (whose gap is 6.0 pp). All three interpretations drop the caveat. The direction is
  unaffected; the magnitude quoted for menu-size sensitivity is not the LLM's.
- TwNV's "Discrete" condition being the least-documented result in its own paper is
  carried correctly everywhere — good; keep it as the reason E3 exists.

---

## 3. Constraint compliance (checked, all pass)

- **Incremental partial-cloud fitting:** respected — menu identity/diff machinery
  exists *because* of per-round refits; E2 is loop-shaped, not survey-shaped.
- **Kinematic reachability:** respected — pre-verified items (exact IK + collision on
  filter survivors), reachability-exhaustion-as-valid-stop imported from Branch 4.
- **Discrete selection:** respected everywhere; no metric emission anywhere.
- **Geometry never exposed:** respected. Angle-style labels are numbers-as-labels of
  an enumerated set (Think3D's deployed pattern, sanctioned by Branch 6 encoding
  rule 4), not coordinates or knobs; internal pose stays an executor side-channel.
  Note feedback.md's own sketch says "view number + plain directional words" while
  the architect's stated preference is angle labels — round 5 correctly resolves the
  tension by making it E3's job rather than picking silently.
- **ADAM regression:** none found. Names are content/scene-derived at runtime
  (table+gravity frame, within-episode stability only — industry §1.4.3 correctly
  refuses cross-episode promises); demo v2's code face is found by image content,
  not a stored per-object label. No per-object recipe anywhere in the part.
- **No VLA/RL/defect detection:** clean.
- **Benchmark-as-auxiliary:** E3/E7/E10 are the feedback's *named* use of the
  benchmark (representation choices), so the load is sanctioned; all run on Tier
  0/1 replay (API calls, no robot time). Watch only that E3's n-scaling (issue 2)
  doesn't quietly turn the auxiliary harness into the month's main deliverable.

## 4. Would it survive the architect's standard?

**Falsification:** present for two-stage (E7/E9), label content (E3 + tie-rule),
identity feasibility (E2), abstention uptake (E10) — genuinely disprovable, with
consequences stated. **Absent** for disclosure-vs-silent and stable-vs-anonymous
naming (issues 1, 4): those two ship as taste unless an arm is added. **Rendered
evidence + usage story:** demo v2 exercises the full menu path under the grounded
2×2; adequate. So: survives on four of six elements; the two unfalsified elements
are, not coincidentally, the two the field has never measured either.

## 5. Suggestions

1. Add a **disclosure ablation arm** to Tier-1 replay: {silent removal, listed-
   unavailable-with-reason} × {frontier, open}, same episodes. One serializer flag;
   fills the field's gap #7; converts the part's only untested element into its
   cheapest novel result. Score decisiveness (moves, repeat-hunting of unavailable
   surfaces) not just accuracy — that is where MapGPT says the damage shows.
2. Add a **churn arm**: same episodes, stable names vs deliberate per-round
   renumbering. Prices the identity layer; becomes the fallback decision number if
   E2's flip-rate disproof fires.
3. Restore the **instruction-following pre-check** as a gate on the composite menu
   prompt, per Branch 6 §1.1's explicit prescription; run it per model tier before
   E3 spends seeds on a below-threshold model.
4. State E3's **minimum detectable effect and required n**; keep the tie→cheapest
   rule. If the goal is only catastrophe detection, say so and cut E3's cell count
   (drop the hybrid variant first).
5. Downgrade the NBV-Net citation in the disclosure rationale from equation to
   analogy ("pre-verification prevents the measured mode; disclosure addresses the
   unmeasured epistemic mode") — one sentence, keeps the argument honest.
6. One-line rules for: stage-1 menu overflow (group by primitive above ~8 surfaces),
   and anti-repeat vs post-refit recapture (repeat allowed when the diff announces a
   fit change).
7. Fix the 30-pp attribution (embedding scorer, not LLM chooser) wherever
   How-Many-Tools is quoted.
