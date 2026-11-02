# Kweon & Medeiros (1998) — "Part orientations for CMM inspection using dimensioned visibility maps"

Computer-Aided Design 30(9): 741–749. DOI: 10.1016/S0010-4485(98)00034-7.
https://www.sciencedirect.com/science/article/abs/pii/S0010448598000347

## Access note — read carefully before trusting any claim below

**The full text was not obtainable.** ScienceDirect returned HTTP 403 (paywalled abstract page). I checked
programmatically for any legitimate open-access copy:
- Semantic Scholar API (`DOI:10.1016/S0010-4485(98)00034-7`) → `openAccessPdf.status: "CLOSED"`, no PDF URL.
- Unpaywall API → `is_oa: false`, `oa_locations: []`, `has_repository_copy: false`.
- No author-hosted PDF, university repository copy, or ResearchGate/academia.edu full text found via search.
- No Penn State ETDA dissertation record found (Kweon's PhD would predate PSU's digital ETD system; this
  1998 journal article appears to be the only accessible trace of the work).

This is a genuinely closed 1998 Elsevier paper with zero OA footprint — not a fetch failure on my part. I did
not use unauthorized-access routes (e.g. Sci-Hub) to get around this.

**What the notes below are built from instead**, in decreasing order of reliability:
1. **Full text, directly read** of Sadaoui & Phan (2021), *"Touch Probe Measurement in Dimensional Metrology:
   A Review,"* Int. J. Automotive and Mechanical Engineering 18(2):8647–8657 (CC BY, open access) — a peer-reviewed
   survey that cites, summarizes, and tabulates Kweon & Medeiros [ref 35 in that paper] alongside seven other
   part-setup/orientation methods (their Table 2). This is the strongest evidence in these notes and every claim
   attributed to it is a direct read, not a search snippet.
2. **The paper's own abstract/summary text**, reconstructed by cross-checking multiple independent search results
   (web search engine indexing + Semantic Scholar `tldr`). The wording was consistent and specific enough across
   independent sources (not paraphrased identically by a single secondary source) that I treat it as a reliable
   reconstruction of the actual abstract, but it is *not* a verbatim-verified quote and contains no numbers,
   figures, or worked examples.
3. **Directly-cited foundational/companion papers** the target document's title explicitly invokes — Spyridi &
   Requicha's accessibility cones and Vafaeesefat & ElMaraghy's clustering — sourced the same way (search-engine
   reconstruction of abstracts; also behind paywalls, also not independently full-text-verified).

**Consequence for this branch:** I cannot report Kweon & Medeiros' actual algorithm steps, worked example part,
or any quantitative result (part count, orientation count, runtime) — those never surfaced anywhere accessible.
Every number-shaped branch question (Q6, Q7, Q13) gets **no answer** from this document specifically; the notes
below are scoped to what *is* independently verifiable: the method's problem framing, its place in the
CMM-inspection-planning literature, and its relationship to the two companion techniques named in the document
title.

## Core idea (as verifiable)

Kweon & Medeiros address: given a CAD part description and its dimensional/geometric tolerances, determine a
**minimal set of part orientations** on a CMM such that every tolerance can be inspected by some orientation in
the set — because a single orientation on the machine table almost never gives probe access to every feature
that needs measuring, and re-orienting/re-fixturing the part is the expensive step in CMM inspection planning
(setup change, not probe motion, is the dominant cost these methods are trying to minimize).

The mechanism (per Sadaoui & Phan's direct-read summary, Section "Part Setups," and the paper's own abstract
reconstruction):
1. For each tolerance/feature, compute its **visibility map (VMap)** — the set of accessible directions from
   which a probe can approach and measure that tolerance. VMaps were pre-existing machinery imported from the NC
   machining literature (accessible-tool-direction analysis), not new to this paper.
2. The paper's stated original contribution is **"VMap dimensionality"** — classifying/parameterizing each VMap
   by some dimensional property of its accessible-direction region (the search-recoverable text names the concept
   but I could not recover its precise definition — plausibly a 0D/1D/2D classification of the direction set on
   the Gaussian sphere, by analogy to how Spyridi & Requicha's accessibility cones are themselves regions on the
   sphere, but I am not able to confirm this from the source and flag it as inference, not a verified claim).
3. VMap dimensionality is then used to drive a **clustering heuristic**: group tolerances whose VMaps overlap
   (share accessible directions) into disjoint clusters, each cluster corresponding to one candidate part
   orientation. The heuristic's target is "a reasonably small number of part orientations" — i.e., a
   minimum/small-set-cover objective over orientations, not a fixed orientation budget.
4. Implemented in C++, tested on an SGI workstation (no further detail recoverable).

**Scope limitation, independently confirmed:** Sadaoui & Phan's Table 2 lists Kweon et al.'s disadvantage as
*"only applied for prismatic measurement parts"* — the method is scoped to prismatic (flat-face, box-like)
geometry, not cylinders or free-form surfaces. Their advantage column: *"reduce the number of part setups"*;
their principle column: *"based on visibility maps (VMap) and clustering procedure."* This is a direct read from
an open-access survey, so it is the most trustworthy single fact recovered about this paper.

## The two companion techniques named in the document title

### Spyridi & Requicha — accessibility cones (LAC/GAC)

Original: Spyridi, A.J. & Requicha, A.A.G., *"Accessibility analysis for the automatic inspection of mechanical
parts by coordinate measuring machines,"* Proc. IEEE ICRA 1990, Vol. 2, pp. 1284–1289. (Also behind a paywall;
not independently full-text-verified — reconstructed the same way as above.)

- Defines, per geometric feature, a **Local Accessibility Cone (LAC)** — the set of probe-approach directions
  that can reach that one feature without collision — and a **Global Accessibility Cone (GAC)** — the
  intersection/aggregation across *all* features of a part, giving directions valid for the whole part at once.
- Computed via **Gaussian images and Minkowski (sweeping) operations** — this is a continuous, analytic
  computational-geometry construction, not sampling or discrete candidate generation.
- The stated purpose of clustering these cones is explicitly to **"generate a minimal set of directions for
  inspecting a part"** — i.e., the same minimum-orientation-count objective as Kweon & Medeiros, one step earlier
  in the citation chain (1990 vs. 1998). Kweon & Medeiros's VMap approach is a direct descendant of this LAC/GAC
  machinery, per multiple secondary sources describing the field's genealogy this way.

### Vafaeesefat & ElMaraghy — accessibility analysis and measurement clustering

Vafaeesefat, A. & ElMaraghy, H.A., *"Automated accessibility analysis and measurement clustering for CMMs,"*
Int. J. Production Research 38(10): 2215–2231 (2000). Abstract (reconstructed, consistent across sources):
"presents a methodology to automatically define the accessibility domain of measurement points and then group
them into a set of clusters. The methodology uses the CAD model of the workpiece and tolerance information as
input to an algorithm for defining point accessibility. For each measurement point, the algorithm determines all
feasible inspection probe orientations, as a subset of the total set of available orientations for a given CMM
probe."

- Same two-stage recipe as Kweon & Medeiros: **(1) per-feature/per-point admissible-direction computation, (2)
  clustering to minimize orientation count** — but operating on discretized measurement *points* on the surface
  (from an STL/facet representation) rather than analytic VMaps over whole tolerances. Sadaoui & Phan's Sect.
  "Probe Orientation Analyses" (direct read) confirms this: "Vafaeesefat et al. [47] use the notion of
  accessibility domains in all points of the surfaces (vertices of facets in an STL file). Once the accessibility
  of a point is defined, the feasibility of measuring in any probe direction can be checked by Boolean operations.
  A grouping algorithm is proposed to classify directions according to accessibility domains."
- This is closer to our operating regime than Kweon & Medeiros or Spyridi & Requicha: it works from a
  **facet/vertex representation** (structurally similar to a point cloud/mesh) rather than requiring exact
  analytic CAD primitives, and it explicitly uses Boolean set operations on per-point direction sets, which is a
  discretized analogue of the LAC/GAC intersection.

## What this citation chain establishes for our branch questions (with confidence level marked)

**Confirmed at survey-level (direct read, Sadaoui & Phan 2021):**
- **Precedent chain for "cluster features that share directions into few orientations" (branch Q5):** genuine,
  three-deep, and dates to 1990. Spyridi & Requicha (1990, accessibility cones, continuous/analytic) →
  Vafaeesefat & ElMaraghy (2000, discretized point-accessibility + clustering) → Kweon & Medeiros (1998, VMap
  dimensionality + clustering) are three independent groups converging on the same two-stage recipe:
  *per-feature admissible-direction set, then cluster/intersect to minimize the count of distinct orientations.*
  That convergence, industrially adopted for CMM inspection planning (a mature, blessed practice, not academic
  vaporware), is legitimate precedent for "one orthogonal view per face, a ring per cylinder" as an instance of
  the same general principle applied to a much smaller, coarser instance (primitive faces/cylinders instead of
  individual tolerances/points). Caveat: none of these three papers name or derive "one view per face" as a rule
  — they solve the general clustering problem and a box-face-orthogonal answer would fall out of it as a special
  case for prismatic parts, not as a stated recipe. Cite the chain for the *principle*, not for the specific rule.
- **Sadaoui & Phan's Table 2 also independently surfaces a fourth, directly relevant method for branch Q13**
  (stopping/minimality criteria): *"Hwang et al. [3] introduced a greedy heuristic procedure... based on the
  Chvátal's greedy heuristic"* to minimize the number of part setups. This is the classical greedy set-cover
  algorithm (Chvátal 1979) being applied, by name, to exactly this class of problem (minimize orientations
  covering all features) in the CMM literature (Hwang, Tsai & Chang, *"Efficient inspection planning for
  coordinate measuring machines,"* IJAMT 23(9):732–742, 2004). This directly substantiates branch Q13's listed
  candidate "Chvátal's greedy termination" as a real, applied precedent in a structurally identical problem —
  not just a textbook reference. Table 2's advantage/disadvantage columns for it: *"limit number of part setups,
  easy to implement"* / *"applied for prismatic measurement parts"* (same scope restriction as Kweon & Medeiros).
- **The field frames the objective as minimum set cover, not budgeted k-coverage (branch Q1):** every method
  in Sadaoui & Phan's Table 2 (Hwang/Chvátal, Kamrani/GA+graphical, Stojadinovic/GA, Ziemian/heuristic,
  Kweon/VMap-clustering, Corrigall/PAD, Lai/rough+fine positioning) is described with the same goal —
  "limit/reduce/minimise the number of part setups" subject to a **coverage constraint** ("all tolerances under
  consideration") — never "given k setups, cover as much as possible." Minimum set cover is the field's
  unanimous framing when the shape class is well-understood (prismatic parts). This is a real, industrially
  converged data point in favor of treating our problem the same way (min-cover subject to a quality floor)
  rather than max-k-coverage, *if* our VLM's discrete-set pressure is better modeled as a soft minimality
  preference than a hard k.
- **Negative finding for branch Q4 (generate-candidates-then-select vs. constraint intersection):** none of these
  three CMM methods generate a discrete candidate pool and then select from it in the way our VLM action-space
  pipeline needs to. All three compute continuous admissible-direction regions (cones, VMaps, accessibility
  domains) and intersect/cluster them analytically. This is direct precedent for the "derive views from constraint
  intersections" alternative Q4 poses (closer to Magaña's C-space intersection than to a sample-then-score
  pipeline) — but note it is precedent for the *clustering-for-minimality* half of the pipeline, not for
  presenting a result to a downstream discrete-choice agent. None of these papers have a VLM-shaped consumer
  downstream; they hand the result to a CMM controller that executes every orientation in the output set
  deterministically. **The enumeration-for-presentability problem (RQ4) has no precedent here** — it is a genuine
  gap these classical methods never had to solve, because their "user" was a machine executing a fixed plan, not
  an agent choosing among options.
- **Companion signal on reachability/fallback (branch Q8, Q9):** one search result (not independently
  full-text-verified, so lower confidence) attributes to a related Medeiros paper on probe selection: *"for a CMM
  equipped with an indexable probe, the probe may be used instead of a fixture(s) to achieve some specified
  orientations if the orientations lie within the angular resolution of the probe,"* and separately describes
  a feature-accessibility algorithm that estimates **"percentage accessibility"** per orientation set rather than
  a hard yes/no. If accurate, this is a soft, quantized reachability model (probe angular resolution as a discrete
  feasibility grid, analogous to IK-solution multiplicity) plus a partial-coverage reporting metric — structurally
  close to what branch Q12 asks for ("best-available view per region plus an explicit unreachable-region report").
  I flag this as a lead worth chasing directly (title/DOI unresolved) rather than a settled fact.

## What does NOT transfer, and why (negative findings — first class)

1. **The physical target of "accessibility" is fundamentally different from ours.** These are all **tactile
   contact-probe** systems: a CMM stylus must approach a point and physically touch it without the probe shaft or
   probe holder colliding with the part. "Accessibility" here means *line-of-approach clearance for a rigid
   stylus*, computed from part geometry and stylus/holder dimensions alone. Our problem is **optical**: a camera
   needs a clear line of sight *and* adequate framing/field-of-view/working-distance/lighting for the surface
   region to be legible to a VLM, not just clearance for a rigid probe to touch one point. The accessibility-cone
   /VMap machinery answers "can a rigid stylus reach this point," which is necessary-but-not-sufficient for "is
   this a good image of this region" — it has nothing to say about framing, occlusion by other visible surfaces,
   or image-quality concerns that dominate our RQ1.
2. **No robot-arm kinematics or self-collision in the picture at all.** CMMs are Cartesian gantries (3 orthogonal
   linear axes, optionally + an indexable 2-DOF probe head) — there is no 6-DOF articulated-arm reachability
   problem, no IK multiplicity (branch Q11 — 8 UR5e solutions with different collision behavior — has literally
   no analogue in a Cartesian CMM), no joint-limit coupling between "where the camera points" and "where the rest
   of the arm's links are." The entire "reachability shell / capability map / IK-filtered candidates" cluster of
   branch questions (Q9–Q12) is native to articulated-arm eye-in-hand systems and has no precedent to transfer
   from CMM accessibility-cone literature beyond the general idea that *some directions are physically
   infeasible and must be excluded before or during selection* — the "before or during" distinction (capability
   map pre-filter vs. per-candidate IK post-filter) is not addressed by this citation chain at all, because a
   CMM's feasibility geometry is static and part-independent (fixed by machine architecture and probe/stylus
   dimensions), not workspace- and pose-dependent the way a 6-DOF arm's reachable shell is.
3. **The objective is contact-point accuracy under tolerance, not view "quality" or coverage of a defect.** These
   methods size and cluster orientation sets to guarantee every *specified GD&T tolerance* gets measured with an
   accessible, collision-free probe approach — a pass/fail metrology guarantee. There is no analogue in this
   literature to "how well does the view frame the region for defect detection" (our RQ1) or "orthogonality as a
   proxy for measurement/image quality along a continuum" — CMM accessibility is binary (reachable / not
   reachable within probe geometry), not scored/graded. Branch Q7's "maximum admissible deviation from
   orthogonality" question gets no numeric answer from this citation chain; accessibility cones bound a *feasible
   region* of approach angles, but the papers reconstructed here do not report a specific tolerance-driven
   deviation threshold in degrees, and I could not verify one from any accessible source.
4. **Prismatic-parts-only scope (confirmed disadvantage of Kweon & Medeiros, per Table 2) limits its direct
   reuse for cylinders.** Our setting explicitly needs cylinder-orbit rules too; this specific paper's method is
   stated to be inapplicable there, so it can only be cited for the box/face-orthogonal half of the "one
   orthogonal view per face, a ring per cylinder" precedent claim (branch Q5), not the cylinder half. The
   cylinder-orbit precedent needs to come from elsewhere (GS-NBV's 1D circle, per the branch's own framing, or
   facet-grouping-under-normal-deviation literature).
5. **No exact-optimization vs. heuristic verdict transfers cleanly (branch Q3).** All CMM-literature methods
   found here (Chvátal-greedy, GA-based, VMap-clustering, PAD-based) are **heuristics**, not exact ILP/FPT
   solvers, despite operating at instance sizes (a handful of features, a handful of candidate orientations per
   part) that are comparable to or smaller than ours. This is a mild negative data point against "small instance
   ⇒ obviously just solve it exactly": an entire mature industrial field with small instances still standardized
   on heuristics (mostly greedy/GA) rather than exact solvers — though this could equally reflect 1990s-2000s
   compute norms and unavailability of accessible commercial ILP tooling at the time, rather than a considered
   rejection of exact methods, so it should not be read as a strong argument against ILP/FPT for us. Flagged as
   weak/ambiguous evidence, not a verdict.

## Bottom line for the branch

This document is genuinely useful only as a **citation-chain anchor for Q5's precedent claim** (industrially
blessed, three-paper-deep convergence on "per-feature accessible-direction set → cluster to minimize orientation
count," 1990–2000) and as an **incidental source of a second, independently strong precedent for Q13**
(Hwang et al.'s named use of Chvátal's greedy set-cover heuristic in this exact problem class, surfaced only
because it appears in the same survey table). It contributes essentially nothing to Q6/Q7 (no recoverable
numbers), nothing new to Q8/Q9/Q11/Q12 beyond a single unverified lead, and its core machinery (tactile-probe
accessibility cones) does not transfer to our optical/eye-in-hand/VLM-consumed setting beyond the general
two-stage principle. Given the paywall, further investment reading *this specific paper* has poor ROI; the
Sadaoui & Phan (2021) survey (open access, already fully read for these notes) or Vafaeesefat & ElMaraghy (2000)
directly would be better next stops if more clustering-algorithm detail is needed, since the latter operates on
a facet/point representation closer to our point-cloud setting.

## Sources consulted (for traceability)

- Kweon, S. & Medeiros, D.J. (1998). Comput.-Aided Des. 30(9):741–749. DOI 10.1016/S0010-4485(98)00034-7.
  Abstract page only (403 on full text); Semantic Scholar + Unpaywall confirm no OA copy exists anywhere.
- Sadaoui, S.E. & Phan, N.D.M. (2021). "Touch Probe Measurement in Dimensional Metrology: A Review." Int. J.
  Automotive and Mechanical Engineering 18(2):8647–8657. DOI 10.15282/ijame.18.2.2021.02.0658. Open access
  (CC BY), full text read in full (11 pp., all sections/tables/figures/references) via
  `journal.ump.edu.my/ijame/article/download/5040/1349/23053`.
- Spyridi, A.J. & Requicha, A.A.G. (1990). Proc. IEEE ICRA, Cincinnati OH, Vol. 2, pp. 1284–1289. Abstract
  reconstructed via search only; not independently full-text-verified.
- Vafaeesefat, A. & ElMaraghy, H.A. (2000). Int. J. Production Research 38(10):2215–2231. Abstract reconstructed
  via search only; not independently full-text-verified.
- Hwang, C.Y., Tsai, C.Y. & Chang, C.A. (2004). "Efficient inspection planning for coordinate measuring
  machines." Int. J. Adv. Manuf. Technol. 23(9):732–742. Cited only via Sadaoui & Phan's direct-read summary
  (their ref [3]); not independently checked.
