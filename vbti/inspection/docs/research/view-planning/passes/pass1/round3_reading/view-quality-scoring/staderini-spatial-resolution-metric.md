# Staderini, Glück, Mecca, Schneider, Kugi — "Spatial Resolution Metric for Optimal Viewpoints Generation in Visual Inspection Planning" (ICVS 2023, LNCS 14253, pp. 197–207)

DOI: 10.1007/978-3-031-44137-0_17 · Springer, Cham. Authors: Center for Vision,
Automation and Control, AIT Austrian Institute of Technology GmbH, Vienna +
Automation and Control Institute, TU Wien. Won the IEEE & AIT Women in
Engineering Best Paper Award at ICVS 2023.

## Access status — read this before trusting anything below

**Full text was not obtained.** This is a paywalled Springer LNCS book
chapter with, per Unpaywall (`api.unpaywall.org/v2/10.1007/978-3-031-44137-0_17`),
`is_oa: false`, `has_repository_copy: false`, `oa_locations: []` — i.e. there
is confirmed to be **no legal open-access copy anywhere** (no green OA
repository deposit, no author self-archive, no preprint). Avenues tried and
exhausted, in order:

1. Direct Springer chapter URL → 303 redirect to IdP institutional-login wall.
2. AIT institutional publications page (`publications.ait.ac.at`) → metadata +
   abstract only, no PDF, license marked "Other" (i.e., not open).
3. ResearchGate publication page → 403 Forbidden (request-full-text gate).
4. ACM Digital Library mirror (same DOI) → 403 Forbidden.
5. Semantic Scholar Graph API (`paperId 9fcb4e60914061ba1bb9a5f112081cdf8ce595f4`)
   → `openAccessPdf.status: "CLOSED"`, `abstract: null` (elided by publisher).
6. Unpaywall API → confirmed closed, zero OA locations (see above).
7. CORE.ac.uk, BASE (Bielefeld) → no accessible repository copy (BASE blocked
   the query outright via its anti-scraping gate; CORE returned nothing
   without an API key).
8. TU Wien reposiTUm (`repositum.tuwien.at`) — searched for Staderini; no
   match (her PhD thesis, if it exists in the repository, isn't indexed yet
   or doesn't include this chapter).
9. arXiv author listing for Staderini — no entries; this author group does
   not appear to post preprints.
10. Google Books preview of the LNCS 14253 volume — no snippet/search-inside
    available for this volume.

I did not attempt shadow-library / paywall-bypass routes (e.g. Sci-Hub) —
that is out of bounds regardless of technical feasibility.

**What follows is therefore built from three tiers of evidence, each marked:**
- **[CONFIRMED]** — the paper's own abstract and keyword list, which is
  reproduced *verbatim and consistently* across five independent aggregators
  (Springer's own summary, AIT's institutional page, ACM's listing, DBLP, and
  Semantic Scholar's metadata) — high confidence this is the actual abstract
  text, not a paraphrase drifting across sources.
- **[INFERRED]** — reasoned extrapolation from the companion predecessor paper
  by the *same five authors and same lab*, published months earlier in 2023:
  Staderini, Glück, Schneider, Mecca, Kugi, "Surface sampling for optimal
  viewpoint generation," ICPRS 2023 (Best Student Paper Award) — whose
  abstract/method summary I *was* able to retrieve via AIT's publications
  page. The ICVS abstract explicitly frames itself as extending "existing
  algorithms for optimal viewpoints generation" with a resolution term, and
  the ICPRS paper is that lab's own immediately-prior "existing algorithm" —
  so the ICVS paper is very likely this pipeline plus a density/resolution
  layer. Flagged inference, not read text.
- **[UNKNOWN]** — everything I could not establish at all: the actual
  resolution-vs-sensor-geometry formula, the sensor model used, test objects,
  numeric results, figures, ablations, equations. Do not cite these as if
  sourced from the paper.

## What is confirmed [CONFIRMED — abstract + keywords]

- **Problem framing**: "A fundamental requirement is to inspect the product
  with a certain specific resolution to detect all defects of a particular
  size, which is still an open problem."
- **Method**: "a novel model-based approach to automatically generate optimal
  viewpoints guaranteeing maximal coverage of the object's surface at a
  specific spatial resolution that depends on the requirements of the
  problem," done "by ray tracing information from the sensor to the object to
  be inspected once the sensor model and the 3D mesh of the object are
  known."
- **Explicit positioning claim against prior art**: "in contrast to existing
  algorithms for optimal viewpoints generation, their approach includes the
  spatial resolution within the viewpoint planning process." This is a direct
  claim that prior work (coverage/visibility-only viewpoint generation, e.g.
  their own ICPRS 2023 predecessor and the Gospodnetić et al. 2022 line from
  the same lab, already read separately — see
  `gospodnetic-viewpoint-placement-inspection-planning.md`) treats "coverage"
  as binary visibility and does *not* enforce a minimum achieved sampling
  density/resolution per surface point.
- **Result claim (abstract-level, no numbers given)**: the approach "yields
  optimal viewpoints that achieve complete coverage and a desired spatial
  resolution at the same time, while the number of optimal viewpoints is kept
  small, limiting the time required for inspection."
- **Keywords (the technical vocabulary of the method)**: *Ray tracing,
  Sampling density matrix, Set coverage problem, Spatial resolution,
  Viewpoint planning, Visibility matrix.* Six terms, each independently
  confirmed across all five metadata sources.
- **Load-bearing input requirement, stated directly in the abstract**: the
  method needs a **3D mesh** of the object plus a **sensor model** — not a
  raw or fused point cloud. This is the single most consequential fact for
  our transfer assessment (see below), and it is asserted plainly enough in
  the abstract itself that I count it as confirmed, not inferred.

## Inferred method shape [INFERRED — from the ICPRS 2023 companion paper]

The companion paper's own pipeline (as summarized by AIT's publications page):
Poisson disk sampling on the target mesh → translate each sample point along
its surface normal to a predefined focusing distance (this generates
oriented candidate viewpoints) → **sight ray tracing** to build a
visibility matrix (candidate viewpoint × mesh point/face, binary
visible/occluded) → **integer linear programming** to select the minimum
subset of candidates satisfying full coverage (a classic weighted set-cover
ILP), with Bayesian-optimization hyperparameter tuning on top.

Given the ICVS abstract's explicit "in contrast to existing algorithms... we
add spatial resolution" framing, and given the keyword pair **visibility
matrix / sampling density matrix** appearing side by side, the most
defensible reading is: the ICVS paper generalizes the *binary* visibility
matrix into a **continuous sampling-density matrix** — for each
(candidate-viewpoint, mesh-point) pair, instead of just visible/occluded, it
records the *achieved point/pixel sampling density* the sensor would obtain
at that surface point from that viewpoint — then reformulates the
viewpoint-selection ILP as a set-cover problem with a **per-point minimum
required density constraint** (rather than, or in addition to, plain binary
coverage), still solved by ray tracing against the known mesh + sensor model.

This is a plausible, literature-consistent extrapolation (this pattern — GSD
as a per-point achieved-resolution field over a mesh, computed via ray
tracing from a pinhole/projective sensor model and typically following a
cosine-law-style range/incidence-angle degradation — is standard in
photogrammetric coverage planning) but **it is not verified text**. The
actual formula (isotropic pinhole GSD? anisotropic active-stereo baseline
term? cosine-law incidence weighting? a hard per-point pass/fail resolution
threshold rather than a continuous field?) is unknown. Whether the paper
treats resolution and visibility as one combined matrix or two separate
matrices feeding one ILP is also unknown.

## What transfers to our setting (heavily caveated by the above)

- **Confirms, independently of the Gospodnetić et al. 2022 paper already
  read, that this same AIT lab's viewpoint-planning research line treats
  "achieved spatial resolution / GSD" as a first-class term worth folding
  into the viewpoint *selection* objective, not just gating admissibility.**
  This is mild but real corroborating evidence for RQ5's term table: a
  "sampling density / GSD" row belongs in the table as a PLY-space (mesh-
  space, strictly) term computed via ray tracing, with the *concept*
  (per-point achieved density must clear a per-defect-size threshold)
  citable to this abstract even without the formula.
- **The visibility-matrix / sampling-density-matrix pairing as a design
  pattern** (parallel per-viewpoint-per-surface-point matrices, one binary,
  one continuous, both from ray tracing) is a reusable structural idea for
  RQ11/RQ16's "how is visibility computed without a surface" question — but
  note this is the *same* ray-tracing-against-a-mesh pattern already
  documented from Gospodnetić et al., not new information; the only addition
  from this paper's title/abstract is that resolution can ride the same
  ray-traced-matrix pattern as visibility.
- **Reinforces the "needs-a-mesh-we-do-not-have" bucket for RQ10.** The
  abstract itself requires "the 3D mesh of the object" as an input — this
  method, whatever its internals, is not directly applicable to our fused
  survey PLY (no mesh, mm pose error, dropout holes) without a meshing step
  first, exactly the same negative-transfer pattern already established for
  the Gospodnetić et al. paper from the same lab.

## What does NOT transfer / cannot be assessed from what's available

- **Almost certainly does not answer RQ1's `sigma(range, θ)` depth-error
  curve.** "Spatial resolution" in this literature family means *achievable
  sampling/pixel density given range and sensor optics* (a geometric
  resolving-power/GSD concept — "can this view distinguish a defect of size
  X"), not *measurement noise* (how much does a reported depth value jitter
  at a given incidence angle). These are different failure modes with
  different physics (foreshortening/optics vs. correspondence-matching or
  return-signal noise). Even with full-text access this paper was probably
  never going to supply the D405 noise-vs-angle curve RQ1 asks for — it is
  the *resolving-power* half of "how well does this view resolve the
  region," not the *measurement-precision* half. This is a scoping
  correction worth carrying forward: the document was selected as "the
  clearest answer to how well does this view resolve the region," which is
  accurate for the sampling-density/GSD question (RQ5), but the branch's RQ1
  (sensor noise curve) needs a different source regardless of access.
- **Cannot confirm or deny smooth-vs-threshold angle behavior (RQ2)** — no
  visibility into whether the (inferred) density formula includes an
  incidence-angle term at all, let alone its shape.
- **Cannot address specular/off-normal-is-better (RQ3)** or
  **defect-detectability-vs-angle (RQ4)** — nothing in the confirmed abstract
  suggests either topic is in scope for this paper; it reads as a pure
  geometric-coverage-and-resolution planning paper (ray tracing against a
  mesh with a sensor model), with no mention of reflectance, material, or
  raking light.
- **No numbers of any kind are available**: sensor type (RGB-D? line-scan?
  area camera?), test objects, defect sizes targeted, standoff/range values,
  number of optimal viewpoints found, coverage percentage achieved, runtime.
  Every one of the branch's "concrete results and numbers" asks is
  unanswerable from what I could retrieve.
- **No equations, no figures.** The RQ5 term-table entry for "sampling
  density" that this paper should ideally supply a formula for cannot be
  filled in with a checkable equation — only the *concept name* and *input
  types* (candidate viewpoint pose, sensor model, mesh point + local
  geometry) are known.

## Recommendation

Given the branch flagged this paper specifically as "the clearest answer" to
the resolving-power question, the formula gap matters. Two ways to close it,
neither of which I could execute in this pass:

1. **Get real access** — check whether Anton's TU/AIT-adjacent network or a
   library has Springer LNCS access; failing that, ResearchGate's
   "Request full-text" button is live on the paper's page (a direct request
   to the authors) and AIT typically responds to academic requests.
2. **Read the ICPRS 2023 companion paper instead/first** ("Surface sampling
   for optimal viewpoint generation," Staderini et al., IEEE ICPRS 2023,
   DOI `10.1109/icprs58416.2023.10179043`) — it establishes the visibility-
   matrix/ILP mechanics this paper extends and may be easier to obtain via
   IEEE Xplore or the authors' ResearchGate profile; it will not have the
   resolution term itself but will ground the surrounding machinery so the
   ICVS paper's delta is easier to read once obtained.
3. Note in passing: a 2024 IEEE Access paper by Haneda et al.
   ("Viewpoint Generation Algorithm for Gimbal-Mounted Robots for Visual
   Inspection of Civil Engineering Structures," DOI
   `10.1109/ACCESS.2025.3546992`, **confirmed gold open access, CC-BY**)
   cites this Staderini paper and explicitly works on relaxing the
   orthogonal-viewing restriction under a data-quality-preservation
   condition — directly relevant to this branch's RQ2/RQ4 (smooth-vs-
   threshold angle penalty, defect detectability vs. angle) and fully
   accessible. It was not read in this pass (out of scope for a single-
   document reading task) but is flagged as a high-value, zero-access-cost
   follow-up read for those two RQs specifically.

## Branch-question coverage summary

Cannot directly answer any of the branch's numbered RQs with checkable
section/figure/equation citations, because the full text was never obtained
— this is the load-bearing limitation of this note. At the "informed
inference from abstract + companion paper" tier, it contributes: a
corroborating data point that GSD/sampling-density belongs in the RQ5 term
table as a ray-traced, mesh-space concept (name and input types only, no
formula); reinforcement of the "needs a mesh we don't have" classification
for RQ10 (the abstract itself states the mesh requirement); and a scoping
correction distinguishing "resolving power" (this paper's likely subject)
from "measurement noise" (RQ1's actual ask), which matters for how the
branch synthesis should describe this source. It does not inform RQ2, RQ3,
RQ4, RQ6–RQ9, RQ11–RQ25 at any confirmable level.
