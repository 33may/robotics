# Cyr & Kimia — "A Similarity-Based Aspect-Graph Approach to 3D Object Recognition"

IJCV 57(1):5–22, 2004. DOI 10.1023/B:VISI.0000013088.59081.4c. Earlier conference
version: Cyr & Kimia, "3D Object Recognition Using Shape Similarity-Based Aspect
Graph," ICCV 2001, pp. 254–261, Vancouver.

## Access note — read carefully before trusting anything below

**Full text was not obtained.** This is a load-bearing limitation for the whole
note and the task brief's "cite exact sections/figures/equations so claims are
checkable" instruction **cannot be met** for this document — there is no page or
section number below that I have personally verified against primary text.

What was tried, all unsuccessful: Springer/`link.springer.com` (redirects into an
IdP cookie-auth loop that never resolves, even for the abstract-only page — this
is unusual; typical Springer abstract pages are open, this one gated); DOI
resolver (same loop); Unpaywall API (`is_oa: false`, `oa_locations: []`, no
repository copy indexed); Semantic Scholar API (abstract field explicitly elided
by publisher, `openAccessPdf` empty); ResearchGate (two candidate pages, both
403); IEEE Xplore (conference-version document page returned no content);
Brown University LEMS lab site (`vision.lems.brown.edu` — the historical
publications host — fails DNS resolution from this environment; its successor
`vision2.lems.brown.edu` exists but no longer carries a `/publications` archive,
only current course/software pages); Brown Digital Repository (Cyr's PhD work
predates Brown's 2008 electronic-thesis mandate, so it is not archived there);
`web.archive.org` (blocked at the tool level in this environment, cannot be
fetched at all); CORE.ac.uk (403). No sci-hub or shadow-library route was used,
per policy.

What **was** obtained: the paper's bibliographic metadata (DBLP, confirmed
closed-access) and a convergent, multiply-corroborated description of the
method's *shape* — the same characterization (view sphere sampled at 5°,
dissimilarity metric, region-growing segmentation, prototype per aspect,
contrast with "edge-detection" visual-event computation) appeared verbatim or
near-verbatim across independent tool queries against Springer's own indexed
abstract, Semantic Scholar's summary, and the ICCV 2001 abstract — this is
consistent enough across independently-queried sources that I treat the
*method description* below as reliable. One query additionally surfaced what
appears to be a results/numbers fragment (64-object database, two similarity
metrics, top-3 recall figures) — this is **single-source, unverified against a
second independent citation or the primary text**, and is flagged as such
below; treat it as indicative, not confirmed. No verbatim quotations are given
anywhere in this note because none could be checked against original pagination.

Given this, the note below is organized to extract maximum defensible value from
metadata + convergent secondary description, while being explicit everywhere
about confidence level. This access failure is itself a data point worth
carrying into the synthesis (see Assessment).

## Core method/idea (high confidence — convergent across independent sources)

The paper is a **recognition** method, not a view-planning method: given an
unknown 2D image of an unknown-pose object, identify the object and its pose by
comparing the image's shape against a stored library of prototype views. The
aspect-graph machinery exists to build that library efficiently, and this is the
part relevant to our branch.

The construction pipeline, as consistently described:

1. **Uniform dense sampling, not exact visual-event computation.** The view
   sphere around each object is sampled at **regular 5° intervals** — a
   conventional uniform grid, the same category of structure as our current
   `{h,v}` sphere, not an object-conditioned tessellation. This is the paper's
   explicit rejection of the classical aspect-graph construction path (compute
   visual-event surfaces exactly, partition the sphere by them) in favor of
   discretize-then-cluster.
2. **Per-view shape description.** Each sampled view is reduced to a 2D shape
   descriptor of the object's (segmented) occluding contour — two variants are
   used across the conference/journal work: a curve-matching-based descriptor
   and a shock-graph-based descriptor (both are Kimia-lab shape-matching
   machinery from unrelated prior work, not generic/off-the-shelf).
3. **Pairwise view-to-view dissimilarity metric.** The shape descriptors give a
   distance between any two sampled views' silhouettes — this is the metric
   that "endows the view sphere with a metric of dissimilarity."
4. **Region-growing segmentation into aspects.** Rather than the usual
   "edge-detection" style of aspect-graph computation (find the boundaries
   first, from visual-event theory), the algorithm runs a **region-growing /
   clustering** pass directly over the dissimilarity-scored sample grid:
   adjacent samples are merged into a growing aspect as long as the aspect
   stays internally similar; growth is constrained so that any two resulting
   aspects remain mutually distinguishable under the same metric (a
   separation/distinctness stopping condition, not just a within-cluster
   compactness one).
5. **One prototype view per aspect.** Each resulting aspect is represented by a
   single "most typical" prototype view, and it is this small set of
   prototypes (not the full 5°-sampled grid) that is stored and matched
   against at recognition time.
6. **Recognition = match query against prototypes.** An unknown image is scored
   against the full library of stored prototype views (across all objects)
   using the same shape metric; best match gives object identity and
   approximate pose.

This is exactly the "reformulation that survived the critique" framing the
document was selected for: it produces an aspect-like partition of the view
sphere **without** ever computing an exact visual event, **without** assuming a
polyhedral object (the shape metrics used are for smooth/segmented occluding
contours, not polyhedral edge topology), and **without** the combinatorial
explosion of the classical construction — because the node count is
upper-bounded by the number of *samples* (a fixed, chosen constant), and
clustering can only ever produce fewer aspects than samples, never more.

## Concrete results and numbers

**Reliable (metadata-level, cross-corroborated):**
- Conference version (ICCV 2001): tested on a database of **18 objects**, each
  "viewed in five degree increments along the ground viewing plane" — i.e., in
  the conference version the sampling described is explicitly a **1D azimuthal
  sweep at fixed elevation** (a circle of ~72 views), not the full 2D view
  sphere.
- Journal version (this paper, IJCV 2004): describes sampling the full **view
  sphere** at 5° — i.e. the journal version is the 2D generalization of the
  conference version's 1D sweep. (Inferred from the consistent "view sphere"
  wording across sources vs. the conference abstract's explicit "ground viewing
  plane" wording — I could not confirm this distinction against primary text,
  flagging it as an inference, not a verified fact.)

**Unverified, single-source (flagged, do not cite without independent
confirmation):** a search-engine synthesis surfaced a results fragment
describing a **64-object database**, two similarity metrics (curve-matching and
shock-graph-matching), with **top-3 recall of (90.3%, 74.2%, 59.7%)** for the
curve-matching metric and **(95.2%, 69.0%, 57.5%)** for the shock-graph metric,
and **cumulative top-3 recall of 98% and 100%** respectively. This is plausible
in shape (matches the "64 objects, two metrics" structure expected of a journal
extension of an 18-object conference paper) but I obtained it once, via an AI
search summarization step, with no way to trace it back to a page I could
independently open and check. **Treat as indicative only.**

**Not obtained at all:** node/edge count of the resulting aspect graph (how many
aspects a typical object's 5°-sample grid region-grows into); build/compile time
for the region-growing pass; any accuracy or coverage metric that isn't
recognition-recall (e.g., nothing about view *quality* for inspection, nothing
comparing this aspect-sampling against a plain uniform grid on any non-recognition
task).

## Load-bearing assumptions (as best can be reconstructed)

- **A 2D shape-similarity metric between views exists and is meaningful.** Both
  variants used (curve matching, shock-graph matching) require a **segmented
  occluding contour** per view — i.e., you must be able to reliably extract the
  object's silhouette/contour from each of ~thousands of sampled views. In the
  paper's setting this is presumably done from **rendered views of a 3D model**
  (synthetic, clean silhouettes), consistent with the recognition-database use
  case (build once from a clean model, match against noisy real photos later).
  There is no evidence anywhere I could reach that this was ever attempted on
  noisy or incomplete real 3D data (fused point cloud, dropout, pose error) —
  it is a model-to-image recognition pipeline, and the "model" side is assumed
  clean.
- **The aspect count is not fixed in advance.** It is whatever the
  similarity-threshold-constrained region-growing produces for a given object —
  emergent, not chosen or bounded a priori beyond "at most (number of 5°
  samples)."
- **The similarity metric and region-growing threshold are shared across the
  whole database**, but nothing suggests aspect *labels*/prototypes are given
  any cross-object meaning — each object's aspect set is its own independent
  clustering result.
- **Purpose is recognition (image → object, pose), i.e. inverse/indexing
  direction**, same as the classical aspect-graph literature Faugeras et al.
  critique (see `faugeras-why-aspect-graphs-not-practical.md` in this
  directory) — the prototypes exist to be matched *against*, not to be
  *addressed into* by a downstream planner.

## What transfers to our setting

- **Directly answers Q11 (pruning mechanisms).** "Similarity merging of sampled
  views" is not just one candidate pruning mechanism among several — it is
  this paper's entire method, worked out in full: sample uniformly and densely,
  define pairwise dissimilarity, region-grow with a distinctness-preserving
  stopping rule, keep one representative per region. This is a genuinely
  implementable recipe if we can define an analogous view-to-view similarity
  metric on our PLY-derived renders (e.g., depth-image similarity, projected
  silhouette similarity, or surface-normal-consistency of the visible patch) —
  a concrete, non-hand-authored way to go from "uniform sphere" to "fewer,
  more-meaningful regions" without polyhedral assumptions or exact visual-event
  math.
- **Directly answers Q4's "similarity below threshold" adjacency candidate.**
  The paper's region-growing merge rule *is* a literal instance of that
  candidate: two neighbouring sampled views are "adjacent" (mergeable) exactly
  when their dissimilarity is below the growth threshold, and the resulting
  aspect boundary is wherever that stops holding. This is one of the cleanest,
  most computable adjacency definitions found for the branch — it needs no
  visual-event math, only a defined view-similarity function and a threshold.
- **Speaks to Q9 (which causes of aspect-graph death apply).** This paper is
  best read as a direct, working response to two of the four causes: (i)
  **size** — sidestepped, because node count is capped by sample count, not by
  algebraic object complexity, so it can never explode combinatorially the way
  exact visual-event enumeration can; (ii) **non-computability on curved/real
  input** — sidestepped, because no polyhedral assumption or exact visual-event
  surface computation is required, only a pairwise similarity function that
  works on any renderable/segmentable view. It does **not** address (iii) the
  indexing problem — see negative findings below — and says nothing about (iv)
  use-case compellingness beyond asserting recognition as the use case.
- **Confirms the selection rationale's framing is accurate**: "sample at 5°,
  define similarity, region-grow into aspects, no exact visual-event
  computation, no polyhedral assumption, no explosion" is a correct one-line
  summary of the method, not an oversimplification — this is genuinely the
  most directly implementable recipe of the classical aspect-graph lineage for
  a system (like ours) that already has a renderable/queryable 3D
  representation and wants regions without visual-event math.
- **Bounded build cost, by construction.** Because sampling is at a fixed
  angular resolution, the compute budget for the region-growing pass is
  bounded by the sample count, not by object complexity — back-of-envelope (my
  own arithmetic, not the paper's reported figure): a regular 5°×5° lat-long
  grid over the full sphere gives 37 colatitude rings × 72 azimuth steps, with
  the two pole rings collapsing to single points, i.e. **(37−2)×72+2 = 2522
  sample directions** for the full sphere, or roughly **half that (~1,260) if
  restricted to an upper hemisphere** (the typical case for a tabletop object
  not viewable from below). This is a concrete, checkable number for Q16/Q18's
  "what does it cost to build" question — the *sampling* cost is small and
  fixed; the unbounded part (not reported anywhere I could find) is the
  per-pair similarity computation if done pairwise-exhaustively over ~2,500
  samples (~3.2M pairs at full resolution) versus only-neighbours (cheap,
  since region-growing only needs adjacent-sample comparisons on a grid, which
  is the far more likely implementation and would make this genuinely cheap).

## What does NOT transfer, and why (negative findings)

- **Same directional mismatch as the Faugeras critique, and this paper does
  nothing to resolve it.** The whole point of the aspect graph here is to
  support *image → which prototype/object/pose* matching — the inverse
  direction. Our need is the forward direction (address/label → deterministic
  pose), which this paper never considers; it has no notion of "the agent picks
  an aspect ID and geometry resolves it to a camera pose." The paper's aspects
  are recognition targets, not navigation targets.
- **No bounded-cardinality guarantee — actively argues against one (Q13/Q14).**
  Aspect count is an emergent output of clustering with an internal
  threshold, not a chosen or predictable number. Nothing in what's accessible
  suggests aspect count is stable across objects of different complexity (a
  64-object database almost certainly does *not* produce the same aspect count
  per object) — this is direct evidence *against* being able to promise "always
  between N and M addressed views regardless of shape" using this mechanism as-is;
  the threshold would need to be object-adaptive to give any such guarantee,
  and nothing here addresses how.
- **No cross-object addressing consistency whatsoever (Q20).** Despite testing
  on a multi-object database (18 or 64 objects), each object's aspect
  partition and prototype set is its own independent clustering result with no
  claimed relationship to any other object's aspects — "aspect 3 of object A"
  carries zero meaning relative to "aspect 3 of object B." This is a genuine,
  clean data point supporting Q20's hypothesized gap: even a paper that
  explicitly processes many objects through the same pipeline never attempts
  cross-shape view-naming consistency; it only reuses the *pipeline*, not the
  *addresses*.
- **The similarity metric's precondition (clean segmented occluding contours,
  almost certainly from rendered synthetic models) is unlikely to survive our
  setting as-is.** Our geometry is a fused, noisy point cloud with real
  defects (pose error, depth dropout, ~4cm fused/single-view offset per the
  framing doc) — extracting a clean, stable occluding-contour shape descriptor
  from renders of *that* geometry is a nontrivial, unaddressed problem in this
  paper; it was designed and evaluated (as far as I can tell) against clean
  models, not noisy survey clouds. This paper gives no evidence either way
  about robustness to our kind of input — it simply never considers it.
- **No adjacency/ordering semantics survive beyond the merge step itself.**
  There's no evidence the paper retains or exposes a navigable graph structure
  (node IDs with next/previous/opposite relationships) for anything other than
  the internal clustering computation — the *output* that matters to the paper
  is a flat list of prototype views per object, not a graph an external agent
  could traverse. If we wanted the *aspect graph* (with edges) as a navigable
  structure for an orchestrator, this paper doesn't demonstrate that use case;
  it only demonstrates aspects as a recognition-database compression trick.
- **No benchmark against a uniform sphere on any view-planning-relevant task
  (Q19).** All reported evaluation (to the extent the unverified numbers above
  are trustworthy) is recognition recall — object+pose identification accuracy —
  not view quality, coverage, or orthogonality for inspection. This paper
  cannot be cited as evidence that geometry/similarity-conditioned view spaces
  outperform uniform ones for *our* task; it only speaks to recognition, and
  even there it doesn't compare against a "no-aspect, match against all 2500
  raw samples" uniform baseline as far as I could determine. This reinforces
  the branch's standing finding that the geometry-conditioned-vs-uniform
  comparison, on any task with a metric, appears to be genuinely unbuilt in
  the literature.
- **No seam/degenerate-boundary treatment found.** Nothing in what's
  accessible discusses what happens at the boundary between two aspects (is
  there a "neither" region, is the boundary itself addressable) — silent on
  Q7's seam question.

## Citations (checkability caveat)

**None of the following can be checked against primary-text page/section
numbers** — full text was never obtained (see Access note). What follows is
attribution to the *source of the claim as I encountered it*, not a page
citation into the paper itself:

- Method description (5° view-sphere sampling, dissimilarity metric,
  region-growing vs. edge-detection framing, prototype-per-aspect,
  distinctness-constrained growth): consistently reproduced across
  independent queries against Springer's indexed IJCV abstract, a Semantic
  Scholar paper summary, and the ICCV 2001 abstract text, via web search — not
  independently re-verified by me against a directly-opened primary page.
- 18-object database, "five degree increments along the ground viewing plane":
  attributed to the ICCV 2001 conference-version abstract, as surfaced via web
  search — not independently opened.
- 64-object database, top-3 recall figures: single web-search-synthesis
  result, unverified, explicitly flagged above — do not propagate into the
  synthesis without a second source.
- Sphere-sampling arithmetic (2522 / ~1260 directions at 5°×5°): my own
  computation, not from the paper.
- DBLP metadata (title/authors/venue/pages/DOI, closed access status):
  independently confirmed by direct fetch of the DBLP record.

## Assessment for the branch

This document earns its selection rationale on method description alone: it is
genuinely the most concrete, implementable non-classical aspect-graph recipe
found in this round — uniform dense sampling + pairwise view similarity +
region-growing with a distinctness stopping rule — and it directly answers
Q4's "similarity below threshold" adjacency candidate and Q11's "similarity
merging of sampled views" pruning candidate with a worked example rather than a
hypothetical. It also gives a clean, citable instance of Q20's cross-shape gap:
a paper that pipelines many objects through the identical procedure still never
attempts to make one object's aspect addresses mean anything relative to
another's. Where it fails us is exactly where the classical literature fails
us: it optimizes the *inverse* (image → node) direction we don't need, offers
no bounded-cardinality guarantee (actively cutting against Q14), and was built
and evaluated (as far as reconstructable) against clean rendered models, not
noisy survey clouds, so its core precondition (a reliable per-view shape
descriptor) is unverified for our PLY setting.

The larger, uncomfortable finding for the branch is procedural, not
substantive: **this specific paper could not be read.** Despite it being a
mainstream, 170+-citation IJCV paper from a major CV lab, no route — publisher,
DOI resolver, Unpaywall, Semantic Scholar, ResearchGate, IEEE Xplore, the
author's own lab site (dead link), Brown's institutional repository (predates
its coverage), or the Internet Archive (blocked at the tool level) — produced
the primary text. If other branch documents hit the same wall, the round-3
reading pass should flag paywall coverage as a systematic gap, not a one-off:
several of the "load-bearing recipe" candidates in this literature (pre-2010s
IJCV/PAMI/ICCV papers without arXiv mirrors) may simply be unreadable from this
environment, and future passes should check for open-access alternatives (arXiv
preprint, author PDF, thesis) *before* selecting a document as the anchor for a
branch question, or budget for requesting institutional access out-of-band.
