# Bonaventura, Feixas, Sbert, Chuang & Wallraven — "A Survey of Viewpoint Selection Methods for Polygonal Models" (Entropy 2018, 20(5):370)

Source: full text obtained via Europe PMC full-text XML (`PMC7512891`, open access,
CC-BY), DOI `10.3390/e20050370`. MDPI's own site (`mdpi.com/1099-4300/20/5/370`) and
NCBI PMC (`pmc.ncbi.nlm.nih.gov`) both blocked automated fetch (403 / reCAPTCHA);
Europe PMC's REST API (`ebi.ac.uk/europepmc/webservices/rest/PMC7512891/fullTextXML`)
served the complete JATS XML (263 KB) without blocking. Full paper read (all 6
sections + tables + equations), not abstract-only. Local extract: `/tmp/fulltext.xml`,
`/tmp/fulltext.txt`. Companion note in this same folder,
`dutagaci-best-view-benchmark.md`, covers the underlying Dutagaci et al. 2010
benchmark this survey re-uses as its ground truth — read together, not duplicated here.

## What this document actually is

A comparative survey + benchmark study, not a new method. It (1) collects 22
closed-form "viewpoint quality" measures from the graphics/vision literature
(Section 3), (2) reimplements all of them (except one) in one common framework, and
(3) scores all 22 against the Dutagaci et al. human-labeled best-view ground truth
(Section 4). This is a near-exact structural precedent for our RQ1 experiment: many
candidate scoring functions, one frozen external ground truth, ranked by error against
it.

## Core structure / classification (Section 3, corrects a mis-extraction risk)

**Five categories, not six** — matching the paper's actual section headings (3.2–3.6)
and Figure 2's five color-coded dots (area=red, silhouette=yellow, depth=purple,
stability=black, surface curvature=blue). Critically, **the four information-theoretic
measures (viewpoint entropy, I2, VKL, VMI/I1, I3) are *inside* Section 3.2 "Area
Attributes,"** not a separate category — they're all built on the same projected-area
"visibility channel" formalism (Section 3.1). Don't split them out as a 6th category
when citing this paper's taxonomy.

| # | Category (§) | Measures (VQ#) | Count |
|---|---|---|---|
| 1 | Area Attributes (§3.2) | # visible triangles (1), projected area (2), Plemenos & Benayada (3), visibility ratio (4), viewpoint entropy (5), I2 (6), VKL (7), VMI/I1 (8), I3 (9) | 9 |
| 2 | Silhouette Attributes (§3.3) | silhouette length (10), silhouette entropy (11), silhouette curvature (12), silhouette curvature extrema (13) | 4 |
| 3 | Depth Attributes (§3.4) | Stoev & Straßer (14), maximum depth (15), depth distribution (16) | 3 |
| 4 | Stability Attributes (§3.5) | instability / Jensen–Shannon (17), depth-based visual stability / NCD (18) | 2 |
| 5 | Surface Curvature Attributes (§3.6) | curvature entropy (19), visible saliency (20), projected saliency (21), saliency-based EVMI (22) | 4 |

Unified notation (§3.1): an information channel V→Z between viewpoints V and polygons
Z, built from `p(z|v) = a_z(v)/a_t(v)` (projected-area conditional), `p(v)` (viewpoint
importance, from normalized projected area), `p(z) = Σ_v p(v)p(z|v)`. Viewpoint
entropy, I2, VKL, VMI, I3 are all different functionals of this same channel — a
caution for Q3/Q7: **22 apparently-independent measures collapse to far fewer
independent feature bases** (area/visibility-channel statistics dominate 9 of 22; only
silhouette, depth, stability, curvature are genuinely distinct inputs). When counting
"how many independent candidates did we test" for our own metric family, count feature
bases (~5), not formula variants (22).

**Notably absent from the 22:** Kamada & Kawai's angle-between-view-direction-and-face-
normal measure — the literature's most direct analogue to our "orthogonal to the
inspected surface" criterion. It is cited only as pioneering background (§2, ref [18])
and is not among the 22 measures tested or in Table 2's list. This is a first-class
negative finding for Q7: **this survey's own candidate pool does not include a
surface-normal-alignment score at all** — none of the 22 measures directly reward "view
axis parallel to a specific surface's normal." The closest survivors (Plemenos &
Benayada, viewpoint entropy/I2, projected area) reward *whole-object* coverage/
informativeness, not *per-region* orthogonality. Adapting any of these to our Q7
per-primitive-region metric requires re-deriving a region-restricted version, not
just picking one off the shelf.

## Evaluation methodology (Section 4) — the RQ1 precedent

- Reuses Dutagaci et al.'s ground truth as-is: 68 models, 26 human subjects, "most
  informative view" web task (see companion note for full GT protocol/VSE machinery).
  This paper does **not** collect new human labels; it is entirely a downstream
  re-scoring exercise against a previously-published, frozen GT — a clean "different
  provenance" separation for Q3 (GT collected once by a different paper/team; 22
  candidate measures are all closed-form, non-learned, with zero shared authorship
  assumption with the GT-collection protocol).
- Rendering/benchmark harness parameters, stated explicitly (§4, first paragraph) —
  useful as a concrete precedent for what a reproducible view-scoring harness spec
  should commit to (relevant to Q20 "deterministic and seeded"):
  - Projection resolution 640×640 px; **no back-face culling**, polygons rendered from
    both sides.
  - Candidate sphere: **642 viewpoints**, built by recursive subdivision of an
    icosahedron (this survey's own re-implementation), sphere radius = 6× the radius
    of the model's smallest bounding sphere, camera FOV 19.2° (tuned so only the model
    + minimal background is visible).
  - Exception: the depth-based visual stability measure (VQ18) uses 128×128 px
    (compute-cost reduction) and a similarity threshold of 0.87.
  - Note the discrepancy with the *original* Dutagaci benchmark, which computed its
    own 7 algorithms on a **258-viewpoint** octahedron-subdivision sphere (§2,
    Background, and matches the companion note). This survey's 642-viewpoint
    reimplementation is a **denser candidate sphere at the same GT**, evaluated with
    the same VSE-style scoring — direct historical precedent for Q5's question of
    whether a denser uniform sphere baseline must be reported at matched view budget:
    here, sphere density was silently changed between the original benchmark paper and
    this survey's use of it, and both are treated as valid without re-justifying the
    density choice. Worth flagging as something to *not* replicate uncritically —
    density change is a confound the original authors don't discuss.
  - Framework released publicly: `github.com/limdor/quoniam` (ref [54], accessed by
    the authors 24 March 2018 — check liveness before citing as available).
- Figures 1 & 3 illustrate 3 example Dutagaci models: Stanford Armadillo (17,296
  triangles), a cow (23,216 triangles), Stanford dragon (26,142 triangles) — none are
  primitives or primitive compounds (consistent with the companion note's finding that
  the Dutagaci set has no primitive-fitting relevance for Q11).

## Results (Figure 2, Section 4 text) — the actual numbers

No table of exact error values is printed in the text — the reported result is **two
box plots** (median-ordered top, mean±std-ordered bottom) over the 68-model error
distribution per measure. However, the values ARE extractable: I fetched the actual
Figure 2 JPEG from PMC's CDN (`cdn.ncbi.nlm.nih.gov/pmc/blobs/201c/7512891/70e4fec0901f/
entropy-20-00370-g002.jpg`, since the naive `.../bin/entropy-20-00370-g002.jpg` guess
404s — the real asset path has to be scraped from the article HTML's `<img src>`) and
read it directly. Approximate values off the mean±SD panel (bottom plot, error axis
0–1, best→worst by mean):

| Rank | Measure | Mean error (approx.) | ±SD band (approx.) |
|---|---|---|---|
| 1 | Projected saliency (VQ21) | ≈0.38 | [0.23, 0.53] |
| 2 | # visible triangles (VQ1) | ≈0.38 | [0.23, 0.54] |
| 3 | Viewpoint entropy / I2 (VQ5/6) | ≈0.385 | [0.23, 0.54] |
| 4 | Curvature entropy (VQ19) | ≈0.39 | [0.22, 0.56] |
| 5 | Plemenos & Benayada (VQ3) | ≈0.41 | [0.23, 0.59] |
| 6–18 | (visible saliency, instability, max depth, I3, visibility ratio, depth distribution, saliency-EVMI, silhouette entropy/curvature/extrema, projected area, Stoev–Straßer, silhouette length) | ≈0.43–0.47, heavily overlapping | mostly [0.25–0.30, 0.60–0.65] |
| 19 | VKL (VQ7) | ≈0.48 | [0.30, 0.67] |
| 20 | Depth-based visual stability (VQ18) | ≈0.485 | [0.30, 0.68] |
| 21 | VMI / I1 (VQ8) | ≈0.57 (clear outlier, worst) | [0.41, 0.73] |

Read with appropriate caution — these are pixel positions read off a chart, not
publisher-supplied numbers, so treat as ±0.01–0.02 precision, good enough for
relative comparison but not for citing as exact published figures. Two takeaways
worth having as numbers rather than just rank order: (1) **the gap between best
(≈0.38) and the bulk of the mid-pack (≈0.43–0.47) is small** — roughly 0.05–0.09 on
a [0,1] scale — meaning most of the 22 measures are statistically close, only the
tails (VMI worst, at ≈0.57, standing well outside everyone else's SD band) are
clearly separated; (2) **error never gets close to 0 for any measure, even the
best** — floor is ≈0.38, not because Dutagaci's benchmark accepts an easy target,
but consistent with the companion note's observation that human "best view" picks
are themselves inconsistent across the 26 subjects (Figure 1 below), which puts a
floor on how low *any* algorithm's error against a noisy multi-rater GT can go —
relevant to Q12's effect-size question: don't expect our own candidate metrics to
approach zero error against a human/analytical GT either, and calibrate "meaningful
difference between methods" against this ≈0.38 floor, not against 0.

This is on top of the exact ranking statement (quoted, §4 final paragraph):

> "if we rank the measures in terms of mean and median, the sets of the five best ones
> are the same: projected saliency [35], the number of visible triangles [19],
> viewpoint entropy and I2 [21,41], curvature entropy [13], and Plemenos and Benayada
> [19]. Observe also that the five best measures belong to two categories: area
> attributes and surface curvature attributes. In contrast, the measures from the
> silhouette attributes category perform poorly."

- **Viewpoint entropy and I2 are reported as a single grouped result** because "they
  have the same performance" (proven algebraically: `I2(v;Z) = H(Z) − H(Z|v)` and
  `H(Z)` is constant per mesh, so I2's minimum always coincides with viewpoint
  entropy's maximum — Eq. (8), §3.2). This is a reusable technique for Q7: two
  differently-normalized measures can be *provably* rank-equivalent on a fixed mesh —
  worth checking before treating two candidate scores as independent evidence.
- Top-5 span only 2 of 5 categories (area, surface curvature); silhouette attributes
  are called out as the worst-performing category, with a proposed explanation
  (§4, penultimate paragraph): silhouette-only 2D contour properties may not access the
  same "volumetric primitive" representation the visual system uses for recognition,
  citing Biederman-style part-based object representation (refs [55,56]).
- No confidence intervals, no per-model breakdown table, no statistical significance
  test reported anywhere (Q12 relevance: this is a negative precedent — same gap the
  companion Dutagaci note already flags for the underlying benchmark, now inherited
  unchanged by this 22-measure re-scoring; box plots are the only variance disclosure).

## Discretization sensitivity — the load-bearing finding for Q2/Q9

Table 2 (column 7, "V/D") marks each of the 22 measures Y (sensitive) or N
(insensitive) to polygonal discretization. **Corrected count from the actual table:
10 of 22 are Y, 12 are N** (not an even 11/11 split — verify against Table 2 directly
if re-citing).

- Y (sensitive, 10): # visible triangles (1), Plemenos & Benayada (3), viewpoint
  entropy (5), I2 (6), I3 (9), instability (17), curvature entropy (19), visible
  saliency (20), projected saliency (21), saliency-based EVMI (22).
- N (insensitive, 12): projected area (2), visibility ratio (4), VKL (7), VMI/I1 (8),
  silhouette length (10), silhouette entropy (11), silhouette curvature (12),
  silhouette curvature extrema (13), Stoev & Straßer (14), maximum depth (15), depth
  distribution (16), depth-based visual stability (18).

Exact quotes (§3.2, "Viewpoint entropy" and "Information I2" paragraphs):

> "Viewpoint entropy is sensitive to polygonal discretization as in general the
> entropy increases with the number of polygons."

> "An important drawback of viewpoint entropy is that it goes to infinity for finer
> and finer resolutions of the mesh [35], while I2 presents a more stable behavior due
> to the normalizing effect of H(Z) in Equation (8)."

> "Viewpoint Kullback–Leibler distance is near insensitive to polygonal
> discretization." (§3.2, VKL paragraph)

> "VMI is insensitive to the discretization of the model." (§3.2, VMI paragraph)

> "The measure I3 is sensitive to polygonal discretization." (§3.2, I3 paragraph,
> closing sentence)

All silhouette measures are blanket-declared insensitive as a category (§3.3 intro):
"All these measures are insensitive to the discretization of the model because the
polygons are not directly used." Depth attributes similarly insensitive as a category
(§3.4, Stoev & Straßer paragraph: "insensitive to polygonal discretization because the
projected area and the maximum depth are insensitive too").

**Transfer to Q2/Q9 (mechanism, not just the fact):** the discretization axis in this
paper (mesh triangle count / tessellation density) is the closed-mesh analogue of
*point density* in a survey cloud. A raw Shannon-entropy-style score computed over
segmented point-cloud regions (e.g., entropy of a histogram of visible points per
cluster) would inherit the same failure mode viewpoint entropy has here: it would
trend upward simply because a denser-sampled region of the cloud is present, not
because the region is more informative — exactly the kind of confound our fused-vs-
single-view offset and dropout-driven density variation (Q9, Q10) would inject. The
paper's own fix — normalize by `H(Z)` (I2) or use a KL/mutual-information form (VKL,
VMI) that's provably insensitive — is a concrete, citable design constraint for any
information-theoretic candidate in our own Q7 metric family: **prefer VMI/VKL-style
normalized forms over raw viewpoint-entropy-style forms if the underlying point/region
density is not controlled**, which for a real survey cloud with dropout it never is.

## Figures 1 and 3, viewed directly — GT is a region, not a point; so are most scores

Both figures are image-only content (no numeric data in the surrounding text), fetched
as JPEGs from the PMC CDN and viewed directly.

**Figure 1** (`.../8249fd872b16/entropy-20-00370-g001.jpg`) plots, for 6 of the 68
models (armadillo, cow, dragon + 3 more), every one of the 26 human subjects' selected
"best view" as a red wireframe camera frustum around the model. Direct visual finding,
load-bearing for Q2 and Q8: **the 26 human picks are never a single point — they form
a cluster/fan spanning roughly 20–40° of angular spread**, and for at least one model
(the cow, panel e) the picks are **bimodal**: a dominant cluster from one side plus a
distinct smaller cluster from below. This is the clearest available evidence in this
source that "best view" ground truth, even collected the most direct way possible (ask
26 humans, no algorithm in the loop), is inherently **a set/distribution**, not a delta
function — any benchmark scoring a candidate view against this kind of GT needs an
angular-tolerance-region or set-membership criterion, and needs to handle genuine
multi-modality (not just symmetry) as a possible outcome, not an edge case.

**Figure 3** (`.../b5f895681445/entropy-20-00370-g003.jpg`) shows, for 3 models ×
21 measures (VQ20 excluded from this figure), the selected best view plus a full
642-viewpoint hemisphere heatmap (red = best, blue = worst) of that measure's score.
Direct visual finding, load-bearing for Q7 (per-primitive analytical ceiling): **for
the great majority of measures the heatmap is one broad, smooth, unimodal red
"plateau" covering roughly a third to half of the visible sphere** — not a sharp,
narrow peak. Two measures visibly break this pattern: I3 (VQ9, panel h) gives a
narrower, more textured hot region; depth-based visual stability (VQ18, panel q) gives
a near-binary landscape (large uniform red region interrupted by isolated blue-dot
artifacts), consistent with its thresholded neighbor-counting definition (Eq. 22–23,
§3.5) rather than a smoothly varying score. Implication: on real, non-primitive,
organic meshes, most of these measures don't have a single sharp optimum — "good" is a
region. This cuts two ways for our Q7 metric design: (a) it supports building our
primitive-level analytical GT as a **tolerance band around the face normal / cylinder
radial direction**, not an exact point, matching what empirical measures naturally
produce; but (b) it means we should not expect an empirical/learned scorer to
spontaneously discover a sharp, primitive-face-aligned optimum on its own — the
sharpness has to come from the geometry-aware GT construction itself (Branch 1/RQ1's
analytical track), not from generic viewpoint-quality measures like these 22.

## Load-bearing assumptions

- All 22 measures are computed on a **clean, single, closed watertight mesh** — no
  scanning defects, no injected sensor noise, no distinction between a "survey cloud"
  and a "ground-truth mesh." Discretization sensitivity here means *triangle count of
  the same underlying clean shape*, not *noise/dropout/registration error in a scanned
  reconstruction*. Extending the discretization-sensitivity table's Y/N labels to
  "which of these measures is robust to *scan defects*" is an extrapolation this paper
  does not make and does not test — it is a plausible hypothesis (§ above), not a
  verified transfer.
- The benchmark GT criterion is whole-object "most informative view for recognition,"
  identical in construct to the companion Dutagaci note's finding — **not**
  orthogonality-to-a-specific-surface-region. This paper inherits that mismatch
  unchanged; none of its 22 measures were designed or validated against a per-region
  orthogonality criterion.
- All 22 measures are single-viewpoint, single-shot scores over the *whole visible
  object* from one camera pose — there is no per-region/per-face decomposition inside
  any of the 22 formulas. Applying one of them to "how orthogonal is this view to
  *this particular* surface region" requires restricting the polygon set Z to the
  region's polygons before computing the score — an extension the paper never
  performs or evaluates.
- The public framework (`github.com/limdor/quoniam`) reimplements 21 of 22 measures
  in one codebase for a fair, consistent comparison; the visual/mesh saliency measure
  (VQ20) is the one exception, using Dutagaci's own original implementation instead
  (§4, first paragraph) — a caveat if reusing VQ20 specifically ("we have used
  Dutagaci's implementation... the only measure not included in the framework").

## What transfers to our setting

- **Direct methodological template for Q1/Q3/Q7:** many closed-form candidate scores,
  one frozen, independently-collected, human-labeled GT, ranked by error against it,
  reported as a box-plot distribution over the object set rather than a single number.
  This is close to exactly RQ1's shape, just at whole-object granularity instead of
  per-(object, surface-region).
- **The discretization-sensitivity mechanism and Table 2's Y/N framework (§3.2, Table
  2 col. 7) is directly reusable scaffolding for a "which candidate metrics are robust
  to our specific sensor defects" table (Q9).** We should build the analogous table
  for our own candidate metric family, but with columns for *our* defect axes (pose
  error, dropout, fused-vs-single-view offset) instead of mesh tessellation — this
  paper's structure is the template even though its specific Y/N values don't transfer.
- **The provable-rank-equivalence check (viewpoint entropy ≡ I2 by construction, §3.2
  Eq. 8) is a concrete technique to apply to our own metric family before treating
  multiple candidates as independent evidence in Q7/Q9** — check algebraically whether
  two candidate orthogonality scores are forced to agree before reporting them as
  separately-validated.
- **Rendering/harness parameter disclosure (§4 first paragraph) is a concrete
  reproducibility template for Q20** — resolution, culling policy, sphere radius
  relative to bounding sphere, FOV, per-measure exceptions all stated as exact numbers.
- **The "5 categories collapse 22 formulas" insight is directly useful for Q18's
  intake format** — when other branches report candidate metrics, tag each by
  underlying feature basis (area/visibility, silhouette, depth, stability, curvature,
  or new bases we introduce), not just by name, so we can spot when "5 new candidate
  metrics" are actually 1 feature basis in 5 notations.
- **Kamada & Kawai's angle-to-normal measure (§2, ref [18]), though excluded from the
  22, is the correct historical anchor to cite when introducing our own orthogonality
  metric** — it establishes that "angle between view direction and surface normal" is
  an old, well-known idea in this literature (1988), just one this particular survey's
  22-measure pool didn't carry forward, giving us a clean citation for the concept
  without claiming this survey validated it.

## What does NOT transfer, and why

- **No sequential/multi-view structure, exactly as in the companion Dutagaci note** —
  every one of the 22 measures scores one static viewpoint against a whole clean mesh.
  Q6 (view budget), Q10 (fused-vs-single-view offset), Q14 (oracle-selector vs. VLM),
  Q15 (view-correct × answer-correct grid) have no precedent here at all — this survey
  inherits that gap from Dutagaci and does not close it.
- **No injected sensor defects of any kind** (Q4, Q9, Q10) — "discretization
  sensitivity" is a mesh-resolution artifact of the clean reference model, not a noise/
  dropout/registration model. Do not cite this paper's Y/N sensitivity table as
  evidence about robustness to *scan* defects; it is evidence about robustness to
  *tessellation density* only, a different (if analogically related) axis.
- **No per-region / per-surface scoring anywhere.** All 22 measures are whole-object,
  single-viewpoint scores; none decomposes to "how good is this view of *this*
  surface." Any per-primitive-region analytical-optimum metric for Q7 must be
  constructed by us (e.g., restricting the polygon/point set to a region before
  applying one of these formulas), not lifted directly.
- **No orthogonality-to-surface measure is actually tested.** The single closest
  historical precedent (Kamada & Kawai, angle-to-normal) was excluded from this
  survey's own 22-measure evaluation — meaning even this paper, whose stated purpose
  is comparing viewpoint-quality measures, does not give us a validated "orthogonal is
  good" score with a known human-agreement track record. That validation gap is real
  and unfilled by this source.
- **No VLM or learned scorer anywhere** — same as the companion note, this predates
  and is silent on Q3's VLM-circularity concern and Q17's LLM-judge controls, beyond
  the generic "independent frozen GT + non-learned candidate scores" pattern already
  covered above.
- **Object set is the same 68 familiar, non-primitive objects as Dutagaci** — no
  additional relevance to Q11's stratified primitive/compound/near-primitive object
  set beyond what the companion note already establishes.
- **No statistics beyond box plots** (Q12) — same gap as the underlying benchmark,
  unaddressed by this survey's re-scoring exercise.

## Checkable citations

- 22-measure list, categories, formulas: Section 3 (§3.1 notation, §3.2–3.6 per
  category), Table 2 (master list with V/D discretization-sensitivity column and H/L
  optimum-direction column), Table 1 (notation glossary).
- Kamada & Kawai angle-to-normal pioneer measure (excluded from the 22): Section 2
  (Background), citing ref [18], Kamada & Kawai 1988, *Comput. Vis. Graph. Image
  Proc.* 41:43–56.
- Discretization sensitivity quotes: Section 3.2, "Viewpoint entropy" paragraph
  (entropy sensitivity), "Information I2" paragraph (I2 stability, "goes to infinity"
  quote), "Viewpoint Kullback–Leibler distance (VKL)" paragraph, "Viewpoint mutual
  information (or I1)" paragraph, "Information I3" paragraph (closing sentence);
  Section 3.3 opening sentence (silhouette category blanket insensitivity); Section
  3.4, Stoev & Straßer paragraph (depth category insensitivity rationale).
- Benchmark protocol and harness parameters: Section 4, first two paragraphs (640×640
  px, no back-face culling, 642-viewpoint icosahedron sphere, 6× bounding-sphere
  radius, 19.2° FOV, VQ18 exception at 128×128 px / 0.87 threshold), citing Dutagaci
  et al. [14] for the underlying 68-model/26-subject/error-metric machinery (see
  companion note for that machinery's own citations).
- Results ranking and quote: Section 4, final paragraph (top-5 list, category
  breakdown, silhouette-attributes-worst finding and proposed explanation citing
  [55,56]); Figure 2 (box plots, median-ordered top / mean±std-ordered bottom, 68
  models, 5-category color coding; approximate numeric error values in the table
  above were read directly off this figure's image, not from printed text).
- Human GT multi-modality: Figure 1 (6 of 68 models, 26 subjects' picks as red
  wireframe frusta; cow panel (e) shows a bimodal split).
- Viewpoint-quality-landscape shape (broad plateau vs. sharp peak): Figure 3
  (3 models × 21 measures, 642-viewpoint hemisphere heatmaps; VQ9/I3 and VQ18/
  depth-based visual stability are the visible exceptions to the broad-plateau
  pattern).
- I2 ≡ viewpoint-entropy rank-equivalence proof: Equation (8), Section 3.2,
  "Information I2" paragraph.
- Public framework and VQ20 implementation exception: Section 4, first paragraph, ref
  [54] (`github.com/limdor/quoniam`, accessed 24 March 2018).
- Conclusions / five-shortlist / three-of-five-information-related claim: Section 6
  (Conclusions), final paragraph.
- Original Dutagaci 258-viewpoint sphere (contrast with this survey's 642): Section 2
  (Background), final paragraph, citing Dutagaci et al. [14].
