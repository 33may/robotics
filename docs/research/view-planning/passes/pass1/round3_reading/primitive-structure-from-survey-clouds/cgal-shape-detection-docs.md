# CGAL Shape Detection — User Manual (CGAL 6.2)

- **Source**: https://doc.cgal.org/latest/Shape_detection/index.html (fetched as CGAL 6.2, "Shape Detection: User Manual", raw HTML pulled directly, section anchors below refer to `<h1>/<h2>/<h3>` ids on that page)
- **Type**: library user manual / reference documentation, not a paper. No experiments on our defect modes, no statistical model, no uncertainty treatment — read accordingly.
- **Authors / provenance**: Sven Oesau, Yannick Verdie, Clément Jamin, Pierre Alliez, Florent Lafarge, Simon Giraudot, Thien Hoang, Dmitry Anisimov (§`Chapter_Shape_Detection`, author list).
- **Underlying methods credited**:
  - Efficient RANSAC → Schnabel, Wahl & Klein, *"Efficient RANSAC for Point-Cloud Shape Detection"*, cite key `schnabel2007efficient` (§Introduction, ref [2]).
  - Region Growing → Lafarge & Mallet, cite key `cgal:lm-clscm-12` ("Creating Large-Scale City Models from 3D Point Clouds: A Robust Approach with Hybrid Representation"), ref [1].

## 1. Core method / what the library actually does

Two independent algorithms, both **require pre-computed point normals as input** — this package does not estimate normals itself.

**Efficient RANSAC** (§`Shape_detection_RANSAC`): "From an unstructured point set with unoriented normals, this algorithm detects a set of shapes." Supports exactly five primitive types out of the box: **plane, sphere, cylinder, cone, torus**. Loop: randomly sample minimal point sets → fit candidate shape → count inliers (distance + normal-deviation test) → repeat until probability of having missed the largest shape drops below a threshold → extract largest shape, repeat until no shape clears `min_points`. Each input point is assigned to **at most one** shape; unassigned points remain.

**Region Growing** (§`Shape_detection_RegionGrowing`): generic seed-and-grow framework (pick item → find neighbors → accept neighbors meeting region criteria → repeat → start new region when stuck). CGAL ships five instantiations: line/circle in 2D point sets, line in 2D/3D segment sets, **plane/sphere/cylinder in 3D point sets**, and **plane only on a polygon mesh** (no sphere/cylinder/cone on mesh). Custom item types/regions can be added by the user.

**Custom shapes** (§`Shape_detection_RANSACExample_with_custom_shapes`): Efficient RANSAC can be extended with a user-derived `Shape_base` (must implement `create_shape`, `squared_distance`, `cos_to_normal`, `minimum_sample_size`, optionally `connected_component`, `info`) — this is the mechanism by which anything beyond plane/sphere/cylinder/cone/torus (e.g. a superquadric, a box-with-fixed-orientation) would have to be added; **nothing like that ships with the library**.

## 2. Parameters — exact semantics (load-bearing for tuning on our cloud)

### Efficient RANSAC (§`Shape_detection_RANSACParameters`)

| Param | Exact meaning (quoted) |
|---|---|
| `epsilon` | "absolute maximum tolerance Euclidean distance between a point and a shape." Interacts directly with noise floor: "Over-segmentation translates into detection of fragmented shapes when epsilon is within or below the noise level." Free-form/non-primitive parts get absorbed into detected shapes if epsilon is raised: "a higher tolerance epsilon enables to detect more primitive shapes that approximate some of the free-form surfaces" — i.e. loosening epsilon does not reject non-primitive geometry, it silently approximates it as a primitive. |
| `normal_threshold` | Threshold on normal deviation between the point's normal and the shape's normal at the closest projection point; used as a `cos(angle)` gate (code example: `0.9 < dot(surface_normal, point_normal)`). |
| `cluster_epsilon` | Grid-cell spacing for a connectivity check on shape support: "two points separated by a distance of at most 2√2·cluster_epsilon are considered adjacent." For developable shapes (plane, cylinder, cone) this connectivity test runs in a distortion-minimizing 2D parameterization of the shape; for non-developable shapes it uses a 3D neighbor graph. This is the parameter that decides whether a shape's support has to be spatially contiguous — directly relevant to dropout gaps (see §5). |
| `min_points` | Termination threshold, **not strict**: "depending on the chosen probability, shapes may be extracted with a number of points lower than the specified parameter." |
| `probability` | Probability of missing the largest candidate shape per iteration; lower = more search effort, longer runtime, marginally higher coverage (see §4). |

**Exact default values** (from the `Efficient_RANSAC::Parameters` struct reference page,
`structCGAL_1_1Shape__detection_1_1Efficient__RANSAC_1_1Parameters.html` — not visible
via the summarized manual page, pulled from the class-reference doc directly):

| Parameter | Default | Valid range |
|---|---|---|
| `probability` | **0.05** | [0, 1] |
| `min_points` | **1% of total number of input points** | [0, ∞) |
| `epsilon` | **1% of the bounding box diagonal** | [0, ∞) |
| `normal_threshold` | **0.9** (≈ 25°) | [0, 1] |
| `cluster_epsilon` | **1% of the bounding box diagonal** | [0, ∞) |

All three geometric defaults **self-scale to the input's bounding box** — a materially
different design choice from Region Growing's point-set defaults below, which do not.

### Region Growing on point sets (§`Shape_detection_RegionGrowingPoints_parameters`)

| Param | Exact meaning |
|---|---|
| `sphere_radius` / `k_neighbors` | Neighborhood definition (fuzzy ball vs. K-NN) used by `NeighborQuery`. |
| `maximum_distance` | Max distance from a point to the region's current best-fit line/plane/sphere/cylinder. |
| `maximum_angle` | Max angle (degrees) between the point's normal and the shape's normal — this is the region-growing analogue of `normal_threshold`, but expressed in degrees, not cosine. |
| `minimum_region_size` | Min point count for a region to be kept — same over/under-segmentation trade-off as `min_points`, illustrated concretely: on a house roof, `maximum_angle = 60°` merges two roof planes into one; `maximum_angle = 25°` splits them correctly (Fig. 94.9, §`Shape_detection_RegionGrowingPoints_parameters`). Neighborhood size trade-off shown numerically on a 2D street-map cloud (Fig. 94.8): `sphere_radius = 0.1` → 17 regions, `0.3` → 8 regions, `1.2` → 4 regions (finer radius = more, smaller regions).

Region growing on a polygon mesh (§`Shape_detection_RegionGrowingMesh_parameters`) uses the same three (`maximum_distance`, `maximum_angle`, `minimum_region_size`) but distance is measured "from the furthest face vertex to a plane," and there is no `NeighborQuery` parameter (adjacency comes for free from mesh connectivity) — **not applicable to us since our input is a fused point cloud, not a mesh**.

**Exact default values for Region Growing point-set classes** (pulled from
`Least_squares_plane_fit_region`, `Least_squares_cylinder_fit_region`,
`K_neighbor_query`, `Sphere_neighbor_query` class-reference pages — again, not present
on the summarized manual page):

| Parameter | Default | Note |
|---|---|---|
| `maximum_distance` | **1** (absolute units) | **not** scale-relative to the input |
| `maximum_angle` | **25°** | `cosine_of_maximum_angle` = cos(25°) can be substituted directly |
| `minimum_region_size` | **3** | absolute point count |
| `k_neighbors` (`K_neighbor_query`) | **12** | |
| `sphere_radius` (`Sphere_neighbor_query`) | **1** (absolute units) | **not** scale-relative |
| `minimum_radius` / `maximum_radius` (cylinder region only) | **0 / +infinity** ("no limit") | validity gate: rejects a fitted cylinder outright if its radius falls outside this band |

The manual itself flags this directly: "The classes... depend on a few parameters that
should be defined by the user. They also have default values, **but these values do not
necessarily guarantee to produce pleasant results**" (§`Shape_detection_RegionGrowingPoints_parameters`).
**Load-bearing contrast with Efficient RANSAC**: RANSAC's defaults self-scale to the
bounding box; Region Growing's `maximum_distance=1` and `sphere_radius=1` are hardcoded
absolute numbers with no relation to model scale. On a real point cloud in meters, a
default 1.0 m distance/radius threshold is close to meaningless — every deployment must
set these explicitly relative to the object's own scale and the sensor's noise floor.
This is a concrete integration trap specific to this half of the library, not discussed
anywhere in Schnabel's paper (which only concerns the RANSAC side).

The cylinder region type's `minimum_radius`/`maximum_radius` band is worth flagging
separately: it is a free, deterministic, geometric plausibility gate ("the radius below/above
which an estimated cylinder is considered as invalid and discarded") usable even before
any residual-based confidence score is computed — directly relevant to a two-tier
fallback design (RQ16) since it needs no learned component and no retraining.

## 3. What the library outputs — and does NOT output (critical for several branch RQs)

This is the load-bearing negative finding of the whole document.

- **Primitives are returned unbounded.** Plane is recovered as `plane_normal()` + convertible to `Kernel::Plane_3` (an infinite plane equation, point+normal). Cylinder is recovered as `axis()` returning `Kernel::Line_3` (**infinite line**) + `radius()`. The Region-Growing cylinder example section is explicitly titled and captioned "detecting **(infinite)** cylinders" (§`Shape_detection_RegionGrowingPoints_example_3D_cylinders`, `region_growing_cylinders_on_point_set_3.cpp`). **No boundary polygon, no extent, no centroid, no length is computed or returned by either algorithm.**
  Verified by dumping every public accessor on every primitive class in the RANSAC
  reference docs, not just the two examples above: `Sphere` → `center()`, `radius()`;
  `Cone` → `apex()`, `axis()`, `angle()` (an infinite double cone — no height); `Torus` →
  `axis()`, `center()`, `major_radius()`, `minor_radius()` (a complete torus — no arc
  extent). On the Region Growing side, `Least_squares_plane_fit_region::Primitive` is
  typedef'd directly to `GeomTraits::Plane_3` — same infinite-plane story. The **only**
  point where boundedness exists at all, on either algorithm, for any shape type, is
  `indices_of_assigned_points()` (RANSAC) / the region's `std::vector<Item>` (Region
  Growing) — i.e. implicitly, in which points were assigned, never as a returned extent.
- The only way to get a bounded face (centroid, extent, footprint) is to post-process: take `indices_of_assigned_points()` (RANSAC) or the region's item list (region growing), then compute a bounding box / centroid yourself in the shape's local parameterization. The library gives you the point-to-shape assignment; it does not give you "face 3's centre and size."
- **No uncertainty output.** No API surface, parameter, or discussion anywhere in this manual produces a covariance, confidence interval, or per-primitive error bar on a fitted normal/axis. The only per-shape "quality" numbers shown are computed by user code in the examples: `average_distance` (mean residual, §`Shape_detection_RANSACExample_with_point_access`, computed as `sum(sqrt(shape->squared_distance(p))) / indices_of_assigned_points().size()`) and global `coverage` = `(points.size() - number_of_unassigned_points()) / points.size()` — that exact variable name and formula appear in CGAL's own example code, used there to pick the best of several repeated detection runs. There is no MDL/energy score, no posterior outlier probability, no optimality certificate anywhere in this component.
- **No preprocessing.** No denoising, outlier removal, or downsampling step is part of this module (would live in `Point_set_processing_3`, which is not referenced or linked from this page at all — checked directly, zero mentions of PCA/jet-fitting/normal estimation).
- **Determinism differs by algorithm** — directly stated in §`Shape_detection_Comparison`: "[region growing] is deterministic (for a given input and a given set of parameters, it always returns the same output), whereas the Efficient RANSAC algorithm is randomized and so the output may vary at each run." Even holding the *same* point cloud and *same* parameters fixed, RANSAC's output (which shapes get found, in what order, how large) is not guaranteed reproducible run-to-run. This is corroborated independently at the API level: the `Efficient_RANSAC` class (constructor, `set_input()`, `detect()`, the whole `Parameters` struct) exposes **no random-seed parameter anywhere**, so there is no documented way to force a reproducible run even if we wanted one. `shapes()`/`planes()` are stated to return results "in the order of detection... ordered with decreasing size" — order is a *consequence* of the randomized search, not a stable identifier.
- No sensor/range-error model, no incidence-angle dependence, no discussion of dropout/holes as a first-class defect — the connectivity constraint (`cluster_epsilon`) is the only mechanism in this doc that would interact with a dropout hole, and it interacts adversarially (see §5).
- **Custom-shape extension confirmed concrete but unshipped.** The `Shape_base` interface requires implementing `minimum_sample_size()`, `create_shape()`, `squared_distance()` (single-point and batch overloads), and `cos_to_normal()`, with optional `connected_component()` and `info()` overrides (full worked `My_Plane` example given in the manual). This is a real, documented path to adding a cuboid or superquadric primitive to the RANSAC search loop — but it is engineering we would have to do ourselves; nothing beyond plane/sphere/cylinder/cone/torus ships.

## 4. Quantitative results (the only numbers this doc actually contains)

All numbers are from a synthetic/aerial-lidar-scale demo scene, **not from an object-scale tabletop scan**, and are qualitative-guidance-oriented, not a controlled noise-vs-accuracy study.

**Fig. 94.4 — epsilon sweep, 5M-point complex scene, plane detection** and **Fig. 94.5 —
probability sweep, same scene** (§`Shape_detection_RANSACPerformance`): **correction —
an earlier draft of these notes quoted specific numeric tables for these two figures
("epsilon=2.0 → 555 primitives, 98% coverage, 50s," etc.). Those numbers do not exist
anywhere in the page's text or HTML** — Fig. 94.4/94.5 are chart *images*
(`epsilon_graph.png`, `prob_graph.png`) with no accompanying data table in the manual;
fetching/rendering the page as text (which is how this document was actually read,
including on the re-verification pass for this revision) never exposes numeric axis
values, only the prose captions. Those tables were fabricated by an earlier
tool-summarized pass and are removed here — **do not cite them**. What the page's actual
prose captions say, verbatim:
- Fig. 94.4: "The graph depicts the number of detected shapes (purple) and the coverage
  (green), that is the ratio assignedPoints / totalPoints, against the `epsilon`
  tolerance parameter. A higher value for `epsilon`, that is a more tolerant error, leads
  to fewer but larger shapes and shorter running times."
- Fig. 94.5: "The graph depicts the runtime, coverage, and the number of detected
  primitives against the search endurance parameter, that is `probability` to miss the
  largest shape at each iteration. The number of shapes is stable and the coverage
  increases when the `probability` is lowered. The running time increases significantly
  as many more candidates are generated during each iteration of the algorithm."
- Directly stated in prose (§`Shape_detection_RANSACPerformance`, ahead of both figures):
  "The `probability` parameter... barely impacts the number of detected shapes, has a
  moderate impact on the size of the detected shapes, and increases the running time" —
  this qualitative claim (not a number) is the real, citable content: **`probability` is
  not the parameter to fight noise/incompleteness with; `epsilon`/`cluster_epsilon`/`normal_threshold`
  are**, since only `epsilon` is described as controlling the fragmentation/coverage
  trade-off directly.
- Test scene for both figures: "a complex scene with 5M points" — i.e. an
  architectural/aerial-lidar-scale demo, not object-scale, same caveat as everywhere else
  in this section.

**Region growing runtime, 3D point set, K-NN neighborhood** (§`Shape_detection_RegionGrowingPoints_performance`):

| points | time (s) |
|---|---|
| 300k | 0.76 |
| 600k | 1.69 |
| 900k | 2.80 |
| 1200k | 4.06 |

Roughly linear, ~3.4 µs/point once parameters are "well-defined." A survey PLY at object scale (likely << 1M points after any downsampling) is trivially fast on this axis — runtime is not the constraint for our use case, correctness is.

**2D 70k-point neighborhood-radius sweep** (§`Shape_detection_RegionGrowingPoints_performance`): `sphere_radius=1.0` → 794 regions / 4483 assigned points (i.e. under-connected, most points isolated and discarded — 6.4% coverage); `radius=3.0` → 3063 regions / 63038 points (90%); `radius=6.0` → 2508 regions / 64906 (93%); `radius=9.0` → 2302 regions / 65334 (93%). Shows a cliff: too-small a neighborhood doesn't just over-segment, it **collapses coverage** because regions never reach `minimum_region_size` and get discarded outright — a mechanism directly relevant to sparse/partial coverage on a defective survey cloud.

**Fig. 94.11 — accuracy/completeness head-to-head, same real scene** (§`Shape_detection_Comparison`): Efficient RANSAC: 78% of shapes correctly detected in 8s. Region growing: 100% of shapes detected in 15s. This is the one place the doc gives a concrete "% correct" number, and it favors region growing on completeness at ~2× the cost — relevant to any choice between the two algorithms for a correctness-over-speed inspection setting.

## 5. Load-bearing assumptions (what has to hold for these numbers/behaviors to apply)

1. Normals are already computed and are of usable quality — both algorithms consume normals as input, don't estimate or refine them; garbage-in/garbage-out is total (unaddressed by this doc; see `Point_set_processing_3`, out of scope of this page — not linked from it).
2. Efficient RANSAC's normal-deviation test explicitly assumes **unoriented** normals (custom-shape example computes `abs(dot(...))`, §`Shape_detection_RANSACExample_with_custom_shapes`) — sign-robust. Region growing's `maximum_angle` semantics are stated without the same explicit unoriented/abs treatment — worth checking source before assuming it tolerates flipped normals equally well; not resolved by this manual page.
3. The demo numbers (Fig. 94.11, the runtime tables) come from architectural/aerial scenes with **millions of points and genuinely planar man-made structure** — a different regime from a single tabletop object's fused PLY (likely 10⁴–10⁵ points, mixed plane/cylinder/rounded geometry). Extrapolating the specific numbers to our scale is not justified by anything in the doc; only the qualitative trade-off shapes (tighter tolerance → more fragments + lower coverage + slower; too-large a neighborhood → coarser regions; too-small → coverage collapse) should be assumed to transfer.
4. **Region Growing's point-set defaults are absolute, not scale-relative** (`maximum_distance=1`, `sphere_radius=1`, §2 above) — load-bearing because "just use the defaults" is unsafe advice for this half of the package, unlike Efficient RANSAC where the bbox-relative defaults are at least a sane scale-independent starting point.
5. `cluster_epsilon`'s connectivity constraint assumes the true surface's sampled support is spatially contiguous at the chosen grid resolution. **Under our dropout defect mode** (glossy/dark-surface depth dropout), a real cylinder's surface is not contiguous — it has a real hole. The documented behavior ("we enforce a connectivity constraint by considering only one connected component... selected as the one covering the largest number of inliers," §`Shape_detection_RANSAC`) means a dropout gap large enough relative to `cluster_epsilon` will cause the algorithm to **keep only the larger of the two resulting fragments** and silently drop the rest as unassigned/a separate smaller shape — not flagged as a defect, just lower coverage or an extra spurious small shape. This is inference from the documented mechanism, not a number the doc gives us; it is the single most concrete mechanistic link this doc offers to our "does dropout bias fitted primitives" branch question, and it points at **coverage loss / fragment loss**, not necessarily silent axis bias — the doc gives no evidence either way on whether the *retained* fragment's fitted axis/normal is systematically biased by asymmetric dropout, only that dropout will fragment/shrink the assigned support.

## 6. Mapping to branch questions

**Directly answered by this doc:**
- *"Does the object set close with plane+cylinder+sphere+cone RANSAC, or do real parts force superquadrics/cuboid decomposition?"* → CGAL's shipped vocabulary is plane/sphere/cylinder/cone/**torus** (RANSAC) — a superset of the contract's list, but region growing (the more complete/deterministic algorithm per §4) only covers plane/sphere/cylinder for point sets, no cone, no torus. Neither algorithm has native superquadric or cuboid-decomposition support; chamfered/rounded parts would either (a) get approximated into a plane/cylinder/cone/torus with loosened `epsilon` (silently, with no indication the fit is an approximation of a rounded feature), (b) get left as unassigned points, or (c) require a hand-written `Shape_base` subclass — this doc gives us no evidence on how well option (a) actually works geometrically.
- *"Which methods recover bounded faces (extent, boundary, centroid)?"* → **Neither CGAL algorithm does, natively.** Both return infinite-primitive parameters (plane equation; cylinder axis as an infinite `Line_3` + radius). Confirmed by direct API inspection (§`Shape_detection_RANSACExample_with_parameters`) and an explicit doc label ("(infinite) cylinders", §`Shape_detection_RegionGrowingPoints_example_3D_cylinders`). Getting "face 3's centre and size" requires writing the extent/centroid computation ourselves on top of the returned point-to-shape assignment — this is a concrete implementation task this doc surfaces, not something to expect off the shelf.
- *"Is there a closed-form covariance for a fitted plane's/cylinder's orientation?"* → No. Nothing in this component produces one. This branch question must be answered from elsewhere (M3C2-EP literature, or derived independently) — CGAL Shape Detection is silent on uncertainty entirely.
- *"What computable signals do papers use to declare a fit unreliable... obtainable from an off-the-shelf implementation with no retraining?"* → From this library specifically: `coverage` (assigned/total points, computed by the *caller* using `number_of_unassigned_points()`, not returned by the API as a named quantity) and mean/per-point residual distance (also caller-computed via `squared_distance()` per assigned point, §`Shape_detection_RANSACExample_with_point_access`), plus — for cylinders specifically — a free, deterministic `minimum_radius`/`maximum_radius` plausibility gate on the Region Growing side (§2). No MDL, no posterior, no certificate. If we want a fit-reliability score, CGAL gives us the raw ingredients (residuals, assigned-point count, one geometric plausibility band) but zero packaged score.
- *"How repeatable is decomposition across re-scans... does 'face 3' stay face 3?"* → Negative finding, explicit: RANSAC is **randomized, not reproducible run-to-run even on the identical input** (§`Shape_detection_Comparison`), and confirmed at the API level — no seed parameter exists anywhere on `Efficient_RANSAC` or its `Parameters` struct, so this cannot be worked around through the documented public interface. Region growing is deterministic **given identical input ordering and parameters**, but nothing in the API assigns persistent, semantic IDs to regions — "face 3" is just the third region discovered in whatever order growing/seeding proceeds; a re-scan changes point count/order/coverage, which will change which region is found first. **Identity/ordering stability across re-scans is not provided by either algorithm and would have to be built as a separate matching step** (e.g. by geometric proximity of returned plane/cylinder parameters between runs).
- *"Which preprocessing steps measurably change fitting outcomes?"* → Out of scope of this component entirely; not addressed.
- *"Does the introspection literature support a two-tier design (primitive decomposition, else uniform sphere)?"* → This doc doesn't discuss failure/introspection design at all; it only exposes coverage, residuals, and (for cylinders) a radius plausibility band as raw signals — the last of these is a working example of a deterministic geometric guardrail usable in a two-tier design without any learned component, but the manual itself makes no claim about sufficiency.

**Not addressed at all by this doc (must come from elsewhere):**
- Angular error on recovered normal/axis vs. registration error / noise sigma — no sensor-noise or registration-error modeling of any kind.
- Range-/incidence-angle-dependent error propagation, M3C2-EP transfer — not mentioned.
- Symmetry-based completion (mirror-plane, PotSAC) for partial cylinders — not mentioned; CGAL's answer to partial coverage is purely "did the point count clear `min_points`/`minimum_region_size` and pass the connectivity check," nothing about exploiting symmetry to complete a partial fit.
- Learned abstraction methods (SuperDec, SuperFlex), datasets/benchmarks (Fit4CAD, SHREC 2022, ABC), ground-truth protocols — entirely outside this doc's scope (it's a C++ library manual, not a survey).
- Depth dropout mechanisms/magnitudes on glossy/dark surfaces for active stereo — not addressed; only the *consequence* (fragmented support under a connectivity constraint) is inferable, not the sensor-side mechanism.

## 7. What transfers to our setting vs. what does not

**Transfers:**
- Parameter semantics are exact and load-bearing if we actually integrate this library (Open3D doesn't have an efficient-RANSAC-with-connectivity-clustering equivalent as clean as CGAL's; PCL's SampleConsensus is comparable but lacks the `cluster_epsilon` 2D-parameterization connectivity check). If we prototype with CGAL, `epsilon`, `cluster_epsilon`, `normal_threshold`, `min_points`/`minimum_region_size`, `maximum_angle` are the exact knobs and their qualitative failure directions (too tight → fragmentation + coverage loss + slow; too loose → merges distinct features / silently approximates non-primitive geometry) are directly actionable.
- The determinism gap (RANSAC non-reproducible, region growing reproducible-but-unordered) is a real constraint on RQ3/RQ5 (action-space consistency) regardless of which library we end up using — any RANSAC-family method has this property by construction (randomized sampling), so this is a family-level finding, not CGAL-specific.
- The "infinite primitive, no bounded face" fact is very likely to hold for PCL's and Open3D's RANSAC-family implementations too (same lineage from Schnabel et al. 2007) — worth verifying directly rather than assuming, but this doc gives strong prior that "recover a face centroid/extent" is *never* free and always a downstream step we have to write.
- The dropout-vs-connectivity-constraint interaction (§5) is a mechanism, not a CGAL quirk — any connectivity-enforcing clustering step (which any efficient-RANSAC descendant needs to avoid spurious shapes) will have the same failure mode against our glossy-surface dropout defect.

**Does NOT transfer / not covered:**
- None of the quantitative numbers (Fig. 94.11, the two runtime tables) are from object-scale, tabletop, real-defect data — they're aerial/architectural synthetic-scale scenes. Do not use these absolute numbers as a noise/incompleteness threshold for our objects; only the qualitative trade-off shapes transfer. (Fig. 94.4/94.5 carry no usable numbers at all, fabricated or otherwise — see the correction in §4.)
- No uncertainty, no failure score, no learned/superquadric method, no symmetry completion — these branch questions need separate sources (M3C2-EP papers, EMS/MonteBoxFinder/GNC papers, SuperDec/SuperFlex papers, Fit4CAD/SHREC benchmark papers), all explicitly out of scope of this manual page.
- Region Growing's point-set parameter defaults (`maximum_distance=1`, `sphere_radius=1`) are absolute numbers tied to CGAL's own demo files' units, not to ours — cannot be used as-is on a metric survey cloud.

## 8. Bottom line for integration planning

If Branch 1/2 prototype with CGAL Shape Detection: expect to (a) supply your own normals (from Open3D or `Point_set_processing_3`, not from this component), (b) get infinite plane/cylinder/cone/sphere/torus parameters back and write the bounded-extent/centroid extraction yourself — confirmed by dumping every accessor on every primitive class, not just the two documented examples, (c) treat RANSAC output as non-reproducible run-to-run (no seed parameter exists anywhere in its public API) and either switch to region growing for anything needing stable identity, or add your own matching/ID-assignment layer across runs, (d) treat `coverage` + mean residual + (for cylinders) the `minimum_radius`/`maximum_radius` plausibility band as the only off-the-shelf reliability signals, insufficient on their own to answer "is this fit trustworthy" without a validation experiment on our own objects, (e) expect `cluster_epsilon`'s connectivity constraint to interact badly (fragment or truncate) with dropout holes on glossy surfaces — worth testing directly on a defective cylinder scan before trusting axis recovery on partially-dropout-affected objects, and (f) **do not use Region Growing's point-set defaults as-is** — `maximum_distance=1`/`sphere_radius=1` are absolute numbers unrelated to our object scale and must be set explicitly (Efficient RANSAC's bbox-relative defaults are the safer starting point of the two).

---

## Provenance / fetch notes

Initial `WebFetch` calls against `index.html` and the `Efficient_RANSAC` class page
returned a lossy, summarized rendering — no parameter default values, incomplete
member-function lists, and (established on this revision pass) at least two fabricated
numeric tables purporting to be read off chart images (Fig. 94.4/94.5) that the page's
actual text never contains. Re-fetched as raw HTML via `curl -A "Mozilla/5.0"` and
converted with `html2text` (miniconda's `html2text` CLI) to recover exact text, and
every quoted number in this revision was checked against that raw text directly. Pages
fetched and read in full (local copies in `/tmp`, not committed to the repo):

- `Shape_detection/index.html` (main user manual, 2037-line raw HTML / ~3500-line
  converted text — Introduction, Efficient RANSAC incl. all 5 examples' full source code,
  Region Growing Framework, Point Set section + parameters + performance, Polygon Mesh
  section + parameters + performance, Comparison section, History, Acknowledgments)
- `structCGAL_1_1Shape__detection_1_1Efficient__RANSAC_1_1Parameters.html` (exact
  defaults for all 5 RANSAC parameters — absent from the main manual page's summarized
  rendering)
- `classCGAL_1_1Shape__detection_1_1Efficient__RANSAC.html` (full member-function list;
  confirms no seed parameter anywhere; `shapes()`/`planes()` ordering semantics)
- `classCGAL_1_1Shape__detection_1_1{Plane,Cylinder,Sphere,Cone,Torus}.html` (every public
  accessor method on every primitive type, confirming no bounded-region output anywhere)
- `classCGAL_1_1Shape__detection_1_1Shape__base.html` (custom-shape extension interface)
- `classCGAL_1_1Shape__detection_1_1Point__set_1_1Least__squares__{plane,cylinder}__fit__region.html`
  (Region Growing named-parameter exact defaults, incl. cylinder radius validity bounds)
- `classCGAL_1_1Shape__detection_1_1Point__set_1_1{K__neighbor__query,Sphere__neighbor__query}.html`
  (default `k_neighbors=12`, `sphere_radius=1`)
- `group__PkgShapeDetectionRGOnPointSet3.html` (Point_set_3 convenience-function typedefs)
- `citelist.html` (Schnabel et al. 2007 and Lafarge & Mallet 2012 full citations)

Cross-checked against the sibling note `schnabel-efficient-ransac.md` (same branch,
already read in full) to avoid duplicating algorithm-level findings and to isolate what
specifically changes/adds at the *implementation* level (exact defaults, API-level
bounded-output confirmation, no-seed finding, cylinder radius gate).
