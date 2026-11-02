# Open3D PointCloud API and geometry tutorials

**Source:** https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
(+ https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html,
https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html,
https://www.open3d.org/docs/release/python_api/open3d.geometry.OrientedBoundingBox.html,
GitHub `isl-org/Open3D` issue #6602 and source `cpp/pybind/geometry/pointcloud.cpp`,
`cpp/open3d/geometry/PointCloudPlanarPatchDetection.cpp`)
**Type:** library documentation + API reference + issue tracker (not a paper — no experiments, no numbers on accuracy/noise)

## What this document actually is

Not a method paper. It's the integration surface we'd build on top of. The value here
is purely: what does Open3D's `PointCloud` class give us out of the box, in what shape
(bounded vs. infinite primitive, what return type), and where does the vocabulary stop.
No angular-error numbers, no noise-robustness numbers, no ground-truth protocol — none
of that is in scope for API docs. This document answers integration-surface questions
only (RQs about "which methods recover bounded faces," "does Open3D alone close the
object set," "what does Open3D + CGAL/PCL cover").

## Core API surface (exact signatures, from the pybind source and docs)

**Plane RANSAC — infinite plane, unbounded:**
```
segment_plane(distance_threshold: float, ransac_n: int, num_iterations: int,
              probability: float = 0.99999999)
  -> tuple[ndarray[4,1], list[int]]   # (a,b,c,d) s.t. ax+by+cz+d=0, inlier indices
```
Tutorial example values: `distance_threshold=0.01, ransac_n=3, num_iterations=1000`.
Docstring for `probability`: "Expected probability of finding the optimal plane" — this
is the standard RANSAC confidence parameter (adaptive iteration count), not a returned
confidence score on the fit itself. **No per-fit residual, RMS, or inlier-ratio is
returned automatically** — caller must compute residuals from the returned indices and
plane equation itself if a confidence signal is wanted. Relevant to the "computable
failure signals" RQ: Open3D gives you the raw material (points, plane eq) but not a
packaged score; inlier_ratio = len(inliers)/len(pcd.points) is trivial to compute
post-hoc, RMS residual likewise, but neither is provided as a method.

**Planar patch detection — BOUNDED, this is the answer to the "bounded faces" RQ:**
```
detect_planar_patches(normal_variance_threshold_deg: float = 60,
                       coplanarity_deg: float = 75,
                       outlier_ratio: float = 0.75,
                       min_plane_edge_length: float = 0.0,
                       min_num_points: int = 0,
                       search_param: KDTreeSearchParam = KDTreeSearchParamKNN(knn=30))
  -> list[OrientedBoundingBox]
```
Docstring (verbatim, from `pointcloud.cpp` pybind bindings): *"Detects planar patches
in the point cloud using a robust statistics-based approach. Returns: A list of
detected planar patches, represented as OrientedBoundingBox objects, with the third
column (z) of R indicating the planar patch normal vector. The extent in the z
direction is non-zero so that the OrientedBoundingBox contains the points that
contribute to the plane detection."*

This is the single most directly load-bearing fact for our contract. Each returned
`OrientedBoundingBox` carries `.center` (float64[3]) — the face centroid — `.extent`
(float64[3]) — face size, with the z-component the (small, non-zero) point-scatter
thickness rather than true face depth — and `.R` (float64[3x3]) whose third column
is the face normal. That is exactly "face centre + face size + orientation," not a
bare plane equation. **This directly answers the RQ "which methods recover bounded
faces (extent, boundary, centroid) rather than infinite primitives?"** — for planes,
Open3D's own `detect_planar_patches` does, with no extra code. `min_plane_edge_length`
and `min_num_points` are the false-positive rejection knobs (edge length defaults to
1% of the cloud's largest dimension per the tutorial page, count defaults to 0.1% of
cloud size). `normal_variance_threshold_deg` / `coplanarity_deg` are the robust-
statistics planarity thresholds (larger `normal_variance_threshold_deg` and smaller
`coplanarity_deg` each admit noisier planes, per the pybind doc-injection strings).
Algorithm description is "robust statistics-based" with a breadth-first octree
(`BoundaryVolumeHierarchy`) for hierarchical subregion search — the source doesn't
cite a paper inline in the header comments I pulled; this matches (unconfirmed,
would need a literature cross-check) the general family of RANSAC-free plane
detectors that dominate object surfaces via per-region PCA + region growing on
patches, rather than global RANSAC.

**Cylinder / sphere / cone: absent.** Confirmed by three independent checks:
1. The full `PointCloud` method listing (pulled from the live API page) has no
   `segment_cylinder`, `segment_sphere`, `segment_cone`, or generic multi-primitive
   RANSAC entry point. Plane is the only implicit-surface RANSAC primitive.
2. GitHub issue **#6602** ("Request to fit RANSAC Cylinder," opened 2024-01-13, still
   **open** as of this reading, 4 comments, zero maintainer response) — a user's
   `pcd.segment_cylinder(radius=0.03)` throws `AttributeError` because the method
   doesn't exist. Comment thread (informal, Chinese-language, non-maintainer users):
   one commenter explicitly notes *"也没有球面的ransac，同求"* ("there's also no
   sphere RANSAC, same request") — so this isn't just a cylinder gap, sphere RANSAC
   is equally absent from Open3D's `PointCloud` API. Another commenter's suggested
   workaround is informal: "project first, then compute" (project points to 2D via
   the cylinder-axis hypothesis, fit a circle) — i.e., roll your own, no library
   support offered.
3. A repo-wide issue search for "cylinder" in title turns up only 4 issues total,
   none of which are a fitting feature ever shipped: #6602 (open, unresolved), #3089
   (closed — "create a cylinder from two points," a **mesh primitive constructor**,
   not point-cloud fitting), #3275 (closed — UV maps for cone/cylinder/sphere
   **meshes**, rendering not fitting), #769 (closed — selecting points inside a
   cylindrical *region*, not fitting a cylinder *to* points). No roadmap signal, no
   open PR, no announced timeline.

**Consequence for the branch's "does the contract close with Open3D alone" question:**
No. Open3D's `PointCloud` closes the **plane** half of the box+cylinder contract
(with bounded-patch support, which is a genuine asset — better than plain RANSAC plane
fitting elsewhere in that it returns extents for free). It does **not** close the
cylinder half at all — cylinder-orbit, which the framing doc explicitly calls out as
one of the two canonical actions ("a cylinder orbited with the camera normal to the
surface"), has zero first-party support. Any implementation needs a second library or
hand-rolled RANSAC/least-squares cylinder fit (PCL has `SACMODEL_CYLINDER` via
`pcl::SACSegmentation`; CGAL has `Efficient_RANSAC` with a `Cylinder` shape primitive
registered by default alongside Plane/Sphere/Cone/Torus — this is the natural
complement to fill Open3D's gap, though this document did not itself fetch CGAL's
docs to verify current API details).

## Normal estimation

```
estimate_normals(search_param: KDTreeSearchParam = KDTreeSearchParamKNN(knn=30),
                  fast_normal_computation: bool = True)
```
Tutorial example uses `KDTreeSearchParamHybrid(radius=0.1, max_nn=30)` instead of the
KNN default. Method is PCA-based (implicit — "covariance analysis" per the tutorial
text: *"The covariance analysis algorithm produces two opposite directions as normal
candidates"*), i.e., this is the classical Hoppe-style local-PCA normal estimator,
sign-ambiguous by construction. Orientation must be fixed afterward with one of:
- `orient_normals_to_align_with_direction(orientation_reference=[0,0,1])`
- `orient_normals_towards_camera_location(camera_location=[0,0,0])`
- `orient_normals_consistent_tangent_plane(k, lambda_penalty=0.0, cos_alpha_tol=1.0)`
  (graph-propagation / MST-style consistent orientation across the whole cloud)

**No angular-error-vs-noise-vs-k relationship is documented anywhere on these pages.**
This is a hard negative for the RQ asking for a quantified noise-sigma / neighbourhood-
size / angular-error relationship per estimator family — Open3D's docs state the
method exists and how to call it, not its error behavior under any noise model. That
number has to come from the PCA-normal-estimation literature (e.g., Mitra et al. on
optimal neighborhood size, or later jet/VCM comparisons), not from here.

## Outlier removal — exact tutorial parameters

```
remove_statistical_outlier(nb_neighbors: int, std_ratio: float, print_progress=False)
  -> tuple[PointCloud, list[int]]
remove_radius_outlier(nb_points: int, radius: float, print_progress=False)
  -> tuple[PointCloud, list[int]]
```
Tutorial example values: `nb_neighbors=20, std_ratio=2.0` and `nb_points=16,
radius=0.05`. Doc note on `std_ratio`: "the lower this number the more aggressive the
filter will be." **No stated caveat about edge/detail loss or about harming downstream
primitive fitting** — the tutorial page only shows a before/after visualization
(removed points in red). This is a negative finding for the branch's preprocessing
question: Open3D's own docs make no claim about denoising harming face-boundary
recovery; that risk (if real) is not something Open3D's documentation warns about or
quantifies — it would have to be established empirically on our own defect profile or
sourced from a paper that specifically studies over-smoothing vs. edge/segmentation
fidelity.

## Downsampling family

`voxel_down_sample(voxel_size)`, `uniform_down_sample(every_k_points)`,
`random_down_sample(sampling_ratio)`, `farthest_point_down_sample(num_samples,
start_index=0)`. Only mentioned for completeness — no numbers on how voxel size trades
against fitting outcome are given in these docs.

## OrientedBoundingBox — the container type for bounded planar patches

`.center` (3,), `.extent` (3,), `.R` (3x3), `.color` (3,); `.volume()`,
`.get_box_points()`, `.get_axis_aligned_bounding_box()`. Static factories
`create_from_points(points, robust=False)` and `create_from_points_minimal(points,
robust=False)` (the `robust` flag switches to a more outlier-tolerant convex-hull-based
fit — relevant if we ever need to box a partial/defective sub-cluster ourselves rather
than relying on `detect_planar_patches`'s internal boxing).

## Load-bearing assumptions in this source

- Everything here assumes a clean-ish, already-loaded `PointCloud` — no claims about
  behavior under our specific defect modes (registration jitter, glossy dropout, depth
  bias). API docs are defect-agnostic by construction; they describe the function
  contract, not its degradation under adversarial input.
- `detect_planar_patches`'s bounded-patch guarantee is architecture-level (it's what
  the algorithm is designed to output), not an empirical claim we've verified on our
  own PLYs — still needs a runtime check on our fused clouds (this doc doesn't include
  a runtime/performance number for our cloud sizes; that's an open item, not covered
  by API docs).

## What transfers to our setting

- `detect_planar_patches` → `OrientedBoundingBox` is a plausible direct source for
  box-face actions: center = face aim point, extent(x,y) = face size for framing, R's
  z-column = orthogonal approach direction. This could be the concrete implementation
  for the "box face" half of the primitive-level deterministic rules the framing doc
  allows.
- `segment_plane` is a viable fallback/building block if `detect_planar_patches`
  under- or over-segments on our defect profile, since it gives raw inliers to inspect.
- The cylinder gap is a hard integration fact: cylinder-orbit action generation cannot
  be built on Open3D alone; must pull in PCL/CGAL or hand-write a RANSAC cylinder fit
  (project-to-2D-circle approach is the informal workaround surfaced even in Open3D's
  own issue tracker).

## What does NOT transfer / negative findings

- No angular-error, noise-robustness, or fit-confidence quantification anywhere in
  this source — it cannot answer any of the branch's "how many degrees of error"
  questions. Those need dedicated normal-estimation and RANSAC-robustness papers.
- No sphere or cone RANSAC either (confirmed via the same issue-thread comment) — so
  "plane + cylinder + sphere + cone" as a vocabulary is NOT closed by Open3D even
  partially beyond plane; three of four primitive types are entirely outside this
  library's scope.
- No introspection/failure-detection API (no MDL score, no outlier-probability output,
  no fit-confidence field) — `probability` in `segment_plane` is a RANSAC search
  hyperparameter, not a returned confidence value. Any "declare this fit unreliable"
  logic has to be built by the caller from raw outputs (inlier count, indices,
  residuals computed manually).
- No denoising-harms-edges caveat is documented — silence here is not evidence of
  absence of the effect, just evidence Open3D's docs don't discuss it.

## Checkable citations

- `segment_plane` signature/defaults, `detect_planar_patches` signature/defaults,
  `estimate_normals`/`orient_normals_*`/outlier-removal/downsampling signatures: Open3D
  Python API reference page `open3d.geometry.PointCloud.html` (live doc, no fixed
  section numbers).
- `detect_planar_patches` bounded-patch docstring quoted verbatim from
  `cpp/pybind/geometry/pointcloud.cpp` in `isl-org/Open3D` (GitHub `main` branch,
  fetched via raw.githubusercontent.com).
- Tutorial parameter values (`distance_threshold=0.01` etc., `nb_neighbors=20,
  std_ratio=2.0`, `nb_points=16, radius=0.05`, `radius=0.1, max_nn=30` for normals):
  `docs/release/tutorial/geometry/pointcloud.html` and
  `.../pointcloud_outlier_removal.html`.
- Cylinder/sphere absence: GitHub issue `isl-org/Open3D#6602` (state=open, created
  2024-01-13, 4 comments, fetched via GitHub REST API), cross-checked against
  issue-title search `cylinder in:title` (4 total results, none a shipped fitting
  feature) via `api.github.com/search/issues`.
- `OrientedBoundingBox` attributes/methods: `open3d.geometry.OrientedBoundingBox.html`
  API reference page.
