# SuperDec: 3D Scene Decomposition with Superquadric Primitives (ICCV 2025, Oral)

**arXiv:** 2504.00992 (v2, 2026-03-19) | **PDF:** https://arxiv.org/pdf/2504.00992 | **HTML:** https://arxiv.org/html/2504.00992v2
**Project page:** https://super-dec.github.io | **Code + weights:** https://github.com/elisabettafedele/superdec
**Authors:** Fedele, Sun, Guibas, Pollefeys, Engelmann (ETH Zurich / Stanford / Microsoft)

Read via arXiv HTML render (automated fetch tool converts LaTeX→text; equation
symbols/superscripts below may have minor transcription loss from the original
typesetting — flagged where confidence is lower). Section/table/equation numbers
are as reported by the fetch tool against the HTML source, not independently
re-verified against the PDF.

## Core method

Per-object superquadric decomposition network + optional post-hoc refinement:

1. **Encoder:** PVCNN point encoder on the input point cloud → per-point features
   `F_PC ∈ R^{N×H}`.
2. **Decoder:** a fixed bank of `P=16` learned "superquadric query" slots
   `F_SQ ∈ R^{P×H}` (sinusoidal positional encoding, DETR/set-prediction style)
   pass through a 3-layer Transformer decoder with self- and cross-attention
   against the point features.
3. **Heads:** (a) a soft point→primitive assignment matrix `M` (segmentation),
   (b) per-slot superquadric parameters: 11 total — 5 shape params
   `(s_x, s_y, s_z, ε₁, ε₂)` + 6 pose params (3 translation + 3 rotation) — plus
   1 scalar existence probability `α_j`.
4. **Post-processing:** Levenberg-Marquardt optimization refines the predicted
   parameters against radial-distance residuals to the point cloud; paper
   describes this as giving only "modest improvements beyond the neural network
   predictions," and larger gains specifically on **out-of-category** shapes
   (Fig. 11), which the authors read as evidence the network's raw predictions
   sit in local minima that LM can polish but not escape.

Superquadric implicit surface (Eq. 1, as rendered):
`f(x) = ((x/s_x)^(2/ε₂) + (y/s_y)^(2/ε₂))^(ε₂/ε₁) + (z/s_z)^(2/ε₁) = 1`
Radial distance (Eq. 2): `d_r = |x| · |1 − f(x)^(−ε₁/2)|` — chosen specifically
because it has a **closed form**, which the paper states is the reason
superquadrics were picked over plain ellipsoids (matters for downstream use,
e.g. grasp planning, collision checking).

**Loss (Eq. 4):** `L = L_rec + λ_par·L_par + λ_exist·L_exist`
- `L_rec` (Eq. 5) = bidirectional Chamfer (point→SQ, Eq. 6ish; SQ→point, Eq. 7)
  **plus a normal term `L_N`**. The paper states `L_N` "is defined as the
  reconstruction loss from Yang et al. [CSA, 2021], and is used to incorporate
  normal information during training which leads to accelerated convergence."
  — No further formula, no ablation isolating its effect on accuracy, and no
  clarification of whether the normals it uses are analytic (from the SQ
  surface) vs. ground-truth mesh vs. estimated from the point cloud. This is a
  training-time regularizer for convergence, not a claimed normal-accuracy
  contribution — the paper reports no normal-error metric anywhere.
- `L_par` (Eq. 8), parsimony: `(1/P · Σ_j √m_j)²` (as rendered) — penalizes the
  total "mass" assigned across primitives, pushing the model toward fewer,
  larger primitives.
- `L_exist` (Eq. 9): BCE between predicted `α_j` and a derived target
  `α̂_j := [m_j > ε_exist]`, threshold `ε_exist = 24` (a point-count-like
  threshold on the soft assignment column sum). Ablation (Table 4) reports the
  existence head gives only marginal improvement over just thresholding `m_j`
  directly — i.e., the "is this primitive real" signal is basically derived
  from segmentation mass, not a separately learned confidence.
- Training schedule: two phases, `λ_par` ramped from a low value to `0.6` over
  500+500 epochs (fetch tool gave slightly inconsistent numbers for the
  low-phase value, 0.06 vs 0.1 — treat the ramp-up structure as reliable, the
  exact low value as unverified).

**Data/augmentation (training, ShapeNet):** 4096 points per object via farthest
point sampling. Rotation augmentation is **asymmetric**: 0–180° around z, but
only **0–7.5° on x and y** — i.e., training assumes objects are close to
upright/canonically aligned and only randomizes yaw, not full SO(3). Small
random translation (radius 0.05) also applied. The paper does not state
whether ShapeNet point clouds are sampled from complete watertight meshes or
from simulated partial camera views — this was not resolved even after a
targeted second pass; treat "does SuperDec train on partial views" as
**unconfirmed**, which matters directly for the branch's incompleteness
question.

**Two released checkpoints** (from the GitHub README, not the paper body):
`shapenet` (unnormalized, for ShapeNet-distribution objects) and `normalized`
(for "objects from generic 3D scenes") — the authors explicitly ship a
checkpoint meant for exactly our use case (arbitrary real-object point cloud,
not ShapeNet canonical pose), which is a meaningfully stronger transfer claim
than "we also tried it on ScanNet++ once."

## Scaling to scenes

Full scenes are handled by running **Mask3D** instance segmentation first,
then decomposing each segmented object mask independently: "Each object is
centered and uniformly rescaled to a sphere of radius 0.5" before being fed to
the per-object network. This is architecturally identical to our pipeline
(single fused per-object PLY in, primitives out) — SuperDec's "scene" mode is
just our single-object mode looped over externally-provided segments. No
scene-level joint reasoning happens inside SuperDec itself; Mask3D's
segmentation failure modes (over/under-segmentation, class-agnostic instance
boundary errors) are inherited but not discussed in the paper.

## Quantitative results

**ShapeNet (Table 1), Chamfer-L2 (×10², lower better), and mean # primitives:**
| Split | SuperDec L2 | SuperDec #prims | SQ (Paschalidou) L2 |
|---|---|---|---|
| In-category | 0.047 | 5.8 | 0.279 |
| Out-of-category | 0.061 | 5.26 | (SuperDec ≈6× lower than SQ) |

**Scene-level real data (Table 2), Chamfer-L2 (×10²) and mean # primitives:**
| Dataset | SuperDec | Best baseline |
|---|---|---|
| ScanNet++ | 0.11, 5.18 prims | CSA: 0.41, 11.64 prims |
| Replica | 0.19, 6.58 prims | EMS: 2.12, 3.61 prims |

So on real, noisy, partial captures the reconstruction error is ~3–11×
better than the compared baselines while using fewer primitives — but note
**neither table reports whether the recovered primitive *type/count* is
semantically correct** (e.g., "this table leg should be 1 cylinder, got 3
blobs") — only geometric fitting error (Chamfer distance) and raw primitive
count. This is a direct gap against the branch's question about "at what
noise/incompleteness level do methods stop returning the correct primitive
type" — **SuperDec's own evaluation protocol cannot answer that question**;
it never scores primitive-type correctness, only surface-distance error.

**Robotics path planning (Table 3, ScanNet++, 15 scenes), success rate / memory:**
| Representation | Success rate | Memory |
|---|---|---|
| Occupancy grid | 100% | (not given) |
| Voxels | 98.78% | — |
| Point cloud | 89.57% | — |
| **SuperDec** | **91.71%** | **0.042 MB** |
| Cuboids | 61.23% | — |

SuperDec beats raw point clouds and cuboid abstraction on planning success
while using drastically less memory than occupancy/voxel grids, but is not as
safe as dense occupancy (91.71% vs 100% — i.e., primitive abstraction can miss
collision geometry that a dense representation would catch). Per-query
collision-check compute: 0.150 ms; forward pass for a full Replica scene:
0.13 s; Mask3D segmentation: 0.3 s; each LM refinement step: <1 s.

**Grasping:** "Given the superquadric parameters, we employ a superquadric-based
geometric method to compute grasping poses" — geometric/closed-form, not
learned, using the fitted shape parameters directly. No further method detail
or citation surfaced in the fetch (likely the Vezzani/Pattacini-style
superquadric grasp-planning line of work, common in that community, but this
paper does not spell out the reference — treat as plausible, not confirmed).

**Not present in this paper at all:** any comparison to RANSAC-family
plane/cylinder/sphere/cone fitting, efficient-RANSAC, or region growing.
Related work compares only to other learned/optimization abstraction methods
(SQ, CSA, EMS) and other scene representations (DBW, 3DGS variants). This is a
genuine gap for our Branch question "does the vocabulary need
plane+cylinder+sphere+cone RANSAC, or superquadrics" — SuperDec offers no
head-to-head evidence either way.

## Load-bearing assumptions (what could break transfer)

1. **Near-canonical/upright input pose.** Rotation augmentation caps x/y tilt
   at 7.5°; only yaw is fully randomized. Yet the paper separately
   characterizes ScanNet++ real objects as "noisier, partially observed, and
   subject to random orientation and translation" (Sec. 4.1) — i.e., the
   authors themselves flag a train/test pose-distribution mismatch and report
   the method still generalizes reasonably (Table 2 numbers above), but there
   is no ablation quantifying *how much* accuracy degrades as tilt grows
   beyond the ~7.5° training range. For our tabletop setting this is probably
   survivable (objects usually rest upright on a table under gravity, limited
   tilt) but is not free — an object lying on its side or grasped/re-posed
   would be pushing outside the demonstrated envelope.
2. **Slot-based, permutation-style output — primitive identity is not
   guaranteed stable.** The architecture is a fixed bank of 16 query slots
   refined by cross-attention (DETR-family design). Nothing in the paper
   claims or evaluates that a given semantic part (e.g., "the front face")
   consistently lands in the same slot index across two different scans of the
   same object, across small pose perturbations, or across re-runs. This is
   directly relevant to Branch's re-scan consistency question (RQ3) — the
   paper is silent on it, and the architecture gives no structural reason to
   expect index-level stability (slot correspondence at training time is
   presumably resolved by the segmentation-mass loss / an implicit matching,
   not by any canonical geometric rule like "largest primitive first" or
   "sorted by surface normal"). Treat repeatable identity as **unaddressed,
   likely unreliable without extra post-hoc canonicalization** (e.g., sort
   primitives by volume, position, or normal direction after the fact).
3. **No explicit boundedness/extent guarantee beyond the superquadric's own
   scale parameters.** Superquadrics are inherently bounded (finite `s_x, s_y,
   s_z`), which is a real advantage over infinite-plane RANSAC output for the
   Branch's "recover bounded faces (extent, boundary, centroid)" question —
   each primitive directly gives a centroid (its pose translation) and an
   extent (its scale params) with no extra fitting step. But a superquadric
   face is a *curved* patch parameterized by `ε₁, ε₂`, not a flat rectangle
   with a boundary polygon — extracting "face centre + face size" for a
   near-planar side of a box-like superquadric is a straightforward
   downstream calc (it degenerates toward a rectangle as `ε→0`), but the paper
   doesn't discuss this or provide the utility.
4. **No noise-level ablation.** The paper reports results at whatever noise
   level ScanNet++/Replica naturally have (unspecified magnitude, unspecified
   sensor), with no synthetic noise sweep. It cannot supply the quantified
   "noise σ → primitive-type-recovery-rate" curve Branch 1 asks for; only
   before/after point that "real capture is noisier and it still works,
   somewhat, better than baselines."
5. **Existence/confidence signal is weak.** `L_exist`/`α_j` ablates to
   "marginal" improvement over a naive mass threshold — this is not a
   calibrated per-primitive confidence score usable as a fit-reliability
   signal for a fallback trigger (Branch's introspection questions). SuperDec
   offers no MDL-style score, no residual-based confidence, no
   outlier-probability output analogous to EMS/MBF/GNC.

## What transfers to our setting

- **Architecture pattern is directly reusable as a design reference:**
  per-object point cloud in (already isolated, as ours is — no scene
  segmentation needed on our side), fixed small bank of primitives out, each
  with closed-form pose + bounded extent. This maps cleanly onto "give me face
  3's centroid and size," better than raw RANSAC plane equations.
- **Released "normalized" checkpoint is a real, no-retraining-needed candidate**
  to try directly on our fused PLYs (mentioned in Branch questions about
  released weights + real-scan generalization) — worth an actual runtime
  smoke test given code+weights are public.
- **Superquadric closed-form radial distance → usable for grasp/approach-pose
  geometry** the way the paper already does for grasping; likely portable to
  "compute an orthogonal viewpoint from the fitted primitive's surface normal
  at closest approach" for our view-planning use.
- **Runtime is cheap** (0.13 s/scene forward pass, LM step <1 s) — compatible
  with our per-object, on-demand budget.

## What does NOT transfer / negative findings

- **No primitive-type-correctness metric exists in this paper.** Everything is
  Chamfer distance + primitive count; there is no ground truth for "is this a
  cylinder" vs "is this 3 misshapen blobs," so SuperDec cannot answer our
  central Branch-1 question about correctness thresholds under noise.
- **No RANSAC baseline whatsoever** — cannot settle the
  RANSAC-vs-superquadric vocabulary question from this paper alone.
- **No calibrated confidence/fit-reliability output** — existence head is a
  near-duplicate of the segmentation-mass threshold, not an independent
  uncertainty estimate; nothing here satisfies the "fit score → trust the
  primitive" introspection requirement.
- **No demonstrated repeatability/identity-stability across re-scans** — an
  open risk for RQ3, not a validated capability.
- **Small-tilt training assumption vs. paper's own admission that real test
  objects have "random orientation"** is an unresolved tension the authors
  note but do not quantify — we should not assume robustness to large,
  uncontrolled object tilt without testing it ourselves.
- **No view-planning or per-primitive viewpoint generation anywhere in this
  paper.** Downstream uses shown are: path planning (collision avoidance
  through/around primitives), grasping (pose from primitive geometry), and
  image generation control (ControlNet depth conditioning) — not camera view
  selection for inspection. This paper is not a precedent for "decompose then
  generate per-primitive viewpoints"; it's evidence primitives are usable for
  *some* downstream robot geometry tasks, not that one for ours exists yet.

## Answers to specific Branch questions (direct hits only)

- **"Which methods recover bounded faces (extent, boundary, centroid)?"** —
  SuperDec: yes, structurally, via `(s_x,s_y,s_z)` + pose per primitive; no
  boundary-polygon extraction shown but geometrically derivable.
- **"Assume canonical object orientation — a direct constraint violation?"** —
  Partially yes: training augmentation only randomizes yaw (±180° z) with
  small x/y tilt (≤7.5°), even though the authors' own real-data description
  admits test objects have arbitrary orientation. Not a hard requirement (no
  code-level canonicalization step at inference beyond centering+rescaling)
  but an unquantified robustness gap for objects tilted well past their
  training envelope.
- **"Released weights and real-scan generalization?"** — Yes: code +
  checkpoints public on GitHub, with a checkpoint specifically for
  "normalized"/generic-scene objects (i.e., not ShapeNet-only evaluation),
  and Table 2 numbers are exactly the "real ScanNet++/Replica" evaluation the
  Branch asks about.
- **"Does introspection literature support two-tier (decomposition, else
  fallback)?"** — Not from this paper; its existence gate is a segmentation-mass
  threshold, not a validated failure predictor.
