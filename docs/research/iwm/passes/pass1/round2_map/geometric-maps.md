# IWM pass 1, round 2: branch 1, `geometric-maps`

**Date:** 2026-10-06
**Question:** which explicit and neural geometric maps exist, and which run in real systems, as
compression of depth and point measurements? What do they say about RQ4 to RQ13?

**Tags.** READ: read in the body today. ABSTRACT: abstract, landing page, README or search
result text today. RECALLED: my memory, not checked today. DERIVED: our own arithmetic from
stated assumptions. VP-MAP: taken from our view-planning map
(`docs/research/view-planning/passes/pass1/round2_map/view-quality-scoring.md`), not re-checked.
Body reads went through a summarising fetch tool, except ElasticFusion and BundleFusion, whose
PDFs I converted to text and read directly.

---

## 1. Headline findings

**H1. At object scale, keeping every raw view and re-fusing is cheap. That is the proven way to
keep fused data correctable.**
BundleFusion stores every RGB-D frame with an "integrated pose" and an "optimized pose", and
de-integrates and re-integrates up to 10 frames per new frame, largest pose change first (READ,
Sec. 5.2 to 5.3). It costs 20 GB of CPU RAM at room scale because frames number in thousands (READ,
Sec. 7, Tab. 1). We have tens of views. 40 D405 depth frames at 848x480x2 bytes come to about
33 MB (DERIVED). So the fused grid can be a cache that is rebuilt from per-view data.

**H2. The serious systems never fuse across views until poses are final. They keep a per-view or
per-submap unit, and fusion is a view over those units.**
MASt3R-SLAM keeps one canonical pointmap plus confidence per keyframe, moves keyframes rigidly in a
global optimisation, and has no global fused volume (READ, html, method and limitations sections).
Voxgraph (SDF submaps aligned on CPU), LoopSplat (Gaussian submaps), Neural Graph Map (small fields
anchored to pose-graph keyframes) and SEAM (2026, evidence anchored to submaps so it survives
trajectory changes) follow the same pattern (ABSTRACT). Our error is one rigid shift per view, so
"one view = one rigid submap" matches it exactly.

**H3. Per-element observation quality is stored everywhere, but only as one scalar weight or
count. Direction, distance and which views contributed are thrown away.**
TSDF keeps (distance, weight) (nvblox READ; BundleFusion READ). Surfels keep a weight, a radius and
two timestamps (ElasticFusion READ, Sec. III). Occupancy keeps a log-odds value. Only a few keep
direction: Directional TSDF (6 sub-volumes per voxel, ABSTRACT), GAVIS (2026, per-Gaussian
visibility as spherical harmonics, ABSTRACT), SEAM (direction-aware voxel evidence, ABSTRACT).
No map keeps a per-element list of contributing views. The view-to-element link must be added by us.

**H4. Multi-resolution saves 2x to 10x in memory, and the 2025 methods refine where the
measurements disagree, not where the robot is close.**
Wavemap at 2 cm: 39.09 MB against Octomap 50.94, supereight2 285.07, Voxblox 348.15 (READ via ar5iv,
Tab. II, Panoptic Flat). MrHash (2025, ACM TOG) splits voxels where the running SDF variance is
high: 155.7 MB multi-resolution against 242.2 MB single-resolution at 1 to 2 cm RGB-D (READ, Tab. 3).
"Fine where the data disagrees" is a usable rule for "fine detail only where needed".

**H5. No geometric map of this family has been handed raw to a language model. What reaches a
model is always a few derived, discrete items, behind a constrained interface.**
Navigation work turns the occupancy grid into frontier candidates shown on images or ranked by the
model (LGR, ABSTRACT; SG-Nav, ABSTRACT). Reasmory (2026) keeps point cloud, cameras, depth and
object instances, and exposes them through a small DSL. Unverified free-form primitives lowered
GPT-5-mini on MindCube from 58.0 to 53.2. The verified DSL gave the best result everywhere
(READ, Tab. 8). That agrees with MIT 2510.16643.

**H6. Our 10 mm cross-view shift is larger than every voxel size these systems are tuned for, so
fusing as-is gives silent blur, not an error.**
Object and room RGB-D systems run at 4 mm (BundleFusion default, READ) to 1 cm (nvblox, MrHash,
READ), and assume pose error below a voxel. TSDF weighted averaging of two copies 10 mm apart gives
one smeared surface whose weight looks high (RECALLED mechanism; matches our own fused-cloud
observation). A high weight then falsely reads as "well observed".

---

## 2. Approach cards

### 2.1 Occupancy grids and octrees: OctoMap, UFOMap, Bonxai, OpenVDB-based grids
- **Atomic unit:** voxel holding a log-odds occupancy value (RECALLED).
- **Kept:** one probability per cell. Colour optional (UFOMap) (ABSTRACT).
- **Discarded:** raw points, view direction, distance, per-view identity.
- **Hierarchy:** octree. OctoMap prunes 8 equal children into one parent (RECALLED). Bonxai and
  VDB grids use a fixed 3-level sparse tree, not adaptive (ABSTRACT).
- **Update, drift, conflicts:** Bayesian log-odds update with clamping; conflicts average out.
  No drift handling: the map is in the world frame and wrong poses are baked in (RECALLED).
- **Known vs unknown:** OctoMap: unknown means "no node" (RECALLED). UFOMap stores occupied,
  free and unknown explicitly and finds this both faster and smaller (ABSTRACT).
- **Identity and merging:** not addressed (no entities).
- **Links geometry, image, words:** not addressed.
- **Frame:** world.
- **Reader and interface:** planners (collision, frontier search) through C++ queries.
- **Cost:** Bonxai README: create 1165 us against OctoMap 25522 us. Memory 20 to 40 % more than
  OctoMap on sparse clouds, possibly less than half on dense ones (ABSTRACT, README).
- **Training:** none.
- **Used where:** OctoMap is the ROS and MoveIt standard for 3D occupancy (RECALLED). UFOMap and
  Bonxai: open source, ROS (ABSTRACT).

### 2.2 Wavelet occupancy: wavemap (RSS 2023)
- **Atomic unit:** octree node holding Haar wavelet coefficients (floats), not absolute values
  (READ, ar5iv).
- **Kept:** occupancy log-odds at all resolutions, implicitly in sync.
- **Discarded:** detail below the error bound; everything except occupancy.
- **Hierarchy:** yes, coarse-to-fine integration refines only where the approximation error needs
  it (READ, Alg. 1).
- **Update, drift, conflicts:** log-odds with upper and lower bounds; no drift handling.
- **Known vs unknown:** classes such as "FullyUnobserved", "PossiblyOccupied",
  "FreeOrUnobserved" (READ).
- **Identity, links:** not addressed.
- **Frame:** world.
- **Reader:** planners. Waverider (2405.13617) uses the hierarchy for reactive avoidance (ABSTRACT).
  Has a Python API (ABSTRACT, docs).
- **Cost:** Tab. II, see H4. Newer College at 5 cm: 241.84 MB against Octomap 981.02 MB (READ). Docs:
  newer code uses up to 50 % less RAM (ABSTRACT, docs).
- **Training:** none. **Used where:** open source (ETH ASL), research robots. No product found.

### 2.3 TSDF with voxel hashing: KinectFusion lineage, Voxblox, nvblox, VDBFusion, DB-TSDF
- **Atomic unit:** voxel holding truncated signed distance and weight; colour optional. nvblox
  groups 8x8x8 voxels into hashed blocks (READ, nvblox html). BundleFusion likewise (READ, Sec. 5.1).
- **Kept:** weighted mean surface distance near the surface; mesh extracted by marching cubes.
- **Discarded:** individual measurements, per-view identity, direction. Weight is "constant or from
  a sensor error model" (nvblox READ).
- **Hierarchy:** two-level (hash plus block). ESDF derived on top for planning (Voxblox, nvblox).
- **Update, drift, conflicts:** weighted running average. nvblox assumes poses are given and does
  not handle pose corrections (READ). Dynamic content goes to a separate occupancy layer that
  decays towards 0.5 (READ, Isaac ROS technical details).
- **Known vs unknown:** weight 0 or unallocated block means unknown. No free-space semantics
  beyond the truncation band unless a separate freespace layer is used (READ, layer list).
- **Identity, links:** not addressed. Colour layer is the only image link.
- **Frame:** world.
- **Reader:** motion planners (nvblox feeds cuRobo, ABSTRACT; Nav2 costmap in Isaac Perceptor, ABSTRACT).
- **Cost:** nvblox tested 1 to 10 cm TSDF (READ). Default voxel 5 cm in Isaac ROS, about 1 cm
  mode added (ABSTRACT, search text). Paper gives speed-ups (177x surface, 31x ESDF) but no MB
  (READ, abstract).
- **Training:** none.
- **Used where:** nvblox in NVIDIA Isaac Perceptor and Isaac ROS (shipped SDK). Voxblox, VDBFusion,
  Open3D TSDF widely used open source (RECALLED). DB-TSDF: ICRA 2026, CPU, ROS 2 (ABSTRACT).

### 2.4 Directional and variance-adaptive TSDF: DTSDF (IROS 2019), MrHash (2025)
- **Atomic unit:** DTSDF: six sub-voxels per voxel, one per axis direction, fused by normal
  alignment, so opposite surfaces stay apart (ABSTRACT plus search text). MrHash: distance, weight,
  colour and SDF variance per voxel (READ).
- **Hierarchy:** MrHash merges low-variance blocks to coarse (2 cm) and keeps high variance fine
  (1 cm), variance by Welford running update (READ).
- **Known vs unknown:** weight; MrHash adds a disagreement measure (variance).
- **Cost:** see H4 (READ, Tab. 3). No object-scale experiments (READ).
- **Used where:** papers plus open source code. Relevance: variance is exactly the signal that
  exposes our rigid cross-view shift instead of hiding it.

### 2.5 Surfels: ElasticFusion and successors
- **Atomic unit:** surfel with position, normal, colour, weight, radius, initialisation time t0,
  last update time t (READ, Sec. III).
- **Kept:** surface disc per patch; radius stands for local area. Weight acts as confidence.
- **Discarded:** raw frames; direction of observation.
- **Hierarchy:** none. Active and inactive sets split by a time window (READ).
- **Update, drift, conflicts:** moving average fusion. Loop closure deforms all surfels with an
  embedded deformation graph whose nodes are sampled from surfels, each surfel moved by its k
  nearest nodes (READ, Sec. IV, Alg. 1). No pose graph.
- **Known vs unknown:** weight low means unstable. Unobserved space is simply absent.
- **Identity:** surfels merge by proximity and normal (RECALLED, Keller 2013 rules).
- **Links:** colour only. Timestamps give a coarse link back to frames.
- **Frame:** world.
- **Reader:** the tracker itself. **Training:** none.
- **Used where:** ElasticFusion open source, research. Surfels also underlie SurfelMeshing
  (ABSTRACT). No product found.

### 2.6 Meshes with deformation: Kimera mesher, Kimera-PGMO
- **Atomic unit:** mesh vertex and face. **Kept:** surface only.
- **Update, drift:** PGMO builds one deformation graph from pose nodes and mesh vertices and
  optimises trajectory and mesh together after loop closure. Simplifies by merging vertices that
  fall in the same octree voxel (ABSTRACT, GitHub).
- **Known vs unknown:** not addressed. **Identity, links:** not addressed at this layer.
- **Used where:** MIT SPARK Kimera and Hydra stacks, open source (ABSTRACT).

### 2.7 Drift-correctable fusion: BundleFusion, keyframe re-integration, voxgraph, Open3D fragments
- **Atomic unit:** TSDF voxel, plus the stored frame or keyframe or submap that can be re-fused.
- **Kept:** BundleFusion keeps every frame with depth, colour, integrated pose and optimised pose
  (READ, Sec. 5.3). Maier et al. (BMVC 2017) fuse consecutive frames into keyframes and re-integrate
  only those, with less memory and up to 93 % faster (ABSTRACT). Voxgraph keeps overlapping SDF
  submaps and aligns them correspondence-free on a CPU (ABSTRACT).
- **Update, drift:** symmetric integrate and de-integrate:
  `D' = (D W - w d) / (W - w)`, `W' = W - w` (READ, Sec. 5.2). Ten frames re-fused per new frame,
  sorted by pose change (READ). Open3D's pipeline registers fragments first and integrates the TSDF
  last (ABSTRACT, docs).
- **Known vs unknown:** as TSDF.
- **Cost:** BundleFusion: 20 GB CPU for sequence Apt 0; GPU 5.3 GB at 4 mm and 1.9 GB at 1 cm;
  SUN3D 14785 frames: 34.7 GB CPU, 7.3 GB GPU (READ, Sec. 7). Voxgraph: 120 x 80 m map optimised in
  under 4 s on an i7-8650U (ABSTRACT).
- **Training:** none. **Used where:** Open3D pipeline widely used; voxgraph open source; BundleFusion
  research code.

### 2.8 Keyframe pointmaps from feed-forward reconstructors: MASt3R-SLAM, VGGT-SLAM
- **Atomic unit:** keyframe with a canonical pointmap and per-pixel confidence (READ, MASt3R-SLAM).
- **Kept:** per-keyframe dense geometry; confidence-weighted running average of new frames into the
  keyframe's own pointmap (READ).
- **Discarded:** non-keyframes after fusion. No global volume (READ).
- **Update, drift:** pose graph with Sim(3)-style keyframe updates, Gauss-Newton; geometry inside a
  keyframe is not refined globally (READ). VGGT-SLAM aligns submaps with 15-DoF SL(4) homographies
  because similarity transforms fail for uncalibrated cameras (ABSTRACT).
- **Known vs unknown:** per-pixel confidence only.
- **Frame:** per-keyframe camera frame, placed in world by pose. This is the "viewpoint frame"
  option of RQ13, already in use.
- **Cost:** about 15 FPS, keyframe step about 164 ms (READ). **Training:** pretrained network,
  none on our side. **Used where:** open-source research code; spreading fast (VGGT-SLAM++,
  VGGT-Align, VGGT-GS SLAM in 2026, ABSTRACT titles only).

### 2.9 Neural implicit maps: iMAP, NICE-SLAM, Co-SLAM, Neural Graph Map, BundleSDF
- **Atomic unit:** network weights or feature grid; BundleSDF: a neural object SDF centred on the
  object plus a pool of posed memory frames chosen for view diversity (ABSTRACT).
- **Kept:** continuous SDF or density and colour. **Discarded:** raw frames (except memory pool).
- **Update, drift:** retraining online. Neural Graph Map anchors many small fields to keyframes of a
  sparse SLAM pose graph so loop closure moves fields instead of re-fusing (ABSTRACT, WACV 2025).
- **Known vs unknown:** implicit; uncertainty needs extra machinery (ActiveNeRF, FisherRF, VP-MAP).
- **Identity, links:** not addressed. **Frame:** world, or object frame (BundleSDF).
- **Training:** per-scene optimisation always. **Used where:** papers and research code.
  BundleSDF and FoundationPose-style object work is the closest object-scale relative (RECALLED for
  FoundationPose).

### 2.10 3D Gaussian splatting maps: SplaTAM, MonoGS, LoopSplat, GS surfels, compact 3DGS
- **Atomic unit:** anisotropic Gaussian (position, scale, rotation, opacity, colour). SplaTAM uses
  a rendered silhouette to tell mapped from unmapped regions (ABSTRACT).
- **Hierarchy:** none in base form; anchor-based compression adds one (CompGS, ABSTRACT).
- **Update, drift:** per-scene optimisation. LoopSplat: Gaussian submaps, loop edges from
  registering splats, 3DV 2025 (ABSTRACT). GLC-SLAM: global BA then map deformation (ABSTRACT).
- **Known vs unknown, quality:** GAVIS (2026): per-Gaussian anisotropic visibility in spherical
  harmonics, 200 FPS uncertainty (ABSTRACT). OUGS (Eurographics 2026): uncertainty from covariance of
  Gaussian position, scale and rotation, masked to the object (ABSTRACT). PUP 3D-GS (CVPR 2025):
  Fisher-style sensitivity per Gaussian used to prune 90 % of Gaussians (ABSTRACT). Uncertainty and
  compression are the same quantity here.
- **Object scale:** ObjSplat (T-ASE 2026): Gaussian surfels for active object reconstruction with
  geometry-aware view scoring (ABSTRACT).
- **Cost:** LightGaussian: 15x average compression (ABSTRACT). Full-scene MB figures were not
  verified today.
- **Training:** per-scene optimisation; no pretraining. **Used where:** papers, Nerfstudio-style
  tools; no robot product found.

### 2.11 Point cloud codecs: octree coding, MPEG G-PCC, Draco
- **Atomic unit:** occupied octree cell (G-PCC octree mode), or mesh and points (Draco).
- **Kept:** geometry and attributes for transmission; trisoup mode for dense surfaces (ABSTRACT,
  search text). **Discarded:** everything about observation.
- **Reader:** decoders, not planners. Storage format only. **Used where:** G-PCC is an MPEG standard
  (ABSTRACT); Draco is Google open source (RECALLED).

### 2.12 Observation-quality state in active reconstruction and inspection (from VP-MAP plus new)
- **Per-voxel state used by IG metrics:** occupancy probability plus derived classes (unknown,
  rear-side, proximity count) (Isler 2016, Delmerico 2018; RECALLED with VP-MAP).
- **Per-region confidence:** Wu 2014 Poisson confidence map (VP-MAP). SEE uses raw measurement
  density, no grid (VP-MAP). Vasquez-Gomez marks "occplane" voxels (VP-MAP).
- **Per-triangle constraints:** Bircher 2015 checks incidence angle and min and max range per mesh
  triangle; open-source StructuralInspectionPlanner (ABSTRACT). GSD and 1/cos effective resolution in
  UAV inspection (VP-MAP).
- **Direction-wise:** AURORA (Sep 2026): Ray-GPIS uncertainty per viewing ray for in-hand
  objects (ABSTRACT). GAVIS (see 2.10).
- **Pattern:** quality is computed per element, used for the next view, then dropped. Never
  persisted as a readable field, never shown to a language model.

---

## 3. Per research question

- **RQ4 atomic block.** Options in use: voxel scalar (occupancy, TSDF), surfel (point, normal,
  radius, weight, times), Gaussian, mesh vertex, keyframe pointmap. For correctability the proven
  block is the per-view or per-keyframe unit (BundleFusion, MASt3R-SLAM). The voxel is a derived cache.
- **RQ5 hierarchy.** Only spatial: octree pruning (OctoMap), wavelets (wavemap), variance-driven
  levels (MrHash), object submaps (Panoptic Multi-TSDFs, ICRA 2022, ABSTRACT). This family is silent
  on semantic hierarchy.
- **RQ6 runtime update without drift.** Four proven patterns: de-integrate and re-integrate
  (BundleFusion, Maier 2017); rigid submaps (voxgraph, LoopSplat, MASt3R-SLAM keyframes); deformation
  graph (ElasticFusion, Kimera-PGMO, GLC-SLAM); joint pose and map optimisation (Occupancy-SLAM, T-RO
  2025, ABSTRACT). For one rigid shift per view, the rigid per-view unit is the exact fit.
- **RQ7 who resolves conflicts.** Always code: a back-end optimiser and a fixed fusion rule. This
  family is silent on agents owning the map.
- **RQ8 contract.** In: depth, pose, optional colour and masks. Out: occupancy or distance queries,
  meshes, ESDF, renders. Readers are planners in code. Reasmory's DSL (H5) is the only model-facing
  contract found.
- **RQ9 keep vs discard.** Fusion keeps the mean surface and a weight, drops measurement identity.
  BundleFusion and MASt3R-SLAM show the cheap answer at small scale: keep raw views, fuse on demand.
- **RQ10 how well known.** Scalars only in mainstream maps (weight, log-odds, confidence). Direction:
  DTSDF, GAVIS, SEAM. Disagreement: MrHash variance. Explicit unknown: UFOMap, wavemap classes.
  "Not there" vs "not looked" is solved for free space by occupancy maps, not by TSDF or surfels.
- **RQ11 identity and merging.** Geometry merges by position (voxel cell, surfel proximity).
  Wrong merges are undone only by de-integration (BundleFusion) or by keeping submaps apart. Branch 3
  owns the rest.
- **RQ12 linking geometry, image, words.** Colour per element only. Timestamps (ElasticFusion) and
  keyframe ownership (MASt3R-SLAM) are the only back-links to images. Words: this family is silent.
- **RQ13 frame.** World frame for volumetric maps; per-keyframe camera frame (MASt3R-SLAM);
  object frame (BundleSDF, Panoptic Multi-TSDFs submaps). Translation to the asker is a rigid
  transform in code.

---

## 4. What does not transfer, and why

- **Room-scale voxel sizes.** Defaults are 5 cm (nvblox) to 4 mm (BundleFusion). At our 10 mm rigid
  error, a voxel below about 15 mm does not mean the same place across views (contract Sec. 7).
  Fine voxels only make sense after per-view re-registration.
- **Fusion that assumes pose error below a voxel.** TSDF, surfel and occupancy fusion blend shifted
  copies and raise the weight, so "seen often" looks like "seen well" (H6).
- **Free-space machinery.** OctoMap, ESDF and frontiers serve collision and exploration. For one
  object on a table, free space is mostly known; what matters is surface coverage and quality.
- **Neural and Gaussian maps as the store.** They need per-scene optimisation (GPU minutes), have no
  discrete entities to name, and give uncertainty only through extra machinery. Their uncertainty
  ideas (direction-wise visibility, parameter covariance) transfer; the store does not.
- **Feed-forward reconstructors as truth.** VGGT and MASt3R give confidence, not metric error; with a
  calibrated D405 and arm poses we already have better metric geometry. Their map pattern
  (keyframe units) transfers, their geometry is not needed.
- **Raw map to the model.** Nothing here is text-shaped. A text-only planner needs derived items
  and queries (H5).
- **Point cloud codecs.** Compress bytes for transport; they say nothing the planner can use.

---

## 5. Ranked reading list for deep reading

1. **BundleFusion**, https://arxiv.org/pdf/1604.01093. Already read Sec. 5 and 7. Extract the chunk
   and keyframe hierarchy (Sec. 4) as a model for "per-view unit plus re-fused cache".
2. **MASt3R-SLAM**, https://arxiv.org/html/2412.12392. Keyframe pointmap and confidence update rule,
   when a new keyframe starts. Template for a per-view unit with a quality field.
3. **MrHash**, https://arxiv.org/html/2511.21459. Variance per voxel and its thresholds; test whether
   variance exposes our rigid shift on our data. Last 12 % of the html not read.
4. **SEAM**, https://arxiv.org/abs/2609.18819. How submap-anchored, direction-aware evidence is
   stored and reprojected after trajectory change. Closest to "quality that survives correction".
5. **GAVIS**, https://arxiv.org/abs/2605.30342. Spherical-harmonic visibility per element: how many
   coefficients, how updated. Candidate encoding for "seen from which directions".
6. **Reasmory**, https://arxiv.org/html/2606.00963v1. The DSL primitives and the verifier; how an
   explicit 3D memory is exposed to a model. Note it renders images; we need a text-only variant.
7. **Wavemap**, https://ar5iv.labs.arxiv.org/html/2306.01279. Coarse-to-fine integration and the
   three-way known classes, for a hierarchical coverage field.
8. **Voxgraph**, https://arxiv.org/abs/2004.13154. Correspondence-free SDF submap alignment, a
   candidate method to estimate our per-view rigid shift.
9. **Directional TSDF**, https://arxiv.org/abs/1908.05146. Per-direction sub-volumes; cheap way to
   keep "which side was seen".
10. **ElasticFusion**, https://www.roboticsproceedings.org/rss11/p01.pdf. Already read Sec. III and IV.
    Only needed if we choose surfels.
11. **Panoptic Multi-TSDFs**, https://arxiv.org/abs/2109.10165. Per-object submaps at their own
    resolution, for the multi-object step.
12. **ObjSplat**, https://arxiv.org/abs/2601.06997 and **OUGS**, https://arxiv.org/abs/2511.09397.
    Object-scale quality signals in Gaussian maps.

---

## 6. Gaps and open questions

- **Not found:** any map that stores, per surface element, the list of views that saw it with angle
  and distance. Inspection planners compute it and discard it (VP-MAP; Bircher ABSTRACT).
- **Not found:** any geometric map read by a text-only model through tools. Closest: Reasmory (VLM,
  renders images), LGR and SG-Nav (frontiers, navigation), VoxPoser (model writes into voxels).
- **Not found:** object-scale benchmarks for wavemap, nvblox, MrHash (MrHash has none, READ).
- **Not found:** memory figures for nvblox in MB (paper gives none, READ). 3DGS per-scene MB figures
  seen only in search text; not verified, so not reported.
- **Not verified:** G-PCC "about 2 bits per occupied voxel" appeared in search text without a
  primary source; not used. BundleSDF chamfer numbers came from secondary sites; not used.
- **Open:** whether to estimate our per-view rigid shift (voxgraph-style SDF alignment, ICP of view
  against the rest) before fusing, or to keep views apart and let the variance show it.
- **Open:** whether a view-quality field should be a scalar weight (cheap, standard) or directional
  (DTSDF, GAVIS, SEAM). Our questions ("which view shows it", "is this view good enough") point to
  directional plus view identity.
- **Search trail (main queries):** LLM plus occupancy or voxel; wavemap memory; nvblox; submaps and
  loop closure 2025; Gaussian SLAM loop closure; BundleFusion re-integration; scene graph plus
  frontier plus LLM; VGGT-SLAM; active object reconstruction with Gaussians 2025; BundleSDF; TSDF
  per-voxel provenance; 3DGS compression; supereight2 multi-resolution; Bircher inspection; agentic
  3D reasoning 2026; uncertainty in 3DGS 2025; volumetric mapping 2026.
