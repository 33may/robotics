# MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting

**arXiv:2403.03174v3** (Liu, Fang, Abbeel, Levine — UC Berkeley). Read from full PDF text (arxiv.org/pdf/2403.03174), all sections + appendix.

## Core method

MOKA converts a manipulation-affordance-selection problem into a series of **visual multiple-choice questions** answered by GPT-4V over a single annotated 2D image, instead of asking the VLM to emit continuous coordinates.

**Point-based affordance representation** (Sec. IV-A): a fixed dictionary of keypoints/waypoints defined once, reused across all task types —
- `x_grasp`, `x_function` (on the in-hand object `o_in-hand`), `x_target` (on the unattached object `o_unattached`)
- `x_pre-contact`, `x_post-contact` (free-space waypoints)
- `R_grasp`, `R_manipulate` (orientations, predicted as one of a **finite set of options**: forward/backward/upside/downside/left/right — Sec. IV-D)

**Hierarchical prompting** (Sec. IV-B, Fig. 2, Eq. 1–2):
1. High-level: `y_high = M([ρ_high, l, s_0])` — decomposes the free-form instruction into an ordered list of subtasks, each a dict with `instruction`, `object_grasped`, `object_unattached`, `motion_direction` (TABLE IV).
2. Low-level: `y_low^k = M([ρ_low, y_high^k, f(s_t(k))])` — per subtask, queries again on a **freshly marked image** to fill in the affordance dict (TABLE V–VII).

**Mark-based visual prompting** (Sec. IV-C, explicitly inspired by Set-of-Mark [Yang et al., arXiv:2310.11441]):
- Object keypoints: GroundedSAM (GroundingDINO + SAM) segments `o_in-hand`/`o_unattached` from a text prompt of their names; farthest-point sampling picks **K boundary points + 1 geometric-center point** on the mask contour, each overlaid and indexed — red `P[i]` for the in-hand object, blue `Q[j]` for the unattached object. **K's numeric value is never stated in the paper body** (not in main text or appendix — only "K" as a symbol; Sec. IV-C, Appendix B-B).
- Free-space waypoints: the whole image is divided into a **5×5 grid**, columns `a,b,c,d,e` left→right, rows `1,2,3,4,5` bottom→top (**chess notation**, explicit design choice — Sec. IV-C, Appendix B-B, Fig. 6). The VLM picks a *tile*, not a coordinate; the exact waypoint is sampled uniformly inside the chosen tile at execution time.
- Height (z) of free-space waypoints is not spatially marked at all — the VLM just picks one of two categorical labels, "same" or "above" the target keypoint's height (Sec. IV-D, TABLE VII). This is the paper's second multiple-choice reduction, applied to the one axis the 2D marking scheme cannot resolve.
- Orientation is reduced the same way: a finite set of textual options relative to the grasp→function axis (Sec. IV-D).

**Lifting to 3D / execution** (Sec. IV-D): object-attached keypoints are deprojected via the registered depth image at the chosen pixel; free-space waypoints are deprojected using the VLM-chosen height label. Grasping itself is **not** left to the VLM — a DexNet-2.0-style antipodal grasp sampler proposes 30 grasp candidates from the depth crop, and the system executes the sampled grasp *nearest to* the VLM's chosen `x_grasp`, not the raw VLM point (Sec. IV-D, Appendix B-D). This is a load-bearing design choice: the VLM is only ever trusted to disambiguate *which region*, not to output anything that needs sub-cm precision — geometry/physics-sensitive precision is handed to a classical planner.

## Concrete results (TABLE I, TABLE IX)

- 4 tasks × 2 subtasks each (Table Wiping, Watch/Ultrasound Cleaning, Gift Preparation, Laptop Packing), 10 trials/subtask, real 7-DoF Franka Emika + 2F-85 gripper, 3 cameras (top-down primary for reasoning; front + wrist ZED added only for distillation), 5 Hz control.
- MOKA zero-shot ≈ matches or beats Code-as-Policies and VoxPoser baselines on 7/8 subtasks; VoxPoser ties MOKA except on the tool-use subtask (table-wipe subtask 2: VoxPoser 0.0 vs MOKA 0.6) — VoxPoser's voxel-map resolution is reported as brittle/hyperparameter-sensitive (Sec. V-B prose, no ablation table given).
- MOKA in-context (2 examples) reaches 0.9–1.0 on most subtasks; MOKA-distilled (Octo, 50 demos/task, full fine-tune) also improves over zero-shot and removes the VLM from the loop entirely at inference (no more "reasoning failures" possible, Fig. 4 caption).
- **Ablation, TABLE IX** (reasoning success /10, not end-to-end task success):
  - MOKA (full): 0.6–1.0 range.
  - **MOKA w/o hierarchy** (skip subtask decomposition, annotate all scene objects at once, one-shot VQA): collapses to **0.0–0.2** — "decreases performance by a large margin" (Sec. Appendix C-A prose). This is the single largest ablated effect in the paper.
  - MOKA w/o keypoint/tile description (remove the definitions in TABLE VI from the prompt): drops modestly, e.g. Table Wiping 0.7→0.5, Laptop Packing 0.8→0.4.
  - MOKA w/o chain-of-thought (remove the step-by-step instruction in TABLE VII): similar modest drop, e.g. Watch Cleaning 1.0→0.8, Laptop Packing 0.8→0.6.
  - All three ablations compare against the same full MOKA row, same 10-trial protocol — this is an **ablated, ranked claim** (hierarchy >> description ≈ CoT), not a bare demonstration.
- **Failure breakdown** (Fig. 4, qualitative only, no counts given): reasoning failures = grasping broom upside-down (confused grasp/function point), pointing tool in wrong direction (wrong target angle choice), pressing laptop hinge (wrong target keypoint). Execution failures = gripper narrowly missing button, filler disassembling mid-trajectory, cable slipping through fingers. No numeric breakdown table — purely illustrative in the figure.
- No random-baseline or menu-size sweep is reported anywhere. No comparison of the P/Q/tile-index encoding against an alternative encoding (e.g., raw pixel coordinates, verbal description) — the paper's "coordinates are ill-defined/challenging for VLMs" claim (Sec. IV-B prose, Sec. IV-C prose) is **asserted, not ablated**.

## Load-bearing assumptions

1. **Single fixed top-down RGB(D) view per decision.** All marking and all VLM querying happens on one image from "the primary camera used in this paper" (Appendix A-A). There is no viewpoint choice anywhere in the pipeline — the camera pose is not part of the action space at all.
2. **Objects are cleanly segmentable and largely unoccluded from that one view.** GroundedSAM must find both named objects; the system explicitly re-poses the arm to a neutral pose and re-captures a "clean and non-occluded image" before every subtask's low-level query (Appendix B-D) — segmentation quality is a hard dependency, worked around procedurally, not solved.
3. **The action space is flat and per-object, not spatial/relational.** Candidate keypoints are independent boundary/center samples on a single object's silhouette; the grid is a static, object-agnostic tiling of the whole frame. Nothing in the representation encodes adjacency, relative direction, or "this option is opposite/orthogonal to that one" — every choice is presented as an unordered flat menu with no inter-option relation.
4. **VLM is trusted only for coarse-region selection, never precision.** All precision-critical mappings (grasp pose, exact waypoint within a tile, exact 3D height) are delegated to classical/geometric components (grasp sampler, uniform-in-tile sampling, one of two height labels).
5. **Depth/point-cloud quality is not a stated concern.** Keypoints are deprojected from a registered depth image at a single pixel — no fused multi-view cloud, no discussion of depth dropout, sparsity, or glossy-surface holes anywhere in the paper.

## What transfers to our setting

- **The central mechanism** — "convert coordinate generation into a multiple-choice/VQA problem by overlaying indexed marks, because VLMs are much better at discrete selection than continuous regression" (Sec. IV-C, Sec. VI conclusion) — is direct, ablation-free-but-directly-stated support for our RQ4 commitment to a discrete, pre-enumerated view action space instead of asking the VLM for metric poses. This is the paper's strongest and most repeated claim.
- **Symbolic, addressable labels over raw coordinates**: `P[i]`/`Q[j]` indices and chess-notation grid tiles (`c4`, `e2`) are legible, ordered-but-arbitrary identifiers — evidence (not ablated against alternatives, see caveat above) for the "index/name > raw coordinate" side of the RQ about pose encoding, though only within a single frame, not across camera viewpoints.
- **Hierarchical decomposition before menu selection matters far more than the menu presentation itself.** The ablation (0.7→0.1 Table Wiping subtask I, Table IX) shows that pre-structuring *what* the VLM should decide (subtask ID, involved objects, direction) before showing the marked menu is the dominant lever — presenting *all* options at once without that scaffolding is the worst-performing configuration tested. This is a direct data point for the branch's "hierarchical/on-demand vs. dump" question: MOKA's version of "dump" (w/o hierarchy) is not "show the full flat action space," it's "skip the high-level framing query entirely," and that specific ablation is catastrophic. It argues for keeping a cheap high-level framing step (e.g., "which surface are we inspecting" before "which view of it") even if the low-level menu itself stays flat and small.
- **Precision is not asked of the VLM.** Directly consistent with our framing doc's given ("VLMs cannot emit metric poses") — MOKA is a real-robot, published existence proof of the same constraint, with an explicit workaround (nearest-candidate snapping to a classical sampler) that is architecturally analogous to snapping a VLM-chosen view index to the nearest IK/collision-feasible view.
- **In-context examples give a large, cheap boost** (0.6→0.9+ on several subtasks with just 2 examples, TABLE I) — relevant if our benchmark harness wants a cheap ceiling-raising baseline before more complex geometry-aware prompting.

## What does NOT transfer, and why

- **MOKA never selects a camera viewpoint.** This is the single most important negative finding for this branch: despite being "the annotated action menu in its purest robotic form," the menu is always over *contact points and free-space regions within one fixed image*, never over *which view to take next*. MOKA supplies zero evidence — positive or negative — about relating one viewpoint to another, ego/allocentric framing, view adjacency, or any of the "capability floor" questions (MMSI-Bench, ViewSpatial, SpinBench) this branch is trying to resolve. It cannot be cited for RQ3 topology or RQ4 view-menu tractability beyond the generic "discrete beats continuous" point above — that generic point transfers, the viewpoint-specific claims do not exist in this paper.
- **No point-cloud, no fused-survey geometry, no glossy/sparse-depth handling.** The paper's whole geometric grounding is a single registered depth frame at chosen pixels; it has nothing to say about the branch's PLY-sparsity/hole/4cm-offset questions (the "GraspView, SPAZER, VLM3D, SeqVLM" handoff question). Flag as **no evidence**, not negative evidence — the paper simply operates in a regime (dense single-view RGBD, no fused survey) that doesn't encounter this problem.
- **No menu-size or option-similarity ablation.** K (candidate keypoints per object) is left symbolic; the 5×5 grid size is fixed and unmotivated by any sweep. The branch's "how many options, how similar may they be" question gets no quantitative answer here — only an implicit existence proof that ~K+1 points plus 25 grid cells (roughly 25–40 simultaneous options across both menus) is workable for GPT-4V on this task family. Do not over-read this as a measured tolerance; it's an untested operating point.
- **No ordering/permutation test.** Grid tile names are geometrically meaningful (chess notation) but the paper never tests whether relabeling or reordering them changes selection accuracy — relevant to the branch's ordering-effects question, but as an absence, not an answer.
- **No stopping criterion / view-count question at all** — MOKA always takes exactly one low-level decision per subtask from one image; there is no "keep looking vs. stop" decision, so Explore-until-Confident-style calibration questions are simply out of scope for this paper.
- **Grasp precision is explicitly NOT trusted to the VLM** (nearest-candidate snapping) — if our design were tempted to let the VLM's view choice directly set final camera pose without a downstream IK/collision snap-to-nearest-feasible step, this paper argues against that: even for MOKA's coarser 2D-pixel-selection task, the authors judged VLM output too imprecise to execute raw and added a classical snap-to-nearest layer. Direct architectural precedent for "selected view index → nearest reachable/orthogonal view," not "selected view index → executed pose."

## Failure modes catalogue (for Branch 2 handoff)

From Fig. 4 (qualitative, uncounted) and Appendix C-A/Fig. 8 (ablation qualitative examples):
- Grasp/function-point confusion (grasping tool upside-down)
- Wrong target/direction selection (tool pointed wrong way — wrong categorical orientation choice)
- Wrong target keypoint on a visually cluttered object (pressed laptop hinge instead of button)
- Subtask mis-ordering when hierarchy is skipped (catastrophic, not a minor error — Appendix C-A)
- Execution-layer failures independent of VLM choice (missed contact, dropped grasp) — these are downstream-planner failures, not VLM reasoning failures, and the paper's own accounting keeps the two failure classes separate (Sec. V-B "Failure breakdown").

None of these are viewpoint/spatial-relation failure modes (no wrong-face, no ego/allo confusion, no view-ID hallucination) since the task never poses that kind of question — worth stating explicitly to Branch 2 as an **absence**, not a clean bill of health for that failure class.

## Section/figure map for verification

- Sec. III: problem statement, object-centric two-phase (grasp/manipulate) formulation.
- Sec. IV-A (p.4): point-based affordance representation, definitions of all keypoints/waypoints.
- Sec. IV-B, Eq. 1–2, Fig. 2 (p.3–4): hierarchical high-/low-level prompting.
- Sec. IV-C, Fig. 1/Fig. 6 (p.5): mark-based visual prompting — GroundedSAM keypoints, 5×5 grid, chess notation.
- Sec. IV-D (p.5–6): lifting to SE(3), grasp-sampler snapping, orientation as finite-option choice.
- Sec. IV-E: in-context learning + policy distillation (Octo, 50 demos/task).
- TABLE I (p.6): main quantitative results, 4 tasks × 2 subtasks, /10 trials.
- Fig. 4 (p.6): failure taxonomy (qualitative).
- Fig. 5 (p.6): robustness to instruction/object/pose variation (qualitative).
- Appendix B-B (p.9): K boundary points + center, farthest-point sampling — K left symbolic.
- Appendix B-D (p.9): grasp-sampler snapping (DexNet-2.0-style, 30 antipodal proposals), neutral-pose re-observation between subtasks.
- Appendix C-A, TABLE IX, Fig. 8 (p.10): ablations (w/o hierarchy, w/o description, w/o CoT) — only comparative/ablated evidence in the paper.
