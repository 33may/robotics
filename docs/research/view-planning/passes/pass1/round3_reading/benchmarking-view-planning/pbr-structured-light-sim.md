# Close the Sim2real Gap via Physically-based Structured Light Synthetic Data Simulation

- **arXiv:** 2407.12449v1 [cs.CV], 17 Jul 2024 (ICRA 2024)
- **Authors:** Kaixin Bai, Lei Zhang, Zhaopeng Chen, Fang Wan, Jianwei Zhang — TAMS, Universität Hamburg / Agile Robots AG
- **Fetched from:** https://arxiv.org/html/2407.12449v1 (full HTML, section-by-section, not abstract-only)
- **Project page:** https://baikaixin-public.github.io/structured_light_3D_synthesizer/

## IMPORTANT CORRECTION TO SELECTION RATIONALE — read first

The task brief selected this paper because it supposedly "simulates the active-stereo pipeline (stereo pair plus IR speckle projector, PBR) — the D405's actual modality." **This is not what this paper does.** Checked the full method section (§III) line by line: the simulated sensor is a **single projector + single camera gray-code structured-light scanner** (sequential binary pattern projection, multi-shot temporal decoding, §III-B/C, Eq. 1–2) — the same class of device as an industrial Photoneo scanner (which is in fact the real camera used in their robot experiments, §IV-D: Photoneo L / Photoneo M). There is no stereo pair and no IR speckle projector anywhere in this paper.

The paper the brief is actually describing is its own reference **[32]: X. Zhang et al., "Close the visual domain gap by physics-grounded active stereovision depth sensor simulation," arXiv:2201.11924 (2022)** — cited here only in passing, for AI-denoiser render-speed numbers (§III-B). That is the IR-speckle/active-stereo paper matching the D405's real modality (D405 is active stereo: two IR cameras + speckle projector + block matching, single-shot). **Recommend reading arXiv:2201.11924 instead/also** if the goal is "believable dropout on glossy/dark surfaces for our exact sensing modality" — it is the correct paper for that specific rationale. This paper's noise mechanism (projector-occlusion shadow + gray-code binarization ambiguity, accumulated over ~dozens of sequential frames) is a different failure mode than stereo-correspondence dropout (textureless/specular IR return failing block matching in a single shot), so its concrete defect statistics do not transfer to D405 even though the general methodology (below) does.

## Core method (§III)

Pipeline (Fig. 1, Fig. 2): Blender Cycles ray tracing (NVIDIA OptiX GPU) → drop objects into a scene with physics (gravity, collisions) → render RGB + a Z-buffer "ground-truth depth" (perfect, pre-pattern) → separately project sequential gray-code binary patterns via a simulated spotlight-with-textured-light "projector," ray-traced per frame (20 frames/pattern, chosen by qualitative tuning, AI-denoised via OptiX to cut render time) → binarize each pixel by its temporal max/min threshold at 0.5 → decode gray code → triangulate camera-ray/projector-ray correspondence (pinhole models for both, Eq. 1–2) → **"proposed synthetic depth."**

Key design idea (the one thing worth transferring): **don't hand-parametrize a depth-noise model (Gaussian blur, dropout probability tables, etc.) — instead re-simulate the sensor's own physical decoding algorithm inside the renderer**, so that shadow zones, flying-pixel edges, and decoding ambiguity emerge from actual geometry/material/lighting interaction rather than being hand-tuned. This is stated as the reason rasterization-based engines are unsuitable (§III-A: "bypasses the shortcomings of rasterization-based methods, which neglect light path calculations").

## Ground truth strategy (§III-A, §IV-B)

Two flavors of "ground truth" are both present per scene, letting them isolate what the noise model buys:
1. **"Ground Truth Depth"** = ray-traced Z-buffer depth of the CAD mesh, no pattern/noise applied — i.e., a mesh-based, perfectly clean depth.
2. **"Proposed Synthetic Depth"** = the noisy reconstruction described above.
3. Object pose / 2D-3D bbox / instance-seg masks come directly from the physics-sim scene state (RLE encoding for synthetic, manual polygon encoding for the 100 real test images/object).

This is directly relevant to Branch-3 Q4's "exit 1" (simulate from mesh with injected scanning defects): this paper is a working instance of exactly that pattern, with a controlled A/B (clean-mesh depth vs. defect-injected depth) baked into the experimental design, not just the injected version.

## Concrete results (Table I, §V-B1, §V-C) — verified against primary HTML text, not just the abstract

Sim2real Gap is defined explicitly (Table I footnote) as **Synthetic-test AP − Real-test AP** (AP@IoU=0.50, YOLOv3 for detection, SOLOv2 for segmentation, both trained on synthetic-only data):

| Object | Input | Det AP synth/real/gap | Seg AP synth/real/gap |
|---|---|---|---|
| Metal Workpiece 1 | RGB | .624/.547/**.077** | .711/.620/**.091** |
| Metal Workpiece 1 | GT Depth (clean) | .613/.527/**.086** | .692/.557/**.135** |
| Metal Workpiece 1 | Proposed Depth (noisy) | .634/.617/**.017** | .698/.646/**.052** |
| Metal Circle WP2 | RGB | .673/.614/.059 | .646/.565/.081 |
| Metal Circle WP2 | GT Depth | .662/.633/.029 | .636/.544/.092 |
| Metal Circle WP2 | Proposed Depth | .674/.628/.046 | .634/.627/**.007** |
| Toothpaste | RGB | 1.0/.885/.115 | 1.0/.953/.047 |
| Toothpaste | GT Depth | .999/.950/.049 | 1.0/.970/.030 |
| Toothpaste | Proposed Depth | 1.0/.980/**.020** | 1.0/.979/.021 |
| YellowSaltCube | RGB | 1.0/.826/.174 | 1.0/.621/**.379** |
| YellowSaltCube | GT Depth | .996/.678/**.318** | .998/.802/.196 |
| YellowSaltCube | Proposed Depth | 1.0/.827/.173 | 1.0/.863/**.137** |
| HygieneSpray | RGB | 1.0/.947/.053 | 1.0/.963/.037 |
| HygieneSpray | GT Depth | .999/.797/.202 | 1.0/.875/.125 |
| HygieneSpray | Proposed Depth | .980/.953/**.027** | 1.0/.966/.034 |

**The headline empirical claim**: for 4 of 5 objects, Proposed (noisy) Depth has the *smallest* sim2real gap of the three input types — smaller than clean GT depth, and usually smaller than RGB. Clean GT depth is consistently the *worst or near-worst* transferer (e.g., Metal WP1 seg gap .135 GT vs .052 proposed; YellowSaltCube det gap .318 GT vs .173 proposed). This is the paper's actual finding: **training on physically-defect-injected synthetic depth transfers better than training on either clean synthetic depth or synthetic RGB.** Directly useful evidence for our Branch's premise (in-scope: "injected sensor defects... pose error, dropout") — it's not decorative, it measurably changes downstream ranking by 0.02–0.3 AP.

Other numbers:
- YOLOv7 vs. domain-randomization (Isaac Sim) baseline, Metal Workpiece 1 only (§V-B1): proposed-depth model IoU 0.648 > domain-adaptation RGB model IoU 0.628 > domain-randomization baseline IoU 0.584. (Caveat: the sentence describing this in the paper text is internally inconsistent/likely a typo — it names "domain adaptation-based RGB dataset" twice with two different IoU values, 0.628 and 0.584; the second occurrence almost certainly should read "domain randomization." Flagging as an error in the source, not a mistake in this extraction.)
- Robotic grasping (§V-C, Dex-Net 3.0 half-model pipeline, UR5e + Photoneo M): grasp success rate 95.6% → 98.0% after switching to the sim2real-trained detector; task-development time "2 weeks → 2 days" (qualitative, no CI given).
- Dataset scale (§IV-B): 1,000 synthetic scenes/object (train), 100 real scenes/object (test), single-class multi-instance bin scenes, up to 255 metal parts or 10 KIT objects per scene.

## Load-bearing assumptions

1. **Ground truth is a CAD/known mesh + physics-sim placement** — pose, bbox, and segmentation labels all come "for free" from the simulator's scene graph, not from any measurement of a real object. This only works because the objects are known CAD models (2 industrial metal parts + 3 KIT household objects), not scanned survey clouds with real defects.
2. **The noise model is entirely mechanistic/emergent** — no explicit numeric defect magnitudes (no "X mm pose error," no "Y% dropout rate") are ever reported; validation of the noise model is qualitative only (Fig. 7: point-cloud overlay of real vs. synthetic showing "similar" shadow/flying-noise patterns by eye). There is no quantitative accuracy metric for the depth simulation itself (e.g., no RMSE-vs-real-depth number) — only downstream task AP is quantified.
3. **Validation is entirely via downstream 2D task performance** (detection/segmentation AP), never via any 3D/geometric metric (no coverage, no Chamfer distance, no reconstruction error) — because the task is bin-picking perception, not surface reconstruction or view planning.
4. Real-world test data comes from a **different physical sensor family** (Photoneo industrial structured-light scanners) than the simulated one only in the sense that Photoneo is itself sequential-pattern structured light — so sim and real modality do match each other in this paper's own use case. That match is exactly what's absent for us (D405 = active stereo, not gray-code SL).

## What transfers to our setting

- **The methodological principle**: simulate defects by re-implementing the sensor's own reconstruction algorithm inside a physically-based renderer (ray trace the actual signal path — pattern/speckle projection, decoding/matching) rather than hand-authoring a parametric dropout/noise model. If we want believable dropout on glossy/dark surfaces for the D405 specifically, the right move (per this paper's logic, but using the *correct* reference for our modality) is to simulate IR-speckle projection + stereo block-matching failure physically, not to hand-tune a per-material dropout probability. This supports Branch 3/Q9's "inject sensor defects" item as a real, implementable technique, and specifically points at ref [32] (arXiv:2201.11924) as the concrete recipe to adapt.
- **The "sim2real gap = synthetic-score − real-score, per condition" evaluation pattern** (Table I design) is a clean, reusable benchmark-design idiom: report the *same* metric on a synthetic and a real held-out set for each candidate input/method, and report the gap as a first-class number, not just the real-set score. Could be adapted for view planning: e.g., report a candidate view-selection method's [orthogonality/coverage score] computed on the fused survey cloud vs. computed on a clean CAD-equivalent, and treat the delta as the quantity of interest — analogous to how Branch's Q10 (fused-vs-single-view offset: benchmark condition or subject?) could be operationalized as an explicit "gap" row rather than folded silently into one number.
- **A/B methodology of "clean GT vs. defect-injected" as two conditions in the same experiment** (their GT-Depth vs. Proposed-Depth rows) is a directly reusable ablation design for Q9 ("does ranking change under injected defects") — this paper is existence proof that method ranking (here, "which depth variant transfers best") *does* flip depending on whether defects are injected, which is exactly the kind of finding Q9 is fishing for.

## What does NOT transfer, and why

- **Sensor modality mismatch (the central issue, see correction above)**: gray-code sequential structured light ≠ D405 active stereo. Different physical failure mode → different injected-defect statistics; the concrete simulator described here (Blender spotlight-projector + 20-frame-per-pattern gray-code decode) cannot be dropped in for D405 dropout simulation without a rewrite around stereo matching.
- **Task and unit of evaluation don't match ours at all**: this is 2D object detection/instance-segmentation AP on cluttered bin-picking scenes, evaluated per-object-class over many piled instances — not per-surface-region view scoring, not single isolated primitive-shape tabletop inspection, not multi-view coverage. Nothing here measures view quality, orthogonality, coverage %, or reconstruction error vs. number of views — the paper never selects or evaluates *views* at all; camera pose is fixed (top-down bin-picking rig), so it has zero direct bearing on Q1, Q2, Q5–Q8, Q11 (no view planning happens).
- **No survey-cloud / real-defect input**: their "real" data is a clean test set captured with an industrial scanner of known-good CAD objects; there's no analog to our "fused multi-view point cloud with pose error, dropout, ~4cm fused-vs-single-view offset" as an *input* to be planned over — their defects are purely in the *simulated training data*, not in a real geometry the system must plan against. So it doesn't inform Q2 (does an analytical GT normal survive contact with a survey cloud) or Q10 (score in fused-cloud space vs execution space) directly.
- **No statistical treatment worth reusing for Q12**: no confidence intervals, no variance across trials/seeds reported anywhere; single-run AP numbers only. Not a positive example for small-N statistical rigor.
- **Residual gap on glossy/reflective objects remains large even with their method** — YellowSaltCube keeps a 0.137–0.173 AP gap after defect injection (vs. ~0.02–0.05 for the other 4 objects), attributed (§V-B1) to "shadows and sharp noise" specific to that object's glossy packaging. Negative finding worth carrying forward: **even physically-based sensor simulation has a documented residual sim2real gap specifically on glossy/reflective surfaces** — i.e., simulating "dropout on glossy/dark surfaces" (the stated reason for selecting this paper) is exactly the case this paper's own results show is hardest to close, not a solved problem. If we build a PBR sensor sim for D405, budget for this residual gap on shiny industrial parts rather than assuming full fidelity.

## Relevance to Branch-3 RQs — summary

Low direct relevance (no view planning, no NBV, no coverage/reconstruction metrics, no per-region scoring — outside this paper's scope entirely). Its value is narrow and methodological: (a) it's a concrete, checkable instance of "simulate from mesh + inject sensor defects" for Q4/Q9's defect-injection design, (b) its Sim2real-Gap-as-a-column table format is a reusable reporting idiom, (c) it is hard evidence that defect injection changes measured rankings, supporting Q9's premise. Its single biggest contribution to this branch is the **correction**: the paper actually matching the "IR speckle + active stereo, D405-like" rationale is a different arXiv ID (2201.11924, its own ref [32]) and should be read instead/in addition for that specific purpose.
