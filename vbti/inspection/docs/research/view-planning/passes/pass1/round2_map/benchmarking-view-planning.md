# Branch 2 — `benchmarking-view-planning`
## Round-2 document map: Benchmarks and evaluation methodology for view planning

**Date:** 2026-08-27
**Role:** literature mapper (judgment tier), pass 1 / round 2.
**Branch plan:** [`../round1_plan.md`](../round1_plan.md) § Branch 2
**Contract:** [`../../../../geometry-aware-view-planning.md`](../../../../geometry-aware-view-planning.md)
**Method:** abstracts and landing pages only. No deep reads. Every entry below is a
*candidate for reading*, ranked, with the extraction target named.

---

## Headline findings from the mapping sweep

Four things the sweep established before any paper is read in full. They shape
the reading order.

1. **There is no standing benchmark for our problem, but there is now one for
   the adjacent problem.** `ObjView-Bench` (2026) is the first artifact that
   treats object-centric view planning evaluation *as the research object* —
   and it independently arrived at two of our constraints: **reachable-view
   subsets filtered by IK and collision**, and **budget regime as a first-class
   variable**. It reports that these factors *change method rankings*. If that
   holds, it is the strongest external validation the contract's constraint
   stack could get, and the strongest argument that a benchmark built without
   reachability is measuring the wrong thing.

2. **The field converged on a protocol shape, and it does not fit us
   unmodified.** The modern standard is: fixed view budget → surface coverage
   ratio (final + AUC over the trajectory) → Chamfer distance to a GT mesh.
   `GenNBV` established the AUC-at-fixed-budget convention explicitly to fix a
   methodological flaw (coverage and view count are confounded). But the whole
   stack assumes (a) a ground-truth mesh, which we do not have — our input *is*
   the defective survey cloud — and (b) that reconstruction completeness is the
   goal, which for us it is not: we want *orthogonal evidence views of a named
   surface*, not maximal coverage. Adopting coverage-AUC uncritically would
   import the wrong construct.

3. **The circularity trap has a name and a literature outside our field.** Two
   independent lines are directly usable: the LLM-as-judge *self-preference /
   familiarity* result (a judge rewards outputs that look like what it would
   have produced, so optimizing against it converges to the judge's prior, not
   to quality), and the `Embodied-IQA` / `EPD` result (classical quality metrics
   correlate with downstream embodied task success at SRCC ~0.13–0.65, i.e.
   proxy metrics *do* diverge from task outcomes and the divergence is
   measurable). Together these give the branch its defensive tooling: hold out
   a differently-constructed metric, and treat judge-score-vs-held-out-metric
   *divergence* as the Goodhart alarm rather than asserting non-circularity.

4. **"Best view" ground truth has exactly three established answers, and the
   field knows the human one is noisy.** Human-labelled (Dutagaci's 68-model /
   26-subject benchmark, still the de facto standard after 15 years; inter-rater
   agreement on pairwise view preference measured at ~78.5% elsewhere — that is
   the realistic ceiling, not 100%); metric-based-against-humans (Bonaventura's
   22-measure comparison scored *against* Dutagaci); and oracle-by-construction
   (`VIN-NBV`'s Oracle-NBV, a greedy planner with GT access, used as an upper
   bound rather than as a label). For primitive shapes we have a fourth option
   the field does not need — **analytical** GT (face normal through face
   centroid, cylinder radial). That is a genuine advantage of the primitive
   restriction and should be exploited, not apologized for.

**Standing critic note carried through this whole map:** judge–benchmark
circularity. Flagged per-entry as `[CIRC]` where the paper's evaluated method
optimizes its own evaluation metric, and `[CIRC-FIX]` where the paper does
something about it.

---

## Document list

Ranked within groups. Priority is for *this branch's* reading order, not
general importance. `⚠` marks a constraint the candidate violates (per the
cross-branch constraint lens).

### A. Spine — benchmark artifacts and protocol design in view planning

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 1 | `objview-bench` | ObjView-Bench: Rethinking Difficulty and Deployment for Object-Centric View Planning — [arXiv:2605.10707](https://arxiv.org/abs/2605.10707) | paper | **The closest existing artifact to what we must build.** Separates object difficulty (omnidirectional self-occlusion, observation saturation) from *protocol-dependent* planning difficulty via a set-cover formulation; evaluates under three budget regimes (K=5, K=30, auto-stop with a 128-view cap) and under **reachable-view constraints from IK + collision filtering**; reports that budget and reachability *change method rankings and failure modes*. Also the source of the `MS@δ` map-stabilization stopping rule for planners with no native stop. Read first, cover to cover. | **high** |
| 2 | `gennbv` | GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction (CVPR 2024) — [arXiv:2402.16174](https://arxiv.org/abs/2402.16174) | paper | Established the field's current metric convention and stated the flaw it fixes: coverage ratio is confounded with view count, so unify the budget and report **AUC of the coverage curve** + final coverage + Chamfer accuracy. Isaac Gym + Houses3K/OmniObject3D benchmark setup. `[CIRC]` — the policy is RL-trained on a coverage reward and then scored on coverage; extract how (or whether) they address it. | **high** |
| 3 | `nbv-taxonomy-survey` | A structured review and taxonomy of next-best-view strategies for 3D reconstruction — ISPRS Open J. Photogrammetry & RS 17:100098 (2025) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667393225000171) | paper | Spine survey, open access, 100 papers from 150 candidates, and — critical for us — **"described implementation details or evaluation metrics" was an explicit inclusion criterion**, so its summary tables are a pre-built harvest of evaluation protocols across five strategy families. Use it as the index for which method papers' eval sections to mine. | **high** |
| 4 | `vin-nbv` | VIN-NBV: A View Introspection Network for NBV Selection for Resource-Efficient 3D Reconstruction — [arXiv:2505.06219](https://arxiv.org/abs/2505.06219) | paper | Carries an explicit **Oracle-NBV upper bound** (greedy planner with GT access) alongside Cov-NBV and the learned policy, all sampling from an identical fixed candidate set (120 views, 3 hemispherical shells). That is the cleanest template in the sweep for "how do you bound the achievable score without making the oracle the metric". Also a direct precedent for our uniform-sphere baseline being a *shell* structure. | **high** |
| 5 | `ma-scvp-longtail` | Integrating One-Shot View Planning with a Single NBV via Long-Tail Multiview Sampling (T-RO 2024) — [arXiv:2304.00910](https://arxiv.org/abs/2304.00910) | paper | One-shot set-covering planning where **the plan length *is* the stopping rule** — the nearest analogue to our "derive the view set from geometry, then execute" analytical track. Reports movement cost as the comparison axis (baselines needed 40–110% more movement for equal coverage) and explicitly criticises "oracle view counts" derived from training objects as unrealistic for unknown test objects. Extract: the movement-cost metric and the oracle-count critique. | **high** |
| 6 | `dm-osvp-plus-plus` | DM-OSVP++: One-Shot View Planning Using 3D Diffusion Models for Active RGB-Based Object Reconstruction — [arXiv:2504.11674](https://arxiv.org/abs/2504.11674) | paper | Same group as #1/#5. Extract the **matched-input fairness control** (reference view count held at 3 across all compared methods) and the reconstruction-quality-vs-movement-cost trade-off plot format. Journal-length eval section, so the protocol is spelled out. | med |
| 7 | `pb-nbv` | PB-NBV: Efficient Projection-Based NBV Planning Framework for Reconstruction of Unknown Objects — [arXiv:2501.10663](https://arxiv.org/abs/2501.10663) | paper | 55-object simulation protocol plus real-world confirmation, open-source. Candidate as a **runnable classical baseline** for our benchmark rather than only as a protocol source. Check whether the object set includes primitives. | med |
| 8 | `neu-nbv` | NeU-NBV: NBV Planning Using Uncertainty Estimation in Image-Based Neural Rendering — [arXiv:2303.01284](https://arxiv.org/abs/2303.01284) | paper | Contains the single best **confound-isolation trick** in the sweep: hold the rendering network fixed across all planners so that all measured difference is attributable to the planning strategy. Our analogue: hold the reconstruction/judging pipeline fixed and vary only the action space. Also the fixed-measurement-budget stop (9 images on DTU). | med |
| 9 | `gleam-bench` | GLEAM: Learning Generalizable Exploration Policy for Active Mapping (ICCV 2025) + GLEAM-Bench — [arXiv:2505.20294](https://arxiv.org/abs/2505.20294) · [repo](https://github.com/zjwzcx/GLEAM) | paper/repo | Benchmark-construction methodology at scale: 1,152 scenes curated from four heterogeneous sources via filtering, **geometric repair to near-watertight**, and task-oriented preprocessing; ships GT surface points and URDF indexing. Read for *how a benchmark asset pipeline is specified and released*, and for the "prior benchmarks used <100 homogeneous scenes" critique. ⚠ scene-scale indoor mapping, not tabletop objects. | med |
| 10 | `how-many-views-prv` | How Many Views Are Needed to Reconstruct an Unknown Object Using NeRF? — [arXiv:2310.00684](https://arxiv.org/abs/2310.00684) | paper | Directly attacks "how many views is enough" as an **object-dependent quantity to be predicted** (20 for a simple object, 50 for a complex one). Relevant because our benchmark must decide whether the view budget is a constant, a per-object constant, or a per-primitive constant. | med |
| 11 | `see-surface-edge-explorer` | The Surface Edge Explorer (SEE): a measurement-direct approach to NBV planning — [arXiv:2207.13684](https://arxiv.org/abs/2207.13684) | paper | Plans directly on measurements rather than a volumetric proxy — the closest existing stance to "judge in PLY space". Extract its evaluation section: how does a measurement-direct method get scored when there is no volumetric occupancy to count? | med |
| 12 | `hestia-nbv` | Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition — [arXiv:2508.01014](https://arxiv.org/abs/2508.01014) | paper | Recent, face-aware (i.e. surface-orientation-aware) NBV. Harvest the eval section only: which coverage definition does a face-aware method use, and does it report per-face rather than aggregate coverage? Per-face reporting is what our benchmark needs. | low |

### B. Reconstruction-error and coverage metric protocol precedent

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 13 | `tanks-and-temples` | Tanks and Temples benchmark (Knapitsch et al., ACM TOG 2017) — [tanksandtemples.org](https://www.tanksandtemples.org/) | docs | The canonical **accuracy/completeness → precision/recall → F-score at threshold τ** protocol, plus the online-leaderboard and time-stamped-claim conventions. Our "reconstruction error vs. number of views" curve inherits from here. Also the source of the caveat that accuracy and completeness trade off, so a single aggregate is required to prevent cherry-picking one axis. | med |
| 14 | `dtu-mvs-eval` | DTU MVS evaluation protocol (Aanæs/Jensen et al.) — reached via [IB-MVS](https://arxiv.org/abs/2111.14420) eval sections | docs | The other half of #13: distance-from-reconstruction-to-GT (accuracy) and GT-to-reconstruction (completeness), in mm, averaged into "Overall"; official evaluation script; reports medians and variances, not just means. Read for the *reporting convention* (median + variance), which we should copy given small object counts. ⚠ requires a GT point cloud we will have to manufacture. | low |

### C. Ground truth for "best view" — the three (four) strategies

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 15 | `dutagaci-best-view-benchmark` | A Benchmark for Best View Selection of 3D Objects (Dutagaci, Cheung & Godil, 3DOR 2010) — [NIST](https://www.nist.gov/publications/benchmark-best-view-selection-3d-objects) · [data](https://data.nist.gov/pdr/lps/ark:/88434/mds2-2209) | paper/repo | **The human-label answer.** 68 models × 26 subjects, released viewpoints + a "view selection error" scoring script, and — the subtle part we must copy or reject — an explicit **symmetry-set treatment** (a view is scored against the equivalence class of symmetric viewpoints, via Fourier-Mellin matching). Boxes and cylinders are maximally symmetric; any GT scheme we design *must* handle view equivalence or it will punish correct answers. Highest-value single read in group C. | **high** |
| 16 | `bonaventura-viewpoint-survey` | A Survey of Viewpoint Selection Methods for Polygonal Models (Entropy 2018, 20(5):370) — [doi](https://doi.org/10.3390/e20050370) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7512891/) | paper | **The metric-based answer, scored against the human answer.** 22 viewpoint-quality measures compared head-to-head *on the Dutagaci benchmark* — i.e. an existing worked example of exactly the "which metric best predicts the agreed-upon best view" experiment we need for RQ1. Also flags discretization sensitivity (viewpoint entropy diverges with mesh resolution; VMI/VKL do not) — a direct warning for metrics computed on a point cloud of unstable density. Cross-tag Branch 1. | **high** |
| 17 | `quoniam` | Quoniam — viewpoint selection framework implementing the Entropy-2018 measures — [github.com/limdor/quoniam](https://github.com/limdor/quoniam) | repo | Reference implementation of the 22 measures. Value here is as an **independent third-party metric implementation** to serve as a held-out benchmark metric against our own RQ1 judge — the concrete anti-circularity instrument. ⚠ mesh/polygon-based, will need adaptation to a point cloud. | med |
| 18 | `secord-viewpoint-preference` | Perceptual Models of Viewpoint Preference (Secord et al., ACM TOG 30(5) art. 109, 2011) — [ACM DL](https://dl.acm.org/doi/10.1145/2019627.2019628) | paper | The **human-study design template**: pairwise view comparison (16 models, 3,840 images, hundreds of subjects), attribute-difference-vs-preference regression, correlation clustering of attributes. If we run any human labelling of "which of these two views is better evidence", this is the protocol to copy. Finding worth carrying: *surface visibility* dominated, silhouette curvature underperformed prior expectation. | **high** |
| 19 | `which-viewpoint-shows-it-best` | Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection — [arXiv:2411.08753](https://arxiv.org/abs/2411.08753) | paper | Supplies the number the benchmark design will need to defend itself: **~78.5% inter-evaluator agreement** on pairwise view preference (10 judges, 70 test pairs), measured independently of the training signal. That is the ceiling any view-selection accuracy metric should be read against. | med |

### D. Simulation-based evaluation with synthetic GT and injected sensor defects

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 20 | `pose-noise-robustness-plant-vp` | Efficient View Planning Guided by Previous-Session Reconstruction for Repeated Plant Monitoring — [arXiv:2510.07028](https://arxiv.org/abs/2510.07028) | paper | **The template for our pose-error injection**, and the only paper found that treats it as a first-class evaluation axis. Eye-in-hand RGB-D on a manipulator; names our exact error sources (hand-eye calibration error, joint imprecision, drift); runs a pre-deployment sweep with zero-mean Gaussian pose noise at three levels starting at (σ_t=0.002 m, σ_r=0.5°) and (0.005, 1.0); tests whether lightweight ICP recovers fidelity. Our contract's "few mm pose error" sits inside this range. Read early. | **high** |
| 21 | `depthsynth` | DepthSynth: Real-Time Realistic Synthetic Data Generation from CAD Models for 2.5D Recognition — [arXiv:1702.08558](https://arxiv.org/abs/1702.08558) | paper | The foundational **defect taxonomy** for injected depth noise: axial noise (pattern density falls quadratically with distance), lateral noise at edges, **shadow noise from occluders blocking emitter or camera**, structural noise from sensor resolution/lens warp. Gives our injected-defect list a citable structure rather than ad-hoc Gaussian jitter. | **high** |
| 22 | `pbr-structured-light-sim` | Close the Sim2real Gap via Physically-based Structured Light Synthetic Data Simulation — [arXiv:2407.12449](https://arxiv.org/abs/2407.12449) | paper | Simulates the **active-stereo** pipeline (stereo pair + IR speckle projector, PBR) — i.e. exactly the D405's sensing modality, not generic structured light. This is the right fidelity level if we want simulated dropout on glossy/dark surfaces to be believable rather than stipulated. | **high** |
| 23 | `omni6dpose` | Omni6DPose: A Benchmark and Model for Universal 6D Object Pose Estimation and Tracking — [arXiv:2406.04316](https://arxiv.org/abs/2406.04316) | paper | Harvest one thing: the **ablation proving physics-based depth simulation materially closes the sim-to-real gap**, with structured-light noise most pronounced on transparent/reflective regions. That is the citation that justifies spending effort on realistic defect injection instead of clean synthetic depth — a defence the critic will ask for. | med |
| 24 | `as-scanned-sl-simulation` | As-scanned point cloud generation using structured-light simulation and ML-based coverage prediction — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0965997825001346) | paper | Industrial-metrology-side simulator that reproduces shadow and sharp noise and reports point-cloud noise matching a real structured-light camera — plus **coverage prediction**, i.e. it directly ties simulated scanning defects to a coverage metric. Nearest neighbour to "simulate our survey pipeline's defects and score coverage on the result". | med |
| 25 | `nbv-under-positioning-error` | Next-Best-View Planning for 3D Object Reconstruction under Positioning Error (Springer LNCS) — [link](https://link.springer.com/chapter/10.1007/978-3-642-25324-9_37) | paper | The classical precedent that positioning error belongs *inside* the view-planning problem, not outside it; entry point to the Scott et al. lineage on pose error effects in range sensing and view planning. Historical grounding for why our ~4 cm fused-vs-single-view offset is a planning-relevant quantity. | med |
| 26 | `motion-uncertainty-nbv` | Motion-Uncertainty-Aware NBV Planning for Moving Object Reconstruction — [arXiv:2605.17593](https://arxiv.org/abs/2605.17593) | paper | Real-world noise sweeps (σ ∈ {0.10, 0.15, 0.20} m) with **completeness-over-iterations reported as mean ± std over 4 trials per setting** — a concrete, minimal statistical reporting format for a noise sweep. Its limitations section articulates the localization/extrinsics coupling that our fused-cloud offset instantiates. ⚠ moving-object setting. | low |
| 27 | `isaac-determinism` | Deterministic replay for reproducible Isaac Sim benchmarks (NVIDIA) — [perspectives.nvidia.com](https://perspectives.nvidia.com/isaac-sim/task/faq/deterministic-replay-mechanisms-reproducible-benchmarks-1/) | docs | Practical: step-locked runtime, fixed randomization seeds, pinned assets, locked physics config. If the benchmark harness runs in the kinematic sandbox, this is the reproducibility checklist to satisfy. | low |

### E. Benchmark design methodology, statistics, and the circularity trap

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 28 | `self-preference-bias-judge` | Self-Preference Bias in LLM-as-a-Judge — [arXiv:2410.21819](https://arxiv.org/abs/2410.21819) | paper | **The branch's designated circularity reference.** Quantifies self-preference and then reframes it: the bias tracks *perplexity/familiarity*, not authorship — judges reward outputs that resemble what they would have produced, regardless of who produced them. Transposed to us: a benchmark metric built from the same geometric assumptions as the RQ1 judge will reward candidates that share those assumptions, whether or not they share code. That reframing is what makes "we used a different implementation" an insufficient defence. `[CIRC-FIX]` — extract the mitigation set (anonymization, position swap, held-out human slice). | **high** |
| 29 | `embodied-iqa` | Image Quality Assessment for Embodied AI (Embodied-IQA) — [arXiv:2505.16815](https://arxiv.org/abs/2505.16815) | paper | The empirical proof that **a plausible view-quality proxy can diverge from task success, measurably**: FR-IQA methods reach SRCC <0.65 against embodied labels while scoring ~0.9 against human perception on the same style of task; VLM-cognition results correlate with real execution at SRCC <0.5. Gives us the *form* of the anti-circularity experiment: report rank correlation between the judge score and an independently-constructed downstream outcome, and treat a low correlation as a finding rather than a failure. | **high** |
| 30 | `embodied-preference-db` | Embodied Image Quality Assessment for Robotic Intelligence (EPD) — [arXiv:2412.18774](https://arxiv.org/abs/2412.18774) | paper | Companion to #29 with a starker number: human-vs-robot preference correlation of 0.13–0.21, and the observation that embodied perception weights texture detail, edge outline and structural integrity over aesthetics. Directly relevant to whether *human* "best view" labels are even the right GT for an *inspection* evidence view. | med |
| 31 | `benchmark-lottery` | The Benchmark Lottery (Dehghani et al.) — [arXiv:2107.07002](https://arxiv.org/abs/2107.07002) | paper | Names the failure mode we are most exposed to as the party that both builds the benchmark and submits to it: "rigging the lottery" — freedom to choose tasks and criteria means researchers fit the benchmark to the method. Also the statistical-validity-under-benchmark-reuse argument. Short, high leverage, read early. | **high** |
| 32 | `measuring-what-matters` | Measuring what Matters: Construct Validity in LLM Benchmarks — [arXiv:2511.04703](https://arxiv.org/abs/2511.04703) | paper | Systematic review, 445 benchmarks, 29 reviewers, and **eight concrete recommendations** — the most actionable checklist found. Its diagnosed failure modes are ours verbatim: contested/absent definitions of the measured phenomenon (what *is* "a good evidence view"?), tasks borrowed without representativeness adjustment (borrowed object sets), and statistical testing rarely performed (only 16% used uncertainty estimates). | **high** |
| 33 | `everything-benchmark` | AI and the Everything in the Whole Wide World Benchmark (Raji et al.) — [arXiv:2111.15366](https://arxiv.org/abs/2111.15366) | paper | The construct-validity framing in its sharpest form, plus the warning against benchmarks that claim general capability. Guards against our benchmark quietly drifting from "orthogonal evidence views on primitives" to "view planning quality" in the write-up. | med |
| 34 | `benchmarking-epistemology` | The Benchmarking Epistemology: Construct Validity for Evaluating ML Models — [arXiv:2510.23191](https://arxiv.org/abs/2510.23191) | paper | Develops construct-validity conditions from psychometrics with worked case studies; the theory layer under #32. Read only if #32 leaves the "what extra assumptions license the inference from score to claim" question open. | low |
| 35 | `ycb-protocols` | Benchmarking in Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols — [arXiv:1502.03143](https://arxiv.org/abs/1502.03143) | paper | The robotics-native precedent for **object set + protocol template released together**, including an explicit protocol template in the appendix. Our deliverable has the same shape (primitive object set + evaluation protocol), and YCB's framing of *why* self-chosen objects prevent comparison is the argument for publishing ours. | **high** |
| 36 | `lbm-careful-examination` | A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation (TRI) — [arXiv:2507.05331](https://arxiv.org/abs/2507.05331) | paper | Current best-practice exemplar for rigorous robot evaluation: **blind, randomized A/B trials** against single-task baselines, paired sim and real campaigns, explicit statistical analysis. The blinding and randomization ideas transfer directly to comparing action spaces (uniform sphere vs. geometry-conditioned) without experimenter bias. | med |
| 37 | `binomial-eval-statistics` | Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators — [arXiv:2510.04354](https://arxiv.org/abs/2510.04354) | paper | Supplies the numbers that will size our experiment: at 90% observed success over 70 rollouts the 95% Clopper–Pearson interval spans 15.4 points; tightening 10 → 2 points costs ~15× more rollouts. Also prediction-powered inference to combine many simulated trials with few real ones — directly applicable given the arm is available only for calibration and demo. | **high** |
| 38 | `mirrer` | Towards Using Multiple Iterated, Reproduced, and Replicated Experiments with Robots (MIRRER) — [arXiv:2408.04736](https://arxiv.org/abs/2408.04736) | paper | Separates generalizability / reproducibility / replicability, and reports **~20% performance variation when the same experiment was reproduced across two labs**, arguing for *local relative ranking* against an in-house baseline instead of absolute numbers. That is a direct argument for making the current uniform {h,v} sphere a permanent, always-reported control rather than a one-off comparison. | med |
| 39 | `reproducible-robotics-nmi` | Towards reproducible robotics research — Nature Machine Intelligence (2025) — [nature.com](https://www.nature.com/articles/s42256-025-01114-7) | paper | Current statement of the reproducibility position for robotics; useful for the methodology chapter's framing and for the R-article / artifact-release conventions (IEEE RAM). | low |
| 40 | `reality-gap-best-practices` | The Reality Gap in Robotics: Challenges, Solutions, and Best Practices — [rpg.ifi.uzh.ch](https://rpg.ifi.uzh.ch/docs/AR25_Aljalbout.pdf) | paper | Distinguishes metrics that measure *the gap itself* from metrics that measure *transfer performance*, including sim-real correlation and open-loop replay error. Since our benchmark will be simulation-first with a kinematic sandbox, we need a stated position on which of these we can claim. | med |
| 41 | `llm-judge-bias-audit` | Your LLM Judge Has a Length Bias, a Position Bias, and a Format Bias — [tianpan.co](https://tianpan.co/blog/2026-04-27-llm-judge-bias-audit-length-position-format) | blog | Practitioner framing of the Goodhart mechanism ("prompt iteration becomes a Goodhart machine") plus three concrete audits and the versioned-judge discipline. Cheap read; useful vocabulary for the methodology section if the VLM ends up scoring anything. | low |

### F. How VLM / agentic view selection is measured (protocol extraction only)

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 42 | `view-level-evidence-benchmark` | Where Does the Answer Come From? Benchmarking View-Level Visual Evidence Identification in Multi-View MLLMs — [arXiv:2606.09644](https://arxiv.org/abs/2606.09644) | paper | **The single most transferable VLM protocol found.** Three settings — view selection scored by exact match against a golden view; Oracle QA with the gold view supplied (upper bound, computed only on answerable items); joint prediction — plus a 2×2 view-correct × answer-correct grid that separates *grounded success* from *shortcut answers*. That grid is exactly how we detect a VLM that names the right defect from the wrong view. Adopt near-verbatim. | **high** |
| 43 | `planning-with-the-views` | Planning with the Views (ViewSuite) — [arXiv:2605.29563](https://arxiv.org/abs/2605.29563) | paper | Benchmarks 13 frontier VLMs on multi-turn view planning in a 3D point-cloud environment over real ScanNet scenes, and reports the failure shape we most need to design against: models grasp single-step view-action relations but **fail to compose them across multi-turn plans, with the gap widening as viewpoint distance grows**. Extract the task construction and the distance-stratified reporting. | **high** |
| 44 | `e3vs-bench` | E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3DGS Scenes — [arXiv:2604.17969](https://arxiv.org/abs/2604.17969) · [repo](https://github.com/K0uya/E3VS-BENCH) | paper/repo | 99 3DGS scenes, 2,014 episodes, 5-DoF viewpoint control, six question types, **GT goal view used for evaluation only and never given to the agent**, VLMs benchmarked against humans. The key design principle to steal: *construct questions that are unanswerable from a single view*, so the metric cannot be satisfied without genuine view selection. Also a defensible answer to "how do you render candidate views for evaluation without a mesh". | **high** |
| 45 | `scoutvla` | ScoutVLA: UAV-Centric Active Perception via a Dual-Expert VLA for Open-World EQA — [arXiv:2606.14772](https://arxiv.org/abs/2606.14772) | paper | Multi-axis metric suite worth copying selectively: target detection rate, terminal distance to target, trajectory length, LLM-judged answer score/correctness, and a **strict success rate** combining them; uses **three independent LLM judges** to damp evaluator variance; and — most valuable — explicitly ablates *whether the active perception process actually yielded more answerable evidence at the final viewpoint*, which is the causal test for view-selection utility. ⚠ UAV, open-world. | med |
| 46 | `openeqa-protocol` | OpenEQA — open-vocabulary embodied QA and the LLM-Match scoring protocol — [open-eqa.github.io](https://open-eqa.github.io/) | docs | The upstream scoring convention (#45 and others state they follow it). Extract the LLM-Match rubric and its human-agreement calibration. Verify the landing page resolves before assigning a reader. | low |
| 47 | `viewspatial-bench` | ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in VLMs — [arXiv:2505.21500](https://arxiv.org/abs/2505.21500) | paper | Bounds what our action-space encoding may assume: VLMs do egocentric spatial reasoning reasonably and **degrade on allocentric / other-perspective reasoning**. Any "orthogonal to face 3" addressing scheme is an allocentric claim. Protocol interest: the automated 3D annotation pipeline that generates directional labels without human labelling — a cheap GT-generation pattern we could reuse for view addressing. Shared with Branch 6; this branch owns the protocol. | med |
| 48 | `vista-semantic-coverage` | VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting — [arXiv:2507.01125](https://arxiv.org/abs/2507.01125) | paper | Introduces a **viewpoint-semantic coverage metric** combining geometric view diversity with task relevance — i.e. an existing argument, with a metric attached, that raw geometric coverage is the wrong objective for a task-driven inspection. Closest published thing to "coverage of the surfaces the question is about". | med |

### G. Defect-visibility ground truth (the sanctioned exception to the 2D-AD exclusion)

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 49 | `m2ad` | Visual Anomaly Detection under Complex View-Illumination Interplay (M2AD) — [arXiv:2505.10996](https://arxiv.org/abs/2505.10996) · [repo](https://github.com/hustCYQ/M2AD) · [HF dataset](https://huggingface.co/datasets/ChengYuQi99/M2AD) | paper/repo | **The transferable defect-visibility ground truth this branch was told to look for.** 999 specimens × 12 synchronized views × 10 illuminations = 120 configurations, 119,880 images at 3648×5472, with two protocols: *Synergy* (fuse across configurations) and *Invariant* (single-image robustness). Findings that bear directly on our design: methods at 99.6% on MVTec drop to 90.0% here; **multi-configuration fusion shows diminishing or negative returns with excessive view/illumination addition** (i.e. more views is not monotonically better — a fact our view-efficiency metric must be able to express); sub-millimetre defect detection is resolution-bound (+5.8% AUROC from 256² to 512²). Read early. | **high** |
| 50 | `real-iad` | Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial AD (CVPR 2024) — [arXiv:2403.12580](https://arxiv.org/abs/2403.12580) | paper | Introduces **sample-level (part-level) metrics aggregating multiple views of one physical part** rather than judging each view independently — the aggregation question our benchmark faces when several views cover one face. Also the saturation argument (>99% AUROC on MVTec makes methods indistinguishable) as a cautionary tale about metric ceilings. | med |
| 51 | `real-iad-mvn` | Real-IAD MVN: A Multi-View Normal Vector Dataset and Benchmark — [arXiv:2605.07149](https://arxiv.org/abs/2605.07149) | paper | Five viewpoints of surface-normal maps chosen specifically to make **side-wall and occluded defects explicitly detectable** — an empirical statement of the "defects hide from the wrong viewing angle" premise that our orthogonality requirement rests on. Grounds Albert's requirement in data rather than intuition. | med |

### H. Applied inspection-planning evaluation protocols

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 52 | `cpp-measurement-uncertainty` | Coverage Path Planning for Robotic Quality Inspection with Control on Measurement Uncertainty — [arXiv:2201.04310](https://arxiv.org/abs/2201.04310) | paper | Evaluation design that mirrors ours closely: cost function over **viewpoint count + average measurement uncertainty across selected FOVs**; reachability encoded as a **binary IK-feasibility indicator plus a collision indicator per viewpoint**; and — the modelling lesson — reports a *trade-off* honestly (traversal time +18.64% while the baseline left 37.33% of holes/slots failing the uncertainty requirement), which is the antidote to single-metric comparison. Sim case + real hardware case pairing. | **high** |
| 53 | `targetted-viewpoint-sampling` | Coverage path planning with targetted viewpoint sampling for robotic free-form surface inspection — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0736584518305246) | paper | The de facto baseline method in this sub-literature (later papers benchmark against it) evaluated on two industrial sheet-metal parts. Read for what an accepted baseline looks like in applied inspection, and whether a primitive-level analogue exists. | med |
| 54 | `nbv-photography-inspection` | Take Your Best Shot: Sampling-Based NBV Planning for Autonomous Photography & Inspection — [arXiv:2403.05477](https://arxiv.org/abs/2403.05477) | paper | Its evaluation metric is **perspective distortion + target scale in frame + estimated target coverage** — coverage demoted to one of three terms. That decomposition is close to what RQ1 needs (orthogonality, framing, region coverage) and gives Branch 2 a precedent for a composite view-quality benchmark metric. `[CIRC]` — check whether the same composite is also the planner's objective. Cross-tag Branch 1. | med |
| 55 | `semantic-drl-inspection` | Semantically-driven Deep RL for Inspection Path Planning — [arXiv:2505.14443](https://arxiv.org/abs/2505.14443) | paper | Uses **semantic surface coverage** in place of uniform surface coverage, and attributes coverage drops in cluttered scenes to *fewer feasible viewpoints* — an explicit acknowledgement that feasibility filtering confounds coverage comparisons. Relevant to how we report coverage when reachability removes candidate views. | med |
| 56 | `active-vision-dataset` | A Dataset for Developing and Benchmarking Active Vision (AVD) — [arXiv:1702.08272](https://arxiv.org/abs/1702.08272) | paper | The long-standing precedent for **densely pre-captured real imagery replayed as a simulator**, so active-vision policies are evaluated on real images without a robot in the loop. Directly applicable: densely survey each benchmark object once, then evaluate any number of view-selection policies offline against the captured set. Possibly the cheapest viable design for our real-data track. | med |

### I. Object sets and CAD ground truth for a primitive-shaped benchmark

| # | Slug | Title / source | Type | Why it matters here | Prio |
|---|------|----------------|------|---------------------|------|
| 57 | `abc-cad-dataset` | ABC: A Big CAD Model Dataset for Geometric Deep Learning — [deep-geometry.github.io/abc-dataset](https://deep-geometry.github.io/abc-dataset/) | repo | One million B-rep models with **explicit parametric surface ground truth (plane / cylinder / cone / sphere / torus / freeform) and patch segmentation** — the standard source when analytic primitive labels are required. This is the natural substrate for analytical "best view" GT: known face, known normal, known extent. Cross-tag Branch 3. ⚠ CAD, not survey clouds — we would render/scan-simulate to get our input. | med |
| 58 | `fusion360-gallery` | Fusion 360 Gallery: Dataset and Environment for Programmatic CAD Reconstruction — [github.com/AutodeskAILab/Fusion360GalleryDataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset) | repo | Operation-derived B-rep face segmentation (extrude side, fillet, chamfer) on ~8.6K human-authored designs. Alternative to #57 when semantic face grouping matters more than analytic surface type — e.g. defining "one inspectable face" for objects whose geometry is a compound of primitives. | low |

---

## Questions this branch must answer

These are the reading tasks and the synthesis agenda. Grouped by what they
decide. Q1–Q6 are the ones that block benchmark design; the rest refine it.

### Blocking — the measuring stick's construction

**Q1 — What is the unit of evaluation?**
Is the benchmark item a *(object, surface region)* pair scored by "did the
system produce a good evidence view of this region", or a *(object, question)*
episode scored end-to-end? #42's view/answer 2×2 grid and #44's
unanswerable-from-one-view construction argue for the episode; #15 and #16
argue for the per-region view score. Decide, and state what the other one
would have measured.

**Q2 — What is ground truth for "the orthogonal/best view" on a primitive,
and does the analytical answer survive contact with a survey cloud?**
Collect and compare the four strategies (analytical from fitted primitive;
metric-based; human-labelled; oracle-by-construction). For the analytical one
specifically: the face normal comes from a fitted primitive on a defective
cloud, so the GT itself carries uncertainty — what is the angular error of
that GT, and is it small relative to the differences between the methods we
want to distinguish? If not, the analytical GT cannot referee. (Depends on
Branch 3's normal-error numbers; ask for them explicitly.)

**Q3 — How does the benchmark avoid being circular with the RQ1 judge?**
Enumerate the available separations and pick one to defend: (a) different
*construct* — GT from geometry, judge from image; (b) different *provenance* —
third-party metric implementation (#17) as the held-out score; (c) different
*outcome* — downstream defect-detection success rather than view quality
(#29/#30 give the correlation methodology, #49 gives the labels). State
explicitly which candidate solutions would be advantaged by our metric and
why that is acceptable. Per #28, "we reimplemented it" is not a separation;
shared assumptions are the bias, not shared code.

**Q4 — Is view-set quality even scorable without a ground-truth mesh?**
Every protocol in group A/B assumes a GT mesh for coverage and Chamfer. We
have a defective survey cloud. Three exits to evaluate: (i) simulate — CAD or
scanned mesh is GT, our defective cloud is generated from it by injected-noise
scanning (#21, #22, #24); (ii) densely pre-capture real objects and treat the
dense set as GT (#56); (iii) abandon coverage/Chamfer entirely for a
per-region view-angle metric that needs no mesh. Recommend one and cost the
other two.

**Q5 — What are the baselines, and is the uniform {h,v} sphere strong enough
to be a fair one?**
The built-in baseline is the current 12×30° × 3-vertical sphere. Establish:
what the field uses as baselines (random-sphere, uniform-sphere, greedy
volumetric, oracle — #4 has all four in one table); whether we should also
report a *denser* uniform sphere at matched view budget, since beating a
36-cell sphere with a geometry-conditioned 8-view set is a different claim
from beating a 200-cell sphere; and whether the oracle upper bound (#4) is
computable in our setting.

**Q6 — Which view budget regimes do we report, and how is stopping handled?**
#1 reports small-fixed / large-fixed / auto-stop with a hard cap; #2 argues
budget must be unified because coverage and view count are confounded; #5 has
plan length as the stopping rule; #1 supplies `MS@δ` as a generic stop for
planners with none. Choose our regimes and, critically, decide whether the
VLM's own stopping decision is part of what we measure or is held fixed.

### Protocol detail

**Q7 — What metric family expresses "the evidence view is orthogonal to the
inspected surface" as a score over a whole run?**
Coverage-AUC (#2) measures completeness over a budget; it cannot express
angular quality. Candidates to compare: mean/worst incidence angle over
inspected regions; fraction of regions with an evidence view under an angular
threshold; the composite of #54 (distortion + scale + coverage). Which admits
a per-primitive analytical optimum, so that the score has a known ceiling?

**Q8 — How do we handle view equivalence under symmetry?**
Boxes and cylinders are highly symmetric; a cylinder's "correct" orbit view is
a continuum, and a cube's six faces are interchangeable under relabelling.
#15 solved this with symmetry sets and image-matching-based equivalence.
What is our equivalence relation, and does the addressing scheme from Branch 5
respect it?

**Q9 — What sensor defects do we inject, at what magnitudes, and what is the
evidence they matter?**
Build the injected-defect list from #21's taxonomy (axial, lateral, shadow,
structural) plus our three measured givens (mm-level pose error, glossy/dark
dropout, ~4 cm fused-vs-single-view offset). Use #20's sweep design (σ_t,
σ_r at three levels) as the template. Then answer the critic's question: does
any candidate method's *ranking* change under injected defects, or only its
absolute score? Ranking changes are the finding; score drops are not.

**Q10 — Is the ~4 cm fused-vs-single-view offset a benchmark condition or a
benchmark subject?**
If the planner reasons in fused-cloud coordinates but the camera reports
single-view depth, a "perfect" orthogonal view is off by that offset in
practice. Does the benchmark score plans in fused-cloud space (offset
invisible) or in execution space (offset penalised)? This is the highest-risk
unexamined assumption in the whole evaluation design.

**Q11 — What object set, and how does it stratify difficulty?**
#1 stratifies by self-occlusion and observation saturation; #57 gives analytic
primitive labels; #35 gives the release convention. We need: pure primitives
(box, cylinder, sphere, cone), compounds of 2–3 primitives, and at least one
near-primitive object where fitting is ambiguous — the last one is where the
uniform-sphere fallback should trigger, and the benchmark must be able to
detect that it correctly did.

**Q12 — What statistical treatment, given how few objects and trials we will
have?**
From #37: Clopper–Pearson / Wilson intervals for any success-rate metric, and
the rollout counts required to resolve the effect sizes we expect. From #38:
local relative ranking against a permanent in-house baseline rather than
absolute numbers. From #14: report medians and variances, not means alone.
From #32: 84% of surveyed benchmarks skipped statistical testing — do not be
one of them. Concretely: what effect size do we need to detect, and how many
objects × trials does that require?

**Q13 — What does a negative result look like, and is it reportable?**
State in advance the outcome under which the geometry-conditioned action space
loses to the uniform sphere, and commit to reporting it. Pre-registering the
failure condition is the cheapest available defence against #31's "rigging the
lottery".

### VLM-layer protocol

**Q14 — How do we separate "the action space is better" from "the VLM used it
better"?**
The confound is fatal if unaddressed: a geometry-conditioned space with 8
well-named views may win because it is smaller and more legible, not because
its views are more orthogonal. Required: an oracle-selector condition (best
view chosen by the metric, VLM removed) run over both action spaces, so
action-space quality and agent competence are measured separately. #8's
fixed-renderer trick is the pattern.

**Q15 — How do we detect grounded success versus a lucky answer?**
Adopt #42's 2×2 grid (view correct × answer correct). Wrong-view/right-answer
is the case that inflates every end-to-end inspection metric and is invisible
without view-level labels.

**Q16 — Which VLM spatial failure modes become explicit benchmark test cases?**
From #43: multi-turn composition failure, degrading with viewpoint distance.
From #47: allocentric/perspective-taking failure. From Branch 6's mandate:
wrong-face selection, left/right confusion, count sensitivity. Each should
become a named test case with its own reported number, not be buried in an
aggregate.

**Q17 — When the VLM or an LLM does the scoring, what are the controls?**
From #45: multiple independent judges to damp variance. From #28/#41:
position swapping, anonymization, versioned judge, and a held-out
human-labelled slice checked periodically for divergence from judge scores.

### Cross-branch and scope

**Q18 — Which evaluation sections from the other branches' candidates get
harvested here?**
Branch 2 aggregates benchmark tags per the cross-branch rules. Establish the
intake format now (metric / dataset / GT source / baselines / does it survive
our constraints) so the other five branches can fill it in as they read.

**Q19 — Does the benchmark measure view *quality* or inspection *outcome*, and
can it do both without one contaminating the other?**
#29/#30 show the two can diverge sharply. If both are reported, the
relationship between them is itself a finding — and it is the empirical
version of the circularity check in Q3.

**Q20 — What do we release, and does releasing it change what we should build?**
#35 and #9 both argue the object set + protocol + assets are the deliverable,
not the numbers. If the benchmark is intended to outlive this project, that
constrains the input format (survey clouds must be shipped, not just objects)
and the harness (deterministic, seeded — #27).

---

## Search trail

Executed 2026-08-27. Engine: WebSearch (US), plus WebFetch on arXiv landing
pages for verification. Web-search budget for the session was exhausted at the
end of this sweep; three intended queries were not run (noted below).

**Queries run**
1. `next best view planning benchmark evaluation metrics comparison study`
2. `arXiv benchmark active 3D reconstruction next-best-view evaluation protocol coverage surface`
3. `GLEAM benchmark active mapping exploration 3D scenes evaluation`
4. `"view planning" survey taxonomy 2025 ISPRS next best view 3D reconstruction structured review`
5. `robot inspection viewpoint planning evaluation metrics coverage industrial surface benchmark comparison`
6. `Dutagaci benchmark best view selection 3D objects human generated ground truth`
7. `Sicong Pan one-shot view planning "how many views" object reconstruction evaluation benchmark Bonn`
8. `Bonaventura survey viewpoint selection methods polygonal models Entropy 2018 comparison viewpoint quality measures`
9. `realistic depth sensor noise model simulation RealSense structured light synthetic point cloud benchmark`
10. `LLM-as-a-judge self-preference bias circularity evaluation metric optimizing the judge Goodhart`
11. `VLM active perception next best view evaluation protocol success metric 2025 2026 embodied benchmark`
12. `Real-IAD multi-view industrial anomaly detection benchmark viewpoint defect visibility per-view`
13. `E3VS-Bench viewpoint-dependent active perception 3D Gaussian Splatting benchmark arXiv`
14. `"Tanks and Temples" OR "DTU" multi-view stereo benchmark accuracy completeness F-score evaluation methodology`
15. `good experimental methodology robotics benchmarking reproducibility statistics guidelines IEEE`
16. `evaluating real-world robot manipulation policies statistical significance number of trials confidence intervals`
17. `simulation benchmark environment view planning object reconstruction open source Isaac Sim virtual scanner reproducible`
18. `Secord perceptual model of viewpoint preference SIGGRAPH viewpoint goodness human study regression`
19. `active object recognition benchmark dataset viewpoint selection evaluation GERMS "Active Vision Dataset"`
20. `defect detectability viewing angle illumination inspection planning visibility ground truth dataset scratch specular`
21. `next best view stopping criterion view budget termination evaluation how many views enough`
22. `"view-illumination" anomaly detection benchmark arXiv large-scale view illumination interplay M2AD`
23. `photogrammetric network design simulation accuracy prediction evaluation camera placement optimization metrology`
24. `"benchmark lottery" machine learning benchmark design pitfalls construct validity evaluation`
25. `construct validity measurement machine learning evaluation "proxy metric" pitfalls benchmarks NLP survey`
26. `YCB object and model set benchmarking manipulation protocols Calli benchmark design guidelines`
27. `ABC dataset CAD models Fusion 360 Gallery segmentation primitive surfaces benchmark ground truth`
28. `next best view robustness localization pose noise evaluation sensitivity ablation simulation real gap view planning`
29. `VLM choose best viewpoint evaluation oracle view agreement "view selection" language model quantitative accuracy`
30. `Embodied-IQA image quality metrics predict embodied task success correlation PSNR SSIM poor`
31. `task-driven view planning evaluation critique coverage not sufficient metric semantic utility inspection`

**Landing pages fetched for verification:** arXiv 2605.10707 (ObjView-Bench),
2606.09644 (view-level evidence), 2605.29563 (Planning with the Views).

**Not run — budget exhausted; carry to the next sweep**
- `VDI/VDE 2634` / `ISO 10360-13` acceptance testing of optical 3D measuring
  systems — the metrology-standards route to artefact-based ground truth and
  instrument verification. Potentially the most rigorous available answer to
  Q2 and Q4, and entirely unexplored.
- Photogrammetric **first/second/third-order network design** and criterion
  matrices — surfaced in query 23 as the classical framework for *predicting*
  accuracy from a camera configuration, i.e. an analytical benchmark score
  that needs no ground-truth reconstruction at all. Search terms noted by that
  result: "Fraser network design", "criterion matrix",
  "simulation-based precision estimation".
- A dedicated look for **negative-result and pre-registration practice in
  robotics venues** — needed to make Q13 more than an intention.

**Coverage assessment.** Groups A, C, E and F are well covered. Group D is
adequate but thin on the specific question of *pose error inside the planning
loop* (only #20 and #25). Group G reached its sanctioned boundary. Group I is
deliberately shallow — object-set selection is downstream of the Q4 decision
and should be revisited once that is made. The largest known gap is the
metrology-standards line above.
