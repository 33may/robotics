# Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V

**arXiv:2310.11441** (v1: 2023-10-17, v2: 2023-11-06). Yang, Zhang, Li, Zou, Li, Gao (Microsoft). Source used: `arxiv.org/html/2310.11441v2` (full HTML), cross-checked table contents against a second targeted fetch.

Branch: Presenting spatial action spaces to LLM/VLM agents. Selected as the founding technique of the "seeing-orchestrator" alternative: overlay alphanumeric marks on an image so the model can refer to regions by label instead of emitting coordinates.

---

## 1. Core method

**Problem framing (§2.1):** GPT-4V (as of Oct 2023) is strong at open-vocabulary recognition but weak at *fine-grained visual grounding* — asked to output pixel coordinates or boxes for objects, it fails. SoM's fix is not to make the model better at coordinates; it is to remove coordinates from the interface entirely.

**Pipeline (§2.2–2.4):**
1. **Image Partition (§2.2):** an off-the-shelf segmentation model (MaskDINO / SEEM / SAM / Semantic-SAM, interchangeable, at various granularities) partitions the image into a set of regions/masks. SoM does not train a new segmenter — it consumes whatever partition is handed to it.
2. **Set-of-Mark Generation (§2.3):** each region is overlaid with a mark — alphanumeric ID, mask outline, box, and/or shaded mask, chosen to be maximally legible and non-conflicting with image content ("given an arithmetic image full of numbers, avoid numeric marks").
   - **Mark placement algorithm (Fig. 5):** marks are *not* placed at region centroids (this causes overlaps/conflicts on crowded or concave regions). Instead: sort masks ascending by area → for each mask, exclude pixels already covered by previously-placed marks → distance-transform to find the point maximally far from the region's boundary → place mark there; very small/thin regions get an offset placement just outside the region. This is a concrete, reusable engineering detail, not a hand-wave.
3. **Interleaved Prompt (§2.4):** two prompt modes — "plain text" (ask a normal question; GPT-4V spontaneously grounds its answer in marks) and "interleaved" (the question itself references marks, e.g. "Which can be toasted, 1 or 2?"). Fig. 6 gives exact example dialogue.

The output is a set of `⟨mark, region⟩` pairs; the model's text output referencing a mark ID can be mechanically mapped back to a pixel region — i.e., the mark is the API between free-form language and 2D spatial location.

---

## 2. Concrete results (numbers, with citations)

All results are **zero-shot, GPT-4V, no fine-tuning**, on **subsampled** validation sets (Table 1: "we select a small subset of images or videos... constrained by GPT-4V interface [quota]"). Table 1 subset sizes: COCO 100 img/567 instances, ADE20K 100/488, Flickr30K 100/274, RefCOCO(g) 100/177, DAVIS 71/157.

**Table 2 (main results), the load-bearing number for this branch:**
- RefCOCOg Referring Expression Comprehension (ACC@0.5): **GPT-4V raw coordinate prediction = 25.7% → GPT-4V+SoM = 86.4%.** A 60.7-point jump from replacing "emit a bounding box/coordinates" with "emit a mark ID."
- RefCOCOg Referring Expression *Segmentation* (mIoU): GPT-4V+SoM = 75.6, vs. PolyFormer (fully-finetuned, full dataset) 67.2, SEEM* 65.7, X-Decoder* 64.6. SoM (zero-shot, subsampled) beats the finetuned SOTA baseline on this cut — this is the paper's headline claim.
- Flickr30K Phrase Grounding (R@1): GPT-4V+SoM = 89.2, vs. GDINO 90.5 (finetuned), GLIPv2* 87.7. Roughly on par with a finetuned detector.
- COCO generic segmentation (precision): GPT-4V+SoM 75.7 vs. MaskDINO 80.7 (SoM below the segmenter it's built on).
- ADE20K open-vocab segmentation (precision): GPT-4V+SoM 63.4 vs. OpenSeeD zero-shot 23.4 (large win here).
- DAVIS2017 video object segmentation (J&F): GPT-4V+SoM 78.8 vs. SegGPT 75.6.
- (asterisked numbers in Table 2 are the *original papers'* full-validation-set numbers, not evaluated by this paper on the same subsample — mixed-condition comparison, flagged below.)

**Table 3 — mark-type ablation (Flickr30K R@1), the paper's only true within-method ablation of encoding choice:**
- Number & Mask: **84.4**
- Number & Mask & Box: **89.2**
Adding a redundant visual cue (box outline on top of numeric ID + mask) bought +4.8 R@1. No further breakdown (e.g., Number alone, Box alone) is reported — so we can't isolate which single cue matters most, only that *stacking* cues helps.

**Table 4 — mask-source ablation (RefCOCOg mIoU), i.e. how much the *partition* (not the marking) bottlenecks grounding:**
- MaskDINO-predicted masks: 75.6
- Ground-truth masks: **90.1**
- Gap: 14.5 mIoU (§4.3, "Golden mark location": "using ground-truth masks in our SoM improves the RefCOCOg performance by 14.5 mIoU"). The marking scheme itself is not the ceiling — segmentation quality is.

**Qualitative-only findings (§4.4, no numbers):**
- Fig. 8: failure cases — (a) some benchmark "ground truth" is itself wrong (SoM's answer was arguably more correct); (b) mark-placement failures, e.g. asked "which one is the person," the model selected a backpack's mark because the mark for the person happened to sit near the backpack — "centered marks are not always the best," i.e. the model's attention doesn't reliably track to geometric centroids even with the anti-overlap placement algorithm.
- Fig. 9: cross-image reference — the model can align marks *across two separate images* in one response ("the common objects... In the first image, elements 8, 17, 19; in the second image, element a") — relevant as an existence proof that mark-ID reasoning composes across frames, not just within one.
- Fig. 15: simulated indoor navigation over a floor-plan image with numbered rooms — multi-turn dialogue chaining marks ("go to 10 for food, then 14, then 13, then 19"). This is the paper's closest analogue to a "topological navigation over marks" demo, but it is a 2D floor-plan photo with pre-drawn room labels, not a 3D scene, not camera viewpoints, and not evaluated quantitatively — purely illustrative.
- Model-specificity: "not all LMMs have the ability to 'speak out' about the marks" — LLaVA-1.5 and MiniGPT-v2 "can hardly interpret the marks." This is stated qualitatively, not benchmarked with a number.
- §6 Discussion offers three *unverified* hypotheses for why SoM works on GPT-4V specifically: model/data scale, training data containing labeled figures/charts, and admittedly "mysterious" unknown factors — the authors explicitly flag these as "not grounded on the facts but on some empirical studies."

---

## 3. Load-bearing assumptions (what has to be true for these numbers to hold)

1. **A decent automatic partition exists.** SoM assumes a segmentation model can propose the regions worth marking. Table 4's 14.5-mIoU gap between predicted and GT masks shows this assumption is not free — bad partitions directly cap grounding accuracy.
2. **The target model has an emergent, GPT-4V-specific ability to "read" marks as symbols.** This is demonstrated by omission, not ablated: open-source LMMs of the same era "can hardly interpret marks." No mechanism is given (§6 admits this). The entire result is conditioned on model choice in a way the paper cannot explain, only observe.
3. **Marks don't collide with pre-existing image content.** The paper's own mitigation ("avoid numeric marks on an arithmetic image") is a manual, image-dependent judgment call, not an automated safeguard — i.e., OCR/mark confusion is a known live failure mode that the paper works around rather than solves.
4. **Evaluation is small-N and quota-limited** (100 images/task, one video benchmark at 71 clips) with **no variance/error bars, no confidence intervals, no repeated runs** reported anywhere in the tables. Table 2's comparison also mixes evaluation conditions (own subsample vs. asterisked full-dataset numbers from other papers) — not apples-to-apples.
5. **The task is region reference in a single already-rendered 2D photo**, not selection among discrete *viewpoints*, not 3D spatial relation, not camera pose. SoM never asks the model to reason about where a region *is in 3D* or how it relates to a *different camera position* — it asks "which visible 2D area does this word refer to."

---

## 4. What transfers to our setting

- **The core interface pattern is directly reusable for our discrete view action space**: instead of asking the VLM to emit a pose or index number from memory, overlay a label on each candidate (view thumbnail / render / marker) and let the model answer with a label. This is exactly the framing doc's constraint ("VLMs cannot emit metric poses... selection stays discrete") — SoM is the concrete mechanism, and the 25.7%→86.4% jump (Table 2, RefCOCOg REC) is the strongest evidence in the literature that *replacing coordinate-emission with discrete-symbol-selection* is where the accuracy comes from, at least for GPT-4V-class models on 2D grounding.
- **The mark-placement algorithm (Fig. 5, §2.3)** — avoid centroid placement on crowded/concave regions, use distance-transform-from-boundary placement with collision avoidance against already-placed marks — is a reusable engineering recipe if we ever overlay markers on a rendered scene (e.g., candidate camera positions drawn on a top-down render, or face labels drawn on a rendered mesh). It's built for 2D segmentation regions, so the transfer to markers hovering in 3D space over a point-cloud render is analogical, not literal — would need re-deriving for our geometry.
- **Table 3's "stack redundant cues" ablation (Number+Mask+Box beats Number+Mask alone)** supports designing our view-menu encoding with multiple redundant cues (e.g., label ID + face-name text + outline) rather than a bare integer, if we go the visual-menu route.
- **The explicit image-dependent mark-type caution** ("avoid numeric marks on an image full of numbers") is a direct transfer warning for our inspection domain: inspected parts can carry serial numbers, gauges, or printed text, so numeric view-marks risk exactly the OCR-confusion failure mode SoM flags — Branch 2 should test this as a named case.
- **Model-specificity finding** ("not all LMMs can speak marks") is a load-bearing caveat we must re-verify on whatever model we deploy (framing doc doesn't name the model) — SoM's headline numbers do not transfer to arbitrary VLMs without re-validation.

## 5. What does NOT transfer, and why (negative findings, first-class)

- **No viewpoint/camera-pose reasoning is tested anywhere in this paper.** SoM operates entirely within one already-captured 2D photograph; it never asks the model to relate one viewpoint to another, judge orthogonality, or reason about "next-best-view." It answers **none** of this branch's viewpoint-relation questions (MMSI-Bench-style camera-camera/camera-object, ViewSpatial allocentric, SpinBench rotation) directly — it is evidence about *labeling interfaces*, not about *spatial reasoning capability*. Citing SoM as evidence that "relating viewpoints is above chance" would be a misuse of the paper.
- **No chicken-and-egg viewpoint-selection problem is addressed.** The image being marked is a given, single photograph — nobody in this pipeline chooses *which view to render*; that problem is entirely orthogonal to SoM's contribution. Zero transfer to Branch's "who picks the render" question (SeeGround / 3D-Mem / Manipulate-Anything territory, not this paper).
- **No 3D geometry, point clouds, or sparse/defective data appear anywhere.** All inputs are clean, complete, colored 2D photographs from standard vision datasets (COCO, ADE20K, Flickr30K, RefCOCO, DAVIS). Nothing here speaks to marks-on-a-defective-point-cloud-render, holes, colorless geometry, or the ~4 cm fused-vs-single-view offset our setting has. Zero evidence, positive or negative, for that question.
- **No option-count or option-order sensitivity analysis.** Confirmed absent on a second, targeted fetch: "sensitivity analysis is absent." The mark-allocation algorithm processes masks "in ascending order by area," which is an *implementation* detail of where pixels get labeled, not a *tested* claim about whether numbering order affects the model's answer. This branch's ordering/token-bias RQ gets nothing from this paper — must be sourced elsewhere and is a genuine gap SoM leaves open.
- **No quantified mark-density / crowding degradation curve.** Fig. 5's motivation ("some conflicts caused by putting all marks in the center... images with dense objects... concave regions") is presented as the *reason the placement algorithm exists*, not as a measured curve of accuracy vs. mark count. We cannot extract a "how many options before degradation" number from this paper — it only shows the failure mode existed before their fix, not how it scales.
- **Ablations are shallow and unreplicated.** Only two within-method ablations exist (Table 3: 2 conditions; Table 4: 2 conditions), both single point-estimates with no variance reported, on 100-image subsamples. This paper cannot support any claim stronger than "in this one small sample, stacking marks and using GT masks each helped" — it should be flagged, not cited as strong evidence, per the branch's evidence-discipline requirement.
- **All results are 2023-era GPT-4V.** No claim here should be assumed to hold for current (2026) VLMs without re-verification; the paper itself attributes its effect to unexplained, possibly model-specific factors (§6).

---

## 6. Evidence-discipline entry (for Branch's ablation table, Q14)

| Comparison | Ablated within-paper? | Delta | Variance reported? | N |
|---|---|---|---|---|
| Coordinate output vs. SoM mark-ID output (RefCOCOg REC, Table 2) | Yes (same model, same benchmark, two prompting modes) | +60.7 pts (25.7→86.4) | No | 100 img / 177 instances |
| Mark type: Number+Mask vs. Number+Mask+Box (Flickr30K, Table 3) | Yes | +4.8 R@1 (84.4→89.2) | No | 100 img / 274 instances |
| Partition source: predicted (MaskDINO) vs. GT masks (RefCOCOg, Table 4) | Yes | +14.5 mIoU (75.6→90.1) | No | 100 img / 177 instances |
| SoM vs. other models' finetuned SOTA (Table 2, most rows) | No — cross-paper, cross-condition comparison (some asterisked as different eval protocol) | varies | No | mixed / not comparable |
| GPT-4V vs. open-source LMMs' ability to use marks | No — qualitative claim only ("can hardly interpret marks") | not quantified | No | not stated |

**Flag:** no confidence intervals, no repeated-run variance, and small quota-limited samples throughout. Treat all numbers as point estimates, not statistically robust claims.

---

## 7. Failure modes catalog (handoff to Branch 2)

From Fig. 8 and §4.4, named for benchmark test-case use:
- **Mark-attention mislocalization**: model selects a *nearby* mark rather than the *correct* mark when marks are placed off-centroid or regions are adjacent/overlapping (backpack mark chosen instead of person mark).
- **Crowded/concave-region mark conflicts**: naive centroid placement causes ambiguous or overlapping marks; not fully solved even by the proposed placement algorithm (only mitigated).
- **Content/mark symbol collision (OCR confusion)**: numeric/alphabetic marks conflict with pre-existing numbers/letters in the image content; paper's only mitigation is manual mark-type selection per image, not automatic detection.
- **Ground-truth-label error masking as model error**: some benchmark "failures" are actually incorrect human annotations (Fig. 8) — a benchmark-hygiene warning for whatever view-selection benchmark Branch 2 builds: manually verify "wrong" answers before scoring them wrong.
- **Model-specific non-transfer**: identical mark-based prompting fails to elicit grounding in some open-source LMMs entirely (not a graded degradation — a categorical capability gap per the paper's qualitative claim).

---

## 8. Answers to / effect on branch questions

- **Q4 (pose encoding: integers/angles vs indices vs names):** Best available evidence in this paper is indices/discrete-symbol beating raw coordinate emission by a huge margin (Table 2, +60.7 pts) — but this is 2D region-reference, not camera-pose encoding, so it supports the *general principle* "give the model a symbol to point at, not a number to compute," not a specific verdict on angle-vs-index for viewpoints. Cite carefully: transfers the *interface philosophy*, not the *numeric result*.
- **Q5 (text vs image encoding of identical info):** SoM is itself a hybrid (visual overlay read back as a text-token symbol) that beats a pure-numeric/coordinate approach — a data point for "encode spatial choice as a labeled visual affordance, not as a number," but doesn't cleanly separate from Q4's answer since SoM conflates "no coordinates" with "visual marks" in one design.
- **Q9 (option count / distinctiveness):** No usable quantitative data — flagged above as absent. Do not cite SoM for a degradation curve.
- **Q11 (option ordering):** No usable data — confirmed absent, a genuine literature gap.
- **Q14 (evidence discipline table):** Directly served — table in §6 of these notes is ready to merge into the Branch 2 evidence table.
- **Q16 (failure-mode catalog):** Directly served — §7 above.
- **Handoff to Branch 5 / RQ4 tractability:** No capability limit found here that forbids a view-adjacency-graph structure — but also no positive evidence it would work for *viewpoints* specifically, since SoM never tests spatial/viewpoint relations. Silence, not permission.
- **Architecture verdict (blind-orchestrator vs seeing):** SoM is founding evidence *for* the seeing option existing as a viable, implementable pattern (mark → label → select) — but it says nothing about whether seeing helps *view selection* specifically, since the paper's whole task presupposes the image (viewpoint) is already fixed. It cannot adjudicate the orchestrator-worker vs. single-agent question for our task; it only proves the labeling mechanism is technically available if we do go with a seeing orchestrator.

---

## 9. Bottom line

SoM is the right technique to cite for **"how do you let a VLM select among discrete, already-rendered options without asking it to emit coordinates"** — and its 25.7%→86.4% RefCOCOg jump (Table 2) is the paper's one number worth quoting directly. But it is *2D single-image region reference*, evaluated at small, unreplicated scale, on a now-superseded model generation, with **zero** engagement with viewpoints, 3D geometry, option ordering, option count, or point-cloud rendering — all central to this branch's other questions. Use it to justify the *mechanism* (marks + discrete symbol answers), not as evidence for spatial/viewpoint reasoning capability, which must come from the MMSI-Bench/ViewSpatial/SpinBench-class papers instead.
