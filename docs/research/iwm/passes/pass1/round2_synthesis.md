# IWM pass 1, round 2: what the five maps say together

**Date:** 2026-10-06
**Question:** which approaches exist, and which are actually used, to compress a robot's
internal representation of the world?
**Sources:** the five maps in [`round2_map/`](round2_map/), written by five mappers.
Numbers below marked (checked) were re-read by the lead against the raw page text of the
paper on the same day. Everything else is as the mapper reported it.

**Correction, 2026-10-06 (Anton):** the structure is read directly by the IWM agent, a
vision-language model that sees the frames. Only the orchestrator is text-only. Where
this document says "text-only reader", read "the orchestrator". Work whose reader sees
images (GraphPad, Reasmory, 3D-Mem, ZoomEye) therefore transfers to the IWM agent
directly, and the notes below that discount it for showing images no longer apply.

## 1. The five families side by side

| Family | Atomic unit | Compresses by | What is lost | Who reads it | Used where |
|---|---|---|---|---|---|
| Geometric maps | Voxel with one weight, surfel, Gaussian, per-keyframe pointmap | Fusing many measurements into one cell, sparsity, multi-resolution | Which view measured it, from what angle and distance | Planners in code, never a language model | Shipped: nvblox in NVIDIA Isaac. Open source: OctoMap, Voxblox, Open3D |
| Learned representations | Feature vector per point, scene tokens, latent state | A learned vector in place of raw data | Provenance, "not looked" (latents fill in what was never seen), fine text | A trained decoder or a similarity query | Open source and research. The one production memory found (Physical Intelligence) keeps video plus text summaries, not a latent |
| Scene graphs | Object, fragment, part or place node, with edges | Naming things and dropping the rest | Everything below the named level. Merges cannot be undone | Language models, through JSON or graph queries | Open-source research stacks (Hydra, Khronos, Clio). No shipped product found |
| Language-agent memory | Fact, timestamped fact edge, typed record, object node with an id | Summarising, indexing, fetching on demand | Fine-grained cues, and stale facts that are never revoked | Language models, through a short brief plus tools | Shipped products (Letta, Zep, Mem0, Claude Code, Aider, Playwright). On robots: Thea, three real robots |
| Task-driven abstractions | Keypoint, predicate, shape primitive, inspection characteristic | Keeping only what the task names | Everything the task did not name. Thrown away after the task | A planner, for one task | Industry: QIF and STEP in metrology. Robotics: papers |

## 2. What the families agree on

**C1. Keep every sighting. Everything fused, merged or summarised is a cache over the
sightings.**
- Geometry: BundleFusion keeps every frame and re-fuses when poses change. MASt3R-SLAM,
  voxgraph and SEAM keep per-view or per-submap units and have no global fused volume.
- Graphs: Khronos keeps every object fragment and infers objects from them ("fragment
  reconciliation", checked).
- Agent memory: Zep, Mem0g and ObsMem mark old facts invalid and never delete the source.
- Learned: WorldMem and Physical Intelligence's memory keep frames plus pose, or text.
- For us: the atom is one sighting of one detail in one view. The detail node, the voxel
  and the digest are derived. This is also the only basis found for undoing a wrong merge.

**C2. Our 10 mm shift per view is ordinary drift, and the field corrects it by moving
views, not by widening merge thresholds.**
Hydra and Khronos solve a pose correction from landmarks seen more than once. Our error
is one rigid shift per view with 1.6 mm left after removal, so details seen twice can
serve as the landmarks.

**C3. How well something was observed is kept as one number, and no language model ever
sees it.** Confirmed in all five families.
- Geometry keeps one weight per element. Averaging two shifted copies raises the weight,
  so "seen often" reads as "seen well".
- Direction is kept by a few: Directional TSDF, GAVIS, SEAM (SEAM checked at abstract
  level, the other two as mapped).
- What reaches a model today: freshness and confidence (Thea, checked), labels fresh,
  stale, uncertain, contradicted (ObsMem), frontier snapshots for "not looked" (3D-Mem),
  an admissible flag and an uncovered set (FRAME).
- Viewing angle, resolution and "which views saw this" appear nowhere. This gap is ours.

**C4. The read path converges on a short pushed brief of ids plus fetch-by-id tools, and
the tools must be checked.**
- Thea (2608.11246, checked): each turn the model gets every object id with coarse
  position, confidence and freshness. Details come from tools keyed by that id, and
  action tools take the same id. Three real robots. No ablation without the brief.
- Reasmory (2606.00963, Table 8, checked): GPT-5-mini scores 58.0 with no tools, 53.2
  with spatial tools and no checker, 76.0 with the same tools behind a checker. The
  model sees rendered images there, so it is not text-only.
- At the size of one object, pasting the whole digest still competes (MIT 2510.16643,
  small graph). Fetching matters for images, raw readings and several objects.

**C5. On the write path, the gain comes from structuring the evidence before deciding.
The source is never rewritten.**
- 2606.01435, Table 1 (checked): direct answer 67.2%, extract candidates then the model
  applies the rule 76.0%, extract then code applies the rule 78.0%. The split gives
  10.8 points; code as the final executor gives 2 of them.
- TEPA (2608.07429, checked): memory that never revokes scores 0.210, below no memory at
  all (0.309). With revocation by key, 0.950.
- Where a model rewrites old entries in place, errors enter and spread (A-MEM, Mem0
  delete).
- Our world is static, so our conflicts are competing readings of one thing, not changes
  over time. The shape transfers with "better view" in place of "newer".

**C6. Compact learned latents are not what working memory is built from.**
- LL3DA on ScanQA, same language model: CIDEr 77.44 with the 3D encoder, 76.64 with its
  output zeroed (2506.05318, Table 1, checked).
- Dropping 90% of Chat-Scene's 3D tokens keeps 96.87% of the score (2507.09334, Table 1,
  checked).
- Latent states predict regions that were never observed, which destroys the difference
  between "not there" and "not looked".

**C7. Compressing by task needs the task in advance. Inspection asks its questions after
the views are taken.**
- Clio keeps 49 objects where its task-free primitives number 1883 (2404.13696, Table I,
  Cubicle, checked), given a task list at the start.
- Manipulation systems (ReKep, MOKA) build an abstraction per task and discard it.
- ViPlan (2505.13180, abstract, checked): symbols beat direct planning in Blocksworld
  (46% against 9%) and lose in household tasks (5% against 34%), where the scene is only
  partly observed. Ours is partly observed.
- So for us: compress by indexing and deriving, not by discarding.

**C8. The yardstick can be stated.**
- Every formal definition has one shape: a structure is sufficient when it answers what
  the task needs as well as the full history does. The planner's questions take the
  place of the task.
- "Smallest" cannot be proven (finding the minimal equivalent filter is NP-hard). It is
  measured as a curve: size in bytes or tokens against questions answered.
- "Sufficient context" (2411.06037, Table 1, checked) gives a test per question that
  needs no ground truth: an automatic rater agreed with human labels 93.0% of the time
  on 115 pairs. It can be run on our 100 logged questions.
- Seven candidates are written out in `round2_map/task-driven-abstractions.md`, section 4.

**C9. Relations are computed when asked. Only part-of and functional relations are
stored.**
- BBQ (2406.07113, Table II, checked): on Nr3D, metric edges lift Llama3-8B from 36.1 to
  43.8 and GPT-4o from 61.8 to 68.6. Semantic edges alone drop GPT-4o to 50.5.
- BBQ, MIT 2510.16643 and ThinkGraphs all compute spatial relations per query.

## 3. Nearest precedents

| Work | What it is to us | State |
|---|---|---|
| Thea, 2608.11246 | The read path: brief of ids, fetch by id, writes gated by an evaluator | checked |
| FRAME, 2609.14219 | Evidence memory per inspected feature: provenance, admissibility, coverage | abstract checked, record fields as mapped |
| QIF | Part, Feature, Characteristic, each with a permanent id | recalled by the mapper |
| RoboSeg, 2608.09778 | Our rig class: UR5, eye-in-hand RealSense, top 3 candidate ids with votes per voxel, 10.7 mm Chamfer | checked |
| Khronos, 2402.13817 | Fragments as the stored unit, objects inferred from them | checked |
| Reasmory, 2606.00963 | A checked query language over a 3D memory | checked |

## 4. Still not found anywhere

- A node type for text, labels or markings on an object.
- Any operation that undoes a merge, or any measured wrong-merge rate.
- View quality (angle, distance, resolution) shown to a language model.
- A controlled comparison of pushing a brief against pulling by query, for a robot.
- A structure at the scale of one object, built without training, for a text-only reader.
- A yardstick defined over natural-language questions for a robot's world representation.

## 5. Per research question

| RQ | What the literature offers |
|---|---|
| 4 Atomic block | The sighting (one detail, one view). Fused cells and merged nodes are derived |
| 5 Hierarchy | Fixed layers by type, or a cut through a tree chosen by relevance (Larsson, Clio). For one object: object, face or region, detail, characteristic |
| 6 Update without drift | Rigid per-view units with a pose correction each; invalidate, never delete; rebuild the cache from the source |
| 7 Who manages | The model proposes and names; an explicit rule orders, compares and revokes. A trained writer is the other option, excluded here |
| 8 Contract | In: posed views plus what the vision agent proposes. Out: a short brief of ids and checked queries by id |
| 9 Preserve or discard | Keep raw views outside with a pointer from every entry. Do not put raw images in context (36.6% to 26.2%, checked) |
| 10 Known and unknown | Open. Pieces exist: direction-aware evidence, variance between measurements, epistemic labels, frontier, uncovered set |
| 11 Identity | Merge rules exist everywhere, undo exists nowhere. Text read by OCR is a stronger key than appearance, and nobody uses it |
| 12 Linking kinds | One id with several indices: position, image, text |
| 13 Frame | Object frame with one pose in the world (object SLAM). Translating into the asker's view is not addressed by anyone |

## 6. Verification log

Checked by the lead against the raw page text on 2026-10-06:

| Claim | Source | Result |
|---|---|---|
| Zeroed 3D encoder, 77.44 against 76.64 | 2506.05318, Table 1 | holds |
| 90% of tokens pruned, 96.87% kept | 2507.09334, Table 1 | holds |
| 58.0, 53.2, 76.0 with a checker | 2606.00963, Table 8 | holds; mapper had left out the 76.0 |
| Evidence anchored to submaps, direction-aware | 2609.18819, abstract | holds |
| Sufficient-context rater 93.0%, F1 0.935, 115 pairs | 2411.06037, Table 1 | holds |
| 46 against 9, 5 against 34 | 2505.13180, abstract | holds |
| VoxPoser 10.0, ReKep 44.3, annotated 68.6 | 2409.01652, Table 1 | holds |
| Brief of ids, fetch by id, gated writes, three robots | 2608.11246 | holds |
| UR5, top 3 ids per voxel, 10.7 mm | 2608.09778, Table I | holds |
| 1883 primitives, 49 objects | 2404.13696, Table I | holds |
| 36.6% to 26.2%, 41,770 to 206,489 tokens | 2609.28236, Table 3 | holds |
| Part recall 83.6, 25.8 at IoU 0.10 | 2606.29786, Tables 3 and 4 | holds |
| 71.2% at 1.6K against 60.2% at 115K tokens | 2501.13956 | holds |
| 0.210, 0.309, 0.950 | 2608.07429 | holds |
| "Code decides" gains 10.8 points | 2606.01435, Table 1 | **wrong attribution, corrected**: the split gains 10.8, code as executor gains 2.0 |
| BBQ edges lift 0.52 to 0.74 on Sr3D | 2406.07113 | **not in the paper, withdrawn**: real figures are in C9 |

Not checked by the lead, quote with care: BundleFusion memory figures, wavemap and MrHash
memory figures, GPT4Scene numbers, ObsMem and Mem0 scores, the FRAME record fields, the
QIF structure, Thea's success rates (read from a figure), everything tagged RECALLED in
the maps.

## 7. Reading lists

Each map ends with a ranked list for a deep-reading round, about 50 sources in total.
