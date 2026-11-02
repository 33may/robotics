# IWM pass 1, round 2: branch 2, learned representations

**Date:** 2026-10-06
**Branch:** `learned-representations` (a learned vector replaces raw data)
**Plan:** [`../round1_plan.md`](../round1_plan.md)

Tags: **READ** = read in the body today. **ABSTRACT** = abstract, landing page or README
today. **RECALLED** = my memory, not checked today. **SNIPPET** = only a search-result
snippet, page not opened.

## 1. Headline findings

**H1. The language-feature field is the one learned representation that has reached
common open-source stacks, and it can now be built without training.**
LERF and Feature Splatting ship as nerfstudio methods (nerfstudio docs page seen in search, SNIPPET).
Training-free lifting is now a family of its own: LUDVIG (ICCV 2025, ABSTRACT), the
superpoint-graph method 2504.13153 (ABSTRACT, "over 30x faster"), NormLift 2609.18898
(ABSTRACT), CoSAG 2607.10237 (ABSTRACT, "closed-form transmittance-weighted lift",
sub-megabyte, 37 to 76x smaller than LangSplatV2). All need only pretrained 2D models.

**H2. Learned 3D tokens fed to language models carry less than they seem to.**
LL3DA on ScanQA: CIDEr 77.44 with the 3D encoder, 76.64 with its output zeroed
(2506.05318, Table 1, READ). Fast3D prunes 90% of Chat-Scene's tokens and keeps 96.87%
of the score (2507.09334, Table 1, READ). "Revisiting 3D LLM Benchmarks" (ACL Findings
2025, ABSTRACT) shows 2D VLMs pass 3D benchmarks from rendered images. The benchmarks do
not test the fine, local recall we need, so their scores say little about it.

**H3. A frozen language model gains little from 3D-derived visual prompts alone.**
GPT4Scene: frozen GPT-4o on SQA3D goes from 40.3 to 42.8 EM-1 with BEV image plus
object-ID markers; a fine-tuned 7B reaches 60.7 (2501.01428, READ, numbers as reported
by the fetch). Small models (2B to 8B) barely gain zero-shot. And these readers still see
images; nothing in this family is designed for a text-only reader.

**H4. Memory that works in world models and VLAs keeps raw frames plus pose, or text,
not a compact latent.** WorldMem stores frame latents with pose and time and retrieves by
field-of-view overlap (2504.12369, Sec 3.3, READ); Context-as-Memory retrieves frames by
FOV overlap (2506.03141, ABSTRACT). Echo-Memory: "compactness is not a free substitute for
capacity" (2606.09803, ABSTRACT). Physical Intelligence's MEM keeps 54 s of video and
writes long-horizon memory as LLM-written text summaries, up to 15 min (2603.03596,
Sec III, READ). This is the closest production-lab evidence and it chose text.

**H5. Learned latent states hallucinate what was not seen, which breaks "absent versus
not looked".** CUT3R's 768 x 768 state is queried with a raymap to predict "unobserved
regions" (2501.12387, Sec 3.2, 3.4, READ). Genie 3: memory about one minute, legible text
"often only" when given in the prompt (DeepMind blog, READ). No approach in this family
stores an explicit "never observed" flag.

**H6. Per-element reliability and provenance appear only as side products, never exposed
to a reader.** OpenMask3D picks top-5 views per instance by visibility score (Sec 3.2.1,
READ): an implicit provenance and quality signal, then averaged away. NormLift derives a
per-Gaussian reliability from feature norm, "calibrated by effective multi-view support"
(ABSTRACT), used for voting, not output.

## 2. Approach cards

### A. Distilled language-feature fields (LERF, LangSplat, Feature 3DGS, F3RM, GARField)
- Atomic unit: a point in a NeRF or a Gaussian, carrying a feature vector (LERF 512-d
  CLIP, Sec 3.6, READ; LangSplat compresses 512-d to 3-d with a per-scene autoencoder,
  Sec 3.3, READ).
- Kept: a dense, view-consistent semantic embedding per location.
- Discarded: the source images and which views contributed; anything CLIP does not encode
  (negation, spatial words: LERF Sec 5, READ).
- Hierarchy: LERF 7-level scale pyramid (Sec 3.2, READ); LangSplat SAM whole, part,
  subpart (Sec 3.2, READ); GARField scale-conditioned affinity tree (ABSTRACT).
- Update, drift, conflicts: per-scene gradient optimisation, offline. Conflicts averaged
  by the loss. No incremental update.
- Known vs unknown: not addressed. LERF notes blurred embeddings where side views are
  missing (Sec 5, READ).
- Identity and merging: none; the field is continuous. GARField groups by affinity.
- Links: text to place via embedding similarity (heatmap). Place to words only by
  testing candidate phrases. No link back to images.
- Frame: world (reconstruction frame).
- Reader: a human or code with a text encoder; relevancy against canonical phrases
  ("object", "things", "stuff", "texture": LangSplat Sec 3.4, READ).
- Cost: LERF about 45 min, about 20 GB on an A100 (Sec 3.6, READ). LangSplat: raw CLIP
  per Gaussian would cost "over 35 times" the memory (Sec 3.3, READ). LangSplatV2 476 FPS
  rendering via sparse codes over a global dictionary (ABSTRACT).
- Training: per-scene optimisation (no labels, pretrained encoders). Not "no training" in
  the strict sense, but no model training.
- Used where: LERF and Feature Splatting in nerfstudio (SNIPPET). F3RM on a real arm,
  few-shot grasping (ABSTRACT). Papers otherwise.

### B. Training-free lifted fields (LUDVIG, NormLift, CoSAG, superpoint graph, D3Fields, OpenScene 2D-fusion)
- Atomic unit: a Gaussian or superpoint with a feature, or (D3Fields) any 3D query point
  projected into the stored images on demand (ABSTRACT).
- Kept: a visibility-weighted average of 2D features. Superpoint method keeps
  "spatially compact and semantically coherent regions" (ABSTRACT).
- Discarded: per-view features after averaging. D3Fields is the exception: it keeps the
  images and computes features at query time.
- Hierarchy: superpoint graph gives coarse and fine levels (ABSTRACT).
- Update: closed-form lift; can be redone as views arrive. NormLift replaces averaging
  with mode voting because linear averages "corrupt CLIP embeddings" (ABSTRACT).
- Known vs unknown: NormLift reliability per Gaussian from intra-view and inter-view
  consistency (ABSTRACT). Closest thing in this family to an observation-quality score.
- Identity and merging: superpoints; otherwise not addressed.
- Links: text to place by similarity. D3Fields links point to images by projection.
- Frame: world.
- Reader: code.
- Cost: CoSAG sub-megabyte field (ABSTRACT).
- Training: none beyond pretrained 2D models (DINOv2, CLIP, SAM).
- Used where: papers, 2024 to 2026. D3Fields on real robot, zero-shot rearrangement
  (ABSTRACT).

### C. Instance-level feature maps (OpenMask3D; Chat-Scene's object features)
- Atomic unit: a 3D instance mask with one CLIP vector (768-d, ViT-L/14, OpenMask3D
  Sec 3.2, READ).
- Kept: one feature per instance, averaged over top-k = 5 views by visibility score and
  L = 3 crop scales (Sec 3.2.1, 3.2.2, READ).
- Discarded: everything below instance level; which views were used is computed, then
  dropped.
- Hierarchy: none.
- Update: offline batch.
- Known vs unknown: the visibility score exists in code, not stored as output.
- Identity and merging: 3D mask proposals (class-agnostic segmenter).
- Links: text to instance by similarity; instance to views via the top-k selection.
- Frame: world.
- Reader: code; Chat-Scene feeds object embeddings plus identifier tokens to a fine-tuned
  LLM (2312.08168, ABSTRACT).
- Cost: Chat-Scene 300 visual tokens for a 100-object scene (Fast3D intro, READ).
- Training: OpenMask3D none (pretrained parts). Chat-Scene fine-tuned.
- Used where: papers. Its idea (object list plus best views) underlies many open-vocab
  maps of branch 3.

### D. 3D tokens for language models (3D-LLM, LEO, LL3DA, LLaVA-3D, Chat-Scene, Video-3D LLM, SplatTalk, Spatial-MLLM, VLM-3R)
- Atomic unit: a token: a 3D patch (LLaVA-3D: 2D CLIP patch plus 3D position embedding,
  ABSTRACT), an object embedding (Chat-Scene), a Gaussian feature (SplatTalk), or
  geometry-encoder tokens (Spatial-MLLM from a visual-geometry backbone, VLM-3R,
  ABSTRACT).
- Kept: whatever the projector learned to keep for QA training data.
- Discarded: not inspectable. Fast3D shows 90% of tokens can go at 3% loss (READ).
- Hierarchy: none in tokens; HiSC (MM 2026, SNIPPET) clusters tokens object-centrically.
- Update: rebuild per query or per scene; no persistent store.
- Known vs unknown: not addressed.
- Identity and merging: Chat-Scene object identifier tokens only.
- Links: language out, sometimes boxes out (LLaVA-3D, ABSTRACT).
- Frame: world or camera.
- Reader: a fine-tuned multimodal LLM. SplatTalk feeds 32,076 tokens ("4 images-worth")
  into LLaVA-OV without 3D QA training and gets ScanQA CIDEr 61.7 vs 50.0 for LLaVA-OV on
  images (Table 1, READ). Its 3DGS encoder is trained on 500 ScanNet scenes.
- Cost: hundreds to tens of thousands of tokens per scene.
- Training: all need training of the encoder or projector; SplatTalk's LLM stays frozen.
- Used where: papers and research checkpoints only.
- Structured-output variant: SpatialLM (2506.07491, ABSTRACT) reads a point cloud and
  writes walls, doors and object boxes as text "scripts". Fine-tuned, room scale, but the
  output is text a frozen planner can read.

### E. Visual-prompt bridges for frozen VLMs (GPT4Scene)
- Atomic unit: an object marker drawn on a BEV render and on frames.
- Kept: global layout (BEV) plus consistent IDs across frames.
- Discarded: depth, quality, everything not drawn.
- Reader: a VLM with images, frozen or fine-tuned.
- Cost: 8 frames at 128 x 123 base (READ).
- Training: optional; zero-shot gains are small (H3).
- Used where: papers. Not usable by a text-only planner.

### F. Feed-forward reconstruction memory (CUT3R, Point3R, StreamVGGT with Evict3R, Mem3R, FILT3R)
- Atomic unit: CUT3R: 768 state tokens of 768-d (Sec 3.4, READ). Point3R: a pointer =
  3D position (mean of a patch's pointmap) plus a feature (Sec 3.2, READ). StreamVGGT: KV
  cache of all past frames.
- Kept: what the decoder needs to regress pointmaps, poses, confidences.
- Discarded: not inspectable.
- Hierarchy: Point3R hierarchical 3D position embedding (ABSTRACT).
- Update: recurrent update per frame. Point3R fuses a new pointer with its nearest
  neighbour below an adaptive distance threshold, otherwise appends (Sec 3.2, App A.1,
  READ). This is position-based identity, the same limit our 10 mm error hits.
- Drift: CUT3R "may eventually drift over very long sequences" (Sec 5, READ). Point3R:
  growing pointers interfere with later pose estimation (Sec 5, READ).
- Known vs unknown: per-pixel confidence output, but CUT3R fills unobserved regions
  (H5).
- Identity and merging: Point3R distance threshold; no undo.
- Links: geometry only. "Exclusively queried for pointmap reconstruction" (Point3R, READ).
- Frame: first-frame or world.
- Reader: its own decoder.
- Cost: CUT3R state 768 x 768 floats = 589,824 values, about 2.4 MB in fp32 (my
  arithmetic). Evict3R cuts StreamVGGT peak memory 18.63 to 9.39 GB, accuracy drop 0.003,
  training-free (ABSTRACT).
- Training: pretrained weights public (CUT3R, Point3R GitHub, ABSTRACT); no finetune
  needed for use.
- Used where: research code, widely forked; no shipped robot product found.

### G. Latent world models (DreamerV3, TD-MPC2, V-JEPA 2-AC, Genie 3, Cosmos)
- Atomic unit: a latent state vector per time step (DreamerV3, TD-MPC2, RECALLED), or
  frame embeddings (V-JEPA 2).
- Kept: what predicts the next latent or reward.
- Discarded: everything else; not inspectable.
- Update: recurrent; no persistent spatial store. Genie 3 memory about one minute
  (READ).
- Known vs unknown: not addressed; generative models invent.
- Reader: a planner in latent space (V-JEPA 2-AC plans to image goals, zero-shot on Franka
  after under 62 h of DROID video, ABSTRACT).
- Training: heavy; pretrained V-JEPA 2 and Cosmos weights exist (Cosmos RECALLED).
- Used where: V-JEPA 2 and Cosmos weights public; Genie 3 research preview. Not used as a
  recallable memory anywhere I found.

### H. Memory in world-model video generators (WorldMem, Context-as-Memory, Echo-Memory)
- Atomic unit: a memory unit = frame latent tokens + pose (x, y, z, pitch, yaw) +
  timestamp (WorldMem Sec 3.3, READ).
- Kept: the frames themselves, compressed by the video VAE.
- Discarded: nothing semantic; redundancy pruned by similarity filtering.
- Update: append; retrieval by confidence = FOV overlap minus time distance (Alg. 1,
  READ).
- Known vs unknown: implicit; a region is "seen" if some stored frame's frustum covers
  it. Fails "when views are blocked by obstacles" (Sec 5, READ).
- Identity: none; frames, not entities.
- Links: pose to frames. No words.
- Frame: world.
- Reader: cross-attention in the generator; "not independent querying" (READ).
- Cost: 600 frames about 21 MB (Sec 7.2, READ). Linear growth (Sec 5).
- Training: yes for the generator; the retrieval rule is plain geometry.
- Used where: papers.

### I. Memory in VLA policies (MemoryVLA, CronusVLA, MEM, VLA-Cache)
- Atomic unit: perceptual and cognitive tokens (MemoryVLA, ABSTRACT); per-frame features
  (CronusVLA, ABSTRACT); video encoding plus text summary (MEM, READ).
- Kept: MemoryVLA merges redundant entries on write (ABSTRACT). MEM keeps up to 18 frames
  / 54 s and an LLM-written summary (Sec III-D, III-B, READ).
- Update: MEM's summary written by an off-the-shelf LLM from subtask outcomes; it
  compresses ("I placed three bowls in the top right cabinet", READ). Naive appending
  caused repeated entries and failure (Sec IV-A, READ).
- Known vs unknown, provenance, frame: not addressed.
- Audit: frozen VLAs keep past frames linearly decodable but largely ignore them unless
  the current view is degraded (2607.03372, ABSTRACT).
- Training: all trained. VLA-Cache is training-free but is compute caching, not memory
  (ABSTRACT).
- Used where: MEM inside Physical Intelligence's pi-0.6-based policies (paper and blog);
  MemoryVLA and CronusVLA code public.

### J. Object-centric slots (Slot Attention, DINOSAUR, STORM)
- Atomic unit: K slot vectors per image.
- Kept: per-object appearance in a fixed budget.
- Discarded: count beyond K; geometry; persistence across views not standard.
- Training: slot module trained on frozen VFM features (STORM 2601.20381, ABSTRACT).
- Used where: policy-learning papers only.
- For us: no off-the-shelf use at object-part level; skip.

### K. Part-level and object-centric fields (Find3D, PA3FF, 2607.03163, GeFF)
- Find3D (2411.13550, ABSTRACT): feed-forward point-cloud part segmentation by text,
  zero-shot, pretrained. Atomic unit: per-point feature queried by part name.
- 2607.03163 (ABSTRACT): point-attached features "vary with viewpoint and sensor noise";
  they resample a continuous object field instead. Trained.
- Relevance: the only line at single-object, part scale. Find3D is usable off the shelf.

## 3. Per research question

**RQ4 atomic unit.** Options seen: point or Gaussian with a feature (A, B), instance
with one vector (C), token (D), positioned pointer (F, Point3R), frame plus pose (H), text
sentence (MEM). Only C and Point3R pointers are discrete entities that could become nodes.

**RQ5 hierarchy.** Scale as the hierarchy axis: LERF pyramid, LangSplat whole, part,
subpart, GARField scale-conditioned groups, superpoint graph. All are segment hierarchies
from SAM, not a tree of named details.

**RQ6 update and drift.** Feed-forward memories drift (CUT3R, Point3R Sec 5). Fields are
offline or closed-form re-lifts. MEM updates text memory only on success to avoid
repetition. No method here re-anchors or rolls back.

**RQ7 who manages.** Silent in practice: the network or the loss resolves conflicts.
MEM is the one case where a separate language model writes the memory.

**RQ8 contract.** In: posed RGB(-D) frames. Out: similarity heatmaps (A, B), pointmaps
(F), generated frames (H), tokens to a fine-tuned model (D). None offers a query API for
a text-only planner; LLM-Grounder (2309.12311, SNIPPET) wraps OpenScene and LERF as tools.

**RQ9 what to keep.** Evidence that compact latents lose what matters: Echo-Memory;
Fast3D shows benchmarks need little. Practical systems keep raw frames (WorldMem,
Context-as-Memory, D3Fields) and fetch on demand by pose or projection.

**RQ10 known vs unknown.** Mostly silent. Partial: NormLift reliability, OpenMask3D
visibility score, WorldMem FOV overlap. Negative evidence: CUT3R and Genie 3 fill unseen
areas.

**RQ11 identity.** Point3R: nearest-neighbour fusion under a distance threshold; no
wrong-merge detection. Feature similarity as a second cue (instance features) appears in
C and branch 3. 2607.03163 reports point features vary with view, so features alone are
also noisy.

**RQ12 links.** Text to place via embedding similarity is the core offer of A, B, C.
Place to text only by scoring candidate phrases or by asking a VLM. Place to image only
where frames are kept (D3Fields, WorldMem, OpenMask3D top-k).

**RQ13 frame.** World frame everywhere; WorldMem stores pose per frame so view-relative
retrieval is computed. No object-frame storage found.

## 4. What does not transfer

- **Text-only reader.** Families D, E, F, G, H output tokens, images or latents for a
  trained or multimodal reader. A frozen text LLM cannot read a 512-d vector or a latent
  state. Only text outputs (SpatialLM-style scripts, MEM summaries, code-computed answers
  from a field) reach our planner.
- **No training.** D, G, H, I, J need training. Off the shelf at one-object scale:
  training-free lifts (B), OpenMask3D-style instance features (C), Find3D (K), pretrained
  CUT3R or Point3R as a geometry front end (F).
- **10 mm cross-view error.** Distilled fields average features across views; a 10 mm
  shift blurs small features into neighbours (boundary mixing is a known issue: search
  results on semantic bleeding, SNIPPET). Point3R's distance-threshold merge would merge
  features about 3 cm apart or split one into two.
- **Single object, fine detail.** CLIP-style features are bag-of-words and miss small or
  far objects (LERF Sec 5, READ; CLIP-DINOiser, SNIPPET). Legible text is not a strength
  of any learned representation found (Genie 3 READ). OCR must stay a separate channel.
- **Benchmarks.** ScanQA, SQA3D and VSI-Bench are room scale and answerable with weak 3D
  use (H2); they cannot validate our structure.

## 5. Ranked reading list (for deep reading)

1. NormLift, https://arxiv.org/abs/2609.18898. Training-free per-Gaussian reliability
   from feature norm and multi-view support. Extract the formula; test as a per-region
   observation-quality score.
2. WorldMem, https://arxiv.org/html/2504.12369v3. FOV-overlap retrieval of stored frames.
   Extract Algorithm 1 as a "give the images it was seen in" query.
3. OpenMask3D, https://arxiv.org/html/2306.13631. Top-k views by visibility per instance.
   Extract the scoring and how to keep the view ids instead of discarding them.
4. MEM, https://arxiv.org/html/2603.03596v2. LLM-written text memory and its failure
   (repetition). Extract the summarisation prompt and update rule.
5. CoSAG, https://arxiv.org/abs/2607.10237, and the superpoint graph,
   https://arxiv.org/abs/2504.13153. Training-free lift, storage, hierarchy. Extract cost
   at one-object scale.
6. Point3R, https://arxiv.org/html/2507.02863. Pointer fusion rule and threshold.
   Compare with our 10 mm error.
7. Echo-Memory, https://arxiv.org/abs/2606.09803. What compressed memories lose on
   return. Extract the three-probe evaluation; could inform our benchmark.
8. Find3D, https://arxiv.org/abs/2411.13550. Off-the-shelf part queries on one object.
   Extract input requirements and failure cases.
9. Does your 3D encoder work, https://arxiv.org/html/2506.05318v2, and Revisiting 3D LLM
   Benchmarks, https://aclanthology.org/2025.findings-acl.1222/. For benchmark design.
10. SpatialLM, https://arxiv.org/abs/2506.07491. Structured text as the output format of
    a learned 3D model.

## 6. Gaps and open questions

- Not found: any learned representation exposing provenance (which image), observation
  quality, or explicit "not observed" to a reader. Partial signals exist only inside code
  (H6).
- Not found: a language-feature field queried by a text-only LLM through tools at object
  scale. LLM-Grounder is the nearest (room scale, 2023, SNIPPET only).
- Not found: learned representations that handle fine text or OCR in 3D. Searched for
  "scene text OCR 3D language field"; results were generic language-splatting papers.
- Unverified: whether nvblox ships a deep-feature layer. The Isaac ROS technical page
  lists TSDF, color, mesh, ESDF and others, no feature layer (READ). An NVIDIA-research
  article says nvblox_torch can "fuse image features from vision foundation models"
  (edge-ai-vision.com, READ). Status in the shipped package unclear.
- Not checked today: exact token counts for 3D-LLM, LEO, LL3DA, Video-3D LLM; DreamerV3
  and TD-MPC2 state sizes; OpenScene, ConceptFusion, VLMaps, CLIP-Fields details
  (all RECALLED or skipped).
- GPT4Scene zero-shot numbers came through a summarising fetch; recheck against the
  table before quoting.
- Open: is a feature field worth keeping at all, or only as a code-side index that
  answers "where is X" and returns region ids plus source views to the planner?
