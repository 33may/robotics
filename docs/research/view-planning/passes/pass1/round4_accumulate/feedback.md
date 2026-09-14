Branch 1 — view-quality-scoring — Round 4 accumulation

Verdict: we measure it ourselves; the literature only lends us the formula template to fit. lets discuss what it will actually give us and if we need to go that deep


camera, we could request other camera if we will need one for our research

Q3 — Can orthogonal ever be worse?

Contested, and interestingly so. Cases where looking straight-on hurt: laser scanners saw noise spikes exactly at perpendicular on shiny surfaces (the return blinds the detector — doesn't transfer to our camera though); and human inspection experts deliberately chose non-orthogonal positions because one tilted view covered more surface than one straight view. Cases for orthogonal: every automated system in the field places cameras on the normal by construction and never questions it. For our specific camera, the old worry ("the projector reflects back") is void — no projector. The remaining open worry: a shiny surface reflecting the room light straight into the lens at perpendicular. No paper we read answers that; the one paper that would (a 2024 work that models glare and excludes blown-out regions) is the top unread item.

we dont care about the sensors once we are in what we consider good view, when we are in orthogonal view we care only about the rgb image nad in the context of inspection the orthogonal view is what we want to get.

Q4 — Are defects actually easier to see from an orthogonal view?

The literature simply doesn't know — and this is the most uncomfortable finding for Albert's claim. The entire inspection-planning field defines "inspection" as measuring dimensions, not spotting flaws — the canonical survey never uses the word "defect" once. The only solid angle-effect that is proven is resolution: tilt the surface and each cm² of it gets fewer pixels/points (measured: 38,500 points looking straight vs. 4,900 at a steep tilt — 8× loss). But classic industrial practice for finding shallow scratches uses raking light — deliberately grazing angles — which points the opposite way. So the honest position is: orthogonal is provably best for measurement quality, resolution, and readable evidence images; it is not proven best for defect visibility, and for some defects it may be wrong.

I dont know, but now we are actually doing not defect detection but object inspection, we want the orthogonal view now, maybe move later


## benchmarking-view-planning.md

Okay here again, we dont focus that match on the scratch or smth like that detection, the inspecxtion now is answering the questions about the objects.

Also we could separate the bencham,arks, now it is really targeted at the orthogonality objective, but I would say we want to build the general benchamark framework, that we will be able to use as general view planning, optimize for number of viewws, test probmpts maybe, check the moving path statistics, maybe think of simulation and RL, maybe the general benchmark that will give us statistics and prices, and orthoogonality is just one of the modes we mightr consider in the benchmark.

F8 — Frontier VLMs compose multi-turn view plans badly, and the failure grows with viewpoint distance.

we have this fine in our design.

### Benchmark vision — 3 tiers (recorded; benchmark is auxiliary for now)

- Tier 0 — single-image VLM benchmark: one image, one decision (read text, spot logo, etc.). Images sampled from existing data.
- Tier 1 — replay episodes from dense capture: robot travels all viewpoints once and records; episodes replay against the dataset (a "move" = fetch image from dataset). Evaluates: how well LLM/VLM picks views, object coverage, moves to answer, travel distance. Later: unattended runs — every real application run records benchmark-grade data for future benchmarks and model training.
- Tier 2 — full-scale quantitative eval: pick N objects × M runs, collect statistics (% correct, moves, duration). Supports real robot in the loop, replay, and simulation — sim will definitely be used at some point in the project.

Answers to open questions:
- per-view data: only the real useful data — RGB, depth, camera pose/intrinsics. No joint configs, timestamps, object placement poses, lighting notes. Travel time maybe later, not now.
- travel costs in replay: computed in the kinematic sandbox from poses — fast, already implemented, no robot motion needed.
- QA pairs: the LLM writes them under my guidance.
- orthogonality is just one pluggable mode of the framework, not the benchmark target.

Status: benchmark is an auxiliary task now — other research focuses come first.

## view-set-selection-under-constraints.md

Important correction to the branch's premise: we do NOT have the scanned object up front. It is a loop — scanning, picking, replanning. From the first view we reconstruct only part of the object; from the next view we reconstruct more and can build a new, better set of views. There is no complete reconstruction from the beginning.

Consequences (agent): this changes the contract's survey-first framing. The primitive fit and the view set are recomputed every iteration on the partial cloud; the exact set-cover solve is cheap (~1 s) so re-solving per round is fine. The branch's anti-NBV argument ("the condition that forces the field toward iterative/learned planning does not hold for us") loses its premise — the iterative NBV structure IS our structure, just with primitive fitting in the middle and no learned selector. Pan 2025's treat-prior-as-fallible pattern and Branch 3's partial-shape fitting (the least-supported case: half-scanned cylinder) become central rather than edge cases.

## structuring-the-viewpoint-space.md

Correction on representation: exposing primitive tuples/parameters (cyl_1, normals, θ knobs) to the LLM is unrealistic — messy, not interpretable, hard to reason about. The viewsphere existed to abstract the geometry away. The idea is an **intelligent viewsphere**: first compute the good views (from the fitted geometry — orthogonal, reachable, framed), then present those computed views to the LLM as a simple finite menu (view number + plain directional words) to select from. Geometry stays entirely internal; the LLM never sees primitives or parameters.

Addition on the benchmark's purpose: open design choices like the view representation (angle labels vs directional words vs anything else) are exactly why we need the benchmark framework — evaluate the different options as swappable components and select the best by numbers, instead of locking my intuitions as decisions.

## vlm-spatial-interfaces.md

I like the idea of the 2-stage selection (first which surface/primitive, then which view of it). Nothing more to add.
