# Structured Interfaces for Automated Reasoning with 3D Scene Graphs

**arXiv:2510.16643v1** (MIT-SPARK lab — Hughes, Carlone et al.; builds on Hydra [8]).
Full text: https://arxiv.org/html/2510.16643v1

## 0. One-line identity

Not a spatial/viewpoint-reasoning paper — it's a **symbolic-grounding architecture** paper: how should an LLM query a text/graph representation of a scene (rooms, objects, positions) to answer questions and generate PDDL goals. No images, no cameras, no viewpoints. Relevant to us only via the **interface-design analogy**: context-dump vs. tool-mediated-query vs. code-gen, as three ways to expose a structured space to an LLM.

## 1. Core method

- Encode the 3D scene graph (3DSG) in a **Neo4j graph database**. Nodes: `Object` (id, class, position=POINT, parent-place ids), `MeshPlace`, `Place`, `Room` — same schema Hydra [8] already produces from Spot-robot SLAM. **Purely symbolic**: no image embeddings, no visual crops, no color/appearance data stored per node (verified quote: "Each node has a label... and a set of key-value attribute pairs... {class: vehicle}"). Edges: `[:CONTAINS]` (hierarchy, supports transitive `[:CONTAINS*]`), `[:OBJECT_CONNECTED]`, `[:PLACE_CONNECTED]`, etc. Spatial ops available via Cypher's `point.distance`.
- Expose the DB to the LLM as an **agentic tool**: the model emits `(chain-of-thought, Cypher query, pause-token)`, the harness executes the query against Neo4j, appends the result, and lets the model continue — up to **M=5** tool calls (prompt: "You can call the cypher query tool up to 5 times"). No report of the actual average calls used.
- Two downstream tasks: (1) **Q&A** — answer scene questions in a typed mini-language (Set/List/Dict/Point — "SLDP"); (2) **Instruction-following** — emit a **PDDL** goal, executed by Fast Downward on the real robot in the supplemental demo (not part of the main quantitative eval).
- Three architectures compared, each in agentic (multi-call, "(A)") and single-shot variants: **Context Window** (serialize a filtered subset of the graph as text into the prompt), **Python** (LLM writes code against the real `spark_dsg`/Hydra Python API, executed against the graph object), **Cypher** (the proposed method).

## 2. Concrete numbers (all from the HTML full text, Tables I–IV; §Experiments)

**Scene graphs**: Small = indoor academic building floor, "65 total objects over 12 categories, 96 mesh places, 5 regions (2 categories: hallway, lounge)." Large = outdoor summer-camp environment, "314 objects, 15944 places, 124 regions, 4 region categories." Both built via Spot teleop + Hydra.

**Datasets**: Q&A = 100 queries per graph (200 total), scored by exact/DNF logical equivalence, not LLM judge. PDDL = 102 instructions (small graph), 82 instructions (large graph), scored by disjunctive-normal-form logical equivalence with numeric tolerances (ε=0.01, δ=0.01). **No variance/seeds/error bars reported anywhere** — single-run numbers.

**Table I, GPT-4.1, task success rate:**

| Method | Q&A Small | Q&A Large | PDDL Small | PDDL Large |
|---|---|---|---|---|
| Cypher (A) | 0.85 | 0.77 | 0.65 | 0.56 |
| Cypher | 0.82 | 0.77 | 0.55 | 0.46 |
| Context Window | 0.75 | 0.33 | 0.65 | 0.41 |
| Python (A) | 0.51 | 0.32 | 0.38 | 0.51 |
| Python | 0.38 | 0.31 | 0.35 | 0.55 |

**Table III, flagship models, best variant:**

| Model | Q&A Small | Q&A Large | PDDL Small | PDDL Large |
|---|---|---|---|---|
| Claude Opus-4.1 | 0.94 | 0.83 | 0.88 | 0.80 |
| GPT-5 | 0.86 | 0.81 | 0.77 | 0.70 |
| GPT-4.1 | 0.85 | 0.77 | 0.65 | 0.56 |
| Qwen3-32B | 0.74 | 0.67 | 0.56 | 0.55 |

**Table IV, PDDL token counts (input / tool / output), GPT-4.1:**

| Method | In-S | Tool-S | Out-S | In-L | Tool-L | Out-L |
|---|---|---|---|---|---|---|
| Cypher (A) | 2366 | 205 | 319 | 2395 | 1596 | 942 |
| Context Window | 8735 | — | 134 | **582202** | — | 56 |
| Python (A) | 6161 | 82 | 390 | 6237 | 33 | 337 |

No Q&A-task token table given (only PDDL). No latency/wall-clock and no $-cost numbers anywhere in the paper — token counts only.

## 3. The one number this branch actually needs: small-scale performance is close, not decisively won

This is the load-bearing nuance the framing doc's selection blurb ("beat context-window baselines with far fewer tokens") elides. On the **large** graph, Cypher's win is dramatic and token-driven (0.77 vs 0.33 Q&A; 2395 vs 582202 input tokens — a >240x token blowup for context-window). But on the **small** graph — 65 objects, 96 places, 5 regions, roughly the scale of our discrete view-menu problem — the picture is much softer:
- Q&A small: Cypher(A) 0.85 vs Context-Window 0.75 (+10 pts, real but modest)
- PDDL small: Cypher(A) 0.65 vs Context-Window **0.65 — tied**
- Token cost at small scale: Context-Window 8735 in-tokens vs Cypher 2366+205=2571 — still a ~3.4x reduction, but nowhere near the 240x large-graph gap.

Paper's own framing: "the context window baseline is a competitive method on small scene graphs." No deeper mechanistic analysis is offered for *why* it's competitive beyond "it still fits" — this is asserted, not ablated further.

## 4. Why context-window degrades at scale — mechanisms given (not just token count)

Three distinct causes named in the text, not merely "doesn't fit":
1. Context-window overflow at large scale (582k tokens > most model limits) — brute capacity.
2. **Distant-reference / needle-in-haystack**: "even simple natural language expressions like 'the bag in the road' may require attending to distant parts of the context window... which remains a challenge for even the most recent LLMs" — this is a reasoning-over-long-context failure, independent of whether the tokens technically fit.
3. **Ontology idiosyncrasy**: "some scene graph ontologies are useful for robots but are idiosyncratic, making inter-layer associations in natural language more difficult to ground" — a labeling/schema-design failure mode, not a scale failure.

## 5. Failure mode that survives the fix: hierarchy traversal

Even the winning Cypher method doesn't solve hierarchical reasoning: "all methods struggle with language that requires traversing the hierarchy of the scene graph. While Cypher does provide a transitive containment operation, the GPT-4.1 inconsistently chooses to generate queries that take advantage of it." I.e., **giving the model the right tool (transitive `[:CONTAINS*]`) does not guarantee it reliably invokes it** — a capability/tool-affordance gap, not just a schema gap. This is a directly transferable warning for any adjacency/hierarchy structure (view graphs, face-adjacency) we might expose via query interface: exposing the operation doesn't make the model consistently use it correctly.

## 6. Agentic (multi-call) vs single-call ablation

Agentic generally helps but inconsistently, and can hurt: Cypher(A) 0.85 vs Cypher 0.82 (small Q&A, +3pts); Python(A) vs Python: +13pts small Q&A (0.51 vs 0.38) but only +1pt large Q&A, and Python(A) *underperforms* single-call Python on PDDL-large (0.51 vs 0.55). Paper's honest conclusion: "Treating the retrieval step of GraphRAG as a tool is a useful component," but the benefit is task/model-dependent, not uniform. No mechanism is offered for the Python-PDDL-large inversion — flagged, not explained.

## 7. What is explicitly NOT compared (evidence-discipline note)

- No experimental comparison to SayPlan, ConceptGraphs, or EmbodiedRAG — these are cited as related/concurrent, not run as baselines. Any claim that Cypher beats SayPlan-style collapse/expand is **not supported by this paper** — it only beats context-window-dump and code-gen, both built by the same authors as baselines.
- No robot-embodied quantitative eval — the Spot demo (correcting a mislabeled object, executing a PDDL plan) is a qualitative supplemental-video illustration, not scored.
- No cost ($) analysis, no latency, no variance/seeds. Numbers should be read as single-run point estimates, not statistically robust deltas — flag per branch's "evidence discipline" requirement: **demonstrated, not ablated for variance**.

## 8. What transfers to our setting

- **Architecture analogy only, not the mechanism.** Our action space (a discrete menu of candidate camera views over one tabletop object) is closer in scale to their *small* graph (65 objects) than their *large* one (15,944 places) — and at that scale their own data says context-dump is competitive (tied on PDDL, +10pts on Q&A). This directly informs the branch question "is dumping a small space worse than exposing it through a query?": **on this paper's evidence, not dramatically — the query interface's advantage is concentrated at scales far larger than ours.** If our view-menu stays in the tens of options, this paper does not license building a Cypher-style query tool on cost/accuracy grounds; the case for a query interface would have to come from elsewhere (e.g., hierarchy/relational reasoning quality, not token budget).
- **Mechanism worth stealing regardless of scale**: separating "what the model reasons over" from "what's dumped in-context" via a tool call, with chain-of-thought interleaved between calls, is a clean pattern if we ever want the orchestrator to interrogate the geometry (e.g., "what's the adjacency of face 3") rather than have it pre-flattened. But per §5, even with the tool present, models don't reliably use hierarchical/transitive operations — a caution against assuming a view-adjacency query tool will get *used* correctly just because it's *offered*.
- **Symbolic-only content is a hard boundary.** The whole 3DSG here is class labels + centroid positions; there is no visual grounding, no image, no notion of "orthogonal to a surface" or camera framing. Nothing here informs render encoding, egocentric/allocentric framing, marks-on-image, or any of the vision-facing branch questions (MMSI-Bench, ViewSpatial, SpinBench, FlipSet, ego/allo gap, viewpoint-selection chicken-and-egg) — **this paper is silent on all of them.** Do not cite it for anything vision/viewpoint-specific.

## 9. What does NOT transfer, and why

- **The large-graph win (0.77 vs 0.33, 240x token reduction) does not transfer** to a tabletop discrete-view action space of our size — the win is a function of graph size (15,944 places), which is 2–3 orders of magnitude beyond any plausible view-menu we'd build (tens to low hundreds of candidate views).
- **PDDL/Q&A grounding tasks are not view-selection tasks.** The model here is answering "what/where" questions about static labeled entities, not choosing among near-identical geometric options (our "count vs description distinctiveness" problem, or ego/allo viewpoint relations) — no evidence here bears on those.
- **No metric/continuous pose reasoning tested** — irrelevant to us anyway per branch scope, but noting it's absent even as a control.
- **Hierarchical containment traversal remains broken even with the best interface** — so if RQ3's topology ambition (adjacency, "opposite face") leans on hierarchical/transitive reasoning through a query tool, this paper's own failure mode (§5) is a warning, not encouragement: the tool existing is necessary but insufficient.

## 10. Branch-question-by-branch disposition

- **"Hierarchical/on-demand pattern... is dumping a small space worse than exposing through a query?"** — Directly answered by §3/§8: on a small graph (our scale), dumping is close to competitive (tied to -10pts), not clearly worse. The dramatic win requires graph sizes far beyond ours. **Verdict: weak evidence against building a query interface purely for our current scale; motivation would have to be reasoning quality (hierarchy), not token/accuracy economics.**
- **"Evidence discipline" table entry**: Paper = demonstrated comparison (own baselines, own authors, single-run, no variance), not ablated against third-party systems (SayPlan/ConceptGraphs cited but not run). Flag: **no variance reported (Y)**, comparison is against author-built baselines only.
- All other branch questions (MMSI-Bench/ViewSpatial/SpinBench/FlipSet capability floors, ego/allo, render encoding, marks, stopping criteria, random baseline, chicken-and-egg viewpoint selection) — **this paper has nothing to say**; it is a text/symbolic grounding paper, not a vision/viewpoint paper. Do not force citations here.

## 11. Exact interface schema (for anyone building a similar tool)

- LLM output format per turn: `(chain-of-thought, Cypher-query, <pause>)`.
- Harness executes query against Neo4j, returns result as `X_rag`, appended to context, generation resumes.
- Prompt includes: scene-graph ontology description, Cypher language primer, PDDL domain predicates (for the instruction task), and the tool-call budget statement ("You can call the cypher query tool up to 5 times").
- No verbatim example Cypher query shown in the main text; Fig. 2 caption gives only a conceptual description ("MATCH all instances of lounge type Rooms 'r' that contain a bag type Object 'o' and return the id").
- Python baseline used the real `spark_dsg` (Hydra) API, e.g. `objects = [o for o in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes]` — a real robotics library, not a synthetic one, so the Python baseline's underperformance (0.38–0.55 range) is a meaningful data point about code-gen against a real but LLM-unoptimized API, not a strawman.
