# Round 2 map, branch 4: memory for language-model agents (`llm-agent-memory`)

**Date:** 2026-10-06. **Mapper:** one agent, abstracts and landing pages for breadth, about 10 bodies read.
**Tags:** READ = read in the body today; ABSTRACT = abstract or landing page today; RECALLED = own memory, not checked today; SNIPPET = seen only in a search-engine summary, not on the source.
**Caveat on READ:** bodies were read through a fetch tool that summarises long pages. Numbers marked READ came back with their table or section; spot-check the load-bearing ones in round 3.

## 1. Headline findings

**H1. Extract candidates into a structure first, then apply an explicit rule. Whether code or the model applies the rule matters much less.** The strongest 2026 results on stale facts come from separating "find the evidence" from "decide".
- Post-retrieval assembly (arXiv 2606.01435, Table 1; corrected by the lead against the raw table on 2026-10-06, the first version of this bullet credited the whole gain to code). MemoryAgentBench FactConsolidation single-hop, gpt-4o-mini, average over 6K to 262K context: direct free-text answer 67.2% (61 at 262K); extract candidates, then the LLM applies the newest-version rule 76.0% (82 at 262K); same extraction, then Python applies the rule 78.0% (82 at 262K). The pipeline gains 10.8 pp (21 pp at 262K). The deterministic executor accounts for 2.0 pp of that, and 0 pp at 262K. The paper itself assigns most of the gain to the structured decomposition. Prior best on the multi-hop variant was 7%.
- TEPA (2608.07429, ABSTRACT): keyed precedents revoked by rule when fresh evidence lands under the same key. 0.950 accuracy on full reversals, against 0.210 for append-only memory and 0.309 for **no memory at all**. Memory that is never revoked is worse than none.
- Control-plane placement (2606.15903, ABSTRACT), 13 configurations, 385 cases: code alone fails at canonicalisation (5% on obfuscated identifiers, 0% cross-lingual). LLM at write time gets canonicalisation right but fails intent-aware deletion. LLM at mutation time (hybrid) is best overall: 91.7 to 93.2%.

**H2. Running systems converge on a hybrid read path: a small pushed brief carrying handles, plus fetch-by-handle tools.** This is the accessibility-tree pattern, and it now exists for a robot.
- Thea (2608.11246, READ): a per-turn "scene graph brief" lists every object ref (e.g. `cup_2`) with coarse position, confidence and freshness. Details come from ref-keyed tools (`get_object_relations`, `get_image`), and the same ref goes to action tools (`navigate_to(target='cup_2')`). Quote: "anything omitted costs one extra query." GPT-5.5 decides; it ran on an Astribot S1, an AgileX Cobot Magic and a Unitree G1.
- The same shape shows up in products: Claude Code (CLAUDE.md pushed, grep and glob pulled; Anthropic context-engineering post, READ), Aider (repo map, 1,000 tokens by default, graph-ranked; READ), Letta (core memory blocks always in the prompt, archival memory searched; READ for blocks), Playwright (element refs like `e15`).
- Microsoft's playwright-cli (READ, README) goes further than Playwright MCP: snapshots are saved to disk and the model gets refs, because pushing "verbose accessibility trees into the model context" costs too many tokens.

**H3. Memory beats full context only once the history is large. Raw images in context hurt.**
- EmbodiedMemory-Bench (2609.28236, READ, Table 2 and Table 3; this independently verifies the earlier agent's numbers): full context 36.6% success at 41,770 tokens. Adding the maximum historical RGB drops it to 26.2% at 206,489 tokens (4.9x tokens, 9.5x latency). Three of the four published memory systems scored **below** full context: MIRIX 28.2, MemVerse 28.9, MMA 23.4, full context 35.3, TeleMem 37.8. The trained EMem scored 58.9.
- Mem0's own paper (2504.19413, READ, Tables 1 and 2) on LoCoMo, where conversations average 9K tokens (LoCoMo abstract): full context 72.90 J beats Mem0 66.88 and Mem0g 68.44. Mem0's p95 latency was 1.44 s against 17.1 s for full context.
- At about 115K tokens the result flips (Zep 2501.13956, READ, LongMemEval): Zep with gpt-4o 71.2% at 1.6K context tokens against 60.2% for full context at 115K tokens; latency 2.58 s against 28.9 s. This matches the MIT 2510.16643 crossover we already hold.
- Chroma's "Context Rot" (READ, 2025-07-14): all 18 models get worse with input length, even on simple tasks. A focused ~300-token LongMemEval prompt beats the full ~113K-token prompt. One distractor already hurts. The post gives no single headline number.

**H4. The better designs never destroy the source: they invalidate, keep history, and link every item back to its raw episode.**
- Zep/Graphiti (READ): the episode subgraph is kept as non-lossy raw input. A fact edge carries four timestamps. A contradiction sets `t_invalid` on the old edge, and the edge is never deleted.
- Mem0g (READ): conflicting relations are marked invalid, not removed. ObsMem (2606.18847, READ) keeps a history and a current snapshot per (entity, attribute).
- Thea (READ): each node stores an image path, and graph metadata records "provenance, coordinate frame, and update time."
- Where an LLM rewrites in place, error enters and compounds: A-MEM's "memory evolution" rewrites old notes (ABSTRACT); Mem0's plain DELETE drops facts (READ). "Experience-following" (2505.16067, ABSTRACT): stored mistakes propagate into later outputs.

**H5. "How well known" reaches a language model only as freshness, confidence, provenance class, or a frontier.** Viewing angle and resolution never do.
- ObsMem (READ) labels each belief fresh, stale, uncertain or contradicted, and each event observed or reported. Rules write it; no LLM writes it. On Memory QA it scores 0.713 judge, against A-Mem 0.575 and Mem0 0.554 (Table 2).
- 3D-Mem (2411.17735, READ) represents "not yet looked" as frontier snapshots: an unexplored region, a navigable pose and an image.
- This is the closest analogue to RQ10 in this family. Viewing angle and resolution, our main concern, appear nowhere.

**H6. The 2026 embodied trend is trained memory writers. Training-free embodied memory is retrieval-centric and room scale.**
- Trained writers: EMem writes through deterministic tool calls (`update_object`, `remember_event`, `remember_scene`), but the policy that chooses them is fine-tuned (READ, §5.2). Also MemCtrl (trained gate, ABSTRACT) and MemCompiler (learned compiler, ABSTRACT; injecting memory up front "may degrade lightweight executors below the no-memory baseline").
- Training-free embodied memory (ReMEmbR, Embodied-RAG, eMEM, EmbodiedLGR, Thea) stores captions or objects at room to building scale. None works at the scale of one object's surface.

## 2. Approach cards

### C1. Caption memory with retrieval tools: ReMEmbR, Embodied-RAG, eMEM, EmbodiedLGR
- **Atomic unit:** ReMEmbR: one VLM caption per 3 s video chunk (VILA1.5-13b), with its embedding, position and timestamp (READ). eMEM: graph nodes indexed by SQLite, hnswlib and an R-tree (ABSTRACT).
- **Kept:** text plus pose plus time. **Discarded:** pixels after captioning. In ReMEmbR the raw frames are not used at query time (READ).
- **Hierarchy:** none in ReMEmbR. Embodied-RAG: a "semantic forest" of descriptions "at varying levels of detail" (ABSTRACT). eMEM: tiered consolidation into summaries (ABSTRACT).
- **Update, drift, conflicts:** append only in ReMEmbR. Not addressed elsewhere.
- **Known vs unknown:** not addressed.
- **Identity and merging:** not addressed. Every caption is its own record.
- **Links:** strong. ReMEmbR gives the LLM three tools, retrieval by text, by (x,y,z) and by time ("HH:MM:SS"), up to 3 retrieval steps, and the answer comes back as JSON (READ). eMEM exposes 10 recall tools, including "concept-to-location resolution" (ABSTRACT).
- **Frame:** world (map frame).
- **Reader and interface:** an LLM agent pulls through tools.
- **Cost:** ReMEmbR about 25 s per question on a 21.5-minute video (READ).
- **Training:** none.
- **Used where:** ReMEmbR was demonstrated on a Nova Carter with a Jetson Orin 32GB (READ); NVIDIA released the code (RECALLED). EmbodiedLGR ran on a physical robot (IROS 2026, ABSTRACT). eMEM is simulation only (ProcTHOR, ABSTRACT).
- **Numbers:** ReMEmbR with GPT-4o, descriptive accuracy 0.62 (short videos) and 0.65 (long), against 0.57 and 0.55 for an LLM fed all captions. The multi-frame VLM baseline cannot process long videos (READ, Tables I and II).

### C2. Image-snapshot memory for VLM readers: 3D-Mem, Mobility VLA, FindingDory
- **Atomic unit:** 3D-Mem: a memory snapshot, i.e. one frame plus the set of objects it covers, chosen by co-visibility clustering. Plus frontier snapshots (READ). Mobility VLA: a tour video plus a topological graph (ABSTRACT).
- **Kept:** a few frames that cover all objects: 10.94 snapshots from 39.76 observations on A-EQA. **Discarded:** redundant frames. Prefiltering by K=10 relevant classes leaves 3.26 snapshots for the VLM (READ).
- **Hierarchy:** none. **Update:** clustering is recomputed as objects arrive. Conflicts not addressed.
- **Known vs unknown:** frontier snapshots mark unexplored regions (READ).
- **Identity:** from the underlying object map (ConceptGraphs style). Not an LLM decision.
- **Links:** image to object set to 3D position.
- **Frame:** world.
- **Reader:** a VLM that sees the images. **Cost:** 3.3 to 3.6 s VLM time per step (READ, Table 6).
- **Training:** none for 3D-Mem.
- **Used where:** papers. Mobility VLA ran in an 836 m² real space (ABSTRACT).
- **Numbers:** A-EQA 52.6 LLM-match, against 46.9 for Explore-EQA and 47.2 for ConceptGraph plus frontier. GOAT-Bench 69.1% success, against 61.5 (READ, Tables 1 and 3).

### C3. Scene graph as context, a brief plus ref-keyed queries: Thea, SayPlan, MoMa-LLM, KARMA, RakuAI
- **Atomic unit:** Thea: an object node holding ref id, 3D box, confidence, freshness timestamp and an image path. Edges "on", "inside", "holding", "near" with measured distance. Containers keep state and contents (READ).
- **Kept:** symbols, boxes, relations, an image pointer. **Discarded:** dense geometry, which stays in perception.
- **Hierarchy:** SayPlan shows a collapsed graph (top levels only), and the LLM calls expand and contract (ABSTRACT, project page). An 82.1% token reduction is quoted in a SNIPPET only, not found on the pages opened. Thea's brief "selects fields rather than pruning nodes" (READ).
- **Update, drift, conflicts:** Thea has two writers, perception and execution. Action outcomes edit the graph **only when the evaluator confirms them** (READ). The evaluator is 93.3% accurate, with 6.7% false "success" (READ, Fig. 9). KARMA: short-term memory records changes, with a "memory replacement" strategy (ABSTRACT).
- **Known vs unknown:** confidence and freshness per node. Staleness is resolved by active perception (READ).
- **Identity:** the ref is minted by the perception backend. Ambiguity ("which red cup") is resolved by the model calling `get_image` and then binding the ref (READ). Undoing a wrong merge is not addressed.
- **Links:** ref to box, image and relations. **Frame:** world. The frame is recorded in metadata (READ).
- **Reader:** brief pushed every turn, details pulled by ref. RakuAI: the model passes names ("pallet-007"), code returns JSON with distance and clearance (READ, vendor page).
- **Cost:** not quantified in Thea.
- **Training:** none. The planner is GPT-5.5, the evaluator Qwen3.7-Plus.
- **Used where:** Thea on three real robots, research. Success is about 95, 85 and 75% on levels L1 to L3, read approximately from a figure, and there is **no ablation without the graph** (READ). RakuAI MCP server in production since March 2026, but no robot deployments yet (READ, vendor). MoMa-LLM and KARMA real-robot demos (ABSTRACT).

### C4. Typed embodied memory with tool-call writes: EMem, ObsMem (WorldLines), RoboMemory, ABot-AgentOS
- **Atomic unit:** EMem: spatial entity graph, events, and scene snapshots (READ). ObsMem: typed atoms on Event, State, Belief and Commitment tracks, each with entity, attribute, value, time and provenance (READ).
- **Kept:** state trails including overwritten values. **Discarded:** raw images in context. EMem evidence shows they hurt.
- **Hierarchy:** ObsMem plans a query over "views". RoboMemory has Spatial, Temporal, Episodic and Semantic submodules over a knowledge graph (SNIPPET).
- **Update, drift, conflicts:** ObsMem writes by rule through an "observer gate" and keeps history plus a current snapshot (READ). EMem: a trained policy issues deterministic tools (READ).
- **Known vs unknown:** ObsMem labels fresh, stale, uncertain, contradicted, and observed vs reported (READ).
- **Identity:** keyed by (entity, attribute). Merging not addressed.
- **Links:** EMem scene memory ties objects to snapshots (READ).
- **Frame:** not addressed (household symbolic).
- **Reader:** pulled queries (`query_spatial`, `query_event`, `query_scene` in EMem; READ).
- **Training:** EMem yes (SFT). ObsMem no.
- **Used where:** papers and benchmarks.
- **Failure taxonomy (EMem-Bench intro, READ):** 100 failed trajectories: fine-grained visual cues forgotten 35%, world state from interaction outcomes not recorded 31%, changes overlooked 18%, no generalisation from experience 7%.

### C5. OS-style paging managed by the model: MemGPT and Letta, Anthropic memory tool, context editing, compaction, note files
- **Atomic unit:** MemGPT and Letta: labelled core blocks (label, value, character limit) plus archival and recall stores (READ for blocks; archival RECALLED). Anthropic: files in a memory directory (READ).
- **Kept:** whatever the model writes. **Discarded:** old tool results. Context editing clears stale tool calls and results (READ). Compaction summarises the history and restarts the window (READ, context-engineering post).
- **Hierarchy:** two tiers, in context and out of context.
- **Update, drift, conflicts:** the model edits through tools. Letta sleep-time agents rewrite memory asynchronously (ABSTRACT, 2504.13171). No conflict mechanism; drift is not measured.
- **Known vs unknown, identity:** not addressed.
- **Links:** file paths and ids act as "lightweight identifiers" that get loaded just in time (READ).
- **Frame:** not applicable.
- **Reader:** a hybrid of pushed blocks or CLAUDE.md and pulled files.
- **Cost:** Anthropic reports +39% from the memory tool plus context editing and +29% from context editing alone on an internal agentic evaluation, and 84% fewer tokens on a 100-turn web search (READ, blog, 2025-09-29; internal evaluation, not reproducible).
- **Training:** none.
- **Used where:** shipped. Claude API memory tool (beta), Claude Code, Letta (OSS plus cloud), and the Pokémon agent's notes (READ, as an anecdote).

### C6. LLM-decided fact extraction: Mem0, A-MEM, LangMem
- **Atomic unit:** a fact sentence (Mem0). A note with description, keywords, tags and links (A-MEM, ABSTRACT).
- **Update:** Mem0: the LLM picks ADD, UPDATE, DELETE or NOOP per fact by function call, using a conversation summary plus recent messages (READ). A-MEM: new notes can rewrite old notes (ABSTRACT).
- **Hierarchy:** none (Mem0g adds a graph).
- **Known vs unknown, frame:** not addressed.
- **Identity:** semantic similarity to existing memories.
- **Links:** text only.
- **Reader:** retrieval pushed into the prompt per query.
- **Cost:** Mem0 about 7K tokens of memory per conversation, Mem0g about 14K (READ).
- **Training:** none.
- **Used where:** Mem0 OSS and hosted. The vendor blog claims 21 framework integrations (LangChain, CrewAI, OpenAI Agents SDK, Google ADK and others; READ, vendor). It also reports LoCoMo 92.5 for its 2026 algorithm, against 66.88 in its 2025 paper. Vendor numbers move a lot. Zep publicly disputed Mem0's Zep figure (RECALLED). LangMem is a LangChain OSS library (RECALLED).
- **Weakness:** on MemoryAgentBench, selective forgetting reaches at most 7% multi-hop across all systems (via 2606.01435, READ).

### C7. Temporal knowledge graph with invalidation: Zep/Graphiti, Mem0g, TEPA, deterministic freshness
- **Atomic unit:** an entity edge (fact) with `t_valid`, `t_invalid`, `t'_created`, `t'_expired` (READ).
- **Kept:** raw episodes, entities, facts, community summaries. **Discarded:** nothing. Old facts are invalidated.
- **Hierarchy:** three layers: episode, semantic entity, community (READ).
- **Update, drift, conflicts:** an LLM compares a new edge with semantically related edges, and code sets `t_invalid` (READ). TEPA revokes by key match (ABSTRACT). 2606.01435: LLM extraction, then code `max()` (READ).
- **Identity:** entity names are embedded (1024-d), searched by cosine plus BM25, and an LLM resolves duplicates and rewrites the summary (READ). Undoing a wrong merge is not addressed. Raw episodes are kept, so re-derivation is possible in principle.
- **Links:** bidirectional episode-to-fact indices, used for citation (READ).
- **Frame:** time only.
- **Reader:** retrieval by cosine, BM25 and BFS, with RRF, MMR or cross-encoder reranking. Returns a context string of facts with validity ranges, about 1.6K tokens (READ).
- **Training:** none.
- **Used where:** Zep is a commercial product, Graphiti is OSS (RECALLED for adoption).

### C8. Summary hierarchies: RAPTOR, GraphRAG, HippoRAG 2, Generative Agents
- **Atomic unit:** a text chunk, with summaries of clusters above it (RAPTOR); entity communities with summaries (GraphRAG); passages as nodes in an open knowledge graph ranked by Personalized PageRank (HippoRAG 2). All ABSTRACT.
- **Hierarchy:** the core idea. RAPTOR builds bottom-up by clustering and summarising, and retrieves across levels. GraphRAG answers "global" questions by map-reduce over community summaries.
- **Update:** mostly batch rebuilds. Generative Agents (RECALLED): a memory stream plus periodic reflection, retrieval scored by recency, importance and relevance.
- **Known vs unknown, identity, frame:** not addressed.
- **Numbers:** RAPTOR with GPT-4 +20% absolute on QuALITY. HippoRAG 2 +7% on associative tasks (both ABSTRACT).
- **Training:** none.
- **Used where:** GraphRAG is Microsoft OSS (RECALLED); the others are research code. Summaries are the place drift enters ("semantic drift through repeated summarisation", SSGM 2603.11768, SNIPPET only).

### C9. Structure plus refs plus detail on demand, outside robotics: Aider repo map, Playwright snapshots, language servers
- **Atomic unit:** a symbol and its signature (Aider); an accessibility node with a ref (Playwright).
- **Kept:** the skeleton. **Discarded:** bodies and pixels, which are fetched by reference.
- **Hierarchy:** the file and symbol tree. Playwright has a `depth` limit on snapshots (READ).
- **Update:** recomputed from the source of truth every turn, so it cannot drift.
- **Identity:** given by the source (symbol, DOM node).
- **Links:** the ref goes directly into action tools. Playwright `boxes` adds pixel boxes (READ).
- **Reader:** pushed map under a token budget (Aider ranks files by dependency graph; READ) or a pulled snapshot.
- **Cost:** Playwright MCP vs CLI token figures exist only in blogs (SNIPPET, not trusted).
- **Training:** none.
- **Used where:** shipped and widely used (Aider, Playwright MCP and CLI, LSP in every coding agent). Language servers as a query interface: RECALLED, no measurement found.

### C10. Learned memory controllers: MemCtrl, MemCompiler, EMem writer, VLA memories
- A trained gate or compiler decides what to keep. MemCtrl: about 16% average gain on EmbodiedBench (ABSTRACT). MemCompiler: up to +129% over no memory and 60% lower per-step latency (ABSTRACT). VLA memories (MemoryVLA, MEM by Physical Intelligence, HAMLET; from the Awesome-Memory-for-Robotics list) are latent.
- Out of scope for us (training). Their lesson: what gets selected should depend on the current state, not be injected once at episode start.

## 3. Research questions RQ4 to RQ13

- **RQ4, atomic block:** the family's units are caption plus pose plus time (ReMEmbR), fact sentence (Mem0), timestamped fact edge (Zep), object node with ref, box, confidence, freshness and image path (Thea), image plus object set (3D-Mem), and typed (entity, attribute, value, time, provenance) atom (ObsMem). The evidence favours keyed, typed atoms over free text: ObsMem beats Mem0 and A-Mem, and TEPA revokes by key.
- **RQ5, hierarchy:** summary trees (RAPTOR), communities (GraphRAG, Zep), collapse and expand under LLM control (SayPlan), semantic forest (Embodied-RAG), budgeted skeleton (Aider). No result shows hierarchy helping at single-object scale. At small size, pasting everything is competitive (H3).
- **RQ6, runtime update without drift:** invalidate, never delete (Zep, Mem0g, ObsMem). Revoke by key (TEPA). Writes gated by an evaluator (Thea). Recompute from the source of truth (Aider, Playwright). The drift sources named are repeated LLM summarisation and rewriting of old notes (A-MEM, SSGM) and stored mistakes reused (2505.16067).
- **RQ7, who maintains it:** four regimes are reported: LLM decides operations (Mem0, MemGPT, A-MEM); rules (ObsMem, TEPA, 2606.01435); hybrid, where the LLM detects and canonicalises and code applies (Zep, control-plane placement: best overall); trained policy (EMem, MemCtrl). Background maintenance by a separate agent exists (Letta sleep-time). The evidence says to split semantic matching (model) from ordering, comparison and deletion (an explicit rule). In 2606.01435 the split itself gives the gain; running the rule in code instead of in the model adds only 2 pp.
- **RQ8, contract:** in: raw episodes, kept non-lossy (Zep), or perception and execution outcomes (Thea). Out: a compact brief with handles plus typed query tools returning JSON. ReMEmbR: text, position, time. eMEM: 10 recall tools including concept-to-location. Thea: relations, image. RakuAI: distance, clearance, occupancy computed in code.
- **RQ9, what to preserve:** keep raw observations outside, with a pointer from every item (Zep episodes, Thea image path). Do not push raw images (EMem-Bench: 36.6% to 26.2%). Keep a few covering frames, not all (3D-Mem: about 11 of 40). Record interaction outcomes, whose absence is 31% of failures. Branch 5 owns the rest.
- **RQ10, how well known and not looked:** freshness and confidence (Thea). Epistemic labels plus observed vs reported (ObsMem). Frontier snapshots for unexplored space (3D-Mem). Abstention as a scored ability (LongMemEval, ABSTRACT). Nothing exposes angle, resolution or coverage to the model.
- **RQ11, identity and wrong merges:** Zep: embedding plus BM25 candidates, the LLM confirms. Thea: perception mints the ref, and the model disambiguates by fetching images. No system in this family detects or undoes a wrong merge. Keeping raw episodes is the only enabler seen. Branch 3 owns this.
- **RQ12, linking geometry, image and words:** ReMEmbR (ask by text, get a position; ask by position, get captions), eMEM (R-tree plus vector index behind one graph), Thea (ref to box, image, relations), 3D-Mem (image to objects). The pattern is one id with several indices.
- **RQ13, frame:** this family is nearly silent. Everything stores in the world or map frame. Thea records the frame in metadata. Translating into the asker's point of view is not addressed.

## 4. What does not transfer, and why

- **Conversation benchmarks and their memory systems are tuned to chat.** LoCoMo averages 9K tokens, which fits in context, and full context wins there. Entities arrive with names. Our details have no names until the vision agent coins one, so ids must be minted by code, as Thea's perception backend does.
- **Text-similarity deduplication (Zep, Mem0, A-MEM) will over-merge us.** Two stickers described as "white rectangular label" embed as near-identical. Position alone separates features only at about 3 cm (contract §7). Text similarity can propose candidates, but it cannot decide.
- **Freshness machinery assumes the world changes.** Ours is static: our conflicts are competing **readings** of the same thing (for example "VATI" vs "YAHBOOM"), not state changes. What transfers is the shape (versions with provenance, code picks by an explicit rule), with "better view quality" in place of "newer."
- **VLM-reader designs (3D-Mem, Mobility VLA, FindingDory baselines) assume the planner sees images.** Ours is blind. These designs fit the vision agent as the reader of snapshots, not the planner.
- **Trained controllers (EMem writer, MemCtrl, MemCompiler, VLA memories)** are excluded by the no-training rule.
- **Room and building hierarchies (SayPlan, Embodied-RAG, ReMEmbR time queries)** solve a scale we lack. For one object the whole digest may fit under about 1 to 2K tokens, and pushing it may beat pulling it (MIT small-graph result). Pull matters for images, raw readings and the multi-object future.

## 5. Ranked reading list for round 3

1. **Thea, arXiv 2608.11246** (https://arxiv.org/html/2608.11246; the last 44K characters are unread). Extract the brief format (Listing 9), the full tool list, the evaluator gating rules, and whether any ablation of brief vs no brief exists. It is the nearest working analogue to the IWM read path.
2. **Deterministic freshness, 2606.01435** (https://arxiv.org/html/2606.01435v1). Extract the extraction schema, the LongMemEval port (§5.7), and the failure analysis. It models "model refers, code computes" for conflicts.
3. **WorldLines / ObsMem, 2606.18847** (https://arxiv.org/html/2606.18847). Extract the observer-gate rules, the belief-label transitions, and the query-plan routing. Template for RQ10 and for rule-based writes.
4. **EmbodiedMemory-Bench, 2609.28236** (https://arxiv.org/html/2609.28236). Extract the EMem tool signatures (App. B.2, B.3), the failure taxonomy definitions, and why MIRIX and MemVerse fall below full context.
5. **Zep, 2501.13956** (https://arxiv.org/html/2501.13956). Extract the invalidation and dedup prompts and the returned context-string template.
6. **3D-Mem, 2411.17735** (https://arxiv.org/html/2411.17735). Extract the co-visibility selection (Alg. 1) as a recipe for "which few views show these details," and frontier snapshots as unknown regions.
7. **Control-plane placement, 2606.15903** (https://arxiv.org/abs/2606.15903). Extract the 13 configurations and the categories where each fails.
8. **ReMEmbR, 2409.13682** (https://arxiv.org/html/2409.13682). Extract the exact tool schemas and prompts, and confirm the units of the temporal error in Table I.
9. **SayPlan** (https://proceedings.mlr.press/v229/rana23a/rana23a.pdf; too large for the fetch tool, needs a local download). Verify the 82.1% figure and the semantic-search failure modes.
10. **eMEM, 2606.03374** (https://arxiv.org/abs/2606.03374). Extract the list of 10 recall tools, as a menu for the IWM query set.
11. **Anthropic memory tool docs** (https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool). Extract the API semantics, since our planner is Claude.
12. **TEPA, 2608.07429** (https://arxiv.org/pdf/2608.07429). Extract the key design and why append-only memory falls below no memory.

## 6. Gaps, open questions, not found

- **No controlled push-vs-pull ablation for an embodied LLM.** Thea pushes a brief and pulls details but does not ablate it. The only clean comparison remains MIT 2510.16643.
- **No study measures LLM-written summaries drifting from their source in a robot memory.** Drift claims are surveys (SSGM, SNIPPET) or chat settings.
- **No system detects or undoes a wrong merge.**
- **No LLM-facing memory exposes view quality** (angle, distance, resolution). The closest are freshness, confidence and epistemic labels.
- **Not opened or not verified:** OpenEQA EM-EQA baseline numbers (the paper PDF was too large; the landing page has no numbers); STMA, VeriGraph, CoELA, JARVIS-1, Voyager, Uni-NaVid and NaVILA (all RECALLED only); LangMem; the Gemini Robotics-ER 1.5 memory mechanism (SNIPPET: the orchestrator "memorizes the state" in some tasks, details not found); RoboMemory numbers (SNIPPET only); Mem0 vs Zep benchmark dispute (RECALLED).
- **Open for us:** if conflicts are competing readings, what is the code-side rule (view quality, vision-agent confidence, or keep both and ask)? Should a single-object digest be pushed whole every turn, with only evidence (images, raw readings) pulled by ref?
