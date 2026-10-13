# Project report — interview notes

Context collected before writing. Anton's answers, lightly structured, his wording kept.
Report state: as at the 22 Sep 2026 demo. First person (+ passive where natural).
Standalone — no references to other repo docs; anything that must be referenced is cited
inside the report itself.

## Settled structure

1. Introduction — the problem, one sentence on ADAM, what this report is
2. Scope of the MVP — what we took on, what we deliberately left out
3. How the system works — short, so the later path makes sense to a reader
4. The path — the design bets and pivots that shaped the system
5. What was achieved — the runs; what was and was not measured
6. Limitations
7. Insights — what we learned about the problem
8. Next steps

Working title (not yet confirmed): "AI-driven inspection on a UR5e — project report".

## 1. Introduction

**Problem statement (Anton):** Given ANY object and ANY question about it, the robot
should understand, plan and execute the movements that gather the information needed
to answer the question, and collect the evidence that proves the answer is correct.

**Framing — human vs robot inspector (Anton's words):** hand a human inspector any
object and he knows what to do with it, how to adapt, where to check — but humans are
slow and hard to maintain. Robot inspection is fast and relatively easy to maintain,
but it is usually fixed to a single object or program and not robust. The idea is to
bring robot efficiency together with the robustness of human-like cognition to reach
autonomous AI inspection.

**ADAM:** one sentence only, in the introduction — the system is inspired by the
existing product ADAM; here we work on an even higher level of autonomy in inspection.
Albert knows ADAM; no explanation.

**Evidence in the problem statement:** mostly for the human who must trust the answer.
It started as a feature, but because every answer card always carries it, it became
part of the solution. The system was built gradually, so some features are logically
part of the product.

## 2. Scope

**Phase-one success criterion:** make it work qualitatively, and give the demo audience
the feeling "okay, this is reasonable, it might actually work". The good demo effect
was the success criterion: it works and it might have a future. Quantitative evaluation
is the next phase.

**Scope boundaries — set on day one:** single object on a table; questions answerable
from the object's visible surface; no training or fine-tuning of any model.

**Scope boundaries — emerged during the build:** cognition in cloud models, detect /
segment / OCR run locally; geometry-aware evidence views for box-shaped objects only,
for now; opaque objects (stereo physics).

(Operator approval gate: Anton — "remove human gate" from the scope list. It is a safety
mechanism described under How the system works, not a scope boundary.)

**Deliberately left out:**
- VLAs, for now — motion planning and execution use traditional methods and that
  machinery exists and works.
- Simulation — everything was developed in the real world.
- A continuously moving robot that decides in real time. Instead: a step-based plan
  over view positions around the object. The problem was framed as a view-selection
  decision problem rather than a robot-motion problem.

Nothing else was deliberately left out.

## 3. How the system works

**Anton's spine (use as the section's opening paragraph):** the system reconstructs the
3D object being inspected and generates the best set of viewpoints to inspect it. The
AI agent picks the viewpoint for the next capture. That image is analysed by the vision
agent, which collects all the details, answers the question given to that vision pass,
and the inspection state is updated with the findings. With one loop finished we have
more information, and the loop repeats until the agent decides it has enough to answer
the starting question.

**The one thing readers must take away:** the main orchestrator is the "top-tier
manager". It never says exactly what to do and never sees raw data: it has commands to
give, and reads structured reports written over the raw data. It never outputs a pose
or a joint angle and never sees an image — that is what makes this not a VLA and not a
vision model with a robot attached.

**Depth per subsystem:** plan/move = brief (classic motion planning, not interesting
here); rays/viewpoints = the closer look. Order follows the demo deck
(oficial/docs/visuals/decks/inspection-demo.html): Agentic systems → Harness → Two
agents → Four actions → One step of the loop → Motion → Motion & safety → Viewsphere →
Vision capture → Vision agent → Evidence views → The answer → Dataset → Limits.

**Figures:** reuse the deck figures; inspect each together while composing — some may
be slop. **Names/numbers:** decide together while composing; few names, it is a PoC.

## 4. The path

**The two real bets (Anton):** (1) a general agent on the Claude Agent SDK plus a thin
harness, instead of a hand-rolled loop or a VLA; (2) the orchestrator reads text only,
a vision agent reads one image. Everything else "would be the same system, slightly
different" — not a bet.

**Where the work went:** the viewpoints, and how to get the robot to those points. The
motion-planning algorithm was stable once built; it only needed the proper sequence to
plan — the ray hop/slide ("ray bouncing") is the important step there. The LLM–VLM
communication was fine from day one; behaviour was tuned so angles seen in the image map
correctly onto the menu. (My compiled pivot list — object soft→hard, cross hull, cuRobo→
OMPL, corridor hint, retreat-before-refit, RTDE, D405 res — is NOT the path section's
shape. At most a compact aside.)

**The ray story (Anton's words, keep):** the open question was how to select the
distance. The MVP camera is quite bad: far from the object, text and details are not
readable. Experiments were done, but computing a fixed distance per cell on the
viewsphere needed a lot of parameters and still did not give the satisfying result,
namely the closest possible view for each cell. So instead of a capture viewpoint with
3D coordinates, a view cell became a RAY from the object centre: "when we say we want
some view, we refer to the whole ray." Motion planning to a ray is easier — hop onto any
point along the ray, then zoom in along it until the closest point the guards allow.
Result: the closest views, and many motion-planning problems escaped by giving the
motion more structure.

**The geometry story (Anton's words, keep):** the whole idea of intelligent inspection
is to inspect based on the inspected object. Today the system only offers view
selection and reasoning conditioned on the object; we also want to GENERATE the set of
views from this specific object's geometry. Several approaches were tried; we decided
to restrict ourselves to the cylinder and box primitives and found a solution that
works well for that — but it is not scalable, and this design decision will probably be
reconsidered and built upon. We need the geometry to generate the proper views.
(Check at draft time: box is implemented; cylinder is the decided-next primitive.)

**Dates:** yes — readers should see how fast this was built; an impressive demo in one
month. Week-resolution dates: 12 Aug definition · 18 Aug first commit in inspection/ ·
~25 Aug agent loop · September build month · 22 Sep demo. 196 commits 18 Aug–22 Sep.

## 5. What was achieved

**The demo day (Anton; run facts to be reconstructed from the archive):** first the red
object — count the dots on it. Then experiments with paper and multiple objects:
scenarios the system was not built for. Albert was impressed and curious what the thing
is actually capable of.

**Albert's reaction (report as a reaction, not as a claim):** he admitted he was
surprised that the system could take two images from two cells, see three dots on
both, and still count them as the same three dots rather than six — "the capability of
the LLM with tools of AI vision runs".

**Capability, in Anton's words (the achieved result):** we can now put any object in
the office on the table and ask questions about it; the robot collects images around
it, thinks, and answers — without tuning, without a human helping it, without
preparation. Out of the box: any object, a question, an answer. That was the idea, and
the PoC solves this task. (Draft note: the operator approval gate is safety, not help —
phrase so it does not contradict "no human in the loop".)

**Numbers:** none. They do not add value to this report.

**Not measured:** nothing is really measured — those are next-phase questions. What we
have: a draft of how the system might look, insights on what works and what does not,
a feeling for the capability, and readiness to plan the next phase with the lessons
from the PoC.

**Demo-day runs, from the archive (inspection/data/runs/2209-*, read-only; 20 runs):**
- 12:19–13:35 — AI KIT box "what is the VBTI serial number?" → AAB0007-0001 (demo1 5 min, demo4 6 min);
  small camera box (D3 DesignCore) "complete spec" → P/N, REV, S/N, config read from the x- face (16 min, 9 steps);
  big camera box (RealSense L515) "what are the serial numbers?" → B/N, product code, MM# (22 min, 11 steps);
  fanless PC "complete spec" → I/O per face, DC 12-24 V, no serial plate reachable (13 min).
- 14:14–14:58 — "is there a logo on the cup?" ×3 with the PC on the table: survey declared
  "NOT a cup but an industrial mini PC"; runs aborted / survey unplannable / not approved.
  PC spec retries aborted (demo7/8/9; demo9 brand read varied per view: Seeed/CoolTrig/CropTeq/Neousys/OnLogic).
- 15:23–16:02 — AI KIT again (8 min, answered); red ball "colour of the dot on top" → yellow,
  confirmed from opposite azimuths [1,2] and [7,2] "ruling out a lighting artifact" (3 min);
  "how many green dots on the side?" → 3 (24 min, 10 steps; per-cell counts 3/3/3/1/0/0/0/3 merged
  as ONE cluster near az 60° — answer.json reasoning: "it read as lower-right from az 30°, lower-left
  from az 90°, edge-on from az 150°, absent from 210/240/300"; the plan had said "avoid double-counting
  the same dot seen from overlapping views"); sticky note "could you solve the equation?" ('11 x 3 =',
  aborted — cells failed to plan); "which object is bigger red or blue?" (survey saw only the ball,
  aborted); "which object is bigger?" (blue Intel Wi-Fi box found at [1,0]; answer "red sphere is bigger"
  REJECTED by the evidence hunt — hunt saw "a 3D-printed red plastic spheroid … instead of a foam sphere";
  run never finished).
- 21 Sep had 19 runs (rehearsal). run.json "object" is "box" in every run including the ball.
- Albert's "same three dots" = demo13 (2209-compute-ray-demo13/ai/000/answer.json).

## 6. Limitations

**The warning to Albert (Anton's words, keep):** we talked a lot with Albert about 3D
space, voxels and so on. But we should understand that the point cloud (PLY) is barely
used today — we work at the primitive level and with key measures like the centroid.
Starting to work with voxels and the real point clouds means going from structured 3D
to unstructured 3D. It makes sense, it is the right decision, but spatial reasoning is
still an unsolved problem: we do not yet know how to navigate and make decisions with
3D objects as the input.

**The list — "not candidates, the real problems we have seen" (keep all):**
(a) small text is misread; the model reconstructs characters with no sign of doubt ·
(b) the camera: far views unreadable, images sometimes blurry · (c) box-shaped objects
only for geometry-aware views · (d) text is the only channel to the orchestrator —
untested for anomaly/defect tasks whose evidence may not survive being described ·
(e) ~9 minutes per question, model latency the largest share · (f) one object on the
table; multi-object scenarios tried on demo day did not complete · (g) opaque, textured
surfaces only — passive stereo, depth bad on untextured surfaces · (h) unreachable cells
and refused moves cost turns · (i) a person approves every move.
(My recommendation stands, unconfirmed: keep (a) and (b) separate — model vs camera;
"nothing measured" lives in Achieved, not here.)

**Build-limit vs approach-limit:** not answered by Anton; my tentative split (approach
= (d) only) to be confirmed when drafting the section.

**Camera:** no option to change it for now; maybe later.

**Negative surprise:** nothing — everything worked as intended. The one exception is the
camera: the quality of the 3D reconstruction from its depth, and depth on untextured
surfaces. "The camera is quite bad."

**Open:** which demo-day block was with Albert in the room (12:19 or 15:23)?

## 7. Insights

**My ten candidates: accepted ("the insights are supported").** Anton's own, on top:

1. **Agentic systems in robotics are the next big thing.** The work is writing the tools,
   the robot integration, the monitoring, the perception logic — and building the agent
   from them; cloud or local. The agentic harness is what we need.
2. **The three-layer protocol — raw → structured → raw.** A raw image is read by the
   vision subagent and handed to the manager LLM as structure; the manager uses the menu
   to send a tool call that becomes raw robot motion. It works, but this protocol is
   speculative and lossy. The options run from improving the protocol under strict A/B
   testing to changing the architecture so that it all happens in latent space — which
   needs an end-to-end scenario (VLA training from the collected data, or sim). A
   decision is needed here. We are only starting to get a feeling for spatial reasoning;
   it will be crucial knowledge in the next few years.
3. **Object geometry and view planning have many degrees of freedom.** The most
   restricted end is invariant views (the fixed sphere); the least restricted is views
   completely determined by the object being inspected. We want the latter; it has many
   challenges.

**Spatial reasoning (Q3):** genuinely unsolved in the state of the art — no model really
understands 3D and knows how to plan from it; some robotics models try, without much
progress yet. The physical world is under-represented in every domain the large AI
models are trained on.

**Where the intelligence sits (Q4, Anton's words, keep):** in the main orchestrator
session. We need better state representation — and to think whether it is even
feasible, and through which protocols a complex world (relations, what is seen from
which position) can be represented to a sequential textual model so it can reason over
spatial features: where a feature lies on the object, what is close to it, where to
look. To some extent the LLM captures this, but "it feels like a blind man painting —
it kind of knows what is there from the description, but does not really understand it
at the level we might at some point" (qualitative conclusion from observation). The
research needs to be done.

Q5 (which block Albert attended): dropped by Anton.

## 8. Next steps

**Status:** to be defined with Albert — there is a lot of room. After the report is
finished, Anton starts digging into all directions.

**Quantitative evaluation of THIS system: no.** It is a raw prototype; even at this
stage Anton felt lost; the system is too restricted and there is not much that
evaluation could improve. (SUPERSEDES the round-2 line "next phase = quantitative
evaluation". Sections 2 and 5 must read: phase one proved it qualitatively; the next
phase is structural.)

**The structural directions — "spatial intelligence in the wide sense":**
1. Navigation and decision-making in voxel space (Albert's direction).
2. Approaching VLA models, with sim-enabled end-to-end training.
3. Keeping the same system but developing a data structure that captures the details,
   their relations, the spatial structure of the object, maps — interpretable to the
   language model so it can reason over it.

**Why structural (Anton's words, keep):** the current system does not have the
innovation capacity to scale; improvement and polishing will not make it good enough.
Other innovations must happen for this product to become really valuable.

**Data:** everything recorded is available to work with; it may be transformed once we
know exactly what is needed. From the beginning the data was collected future-ready,
because the next stages — eval, bench or train — will use it.

## Interview complete — open items for composing
- Title (working): "AI-driven inspection on a UR5e — project report".
- Limitations: build-limit vs approach-limit split — confirm while drafting.
- Figures: inspect each with Anton; some may be slop.
- Names/numbers: decide together; few.
