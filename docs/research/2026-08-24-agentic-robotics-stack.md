# The Agentic Robotics Stack, Summer 2026 — Research Synthesis

Five parallel research sweeps (Google, NVIDIA, open-source frameworks,
practitioner stacks, cognitive architectures), 2026-08-24. Full transcripts in
the session log; this is the distilled synthesis with sources.

## Headline

**Nobody ships the cognition layer.** Every vendor and framework covers a
slice; the layer between "frontier model" and "robot that inspects things" is
unclaimed territory, and the people closest to it say so themselves —
DeepMind's own systematic study of hierarchical robot agents (arXiv:2606.10267)
concludes the field "lacks unified design principles."

## The consensus brain decomposition (what everyone converges on)

Across RoboOS (BAAI), Gemini-ER (DeepMind), CoELA descendants, and the 2025-26
surveys, the same seven modules recur:

1. **Perception tools** — open-vocab detectors/VLMs as callable functions, not a fixed pipeline
2. **Spatial world model** — hierarchical 3D scene graph (Hydra→ConceptGraphs→Clio), exposed to the LLM via SayPlan-style collapse/expand text queries; contested by snapshot (3D-Mem) and Gaussian (GaussMemory) alternatives
3. **Memory** — episodic/semantic/procedural + working, with spatial memory as a robot-specific fourth tier; write-gated (surprise-gating, 2606.03787)
4. **Cognitive core / orchestrator** — LLM/VLM doing decomposition, tool selection, progress estimation
5. **Verifier** — generate-then-verify is now the default; unverified free-form plans are legacy
6. **Skill library ("cerebellum")** — named executable skills / VLA policies invoked as tools
7. **Reflection loop** — failures distilled offline into memory/rules (Reflexion-style)

This is the classic 3T architecture reborn with an LLM in the deliberative seat.

## Vendor reality check

- **NVIDIA**: models (GR00T N1.7 GA, Cosmos-Reason2-8B — fits the 5090 at its
  32 GB line) + sim (Isaac 6.0, official MCP server; Lab-Arena for policy
  eval) + data agents. **No planner→robot orchestration product.** Their VSS
  inspection blueprint watches video, commands nothing.
- **Google**: `gemini-robotics-er-2-preview` is the closest thing to a
  cognitive architecture as an API — orchestrator brain + blocking tool calls
  into declared motion primitives + model-level safety (ASIMOV-Agentic). Open
  access, $2/$10 per M. VLA line trusted-tester only. **ADK: zero robotics
  surface — skip.** Two model shutdowns + one breaking API change in 2026:
  budget quarterly migration; `generateContent` is now legacy, Interactions
  API is the front door.
- **Open source**: RAI (Robotec) — only actively-shipped ROS2 embodied agent
  framework, real warehouse-inspection deployment; ROSClaw — only one with a
  pre-execution safety envelope; ros-mcp-server (1.4k★) — the de-facto
  LLM↔ROS wire; openpi π0.5 — the open VLA with UR5e in its pretraining mix.
  Each holds one organ; none is the body.

## The build-vs-adopt verdict (both schools agree)

- **Cognition loop: ADOPT.** FAEA (arXiv:2601.20334): an *unmodified* Claude
  Agent SDK loop hits 85-96% on sim manipulation, matching few-shot VLAs.
  Anthropic's own eval: capability is dominated by **choice of control
  abstraction**, not model — the same model is useless at torque level and
  strong supervising policy-level tools. Practitioners converged on raw
  SDK/MCP + tools; LangGraph is absent from real robot builds.
- **Embodiment harness: BUILD, THIN.** The five components everyone hand-rolls
  because nothing ships them:
  1. two-rate contract (LLM ~0.1-1 Hz over a deterministic 100+ Hz executor;
     watchdogs, staleness, preemption)
  2. spatial state store for LLMs (what is where, how confident, how stale)
  3. safety gate as code between tool call and actuator (prompt-level safety
     fails in 49-73% of vulnerable tasks)
  4. **evidence ledger** — every claim bound to sensor captures + an
     independent verifier verdict ("for inspection, this ledger IS the product")
  5. sim-gated execution (dry-run in a twin with identical tool schemas)
- **Skills: ADOPT** (GR00T/π0.5/SmolVLA via LeRobot) when motion beyond
  viewpoints is needed.

> Own the control flow that touches the robot; rent everything above it.

## Where MAY-186/187 sits in this landscape

- **Closest prior art: AP-VLM** (arXiv:2409.17641) — VLM proposes the next
  viewpoint on a 3D grid over the workspace, Franka + UR5, iterate until
  confident. Our viewsphere + AI decider is the same family. Read before
  building the orchestrator; differentiate on: evidence discipline, benchmark
  with physical ground truth, question-generality, coverage-based absence.
- **We sit on an open research question**: whether "decide where to look"
  belongs to the orchestrator (interpretable, slow, discrete) or inside the
  policy (ActiveVLA — fast, opaque). The field is squeezing the orchestrator
  option from both sides but names it "the only interpretable option at the
  behaviour layer." Our bet is the interpretable side — for inspection, where
  the justification is deliverable, that is arguably the *right* side.
- **The known unsolved problems our project directly touches**:
  - truthful self-monitoring (models hallucinate success — our bench's
    hallucination metric is exactly this)
  - what the world model stores (our grid+coverage text is a minimal instance
    of the "spatial state for LLMs" gap)
  - latency economics (>70% of embodied-agent runtime is waiting on the
    planner, AgenticCache — our views-to-answer metric is also a cost metric)
- **Organ map** (what our inspection MVP already implements of the consensus):
  evidence ledger ≈ RunStore + transcripts + trust-tier writers (built);
  perception tools ≈ verbs_local + inspect subagent (built); spatial state ≈
  grid/coverage + action rendering (in design); verifier ≈ readout bench (in
  design); reflection ≈ the offline improvement cycle (in design); safety
  gate + two-rate contract ≈ loop-v2 RTDE guards (embryonic).

## What to steal, concretely

1. **SayPlan's collapse/expand interface** for our action-space rendering —
   present the compact state, let the model expand task-relevant detail.
2. **RAI's Agent/Connector/Tool decomposition + execution logging** as the
   reference shape for the harness layer.
3. **Generate-then-verify** as the default for any plan the orchestrator emits.
4. **VSS blueprint's everything-is-an-MCP-microservice pattern** for exposing
   our tools when the framework generalizes beyond one repo.
5. **DFKI's separate goal critic** (arXiv:2602.13081) — never trust the
   model's verbal success claim; verify externally. (Independently our bench
   design.)

## Strategic conclusion for the framework ambition

Build it — but the way durable frameworks are built: **extracted from a
working system, not designed in the abstract**. The inspection MVP is the
working system; the harness organs it grows (evidence ledger, spatial state
text, safety gate, bench, reflection loop) are the framework. When a second
robot or second task forces generalization, extract. The field's gap analysis
reads like our roadmap — that is the strongest possible signal the mastery
path and the project are the same work.
