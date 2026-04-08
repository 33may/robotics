# Copilot Skill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a session-long thinking copilot skill for the VBTI robotics project — strategic advisor + growing knowledge base in Obsidian.

**Architecture:** Skill files at `~/.claude/skills/copilot/` define behavior (SKILL.md, project_context.md, decision_framework.md, codebase_reference.md). Knowledge files at `~/Documents/Obsidian Vault/vbti/copilot/` store evolving project knowledge (index, gaps, inbox, experiments, papers, decisions). Skill reads both at session start.

**Tech Stack:** Claude Code skills (SKILL.md + reference files), Obsidian markdown with wikilinks, youtube-transcript skill for ingestion.

**Key paths:**
- Skill dir: `~/.claude/skills/copilot/`
- Knowledge dir: `~/Documents/Obsidian Vault/vbti/copilot/`
- Existing research: `~/Documents/Obsidian Vault/vbti/infra/models/sim_real_train research/`
- Experiment notes: `/home/may33/projects/ml_portfolio/robotics/vbti/experiments/`
- Existing skills (reference pattern): `~/.claude/skills/train-copilot/`
- Memory files: `~/.claude/projects/-home-may33-projects-ml-portfolio-robotics/memory/`

---

## Task 1: Create Skill Directory + SKILL.md

**Files:**
- Create: `~/.claude/skills/copilot/SKILL.md`

- [ ] **Step 1: Create the skill directory**

```bash
mkdir -p ~/.claude/skills/copilot
```

- [ ] **Step 2: Write SKILL.md**

Create `~/.claude/skills/copilot/SKILL.md` with the following content:

```markdown
---
name: copilot
description: Thinking copilot for VBTI robotics project. Strategic advisor for experiment planning, research synthesis, sim-to-real decisions, and knowledge management. Use when starting a thinking session, asking "what should I try next", processing research inbox, discussing project strategy, or reviewing experiment results. Activates on: copilot, strategy, what's next, experiment planning, research, sim-to-real, project status, inbox, knowledge gap, what should I try, stuck, generalization, RL hardening.
---

# Copilot

You are a thinking partner for the VBTI robotics project — a sim-to-real manipulation pipeline using VLA models (SmolVLA, GR00T) on SO-ARM101. You help the user decide *what* to do and *why*. For *how* to execute, defer to the domain copilots: train-copilot, dataset-copilot, hardware-copilot.

You are not the architect — the user is. Present options with trade-offs. The user decides.

## Knowledge Locations

All paths are absolute. Read these at session start.

| What | Where |
|------|-------|
| Master index | `~/Documents/Obsidian Vault/vbti/copilot/index.md` |
| Known gaps | `~/Documents/Obsidian Vault/vbti/copilot/gaps.md` |
| Research inbox | `~/Documents/Obsidian Vault/vbti/copilot/inbox.md` |
| Experiment summaries | `~/Documents/Obsidian Vault/vbti/copilot/experiments/` |
| Paper/video knowledge | `~/Documents/Obsidian Vault/vbti/copilot/papers/` |
| Decision records | `~/Documents/Obsidian Vault/vbti/copilot/decisions/` |
| GTC research (raw) | `~/Documents/Obsidian Vault/vbti/infra/models/sim_real_train research/` |
| Daily session notes | `~/Documents/Obsidian Vault/vbti/sessions/` |
| Project context | Read `project_context.md` in this skill directory |
| Decision framework | Read `decision_framework.md` in this skill directory |
| Codebase & CLI ref | Read `codebase_reference.md` in this skill directory |

## Session Start Protocol

1. Read `copilot/index.md` — what you know
2. Read `copilot/gaps.md` — what's missing
3. Read `copilot/inbox.md` — pending items
4. Skim 2-3 most recent daily notes at `vbti/sessions/`. Check claude session logs if needed.
5. Deliver briefing: where we left off, top gaps. Max 10 lines, bullets.

## Session Modes

### Inbox Processing

User says "let's process the inbox." For each entry:

1. Fetch content — use `youtube-transcript` skill for YouTube links, Read tool for local files
2. Read the content, extract what's relevant to this project
3. Propose a structured summary — show draft to user
4. User guides: "focus on X", "connects to Y", "skip this"
5. Write knowledge file to appropriate subdirectory (`papers/`, `experiments/`, `decisions/`)
6. Update `index.md` — add `[[wikilink]]` to the new file
7. Update `gaps.md` — did this close a gap or reveal a new one?
8. Mark inbox item status as `done`

### Strategy

User asks "what should I try next?" or similar.

1. Read `experiments/` for what's been tried and learned
2. Read `gaps.md` for ranked priorities
3. Read relevant papers/research for evidence
4. Check `codebase_reference.md` to verify what tools/modules exist for each option
5. Propose 2-3 prioritized next steps with reasoning
6. Ground each recommendation in both project state AND research findings
7. For each option, note which domain copilot + CLI tools would execute it
8. Present trade-offs clearly. User decides.

### Experiment Logging

User reports experiment results.

1. Ask targeted follow-up questions: what changed from last version? what surprised you? what failed?
2. Draft experiment summary — show to user for approval
3. Write to `copilot/experiments/<experiment_name>.md` (update if exists, create if new)
4. Update `gaps.md` — did this close a gap or open new ones?
5. Update `index.md` if new file created

### Deep Dive

User asks about a specific topic (e.g., "explain the sim-to-real gap for my setup").

1. Read relevant knowledge files (papers/, decisions/, research/)
2. Check `codebase_reference.md` for relevant modules, CLI tools, or docs
3. Synthesize an answer combining research knowledge + project-specific context
4. Reference specific experiments, GTC findings, numbers, and point to relevant code/docs
5. If knowledge is insufficient, flag the gap and suggest adding it to inbox

### Stuck Mode

User is stuck on something.

1. Understand the block — ask one clarifying question
2. Check if research addresses this (GTC findings, papers)
3. Check if past experiments give evidence
4. Suggest approach-level strategies (not code-level fixes)
5. Example: "v010 shows BC ceiling at 90%. The GTC tutorial confirmed RL hardening solves this. Should we prioritize implementing the reward function?"

## Session End Protocol

1. Update `index.md` with any new knowledge files created this session
2. Update `gaps.md` if priorities shifted
3. Offer to draft a daily note entry for `vbti/sessions/DD-MM-YYYY.md`

## Writing Rules

All output to Obsidian vault:
- Use `[[wikilinks]]` for cross-references between knowledge files
- Use `## Sections`, `- [ ] tasks`, code blocks
- Write in user's voice: casual, first person, stream of consciousness
- Show drafts in backtick blocks — user approves before writing
- Append to existing files by default, new sections only when user says

## Inbox Format

When the user adds items to `inbox.md`, they follow this format:

```markdown
### [date] — [short description]
- **Link:** [url]
- **My thoughts:** [user's notes about why this matters]
- **Status:** pending | processing | done
```

## Knowledge File Template

When creating new knowledge files in `papers/` or `experiments/`:

```markdown
---
created: YYYY-MM-DD
source: [url or experiment name]
tags: [relevant tags]
---

# [Title]

## Key Findings
- Finding 1
- Finding 2

## Relevance to Our Project
[How this connects to VBTI pipeline, current gaps, next steps]

## Links
- [[related_file_1]]
- [[related_file_2]]
```

## Future: MindKeg MCP Integration

When the knowledge base grows beyond ~50-100 files and `index.md` becomes unwieldy:
- Install: `npx mindkeg-mcp init`
- Store atomic learnings in MindKeg alongside Obsidian files
- Session start: semantic search MindKeg first → read only relevant Obsidian files
- Obsidian stays source of truth, MindKeg is the search index
```

- [ ] **Step 3: Verify skill is valid markdown with correct frontmatter**

```bash
head -5 ~/.claude/skills/copilot/SKILL.md
```

Expected output:
```
---
name: copilot
description: Thinking copilot for VBTI robotics project...
---
```

- [ ] **Step 4: Commit**

```bash
cd ~/.claude/skills && git init 2>/dev/null; echo "Skill created at ~/.claude/skills/copilot/SKILL.md"
```

No git repo here — skill lives outside the project. Just verify the file exists.

---

## Task 2: Create project_context.md

**Files:**
- Create: `~/.claude/skills/copilot/project_context.md`

- [ ] **Step 1: Write project_context.md**

This is a static reference file. Content is sourced from the exploration report and memory files. Create `~/.claude/skills/copilot/project_context.md`:

```markdown
# Project Context

Static reference for the copilot. Updated when major infrastructure changes.

## Pipeline

```
iPhone Video → COLMAP (SfM) → Gaussian Splatting (nerfstudio) → MILo (mesh extraction)
→ USD Scene Asset → Isaac Sim (composition + physics) → LeIsaac Task
→ Teleoperation (leader → follower) → Multi-camera HDF5
→ LeRobot v2 Parquet → VLA Training (SmolVLA / GR00T) → Inference → Real Robot
```

Orchestrator: `vbti/logic/reconstruct/master.py`

## Hardware

- **Robot**: SO-ARM101 — 6-DOF + gripper, Feetech STS3215 servos, leader-follower teleop
- **Cameras**: 4× RealSense D405 (top, left, right, wrist) — 640×480 @ 15fps
- **udev**: `/dev/cam_top`, `/dev/cam_left`, `/dev/cam_right`, `/dev/cam_gripper`
- **Local GPU**: RTX 4070 Ti SUPER (16GB, SM 89)
- **Remote GPU**: RTX 5090 @ `vbti@10.11.100.151` — nightly PyTorch 2.12, `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64`

## Training Backends

### SmolVLA (primary)
- 500M params, flow-matching diffusion action head
- LeRobot trainer, batch 64, bf16, cosine LR decay
- 50-action chunks, frozen vision encoder
- Env: `lerobot` (Python 3.12)

### GR00T N1.6 (emerging)
- 3B params, Eagle VLM + 32-layer DiT action head
- Post-training: 20-40 demos, 20k steps (~7hrs), 16-step action horizon
- Needs `modality.json` matching cameras/embodiment
- Env: `groot` (Python 3.10)

## Current Task

Duck-cup pick-place on SO-ARM101. Tabletop manipulation with 4-camera setup.
- DR: poses (±0.21m X, ±0.32m Y, ±180° rot), lighting, camera jitter, physics
- Dataset: 144 episodes (merged 01+02 black), LeRobot format
- Best result: ~90% in-distribution (v010 ckpt 015k-035k), <10% out-of-distribution

## Key Code Paths

| Module | Entry Point |
|--------|-------------|
| 3D Pipeline | `vbti/logic/reconstruct/master.py` |
| Training | `vbti/logic/train/engine.py` |
| Experiments | `vbti/logic/train/experiment_utils.py` |
| Monitoring | `vbti/logic/train/monitor.py` |
| Remote Training | `vbti/logic/train/remote.py` |
| Dataset Tools | `vbti/logic/dataset/check_utils.py`, `convert_utils.py` |
| HDF5 Inspection | `vbti/logic/dataset/hdf5_utils.py` |
| Real Inference | `vbti/logic/inference/run_real_inference.py` |

## Conda Environments

| Env | Python | Purpose |
|-----|--------|---------|
| lerobot | 3.12 | SmolVLA training/inference (primary) |
| groot | 3.10 | GR00T N1.6 training |
| isaac | 3.11 | Isaac Sim |
| gsplat-pt25 | 3.11 | MILo / Gaussian Splatting |

## Data Locations

- Datasets: `/home/may33/projects/ml_portfolio/robotics/datasets/`
- Models: `/home/may33/projects/ml_portfolio/robotics/models/`
- Experiments: `/home/may33/projects/ml_portfolio/robotics/vbti/experiments/`
- Scene assets: `/home/may33/projects/ml_portfolio/robotics/vbti/data/`
```

- [ ] **Step 2: Verify file exists and is readable**

```bash
wc -l ~/.claude/skills/copilot/project_context.md
```

Expected: ~75 lines

---

## Task 3: Create decision_framework.md

**Files:**
- Create: `~/.claude/skills/copilot/decision_framework.md`

- [ ] **Step 1: Write decision_framework.md**

Content sourced from `skill_context.md` section I and `knowledge.md`. Create `~/.claude/skills/copilot/decision_framework.md`:

```markdown
# Decision Framework

Use these frameworks when the user asks strategic questions. Present the relevant framework, apply it to their current situation, and let them decide.

## SmolVLA vs GR00T

| Factor | SmolVLA | GR00T N1.6 |
|--------|---------|-------------|
| Params | 500M | 3B |
| Data needed | 50-100+ episodes | 20-40 demos |
| Training time | Variable (10k-50k steps) | ~7hrs (20k steps) |
| Control | Full (LR, scheduler, layers) | Post-training only |
| Proven on our hardware | Yes (v001-v010) | Tutorial uses SO-101 |
| Multi-frame | Last frame only | Temporal via delta_indices |

**When to choose SmolVLA**: More data available, need fine-grained training control, iterating on hyperparams.
**When to choose GR00T**: Few high-quality demos, want faster iteration, leveraging foundation model generalization.
**Best approach**: Train both on same eval, compare.

## BC vs RL

**Always BC first** — establishes initial policy from demonstrations.

**Add RL when**: BC policy plateaus (our v010: 90% in-dist, <10% out-of-dist), need recovery from novel states, want to scale via sim randomization.

**RL requirements**:
- Reward function: `distance(duck, cup) + grasp_bonus + success_bonus`
- Parallel envs: 1000+ (need convex hull collisions)
- DR across all axes during RL training
- PPO or SAC optimizer

**The canonical pattern** (from 9 GTC talks): BC pretrain → RL harden across 1000s of randomized sim envs → deploy on real robot.

## Real vs Sim vs Mixed Data

| Source | Provides | Scale | When to use |
|--------|----------|-------|-------------|
| Real teleop | Ground truth dynamics, real visuals | Limited (10s-100s eps) | Initial policy, grounding |
| Sim RL/teleop | Closed-loop RL, DR, edge cases | Massive (1000s+ eps) | RL hardening, scaling |
| Cosmos augment | Visual diversity (lighting, texture) | Orders of magnitude | After sim pipeline works |

**Current gap**: No real robot data collected yet. 144 episodes are sim-only.

## Retrain vs Fine-tune

- **Fine-tune from checkpoint**: More data, same distribution, same task
- **Fine-tune from base**: New environment, same robot (our v010 approach)
- **Retrain from scratch**: New task, new robot, new architecture, or new action space

## When to Move On from an Experiment

If 3+ consecutive versions show the same failure mode, it's structural — not a hyperparameter problem. Look at:
1. Data distribution (rest-pose bias, limited position coverage)
2. Architecture (action representation, camera setup)
3. Pipeline stage (missing RL hardening)

Evidence: v005-v010 all show same generalization failure → structural (need RL hardening or more diverse data), not hyperparameter.

## Generalization Failure Triage

When a policy fails to generalize:

1. **Check distribution coverage** — Does training data cover the test region? (v010: no)
2. **Check action representation** — Absolute positions anchor to training distribution. Delta actions may generalize better (trade-off: less stable)
3. **Check camera hypothesis** — Does wrist cam provide pose-agnostic features? (v010: no, hypothesis failed)
4. **Check for data bias** — Rest-pose bias? Clustered demos? (v010: yes, rest-pose)
5. **Consider RL hardening** — BC can't recover from unseen states by definition

## Priority Ranking (current)

From `gaps.md` — update when priorities shift:
1. RL hardening (Stage 3) — highest impact
2. Real robot data — blocks sim-to-real validation
3. GR00T evaluation — may outperform SmolVLA with fewer demos
4. Domain randomization expansion — lighting, materials, camera intrinsics
5. Collision mesh optimization — prerequisite for RL at scale
```

- [ ] **Step 2: Verify file exists**

```bash
wc -l ~/.claude/skills/copilot/decision_framework.md
```

Expected: ~85 lines

---

## Task 4: Create Obsidian Knowledge Directory + index.md

**Files:**
- Create: `~/Documents/Obsidian Vault/vbti/copilot/index.md`
- Create: `~/Documents/Obsidian Vault/vbti/copilot/experiments/` (directory)
- Create: `~/Documents/Obsidian Vault/vbti/copilot/papers/` (directory)
- Create: `~/Documents/Obsidian Vault/vbti/copilot/decisions/` (directory)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p ~/Documents/Obsidian\ Vault/vbti/copilot/{experiments,papers,decisions}
```

- [ ] **Step 2: Write index.md**

Create `~/Documents/Obsidian Vault/vbti/copilot/index.md`:

```markdown
# Copilot Knowledge Index
> Master map of everything the copilot knows. Updated each session.

## Research

- [[gtc_2026_synthesis]] — 9 GTC talks synthesized: canonical pipeline, RL hardening, GR00T on SO-101, convex hull scaling, sim-to-real gap
- Raw transcripts + analysis at `vbti/infra/models/sim_real_train research/`

## Experiments

- [[duck_cup_v001_v010]] — SmolVLA duck-cup: 10 versions, loss≠quality, scheduler scaling, generalization failure at distribution boundary

## Decisions

- [[smolvla_vs_groot]] — SmolVLA as primary, GR00T next experiment. Reasoning: proven pipeline vs foundation model generalization.

## Gaps

See [[gaps]] for ranked list of known unknowns.

## Inbox

See [[inbox]] for pending research items.
```

- [ ] **Step 3: Verify directory structure**

```bash
find ~/Documents/Obsidian\ Vault/vbti/copilot/ -type f -o -type d | sort
```

Expected:
```
.../copilot/
.../copilot/decisions/
.../copilot/experiments/
.../copilot/index.md
.../copilot/papers/
```

---

## Task 5: Create gaps.md

**Files:**
- Create: `~/Documents/Obsidian Vault/vbti/copilot/gaps.md`

- [ ] **Step 1: Write gaps.md**

Seeded from `knowledge.md` §10 action items + `skill_context.md` §B known problems. Create `~/Documents/Obsidian Vault/vbti/copilot/gaps.md`:

```markdown
# Knowledge Gaps
> Ranked by impact on project progress. Updated each copilot session.

## Critical — Blocks Next Milestone

### 1. No RL Hardening (Stage 3 Missing)
- **Impact**: BC ceiling at ~90% in-dist, <10% out-of-dist. Every successful deployment in GTC talks uses RL.
- **What's needed**: Reward function, convex hull collisions, 1000+ parallel envs, PPO/SAC integration
- **Evidence**: [[duck_cup_v001_v010]] — v005-v010 all show same failure mode
- **Research**: [[gtc_2026_synthesis]] §6

### 2. No Real Robot Data
- **Impact**: Can't validate sim-to-real transfer
- **What's needed**: Collect real teleop episodes, compare policy performance sim vs real
- **Blocker**: Need working inference on real hardware first

## High — Next Sprint

### 3. GR00T Not Tested on Our Task
- **Impact**: May outperform SmolVLA with 3.6x fewer demos needed (40 vs 144)
- **What's needed**: Verify `_build_modality_config()`, run post-training, compare
- **Research**: [[gtc_2026_synthesis]] §4 — tutorial uses our exact robot

### 4. Limited Domain Randomization
- **Impact**: Policy overfits to narrow visual/physics distribution
- **Missing axes**: Material properties, camera intrinsics, articulated scene elements
- **Current**: Object poses, lighting intensity/angle, camera jitter, physics params

### 5. Collision Mesh Not Optimized
- **Impact**: Triangle mesh limits to 10-100 parallel envs/GPU. Convex hulls → 10,000+.
- **What's needed**: V-HACD or CoACD decomposition of duck + cup meshes
- **Prerequisite for**: RL hardening at scale

## Watch — Future

### 6. Newton Physics Engine
- Differentiable, GPU-accelerated, Disney + DeepMind co-developed
- Switch from PhysX when it exits beta

### 7. GR00T N2 / DreamZero
- 2x generalization vs current VLAs on novel tasks. End 2026.

### 8. GS-in-USD
- Eliminate MILo mesh extraction step. Spec landed, tooling not ready.
```

---

## Task 6: Create inbox.md

**Files:**
- Create: `~/Documents/Obsidian Vault/vbti/copilot/inbox.md`

- [ ] **Step 1: Write inbox.md**

Create `~/Documents/Obsidian Vault/vbti/copilot/inbox.md`:

```markdown
# Research Inbox
> Drop links + thoughts here between sessions. Process together with copilot.

<!-- Format:
### [date] — [short description]
- **Link:** [url]
- **My thoughts:** [why this matters]
- **Status:** pending | processing | done
-->
```

---

## Task 7: Create experiments/duck_cup_v001_v010.md

**Files:**
- Create: `~/Documents/Obsidian Vault/vbti/copilot/experiments/duck_cup_v001_v010.md`

- [ ] **Step 1: Write experiment summary**

Synthesized from memory files (`training_experiments.md`) and experiment notes (`vbti/experiments/duck_cup_smolvla/v010/notes.md`, `experiment.md`). Read these files first, then create `~/Documents/Obsidian Vault/vbti/copilot/experiments/duck_cup_v001_v010.md`:

```markdown
---
created: 2026-04-02
source: vbti/experiments/duck_cup_smolvla/
tags: smolvla, duck-cup, bc, generalization
---

# Duck Cup SmolVLA — v001 to v010

## Overview

10 versions of SmolVLA fine-tuning on duck-cup pick-place task. Progressive learning about BC limitations, scheduler behavior, and generalization boundaries.

## Key Findings

### Loss ≠ Policy Quality
- v004: train loss 0.004, aggressive cosine decay over 10k steps → poor real performance
- v006: train loss 0.041, gentle decay over 30k steps → better policy
- **Lesson**: Don't chase low train loss. Eval on real robot is the only reliable metric.

### Scheduler Auto-Scaling
- When `total_steps < num_decay_steps` (30k), SmolVLA auto-scales:
  - 10k steps → scale_factor=0.333, warmup 1000→333, aggressive decay
  - 30k steps → no scaling, full warmup, gentle decay
- Same peak LR produces very different training dynamics depending on step count.

### Domain Boundary Stickiness (v010)
- 90%+ success in training distribution area
- <10% success outside — gripper literally stuck in data collection bounds
- Gripper camera did NOT provide pose-agnostic features (hypothesis failed)
- Tested checkpoints: 5k, 10k, 15k, 20k, 25k, 30k, 35k, 40k, 45k, 50k

### Side Grab Difficulty
- Model overshoots duck on right side approach
- Learned to lean back for close objects (good visual reasoning)
- With enough attempts, completes task but pushes duck out of success boundary

### Rest-Pose Bias
- Real teleop data clusters around resting position
- Model defaults to rest config regardless of input in some cases
- Partial fixes: dataset trimming (v003-v004), but structural issue remains

## Version Summary

| Version | Dataset | Steps | Key Change |
|---------|---------|-------|------------|
| v001-v004 | 50-100 eps | 10k-50k | Initial experiments, LR tuning |
| v005-v009 | Various | Various | Data variations, camera experiments |
| v010 | 144 eps (merged 01+02 black) | 50k | Best attempt — exposed BC ceiling |

## Conclusion

BC alone hits a ceiling at ~90% in-distribution. The failure mode is structural (not hyperparameter): policy can't recover from states outside demo distribution. Next step: RL hardening ([[gaps#1. No RL Hardening (Stage 3 Missing)]]).

## Links
- [[gtc_2026_synthesis]] — confirms RL hardening is the canonical solution
- [[smolvla_vs_groot]] — GR00T may need fewer demos for comparable performance
- [[gaps]] — generalization failure is gap #1
```

---

## Task 8: Create papers/gtc_2026_synthesis.md

**Files:**
- Create: `~/Documents/Obsidian Vault/vbti/copilot/papers/gtc_2026_synthesis.md`

- [ ] **Step 1: Write GTC synthesis**

Migrate and restructure from `~/Documents/Obsidian Vault/vbti/infra/models/sim_real_train research/knowledge.md`. Read that file, then create `~/Documents/Obsidian Vault/vbti/copilot/papers/gtc_2026_synthesis.md`:

```markdown
---
created: 2026-04-02
source: 9 GTC 2026 videos (see Sources below)
tags: gtc, sim-to-real, rl-hardening, groot, newton, convex-hull
---

# GTC 2026 — Synthesized Knowledge

## Canonical 4-Stage Pipeline

Every successful physical AI deployment follows this:

1. **Pretrain** — Foundation model on internet video + human demos [done: SmolVLA base, GR00T N1.6]
2. **Post-train** — Task-specific BC on real + sim demonstrations [partial: real data pipeline, sim pipeline built]
3. **RL Harden** — RL across 1000s of randomized sim envs [MISSING: our #1 gap]
4. **Deploy + Loop** — Real robot → failures → new sim data → retrain [MISSING: no automation]

## Sim-to-Real Gap

Three components (Lightwheel):
1. **Physics solver accuracy** — PhysX vs real dynamics (kinematic gap is solved per ABB)
2. **Asset quality** — collision meshes, friction, mass, articulation
3. **Parameter distribution** — sim params must span real-world variation

Frontier problems: lighting, camera intrinsics, material properties.

Strategy: "Digital cousins, not digital twins" — 5-10 variants per object covering the distribution. Policy learns task structure, not specific geometry.

## GR00T on SO-101

Most actionable finding. Tutorial uses our exact robot.
- 20-40 demos for post-training, 20k steps, ~7hrs
- Needs `modality.json` matching cameras/embodiment
- Resume gotcha: increase `max_steps` or trains zero additional steps
- We have 144 eps (3.6x what's needed)

## RL Hardening

Pattern: BC policy → Isaac Lab (1000s parallel envs) → PPO/SAC → recovers from novel states.

Requirements for our project:
- Reward: `distance(duck, cup) + grasp_bonus + success_bonus`
- Convex hull collisions (10,000+ envs/GPU vs 10-100 with triangle mesh)
- DR across all axes during RL

## Asset Pipeline Evolution

| Now | Future |
|-----|--------|
| GS → MILo mesh → USD | GS → USD directly (particle fields schema) |
| Manual physics params | SimReady SDK validation |
| Triangle mesh collision | Convex hull decomposition (V-HACD/CoACD) |

## Key Numbers

| Metric | Value |
|--------|-------|
| GR00T post-training demos | 20-40 |
| GR00T 20k steps | ~7 hrs |
| Convex hull parallel envs | 10,000+/GPU |
| Triangle mesh parallel envs | 10-100/GPU |
| Isaac Lab 3.0 FPS | 150k (manipulation) |
| Production bar | >99% success, >600 UPH |

## Sources

| # | Video | Key takeaway |
|---|-------|-------------|
| 1 | Jensen Huang Keynote (full) | Physical AI vision, Cosmos 3, GR00T N2 |
| 2 | Keynote Highlights | Condensed announcements |
| 3 | Dev Livestream | Newton engine, Isaac Lab 3.0 |
| 4 | GR00T Post-Training SO-101 | ★ Most actionable — our exact robot |
| 5 | Lightwheel Sim Factory | ★ Digital cousins, physics benchmarking |
| 6 | Physical AI Keynote | Data Factory blueprint |
| 7 | Digital Twins Panel | ABB/Siemens/Agile production insights |
| 8 | Dassault World Models | FoundationPose, industry world models |
| 9 | OpenUSD Physical AI | ★ GS-in-USD, SimReady SDK |

Raw transcripts: `vbti/infra/models/sim_real_train research/transcripts/`

## Links
- [[duck_cup_v001_v010]] — our BC experiments confirming need for RL
- [[smolvla_vs_groot]] — GR00T evaluation planning
- [[gaps]] — RL hardening is gap #1
```

---

## Task 9: Create decisions/smolvla_vs_groot.md

**Files:**
- Create: `~/Documents/Obsidian Vault/vbti/copilot/decisions/smolvla_vs_groot.md`

- [ ] **Step 1: Write decision record**

Create `~/Documents/Obsidian Vault/vbti/copilot/decisions/smolvla_vs_groot.md`:

```markdown
---
created: 2026-04-02
status: active
tags: architecture, smolvla, groot
---

# Decision: SmolVLA vs GR00T

## Context

Two VLA backends available for duck-cup task. Need to decide allocation of effort.

## Options Considered

### A. SmolVLA Only
- Proven pipeline (v001-v010), full training control
- But: hit BC ceiling at ~90% in-dist, 500M params may limit generalization

### B. GR00T Only
- 3B params, foundation model generalization, only 20-40 demos needed
- But: less training control (post-training only), unproven on our specific setup

### C. SmolVLA Primary, GR00T Next (chosen)
- Continue SmolVLA for RL hardening experiments (proven pipeline)
- Run GR00T post-training in parallel as comparison point
- Compare both on same eval protocol

## Decision

Option C. SmolVLA stays primary because the pipeline is proven and RL hardening is the next structural improvement (not model swap). GR00T runs as parallel experiment — if it matches SmolVLA with 40 demos and no RL, it becomes the new primary.

## Evidence

- [[duck_cup_v001_v010]] — SmolVLA works but hits BC ceiling
- [[gtc_2026_synthesis]] §4 — GR00T tutorial uses SO-101, 20-40 demos, ~7hrs
- GR00T N2 (end 2026) may make current comparison moot — but can't wait

## Review Trigger

Revisit after GR00T post-training results are in.
```

---

## Task 10: Create codebase_reference.md

**Files:**
- Create: `~/.claude/skills/copilot/codebase_reference.md`

- [ ] **Step 1: Write codebase_reference.md**

This is the copilot's map of the entire codebase — every CLI tool, module, doc, and domain copilot. The copilot uses this to know *where things are* and *what tools exist* when advising on strategy. It doesn't duplicate full CLI docs (those live in domain copilots), but gives enough to make informed recommendations.

Create `~/.claude/skills/copilot/codebase_reference.md`:

```markdown
# Codebase Reference

Map of all CLI tools, modules, documentation, and domain copilots. Use this to ground strategic advice in what actually exists in the codebase.

## Project Root

`/home/may33/projects/ml_portfolio/robotics/`

## Domain Copilots — Defer Execution Here

When the user needs to *do* something, point them to the right copilot:

| Copilot | Skill Name | When to Defer |
|---------|------------|---------------|
| Training | `train-copilot` | Starting training, configuring experiments, monitoring runs, analyzing results |
| Dataset | `dataset-copilot` | Inspecting datasets, converting formats, checking cameras, comparing distributions |
| Hardware | `hardware-copilot` | Servo ops, camera setup, calibration, teleoperation, real robot inference |

Each copilot has its own `cli_reference.md` with full command docs. Don't duplicate — just point the user to the right copilot.

## Module Map

### Reconstruction Pipeline (`vbti/logic/reconstruct/`)

| Module | CLI Entry | Purpose |
|--------|-----------|---------|
| `master.py` | `python -m vbti.logic.reconstruct.master <stage>` | Pipeline orchestrator: video_processing → gs_reconstruction → ply_to_usda → scene_composition → export_isaaclab_task |
| `video_utils.py` | via master | Frame extraction, rotation fix, sharp frame selection |
| `colmap_utils.py` | via master | SfM reconstruction (COLMAP + nerfstudio) |
| `gs_milo_utils.py` | via master | Gaussian Splatting training + MILo mesh extraction |
| `format_utils.py` | via master | PLY→USD conversion (sRGB, PCA alignment, physics properties) |
| `robot_utils.py` | `python -m vbti.logic.reconstruct.robot_utils <cmd>` | USD robot: inspect, fix_base, set_drives, make_ready |
| `isaac_cfg_utils.py` | via master | LeIsaac/IsaacLab env code generation from USD scenes |
| `cosmos_transfer.py` | `python -m vbti.logic.reconstruct.cosmos_transfer <cmd>` | Sim data augmentation (extract, process, config, transfer) — transfer step incomplete |
| `clean_mesh.py` | `python vbti/logic/reconstruct/clean_mesh.py` | Interactive Polyscope mesh cleaner |

### Training (`vbti/logic/train/`)

| Module | CLI Entry | Purpose |
|--------|-----------|---------|
| `engine.py` | `python -m vbti.logic.train.engine train [opts]` | Unified training loop (SmolVLA + GR00T) |
| `experiment_utils.py` | `python -m vbti.logic.train.experiment_utils <cmd>` | Experiment lifecycle: create, use, active, lse, lsv, status, version, config, dir, complete, log, summary |
| `config_utils.py` | `python -m vbti.logic.train.config_utils <cmd>` | Config management: schema, default, create, show. Dotted overrides for any field. |
| `monitor.py` | `python -m vbti.logic.train.monitor <cmd>` | Training monitoring: snapshot, logs, trend, compare, history, metrics |
| `remote.py` | `python -m vbti.logic.train.remote <cmd>` | Remote RTX 5090 training: train, status, logs, pull, push_data |
| `backends/smolvla.py` | via engine | SmolVLA training backend |
| `backends/groot.py` | via engine | GR00T N1.6 training backend |

### Dataset (`vbti/logic/dataset/`)

| Module | CLI Entry | Purpose |
|--------|-----------|---------|
| `check_utils.py` | `python -m vbti.logic.dataset.check_utils <cmd>` | ls, info, cameras, report, compare_actions, compare_trimmed, trajectories |
| `convert_utils.py` | `python -m vbti.logic.dataset.convert_utils <cmd>` | discover, convert (HDF5→LeRobot), verify, link, roundtrip_test |
| `hdf5_utils.py` | `python -m vbti.logic.dataset.hdf5_utils <cmd>` | info, view, report, tree, episodes |
| `replay_utils.py` | `python -m vbti.logic.dataset.replay_utils <cmd>` | replay (on real robot), show (print actions) |
| `trim_utils.py` | via dataset tools | Trim rest-pose frames from episodes |

### Inference (`vbti/logic/inference/`)

| Module | CLI Entry | Purpose |
|--------|-----------|---------|
| `run_smolvla_inference.py` | `python vbti/logic/inference/run_smolvla_inference.py` | SmolVLA inference in Isaac Sim |
| `run_real_inference.py` | `python vbti/logic/inference/run_real_inference.py <cmd>` | Real robot inference: run, preview |

### Hardware Scripts (`vbti/scripts/` and `vbti/logic/`)

| Script | Purpose |
|--------|---------|
| `vbti/logic/cameras/check_usb.py` | RealSense health check: serials, firmware, USB speed, topology |
| `vbti/logic/cameras/reset_camera.py` | Hardware reset cameras |
| `vbti/logic/cameras/view_cameras.py` | Live camera grid (pyrealsense2) |
| `vbti/logic/servos/scan_all.py` | Scan Feetech bus for servo IDs |
| `vbti/logic/servos/factory_reset_motors.py` | Factory reset with collision avoidance |
| `vbti/logic/servos/load_calibration.py` | Load calibration to servo EEPROM |

### LeRobot CLI (global tools, conda `lerobot` env)

| Command | Purpose |
|---------|---------|
| `lerobot-find-port` | Discover USB serial ports |
| `lerobot-find-cameras realsense` | Find RealSense cameras with serials |
| `lerobot-find-cameras opencv` | Find OpenCV cameras with device paths |
| `lerobot-teleoperate` | Start teleoperation |
| `lerobot-record` | Record dataset or run policy on real robot |
| `lerobot-dataset-viz` | Visualize dataset in browser |

## Documentation Index

All at `/home/may33/projects/ml_portfolio/robotics/vbti/docs/`:

| Document | What it covers |
|----------|---------------|
| `project_knowledge_base.md` | Full project context: vision, pipeline, business context, research direction, stakeholders |
| `module_reference.md` | Per-file function index for all vbti modules |
| `pipeline_processes.md` | Step-by-step pipeline operations |
| `domain_randomization.md` | DR config, axes, current coverage |
| `hardware_setup.md` | Physical hardware setup guide |
| `cosmos_transfer_guide.md` | Cosmos Transfer 2.5 pipeline guide |
| `research/smolvla_training_internals.md` | SmolVLA architecture deep dive |
| `research/groot_training_internals.md` | GR00T N1.6 architecture deep dive |
| `research/camera_handling_comparison.md` | SmolVLA vs GR00T camera handling |
| `sessions/explore_isaac.md` | Isaac Sim exploration notes |
| `sessions/leisaac_train_inference.md` | LeIsaac training + inference session |

## Data Locations

| What | Where |
|------|-------|
| Datasets (LeRobot parquet) | `robotics/datasets/` |
| Trained models | `robotics/models/` |
| Experiment configs + checkpoints | `robotics/vbti/experiments/` |
| Scene assets (USD, meshes, HDRI) | `robotics/vbti/data/` |
| Robot USD | `robotics/robots/SO-101/` |
| Calibration files | `~/.cache/calibration/` |
| LeIsaac framework | `robotics/leisaac/` |
| LeRobot (vendored) | `robotics/lerobot/` |

## Config Structure (Training)

```yaml
model:
  type: smolvla|groot
  pretrained: "lerobot/smolvla_base"  # or nvidia/GR00T-N1.6-3B
  chunk_size: 50                       # action horizon
  freeze_vision_encoder: true
dataset:
  sources:
    - repo_id: "eternalmay33/dataset_name"
      weight: 1.0
      source: sim|real|synthetic|mixed
      role: both|train|val
  cameras:
    names: [top, left, right, gripper]   # ORDER MATTERS for both models
    remap: {cam_top: top}
training:
  steps: 10000
  batch_size: 4
  lr: 1e-5
  decay_lr: 2.5e-6
  warmup_steps: 500
  scheduler: cosine|wsd
  bf16: true
logging:
  log_freq: 100
  save_freq: 1000
  val_freq: 500
  wandb_enabled: false
```

## Experiment Directory Structure

```
vbti/experiments/{name}/
├── experiment.md          # hypothesis, goal
├── base_config.yaml       # starting config
├── compare.md             # cross-version comparison
└── v001/
    ├── config.yaml        # frozen complete config
    ├── run.md             # auto: status, timing
    ├── notes.md           # reasoning + eval observations
    ├── metrics/
    │   ├── training_log.jsonl
    │   └── summary.json
    ├── checkpoints/
    │   ├── best/ final/ step_XXXXXX/
    └── eval/
        ├── sim_results.json
        └── videos/
```

## Key Conversion & Data Flow

```
Collection:  Isaac Sim (radians) → convert → dataset (degrees, [-100,100] body, [0,100] gripper)
Training:    dataset (degrees) → normalize (MEAN_STD) → model (normalized tensors)
Inference:   model output → postprocessor (degrees) → radians → env.step()
```

Gripper is special: [0, 54] sim degrees → [0, 100] real degrees. NOT symmetric like body joints.

## Remote Training (RTX 5090)

- Host: `vbti@10.11.100.151:22`
- Env: `/home/vbti/anton/env` (uv, NOT conda)
- PyTorch: nightly 2.12.0.dev+cu128 (Blackwell sm_120)
- Required: `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64`
- `num_workers: 0` causes GPU starvation → use 8+
- Dataset `eternalmay33/08-merged` is local-only → must rsync before training
- CLI: `python -m vbti.logic.train.remote {train|status|logs|pull|push_data}`
```

- [ ] **Step 2: Verify file exists and line count**

```bash
wc -l ~/.claude/skills/copilot/codebase_reference.md
```

Expected: ~200 lines

---

## Task 11: Verify Complete Skill

- [ ] **Step 1: Verify skill directory**

```bash
ls -la ~/.claude/skills/copilot/
```

Expected: `SKILL.md`, `project_context.md`, `decision_framework.md`, `codebase_reference.md`

- [ ] **Step 2: Verify knowledge directory**

```bash
find ~/Documents/Obsidian\ Vault/vbti/copilot/ -type f | sort
```

Expected:
```
.../copilot/decisions/smolvla_vs_groot.md
.../copilot/experiments/duck_cup_v001_v010.md
.../copilot/gaps.md
.../copilot/inbox.md
.../copilot/index.md
.../copilot/papers/gtc_2026_synthesis.md
```

- [ ] **Step 3: Verify SKILL.md frontmatter parses correctly**

```bash
head -4 ~/.claude/skills/copilot/SKILL.md
```

Expected:
```
---
name: copilot
description: Thinking copilot for VBTI robotics project...
---
```

- [ ] **Step 4: Verify wikilinks are consistent**

Check that every `[[wikilink]]` in the knowledge files points to an existing file:
- `[[gtc_2026_synthesis]]` → `papers/gtc_2026_synthesis.md` ✓
- `[[duck_cup_v001_v010]]` → `experiments/duck_cup_v001_v010.md` ✓
- `[[smolvla_vs_groot]]` → `decisions/smolvla_vs_groot.md` ✓
- `[[gaps]]` → `gaps.md` ✓
- `[[inbox]]` → `inbox.md` ✓

- [ ] **Step 5: Test skill visibility**

Start a new Claude Code session in the robotics project and check if `copilot` appears in the available skills list. The skill should trigger when the user says `/copilot` or mentions strategy/research keywords.
