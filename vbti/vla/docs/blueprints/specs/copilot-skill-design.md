# Copilot Skill — Design Spec

**Date:** 2026-04-02
**Status:** Draft
**Location:** `~/.claude/skills/copilot/`
**Knowledge base:** `~/Documents/Obsidian Vault/vbti/copilot/`

---

## 1. What This Is

A session-long thinking partner for the VBTI robotics project. Knows the full pipeline (3D recon → Isaac Sim → data collection → VLA training → real robot), all experiments, research from GTC and papers, and current gaps. Grows smarter as the user feeds it content.

**Not** a replacement for train-copilot, dataset-copilot, or hardware-copilot. Those handle execution. This handles *thinking* — what to try, why, what the research says, what's missing.

## 2. Activation

Explicit invocation only. User starts a session with `/copilot`. The skill loads project context and becomes the strategic advisor for that session.

Trigger keywords in description: "copilot", "thinking partner", "what should I try", "strategy", "research", "what's next", "experiment planning", "sim-to-real strategy", "project status".

## 3. File Structure

### Skill files (how it thinks)

```
~/.claude/skills/copilot/
├── SKILL.md                ← Session workflow, inbox protocol, behavior rules
├── project_context.md      ← Static: pipeline overview, hardware, envs, key code paths
└── decision_framework.md   ← When to use SmolVLA vs GR00T, BC vs RL, real vs sim, retrain vs fine-tune
```

### Knowledge files (what it knows)

```
~/Documents/Obsidian Vault/vbti/copilot/
├── index.md              ← Master map with [[wikilinks]] to all knowledge
├── inbox.md              ← User drops links + thoughts between sessions
├── gaps.md               ← Known unknowns, ranked by impact on project
├── experiments/
│   └── duck_cup_v001_v010.md
├── papers/
│   └── gtc_2026_synthesis.md
└── decisions/
    └── smolvla_vs_groot.md
```

## 4. Skill Files — Content Spec

### SKILL.md

YAML frontmatter:
- `name: copilot`
- `description:` Thinking copilot for VBTI robotics project. Strategic advisor for experiment planning, research synthesis, sim-to-real decisions, and knowledge management. Use when starting a thinking session, asking "what should I try next", processing research inbox, or discussing project strategy. Activates on: copilot, strategy, what's next, experiment planning, research, sim-to-real, project status, inbox, knowledge gap.

Body sections:

**Identity** — You are a thinking partner, not an executor. You help the user decide *what* to do and *why*. For *how*, defer to the domain-specific copilots (train-copilot, dataset-copilot, hardware-copilot).

**Session Start Protocol:**
1. Read `vbti/copilot/index.md`
2. Read `vbti/copilot/gaps.md`
3. Read `vbti/copilot/inbox.md`
4. Skim 2-3 most recent daily notes at `vbti/sessions/`, check the claude sessions if needed.
5. Deliver briefing: where we left off, top gaps. Max 10 lines, bullets.

**Session Modes:**

- **Inbox processing** — User says "let's process the inbox." Walk through each entry: fetch content (use youtube-transcript skill for videos, read files for local docs), extract project-relevant knowledge, propose a summary. User guides: "focus on X", "connects to Y", "skip." Write structured knowledge file in the appropriate subdirectory. Update `index.md` and `gaps.md`. Mark item as processed.

- **Strategy** — User asks "what should I try next?" or similar. Read experiments + gaps + research knowledge. Propose 2-3 prioritized next steps with reasoning grounded in both project state and research. Present trade-offs. User decides.

- **Experiment logging** — User reports experiment results. Ask targeted follow-up questions (what changed, what surprised, what failed). Write experiment summary to `copilot/experiments/`. Update `gaps.md` — did this close a gap or open new ones?

- **Deep dive** — User asks about a specific topic ("explain the sim-to-real gap for my setup"). Answer by synthesizing research knowledge + project-specific context. Reference specific files, experiments, GTC findings.

- **Stuck mode** — User is stuck. Strategic debugging: not code-level, but approach-level. "Have you tried X? The GTC tutorial showed Y. Your v010 results suggest Z."

**Session End Protocol:**
1. Update `index.md` with any new knowledge files created
2. Update `gaps.md` if priorities shifted
3. Offer to draft a daily note entry for `vbti/sessions/DD-MM-YYYY.md`

**Writing Rules:**
- All output to Obsidian uses `[[wikilinks]]`, `## Sections`, `- [ ] tasks`, code blocks
- Write in user's voice: casual, first person, stream of consciousness
- Show drafts — user approves before writing
- Append to existing files by default, new sections only when told

**Inbox Format:**
```markdown
## Inbox

### [date] — [short description]
- **Link:** [url]
- **My thoughts:** [user's notes about why this matters]
- **Status:** pending | processing | done
```

### project_context.md

Static reference the copilot reads to understand the project. Updated rarely (when major infrastructure changes).

Contents:
- **Pipeline overview** — Video → COLMAP → GS (MILo) → mesh → USD → Isaac Sim → teleoperation → HDF5 → LeRobot → VLA training → inference
- **Hardware** — SO-ARM101 (6-DOF + gripper, Feetech STS3215), 4× RealSense D405 (top/left/right/wrist), local RTX 4070 Ti SUPER (16GB), remote RTX 5090 @ 10.11.100.151
- **Training backends** — SmolVLA (500M, LeRobot, primary), GR00T N1.6 (3B, NVIDIA, emerging)
- **Conda envs** — lerobot (py3.12), groot (py3.10), isaac (py3.11), gsplat-pt25 (py3.11)
- **Key code paths** — Pointers to vbti/logic/ subdirectories, entry points, config locations
- **Current task** — Duck-cup pick-place, SO-ARM101, 4-camera setup
- **Data** — LeRobot v2 parquet format, datasets at robotics/datasets/, models at robotics/models/

### decision_framework.md

Decision trees for common strategic questions:

- **SmolVLA vs GR00T** — SmolVLA: faster iteration, proven on our hardware, 500M lighter. GR00T: better zero-shot from Cosmos backbone, 3B heavier, post-training validated on SO-101 with 20-40 demos at GTC.
- **BC vs RL** — BC first (we're here), RL hardening after BC policy is reasonable (canonical Stage 3). BC alone hits ~90% in-distribution ceiling.
- **Real vs sim data** — Sim for volume + randomization, real for grounding. Mix is the answer. Current gap: no real data yet.
- **Retrain vs fine-tune** — Fine-tune when adding data to same task. Retrain when changing architecture, action space, or camera setup.
- **When to move on from current experiment** — If 3+ versions show same failure mode, it's a structural issue (data, architecture, or pipeline), not hyperparameter.

## 5. Knowledge Files — Initial Content

### index.md

Seeded from existing research:
- Link to `papers/gtc_2026_synthesis.md` (from today's GTC transcript processing)
- Link to `experiments/duck_cup_v001_v010.md` (from memory + experiment notes)
- Link to `decisions/smolvla_vs_groot.md`
- Pointers to existing Obsidian content at `vbti/infra/models/sim_real_train research/`

### gaps.md

Seeded from `00_MASTER_KNOWLEDGE.md` action items:
1. **No RL hardening** (Stage 3 missing) — highest impact, blocks generalization beyond training distribution
2. **No real robot data yet** — blocks sim-to-real validation
3. **GR00T not tested on our task** — GTC showed 20-40 demos sufficient on SO-101
4. **Limited domain randomization** — only object pose, no lighting/texture/physics variation
5. **Collision mesh not optimized** — triangle mesh limits parallel env scaling

### inbox.md

Empty — ready for user to start adding links.

### experiments/duck_cup_v001_v010.md

Synthesized from memory files + experiment notes:
- Versions tested, configs, results
- Key findings: loss ≠ quality, scheduler scaling, domain boundary stickiness
- What each version taught us

### papers/gtc_2026_synthesis.md

Migrated + refined from `00_MASTER_KNOWLEDGE.md`:
- Canonical 4-stage pipeline
- Sim-to-real gap analysis
- GR00T post-training specifics
- RL hardening strategy
- Scaling with convex hull collisions

### decisions/smolvla_vs_groot.md

First architectural decision record:
- Context, options considered, decision (SmolVLA primary, GR00T next experiment), reasoning

## 6. Future Upgrade Path

### Phase 2: MindKeg MCP (when knowledge > ~50-100 files)

When `index.md` becomes unwieldy and Claude can't read everything at session start:
- Install: `npx mindkeg-mcp init`
- Each time copilot writes a knowledge file, also store atomic learnings in MindKeg
- Session start: semantic search MindKeg → read only relevant Obsidian files
- Obsidian stays source of truth, MindKeg is the search index

### Phase 3: Knowledge graph (if Phase 2 feels limiting)

Cognee or Dragon Brain for relationship traversal between concepts. Only if "what connects X to Y?" becomes a frequent question that flat files can't answer.

## 7. Boundaries

- Does NOT write code. Defers to domain copilots for execution.
- Does NOT make architectural decisions. Presents options with trade-offs. User decides.
- Does NOT auto-process inbox. Waits for user to start a session and say "let's process."
- Does NOT push to git or modify project code. Only writes to Obsidian vault.
- Knowledge files are Obsidian-native: wikilinks, frontmatter, clean markdown.
