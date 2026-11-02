# Architecture Foundation — Offline Work Packet

> For Anton's ~2h solo block. Answer the decisions, draft the 3 deliverables, draw the 3 diagrams.
> When tokens are back, I assemble your inputs into the standardized files (AGENTS.md, architecture.md,
> ADRs, path-scoped rules, CI gate). **You produce the content/decisions; I produce the machinery.**

---

## What we're building (the foundation)

A stable, agent-legible foundation so both Claude Code and Codex navigate this repo reliably as it scales.
Six artifacts, derived from July-2026 best-practice research (see Appendix):

| Artifact | Who drafts | Purpose |
|---|---|---|
| `docs/architecture/architecture.md` | **You** (I shape) | The map: ports-and-adapters, the map agents read to navigate |
| `docs/architecture/adr/NNNN-*.md` | **You** (I format) | One record per hard decision + *why* (invariance, glide, comm convention…) |
| `AGENTS.md` (root, ~100-150 lines) | I write from your inputs | Source-of-truth steering file (commands, structure, boundaries, links out) |
| `CLAUDE.md` (root, ~2 lines) | I write | Shim: `@AGENTS.md` import so Claude Code reads the same file |
| `.claude/rules/{brain,world}.md` | I write | Path-scoped rules that load only when editing that area |
| CI gate + invariance test | I write | Deterministic enforcement of the one rule markdown can't guarantee |

Your offline job = the top two rows + the diagrams + the 8 decisions. Everything else I derive from those.

---

## PART 1 — The 8 architecture decisions (only you can make these)

Each is phrased so your answer becomes doc/ADR content. I give **my read** as a starting point — accept, reject, or rewrite. Bullet answers are fine; I'll shape prose.

### D1. The brain/world boundary — what exactly crosses the line?
The single most important decision. Define the *contract surface*: the exact messages each direction.
- **World → Brain:** observations + camera frames.
- **Brain → World:** commands (joint targets and/or base-velocity glide).
- **My read:** `RobotObs` (joint pos/vel, IMU, base pose) + `CameraFrame[]` inbound; `RobotCmd` (joint targets) *or* `GlideCmd` (base velocity) outbound. Boundary sits at **SDK-canonical arrays**, not raw per-sim structs.
- **Answer produces:** the contract table (Deliverable C) + architecture.md §Boundary.

### D2. The invariance contract — what may the brain *assume* about any world?
Invariance ("brain never knows which world") only holds if you pin the canonical assumptions every world must satisfy:
- **Coordinate frame** convention (which axis-up, base frame origin)
- **Units** (rad, m, m/s — confirm)
- **Joint order** (SDK order — we already know this bites; pin it explicitly, cite oli-corpus `sdk_joint_order`)
- **Rates** (control Hz, frame Hz — e.g. 30 Hz frame sub-tick per current cameras work)
- **My read:** these five are the invariance contract; a violation is a *test failure*, not a doc note.
- **Answer produces:** ADR-0001 (Invariance) + the CI test I write.

### D3. Where does world-specific processing live — brain-side adapter or world-side?
Your diagram drew "World-Specific Impl" *inside* Brain/Communication, but the invariance text says processing is world-side. Pick the seam:
- **(a)** Each world emits already-canonical data; brain comm is world-agnostic. *(thin adapter, fat world)*
- **(b)** Worlds emit raw; brain-side adapters translate to canonical. *(fat adapter, thin world)*
- **My read:** **(a)** for the wire format (worlds emit canonical), with a thin brain-side adapter that only *decodes the transport* — keeps the brain hexagon clean and matches "processing happens world-side."
- **Answer produces:** architecture.md §Adapters + ADR-000X.

### D4. dev_app — Brain module, or 4th top-level block? And how does it attach?
Code already hints separate (`devapp/brain_link.py`, `devapp/launcher.py`).
- Consumes: frame channel + brain state. Sends: teleop. Launcher args → its initial state.
- **My read:** **separate process**, a *client* that attaches over the same comm channels (frame + a teleop/state channel) — so brain runs identically with or without it. Top-level block beside Brain/World in the diagram, but architecturally a Brain *client*, not a world.
- **Answer produces:** architecture.md §dev_app + Launcher flag `--devapp`.

### D5. Inner comm convention — ROS-like, real ROS/MROS, or direct calls?
You said "maybe more like a ROS-type convention." This shapes Logic + Runner internals.
- **(a)** In-process typed pub/sub bus (ROS-like topics, *no* ROS runtime dependency)
- **(b)** Real ROS2 / mirror LimX's MROS
- **(c)** Direct function calls, no bus
- **My read:** **(a)** — ROS-like topic ergonomics, zero ROS deploy weight, mirrors LimX's MROS *conceptually* (our MAY-147 bus-topology finding) without inheriting it. Revisit (b) only if we federate with their stack.
- **Answer produces:** ADR-000X (Inner comm convention) + architecture.md §Inner contracts.

### D6. Runner / Logic split — who owns *policy selection*?
Runner = inference, model bank, policies, action banks. Logic = the intelligent modules (`reason/`).
- **My read:** **Logic decides intent (*what/when*), Runner owns execution (*how*)** — loads the model, runs inference, returns an action. Policy *selection* lives in Logic; the model *bank* lives in Runner. The Logic↔Runner boundary is an inner contract (a topic or a call interface per D5).
- **Answer produces:** architecture.md §Runner/§Logic + the inner contract row in Deliverable C.

### D7. Launcher — v1 flag set + process/supervision model.
- **Flags (v1):** `--world {isaac,mujoco,real}`, `--devapp {on,off}`, `--mode {walk,glide}`, + designed-to-extend. Confirm/extend.
- **Process model:** launcher spawns world proc + brain proc (+ dev_app proc if on), supervises, **reaps orphans** (we have the orphan-reaping finding). Confirm shape.
- **Arg flow:** launcher args are passed straight through as dev_app's *initial* setup state (overridable live in dev_app).
- **My read:** above as written; one supervisor, N children, structured shutdown.
- **Answer produces:** architecture.md §Launcher + AGENTS.md commands block.

### D8. Real-world scope (v1).
Thin: LimX SDK + hardware I/O + the *same* World contract. Mostly emits ready data; we can't code the physics.
- **My read:** Real is just another World adapter satisfying D1/D2 — no special-casing in the brain. v1 surface = emit `RobotObs`+`CameraFrame`, accept `RobotCmd`; glide/walk mode handled world-side.
- **Answer produces:** architecture.md §Real + ADR note on "real = adapter, not a special path."

---

## PART 2 — Deliverables you draft offline

### Deliverable A — architecture.md narrative (your voice; I shape to final)
Don't polish. Bullet-dump each section's *content* per the skeleton in Part 4. I'll turn bullets into prose and standardize.

### Deliverable B — ADR stubs (one per hard decision)
For each of D1, D2, D3, D5, D6 (the ones that are real *decisions* vs mechanics), write 4 lines:
`Decision:` / `Why:` / `Alternatives rejected:` / `Consequences:`. Use the template in Part 4. I format into numbered ADRs.

### Deliverable C — the contract table (highest-value artifact)
Fill this — it's the spine of the whole architecture. One row per message that crosses a boundary:

| Message | Direction | Boundary | Fields (rough) | Notes |
|---|---|---|---|---|
| RobotObs | World→Brain | outer | joint q/qd, IMU, base pose | canonical frame/units/order |
| CameraFrame | World→Brain | outer | rgb, depth, intrinsics, pose, stamp | AF_UNIX SOCK_STREAM, 30Hz |
| RobotCmd | Brain→World | outer | joint targets (+kp/kd?) | walk mode |
| GlideCmd | Brain→World | outer | base lin/ang velocity | glide mode |
| (Logic→Runner) | inner | inner | intent / obs slice | you define |
| (Runner→Action) | inner | inner | action / joint targets | you define |
Add/rename rows to match reality — these are seeded from memory, verify against `logic/oli/comm/`.

---

## PART 3 — Diagrams to draw (3)

1. **Container diagram (refine your existing one).** Launcher, World{Isaac,MuJoCo,Real}, Brain{Comm, Logic, Runner}, dev_app. **Highlight the outer contract boundary** as a thick line — that line *is* the invariance surface. Fix the "World-Specific Impl inside Brain" ambiguity per your D3 answer.
2. **Sequence diagram — one control tick.** World emits Obs+Frame → Comm decode → Logic (select policy) → Runner (inference) → Action → Cmd → Comm encode → World applies. Mark exactly where data crosses the boundary and where it becomes/leaves "canonical."
3. **Contract/ports map.** Just the boundaries + the messages on them (from Deliverable C). Two ports: outer (World↔Brain) and inner (Logic↔Runner). This is the ADR-friendly "what must never change lightly" picture.

Tooling: Excalidraw / draw.io / tldraw, or hand-drawn photo — anything. I can also regenerate any of these as Mermaid from your answers, so a rough sketch is enough.

---

## PART 4 — Templates (copy these)

### architecture.md skeleton
```markdown
# Humanoid (Oli) — Architecture

## 1. One-paragraph overview
## 2. The core principle: brain/world invariance   (D2)
## 3. Container map                                 (diagram 1)
## 4. The outer contract: World ↔ Brain             (D1, D3, Deliverable C)
## 5. Worlds: Isaac (walk|glide) / MuJoCo / Real     (D8)
## 6. Brain: Communication / Logic / Runner          (D5, D6)
## 7. The inner contract: Logic ↔ Runner             (D6)
## 8. dev_app                                        (D4)
## 9. Launcher & lifecycle                           (D7)
## 10. Control tick, end-to-end                      (diagram 2)
## 11. Open questions / not-yet-decided
```

### ADR template
```markdown
# ADR-NNNN: <title>
- Status: proposed
- Date: 2026-07-02
## Context
<the forces — why this even needs deciding>
## Decision
<what we chose, one sentence>
## Why
<the reasoning>
## Alternatives rejected
<what we didn't pick and why>
## Consequences
<what this makes easy / hard / commits us to>
```

---

## PART 5 — What I build when you're back (so you don't)

- `AGENTS.md` (root, ≤150 lines): commands, project structure, code style, the invariance boundary rule, links → architecture.md + ADRs. *Derived from your inputs — don't draft it.*
- `CLAUDE.md`: 2-line `@AGENTS.md` shim.
- `.claude/rules/brain.md` + `world.md` with `paths:` frontmatter (load only in-context).
- The **invariance CI test** + a minimal CI workflow (we have zero CI today).
- OpenSpec hygiene: wire `archive` → `openspec/specs/` sync so the spec corpus finally accumulates.
- Then we shape the whole thing into a Linear task via `/manager`.

---

## Appendix — the research-backed rules these files must obey

- **AGENTS.md is the cross-tool standard** (Linux-Foundation-governed, Dec 2025); Claude Code reads CLAUDE.md → bridge via `@AGENTS.md`. One file, both agents.
- **Small + human-written wins.** Optimal root file ~100-150 lines; gains reverse past that. **LLM-generated rules files measured -2% success / +23% cost** — so *you* draft content, I only structure it. Don't let me auto-generate a god-doc.
- **Discovery is the bottleneck:** architecture.md/ADRs only get read if **linked from AGENTS.md**. Orphaned `_docs/` < 10% discovery.
- **Enforcement is layered:** "if a violation blocks merge → CI; if it raises an eyebrow → CLAUDE.md." The invariance rule → a **test/hook**, not a markdown plea.
- **Spec-anchored, not spec-first:** OpenSpec only pays off if `archive` folds deltas into `specs/`. Ours is empty — close that loop.
- **Skip the ceremony for small fixes;** this SDD/ADR weight is for multi-file/foundational work like *this*.

Sources on request when I'm back — full findings are in the session log.
