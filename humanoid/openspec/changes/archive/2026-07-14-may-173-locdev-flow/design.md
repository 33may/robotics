# Design — locdev-flow, the AI-driven localization development loop (localization-dev-flow)

## Context

Locbench (sibling change) turns "does this localizer work?" into a computable verdict: frozen episodes, raw two-tier gates, `report.json` + plots, agent-drivable CLI. This change designs the loop that *drives* it. Design settled with Anton 2026-07-13 (user-story session): his own developer loop is the spec; the AI does the mechanical 90%; judgment is **captured, not encoded** — because for SLAM neither party knows the hard-to-spot rules yet, and pretending otherwise produces blind grinding.

Ground rules inherited: subscription-only compute (Claude Code sessions + Workflow tool, no Agent SDK), sequential research sweeps (mid-sweep insight steers the next run), no unapproved deletion, MD tables.

## The loop (Anton's 7 phases = the spec)

| # | Phase | Agent does | Judgment lives in |
|---|---|---|---|
| 1 | Research | survey approaches; read `docs/research/localization.md`, the board, prior journals | research doc + board evidence |
| 2 | Scope | pick target + written hypothesis *before* building | playbook scoping rules |
| 3 | Build | **find a working GitHub implementation, adapt it**; scratch-build is last resort | playbook build strategy (D3) |
| 4 | Refine | env recipe, contract conformance, "runs at all" milestone | locbench contract + playbook |
| 5 | Run | `locbench run <name> --smoke=3` (grind) / full 10 (certify) | harness (oracle) |
| 6 | Analyze | read `report.json` + **look at the plots** (overlay, timeline); name the failure mode | playbook diagnosis guide |
| 7 | Decide | improve / pivot / abandon — feasibility is a first-class verdict, not a stuck-counter | **Anton (v1) → promoted rules (v2)** |

## Decisions

### D1 — Capture judgment, don't encode it (graduated autonomy)

The core stance. v1 = **pair mode**: the loop runs in a live session; the agent executes phases 3–6 end-to-end, then *stops at phase 7* and presents: failure mode, evidence, its recommendation. Anton decides (or confirms), the decision + reasoning is journaled. **Promotion protocol**: when the same decision pattern appears repeatedly across journal entries, it's proposed as a rule for the playbook's Promoted Rules section (Anton approves the promotion). v2 = **earned autonomy**: unattended runs may take exactly the decisions the playbook covers; anything else pauses (D7). Autonomy is measurable — read the playbook, that's what the agent may decide alone.

### D2 — Runtimes: live-session skill first, workflow second (subscription only)

Workflow subagents are headless — they cannot discuss a plot with Anton. So v1 is a **skill** (`/loc-iterate`) in the live session, where conversation *is* the phase-7 mechanism; the Workflow tool is still used inside v1 for mechanical fan-outs (research sweeps, parallel log analysis). v2 (`loc-enforce` workflow) is deterministic JS: loop of iteration-subagents until PASS / budget / non-covered decision. No Agent SDK, no API tokens — every brain is a subscription-backed Claude Code process. Workflow nesting is one level; portfolio-over-enforcer fits inside it, invention stays a separate run (natural human checkpoint between "proposed an approach" and "spent a night on it").

### D3 — Build strategy: adapt-existing-first, vendor-in-realization

The agent hunts for a proven implementation the way Anton would (working repo, active bindings, licenses checked) before writing SLAM code itself. Adapted upstream is **cloned into `realizations/<name>/vendor/`** and patched in place (Anton's pick — everything the agent touches visible in one repo, no fork/push round-trips mid-loop). Hygiene: strip the clone's `.git` (snapshot, not submodule); `vendor/UPSTREAM.md` records url + commit sha + license + why-this-repo; upstream's fat assets (datasets, models) gitignored, code + patches committed.

### D4 — Iteration discipline

- **One change per iteration** — what makes `JOURNAL.md` causally readable and stops thrashing.
- **All tunables in `config.yaml`** — an iteration's diff is one file; parameter history reconstructable from git.
- **Smoke-grind, full-certify**: `--smoke=3` for the inner loop (~minutes matter at 10–15 min/eval); any smoke-PASS must be confirmed on the full 10 before it counts.
- **Bring-up before tune**: first milestone is "runs at all" (env boots, `step()` emits poses, coverage > 0, no crash); gates are phase two. The skill detects the phase from the report and steers differently.
- **Plots are senses**: the agent reads the overlay PNG and error timeline every iteration — "lost in the north aisle at t=40s" lives in the plot, not in a scalar.

### D5 — JOURNAL.md schema (the capture instrument)

Append-only, one entry per iteration:

```
## it-<N> — <date> — run <run-id>
hypothesis: <what we believed>
change:     <the one thing changed>
result:     <verdict + key numbers vs previous>
decision:   <improve|pivot|abandon|promote> (anton|agent:rule-<id>)
reasoning:  <why — the payload the promotion protocol mines>
next:       <the next hypothesis>
```

`decision(+who)` is the autonomy audit trail: v1 entries say `anton`, v2 entries must cite the playbook rule that authorized them. Traceability stack stays locbench's D12: `report.json` → `JOURNAL.md` → `README.md` → playbook → memory tree (`localization-slam` node via /reflect).

### D6 — Oracle integrity

The dev flow **never modifies** `logic/locbench/`, gate thresholds, or frozen episode sets — the exact analogue of "don't edit the tests to go green." If an iteration concludes a gate or the harness is wrong, that goes into the journal + the pause report *for Anton*; it is never acted on. Enforced by instruction (playbook hard rule) + scope (iteration subagents get the realization dir as their working scope) + review (journal entries cite diffs).

### D7 — Pause + Telegram ping

When a run needs Anton (v2 hits a non-covered decision; a pair-mode mechanical stretch finishes while he's away; DEPLOY/Stage-2 gate), it pauses and pings via the existing telegram plugin: failure summary + key plot + the agent's recommendation. Anton can answer from the phone; the loop resumes on reply. State at pause is already durable (journal + run dir), so a dead session resumes cleanly.

### D8 — Portfolio: sequential, board-ranked, delete-nothing

One candidate at a time — forced by hardware (one Isaac World, one 16 GB GPU, one `bench-<name>` env) and by Anton's sequential-sweeps rule: what candidate N revealed reshapes candidate N+1's starting point (that reasoning is exactly what journals carry). `locbench board` is the ranking. Losers' envs and map caches are *reported* as removable with sizes; removal is Anton's call, never automatic. DEPLOY-tier candidate ⇒ stop, present, Anton decides promotion + Stage-2.

### D9 — Invention: research in, scaffold out, glance before grind

Invention = a research session (deep-research pattern, Workflow fan-out) aimed at the gap the board exposes — including *compositions* (e.g. one candidate's odometry + another's relocalizer). Output is not analysis prose but a **ready-to-grind scaffold** via `/loc-new`: README (approach + hypothesis + why it beats the board), env recipe, vendor plan, module stub. Anton glances at the README before any grind spends a night — the checkpoint costs one read, not a formal gate.

### D10 — Approval gates (locked)

| action | who |
|---|---|
| env create / vendor clone / code edits / smoke + full runs / commits of realization + reports | autonomous |
| phase-7 decisions | Anton (v1) → playbook-covered (v2), else pause |
| playbook rule promotion | proposed by agent, approved by Anton |
| DEPLOY tier declaration, Stage-2 `--localizer` flip | **always Anton** |
| env/map/artifact deletion | **always Anton** |

## File layout (delta over locbench's D7/D12)

```
logic/oli/reason/localization/realizations/
  AGENTS.md              ← the living playbook (+ CLAUDE.md shim)
  _template/             ← /loc-new source: module stub, config.yaml, README/JOURNAL seeds
  <name>/
    module.py  config.yaml  environment.yml  build.sh  lock.yml
    vendor/                ← adapted upstream (D3): UPSTREAM.md committed, fat assets ignored
    map/                   ← gitignored (locbench D9)
    README.md  JOURNAL.md
.claude/skills/loc-new/  .claude/skills/loc-iterate/
.claude/workflows/loc-enforce.js
runs/<candidate>/<run-id>/   ← locbench-owned (D12 there)
```

## Risks / Trade-offs

- **Playbook starts empty → early loop is slow and Anton-heavy.** Accepted on purpose: speed without shared understanding is blind grinding. The promotion protocol is the flywheel.
- **Vendored clones bloat the repo** → fat-asset gitignore + UPSTREAM.md provenance; revisit (subtree/fork) at first real-pain candidate.
- **Session dies mid-iteration** → journal + run dirs are the durable state; `/loc-iterate` re-orients from them by design; workflow runs resume from journal.
- **Agent games the smoke set** (overfits 3 episodes) → certify-on-full rule; the board only ever shows full-run verdicts.
- **Telegram unavailable** → pause state is on disk either way; ping is a notification, not the state.

## Migration Plan

Additive. Sequenced after locbench §8 (reference-candidate triplet green) — the flow needs a proven oracle. Follow-ups this change enables: (1) RTAB-Map as the first real candidate through the loop; (2) further candidates/compositions via invention; (3) the winner's Stage-2 flip = MAY-173's endgame.

## Open Questions

- Promotion threshold: propose a rule after 2 recurrences of the same decision pattern, or leave it to judgment per case? Default: agent proposes at 2, Anton disposes.
- Does `/loc-iterate` auto-commit each iteration (fine-grained history) or only green/milestone states? Default: commit per iteration on the work branch, squash at merge.
