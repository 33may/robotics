# Proposal — locdev-flow: the AI-driven localization development loop

## Why

`may-173-locbench-harness` (sibling change, in build) delivers the **oracle**: any `realizations/<name>/` gets a reproducible pass/fail verdict from frozen episodes. What it deliberately does not deliver is the **developer** — the thing that researches approaches, adapts working implementations, runs them against the oracle, reads the plots, and decides *improve or abandon* until a candidate passes the gates.

Today that developer is Anton's own loop: research → scope → build (find a working GitHub project, adapt it) → refine → run → analyze plots → decide feasibility → repeat. The goal of this change is to make an AI agent run that same loop — spending tokens freely on the mechanical 90% — **with Anton in the loop at the judgment points**, because (his words, 2026-07-13) neither he nor the agent knows the improve-vs-abandon rules for SLAM yet. The system must therefore **capture judgment, not pretend to encode it**:

1. **Pair mode first.** A live-session skill walks the agent through the phases; the agent does everything mechanical and presents its analysis; Anton and the agent decide together; every decision lands in the candidate's `JOURNAL.md` *with its reasoning*.
2. **Autonomy is earned, and measurable.** Recurring decision patterns get promoted from journals into the living playbook (`realizations/AGENTS.md`). Unattended runs may take exactly the decisions the playbook covers — anything novel pauses and pings Anton on Telegram. The trust boundary is a readable document.

Hard constraint: everything runs on the Claude Code subscription (live sessions, subagents, Workflow tool). No Agent SDK / API-token billing.

## What Changes

- **NEW — the playbook, `reason/localization/realizations/AGENTS.md`.** The living methodology document every dev-flow agent reads: Anton's 7-phase loop, build strategy (adapt-existing-GitHub-first), iteration discipline (one change per iteration; smoke-grind, full-certify), diagnosis guide, oracle-integrity rules, and a **Promoted Rules** section that starts near-empty and grows from journaled decisions. Absorbs locbench task 11.1.
- **NEW — realization file conventions.** Beyond locbench's D7 set: `config.yaml` (all tunables in one file → iterations are clean diffs), `vendor/` (adapted upstream cloned in, `.git` stripped, `UPSTREAM.md` provenance, fat assets gitignored), `JOURNAL.md` schema (hypothesis → change → run-id → numbers → decision+who → next).
- **NEW — `/loc-new <name>` skill.** Scaffolds a ready-to-grind realization from a research pick: folder from template, vendor clone + provenance, env recipe draft, README with approach + hypothesis.
- **NEW — `/loc-iterate <name>` skill (pair mode, v1).** One loop cycle in the live session: refine/build → `locbench run --smoke` → read report + plots (plots are senses — the agent looks at the overlay) → summarize the failure mode → **stop at the decision** → journal the outcome. Uses the Workflow tool internally for mechanical fan-outs only.
- **NEW — `loc-enforce` workflow (earned autonomy, v2).** Unattended grind of one candidate, budget-capped; may only take playbook-covered decisions; novel decision or mechanical-stretch completion → pause + Telegram ping (failure summary + key plot + recommendation), resume on Anton's reply. Gated: built only after pair mode has produced real promoted rules.
- **NEW — portfolio + invention protocol (thin, prose).** Sequential candidates only (one GPU/Isaac; mid-sweep insight steers the next pick); `locbench board` is the ranking; losers' envs *reported* removable, never auto-removed. Invention = a research session against the board's failure evidence, output = a `/loc-new` scaffold Anton glances at before it burns compute.
- **Approval gates.** DEPLOY promotion and the Stage-2 (`--localizer`) flip always stop for Anton. Everything below that follows the playbook-coverage rule.
- **NOT in scope:** the locbench harness itself (sibling change), real SLAM adapters (RTAB-Map is the first *user* of this flow, immediately after), real-robot episodes.

## Capabilities

### New Capabilities

- `localization-dev-flow`: the agent-driven development loop around `localization-bench`. Defines the playbook + promotion protocol (journal → rule → autonomy), the pair-mode iteration skill, the earned-autonomy workflow with pause/ping, realization file/vendor conventions, and the portfolio/invention protocol. Observable behavior: a candidate goes from research pick to gate-passing realization through journaled, reproducible iterations — with every judgment either made by Anton or traceable to a written playbook rule.

### Modified Capabilities

- None. Additive: consumes locbench's CLI + report/plot artifacts read-only; never modifies `logic/locbench/`, gate thresholds, or episode sets (oracle integrity — see design D6).

## Impact

- **New files**: `reason/localization/realizations/AGENTS.md` (+ one-line `CLAUDE.md` shim), realization template, `.claude/skills/loc-new/`, `.claude/skills/loc-iterate/`, `.claude/workflows/loc-enforce.js`.
- **Existing code**: none touched. Locbench change: its task 11.1 moves here.
- **Envs**: lifecycle stays locbench-owned (D8 there); this flow adds one rule — never auto-remove.
- **Linear**: [MAY-173](https://linear.app/may33/issue/MAY-173) — locbench builds the oracle; this change builds the developer; RTAB-Map-inside-the-loop is the follow-up both enable.
