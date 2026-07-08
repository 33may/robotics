# AGENTS.md — Humanoid (Oli)

Single source of truth for AI agents (Claude Code + Codex) in this repo. Read first; follow links for depth. Keep it concrete and current.

**North star:** end-of-summer-2026 reasoning demo on the Oli humanoid; acting research runs in parallel. Every change ladders up to that.

## Architecture — read before touching runtime

Two containers talk only through a Communication Protocol, and the **Robot never knows which World it runs in** (world-invariance). Full map: **[docs/architecture/architecture.md](docs/architecture/architecture.md)**; decisions & why in `docs/architecture/adr/`.

- **World** — Isaac (primary) / MuJoCo (vendor ref) / Real (later). Applies commands, reports state.
- **Robot** — `Comm` (world edge/translator) · `Reason` (brain, emits intent) · `PolicyRunner` (own process; intent→actions, buffer + 2 clocks) · `dev_app` (observability UI).

## Golden rules (non-negotiable)

| Rule | Meaning |
|---|---|
| **Invariance boundary** | `brain`/`PolicyRunner` import **neither** `isaacsim` **nor** `limxsdk`; world-specifics live only in `Comm`. Enforced by the `brain` pytest marker + env split. |
| **Single entrypoint** | One command boots the whole stack (`logic/oli/launcher.py`). Never a 2–3 terminal dance; new subsystems plug into the Supervisor. |
| **Comm is the world edge, not a hub** | Internal modules talk over the bus directly; only `Observation`/`PolicyOut` cross the Comm edge. |
| **Tests in-repo, real TDD** | Committed red→green→refactor suite under `tests/`; no throwaway `/tmp` smoke scripts. |
| **Docs = structure, not status** | `architecture.md` = stable structure; blockers/status/roadmap → daily notes / Linear / memory. |
| **Check memory + oli-corpus first** | Before architectural questions, query humanoid memory + the `oli-corpus` MCP + repo state. |
| **MD tables, not ASCII** | In `.md`, always Markdown pipe tables (box-drawing only in terminal/chat). |

## Commands

Run from `humanoid/`; `p` = python.

```bash
p logic/oli/launcher.py --sim isaac  --mode glide --dev-app   # drive Oli, live cameras
p logic/oli/launcher.py --sim isaac  --mode forward --vx 0.3  # constant forward walk
p logic/oli/launcher.py --sim mujoco --mode walk             # vendor pad live-drive
p logic/oli/launcher.py --sim isaac  --mode walk --dry-run   # print boot plan only
#   --sim {isaac,mujoco,real}   --mode {stand,walk,forward,glide}   --dev-app

pytest -m brain    # deployment-invariant core (no isaacsim/limxsdk) — run in `brain` env
pytest -m isaac    # tests needing Isaac — run in `isaac` env
```

(Architecture doc uses the aspirational `--world`; the current flag is `--sim`.)

## Environments — why multiple

Mutually-incompatible ABIs → one conda env each; they talk via files (ONNX/datasets) or the wire (`limxsdk` MROS bus). Setup: `requirements/README.md`.

| Env | Py | Purpose |
|---|---|---|
| `brain` | 3.11 | invariant Oli brain (Reason+Action, walk ONNX, teleop) |
| `limx` | 3.8 | anything importing `limxsdk` (sim + RL deploy) — ABI-locked to 3.8 |
| `isaac` | 3.11 | Isaac Sim / IsaacLab |
| `hum` | 3.12 | oli-corpus extraction, MCP, docs |

## Repo structure

```
logic/oli/         comm/ reason/ action/ devapp/ launch/  (+ launcher.py, brain_main.py)
logic/simulation/  isaacsim/ mujoco/ real/ walkmatch/
docs/architecture/ architecture.md (+ adr/)
docs/oli-corpus/   queryable Oli/LimX docs (via oli-corpus MCP)
openspec/          changes/ (+ specs/) — spec-anchored change flow
vendor/            LimX submodules (mujoco-sim, rl-deploy)
requirements/      per-env pins   ·   tests/  brain + isaac marked suites
```

Per-module rules live in each module's own `AGENTS.md` (with a one-line `CLAUDE.md` shim), lazy-loaded when you edit that directory.

## Operating loop (skill tree)

**/work** (orient+plan+log) → **/manager** (shape Linear MAY-XXX) → **opsx:propose→apply→archive** (spec-anchored; archive folds deltas into `openspec/specs/`) → **/code** (TDD, single-entrypoint, invariance-safe) → **train/eval/dataset/hardware-copilot** (domain) → **/reflect** (→ memory) → **/obsidian** (daily note).

## Conventions

Branches `33may/` prefix · run from `humanoid/` with `p` · agent memory in `.claude/agent-memory/humanoid/` (index `MEMORY.md`) · daily notes in vault `vbti/humanoid/daily/` · session capture is its own repo — integrate, don't duplicate.
