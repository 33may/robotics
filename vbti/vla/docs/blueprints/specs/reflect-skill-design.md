# `/reflect` skill — session distillation design

**Date**: 2026-04-29
**Replaces**: `~/.claude/commands/reflect.md` (current free-form journaling-only command)
**Lives at**: `~/.claude/skills/reflect/`

---

## Goal

End-of-session distillation that produces three real artifacts from a Claude Code session:

1. New / updated **memory files** in `~/.claude/projects/<project>/memory/` (auto-loaded into future conversations via `MEMORY.md` index)
2. New **skill files** in `~/.claude/skills/<name>/SKILL.md` (auto-discovered by Claude Code at session start)
3. Appended block on **today's Obsidian daily note** in casual user voice

The skill drives an analysis-first interview: I read the session, surface candidates, you classify and shape each one, files are written item-by-item.

## Non-goals

- Sprint-board updates (dropped from old `/reflect`)
- Auto-detecting "important" turns without user validation
- Inventing memory/skill content the user didn't actually express
- Forced wrap-up — the skill is interruptible at every level

## File layout

```
~/.claude/skills/reflect/
├── SKILL.md                  # the procedure (frontmatter + phase instructions)
├── find_session.py           # locate source → render via empr → stdout
└── templates/
    ├── memory_feedback.md
    ├── memory_user.md
    ├── memory_project.md
    ├── memory_reference.md
    └── skill_new.md
```

The old `~/.claude/commands/reflect.md` is **deleted**. The slash invocation `/reflect` resolves to the new skill (skill names auto-bind to slash commands).

## Source of truth

- **Default**: live session JSONL at `~/.claude/projects/<encoded-cwd>/<latest>.jsonl`
- **With arg**: `/reflect <path>` — accepts either a raw `.jsonl` or a pre-rendered `.md` (e.g., from the Obsidian vault `<vault>/projects/<name>/sessions/rendered/<uuid>.md`)
- **Rendering**: `empr.claude_sessions.renderer.render_session()` from `/home/may33/projects/3mpr/empr` — it strips navigation noise (Read/Glob/etc.), collapses long tool outputs, cleans system XML tags
- **Fallback**: in-context conversation if disk lookup fails

`find_session.py` is the only helper. It encodes cwd, picks the latest JSONL, imports the renderer with `sys.path.insert(0, "/home/may33/projects/3mpr")`, prints rendered markdown to stdout.

## Phases

```
Phase 0  Silent: locate source, run find_session.py, load digest
Phase 1  Silent: extract candidates from digest into internal list
Phase 2  Per-item walkthrough — count + small list, then one-by-one
         Write files immediately on per-item approval
Phase 3  Free-form journaling — open-ended, may pivot into work
Phase 4  Append to today's daily note (only on explicit signal)
```

The skill is **interruptible at every level**:
- Per-item: user can say `build` to drop the walkthrough format and collaboratively implement the candidate properly. On `next` / `done`, walkthrough resumes.
- Phase 3: user may pivot to actual coding/discussing. No auto-summary, no nudge. Daily-note append happens only on `wrap up` / `write it` / `done with reflect`.
- Mid-build new findings: I queue silently and surface them when user signals wrap-up.

If extraction finds zero candidates, Phase 2 collapses to "nothing notable from analysis" and the skill goes straight to Phase 3.

## Extraction taxonomy

What the silent extraction pass scans for. Each candidate carries: raw chat citation (verbatim), proposed category, proposed file path, draft content, confidence (high/med/low).

| Category | Signal in source | Default destination |
|---|---|---|
| **feedback (correction)** | "stop X", "don't", "not like that", explicit rejection of approach | `memory_feedback.md` |
| **feedback (validation)** | "yes that worked", "perfect, keep doing that", non-obvious approach accepted without pushback | `memory_feedback.md` |
| **user trait** | role / preference / expertise mentions | `memory_user.md` |
| **project state** | who/what/why/by-when changes | `memory_project.md` |
| **external reference** | "X tracked in Linear project Y", "dashboard at Z" | `memory_reference.md` |
| **skill candidate** | a multi-step procedure worth reusing | `~/.claude/skills/<name>/SKILL.md` |
| **method win/loss** | something tried that worked / flopped, with traceable cause | inline note in daily; upgraded to feedback memory if it has rule-shape |

## Interview UX

**Step 1 — upfront list (single message):**

```
Found N candidates from this session:
  1. [feedback] don't dump options at me — terse decisions only
  2. [skill]    /reflect distillation procedure
  3. [user]     prefers blockquote citations over backticks for chat
  ...

Going through one by one. Say "skip" to drop, "later" to defer.
```

**Step 2 — per-item message (one per candidate):**

```
**1/N — feedback memory**

> "<verbatim quote from session, blockquote-cited>"

**Proposed**: ~/.claude/projects/<project>/memory/feedback_<slug>.md
**Draft body**:
> <drafted body>
> **Why:** ...
> **How to apply:** ...

OK to write? (yes / edit / skip / change-bucket / build / merge with X / later)
```

**Step 3 — on approval:** write file via `Write` tool, confirm with one line ("✓ wrote feedback_<slug>.md"), advance to next.

### Per-item controls

| Input | Effect |
|---|---|
| `yes` / `ok` / `good` | Write file as drafted, advance |
| edit text | Reshape body, re-show, re-confirm |
| `skip` | Drop candidate, advance |
| `later` | Save to `pending_candidates.md` for next reflect |
| `change-bucket` | Reclassify (e.g., feedback → skill) and re-draft |
| `merge with X` | Fold into existing memory file instead of creating new |
| `build` / "let's make it properly" | Drop walkthrough format, enter normal collaborative work mode for that candidate. On `next` / `done` resume walkthrough. The drafted file is discarded; whatever was actually built replaces it |

## File output formats

### Memory file template (frontmatter + body)

```yaml
---
name: <slug>
description: <one-line description used to match relevance in future conversations>
type: feedback                # or: user, project, reference
originSessionId: <source-session-uuid>
---
<body>

**Why:** <reason — past incident or strong preference>
**How to apply:** <when/where the rule kicks in>
```

`Why:` and `How to apply:` are mandatory for `feedback` and `project` types; optional for `user` and `reference`.

### Skill file template

```yaml
---
name: <skill-name>
description: <when-to-use line — drives auto-trigger matching>
---

# <Title>

## When to use
...

## Procedure
1. ...
2. ...
```

If the skill needs more than a single SKILL.md (helper scripts, templates, multiple files), the skill prompts: "this needs more structure — switch to build mode?" and lets the user collaboratively build it.

### Daily-note append format

```markdown

---
*reflected at HH:MM*

> <free-form prose drafted from Phase 3, in user's casual voice, blockquoted>

(optional) memories distilled:
- feedback_<slug>
- skill_<name>
```

No rigid `## Reflection` header — matches existing daily-note style (append-only stream-of-consciousness).

### Index update

Every new memory file gets one line appended to `MEMORY.md`:

```
- [<title>](<filename>.md) — one-line hook
```

`MEMORY.md` is the auto-loaded index, kept under 200 lines per CLAUDE.md.

## Approval gates

| Phase | What gets approved | Who decides |
|---|---|---|
| 0 | (none — silent file lookup) | — |
| 1 | (none — silent extraction into internal list) | — |
| 2 | Per-item: bucket, file path, body content, then write | User, item-by-item |
| 3 | Open-ended; no approval needed | User |
| 4 | Daily-note append text | User sees draft, approves, then write |

## Skill self-rules (codified in SKILL.md)

- Never invent quotes — if no verbatim citation exists, don't fabricate; reclassify as "method note" instead of feedback memory
- Never write a file without explicit `yes` / `ok` / `good` for that specific item
- If extraction finds zero candidates, skip Phase 2 entirely
- Honor per-item `build` and Phase 3 pivot at all times — no auto-summary, no nudge
- Daily-note append happens only on explicit user signal

## Persistence — how artifacts become live

| Artifact | Mechanism |
|---|---|
| Memory file in `~/.claude/projects/<project>/memory/<name>.md` | Loaded on-demand when its `MEMORY.md` index entry looks relevant in a future conversation |
| `MEMORY.md` index entry | Auto-loaded into every conversation in this project (verified — appears in system context) |
| Skill file in `~/.claude/skills/<name>/SKILL.md` | Auto-discovered at Claude Code session start, available as `/skill-name` and via auto-trigger description matching |
| Daily-note append | Stored in Obsidian vault — read by future `/reflect` runs and by user during browsing |

These are not dead files. The persistence is the actual mechanism Claude Code uses for memory and skill loading.

## Open questions / risks

- **Confidence threshold**: low-confidence candidates clutter the upfront list. Default behavior: include all, mark low ones with `[?]` so user can fast-skip. Revisit if list noise becomes a problem.
- **Slug collisions**: if a generated filename already exists, prompt for rename rather than overwrite.
- **Long sessions**: even with renderer noise-stripping, very long sessions may exceed reasonable context. If digest > ~50k tokens, truncate to last 80% and warn the user; explicitly note that early-session feedback may be missed.
- **Cross-project skills vs project memories**: skills always go to `~/.claude/skills/` (global). Memories default to project-specific dir; user can override per-item if a finding is genuinely global.

## Implementation order (for the plan that follows)

1. Write `find_session.py` and verify it produces sane digest output for live + arg-passed sources
2. Write the five templates
3. Write `SKILL.md` with phase instructions, controls, self-rules
4. Delete `~/.claude/commands/reflect.md`
5. Manual test: invoke `/reflect` end-of-session, walk through, verify writes land in correct paths
6. Verify next session loads new memories via `MEMORY.md` index and shows new skill in available list
