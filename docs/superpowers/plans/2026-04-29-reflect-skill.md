# `/reflect` skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the journaling-only `/reflect` command with a skill that distills Claude Code sessions into real memory files, skill files, and a daily-note append.

**Architecture:** A pure-procedure skill (`SKILL.md` drives Claude through a 4-phase flow) plus a thin Python helper (`find_session.py`) that locates the source session JSONL and renders it to clean markdown using the existing `empr.claude_sessions.renderer`. Five frontmatter templates seed memory/skill file writes.

**Tech Stack:** Python 3 (helper script + tests), Markdown (skill procedure + templates), Claude Code skill loading (`~/.claude/skills/<name>/SKILL.md`), existing `empr` package at `/home/may33/projects/3mpr/empr` for session rendering.

**Spec:** `docs/superpowers/specs/2026-04-29-reflect-skill-design.md`

**Implementation root:** `~/.claude/skills/reflect/` (this directory is a git repo; commit there, not in the robotics repo)

---

## File structure

Files created/modified by this plan:

```
~/.claude/skills/reflect/                 # new directory, all in this skill
├── SKILL.md                              # procedure: phase instructions + controls + self-rules
├── find_session.py                       # locate source → render via empr → stdout
├── tests/
│   └── test_find_session.py              # smoke tests for the helper
├── templates/
│   ├── memory_feedback.md                # frontmatter + Why/How shell
│   ├── memory_user.md
│   ├── memory_project.md
│   ├── memory_reference.md
│   └── skill_new.md                      # SKILL.md skeleton for distilled skills
└── README.md                             # human-facing notes (optional, last)

~/.claude/commands/reflect.md             # DELETED (replaced by skill)
```

The directory is one focused unit: helper, templates, procedure all live together. Tests are co-located.

---

### Task 1: Bootstrap skill directory + helper test

**Files:**
- Create: `~/.claude/skills/reflect/` (directory)
- Create: `~/.claude/skills/reflect/tests/test_find_session.py`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p ~/.claude/skills/reflect/tests
mkdir -p ~/.claude/skills/reflect/templates
```

- [ ] **Step 2: Write the failing test for find_session.py**

Create `~/.claude/skills/reflect/tests/test_find_session.py`:

```python
"""Smoke tests for find_session.py — runs the helper as a subprocess
and verifies the output looks like rendered session markdown."""

import os
import subprocess
import sys
from pathlib import Path

HELPER = Path(__file__).parent.parent / "find_session.py"
PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _has_any_session() -> bool:
    """Skip tests if no live sessions exist anywhere on disk."""
    if not PROJECTS_DIR.exists():
        return False
    return any(PROJECTS_DIR.glob("*/*.jsonl"))


def test_helper_prints_rendered_markdown_for_default_cwd():
    """Running with no args from a project that has sessions emits markdown."""
    if not _has_any_session():
        return  # nothing to test against on a fresh machine

    # Pick any project dir that has a JSONL and use its decoded path as cwd
    encoded_dir = next(d for d in PROJECTS_DIR.iterdir()
                       if d.is_dir() and any(d.glob("*.jsonl")))
    decoded_cwd = "/" + encoded_dir.name.lstrip("-").replace("-", "/")

    if not Path(decoded_cwd).exists():
        return  # encoded path doesn't map to a real dir, skip

    result = subprocess.run(
        [sys.executable, str(HELPER)],
        cwd=decoded_cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"helper failed: {result.stderr}"
    assert result.stdout.startswith("# Session "), \
        f"output doesn't look rendered: {result.stdout[:200]!r}"


def test_helper_with_explicit_path():
    """Passing --path <jsonl> emits rendered markdown for that file."""
    jsonls = list(PROJECTS_DIR.glob("*/*.jsonl"))
    if not jsonls:
        return

    target = jsonls[0]
    result = subprocess.run(
        [sys.executable, str(HELPER), "--path", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"helper failed: {result.stderr}"
    assert result.stdout.startswith("# Session "), \
        f"output doesn't look rendered: {result.stdout[:200]!r}"


def test_helper_errors_clearly_when_no_session_for_cwd(tmp_path):
    """Running from a cwd with no encoded project dir gives a clear error."""
    result = subprocess.run(
        [sys.executable, str(HELPER)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "no session" in result.stderr.lower() or \
           "not found" in result.stderr.lower()
```

- [ ] **Step 3: Run test — expect failure (no helper yet)**

```bash
cd ~/.claude/skills/reflect
python -m pytest tests/test_find_session.py -v
```

Expected: all 3 tests FAIL with `FileNotFoundError` or similar (find_session.py doesn't exist).

- [ ] **Step 4: Commit**

```bash
cd ~/.claude
git add skills/reflect/tests/test_find_session.py
git commit -m "test: bootstrap test suite for /reflect find_session helper"
```

---

### Task 2: Implement find_session.py

**Files:**
- Create: `~/.claude/skills/reflect/find_session.py`

- [ ] **Step 1: Write find_session.py**

```python
#!/usr/bin/env python3
"""find_session.py — locate the source session for /reflect and render it.

Default behavior: encode cwd → ~/.claude/projects/<encoded>/, find latest .jsonl,
render via empr.claude_sessions.renderer.render_session, print to stdout.

With --path: render that specific .jsonl, OR cat that .md if already rendered.
"""

import argparse
import sys
from pathlib import Path

# empr lives outside site-packages — extend sys.path before import
EMPR_ROOT = Path("/home/may33/projects/3mpr")
if str(EMPR_ROOT) not in sys.path:
    sys.path.insert(0, str(EMPR_ROOT))

try:
    from empr.claude_sessions.renderer import render_session
except ImportError as e:
    print(f"error: cannot import empr.claude_sessions.renderer: {e}", file=sys.stderr)
    print(f"  expected at: {EMPR_ROOT}/empr/claude_sessions/renderer.py", file=sys.stderr)
    sys.exit(2)


def encode_cwd(cwd: Path) -> str:
    """Match Claude Code's encoding: replace '/' with '-', drop leading '-' later."""
    return str(cwd).replace("/", "-")


def find_latest_jsonl(encoded_dir: Path) -> Path | None:
    candidates = list(encoded_dir.glob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate + render Claude Code session source")
    parser.add_argument("--path", help="explicit .jsonl or .md file to use as source")
    args = parser.parse_args()

    if args.path:
        target = Path(args.path).expanduser()
        if not target.exists():
            print(f"error: path not found: {target}", file=sys.stderr)
            return 1

        if target.suffix == ".md":
            sys.stdout.write(target.read_text())
            return 0
        if target.suffix == ".jsonl":
            sys.stdout.write(render_session(target))
            return 0
        print(f"error: unsupported extension {target.suffix!r} (want .jsonl or .md)",
              file=sys.stderr)
        return 1

    cwd = Path.cwd()
    encoded_name = encode_cwd(cwd)
    projects_root = Path.home() / ".claude" / "projects"
    encoded_dir = projects_root / encoded_name

    if not encoded_dir.exists():
        print(f"error: no session directory for cwd ({cwd})", file=sys.stderr)
        print(f"  looked for: {encoded_dir}", file=sys.stderr)
        return 1

    latest = find_latest_jsonl(encoded_dir)
    if latest is None:
        print(f"error: no .jsonl files in {encoded_dir}", file=sys.stderr)
        return 1

    sys.stdout.write(render_session(latest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make executable**

```bash
chmod +x ~/.claude/skills/reflect/find_session.py
```

- [ ] **Step 3: Run tests — expect pass**

```bash
cd ~/.claude/skills/reflect
python -m pytest tests/test_find_session.py -v
```

Expected: all 3 tests PASS (or skip cleanly if no sessions on disk for the path-less test).

- [ ] **Step 4: Manual smoke test from this project**

```bash
cd /home/may33/projects/ml_portfolio/robotics
python ~/.claude/skills/reflect/find_session.py | head -40
```

Expected: starts with `# Session <id>` and shows recent turns of the active session in cleaned markdown.

- [ ] **Step 5: Commit**

```bash
cd ~/.claude
git add skills/reflect/find_session.py
git commit -m "feat: find_session.py for /reflect — locate + render source"
```

---

### Task 3: Write the five templates

**Files:**
- Create: `~/.claude/skills/reflect/templates/memory_feedback.md`
- Create: `~/.claude/skills/reflect/templates/memory_user.md`
- Create: `~/.claude/skills/reflect/templates/memory_project.md`
- Create: `~/.claude/skills/reflect/templates/memory_reference.md`
- Create: `~/.claude/skills/reflect/templates/skill_new.md`

- [ ] **Step 1: Write memory_feedback.md**

```markdown
---
name: {{slug}}
description: {{one-line description used to match relevance in future conversations — be specific}}
type: feedback
originSessionId: {{session_uuid}}
---
{{rule_or_fact}}

**Why:** {{reason — the past incident, strong preference, or constraint that motivates the rule}}
**How to apply:** {{when/where the rule kicks in — which situations, file types, decision points}}
```

- [ ] **Step 2: Write memory_user.md**

```markdown
---
name: {{slug}}
description: {{one-line description of this user trait}}
type: user
originSessionId: {{session_uuid}}
---
{{trait — role, expertise, preference, working style}}
```

- [ ] **Step 3: Write memory_project.md**

```markdown
---
name: {{slug}}
description: {{one-line description of project state, decision, or constraint}}
type: project
originSessionId: {{session_uuid}}
---
{{fact_or_decision}}

**Why:** {{motivation — constraint, deadline, stakeholder ask}}
**How to apply:** {{how this should shape suggestions and decisions going forward}}
```

- [ ] **Step 4: Write memory_reference.md**

```markdown
---
name: {{slug}}
description: {{one-line pointer to where information lives}}
type: reference
originSessionId: {{session_uuid}}
---
{{external_resource_and_purpose — system, URL, channel, project key, what it contains}}
```

- [ ] **Step 5: Write skill_new.md**

```markdown
---
name: {{skill-name}}
description: {{when-to-use line — drives auto-trigger matching, be specific about triggers}}
---

# {{Title}}

## When to use

{{conditions that should activate this skill — user phrases, file types, task shapes}}

## Procedure

1. {{step}}
2. {{step}}
3. {{step}}

## Notes

{{constraints, gotchas, edge cases — only include if truly load-bearing}}
```

- [ ] **Step 6: Commit**

```bash
cd ~/.claude
git add skills/reflect/templates/
git commit -m "feat: memory + skill templates for /reflect distillation"
```

---

### Task 4: Write SKILL.md — the procedure

**Files:**
- Create: `~/.claude/skills/reflect/SKILL.md`

- [ ] **Step 1: Write SKILL.md**

```markdown
---
name: reflect
description: Use at end of working session — analyzes the session, extracts feedback/methods/skill candidates, walks them with the user item-by-item, writes real memory + skill files, then appends to today's Obsidian daily note
---

# /reflect — Session Distillation

## Identity

You are a session distiller. Your job: turn a Claude Code session into persistent artifacts that change future behavior — memory files (auto-loaded into future conversations) and skill files (auto-discovered at session start) — plus a casual journal entry on today's daily note.

You do not invent. Every memory entry must trace to a verbatim citation from the session source. If you can't quote it, you can't write it.

You are interruptible at every level. The user can pivot to actual work mid-walkthrough; you step out, then resume on signal.

## Source paths

- **Vault**: `/home/may33/Documents/Obsidian Vault`
- **Daily note**: `vbti/sessions/sprintN/DD-MM-YYYY.md` (current sprint dir — check `vbti/sessions/` for latest)
- **Project memory**: `~/.claude/projects/-home-may33-projects-ml-portfolio-robotics/memory/`
- **Global memory**: `~/.claude/memory/`
- **Skills**: `~/.claude/skills/`
- **MEMORY.md index**: in each memory dir, one line per memory file

## Phase 0: Locate source (silent)

If user passed `<path>` as arg → use that path with `--path`.
Otherwise → no args (uses cwd).

```bash
python ~/.claude/skills/reflect/find_session.py [--path <file>]
```

Capture stdout as the rendered source. If exit code is nonzero, surface the error to the user and stop.

## Phase 1: Extract candidates (silent)

Read the rendered source. Build an internal list of candidates across these categories:

| Category | Signal |
|---|---|
| feedback (correction) | "stop", "don't", "not like that", explicit rejection |
| feedback (validation) | "yes that worked", "perfect", non-obvious approach accepted without pushback |
| user trait | role / preference / expertise mention |
| project state | who/what/why/by-when changes |
| external reference | "X tracked in Y", "dashboard at Z" |
| skill candidate | multi-step procedure worth reusing |
| method win/loss | something tried that worked / flopped |

Each candidate carries:
- **citation**: the verbatim user quote (must exist — never fabricate)
- **category**: from table above
- **proposed file path**: see destinations
- **draft body**: filled from the template for that category
- **confidence**: high / med / low (low = "include but flag with [?]")

**Default destinations:**
- feedback → `<project-memory>/feedback_<slug>.md`
- user → `<project-memory>/user_<slug>.md` (or global if user trait is general)
- project → `<project-memory>/project_<slug>.md`
- reference → `<project-memory>/reference_<slug>.md`
- skill → `~/.claude/skills/<name>/SKILL.md`
- method note → inline in daily note (no separate file unless it has rule-shape)

If the digest exceeds ~50k tokens, truncate to last 80% and warn the user that early-session feedback may be missed.

If extraction finds zero candidates → skip Phase 2, go straight to Phase 3 with "nothing notable from analysis."

## Phase 2: Per-item walkthrough

### Step 1: Upfront list (one message)

```
Found N candidates from this session:
  1. [feedback] <one-line summary>
  2. [skill]    <one-line summary>
  ...

Going through one by one. Say "skip" to drop, "later" to defer.
```

### Step 2: Per-item message (one per candidate, in order)

```
**i/N — <category> <memory|skill>**

> "<verbatim citation from source>"

**Proposed**: <full file path>
**Draft body**:
<filled-in template body, in a blockquote>

OK to write? (yes / edit / skip / change-bucket / build / merge with X / later)
```

### Step 3: Apply the user's choice

| Input | Action |
|---|---|
| `yes` / `ok` / `good` | Write the file with `Write` tool, confirm one-line ("✓ wrote <name>"), index update for memories (append `- [<title>](<file>) — <hook>` to MEMORY.md), advance |
| edit text | Reshape body per user input, re-show, re-confirm |
| `skip` | Drop, advance |
| `later` | Append to `<project-memory>/pending_candidates.md`, advance |
| `change-bucket` | Reclassify, re-pick template, re-draft, re-show |
| `merge with X` | Append to existing memory file X instead of creating new; show diff before write |
| `build` / "let's make it properly" | Drop walkthrough format, enter normal collaborative work for this candidate. On `next` / `done` / `go next`, resume walkthrough at next item. Discard the drafted file; whatever was actually built replaces it |

## Phase 3: Free-form journaling

Open with one question informed by Phase 2:

> "Anything else from today I missed?"

Or if Phase 2 was light:

> "What stands out from today?"

**Rules** (from old `/reflect`, preserved):
- ONE question at a time. Listen, don't summarize after every answer.
- Ask follow-ups ONLY when user explicitly says ("ask me more", "go deeper")
- Short answer is fine — don't push
- User controls depth and length. Never push uninvited.

**Phase 3 may pivot to actual work** — discussing, planning, building. If it does, you step out of "interview mode" and resume normal coding behavior. No timeout. No auto-summary.

If new findings surface during this phase or any subsequent work, queue them silently. Surface them when the user signals wrap-up.

## Phase 4: Daily-note append

Triggered ONLY by explicit user signal: `wrap up`, `write it`, `done with reflect`, `ok save`.

### Step 1: Locate today's daily

`<vault>/vbti/sessions/<current-sprint>/DD-MM-YYYY.md` using today's date and the latest `sprintN` directory.

If file doesn't exist, create it (just append).

### Step 2: Draft the append in user's voice

Format (always today's daily, regardless of source session):

```markdown

---
*reflected at HH:MM*

> <free-form prose drafted from Phase 3, casual, first person, stream of consciousness — match daily-note voice>

(if any memories distilled this session)
memories distilled:
- <slug-1>
- <slug-2>
```

### Step 3: Show draft, await approval

Present the full append text in a backtick block. State the target file. Wait for `yes` / `good` / `ok` / edits.

### Step 4: Append on approval

Use `Edit` (or `Write` if file is new) to add the block to the end of the daily note. Confirm with one-line: `✓ appended to <path>`.

## Self-rules (never violate)

1. **Never fabricate citations.** If no verbatim quote exists, the candidate isn't a feedback memory — reclassify or drop.
2. **Never write without explicit `yes` for that specific item.** Per-item approval is non-negotiable.
3. **Never push the conversation forward uninvited.** User controls every transition.
4. **Honor `build` and Phase 3 pivot.** No nudge, no auto-resume, no auto-summary.
5. **Daily-note append only on explicit signal.** No automatic write at end of Phase 3.
6. **Keep MEMORY.md index under 200 lines.** Each entry is one line, ~150 chars max.
7. **Skills always go to `~/.claude/skills/`** (global). Memories default to project; user overrides per-item if global.
8. **Verify source-session UUID** before writing `originSessionId` — pull from the rendered source's first line (`# Session <id>`).

## What this is NOT

- NOT a sprint-board updater (dropped from old `/reflect`)
- NOT a forced wrap-up — interview ends only when user signals
- NOT a free-form summary generator — every memory traces to a quote
- NOT auto-launched at session end — only when user invokes `/reflect`
```

- [ ] **Step 2: Verify SKILL.md is valid**

```bash
head -5 ~/.claude/skills/reflect/SKILL.md
```

Expected: shows the frontmatter starting with `---` and `name: reflect`.

- [ ] **Step 3: Commit**

```bash
cd ~/.claude
git add skills/reflect/SKILL.md
git commit -m "feat: SKILL.md procedure for /reflect distillation"
```

---

### Task 5: Delete old command + verify discovery

**Files:**
- Delete: `~/.claude/commands/reflect.md`

- [ ] **Step 1: Delete old command file**

```bash
rm ~/.claude/commands/reflect.md
```

- [ ] **Step 2: Commit deletion**

```bash
cd ~/.claude
git add -u commands/reflect.md
git commit -m "refactor: remove old /reflect command (replaced by skill)"
```

- [ ] **Step 3: Manual verification — restart needed**

The skill auto-discovers at next Claude Code session start. Tell the user:

> "Skill written. Start a new Claude Code session to load it. In the new session, `/reflect` should appear in the available-skills list and be invokable. Run it with no args to test against the current live session, or with `<path>` to reflect on a past session."

---

### Task 6: First real run — collaborative test

**Files:**
- (depends on what gets distilled — files written are determined by walkthrough)

- [ ] **Step 1: Open a new Claude Code session**

User starts a new session in this same project to load the new skill.

- [ ] **Step 2: Run `/reflect`**

User invokes `/reflect` (no args, on the current live session).

- [ ] **Step 3: Walk the upfront list**

Verify Phase 2 step 1 emits the count + categorized list.

- [ ] **Step 4: Walk at least one of each category**

For each candidate, verify the citation, draft body, controls all behave as specified. Use `build` on at least one item to confirm pivot-to-work and resume-on-signal works.

- [ ] **Step 5: Verify writes land in correct paths**

```bash
ls -lt ~/.claude/projects/-home-may33-projects-ml-portfolio-robotics/memory/ | head
ls -lt ~/.claude/skills/ | head
cat ~/.claude/projects/-home-may33-projects-ml-portfolio-robotics/memory/MEMORY.md | tail -10
```

Expected: new memory file(s) present, MEMORY.md index updated, any new skill present in `~/.claude/skills/`.

- [ ] **Step 6: Verify Phase 4 daily-note append**

```bash
ls -lt "/home/may33/Documents/Obsidian Vault/vbti/sessions/sprint4/" | head -3
```

Expected: today's daily file modified or created with the appended `*reflected at HH:MM*` block.

- [ ] **Step 7: Verify next-session loading**

Open another fresh Claude Code session. Confirm:
- New memories appear in the system context (the auto-loaded `MEMORY.md` block at the top of the conversation)
- Any new skill appears in the available-skills list

- [ ] **Step 8: Commit any tweaks discovered during the test**

If the test surfaces issues with SKILL.md, find_session.py, or templates, fix them and commit:

```bash
cd ~/.claude
git add skills/reflect/
git commit -m "fix: <specific fix from first real run>"
```

---

## Implementation order

1. Task 1 — bootstrap dir + failing tests
2. Task 2 — implement `find_session.py`, tests pass
3. Task 3 — five templates
4. Task 4 — `SKILL.md` procedure
5. Task 5 — delete old command, commit
6. Task 6 — manual test in fresh session, iterate on findings
