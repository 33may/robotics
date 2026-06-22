---
name: skills_location
description: Custom Claude Code skills live in ~/.claude/skills/, not inside plugins
type: feedback
---

Custom standalone skills go in `~/.claude/skills/<skill-name>/SKILL.md` — NOT inside the plugins system.

**Why:** Plugins and skills are separate things. The user explicitly corrected this — don't conflate them.

**How to apply:** When creating new skills, always use `~/.claude/skills/` as the base directory.
