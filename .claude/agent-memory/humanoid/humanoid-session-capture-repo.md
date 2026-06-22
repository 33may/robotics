---
name: humanoid-session-capture-repo
description: Where humanoid session transcripts are captured and rendered for /reflect and review
metadata:
  node_type: memory
  type: project
  originSessionId: 7039e4db-0875-47de-a24d-4989de40814b
---

Humanoid session capture pipeline writes to `/home/may33/Documents/vbti/vbti/humanoid/sessions/`:

- `raw/` — raw `.jsonl` copies
- `rendered/` — `YYYY-MM-DD_<shortid>.md` plain-text transcripts (canonical record for `/reflect` and review)
- `index.json` — manifest with `last_run`, per-session metadata

**Why:** Capture logic lives in its own session-capture repo under projects; we don't duplicate it in humanoid command files. The rendered markdown copies in `vbti/humanoid/sessions/rendered/` are the project-canonical, human-readable record — easier to grep, share, and review than the raw `~/.claude/projects/.../<uuid>.jsonl`.

**How to apply:**
- For `/reflect` across multiple sessions in a day, list `vbti/humanoid/sessions/rendered/YYYY-MM-DD_*.md` — filename pattern makes today's batch trivial.
- If a session is missing from `rendered/` (capture not yet run), fall back to `~/.claude/skills/reflect/find_session.py --path <jsonl>`.
- Workflow commands (`/work`, `/reflect`, `/manager`, `/code`) should integrate with this capture, never reimplement it.

Related: [[humanoid-daily-note-path]], [[humanoid-daily-note-style]], [[humanoid-summer-2026-plan]]
