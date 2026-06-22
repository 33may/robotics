---
name: feedback_one_liner_status
description: Before any action (bash, edit, agent dispatch), state in one line what you're doing and why. Don't drop it after a few turns
type: feedback
originSessionId: 672e223a-9b37-4a53-84d3-29a68457f36d
---
Before every non-trivial tool call (bash command that does work, file edit, agent dispatch, multi-tool batch), prefix with **one short line** stating what you're about to do and why. Format: action verb + object + reason. Example: "Re-rendering verify plot at the new clip so you can pick" or "Killing v017 tmux on remote so v018 doesn't OOM".

**Why:** the user said verbatim, "as I explained in every message I one to see one liner what are you doing now and why" — meaning they had already asked once earlier in the session and I'd drifted back into silent tool-batching. Drift is the failure mode, not the initial omission. Without the one-liner, the user can't intercept a wrong direction before the work runs.

**How to apply:**
- The one-liner sits OUTSIDE the tool call — in the prose immediately preceding it. Not in a `description` field, not in a comment.
- Keep it under ~12 words. Multiple parallel tool calls share one combined one-liner.
- Don't drop it after the first few turns of a long session. The user notices when it slips.
- It's not a substitute for prose explanation when the user asks "why" or "what" — it's the always-on default before action.
- Recovery clue: if the user asks "what are you doing?" mid-session, you've already drifted; reset the discipline immediately.
