---
name: feedback_no_destructive_inplace
description: For dataset/file transforms — default to writing a new artifact. Never rename-over-source unless user explicitly asks for in-place
type: feedback
originSessionId: 672e223a-9b37-4a53-84d3-29a68457f36d
---
Default to producing a new artifact (new dataset name, new file path) for any data transform — bake, strip, augment, recalibrate, etc. Never rename-over-source ("in-place via temp + move") even when it seems convenient unless the user explicitly asks for in-place.

**Why:** during the v016 depth-bake the user said "STOP, DONT swap, make new dataset" mid-run because in-place swap would have destroyed the lossless packed-PNG original. Even with a `.old` backup, the cognitive load of "did I just lose 16 min of bake compute + my source of truth?" is unacceptable. Reversibility costs disk; foot-guns cost a session.

**How to apply:**
- For dataset CLIs: prefer `--out-repo-id=...` style, with the safer side as default. If an in-place flag exists, gate it behind explicit user opt-in.
- For file transforms (strip_feature, add_gripper_depth, bake_packed_depth, recalibrate): write to a new path; let the user delete the old one once they've verified.
- If the user has explicitly asked for in-place, fine — but echo the destructive step back ("this will rename X → X.old and replace X with the baked version") before kicking it off.
