---
name: feedback-explain-plainly
description: Explain technical spaces to Anton plainly — anchor to what he already knows, strip jargon from the first pass
metadata:
  type: feedback
---

**When explaining a technical space to Anton, lead with the plain-language mental model anchored to something he already knows — not the textbook jargon.** Strip ATE/RPE/Umeyama-style vocabulary out of the *first* pass; introduce a term only after the idea behind it lands.

**Why:** twice in one session (2026-07-10 localization) Anton pushed back on over-dressed explanations — "is it like colmap or what, I understand nothing" and "again, what are the metrics… or what?". The COLMAP anchor (build-a-map-once, then find-yourself-in-it) and the plain "meter error between GT and estimate" landed instantly where the jargon hadn't. He's the architect making the call; a decision buried in jargon is obscured, not informed.

**How to apply:** when introducing methods/metrics/unfamiliar machinery, open with an analogy to a tool/concept he already owns (COLMAP, a pipeline he's built), give the one-line intuition, then layer in precision on top. If he says "again" / "I understand nothing", that's the signal I over-dressed it — restate *plainer*, don't pile on more detail. Related: [[feedback-show-dont-tell]].
