#!/usr/bin/env python3
"""`inspect(cell, task)` — the inspection variant of the VLM subagent.

The loop lives in `vlm_agent.py`; this module is the inspection SPEC: its
rules text, its `view` appendix vocabulary, and a thin `inspect_view` wrapper
that keeps the tier boundary's signature stable (`brain/loop.py` imports only
this).

What every variant structurally CANNOT do, and why (asserted by tests over
`vlm_agent.py`):
- **Move the robot.** No import of motion/run. Motion budget, approval and
  safety live with the orchestrator, which owns them.
- **Look at another view.** `view_at`/`views_near` are absent from the tier's
  TOOLS. A subagent that wandered the viewsphere would be a second decider,
  and its answer would stop being about one frame.
- **Write geometry.** It holds a FindingWriter, which has no add_view.
"""
from inspection.eyes.agents.vlm_agent import (TOOLS, Finding,  # noqa: F401 — re-
                                       VlmAgent)        # exports; tests and
                                                        # callers import here

# Raised 10 -> 16 (Anton 2026-08-25, duck run). Note what this does NOT fix:
# in `2508-aiduck` the only view that exhausted its budget ([3, 0], twice)
# spent it calling the STUB detector, which answers every phrase with a fixed
# box at (10, 10, 100, 100). More turns buy a longer detect->crop->read chain
# on a curved surface; they do not buy a way out of a lying tool.
MAX_TURNS = 16

_RULES = """You are inspecting ONE captured camera frame of an object.

Rules:
- Describe only what is visible in THIS frame. Never reason about where the
  robot is, world directions, or what other views would show.
- Speak camera-centric: "at the right edge of the frame", "upper third".
- You may call a tool by replying with {"tool": NAME, "args": {...}}.
  Tools: detect{phrase}, segment{box}, read_text{box?}, crop{box}, note{text}.
  read_text reads just the region in `box`, or the whole frame without one —
  give a box when the text is small.
- If you want a view that does not exist, do NOT ask to move — record it with
  note{text}. Something else decides whether it is worth the motion.
- When done reply with {"evidence": [...], "reasoning": "...", "answer": "...",
  "view": {...}} in that order. Give evidence first, reasoning second, the
  answer third.
- Do not report a confidence number.

`view` is an appendix about the VIEWPOINT — what this frame offered, for
whoever plans the next one. Two short strings:
  {"saw":  what the task's subject looked like from here, one clause,
           including how its surface faces the camera
   "recommendation":  where the camera should move for a better look}

For `recommendation`: a surface turned away from the camera is compressed
horizontally — its width in the image is cos(angle) of its true width.
Estimate the turn from the WIDTH you see, then recommend moving that far,
toward the side the surface turns toward. Use these phrases; the degrees are
what the words mean:

  "none"                                    this frame already shows it squarely
  "slightly left" / "slightly right"   30°  mildly compressed, ~85% of its width
  "half left"     / "half right"       60°  about half its width shows
  "far left"      / "far right"        90°  only a sliver, nearly edge-on
  "around left"   / "around right"    150°  cannot see it at all — go around that way

Say the phrase, not an angle. left/right are directions in THIS frame. You are
describing a viewpoint, not commanding a robot — never name a cell or a compass
direction, and never say the arm should move. Something else owns motion; it
only needs what you saw and which way is better."""


#: Keys we ask for in `view`, in the order they read best.
VIEW_KEYS = ("saw", "recommendation")

#: `recommendation` values that mean "nothing to recommend" — dropped at the
#: boundary so downstream renders no line at all instead of "recommends: none".
_NO_RECOMMENDATION = {"none", "n/a", "stay", ""}


def normalise_view(raw):
    """Coerce whatever the model produced into `{str: str}`. Never raises.

    Two failure modes are already on record and both land here: a model that
    returns a bare string where an object was asked for, and one that returns
    an object where the consumer assumed a string (which took the trace panel
    down with React #31 on 2026-08-25). Normalising once, at the tier boundary,
    means neither the orchestrator's renderer nor the UI has to type-check.

    `view` is advisory, so a malformed one is dropped rather than fought
    over — an inspection that saw the object clearly must not be discarded
    because its appendix was the wrong shape. The recommendation phrase is
    deliberately NOT validated against the vocabulary: the contract is prompt
    text on both sides (Anton 2026-08-26), and the orchestrator reads whatever
    was written, so a near-miss phrase is still worth carrying.
    """
    if isinstance(raw, str) and raw.strip():
        return {"saw": raw.strip()}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k in VIEW_KEYS:
        v = raw.get(k)
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        if v not in (None, ""):
            out[k] = str(v).strip()
    if out.get("recommendation", "").lower() in _NO_RECOMMENDATION:
        out.pop("recommendation", None)
    return out


INSPECT = VlmAgent(rules=_RULES,
                   emit=("evidence", "reasoning", "answer", "view"),
                   extras={"view": normalise_view},
                   max_turns=MAX_TURNS)

#: The enforced reply order — re-exported so the CoT-order test keeps pinning
#: it where the rules text lives.
SCHEMA_ORDER = INSPECT.emit


def inspect_view(tools, verbs, writer, model, cell, task, hypothesis=None,
                 max_turns=MAX_TURNS, answer_schema=None, on_turn=None):
    """Run the inspection subagent over one captured view. Returns a Finding.

    Thin: the loop is `INSPECT.run`. The one behaviour that stays here is the
    exhausted-view note — "read failed here" and "looked, saw nothing here"
    are opposite facts for a planner deciding where to go next, and a blank
    ledger row would render them identically. That is a fact about the VIEW
    LEDGER, so the text lives with the view vocabulary, not in the loop.
    """
    f = INSPECT.run(tools, verbs, writer, model, task=task, cell=cell,
                    hypothesis=hypothesis, answer_schema=answer_schema,
                    on_turn=on_turn, max_turns=max_turns)
    if f.exhausted:
        f.extras["view"] = {"saw": f"unread — the inspection of this view "
                                   f"ran out of turns after {max_turns}"}
    return f
