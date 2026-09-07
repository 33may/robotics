#!/usr/bin/env python3
"""The survey agent — one declaration run over the survey frame.

Before `plan()` the orchestrator used to know NOTHING about the object: the
opening prompt carried `survey view · dir 000`, a metadata gloss, and the
recorded plans show the cost — 2608-aicam's plan literally asks "what kind of
product is it (box, bottle, can, device)?" of an object nobody had described.

This run fills that hole, and ONLY that hole. The tier split is Anton's
(2026-08-27): **the eyes declare, the planner infers.** This agent states what
the object is, its shape as a primitive, how each surface is oriented in the
survey frame, and which features are actually visible. Where the *unseen*
features probably are — "logos on film boxes sit on the front panel" — is
world knowledge, and world knowledge choosing where to point the camera is the
orchestrator's job (it is the project's whole claim). A guess dressed as an
observation would poison exactly that inference.

Mechanically this is a `VlmAgent` spec: same tool loop, same transcript on
disk, same trace events, same Finding as every eyes-tier agent — one way to
see, audited once.
"""
from inspection.eyes.agents.vlm_agent import VlmAgent

#: The survey chain is description, not a detect->crop->read hunt; it needs
#: fewer turns than an inspection before the answer should land.
SURVEY_MAX_TURNS = 8

SURVEY_TASK = ("Declare what this object is and how it sits in this frame, "
               "for planning its inspection.")

SURVEY_RULES = """You are looking at the SURVEY frame — the first, widest view \
of an object a robot is about to inspect. Your job is to DECLARE the scene so \
a blind planner can reason about its geometry. You describe; you do not plan.

Declare, in this order:
- WHAT the object is, as specifically as the pixels allow.
- Its gross shape as a geometric PRIMITIVE: a box (say which faces show), a
  cylinder (wall / top / bottom), a sphere, or a composite of those.
- For EACH surface of that primitive, its orientation in THIS frame. A surface
  turned away from the camera is compressed horizontally — its width in the
  image is cos(angle) of its true width, so judge by WIDTH. Use exactly these
  words per surface:
    "face-on"                            0°   full width, not compressed
    "slightly left" / "slightly right"  30°   ~85% of its width
    "half left"     / "half right"      60°   about half its width
    "edge-on left"  / "edge-on right"   90°   a sliver, barely a line
    "hidden left"   / "hidden right"   >90°   not visible, turned away that way
- Every feature you can actually SEE — print, marks, labels, handles, seams,
  openings — and which surface each one sits on.

Rules:
- Declare only what is visible. Do NOT guess where unseen features might be —
  the planner owns that inference, and a guess dressed as an observation
  poisons it.
- Speak camera-centric: "the left of the frame", "the upper face". Never name
  a cell, a compass direction, or a robot motion.
- You may call a tool by replying with {"tool": NAME, "args": {...}}.
  Tools: detect{phrase}, segment{box}, read_text{box?}, crop{box}, note{text}.
  read_text reads just the region in `box`, or the whole frame without one.
- When done reply with {"evidence": [...], "reasoning": "...", "answer": "..."}
  in that order, where `answer` is the complete declaration. No confidence
  number, no `view` block."""


SURVEY = VlmAgent(rules=SURVEY_RULES, max_turns=SURVEY_MAX_TURNS)


def describe_survey(tools, verbs, writer, model, on_turn=None,
                    max_turns=SURVEY_MAX_TURNS):
    """Run the declaration over the run's survey frame. Finding, or None.

    None when the run has no survey capture — a replay of an old sweep, or a
    live run that skipped the survey. The caller falls back to the geometry
    gloss; a missing declaration must not stop an inspection.
    """
    recs = [v for v in tools._views() if v.cell is None]
    if not recs:
        return None
    return SURVEY.run(tools, verbs, writer, model, task=SURVEY_TASK,
                      cell=None, view_rec=recs[0], max_turns=max_turns,
                      on_turn=on_turn)
