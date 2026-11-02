#!/usr/bin/env python3
"""The evidence agent — hunt pixels that back a verdict, or say so honestly.

At answer time the orchestrator cites cells and states EXACTLY what a picture
must show ("the ILFORD wordmark, fully in frame"); one hunt runs per citation
(`brain/loop.py:_collect_evidence`). The tier split holds (Anton 2026-08-27):
the planner owns the inference of WHERE the proof is, this agent only hunts
in ONE frame — locate, VERIFY, frame — and "not found" is a fully successful
hunt, never a stretch. It is deliberately given no `hypothesis` and never the
verdict: a hunter told the expected answer is under confirmation pressure,
and the honest miss is the whole point.

The citation mechanism: the loop numbers every image it hands over
(`vlm_agent.py` — "this is image N"), the emit echoes `evidence_image: N`,
and `resolve()` turns the number back into the transcript's saved path. The
model echoes a digit it just read; no model ever authors a file path.
"""
from inspection.eyes.agents.vlm_agent import VlmAgent

#: A hunt is a detect->crop->read chain like an inspection, not a survey
#: description — it keeps the inspection budget (see inspect_agent.MAX_TURNS).
EVIDENCE_MAX_TURNS = 16

EVIDENCE_RULES = """You are hunting for ONE specific thing in ONE captured \
camera frame. The Task names the target. Your job is to find it, verify it, \
and frame it — or to say plainly that it is not here. You do not judge what \
the target means for any larger question; you hunt, you do not decide.

Work in this order:
- LOCATE: find where the target sits in this frame. detect{phrase} and
  read_text{box?} are your search tools; crop{box} to look closer.
- VERIFY: prove it is the named target and not a lookalike. A claimed
  wordmark, label or code must be READ — quote its text verbatim in your
  evidence. A claimed mark or feature must be described by the detail that
  distinguishes it. If you could not verify it, you have not found it.
- FRAME: crop to the best framing of the verified target — the WHOLE target
  in frame with a small margin, clipped at no edge, as close as it can be
  while staying whole. If the full frame is already the best framing, no
  crop is needed.

Images are numbered: the full frame is image 0, and each crop you take is
announced back to you with its image number. When you answer, cite the
number of the image that best shows the target.

Rules:
- Only THIS frame. Never reason about where the robot is or what other
  views would show. Speak camera-centric: "upper third", "at the left edge".
- You may call a tool by replying with {"tool": NAME, "args": {...}}.
  Tools: detect{phrase}, segment{box}, read_text{box?}, crop{box}.
  read_text reads just the region in `box`, or the whole frame without one.
- The target either is verifiably here or it is not. A near-match, a
  partial glimpse, or a guess is NOT a find. Answering "not found" honestly
  is a fully successful hunt — never stretch.
- When done reply with {"evidence": [...], "reasoning": "...",
  "answer": "...", "evidence_image": N} in that order. `answer` starts
  with "found — " followed by what the cited image shows in one clause, or
  "not found — " followed by what is here instead and where you looked.
  Omit `evidence_image` when not found.
- Do not report a confidence number. No `view` block."""


def normalise_evidence_image(raw):
    """Coerce a citation into an int index, or None. Never raises.

    Lenient by standing decision (2026-08-26): accepts 2, "2", "image 2".
    Anything unparseable is None and the resolver falls back — a good hunt
    must not be discarded over a garbled citation.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw == int(raw):
        return int(raw)
    if isinstance(raw, str):
        digits = "".join(c for c in raw if c.isdigit())
        if digits:
            return int(digits)
    return None


EVIDENCE = VlmAgent(rules=EVIDENCE_RULES, kind="evidence",
                    # `note` feeds the planner's ledger; a hunt reports to a
                    # verdict. Dropping it is the AgentOccam lever applied to
                    # this variant's action set.
                    tool_names=frozenset({"detect", "segment", "read_text",
                                          "crop"}),
                    emit=("evidence", "reasoning", "answer", "evidence_image"),
                    extras={"evidence_image": normalise_evidence_image},
                    max_turns=EVIDENCE_MAX_TURNS)


def _images_of(transcript):
    """The saved images in hand-over order: frame first, then each crop.

    The same join `Brain._images_of` does — index i here IS "image i" as the
    loop announced it, because the loop numbers only saved images.
    """
    out = [transcript["image"]] if transcript.get("image") else []
    out += [t["image"] for t in transcript.get("turns", []) if t.get("image")]
    return out


def hunt_evidence(tools, verbs, writer, model, cell, find, on_turn=None):
    """One hunt over one captured view. Returns the resolved evidence record.

    Everything model-authored in the record is verbatim (`report`); every
    path is harness-resolved from the transcript. The record ships into
    answer.json as-is:
      {cell, find, found, report, crop, frame, transcript, transcript_id}
    """
    f = EVIDENCE.run(tools, verbs, writer, model, task=f"Find: {find}",
                     cell=cell, on_turn=on_turn)
    images = _images_of(f.transcript)
    answer = str(f.answer or "")
    head = answer.strip().casefold()
    cited = f.extras.get("evidence_image")
    valid = cited is not None and 0 <= cited < len(images)
    # "not found" first — it also starts with "found" backwards spelled
    # forwards; then "found"; an answer that says neither is found iff a
    # citation resolves. Lenient always, never a hard failure.
    if head.startswith("not found") or f.exhausted:
        found = False
    elif head.startswith("found"):
        found = True
    else:
        found = valid
    if found:
        # Fallback: the last image the hunt took — where its looking landed.
        # The frame itself is legitimate when the target fills the frame.
        crop = images[cited] if valid else (images[-1] if images else None)
    else:
        crop = None
    return {"cell": list(cell), "find": find, "found": found,
            "report": answer,
            "crop": crop,
            # The "we looked here" receipt — present even on a miss.
            "frame": f.transcript.get("image"),
            "transcript": f.transcript_rel,
            # The record id, not just the path: this is what the answer cites
            # (`AnswerRecord.evidence_images[].transcript_id`).
            "transcript_id": f.transcript_id}
