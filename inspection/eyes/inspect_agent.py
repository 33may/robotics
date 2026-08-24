#!/usr/bin/env python3
"""`inspect(cell, task)` — a subagent RUN over ONE captured view.

Not a function that returns a label: it is a small agent with its own tools,
its own context, and a transcript. It reads one frame, may call local verbs
on it, and returns what it saw plus an answer to its task.

What it structurally CANNOT do, and why:
- **Move the robot.** No import of motion/run, asserted by a test. Motion
  budget, approval and safety live with the orchestrator, which owns them.
- **Look at another view.** `view_at`/`views_near` are absent from TOOLS. A
  subagent that wandered the viewsphere would be a second decider, and its
  answer would stop being about one frame.
- **Write geometry.** It holds a FindingWriter, which has no add_view.

Findings that shape the prompt and schema (researched, replicated):
- Field ORDER is the contract: evidence -> reasoning -> answer. Answer-first
  erases the entire chain-of-thought gain (LaMDA GSM8K 14.3 -> 6.1), and JSON
  mode is precisely what causes it — 100% of GPT-3.5 responses put `answer`
  before `reason`.
- NO confidence field. Verbalized confidence is chance-level (AUROC 51.2) and
  clusters in the 80-100% band. If we need calibration, sample.
- Zooming keeps the FULL FRAME in context alongside the crop: global+crop
  beats crop-only (V*Bench 48.68 -> 83.25); crop-only helps small objects and
  nothing else.
- One image per turn, never a contact sheet (10 separate images 97% vs a 4x4
  grid 26.9%).
- Camera-centric only: the subagent says "fragment at the right edge", the
  orchestrator — which knows the pose — turns that into a direction.
"""
import json
import time

# The verb surface handed to the subagent. `crop` is here and `view_at` is
# not: zooming into THIS frame is reading, fetching another frame is moving.
TOOLS = {"detect", "segment", "read_text", "crop", "note"}

# Enforced order of the answer object — see the module docstring.
SCHEMA_ORDER = ("evidence", "reasoning", "answer")

MAX_TURNS = 6

_RULES = """You are inspecting ONE captured camera frame of an object.

Rules:
- Describe only what is visible in THIS frame. Never reason about where the
  robot is, world directions, or what other views would show.
- Speak camera-centric: "at the right edge of the frame", "upper third".
- You may call a tool by replying with {"tool": NAME, "args": {...}}.
  Tools: detect{phrase}, segment{box}, read_text{}, crop{box}, note{text}.
- If you want a view that does not exist, do NOT ask to move — record it with
  note{text}. Something else decides whether it is worth the motion.
- When done reply with {"evidence": [...], "reasoning": "...", "answer": "..."}
  in that order. Give evidence first, reasoning second, the answer last.
- Do not report a confidence number."""


class Finding:
    """What comes back from one inspect run. The summary is what travels."""

    def __init__(self, cell, answer, evidence, reasoning, transcript):
        self.cell = cell
        self.answer = answer
        self.evidence = list(evidence)
        self.reasoning = reasoning
        self.transcript = transcript

    @property
    def summary(self):
        """The distilled form the orchestrator sees — never the transcript.

        Verbatim stays on disk so "why did it say no logo on view 12" remains
        answerable; only this crosses back, so a long run cannot fill the
        orchestrator's context with pixels and tool chatter.
        """
        ev = "; ".join(self.evidence[:3])
        return f"{self.answer} — {ev}" if ev else str(self.answer)


def _boxes_to_pixels(args, model, img):
    """Convert any box argument into frame pixels using the MODEL's convention.

    Each model states geometry its own way — ER-2 emits [ymin, xmin, ymax,
    xmax] normalised 0-1000, y first — so the adapter owns the conversion and
    the agent stays model-agnostic. Without this the agent silently crops the
    wrong region, which looks like a model failure and is not one.
    """
    if "box" not in args:
        return args
    h, w = img.rgb.shape[:2]
    convert = getattr(model, "to_pixel_box", None)
    out = dict(args)
    out["box"] = (convert(args["box"], w, h) if convert
                  else tuple(int(v) for v in args["box"]))
    return out


def _dispatch(name, args, tools, verbs, writer, img):
    """Run one tool call against THIS view. Returns (text, new_image|None).

    `img` is ALWAYS the original full frame, never the last crop: the model
    states boxes in full-frame coordinates because the full frame is what it
    was shown first. Feeding it the previous crop made every second box land
    on a sliver of the first one.
    """
    if name == "detect":
        dets = verbs.detect(img, args.get("phrase", "object"))
        return ("detect: " + (", ".join(f"{d.label} {d.score:.2f} at {d.box}"
                                        for d in dets) or "nothing found"), None)
    if name == "segment":
        seg = verbs.segment(img, tuple(args["box"]))
        return (f"segment: {int(seg.mask.sum())} px "
                f"({seg.mask.mean() * 100:.1f}% of frame)", None)
    if name == "read_text":
        lines = verbs.read_text(img)
        return ("read_text: " + (", ".join(f"{t.text!r} {t.score:.2f}"
                                           for t in lines) or "no legible text"),
                None)
    if name == "crop":
        sub = tools.crop(img, tuple(args["box"]))
        return (f"crop: {sub.text}", sub)
    if name == "note":
        writer.note(args["text"], cell=img.cell)
        return ("note recorded", None)
    return (f"unknown tool {name!r} — tools are {sorted(TOOLS)}", None)


def inspect_view(tools, verbs, writer, model, cell, task, hypothesis=None,
                 max_turns=MAX_TURNS):
    """Run the inspection subagent over one captured view.

    `tools` is a ViewTools bound to the run, `verbs` a LocalVerbs, `writer` a
    FindingWriter (the only writer this tier may hold), `model` anything with
    `.respond(parts, schema)`.
    """
    img = tools.get_view(cell)
    prompt = (f"{_RULES}\n\nView: {img.text}\nTask: {task}"
              + (f"\nCurrent hypothesis: {hypothesis}" if hypothesis else ""))

    transcript = {"cell": list(cell) if cell else None, "task": task,
                  "hypothesis": hypothesis, "prompt": prompt,
                  "t": time.time(), "turns": []}
    # The full frame stays first in every request: the crop is additional
    # context, never a replacement, and the most relevant image belongs at the
    # START of the sequence (position alone is worth up to a 41% swing).
    parts = [prompt, ("full frame", img.rgb)]
    current = img
    done = {}                       # tool call -> its result, to refuse repeats

    for _ in range(max_turns):
        reply = model.respond(parts, schema={"order": SCHEMA_ORDER})
        transcript["turns"].append(dict(reply) if isinstance(reply, dict)
                                   else {"raw": str(reply)})
        if isinstance(reply, dict) and "tool" in reply:
            name = reply["tool"]
            args = _boxes_to_pixels(reply.get("args", {}) or {}, model, img)
            key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            if key in done:
                # Measured: ER-2 called detect four times with identical
                # arguments and stalled. Removing distractor actions is the
                # single biggest lever on agent loops (AgentOccam WebArena
                # 16.5 -> 25.8%), so say it plainly instead of re-running.
                result = (f"{name} was already called with these arguments and "
                          f"returned: {done[key]} — the result will not change. "
                          f"Use a different tool or give your answer now.")
                transcript["turns"].append({"result": result})
                parts.append(f"[{name}] {result}")
                continue
            try:
                # Always the original frame — see _dispatch's docstring.
                result, new_img = _dispatch(name, args,
                                            tools, verbs, writer, img)
                done[key] = result
            except Exception as e:
                # A bad box is not a crash, it is a message. Categorical
                # failure feedback raises success AND cuts retries (LLM3:
                # 40% -> 60% with fewer calls); an exception here would throw
                # away a run over one malformed argument.
                h, w = img.rgb.shape[:2]
                result, new_img = (f"{name} failed: {e}. The frame is "
                                   f"{w}x{h}; boxes are [x0, y0, x1, y1] in "
                                   f"FULL-FRAME pixels."), None
            transcript["turns"].append({"result": result})
            parts.append(f"[{name}] {result}")
            if new_img is not None:
                current = new_img
                parts.append(("crop of the same frame", new_img.rgb))
            continue

        answer = (reply or {}).get("answer", "unknown")
        finding = Finding(cell, answer, (reply or {}).get("evidence", []),
                          (reply or {}).get("reasoning", ""), transcript)
        transcript["answer"] = answer
        writer.add_finding(cell, finding.summary, json.dumps(transcript, indent=1))
        return finding

    # Out of turns: that is a finding about the model, not an exception.
    transcript["answer"] = "unknown"
    finding = Finding(cell, "unknown", [], f"no answer within {max_turns} turns",
                      transcript)
    writer.add_finding(cell, finding.summary, json.dumps(transcript, indent=1))
    return finding
