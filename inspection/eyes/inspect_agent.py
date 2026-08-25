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
#
# `framing` trails the triple deliberately. The CoT constraint is that the
# ANSWER must not precede the REASONING; nothing is harmed by an appendix
# written after the answer, and putting framing earlier would push `answer`
# out of last place for no gain.
SCHEMA_ORDER = ("evidence", "reasoning", "answer", "framing")

# Raised 10 -> 16 (Anton 2026-08-25, duck run). Note what this does NOT fix:
# in `2508-aiduck` the only view that exhausted its budget ([3, 0], twice)
# spent it calling the STUB detector, which answers every phrase with a fixed
# box at (10, 10, 100, 100). The model kept trying to reconcile a fake box
# with real pixels and never cropped. More turns buy a longer detect->crop->
# read chain on a curved surface; they do not buy a way out of a lying tool.
MAX_TURNS = 16

_RULES = """You are inspecting ONE captured camera frame of an object.

Rules:
- Describe only what is visible in THIS frame. Never reason about where the
  robot is, world directions, or what other views would show.
- Speak camera-centric: "at the right edge of the frame", "upper third".
- You may call a tool by replying with {"tool": NAME, "args": {...}}.
  Tools: detect{phrase}, segment{box}, read_text{}, crop{box}, note{text}.
- If you want a view that does not exist, do NOT ask to move — record it with
  note{text}. Something else decides whether it is worth the motion.
- When done reply with {"evidence": [...], "reasoning": "...", "answer": "...",
  "framing": {...}} in that order. Give evidence first, reasoning second, the
  answer third.
- Do not report a confidence number.

`framing` reports the VIEWING GEOMETRY of whatever the task is about, as seen
in this frame. Three short strings:
  {"target":  where it sits in the frame, or "not visible"
   "facing":  how square-on its surface is and WHICH WAY IT TURNS AWAY —
              "face-on", "turning away to the left", "edge-on to the right"
   "better":  which way the camera would have to shift to frame it better —
              one of: left, right, closer, higher, lower, none}
Both directions are FRAME directions: left means the left of this image. You
are describing a viewpoint, not commanding a robot — never name a cell, an
angle, or a compass direction, and never say the arm should move. Something
else owns motion; it only needs to know what you can see and how well."""


#: Keys we ask for in `framing`, in the order they read best.
FRAMING_KEYS = ("target", "facing", "better")


def normalise_framing(raw):
    """Coerce whatever the model produced into `{str: str}`. Never raises.

    Two failure modes are already on record and both land here: a model that
    returns a bare string where an object was asked for, and one that returns
    an object where the consumer assumed a string (which took the trace panel
    down with React #31 on 2026-08-25). Normalising once, at the tier boundary,
    means neither the orchestrator's renderer nor the UI has to type-check.

    `framing` is advisory, so a malformed one is dropped rather than fought
    over — an inspection that saw the object clearly must not be discarded
    because its appendix was the wrong shape.
    """
    if isinstance(raw, str) and raw.strip():
        return {"target": raw.strip()}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k in FRAMING_KEYS:
        v = raw.get(k)
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        if v not in (None, ""):
            out[k] = str(v).strip()
    return out


class Finding:
    """What comes back from one inspect run. The summary is what travels."""

    def __init__(self, cell, answer, evidence, reasoning, transcript,
                 framing=None):
        self.cell = cell
        self.answer = answer
        self.evidence = list(evidence)
        self.reasoning = reasoning
        self.transcript = transcript
        # Viewing geometry of the target in THIS frame — camera-centric, and
        # the only part of a finding that is about the VIEW rather than the
        # object. The orchestrator turns it into a direction; see brain/render.
        self.framing = normalise_framing(framing)

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


def _save_seen(tools, img, tag, kind="crops"):
    """Persist EXACTLY the pixels handed to the model. Run-relative path back.

    Two reasons this writes a file instead of pointing at the capture on disk:

    - Crops only ever existed in memory, so the most interesting images in a
      run — the ones the model chose to look closer at — were the only ones
      nobody could review afterwards.
    - The full frame on disk is RAW. `ViewTools.get_view` uprights it in
      memory (tools.py:126), so a viewer pointed at `<cap>/rgb.png` shows a
      half-turn capture upside down — a different image from the one the model
      answered about. For an observability surface that is the worst possible
      failure: it looks right and it is lying.

    Best-effort: a failed write must not end an otherwise good inspection.
    """
    try:
        import cv2
        d = tools._store.path / kind
        d.mkdir(parents=True, exist_ok=True)
        name = f"{tag}.png"
        cv2.imwrite(str(d / name), cv2.cvtColor(img.rgb, cv2.COLOR_RGB2BGR))
        return f"eyes/{kind}/{name}"
    except Exception:                            # noqa: BLE001
        return None


def inspect_view(tools, verbs, writer, model, cell, task, hypothesis=None,
                 max_turns=MAX_TURNS, answer_schema=None, on_turn=None):
    """Run the inspection subagent over one captured view.

    `tools` is a ViewTools bound to the run, `verbs` a LocalVerbs, `writer` a
    FindingWriter (the only writer this tier may hold), `model` anything with
    `.respond(parts, schema)`.

    `on_turn(n, max_turns, event, **fields)` is called as the loop runs, so a
    watcher can show it happening instead of a frozen gap. This subagent takes
    tens of seconds and used to surface only when it landed — which made the
    slowest part of a run the only invisible one.
    """
    def _tick(n, event, **fields):
        if on_turn is not None:
            try:
                on_turn(n, max_turns, event, **fields)
            except Exception:                        # noqa: BLE001
                pass          # a watcher must never break the thing it watches
    img = tools.get_view(cell)
    # `answer_schema` is the orchestrator saying what SHAPE of readout it wants
    # back ("yes|no", "the text, verbatim", "count of marks"). It is enforced by
    # prompt text and nothing else: `models.respond`'s schema argument is only a
    # truthiness flag that switches Gemini into JSON mode (models.py:111), so
    # the shape has to be stated here. The field ORDER is never negotiable —
    # answer-before-reasoning erases 100% of the CoT gain — so a caller may
    # constrain the answer's contents, never the order it arrives in.
    prompt = (f"{_RULES}\n\nView: {img.text}\nTask: {task}"
              + (f"\nRequired shape of `answer`: {answer_schema}"
                 if answer_schema else "")
              + (f"\nCurrent hypothesis: {hypothesis}" if hypothesis else ""))

    transcript = {"cell": list(cell) if cell else None, "task": task,
                  "answer_schema": answer_schema,
                  # The uprighted array, not the raw capture — see _save_seen.
                  "image": _save_seen(tools, img, img.cap_dir, kind="frames"),
                  "view_text": img.text,
                  "hypothesis": hypothesis, "prompt": prompt,
                  "t": time.time(), "turns": []}
    # The full frame stays first in every request: the crop is additional
    # context, never a replacement, and the most relevant image belongs at the
    # START of the sequence (position alone is worth up to a 41% swing).
    parts = [prompt, ("full frame", img.rgb)]
    current = img
    done = {}                       # tool call -> its result, to refuse repeats

    _tick(0, "start", cell=list(cell) if cell else None, task=task)
    for turn_i in range(max_turns):
        _tick(turn_i + 1, "thinking")
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
            _tick(turn_i + 1, "tool", tool=name, args=args,
                  result=str(result)[:300])
            turn_rec = {"result": result}
            if new_img is not None:
                turn_rec["image"] = _save_seen(
                    tools, new_img,
                    f"{img.cap_dir}_{len(transcript['turns']):02d}")
                # Announce the crop the moment it exists on disk. Without this
                # the live view streams the subagent's WORDS but none of its
                # pictures, which is backwards for a vision agent — the crop
                # it chose is the most informative thing it does.
                _tick(turn_i + 1, "image", image=turn_rec["image"], tool=name)
            transcript["turns"].append(turn_rec)
            # ORDER IS LOAD-BEARING: the result text goes in BEFORE the crop
            # pixels, never after. Persisting crops is a side effect and must
            # not disturb the sequence the model reads.
            parts.append(f"[{name}] {result}")
            if new_img is not None:
                current = new_img
                parts.append(("crop of the same frame", new_img.rgb))
            continue

        answer = (reply or {}).get("answer", "unknown")
        framing = normalise_framing((reply or {}).get("framing"))
        _tick(turn_i + 1, "done", answer=answer, framing=framing)
        finding = Finding(cell, answer, (reply or {}).get("evidence", []),
                          (reply or {}).get("reasoning", ""), transcript,
                          framing=framing)
        transcript["answer"] = answer
        transcript["framing"] = framing
        writer.add_finding(cell, finding.summary, json.dumps(transcript, indent=1))
        return finding

    # Out of turns: that is a finding about the model, not an exception.
    _tick(max_turns, "exhausted")
    transcript["answer"] = "unknown"
    # Say so in the framing too. "read failed here" and "looked, saw nothing
    # here" are opposite facts for a planner deciding where to go next, and a
    # blank ledger row would render them identically.
    finding = Finding(cell, "unknown", [], f"no answer within {max_turns} turns",
                      transcript,
                      framing={"target": f"unread — the inspection of this view "
                                         f"ran out of turns after {max_turns}"})
    writer.add_finding(cell, finding.summary, json.dumps(transcript, indent=1))
    return finding
