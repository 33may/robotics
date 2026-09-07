#!/usr/bin/env python3
"""The one VLM subagent loop — every eyes-tier agent is a spec over this.

A subagent here is not a function that returns a label: it is a small agent
with its own tools, its own context, and a transcript. It reads ONE frame, may
call local verbs on it, and returns what it saw plus an answer to its task.
There are several of them — inspect, survey, evidence — and they differ in
DATA (rules text, tool subset, emit fields, turn budget), never in control
flow. The loop lives here once; a variant module holds its rules and a
`VlmAgent` spec (Anton 2026-08-27: "don't rebuild the agentic loop every
time").

Findings that shape the loop (researched, replicated — these are properties of
the LOOP, which is why they move with it and no spec can disable them):
- Field ORDER is the contract: evidence -> reasoning -> answer. Answer-first
  erases the entire chain-of-thought gain (LaMDA GSM8K 14.3 -> 6.1), and JSON
  mode is precisely what causes it — 100% of GPT-3.5 responses put `answer`
  before `reason`. The constructor asserts this for every variant ever built.
- NO confidence field. Verbalized confidence is chance-level (AUROC 51.2) and
  clusters in the 80-100% band. If we need calibration, sample.
- Zooming keeps the FULL FRAME in context alongside the crop: global+crop
  beats crop-only (V*Bench 48.68 -> 83.25); crop-only helps small objects and
  nothing else.
- One image per turn, never a contact sheet (10 separate images 97% vs a 4x4
  grid 26.9%).
- Camera-centric only: the subagent says "fragment at the right edge", the
  orchestrator — which knows the pose — turns that into a direction.
- Repeated identical tool calls are refused with the cached result (measured:
  ER-2 called detect four times with identical arguments and stalled).
- Tool failures are categorical messages, never exceptions (LLM3 `2403.11552`:
  40% -> 60% success AND fewer retries from naming the failure class).

Images are NUMBERED as they are handed over — "image 0" is the full frame,
each saved crop announces "this is image N". Inert extra text for most
variants; for the evidence agent it is the citation mechanism: the model
echoes a number it just read, the harness resolves number -> path from the
transcript, and no model ever authors a file path.
"""
import json
import time

# The verb surface of the TIER. Fixed, not injectable: "crop is here and
# view_at is not" is a structural property of the eyes tier — zooming into
# THIS frame is reading, fetching another frame is moving. A spec chooses a
# SUBSET; nothing can add a verb without editing this file, which is where
# the no-motion / no-cross-view test points (test_vlm_agent.py).
TOOLS = frozenset({"detect", "segment", "read_text", "crop", "note"})

#: The mandatory head of every emit, in the only order that keeps the CoT
#: gain. Variant appendices trail it.
EMIT_CORE = ("evidence", "reasoning", "answer")


class Finding:
    """What comes back from one run. The summary is what travels."""

    def __init__(self, cell, answer, evidence, reasoning, transcript,
                 extras=None, exhausted=False):
        self.cell = cell
        self.answer = answer
        self.evidence = list(evidence)
        self.reasoning = reasoning
        self.transcript = transcript
        #: Normalised appendix fields, keyed by their emit name. What a
        #: variant declared beyond the core triple lands here.
        self.extras = dict(extras or {})
        #: True when the run hit its turn budget without answering — a fact
        #: about the model, not an exception, and cleaner than string-matching
        #: the reasoning text.
        self.exhausted = exhausted
        #: Run-relative path of the persisted transcript, set by the loop
        #: after `writer.add_finding`. The audit trail behind any claim a
        #: caller builds on this finding (the evidence agent cites it).
        self.transcript_rel = None

    @property
    def view(self):
        """The one appendix consumed by name across the tier boundary —
        `brain/loop.py` writes it into the move-menu ledger. Variants that
        declare no `view` extra report {} here, structurally."""
        return self.extras.get("view", {})

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


def _dispatch(name, args, tools, verbs, writer, img, fmt_box):
    """Run one tool call against THIS view. Returns (text, new_image|None).

    `img` is ALWAYS the original full frame, never the last crop: the model
    states boxes in full-frame coordinates because the full frame is what it
    was shown first. Feeding it the previous crop made every second box land
    on a sliver of the first one.

    `fmt_box` renders a pixel box in the MODEL's own convention. Every box
    the model reads goes through it — detect results, crop echoes — because
    a model will echo back whatever coordinates it was shown. On 2708-aicam
    (f04) detect printed pixels while the adapter decoded crops as 0-1000
    y-first: the model echoed the detect box into crop, landed on background,
    and burned 16 turns retrying variations of it.
    """
    if name == "detect":
        dets = verbs.detect(img, args.get("phrase", "object"))
        return ("detect: " + (", ".join(f"{d.label} {d.score:.2f} at "
                                        f"{fmt_box(d.box)}"
                                        for d in dets) or "nothing found"), None)
    if name == "segment":
        seg = verbs.segment(img, tuple(args["box"]))
        return (f"segment: {int(seg.mask.sum())} px "
                f"({seg.mask.mean() * 100:.1f}% of frame)", None)
    if name == "read_text":
        # Optional box: OCR just that region. Full-frame OCR on 848x480
        # cannot resolve small label text (2708-aicam f04 read garbage, and
        # the argless call could never be retried past the dedup) — a region
        # read is sharper AND each box is its own call.
        target = tools.crop(img, tuple(args["box"])) if args.get("box") else img
        lines = verbs.read_text(target)
        where = f" in {fmt_box(target.box)}" if target.box else ""
        return (f"read_text{where}: "
                + (", ".join(f"{t.text!r} {t.score:.2f}" for t in lines)
                   or "no legible text"), None)
    if name == "crop":
        sub = tools.crop(img, tuple(args["box"]))
        # Echo the region in the model's convention, not just the pixel
        # provenance — the model must be able to see WHERE its crop landed
        # in coordinates it can act on.
        return (f"crop: {sub.text} · region {fmt_box(sub.box)}", sub)
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
        d = tools._run.path / "eyes" / kind
        d.mkdir(parents=True, exist_ok=True)
        name = f"{tag}.png"
        cv2.imwrite(str(d / name), cv2.cvtColor(img.rgb, cv2.COLOR_RGB2BGR))
        return f"eyes/{kind}/{name}"
    except Exception:                            # noqa: BLE001
        return None


class VlmAgent:
    """A kind of eyes-tier agent: config fixed at import time, next to the
    rules text it belongs to. `run()` is one look at one frame; the run-bound
    resources (tools/verbs/writer/model) arrive there because a Brain builds
    them per run while specs are constants."""

    def __init__(self, *, rules, tool_names=TOOLS, emit=EMIT_CORE,
                 extras=None, max_turns=16):
        # Import-time invariants: a broken spec kills the import, so no
        # variant can silently reorder the schema the way JSON mode does.
        assert tuple(emit[:3]) == EMIT_CORE, "answer never precedes reasoning"
        assert "confidence" not in emit, "verbalized confidence ~ chance"
        assert set(emit[3:]) == set(extras or {}), \
            "every appendix trails answer and has a normaliser"
        assert frozenset(tool_names) <= TOOLS, \
            f"unknown tools {set(tool_names) - TOOLS}"
        self.rules = rules
        self.tool_names = frozenset(tool_names)
        self.emit = tuple(emit)
        self.extras = dict(extras or {})
        self.max_turns = max_turns

    def run(self, tools, verbs, writer, model, *, task, cell=None,
            view_rec=None, answer_schema=None, hypothesis=None, on_turn=None,
            max_turns=None):
        """One subagent run over one captured view. Returns a Finding.

        `tools` is a ViewTools bound to the run, `verbs` a LocalVerbs,
        `writer` a FindingWriter (the only writer this tier may hold — every
        exit persists through it; there is no unaudited way to see), `model`
        anything with `.respond(parts, schema)`. `view_rec` hands in a
        ViewRecord directly for frames with no cell address (the survey).

        `on_turn(n, max_turns, event, **fields)` is called as the loop runs,
        so a watcher can show it happening instead of a frozen gap. This
        takes tens of seconds and used to surface only when it landed — which
        made the slowest part of a run the only invisible one.
        """
        max_turns = self.max_turns if max_turns is None else max_turns

        def _tick(n, event, **fields):
            if on_turn is not None:
                try:
                    on_turn(n, max_turns, event, **fields)
                except Exception:                    # noqa: BLE001
                    pass  # a watcher must never break the thing it watches

        img = tools.get_view(view_rec if view_rec is not None else cell)
        # THE BOX BOUNDARY, both directions: every box the model WRITES is
        # converted in via `to_pixel_box`; every box it READS is rendered via
        # `to_model_box`. Round-trip is identity (pinned by test) — a model
        # shown coordinates it does not speak will echo them anyway, and the
        # mismatch reads as "the model crops random background".
        h_px, w_px = img.rgb.shape[:2]
        _to_model = getattr(model, "to_model_box", None)
        fmt_box = ((lambda b: str(list(_to_model(b, w_px, h_px))))
                   if _to_model else
                   (lambda b: str(tuple(int(v) for v in b))))
        convention = getattr(model, "BOX_CONVENTION",
                             "[x0, y0, x1, y1] in full-frame pixels")
        # `answer_schema` is the caller saying what SHAPE of readout it wants
        # back ("yes|no", "the text, verbatim"). It is enforced by prompt text
        # and nothing else: `models.respond`'s schema argument is only a
        # truthiness flag that switches Gemini into JSON mode (models.py:111).
        # The field ORDER is never negotiable — a caller may constrain the
        # answer's contents, never the order it arrives in.
        prompt = (f"{self.rules}\n\nView: {img.text}\nTask: {task}"
                  + (f"\nRequired shape of `answer`: {answer_schema}"
                     if answer_schema else "")
                  + (f"\nCurrent hypothesis: {hypothesis}" if hypothesis else ""))

        transcript = {"cell": list(cell) if cell else None, "task": task,
                      "answer_schema": answer_schema,
                      # The uprighted array, not the raw capture — _save_seen.
                      "image": _save_seen(tools, img, img.cap_dir,
                                          kind="frames"),
                      "view_text": img.text,
                      "hypothesis": hypothesis, "prompt": prompt,
                      "t": time.time(), "turns": []}
        # The full frame stays first in every request: the crop is additional
        # context, never a replacement, and the most relevant image belongs at
        # the START of the sequence (position alone is worth up to a 41%
        # swing). Numbering counts SAVED images only, so it stays aligned by
        # construction with the enumeration `_images_of` does after the fact.
        n_images = 1 if transcript["image"] else 0
        frame_label = "image 0 — full frame" if transcript["image"] \
            else "full frame"
        parts = [prompt, (frame_label, img.rgb)]
        done = {}                   # tool call -> its result, to refuse repeats

        _tick(0, "start", cell=list(cell) if cell else None, task=task)
        for turn_i in range(max_turns):
            _tick(turn_i + 1, "thinking")
            reply = model.respond(parts, schema={"order": self.emit})
            transcript["turns"].append(dict(reply) if isinstance(reply, dict)
                                       else {"raw": str(reply)})
            if isinstance(reply, dict) and "tool" in reply:
                name = reply["tool"]
                if name not in self.tool_names:
                    # Pruned or unknown: same categorical refusal either way.
                    # A verb a spec dropped is STRUCTURALLY absent — it costs
                    # a message, never an exception, and never runs.
                    result = (f"{name} is not a tool this agent has — tools "
                              f"are {sorted(self.tool_names)}")
                    transcript["turns"].append({"result": result})
                    parts.append(f"[{name}] {result}")
                    continue
                args = _boxes_to_pixels(reply.get("args", {}) or {}, model, img)
                key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
                if key in done:
                    # Removing distractor actions is the single biggest lever
                    # on agent loops (AgentOccam WebArena 16.5 -> 25.8%), so
                    # say it plainly instead of re-running.
                    result = (f"{name} was already called with these arguments "
                              f"and returned: {done[key]} — the result will "
                              f"not change. Use a different tool or give your "
                              f"answer now.")
                    transcript["turns"].append({"result": result})
                    parts.append(f"[{name}] {result}")
                    continue
                try:
                    # Always the original frame — see _dispatch's docstring.
                    result, new_img = _dispatch(name, args, tools, verbs,
                                                writer, img, fmt_box)
                    done[key] = result
                except Exception as e:
                    # A bad box is not a crash, it is a message — stated in
                    # the MODEL's convention, never a different one (teaching
                    # the wrong convention here is what a failure message must
                    # not do).
                    result, new_img = (f"{name} failed: {e}. Boxes are "
                                       f"{convention}, about the full "
                                       f"frame."), None
                _tick(turn_i + 1, "tool", tool=name, args=args,
                      result=str(result)[:300])
                turn_rec, crop_label = {"result": result}, None
                if new_img is not None:
                    rel = _save_seen(
                        tools, new_img,
                        f"{img.cap_dir}_{len(transcript['turns']):02d}")
                    if rel is not None:
                        turn_rec["image"] = rel
                        # Announce the number IN the result text, so the
                        # model can cite the image later by echoing it.
                        result += f" — this is image {n_images}"
                        turn_rec["result"] = result
                        crop_label = (f"image {n_images} — "
                                      f"crop of the same frame")
                        n_images += 1
                        # Announce the crop the moment it exists on disk, so
                        # the live view streams the pictures, not just words.
                        _tick(turn_i + 1, "image", image=rel, tool=name)
                    else:
                        crop_label = "crop of the same frame"
                transcript["turns"].append(turn_rec)
                # ORDER IS LOAD-BEARING: the result text goes in BEFORE the
                # crop pixels, never after.
                parts.append(f"[{name}] {result}")
                if new_img is not None:
                    parts.append((crop_label, new_img.rgb))
                continue

            answer = (reply or {}).get("answer", "unknown")
            raw = reply if isinstance(reply, dict) else {}
            extras = {k: fn(raw.get(k)) for k, fn in self.extras.items()}
            _tick(turn_i + 1, "done", answer=answer, **extras)
            finding = Finding(cell, answer, raw.get("evidence", []),
                              raw.get("reasoning", ""), transcript,
                              extras=extras)
            transcript["answer"] = answer
            for k, v in extras.items():
                transcript[k] = v            # disk and Finding always agree
            finding.transcript_rel = writer.add_finding(
                cell, finding.summary, json.dumps(transcript, indent=1))
            return finding

        # Out of turns: a finding about the model, not an exception.
        _tick(max_turns, "exhausted")
        transcript["answer"] = "unknown"
        finding = Finding(cell, "unknown", [],
                          f"no answer within {max_turns} turns",
                          transcript, exhausted=True)
        finding.transcript_rel = writer.add_finding(
            cell, finding.summary, json.dumps(transcript, indent=1))
        return finding
