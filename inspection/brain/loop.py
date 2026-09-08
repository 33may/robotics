#!/usr/bin/env python3
"""The orchestrator loop — a blind planner over the eyes tier.

Runtime is the Claude Agent SDK (Anton 2026-08-25): we adopt the loop and build
only the harness — the tool bodies, the safety gate, the context render and the
evidence ledger. Nothing here re-implements turn-taking, retries or context
management; that is exactly the part we rent.

Shape (Anton's loop sketch, 2026-08-25):

    survey -> plan() -> { inspect(question, cell) | move(cell) } * -> answer()

Four verbs, deliberately few. Every additional verb is a distractor, and
pruning the action set is the biggest measured lever on agent loops
(AgentOccam `2410.13825`: WebArena 16.5 -> 25.8% from pruning alone).

**The planner never sees pixels.** `inspect` runs the vision subagent
(`eyes/inspect_agent.py`), which burns its own context on the frame and returns
text. That split is what stops a 5-30 step run from filling the planner with
images, and it makes the vision->language boundary the one interface the
layer-1 readout bench can measure.

**Replay first** (memory: orchestrator-objectives-and-replay-first). `move`
here jumps to a cell that a data-engine sweep already captured, so the whole
loop runs and benches with no arm, no IK and no risk. The live executor binds
at exactly one place — `_gate` — and the seam is marked.

    p inspection/brain/loop.py inspection/data/runs/2408-cup1 "is there a logo on the cup?"
"""
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, HookMatcher,
                              ResultMessage, StreamEvent, TextBlock,
                              ThinkingBlock, create_sdk_mcp_server, query, tool)

from inspection.brain import render
from inspection.brain.trace import TraceWriter
from inspection.eyes.agents.evidence_agent import hunt_evidence
from inspection.eyes.agents.inspect_agent import inspect_view
from inspection.eyes.agents.survey_agent import describe_survey
from inspection.eyes.store import FindingWriter, PlanWriter, RunStore
from inspection.eyes.tools import ViewTools
from inspection.record.ai_writer import AIRunWriter, next_seq
from inspection.record.run import Run
from inspection.record.schema import MenuInput
from inspection.view.grid import DEFAULT_R, H_BINS, V_ELEVATIONS

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
MAX_TURNS = 40                 # budget, not a plan: the loop must be able to
                               # stop without an answer. "cannot determine,
                               # coverage N" is a legitimate inspection result.

SYSTEM_PROMPT = """You inspect a physical object with a robot arm-mounted \
camera, to answer one question about it.

You cannot see. A vision subagent looks at frames for you and reports in text;
you own the geometry, the plan, and the answer.

Verbs:
- plan(criteria) — once, first. Nothing else may be called before this. The
  SURVEY VIEW is the vision tier's declaration of the object: what it is, its
  geometric primitive, how each surface is oriented in the survey frame, and
  which features are visible. Build the plan FROM that geometry:
  (1) the question broken into things that can be checked by looking;
  (2) where the asked-about feature most likely sits on that primitive — the
      declaration never guesses about unseen surfaces, that inference is
      YOURS, from what such objects are like;
  (3) the declared orientations turned into cells: the orientation words map
      to degrees (table below), counted from the survey bearing line, and
      image-right in the survey frame is increasing azimuth;
  (4) the first moves, likeliest surface first — never a blind orbit.
- move(cell) — go to a cell, capture it, and inspect what is there against the
  question. One deterministic step; you get the new view's report back.
- inspect(question, cell, answer_schema) — run the inspection model again on a
  view you already hold, asking something of your own choosing. Omit `cell` for
  where you are now. This costs NO robot motion, so chase a detail through the
  views you already have before spending a move. Give `answer_schema` when you
  need the readout in a particular shape — "yes|no", "the text, verbatim", "a
  count" — so the result can be checked against your criteria rather than read
  as prose.
- answer(reasoning, verdict, evidence, evidence_cells) — ends the run.
  evidence_cells names where the PROOF is: up to three {cell, find} entries,
  each a view you hold and EXACTLY what a picture from it must show, stated
  so it can be verified in pixels ("the ILFORD wordmark", "the barcode on
  the top flap"). For a YES cite the feature itself; for a NO cite the
  surfaces that would have shown it, so the record proves you looked. An
  evidence hunt that finds nothing bounces the answer back to you — move
  somewhere better, cite a different view, or re-answer citing the same
  hunt to ship with the gap on record. Once "run complete" returns, the run
  is over: say nothing further and call nothing further.

YOU START WITH NO IMAGES. The survey frame is not on the viewsphere and
cannot be inspected, so at the beginning there is nothing for inspect() to
read — your first action after plan() must be a move(). inspect() only works
on a viewpoint you have already moved to and captured.

Addresses are [azimuth, elevation]. The MOVES block lists every azimuth at
your current elevation level, plus every viewpoint already visited at the
other levels; the bearings beside them are relative to where you stand, and
`right` means increasing azimuth. One azimuth step is 30°. To change
elevation, take an address and change its second number — the STATE block
names the levels and their heights. Prefer moving by a listed address.

WHICH WAY TO GO. Each visited row carries what the vision subagent saw from
there, and sometimes a line `recommends: <phrase>`. The phrase says how far
to orbit and which way, judged in that view's frame — and the camera is
mounted so the frame's left/right and this menu's left/right are the same
words. The words map to degrees:

  slightly = 30°    half = 60°    far = 90°    around = 150°

Count from the cell the recommendation hangs under, NOT from where you
stand: "half right" under move([3, 0]) means the address 60° — two azimuth
steps — to the right of [3, 0]. Resolve the phrase to an address and move
there directly instead of stepping one cell at a time.
- Ask when you do not know. inspect() costs no motion, so a turn spent on
  "which way would frame this better?" is far cheaper than a move spent
  guessing.
- A viewpoint that showed the target edge-on is not worth returning to; one
  that showed it clipped at an edge usually is.

How to answer:
- One view that clearly shows the feature settles a YES. You do not need
  corroboration, and you must not count votes across views.
- A NO is a claim about coverage: it needs the argument that you looked where
  the feature would have been visible had it existed.
- SETTLING THE VERDICT IS NOT THE SAME AS HAVING THE BEST EVIDENCE. Once you
  believe the answer, keep going until the evidence is the best the workspace
  can give you: the feature fully in frame rather than clipped at an edge,
  and any text on it read verbatim. If the current viewpoint cannot give you
  that, move to one that can. Answer from your best view, not your first.
- Never state a confidence number. Reason first, verdict after — in that order.
- Do not answer before you have looked."""


class Brain:
    """One inspection run. Holds the state the render turns into text."""

    def __init__(self, run_dir, question, vlm, verbs,
                 model=DEFAULT_MODEL, max_turns=MAX_TURNS, adaptive_thinking=True,
                 mover=None, trace=None, ai=None):
        self.run_dir = Path(run_dir)
        self.question = question
        self.run = Run.load(self.run_dir)
        # ONE orchestrator session = one `ai/<seq>/` subtree, and this writer
        # owns all of it (trace, store, transcripts, verdict). A caller that
        # already opened one — the live handler, which surveys before the
        # Brain exists — passes it in; the CLI opens the next free seq here.
        self.ai = ai if ai is not None else AIRunWriter.create(
            self.run_dir, seq=next_seq(self.run_dir), orchestrator_model=model,
            menu_id=render.MENU_ID, menu_hash=render.menu_def().content_hash)
        if ai is None:
            self.ai.menu_def(render.menu_def())
        # RunStore SURVIVES here for its OTHER role — the brain's mutable AI
        # state (plan/hypothesis/findings, task-5 ruling); views live on
        # `self.run` now. `open` preserves that state across a resumed
        # session; the grid values below are vestigial once ViewTools reads
        # geometry off `self.run` directly (eyes/tools.py:_grid_of) — they
        # only need to satisfy `RunStore.create`'s signature.
        store_path = self.ai.dir / "store.json"
        self.store = RunStore.open(self.ai.dir) if store_path.exists() else \
            RunStore.create(self.ai.dir, h_bins=H_BINS, v_elevs=V_ELEVATIONS,
                            r=DEFAULT_R)
        self.plan_writer = PlanWriter(self.store)
        self.tools = ViewTools(self.run, writer=self.plan_writer)
        # Bound to the AI session: every subagent transcript is written
        # through it as a `TranscriptRecord`, so `Run.load(...).ai[].transcripts`
        # can see what this run's eyes actually did.
        self.finding_writer = FindingWriter(self.store, ai=self.ai)
        self.vlm = vlm
        # Both models are given, never defaulted. Until 2026-08-26 `verbs`
        # fell back to `LocalVerbs(StubBackend())`, so every live run so far
        # detected the same fixed box (10, 10, 100, 100) for every phrase and
        # the trace looked exactly like a real one. Cognition is assembled by
        # the caller (`eyes/cognition.py`) and labelled in the `run` event.
        self.verbs = verbs
        self.model = model
        self.max_turns = max_turns
        self.adaptive_thinking = adaptive_thinking
        # None -> replay: `move` may only go where a sweep already captured.
        # A SupervisorMover -> live: `move` drives the real arm, gated on the
        # operator's approval (see brain/live.py).
        self.mover = mover

        # Agent progress is NOT store coverage. In replay the sweep already
        # captured everything, so `store.visited()` is full from turn one; what
        # matters is which cells this agent has actually read.
        self.agent_seen = set()
        # cell -> the vision tier's `view` block for it. The run's view
        # ledger: what each viewpoint offered and where it recommends going.
        # Re-rendered into the move menu every turn, never pasted as a
        # standing table (brain/render.py: stamped blocks, not snapshots).
        self.view_notes = {}
        self.cell = None                     # survey pose, off-grid
        self.planned = False
        self.answer = None
        self.turn = 0                        # stamps the state/moves blocks
        # (cell, find) -> resolved evidence record. The hunt cache: mirrors
        # the subagent's own repeated-call refusal, and it is what turns
        # "re-cite the same failed hunt" into an explicit acknowledgment
        # instead of an infinite bounce (each hunt costs tens of seconds).
        self._hunts = {}
        # A caller that already opened the trace (live runs, where the survey
        # happens before the Brain exists) passes it in and has already
        # written the `run` event — appending a second one would give the
        # panel two headers for one run.
        self.trace = trace or TraceWriter(self.ai.dir)
        if trace is None:
            self.trace.event("run", question=question, model=model,
                             run_dir=str(self.run_dir),
                             live=mover is not None,
                             captured=len(self._visited()))

    # ------------------------------------------------------------- state
    def _nav(self):
        """The Supervisor's reachability picture, or None in replay."""
        return self.mover.nav() if self.mover is not None else None

    def _visited(self):
        """Distinct grid cells captured so far — the survey (step 0)
        excluded, since it sits off-grid and was never a cell to visit."""
        return {tuple(s.record.view.address) for s in self.run.captured
                if s.id != 0}

    def _counts(self):
        """(visited, reachable) over the sphere — see render.state_block."""
        nav = self._nav()
        if nav is None:
            # Replay: the sweep IS the reachable world.
            return len(self.agent_seen), len(self._visited())
        return len(nav["visited"]), len(nav["reachable"])

    @staticmethod
    def _finding_text(f):
        """The full finding crosses back, not the one-line summary.

        Anton 2026-08-25 overrode "distilled in context" for v1: Opus carries
        the tokens, and OpenEQA is the one summarise-vs-raw datapoint we have
        (scene-graph captions 36.5 < per-frame captions 43.6 < raw 49.6). The
        transcript still stays on disk — evidence order is kept as the vision
        tier produced it, evidence before reasoning before answer.

        The `view` block does NOT ride along: it re-renders in the move menu
        on every turn (the state hook), and a copy here would only be the
        stale twin of that row.
        """
        ev = "\n".join(f"  - {e}" for e in f.evidence) or "  (none given)"
        return (f"evidence:\n{ev}\nreasoning: {f.reasoning}\n"
                f"answer: {f.answer}")

    def _inspect_cell(self, cell, question, answer_schema=None):
        """Run the vision subagent on one captured view and record it.

        `move` and `inspect` share this: the difference between them is motion
        and whose question gets asked, never how the looking happens. One path
        in means every description in the run is a FindingWriter entry with a
        transcript on disk — there is no second, unaudited way to see.
        """
        def on_turn(n, total, event, **fields):
            # One trace line per subagent step. The watcher republishes on
            # mtime, so the panel shows the loop as it happens rather than a
            # 40-second gap ending in a wall of text.
            self.trace.event("sub_step", cell=list(cell), n=n, of=total,
                             event=event, **fields)

        finding = inspect_view(self.tools, self.verbs, self.finding_writer,
                               self.vlm, cell, question,
                               hypothesis=self.store.hypothesis,
                               answer_schema=answer_schema, on_turn=on_turn)
        self.agent_seen.add(cell)
        # Last read wins. A later, sharper look at the same cell supersedes an
        # earlier one — the ledger is a current picture of what each viewpoint
        # offers, not a history of every time we looked.
        if finding.view:
            self.view_notes[cell] = finding.view
        return finding

    def _survey_text(self, surveys, gloss):
        """What `SURVEY VIEW ·` carries: the declaration, anchored to the grid.

        The vision tier declares the object — identity, primitive, surface
        orientations, visible features (`eyes/survey_agent.py`); the harness
        appends the code-computed bearing that anchors the declaration's
        frame-relative words to addresses. Until 2026-08-27 this was only the
        metadata gloss, and plan() was written knowing nothing about the
        object — 2608-aicam's plan asked "what kind of product is it?".
        Falls back to the gloss: a failed declaration must not stop the run.
        """
        if not surveys:
            return gloss
        def on_turn(n, total, event, **fields):
            self.trace.event("sub_step", cell=None, n=n, of=total,
                             event=event, **fields)
        f = describe_survey(self.tools, self.verbs, self.finding_writer,
                            self.vlm, on_turn=on_turn)
        text = gloss if f is None or f.answer in (None, "", "unknown") \
            else str(f.answer)
        # Exact centre from the live Supervisor when there is one; recovered
        # from the capture rays in replay (`ViewTools.survey_bearing`).
        sphere = getattr(getattr(self.mover, "sup", None), "sphere", None)
        bearing = self.tools.survey_bearing(
            surveys[0], centre=getattr(sphere, "center", None))
        if bearing is not None:
            text += "\n" + render.survey_bearing_line(*bearing,
                                                      self.tools._h_bins)
        return text

    @staticmethod
    def _images_of(finding):
        """Run-relative paths of every image the subagent actually looked at:
        the full frame first, then each crop it chose, in the order it chose
        them. The crop sequence IS the visual reasoning."""
        t = finding.transcript
        out = [t["image"]] if t.get("image") else []
        out += [turn["image"] for turn in t.get("turns", [])
                if turn.get("image")]
        return out

    def _collect_evidence(self, entries):
        """Run the cited evidence hunts. Returns (bounce_text | None, records).

        The planner names WHERE the proof is and WHAT a picture must show;
        the hunt (`eyes/evidence_agent.py`) does the pixels; this resolves
        cells and paths deterministically. A FRESH miss bounces the whole
        answer — the run is not over, the planner can move somewhere better —
        while re-citing an already-missed hunt is the explicit "ship with the
        gap on record". All hunts run before any bounce, so a re-answer pays
        only for what it newly cites.
        """
        records, fresh_misses = [], []
        for e in entries or []:
            if not isinstance(e, dict) or "cell" not in e:
                self.trace.event("text", text=f"dropped evidence entry {e!r}")
                continue
            try:
                cell = tuple(int(c) for c in e["cell"])
            except (TypeError, ValueError):
                self.trace.event("text", text=f"dropped evidence entry {e!r}")
                continue
            find = str(e.get("find") or "").strip() or self.question
            key = (cell, find.casefold())
            if key in self._hunts:
                records.append(self._hunts[key])
                continue
            if self.tools.view_at(cell) is None:
                return (f"cannot hunt {list(cell)}: no capture at that cell. "
                        f"Cite views you hold.", records)

            def on_turn(n, total, event, **fields):
                self.trace.event("sub_step", cell=list(cell), n=n, of=total,
                                 event=event, hunt=find, **fields)

            rec = hunt_evidence(self.tools, self.verbs, self.finding_writer,
                                self.vlm, cell, find, on_turn=on_turn)
            self._hunts[key] = rec
            self.trace.event("evidence", **rec)
            records.append(rec)
            if not rec["found"]:
                fresh_misses.append(rec)
        if fresh_misses:
            gaps = "; ".join(f"hunt at {r['cell']} found no {r['find']!r} — "
                             f"it reported: {r['report']}"
                             for r in fresh_misses)
            return (f"{gaps}. The run is not over: move to a view that shows "
                    f"it, cite a different view, or answer again citing the "
                    f"same hunt to ship with the gap on record.", records)
        return None, records

    def _ai_relative(self, path):
        """A run-relative artifact path -> relative to this AI session.

        The eyes tier writes its frames and crops into `ai/<seq>/artifacts/`
        and reports them run-relative (what the trace and the UI mount read);
        `EvidenceImage.artifact` is declared relative to the AIRun. One
        conversion, at the boundary that knows both. A path from outside this
        session cannot be cited as its evidence.
        """
        if not path:
            return None
        try:
            return Path(path).relative_to(
                self.ai.dir.relative_to(self.run_dir)).as_posix()
        except ValueError:
            return None

    def _answer_record(self, args, records):
        """The verdict as `AnswerRecord` — the write boundary of the AI tier.

        The loop's runtime answer and the schema's are NOT the same shape, and
        this is where they meet:

        - `evidence` is one prose blob from the model; the record is a LIST of
          statements, so a single blob is a one-element list.
        - an `evidence_images` entry is a raw hunt record
          ({cell, find, found, report, crop, frame, transcript}). The schema
          wants a citation: WHICH step, WHICH transcript, WHICH image. `cell`
          resolves through `Run.at` to the step that holds that view, `crop`
          (or the frame, when the hunt cited none) is the artifact, and the
          hunt's own verdict text becomes the note.
        - a hunt that produced no image at all cannot be cited as an image.
          It is dropped HERE and only here: the full record is already in the
          trace (`evidence` event) and in the transcript on disk, so nothing
          is lost — the answer simply does not claim a picture it does not
          have.
        - `artifact` is stored RELATIVE TO THE AI SESSION (schema.py:
          EvidenceImage), while the hunt reports run-relative paths — that is
          what the trace and the UI's capture mount speak. The conversion is
          here, at the one boundary that knows both.
        - the same hunt cited twice (the planner re-citing a cached miss to
          ship with the gap on record) is ONE citation: `_hunts` returns the
          identical record, and a verdict does not gain evidence by repeating
          itself.
        """
        self.run.refresh()
        images, step_ids, transcript_ids, seen = [], [], [], set()
        for r in records:
            artifact = self._ai_relative(r.get("crop") or r.get("frame"))
            step = self.run.at(r["cell"]) if r.get("cell") is not None else None
            if artifact is None or step is None:
                self.trace.event("text", text=(
                    f"evidence at {r.get('cell')} not citable as an image "
                    f"({'no artifact' if artifact is None else 'no step'}) — "
                    f"it stays in the trace"))
                continue
            # The transcript's id as the WRITER stamped it, so the citation
            # names a file `Run.ai[].transcripts` actually returns.
            tid = str(r.get("transcript_id")
                      or Path(str(r.get("transcript") or "")).stem or "unknown")
            if (step.id, artifact) in seen:
                continue
            seen.add((step.id, artifact))
            images.append({
                "step_id": step.id, "transcript_id": tid, "artifact": artifact,
                "note": f"{r['find']} — {'found' if r['found'] else 'NOT found'}"
                        f": {r['report']}"})
            step_ids.append(step.id)
            transcript_ids.append(tid)
        evidence = args.get("evidence")
        return {
            "verdict": args["verdict"], "reasoning": args["reasoning"],
            "evidence": [evidence] if isinstance(evidence, str) and evidence
            else list(evidence or []),
            "evidence_images": images,
            "step_ids": sorted(set(step_ids)),
            "transcript_ids": sorted(set(transcript_ids)),
            "views_inspected": sorted(map(list, self.agent_seen)),
            "coverage": self.tools.coverage(cur=self.cell),
        }

    # ------------------------------------------------------------- verbs
    def _build_tools(self):
        def ok(text):
            return {"content": [{"type": "text", "text": text}]}

        def done(tool_name, text, **extra):
            """Every verb exits through here, so the trace cannot miss one."""
            self.trace.event("tool_result", tool=tool_name, text=text, **extra)
            return ok(text)

        def called(tool_name, args):
            self.trace.event("tool_call", tool=tool_name, args=dict(args))

        @tool("plan", "Record the question broken into checkable criteria. "
                      "Call this once, before anything else.",
              {"criteria": str})
        async def _plan(args):
            called("plan", args)
            self.plan_writer.set_plan(args["criteria"])
            self.planned = True
            self.trace.event("write", tier="plan", what="plan",
                             text=args["criteria"])
            return done("plan", "plan recorded")

        @tool("inspect", "Ask the inspection model about a captured view. Omit "
                         "cell to use the current one. Costs no robot motion. "
                         "answer_schema optionally constrains the readout "
                         "shape, e.g. 'yes|no' or 'the text, verbatim'.",
              {"question": str, "cell": list, "answer_schema": str})
        async def _inspect(args):
            called("inspect", args)
            if not self.planned:
                return done("inspect", "call plan() first — the criteria are "
                                       "what the findings get checked against.")
            cell = tuple(args["cell"]) if args.get("cell") else self.cell
            if cell is None:
                return done("inspect", "no cell given and the arm is still at "
                                       "the survey pose, which is off-grid. "
                                       "move() to a cell first.")
            if self.tools.view_at(cell) is None:
                return done("inspect", f"no capture exists at {list(cell)} — "
                                       f"nothing to inspect. Move somewhere "
                                       f"captured, or pick another.")
            f = self._inspect_cell(cell, args["question"],
                                   args.get("answer_schema"))
            return done("inspect", f"view {list(cell)} —\n{self._finding_text(f)}",
                        cell=list(cell), images=self._images_of(f),
                        sub=f.transcript)

        @tool("move", "Go to a cell, capture it, and inspect it against the "
                      "run's question. One deterministic step.", {"cell": list})
        async def _move(args):
            called("move", args)
            if not self.planned:
                return done("move", "call plan() first.")
            cell = tuple(args["cell"])

            if self.mover is not None:
                # LIVE. Blocks in a worker thread until the operator approves,
                # redirects, or the request is refused — `query()` is async, so
                # blocking the event loop here would stall the whole SDK
                # session including its hooks.
                ok, reason = await asyncio.to_thread(self.mover.request, cell)
                if not ok:
                    return done("move", reason, cell=list(cell), failed=True)
                self.run.refresh()
                self.cell = cell
                f = self._inspect_cell(cell, self.question)
                return done("move",
                            f"view {list(cell)} —\n{self._finding_text(f)}",
                            cell=list(cell), images=self._images_of(f),
                            sub=f.transcript)

            if self.tools.view_at(cell) is None:
                # Categorical, not verbatim — LLM3 (`2403.11552`) measured 60%
                # vs 40% success AND fewer retries from naming the failure
                # class rather than echoing the error.
                return done("move", f"cannot reach {list(cell)}: no capture at "
                                    f"that cell in this run. The arm has not "
                                    f"moved.", cell=list(cell), failed=True)
            # Deterministic robot code, then the inspect call itself — nothing
            # else. `move` returns exactly what `inspect` returns, because it
            # IS `inspect` with a motion in front of it. Position and menu are
            # state, so they arrive via the hook like all other state; putting
            # them here would make one verb quietly render its own context.
            self.cell = cell
            f = self._inspect_cell(cell, self.question)
            return done("move", f"view {list(cell)} —\n{self._finding_text(f)}",
                        cell=list(cell), images=self._images_of(f),
                        sub=f.transcript)

        @tool("answer", "End the run with a verdict. Reasoning first. "
                        "evidence_cells names where the proof is.",
              {"reasoning": str, "verdict": str, "evidence": str,
               "evidence_cells": list})
        async def _answer(args):
            called("answer", args)
            # The hunts run INSIDE the answer: the planner cites where the
            # proof is and what a picture must show, the evidence agent does
            # the pixels, and a fresh miss bounces the whole answer back as a
            # failed tool call (Anton 2026-08-27) — code-level enforcement of
            # "answer from your best view, not your first".
            bounce, records = await asyncio.to_thread(
                self._collect_evidence, args.get("evidence_cells"))
            if bounce:
                return done("answer", bounce, failed=True)
            self.answer = {"reasoning": args["reasoning"],
                           "verdict": args["verdict"],
                           "evidence": args["evidence"],
                           "evidence_images": records,
                           "views_inspected": sorted(map(list, self.agent_seen)),
                           "coverage": self.tools.coverage(cur=self.cell)}
            self.ai.answer(self._answer_record(args, records))
            # The trace keeps the RAW hunt records (cell/find/found/report/
            # crop/frame/transcript) — `_answer_record` below narrows them to
            # what the schema has a home for, and nothing may be lost between
            # the two.
            self.trace.event("answer", **{k: v for k, v in self.answer.items()
                                          if k != "coverage"})
            self.trace.event("write", tier="plan", what="answer",
                             text=args["verdict"])
            return done("answer", "run complete",
                        images=[r["crop"] for r in records if r.get("crop")])

        return [_plan, _inspect, _move, _answer]

    def _menu_input(self):
        """One `MenuInput` line per turn — the AI-tier half of the menu.

        The menu is the pure function (ViewState, MenuInput) -> text. ViewState
        is physical truth and belongs to the step; THIS is what the agent-side
        rendering additionally consumed — which cells this agent has actually
        read, how many findings stand, and the recommendations hanging off the
        view ledger. With both on disk a menu experiment can re-render a
        recorded turn instead of re-flying it. The rendered text itself is
        deliberately not repeated here: it is already in the trace.

        Best effort: losing a menu line must not cost the turn.
        """
        try:
            step = (self.run.at(self.cell) if self.cell is not None
                    else self.run.survey)
            self.ai.menu_input(MenuInput(
                turn=self.turn, step_id=step.id if step is not None else 0,
                t=time.time(),
                agent_context={
                    "seen": sorted(map(list, self.agent_seen)),
                    "findings": len(self.store.findings()),
                    "recommends": {str(list(c)): v["recommendation"]
                                   for c, v in self.view_notes.items()
                                   if (v or {}).get("recommendation")},
                    "nav": {k: sorted(map(list, v))
                            for k, v in (self._nav() or {}).items()},
                }))
        except Exception:                            # noqa: BLE001
            log.exception("could not record the menu input for turn %s",
                          self.turn)

    # ------------------------------------------------------------- hooks
    async def _staple_state(self, input_data, tool_use_id, context):
        """Push state onto every tool result — the model cannot forget to look.

        A pull costs a turn and can be skipped; a push cannot. EVERY turn,
        not only on moves: until 2026-08-26 the menu rode only the move
        results, and on `2608-aicam` four consecutive inspects on held views
        each received nothing but a findings counter — the planner chained
        decisions with no menu in sight. Re-stating is safe because every
        block is stamped `· turn N ·` (render.py module docstring).
        """
        if self.answer is not None:
            return {}       # the run is over; there is nothing left to decide
        self.turn += 1
        visited, reachable = self._counts()
        text = (render.state_block(self.turn, self.cell, self.tools._h_bins,
                                   self.tools._v_elevs, visited, reachable,
                                   len(self.store.findings()))
                + "\n\n"
                + render.moves_block(self.turn, self.tools, self.cell,
                                     self.agent_seen, nav=self._nav(),
                                     seen=self.view_notes))
        # Traced as its own kind: this is context the model was GIVEN, not text
        # it wrote. Telling those apart is the whole point of watching a run.
        self.trace.event("state", text=text)
        self._menu_input()
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                       "additionalContext": text}}

    async def _gate(self, input_data, tool_use_id, context):
        """The safety gate. THIS is where the live executor binds.

        Prompt-level safety fails in 49-73% of vulnerable tasks, so the veto
        lives in code between the tool call and the actuator, not in the
        system prompt. In replay there is no actuator: the only invariant that
        exists is the grid itself. When `move` drives a real arm, this is
        where the motion tier's checks run (`motion/execute.py` refuses on
        world-invalidated paths and non-NORMAL safety mode), and a
        refusal returns here as a denial the model cannot talk its way past.
        """
        cell = (input_data.get("tool_input") or {}).get("cell")
        h_bins, n_v = self.tools._h_bins, len(self.tools._v_elevs)
        bad = (not isinstance(cell, (list, tuple)) or len(cell) != 2
               or not (0 <= int(cell[0]) < h_bins) or not (0 <= int(cell[1]) < n_v))
        if bad:
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"{cell} is not a cell on this viewsphere — h is 0..{h_bins - 1}, "
                    f"v is 0..{n_v - 1}.")}}
        return {}

    # -------------------------------------------------------------- run
    async def run(self):
        server = create_sdk_mcp_server(name="brain", version="0.1.0",
                                       tools=self._build_tools())
        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"brain": server},
            allowed_tools=[f"mcp__brain__{n}"
                           for n in ("plan", "inspect", "move", "answer")],
            tools=[],                      # no Read/Bash/Edit: this agent has
                                           # exactly four things it can do
            strict_mcp_config=True,
            setting_sources=[],            # ignore this repo's .claude config
            permission_mode="dontAsk",     # never block a headless run
            max_turns=self.max_turns,
            # The assembled ThinkingBlock carries only a signature — the actual
            # reasoning text is delivered ONLY as streaming deltas. Without
            # this the trace records that the model thought, but not what it
            # thought, which is the single most useful thing to watch.
            include_partial_messages=True,
            # Thinking models are the one documented immunity to
            # self-conditioning — the mechanism is that they do not refer back
            # to their own prior answers, which is exactly the failure that
            # gets WORSE with model scale. One line to flip if the bench
            # disagrees.
            thinking={"type": "adaptive"} if self.adaptive_thinking
            else {"type": "disabled"},
            hooks={
                "PreToolUse": [HookMatcher(matcher="mcp__brain__move",
                                           hooks=[self._gate])],
                "PostToolUse": [HookMatcher(hooks=[self._staple_state])],
            },
        )
        # NOT `view_at(None)`: `tools._views()` means "every view", so that
        # would hand back the newest capture of the whole run. The survey is
        # the record whose cell IS None (record/run.py: step id 0).
        records = self.tools._views()
        surveys = [v for v in records if v.cell is None]
        survey = self.tools.get_view(surveys[0] if surveys else records[0])
        prompt = render.opening(self.question, self._survey_text(surveys,
                                                                 survey.text),
                                self.tools, self.agent_seen, nav=self._nav())

        result = None
        thinking, from_stream = [], False
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, StreamEvent):
                # Measured 2026-08-25: Opus 5's thinking content is NOT exposed
                # through the SDK. `thinking_delta` carries an empty `thinking`
                # string and an `estimated_tokens` count, and the assembled
                # ThinkingBlock holds only a signature. So the trace records
                # THAT it thought and how much — never invent the content.
                # If visible reasoning is wanted, it has to be narrated as
                # text (a ReAct Thought field), not extracted from here.
                ev = msg.event or {}
                if ev.get("type") == "content_block_delta":
                    d = ev.get("delta") or {}
                    if d.get("type") == "thinking_delta":
                        thinking.append(d.get("thinking", "")
                                        or d.get("estimated_tokens", 0))
                elif ev.get("type") == "content_block_stop" and thinking:
                    text = "".join(t for t in thinking if isinstance(t, str))
                    tokens = max([t for t in thinking
                                  if isinstance(t, int)] or [0])
                    self.trace.event("thinking", text=text, tokens=tokens)
                    thinking.clear()
                    from_stream = True
            elif isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, ThinkingBlock):
                        # Fallback only: if partial events ever stop carrying
                        # the text, still record whatever the block holds.
                        if b.thinking and not from_stream:
                            self.trace.event("thinking", text=b.thinking)
                    elif isinstance(b, TextBlock) and b.text.strip():
                        self.trace.event("text", text=b.text.strip())
                        print(f"[brain] {b.text.strip()[:300]}")
            elif isinstance(msg, ResultMessage):
                result = msg
            # Deliberately NO `break` on self.answer: closing query()'s async
            # generator while it is mid-iteration raises "aclose():
            # asynchronous generator is already running" and buries the run's
            # real output under a traceback. answer() returns a terminal tool
            # result and the model stops on its own; max_turns is the backstop.
        if self.answer is None:
            # Budget exhausted without a verdict. That is a result, not a
            # crash, and the bench needs to see it as one.
            self.answer = {"verdict": "cannot determine",
                           "reasoning": f"no answer within {self.max_turns} turns",
                           "evidence": "", "views_inspected": sorted(
                               map(list, self.agent_seen)),
                           "coverage": self.tools.coverage(cur=self.cell)}
        return self.answer, result


def main(run_dir, question, model=DEFAULT_MODEL):
    from inspection.eyes.cognition import real_cognition

    # No stub fallback (Anton 2026-08-26). Replay is cheap in motion, not in
    # vision: a missing GEMINI_API_KEY used to silently swap the whole eyes
    # tier for scripted text over a fixed box, and the run still printed a
    # verdict. `real_cognition()` raises here instead. Tests and the mock
    # reach the stubs by injecting `stub_cognition()`, never by omission.
    cog = real_cognition()
    brain = Brain(run_dir, question, vlm=cog.vlm, verbs=cog.verbs, model=model)
    answer, result = asyncio.run(brain.run())
    print("\n=== ANSWER ===")
    print(json.dumps({k: v for k, v in answer.items() if k != "coverage"},
                     indent=1))
    print(answer["coverage"])
    if result is not None:
        print(f"turns {result.num_turns} · {result.duration_ms} ms "
              f"· ${result.total_cost_usd}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else "is there a logo on the cup?")
