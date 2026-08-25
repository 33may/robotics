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
import sys
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, HookMatcher,
                              ResultMessage, StreamEvent, TextBlock,
                              ThinkingBlock, create_sdk_mcp_server, query, tool)

from inspection.brain import render
from inspection.brain.live import refresh_views
from inspection.brain.trace import TraceWriter
from inspection.eyes.inspect_agent import inspect_view
from inspection.eyes.replay import load_run
from inspection.eyes.store import FindingWriter, PlanWriter
from inspection.eyes.tools import ViewTools
from inspection.eyes.verbs_local import LocalVerbs, StubBackend

DEFAULT_MODEL = "claude-opus-5"
MAX_TURNS = 40                 # budget, not a plan: the loop must be able to
                               # stop without an answer. "cannot determine,
                               # coverage N" is a legitimate inspection result.

SYSTEM_PROMPT = """You inspect a physical object with a robot arm-mounted \
camera, to answer one question about it.

You cannot see. A vision subagent looks at frames for you and reports in text;
you own the geometry, the plan, and the answer.

Verbs:
- plan(criteria) — once, first: the question broken into things that can be
  checked by looking. Nothing else may be called before this.
- move(cell) — go to a cell, capture it, and inspect what is there against the
  question. One deterministic step; you get the new view's report back.
- inspect(question, cell, answer_schema) — run the inspection model again on a
  view you already hold, asking something of your own choosing. Omit `cell` for
  where you are now. This costs NO robot motion, so chase a detail through the
  views you already have before spending a move. Give `answer_schema` when you
  need the readout in a particular shape — "yes|no", "the text, verbatim", "a
  count" — so the result can be checked against your criteria rather than read
  as prose.
- answer(reasoning, verdict, evidence) — ends the run. Once it returns, the
  run is over: say nothing further and call nothing further.

YOU START WITH NO IMAGES. The survey frame is not on the viewsphere and
cannot be inspected, so at the beginning there is nothing for inspect() to
read — your first action after plan() must be a move(). inspect() only works
on a viewpoint you have already moved to and captured.

Addresses are absolute; the bearings beside them are relative to where you
are standing. `right` means increasing azimuth. Move by address, never by
computing one yourself — the menu tells you which viewpoints are reachable
from where you are, and cells that are not listed cannot be reached.

WHICH WAY TO GO. Every view report ends with a `framing` line: where the
target sat in that frame, how square-on its surface was, and which way the
camera would have to shift to frame it better. Those are FRAME directions,
and the camera is mounted so that the right of the image is the same right
this menu uses. `better: left` means take a move the menu calls "left" — no
conversion, no arithmetic.
- To bring a surface square-on, orbit TOWARD the side it is turning toward.
  A mark "turning away to the left" is fixed by moving left and destroyed by
  moving right; one step the wrong way can rotate it out of sight entirely.
- Ask when you do not know. inspect() costs no motion, so a turn spent on
  "which way is this surface turning, and would shifting left or right frame
  it more squarely?" is far cheaper than a move spent guessing.
- The menu repeats what each visited cell showed, under "seen from there".
  Use it: a viewpoint that showed the target edge-on is not worth returning
  to, one that showed it clipped at an edge usually is, and the direction
  that improved things once will usually keep improving them.

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

    def __init__(self, run_dir, question, vlm, verbs=None,
                 model=DEFAULT_MODEL, max_turns=MAX_TURNS, adaptive_thinking=True,
                 mover=None, trace=None):
        self.run_dir = Path(run_dir)
        self.question = question
        self.store = load_run(self.run_dir)
        self.plan_writer = PlanWriter(self.store)
        self.tools = ViewTools(self.store, writer=self.plan_writer)
        self.finding_writer = FindingWriter(self.store)
        self.vlm = vlm
        self.verbs = verbs or LocalVerbs(StubBackend())
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
        # cell -> the vision tier's framing block for it. The run's view
        # ledger: what the target looked like from each place we have stood.
        # Re-rendered into the action menu on every move, never pasted as a
        # standing table (brain/render.py: deltas, not snapshots).
        self.view_notes = {}
        self.cell = None                     # survey pose, off-grid
        self.planned = False
        self.answer = None
        self._prev = self._snapshot()
        # A caller that already opened the trace (live runs, where the survey
        # happens before the Brain exists) passes it in and has already
        # written the `run` event — appending a second one would give the
        # panel two headers for one run.
        self.trace = trace or TraceWriter(self.run_dir)
        if trace is None:
            self.trace.event("run", question=question, model=model,
                             run_dir=str(self.run_dir),
                             live=mover is not None,
                             captured=len(self.store.visited()))

    # ------------------------------------------------------------- state
    def _nav(self):
        """The Supervisor's reachability picture, or None in replay."""
        return self.mover.nav() if self.mover is not None else None

    def _snapshot(self):
        nav = self._nav()
        if nav is None:
            # Replay: the sweep IS the reachable world.
            reachable, visited = len(self.store.visited()), len(self.agent_seen)
        else:
            reachable, visited = len(nav["reachable"]), len(nav["visited"])
        return render.snapshot(self.cell, self.agent_seen,
                               len(self.store.findings()),
                               reachable, visited)

    def _delta(self):
        cur = self._snapshot()
        out = render.state_delta(self._prev, cur, self.tools._h_bins,
                                 self.tools._v_elevs)
        self._prev = cur
        return out

    @staticmethod
    def _finding_text(f):
        """The full finding crosses back, not the one-line summary.

        Anton 2026-08-25 overrode "distilled in context" for v1: Opus carries
        the tokens, and OpenEQA is the one summarise-vs-raw datapoint we have
        (scene-graph captions 36.5 < per-frame captions 43.6 < raw 49.6). The
        transcript still stays on disk — evidence order is kept as the vision
        tier produced it, evidence before reasoning before answer, framing
        after (it is an appendix about the VIEW, not part of the chain).
        """
        ev = "\n".join(f"  - {e}" for e in f.evidence) or "  (none given)"
        out = (f"evidence:\n{ev}\nreasoning: {f.reasoning}\n"
               f"answer: {f.answer}")
        line = render.framing_line(getattr(f, "framing", None))
        return f"{out}\n{line}" if line else out

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
        if finding.framing:
            self.view_notes[cell] = finding.framing
        return finding

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
                refresh_views(self.store, self.run_dir)
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

        @tool("answer", "End the run with a verdict. Reasoning first.",
              {"reasoning": str, "verdict": str, "evidence": str})
        async def _answer(args):
            called("answer", args)
            self.answer = {"reasoning": args["reasoning"],
                           "verdict": args["verdict"],
                           "evidence": args["evidence"],
                           "views_inspected": sorted(map(list, self.agent_seen)),
                           "coverage": self.tools.coverage(cur=self.cell)}
            (self.store.path / "answer.json").write_text(
                json.dumps(self.answer, indent=1) + "\n")
            self.trace.event("answer", **{k: v for k, v in self.answer.items()
                                          if k != "coverage"})
            self.trace.event("write", tier="plan", what="answer",
                             text=args["verdict"])
            return done("answer", "run complete")

        return [_plan, _inspect, _move, _answer]

    # ------------------------------------------------------------- hooks
    async def _staple_state(self, input_data, tool_use_id, context):
        """Push state onto every tool result — the model cannot forget to look.

        A pull costs a turn and can be skipped; a push cannot. Empty deltas
        emit nothing, so a no-op inspect adds no noise.
        """
        moved = self._prev["cell"] != self._snapshot()["cell"]
        blocks = [self._delta()]
        if moved:
            # The menu only changes when the arm does, so it rides the move
            # rather than repeating every turn — a menu re-stated unchanged is
            # just another snapshot rotting in an append-only log.
            blocks.append(render.action_menu(self.tools, self.cell,
                                             self.agent_seen, nav=self._nav(),
                                             seen=self.view_notes))
        text = "\n".join(b for b in blocks if b)
        if not text:
            return {}
        # Traced as its own kind: this is context the model was GIVEN, not text
        # it wrote. Telling those apart is the whole point of watching a run.
        self.trace.event("state", text=text)
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                       "additionalContext": text}}

    async def _gate(self, input_data, tool_use_id, context):
        """The safety gate. THIS is where the live executor binds.

        Prompt-level safety fails in 49-73% of vulnerable tasks, so the veto
        lives in code between the tool call and the actuator, not in the
        system prompt. In replay there is no actuator: the only invariant that
        exists is the grid itself. When `move` drives a real arm, this is
        where `inspection.safety.config_is_safe` / `path_is_safe` run, and a
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
        # NOT `view_at(None)`: `store.views(cell=None)` means "every view",
        # so that would hand back the newest capture of the whole run. The
        # survey is the record whose cell IS None (replay.py: pose_id 0).
        surveys = [v for v in self.store.views() if v.cell is None]
        survey = self.tools.get_view(surveys[0] if surveys
                                     else self.store.views()[0])
        prompt = render.opening(self.question, survey.text, self.tools,
                                self.agent_seen, nav=self._nav())

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
    from inspection.eyes.models import StubVlm
    import os
    if os.environ.get("GEMINI_API_KEY"):
        from inspection.eyes.models import GeminiVlm
        vlm = GeminiVlm()
    else:
        # Runs the whole loop with no vision stack: the shape is testable
        # before the API keys are.
        print("[brain] GEMINI_API_KEY unset — vision subagent is a stub")
        vlm = StubVlm([{"evidence": ["stub: no vision backend"],
                        "reasoning": "stub", "answer": "unknown"}] * 50)
    brain = Brain(run_dir, question, vlm=vlm, model=model)
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
