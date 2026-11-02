"""e02: the in-loop vision agent (VlmAgent + Gemini ER-2) builds the IWM tree of one view.

The loop (`vbti/inspection/eyes/agents/vlm_agent.py`) is not modified: this script extends
its verb table and dispatcher at import time, for this process only, with the IWM
verbs over an IwmTree bound to the current frame.

Run (robo env, GPU for find, GEMINI_API_KEY set):
  p vbti/inspection/investigation/iwm/e02_agent_tree/agent_tree.py --frames c5.6 [--without find]
Outputs: vbti/inspection/data/iwm/e02_agent_tree/<frame>/<variant>/  (tree before/after, store,
         transcripts, artifacts, run.json)  and  evidence/<frame>_<variant>_{before,after}.png
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from inspection.eyes.agents import vlm_agent as VA
from inspection.eyes.store import FindingWriter, RunStore
from inspection.eyes.tools import ViewRecord, ViewTools
from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e02_agent_tree.iwm_tools import IwmTree
from inspection.record.run import Run

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
DATA = C.ROOT / "inspection" / "data" / "iwm" / "e02_agent_tree"

IWM_VERBS = frozenset({"tree", "show", "find", "name", "group", "drop"})
_ORIG_TOOLS = VA.TOOLS
_ORIG_DISPATCH = VA._dispatch
CURRENT = {"tree": None}            # the IwmTree of the frame being inspected


def _id(x):
    """Node ids arrive as 4, "4" or "#4"."""
    return int(str(x).strip().lstrip("#"))


def _ids(x):
    if isinstance(x, (int, str)):
        x = [x]
    return [_id(i) for i in x]


def _iwm_dispatch(name, args, tools, verbs, writer, img, fmt_box):
    t = CURRENT["tree"]
    if name not in IWM_VERBS or t is None:
        return _ORIG_DISPATCH(name, args, tools, verbs, writer, img, fmt_box)
    if name == "tree":
        return ("tree:\n" + t.tree_text(fmt_box), None)
    if name == "show":
        return t.show(_id(args.get("node", args.get("id", 0))), img)
    if name == "find":
        return (t.find(str(args.get("phrase", "")), verbs, img), None)
    if name == "name":
        return (t.name(_id(args["id"]), args.get("name", ""), args.get("text")), None)
    if name == "group":
        return (t.group(_ids(args.get("ids", [])), args.get("name", "group")), None)
    if name == "drop":
        return (t.drop(_ids(args.get("ids", args.get("id", [])))), None)
    return (f"unknown tool {name!r}", None)


# Extend the tier for this process only. Specs built after this see the new verbs;
# the production specs (inspect/survey/evidence) keep their own subsets.
VA.TOOLS = frozenset(_ORIG_TOOLS | IWM_VERBS)
VA._dispatch = _iwm_dispatch

RULES = """You are looking at ONE captured camera frame of an object. A segmentation
model has already cut the frame into regions and code has nested them by containment
into a draft tree: node #0 is the whole object; every other node is a region inside
its parent. Many draft nodes are noise: plain surface fragments, duplicates, splinters.

Your job: turn the draft into the most logical levelled tree of what is visible, the
way a person would describe the object:
- depth 1: the object's visible SIDES (faces); on a box or device each face is a node.
- depth 2: the THINGS on a side: stickers, labels, connectors/ports, buttons, holes,
  logos, a barcode label. Repeated things of one kind that sit together become one
  group node ("array of ports", "row of barcodes") with the things under it.
- depth 3: the PARTS of a thing: a port's printed label and the port itself; a
  sticker's logo, barcode, QR code, text lines; a label's lines.
- deeper only when there is a real part (a barcode's digits, a logo's symbol).
Every node you keep must be something you can name. A region that is only a patch of
plain surface, a splinter of another node, or the same thing twice is dropped.

Procedure, one level at a time:
1. Call tree to read the draft.
2. Call show{node: 0}: the object with its depth-1 regions outlined and numbered. Name
   the sides (name{id, name}); group or drop the rest.
3. For each side, call show{node: <side id>}; name what you recognise, group repeated
   things, drop fragments. If you see a thing that has no region, call find{phrase}
   with a short noun ("port", "sticker", "barcode", "label", "logo", "text") — it adds
   regions for that noun into the tree, then show the side again to see them.
4. Descend into stickers, labels and ports the same way (show{node}); name their parts,
   drop fragments. Use read_text{box} when you need the exact wording of a label; put
   it in name{id, name, text}.
5. Stop when the levels read like the description above. Name the remaining nodes in
   bulk in the `tree` field of your final reply (every id you keep -> its name); use
   name{} during the work only when you also give the exact text read on it. Nodes
   that end without a name are discarded as noise.

Tools, one per reply, as {"tool": NAME, "args": {...}}:
  tree{}                      the current tree as text
  show{node}                  image of that node with its children outlined and numbered
  find{phrase}                add regions of a named thing (short noun)
  name{id, name, text?}       name a node (text = exact wording read on it, optional)
  group{ids, name}            make one node over several siblings
  drop{ids}                   remove nodes; their children move up one level
  crop{box}                   zoom into a box of the full frame
  read_text{box}              OCR the box
  note{text}                  record something you want to say to the planner
Boxes you write are in the convention stated below; node ids are the #numbers.
Do not call a tool with the same arguments twice. You have a limited number of turns;
do not name every letter — stop at the level a person would stop at.

When done reply with {"evidence": [...], "reasoning": "...", "answer": "...",
"tree": {"<id>": "<name>", ...}} in that order: evidence = what you saw, reasoning =
how you decided the levels, answer = one line describing the tree (sides, things,
parts), tree = every node you named. Do not report a confidence number."""


def normalise_tree(raw):
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        try:
            out[str(_id(k))] = str(v)          # the model writes "#3" as often as "3" (e07 step 6 lost 33 names)
        except (TypeError, ValueError):
            continue
    return out


def make_spec(without=()):
    tools = (VA.TOOLS - frozenset(without)) - {"detect", "segment"}
    # kind must be one the record schema knows (survey|inspect|move|evidence)
    return VA.VlmAgent(rules=RULES, kind="inspect", tool_names=tools,
                       emit=("evidence", "reasoning", "answer", "tree"),
                       extras={"tree": normalise_tree}, max_turns=24)


def step_record(frame):
    run = Run.load(C.RUNS_DIR / frame["run"])
    step = next(s for s in run.captured if s.id == frame["step"])
    return run, ViewRecord(step)


def run_frame(frame, variant, without, model, verbs, trace):
    out = DATA / frame["id"] / variant
    out.mkdir(parents=True, exist_ok=True)
    run, rec = step_record(frame)
    store = RunStore.create(out / "store", h_bins=12, v_elevs=(10.0, 40.0, 70.0), r=0.3)
    writer = FindingWriter(store)
    tools = ViewTools(run, writer=writer)

    tree = IwmTree.from_e01(frame)
    tree.save(out, "before")
    Image.fromarray(tree.render_levels(f"{frame['id']} draft tree (before the agent)")).save(
        EVIDENCE / f"{frame['id']}_{variant}_before.png")
    CURRENT["tree"] = tree
    spec = make_spec(without)
    events = []

    def on_turn(n, max_turns, event, **fields):
        rec_ = {"turn": n, "event": event, "t": round(time.time() - t0, 1), **fields}
        events.append(rec_)
        if trace and event in ("tool", "done", "exhausted"):
            print(f"  t{n:02d} {event:9s} {fields.get('tool', '')} "
                  f"{json.dumps(fields.get('args', ''))[:80]} -> "
                  f"{str(fields.get('result', fields.get('answer', '')))[:110]}")

    t0 = time.time()
    task = ("Build the levelled tree of this view: sides, the things on them, their "
            "parts. Name every node you keep, drop the noise.")
    finding = spec.run(tools, verbs, writer, model, task=task, view_rec=rec, on_turn=on_turn)
    dt = time.time() - t0
    # final names from the appendix, applied to the tree state as well
    for k, v in (finding.extras.get("tree") or {}).items():
        i = int(k)
        if i in tree.nodes and not tree.nodes[i]["name"]:
            tree.nodes[i]["name"] = v
    tree.save(out, "after")
    # the noise rule: what the agent did not name is dropped (children move up)
    unnamed = [i for i, n in tree.nodes.items() if i != 0 and not n["name"]]
    tree.drop(unnamed)
    tree.nodes[0]["name"] = tree.nodes[0]["name"] or "object"
    tree.save(out, "final")
    Image.fromarray(tree.render_levels(
        f"{frame['id']} the agent's tree after dropping unnamed nodes ({variant})")).save(
        EVIDENCE / f"{frame['id']}_{variant}_final.png")
    (EVIDENCE / f"{frame['id']}_{variant}_tree.txt").write_text(tree.as_text() + "\n")
    tool_calls = [e for e in events if e["event"] == "tool"]
    summary = {"frame": frame["id"], "variant": variant, "seconds": round(dt, 1),
               "turns": len([e for e in events if e["event"] == "thinking"]),
               "tool_calls": len(tool_calls),
               "tools_used": [f"{e['tool']}{json.dumps(e.get('args', {}))}" for e in tool_calls],
               "answer": finding.answer, "exhausted": finding.exhausted,
               "nodes_before": len(json.load(open(out / "tree_before.json"))["nodes"]),
               "nodes_after": len(json.load(open(out / "tree_after.json"))["nodes"]),
               "unnamed_dropped": len(unnamed),
               "nodes_final": len(tree.nodes),
               "transcript": finding.transcript_rel}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    json.dump(events, open(out / "events.json", "w"), indent=1)
    C.run_record(out, {"id": frame["id"], "run": frame["run"], "step": frame["step"]},
                 {"name": VA._model_id(model), "verbs": "sam3+paddleocr"},
                 {"variant": variant, "without": list(without), "max_turns": spec.max_turns,
                  "sets": "e01 unsam ladder + sam3_text"}, {"seconds": round(dt, 1)})
    print(f"{frame['id']} {variant}: {summary['turns']} turns, {summary['tool_calls']} tool calls, "
          f"{dt:.0f}s, nodes {summary['nodes_before']} -> {summary['nodes_after']} -> "
          f"{summary['nodes_final']} named, exhausted={finding.exhausted}")
    print(f"  answer: {finding.answer}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*")
    ap.add_argument("--without", nargs="*", default=[])
    ap.add_argument("--variant")
    ap.add_argument("--no-trace", action="store_true")
    args = ap.parse_args()
    EVIDENCE.mkdir(exist_ok=True)
    from inspection.eyes.models import GeminiVlm
    from inspection.eyes.verbs_local import LocalVerbs, SamOcrBackend
    model = GeminiVlm()
    verbs = LocalVerbs(SamOcrBackend())
    variant = args.variant or ("full" if not args.without else "without_" + "_".join(args.without))
    for f in C.frames(args.frames):
        run_frame(f, variant, args.without, model, verbs, trace=not args.no_trace)


if __name__ == "__main__":
    main()
