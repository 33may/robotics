"""e05 keep (RQ3, RQ4): the views of a run arrive one by one; every node of a new view is placed
in the existing S/T/V by the agent (same / under / drop), or by a geometry-only baseline.

First view: coloured with the chosen lift; its root is the object. Each later view:
  1. its own e02 tree is lifted (same lift);
  2. the current T is rendered into the view (label render) -> which existing parts are
     visible, and for each new node the share of its rows each existing part holds;
  3. decisions per node: `same{node, part}` (extend the part), `under{node, part, name}` (new
     part under it), `drop{ids}`; undecided nodes get defaults (root -> same object; node ->
     under its parent's part, own name);
  4. apply, check(), count conflicts.
Modes: `baseline` (same to the deepest candidate child of the resolved parent holding >= 50 % of
the node's rows, else under) and `agent` (the in-loop VlmAgent on Gemini ER-2 with the verbs
tree / show / existing / show_existing / same / under / drop + crop / read_text / note). The
baseline is always computed for the comparison; the agent's decisions are what gets applied in
agent mode. Both are scored against evidence/truth_pairs.json.

Run:  p inspection/investigation/iwm/e05_parts_in_v/keep.py --run 2209-compute-ray-demo5 --mm 1 --mode baseline|agent
Outputs  inspection/data/iwm/e05_parts_in_v/<run>/<mm>mm/keep_<mode>/after_<view>/ (STV, run.json, transcripts)
         evidence/agent_vs_truth.json, evidence/conflicts.json, evidence/keep_<run>_<mm>mm_<mode>.jpg
"""
import argparse
import json
import time
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e02_agent_tree.iwm_tools import COLORS, IwmTree
from inspection.investigation.iwm.e05_parts_in_v import common as C5
from inspection.investigation.iwm.e05_parts_in_v import lift as L
from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface

SAME_SHARE = 0.5        # baseline: same when one candidate holds this share of the node's rows
MIN_VISIBLE_PX = 150    # an existing part counts as visible in a view above this many pixels
SMALL_MM = 20.0


# ====================================================================== decisions (no model)
def overlap_shares(stv, rows):
    """{part: share of `rows` the part HOLDS} by row membership (a part keeps rows it lost the label
    of in a conflict), only parts with a share > 0."""
    rows = np.unique(np.asarray(rows, np.int64))
    if not len(rows):
        return {}
    out = {}
    for p, part in stv.parts.items():
        if len(part.rows):
            n = len(np.intersect1d(rows, part.rows, assume_unique=True))
            if n:
                out[p] = n / len(rows)
    return out


def _best(stv, cands, shares):
    """The deepest candidate holding >= SAME_SHARE, else None."""
    ok = [p for p in cands if shares.get(p, 0.0) >= SAME_SHARE]
    return max(ok, key=lambda p: (stv.depth(p), shares[p])) if ok else None


def baseline_decisions(stv, view, rows):
    out, target = {}, {}
    for nid in L.node_order(view["tree"]):
        nd = view["nodes"][nid]
        sh = overlap_shares(stv, rows.get(nid, np.empty(0, np.int64)))
        if nd["parent"] is None:
            best = _best(stv, [stv.root] + stv.parts[stv.root].children, sh)
            out[nid] = ("same", best if best is not None else stv.root)
        else:
            pt = target[nd["parent"]]
            best = _best(stv, stv.parts[pt].children, sh) if isinstance(pt, int) else None
            out[nid] = ("same", best) if best is not None else ("under", pt, nd["name"])
        target[nid] = out[nid][1] if out[nid][0] == "same" else ("node", nid)
    return out


def resolve_defaults(stv, view, decisions):
    """Fill the undecided nodes: root -> same object; node -> under its parent's part, own name."""
    out, target = dict(decisions), {}
    for nid in L.node_order(view["tree"]):
        nd = view["nodes"][nid]
        if nid not in out:
            out[nid] = ("same", stv.root) if nd["parent"] is None else ("under", target[nd["parent"]], nd["name"])
        d = out[nid]
        target[nid] = d[1] if d[0] == "same" else ("node", nid) if d[0] == "under" else target.get(nd["parent"], stv.root)
    return out


def apply_decisions(stv, view, rows, decisions):
    decisions = resolve_defaults(stv, view, decisions)
    m, res = {}, {"same": 0, "under": 0, "drop": 0, "parent_disagree": 0}
    c0 = len(stv.conflicts)
    for nid in L.node_order(view["tree"]):
        nd = view["nodes"][nid]
        d = decisions[nid]
        r = rows.get(nid, np.empty(0, np.int64))
        det = (view["id"], nid)
        if d[0] == "same":
            part = d[1]
            stv.extend(part, r, detection=det, alias=nd["name"])
            m[nid] = part
            res["same"] += 1
            if nd["parent"] is not None and stv.parts[part].parent != m.get(nd["parent"]):
                res["parent_disagree"] += 1
        elif d[0] == "under":
            tgt = d[1]
            if isinstance(tgt, tuple):              # ("node", k) -> the part that node became (else the tree parent's)
                tgt = m.get(tgt[1], m.get(nd["parent"], stv.root))
            m[nid] = stv.add(tgt, d[2] or nd["name"] or f"#{nid}", rows=r, text=nd.get("text"), detection=det)
            res["under"] += 1
        else:
            m[nid] = m.get(nd["parent"], stv.root)
            res["drop"] += 1
    stv.meta[view["id"]] = m
    res["conflicts"] = len(stv.conflicts) - c0
    res["decisions"] = {str(k): [str(x) for x in v] for k, v in decisions.items()}
    return res


# ====================================================================== scoring against the truth
def expected_of(truth, stv, view_id, nid):
    """The part this node should be (from the truth pairs via the earlier view's meta), 'new', or None."""
    for p in truth["pairs"]:
        if p["b"] == view_id and p["b_id"] == nid:
            return stv.meta.get(p["a"], {}).get(p["a_id"])
    for w in truth.get("within", []):
        if w["view"] == view_id and w["id"] == nid:
            return ("within", stv.meta.get(w["a"], {}).get(w["a_id"]))
    for u in truth.get("uncertain", []):
        if u.get("b") == view_id and u.get("b_id") == nid:
            return None
    for n in truth.get("new", []):
        if n["view"] == view_id and n["id"] == nid:
            return "new"
    return None


def score(decisions, truth, stv, view, rows):
    keys = ("recovered", "wrong", "invented", "missed", "correct_new", "coarse", "within_ok")
    out = {**{k: 0 for k in keys}, "unscored": 0, "per_node": {}}
    by = {"small": {k: 0 for k in keys}, "large": {k: 0 for k in keys}}
    for nid, d in decisions.items():
        nid = int(nid)
        exp = expected_of(truth, stv, view["id"], nid)
        if exp is None:
            out["unscored"] += 1
            continue
        width = L.extent_mm(stv.s, rows.get(nid, np.empty(0, np.int64)))[0]
        cls = "small" if width < SMALL_MM else "large"
        tgt = d[1] if len(d) > 1 else None
        if isinstance(exp, tuple):                                   # within an existing part
            verdict = "within_ok" if d[0] in ("same", "under") and tgt == exp[1] else "wrong"
        elif d[0] == "same":
            verdict = ("recovered" if exp == tgt else "coarse" if exp == "new" and tgt == stv.root
                       else "invented" if exp == "new" else "wrong")
        else:
            verdict = "correct_new" if exp == "new" else "missed"
        out[verdict] += 1
        by[cls][verdict] += 1
        out["per_node"][str(nid)] = {"decision": [str(x) for x in d], "expected": str(exp), "verdict": verdict,
                                     "width_mm": width, "name": view["nodes"][nid]["name"]}
    out["by_size"] = by
    n_pairs = out["recovered"] + out["wrong"] + out["missed"]
    out["pairs_recovered_share"] = round(out["recovered"] / n_pairs, 3) if n_pairs else None
    return out


# ====================================================================== the agent
VA = None
_ORIG_DISPATCH = None
KEEP_VERBS = frozenset({"tree", "show", "existing", "show_existing", "same", "under", "drop"})
CURRENT = {"keeper": None}

RULES = """You are looking at ONE new camera frame of an object the robot has seen before. Two
things are given:
1. The NEW TREE of this frame: regions found in THIS frame, nested and named by an earlier pass
   (node ids #N, #0 = everything visible). Call tree to read it, show{node} to see a node with
   its children outlined and numbered.
2. The MODEL: the parts known from the earlier frames (part ids PN, P0 = the object), projected
   into this frame from the 3D reconstruction. Call existing to list the model's parts visible
   in this frame and, for every new node, which existing parts its pixels fall on. Call
   show_existing{node} to see the model's parts drawn over that node's area, named.

Your job: place EVERY node of the new tree, #0 included:
- same{node, part}: the node IS that existing part, the same physical thing seen again. Use it
  when the model already holds this port / label / face: the projection and the name agree,
  allowing a few millimetres of offset. Another side of the object with a similar name is NOT
  the same part.
- under{node, part, name}: the node is NEW and belongs under `part`: an existing model part
  (P<id>: its face, its group, or the object P0) OR a node of the new tree that you also place
  as new (#<id>, e.g. a new label's barcode goes under:#<label node>). Give the name it should
  carry. Keep the new tree's own nesting unless the model already holds the parent.
- drop{ids}: noise, not a part.
#0 is the whole visible frame: same{0, P0} when the frame shows the object, same{0, <face>}
when it shows one face only.

Procedure: tree, then existing, then show{0} and show_existing{node} where the overlap is
unclear; decide top-down: faces first, then the things on them, then their parts. One tool per
reply. Do not call a tool with the same arguments twice. You have a limited number of turns:
decide the clear nodes in bulk in your final reply.

Tools, one per reply, as {"tool": NAME, "args": {...}}:
  tree{}                       the new tree as text
  show{node}                   image of that node with its children outlined and numbered
  existing{}                   the model's parts visible here, and the overlap of every new node
  show_existing{node}          image of that node's area with the model's parts outlined and named
  same{node, part}             the node is that existing part
  under{node, part, name}      a new part under part = P<id> (model part) or #<id> (a new node placed as new)
  drop{ids}                    remove nodes
  crop{box}  read_text{box}  note{text}   as usual
Boxes you write are in the convention stated below; node ids are #numbers, part ids are P-numbers.

When done reply with {"evidence": [...], "reasoning": "...", "answer": "...",
"decisions": {"<node>": "same:P<id>" | "under:P<id>:<name>" | "under:#<node>:<name>" | "drop", ...}}
in that order; decisions must cover every node of the new tree."""


def _id(x):
    return int(str(x).strip().lstrip("#").lstrip("P").lstrip("p"))


def _ids(x):
    if isinstance(x, (int, str)):
        x = [x]
    return [_id(i) for i in x]


def _ref(x):
    """A parent reference: '#4' or ('node', 4) -> ('node', 4), a node of the new view; 'P4' / 4 / '4' -> part 4."""
    if isinstance(x, (tuple, list)) and len(x) == 2 and str(x[0]) == "node":
        return ("node", int(x[1]))
    s = str(x).strip()
    if s.startswith("#"):
        return ("node", int(s[1:].strip().lstrip("#")))
    return _id(s)


def normalise_decisions(raw):
    """{"<node>": "same:P3" | ["same", 3] | {"op": "under", "part": "P2", "name": ...} | "drop"} -> {node: tuple}."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            nid = _id(k)
            if isinstance(v, dict):
                op = str(v.get("op", v.get("decision", v.get("action", "")))).lower()
                part, name = v.get("part", v.get("id")), v.get("name")
            elif isinstance(v, (list, tuple)):
                op = str(v[0]).lower() if v else ""
                part = v[1] if len(v) > 1 else None
                name = v[2] if len(v) > 2 else None
            else:
                bits = str(v).strip().split(":", 2)
                op = bits[0].strip().lower()
                part = bits[1] if len(bits) > 1 else None
                name = bits[2].strip() if len(bits) > 2 else None
            if op.startswith("same"):
                if isinstance(part, (tuple, list)) or str(part).strip().startswith("#"):
                    continue                                  # same needs a model part, not a node
                out[nid] = ("same", _id(part))
            elif op.startswith("under"):
                out[nid] = ("under", _ref(part), str(name).strip() if name else None)
            elif op.startswith("drop"):
                out[nid] = ("drop",)
        except (ValueError, IndexError, TypeError):
            continue
    return out


def merge_final(tool_decisions, final, view, stv):
    """The final reply completes / overrides the tool-call decisions; unknown nodes, unknown parts
    and unknown new-node parents are dropped, ('node', k) parents are kept."""
    out = dict(tool_decisions)
    for nid, d in normalise_decisions(final).items():
        if nid not in view["nodes"]:
            continue
        if d[0] == "same" and d[1] not in stv.parts:
            continue
        if d[0] == "under":
            if isinstance(d[1], tuple):
                if d[1][1] not in view["nodes"] or d[1][1] == nid:
                    continue
            elif d[1] not in stv.parts:
                continue
        out[nid] = d
    return out


def _upright(a, rot):
    return a[::-1, ::-1] if rot == 180 else a


class Keeper:
    """One new view against the current STV; holds the label render and the agent's decisions."""

    def __init__(self, stv, vbg, size_m, view, rows):
        self.stv, self.view, self.rows = stv, view, rows
        self.lab, _, _, self.col = L.label_render(stv, vbg, size_m, view["T"], view["K"], color=True)
        self.lab_up, self.col_up = _upright(self.lab, view["rot"]), _upright(self.col, view["rot"])
        self.rgb_up = _upright(view["rgb"], view["rot"])
        self.decisions = {}
        self.log = []
        self.tree = self._iwm_tree()
        self.visible = self._visible()

    def _iwm_tree(self):
        t = IwmTree(self.rgb_up)
        for nid, nd in self.view["nodes"].items():
            t.nodes[nid] = {**{k: nd[k] for k in ("parent", "name", "text", "area", "bbox", "source")},
                            "children": list(nd["children"]), "mask": _upright(self.view["masks"][nid], self.view["rot"])}
        return t

    def _visible(self):
        vals, counts = np.unique(self.lab[self.lab >= 0], return_counts=True)
        px = {}
        for l, c in zip(vals.tolist(), counts.tolist()):
            for q in [l] + self.stv.ancestors(l):
                px[q] = px.get(q, 0) + c
        return {p: n for p, n in px.items() if n >= MIN_VISIBLE_PX}

    def pixels(self, part):
        return L.part_pixels(self.lab_up, self.stv, part)

    def box_of(self, part):
        ys, xs = np.nonzero(self.pixels(part))
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) else None

    def overlaps(self, nid, top=4):
        """The deepest substantial holders first (a leaf match must not be pushed out by its ancestors)."""
        sh = overlap_shares(self.stv, self.rows.get(nid, np.empty(0, np.int64)))
        items = sorted(((p, s) for p, s in sh.items() if s >= 0.05),
                       key=lambda x: (x[1] < 0.3, -self.stv.depth(x[0]), -x[1]))
        return items[:top]

    # ------------------------------------------------------------ verbs
    def existing(self, fmt_box):
        lines = ["model parts visible in this frame (P-id, name, where it sits, box, visible pixels):"]
        for p in sorted(self.visible, key=lambda p: (self.stv.depth(p), p)):
            b = self.box_of(p)
            lines.append(f"  P{p} {self.stv.parts[p].name} — {self.stv.path(p)} — box {fmt_box(b)} — {self.visible[p]} px")
        lines.append("new nodes and the model parts their pixels fall on (share of the node):")
        for nid in L.node_order(self.view["tree"]):
            nd = self.view["nodes"][nid]
            ov = self.overlaps(nid)
            txt = ", ".join(f"{100 * s:.0f}% on P{p} {self.stv.parts[p].name}" for p, s in ov) or "nothing (unseen area)"
            lines.append(f"  #{nid} {nd['name']}: {txt}")
        return "\n".join(lines)

    def show_existing(self, nid, img, margin=0.3, max_w=1100):
        if nid not in self.view["nodes"]:
            return f"show_existing: no node #{nid}", None
        nd = self.view["nodes"][nid]
        h, w = self.rgb_up.shape[:2]
        x0, y0, x1, y1 = C.expand_box(nd["bbox"], (h, w), margin)
        sub = self.rgb_up[y0:y1, x0:x1]
        parts = [p for p in self.visible if self.pixels(p)[y0:y1, x0:x1].sum() >= 50 and p != self.stv.root]
        parts = sorted(parts, key=lambda p: (self.stv.depth(p), p))
        masks = [self.pixels(p)[y0:y1, x0:x1] for p in parts]
        cols = [COLORS[j % len(COLORS)] for j in range(len(parts))]
        own = self.tree.nodes[nid]["mask"][y0:y1, x0:x1]
        out = C.outlines(sub, masks + [own], cols + [(255, 255, 255)], thick=2, dim=0.15)
        im = Image.fromarray(out)
        d = ImageDraw.Draw(im)
        font = C._font(15)
        for j, p in enumerate(parts):
            b = self.box_of(p)
            tx, ty = max(0, b[0] - x0), max(0, b[1] - y0 - 18)
            label = f"P{p} {self.stv.parts[p].name}"
            tb = d.textbbox((tx, ty), label, font=font)
            d.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=cols[j])
            d.text((tx, ty), label, fill="white", font=font)
        out = np.array(im)
        if out.shape[1] > max_w:
            out = C.resize(out, max_w)
        ov = ", ".join(f"P{p} {self.stv.parts[p].name} ({100 * s:.0f}%)" for p, s in self.overlaps(nid))
        text = (f"show_existing #{nid} {nd['name']}: region {x0},{y0}-{x1},{y1}; the node's own outline is white; "
                f"model parts drawn here: {', '.join(f'P{p} {self.stv.parts[p].name}' for p in parts) or 'none'}; "
                f"the node's pixels fall on: {ov or 'nothing'}")
        view = replace(img, rgb=out, box=(x0, y0, x1, y1), text=f"{img.text} · show_existing #{nid}")
        return text, view

    def decide_same(self, nid, part):
        if nid not in self.view["nodes"]:
            return f"same: no node #{nid}"
        if part not in self.stv.parts:
            return f"same: no part P{part}"
        self.decisions[nid] = ("same", part)
        self.log.append(("same", nid, part))
        return f"#{nid} {self.view['nodes'][nid]['name']} = P{part} {self.stv.parts[part].name} (will extend it)"

    def decide_under(self, nid, part, name):
        if nid not in self.view["nodes"]:
            return f"under: no node #{nid}"
        name = name or self.view["nodes"][nid]["name"]
        if isinstance(part, tuple):                            # ("node", k): a new node of this view as the parent
            k = part[1]
            if k not in self.view["nodes"] or k == nid:
                return f"under: no node #{k} to hang #{nid} under"
            self.decisions[nid] = ("under", part, name)
            self.log.append(("under", nid, f"#{k}", name))
            return f"#{nid} -> new part '{name}' under the new node #{k} {self.view['nodes'][k]['name']}"
        if part not in self.stv.parts:
            return f"under: no part P{part}"
        self.decisions[nid] = ("under", part, name)
        self.log.append(("under", nid, part, name))
        return f"#{nid} -> new part '{name}' under P{part} {self.stv.parts[part].name}"

    def decide_drop(self, ids):
        for i in ids:
            if i in self.view["nodes"] and i != 0:
                self.decisions[i] = ("drop",)
        return f"dropped {['#%d' % i for i in ids]}"


def _keep_dispatch(name, args, tools, verbs, writer, img, fmt_box):
    k = CURRENT["keeper"]
    if k is None or name not in KEEP_VERBS:
        return _ORIG_DISPATCH(name, args, tools, verbs, writer, img, fmt_box)
    if name == "tree":
        return ("new tree:\n" + k.tree.tree_text(fmt_box), None)
    if name == "show":
        return k.tree.show(_id(args.get("node", args.get("id", 0))), img)
    if name == "existing":
        return (k.existing(fmt_box), None)
    if name == "show_existing":
        return k.show_existing(_id(args.get("node", args.get("id", 0))), img)
    if name == "same":
        if str(args.get("part", "")).strip().startswith("#"):
            return ("same needs a model part P<id>; a #node of the new tree cannot be 'the same'", None)
        return (k.decide_same(_id(args["node"]), _id(args["part"])), None)
    if name == "under":
        return (k.decide_under(_id(args["node"]), _ref(args["part"]), args.get("name")), None)
    if name == "drop":
        return (k.decide_drop(_ids(args.get("ids", args.get("node", [])))), None)
    return (f"unknown tool {name!r}", None)


def install_agent():
    """Extend the in-loop tier for this process only (e02's pattern)."""
    global VA, _ORIG_DISPATCH
    from inspection.eyes.agents import vlm_agent as VA_
    VA = VA_
    if _ORIG_DISPATCH is None:
        _ORIG_DISPATCH = VA._dispatch
        VA.TOOLS = frozenset(VA.TOOLS | KEEP_VERBS)
        VA._dispatch = _keep_dispatch


def make_spec():
    tools = (VA.TOOLS - {"detect", "segment", "find", "name", "group"})
    return VA.VlmAgent(rules=RULES, kind="inspect", tool_names=tools,
                       emit=("evidence", "reasoning", "answer", "decisions"),
                       extras={"decisions": normalise_decisions}, max_turns=24)


def agent_decisions(keeper, view, out, model, verbs, trace=True):
    from inspection.eyes.store import FindingWriter, RunStore
    from inspection.eyes.tools import ViewRecord, ViewTools
    from inspection.record.run import Run
    run = Run.load(C.RUNS_DIR / view["run"])
    step = next(s for s in run.captured if s.id == view["step"])
    rec = ViewRecord(step)
    store = RunStore.create(out / "store", h_bins=12, v_elevs=(10.0, 40.0, 70.0), r=0.3)
    writer = FindingWriter(store)
    tools = ViewTools(run, writer=writer)
    CURRENT["keeper"] = keeper
    spec = make_spec()
    events = []
    t0 = time.time()

    def on_turn(n, max_turns, event, **fields):
        events.append({"turn": n, "event": event, "t": round(time.time() - t0, 1), **fields})
        if trace and event in ("tool", "done", "exhausted"):
            print(f"  t{n:02d} {event:9s} {fields.get('tool', '')} {json.dumps(fields.get('args', ''))[:70]} -> "
                  f"{str(fields.get('result', fields.get('answer', '')))[:100]}")

    task = ("Place every node of this frame's tree in the model: same for parts the model already "
            "holds, under for new parts, drop for noise.")
    finding = spec.run(tools, verbs, writer, model, task=task, view_rec=rec, on_turn=on_turn)
    dt = time.time() - t0
    final = normalise_decisions(finding.extras.get("decisions") or {})      # already normalised by extras=; idempotent
    decisions = merge_final(keeper.decisions, final, view, keeper.stv)
    CURRENT["keeper"] = None
    tool_calls = [e for e in events if e["event"] == "tool"]
    summary = {"seconds": round(dt, 1), "turns": len([e for e in events if e["event"] == "thinking"]),
               "tool_calls": len(tool_calls), "tools_used": [e["tool"] for e in tool_calls],
               "answer": finding.answer, "exhausted": finding.exhausted, "transcript": finding.transcript_rel,
               "decided_by_tools": len(keeper.decisions), "decided_in_final": len(final)}
    json.dump(events, open(out / "events.json", "w"), indent=1)
    return decisions, summary


# ====================================================================== the run
def label_figure(stv, vbg, size, views, path, title):
    """The label render from each view's pose, coloured per deepest part, with a legend."""
    ids = sorted(stv.parts)
    pal = {p: C._palette(len(ids), seed=3)[i] for i, p in enumerate(ids)}
    panels = []
    for v in views:
        lab, _, _, col = L.label_render(stv, vbg, size, v["T"], v["K"], color=True)
        img = (col * 0.55).astype(np.uint8)
        for p in ids:
            m = lab == p
            if m.any():
                img[m] = (0.4 * img[m] + 0.6 * np.array(pal[p])).astype(np.uint8)
        img = _upright(img, v["rot"])
        panels.append(C.label(C.resize(img, 640), v["id"], size=16))
    hmax = max(p.shape[0] for p in panels)
    panels = [np.pad(p, ((0, hmax - p.shape[0]), (0, 0), (0, 0)), constant_values=255) for p in panels]
    g = C.grid(panels, cols=len(panels))
    leg = Image.new("RGB", (g.shape[1], 18 * ((len(ids) + 3) // 4) + 10), "white")
    d = ImageDraw.Draw(leg)
    font = C._font(13)
    for i, p in enumerate(ids):
        x, y = 8 + (i % 4) * (g.shape[1] // 4), 6 + (i // 4) * 18
        d.rectangle([x, y + 2, x + 12, y + 14], fill=tuple(pal[p]))
        d.text((x + 16, y), f"P{p} {'  ' * stv.depth(p)}{stv.parts[p].name[:30]}", fill="black", font=font)
    out = np.concatenate([C.title_band(g, title), np.array(leg)], 0)
    Image.fromarray(out).save(path, quality=88)


def run_views(run, mm, mode, lift_name, truth, model=None, verbs=None, trace=True):
    views = C5.ORDER[run]
    vbg, size = C5.V(run, mm)
    surf = Surface.from_vbg(vbg, size, C5.trunc_m(size))
    base = C5.DATA / run / f"{mm}mm" / f"keep_{mode}"
    v0 = C5.load_view(views[0])
    rows0, _ = L.LIFTS[lift_name](v0, surf, vbg, size)
    stv = STV(surf)
    L.colour(stv, v0, rows0)
    stv.check()
    stv.save(base / f"after_{views[0]}")
    report = [{"view": views[0], "role": "first", **stv.summary()}]
    loaded = [v0]
    for vid in views[1:]:
        view = C5.load_view(vid)
        loaded.append(view)
        rows, _ = L.LIFTS[lift_name](view, surf, vbg, size)
        out = base / f"after_{vid}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        keeper = Keeper(stv, vbg, size, view, rows)
        t_render = time.time() - t0
        base_dec = baseline_decisions(stv, view, rows)
        agent_summary = None
        if mode == "agent":
            dec, agent_summary = agent_decisions(keeper, view, out, model, verbs, trace)
        else:
            dec = base_dec
        base_score = score(resolve_defaults(stv, view, base_dec), truth, stv, view, rows)
        agent_score = score(resolve_defaults(stv, view, dec), truth, stv, view, rows) if mode == "agent" else None
        res = apply_decisions(stv, view, rows, dec)
        stv.check()
        stv.save(out)
        C.run_record(out, {"id": vid, "run": run, "step": view["step"]},
                     {"name": f"keep {mode}", "lift": lift_name, "model": "gemini-robotics-er-2-preview" if mode == "agent" else None},
                     {"mm": mm, "same_share": SAME_SHARE}, {"render_s": round(t_render, 2), **stv.summary()})
        rec = {"view": vid, "role": "later", "applied": res, "baseline_decisions": {str(k): [str(x) for x in v] for k, v in base_dec.items()},
               "baseline_score": base_score, "agent_score": agent_score, "agent": agent_summary,
               "visible_parts": len(keeper.visible), **stv.summary()}
        report.append(rec)
        s = agent_score or base_score
        print(f"{run} {mm} mm {mode} after {vid}: parts {stv.summary()['parts']} conflicts +{res['conflicts']} "
              f"same {res['same']} under {res['under']} drop {res['drop']} | recovered {s['recovered']} wrong {s['wrong']} "
              f"invented {s['invented']} missed {s['missed']} new-ok {s['correct_new']}"
              + (f" | agent {agent_summary['turns']} turns {agent_summary['seconds']} s" if agent_summary else ""))
    label_figure(stv, vbg, size, loaded, C5.EVID / f"keep_{run}_{mm}mm_{mode}.jpg",
                 f"{run} at {mm} mm, {mode}: V's labels after {len(views)} views, from each view's pose")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(C5.ORDER))
    ap.add_argument("--mm", type=int, default=1)
    ap.add_argument("--mode", default="baseline", choices=["baseline", "agent"])
    ap.add_argument("--lift")
    ap.add_argument("--no-trace", action="store_true")
    a = ap.parse_args()
    lift_name = a.lift or json.load(open(C5.EVID / "lift.json"))["chosen"]
    truth = json.load(open(C5.EVID / "truth_pairs.json"))
    model = verbs = None
    if a.mode == "agent":
        install_agent()
        from inspection.eyes.models import GeminiVlm
        from inspection.eyes.verbs_local import LocalVerbs, SamOcrBackend
        model, verbs = GeminiVlm(), LocalVerbs(SamOcrBackend())
    p = C5.EVID / "agent_vs_truth.json"
    results = json.load(open(p)) if p.exists() else {}
    for run in a.runs:
        rep = run_views(run, a.mm, a.mode, lift_name, truth, model, verbs, trace=not a.no_trace)
        results[f"{run}_{a.mm}mm_{a.mode}"] = {"run": run, "mm": a.mm, "mode": a.mode, "lift": lift_name, "views": rep}
    json.dump(results, open(p, "w"), indent=1)
    conf = {k: [{"view": v["view"], "conflicts_total": v["conflicts"], "added": v.get("applied", {}).get("conflicts", 0),
                 "parts": v["parts"]} for v in r["views"]] for k, r in results.items()}
    json.dump(conf, open(C5.EVID / "conflicts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
