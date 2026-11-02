"""e07: one run's captures as INSPECT BLOCKS for the review viewer.

A block is the ordered list of events one capture produced, in the loop Anton laid out (2026-10-12):
capture arrives -> depth fused into V -> agentic segmentation (e02 trace) -> reconcile with the IWM
(e05 keep trace) -> commit (P, T, C before -> after). Each event says which IWM state it sees:
"before" (the state the capture started from) or "after" (the committed state).

Sources (read-only):
  e02  vbti/inspection/data/iwm/e02_agent_tree/<fid>/full/{tree_before,tree_final}.txt, store/
  e05  vbti/inspection/data/iwm/e05_parts_in_v/<run>/<mm>mm/keep_agent/after_<fid>/{parts.json,rows.npz,store/}
The agent's per-turn thinking is not in the stores (only tool calls and the final reply's
evidence/reasoning), so the trace shows what was logged, nothing more.
"""
import json
from pathlib import Path

from inspection.investigation.iwm.e05_parts_in_v import common as C5

MAX_RESULT = 4000


def latest_transcript(store):
    """The newest transcript in a store (stores accumulate transcripts across re-runs)."""
    files = sorted((Path(store) / "transcripts").glob("t*.json"))
    if not files:
        return None
    ts = [json.load(open(f)) for f in files]
    return max(ts, key=lambda t: t.get("t", 0))


def _artifact(store, step, idx, tool, t0):
    """The image a tool returned at turn idx, if it belongs to this transcript (artifacts are
    overwritten by name across re-runs, so an older file with the same name is not shown)."""
    p = Path(store) / "artifacts" / f"{int(step):03d}_{idx:02d}_{tool}.png"
    if p.exists() and p.stat().st_mtime >= t0 - 2:
        return p
    return None


def _fmt(v):
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def turn_events(store, stage):
    """Transcript turns -> events: one per tool call (args + result + image), one for the final reply."""
    t = latest_transcript(store)
    if t is None:
        return [], None
    out, pending = [], None
    for i, turn in enumerate(t["turns"]):
        if turn["type"] == "reply" and "tool" in turn:
            pending = turn
        elif turn["type"] == "tool_result":
            tool = turn.get("tool") or (pending or {}).get("tool", "?")
            out.append({"stage": stage, "kind": "tool", "tool": tool,
                        "args": _fmt((pending or {}).get("args", {})),
                        "result": _fmt(turn.get("result", ""))[:MAX_RESULT],
                        "image": _artifact(store, t["step_id"], i, tool, t.get("t", 0)), "state": "before"})
            pending = None
        elif turn["type"] == "reply":
            fields = {k: _fmt(v) for k, v in turn.items() if k != "type"}
            out.append({"stage": stage, "kind": "final", "tool": "final reply", "fields": fields,
                        "image": None, "state": "before"})
    return out, t


def diff(prev, cur, fid):
    """What the capture changed: new parts, parts extended by this view, renames, re-parents."""
    before = set(prev.parts) if prev is not None else set()
    new = sorted(set(cur.parts) - before)
    ext, moved, renamed = [], [], []
    for p in sorted(set(cur.parts) & before):
        a, b = prev.parts[p], cur.parts[p]
        if any(v == fid for v, _ in b.detections):
            ext.append(p)
        if a.parent != b.parent:
            moved.append(p)
        if a.name != b.name:
            renamed.append(p)
    gone = sorted(before - set(cur.parts))
    return {"new": new, "extended": ext, "moved": moved, "renamed": renamed, "gone": gone}


def tree_lines(stv, changes=None):
    """The IWM tree as indented lines with the capture's change tags."""
    changes = changes or {}
    tag = {}
    for k, mark in (("new", "NEW"), ("extended", "+view"), ("moved", "moved"), ("renamed", "renamed")):
        for p in changes.get(k, []):
            tag.setdefault(p, []).append(mark)
    lines = []

    def walk(p, d):
        part = stv.parts[p]
        t = f"  [{', '.join(tag[p])}]" if p in tag else ""
        det = ",".join(f"{v}#{n}" for v, n in part.detections)
        lines.append(f"{'  ' * d}P{p} {part.name}  ({len(part.rows)} vox; {det}){t}")
        for c in sorted(part.children, key=lambda c: stv.parts[c].name):
            walk(c, d + 1)

    walk(stv.root, 0)
    return lines


def plain_block(run, step, address, traced):
    """A captured step no agent was run on in e02/e05: its depth still fuses into V; the IWM is unchanged."""
    what = "the survey: boots V" if address is None else f"view {address}"
    return [{"stage": "capture", "kind": "capture", "tool": "capture arrives",
             "text": f"step **{step}** · run {run} · {what}", "image": None, "state": "before"},
            {"stage": "V", "kind": "fuse", "tool": "depth fused into V",
             "text": "code, no agent: this step's depth is integrated; the 3D shows V after it.",
             "image": None, "state": "before"},
            {"stage": "segment", "kind": "none", "tool": "no agent trace recorded",
             "text": f"e02/e05 ran the agent only on the frames {', '.join(traced)}; this step was not segmented, "
                     "so the IWM did not change here.", "image": None, "state": "before"}]


def block(run, fid, mm, prev, cur):
    """The inspect block of capture fid: events in loop order."""
    seg_dir = C5.E02 / fid / "full"
    keep_dir = C5.DATA / run / f"{mm}mm" / "keep_agent" / f"after_{fid}"
    run_, step = C5.FRAMES[fid]
    ev = [{"stage": "capture", "kind": "capture", "tool": "capture arrives",
           "text": f"frame **{fid}** · run {run} · step {step}", "image": None, "state": "before"},
          {"stage": "V", "kind": "fuse", "tool": "depth fused into V",
           "text": "code, no agent. NOTE: e05's V is the TSDF of ALL steps of the run, fused up front; "
                   "the incremental V per capture is not built yet.", "image": None, "state": "before"}]
    draft = (seg_dir / "tree_before.txt").read_text() if (seg_dir / "tree_before.txt").exists() else ""
    ev.append({"stage": "segment", "kind": "start", "tool": "segmentation starts (e02, blind to the IWM)",
               "text": "draft tree from e01's masks:\n```\n" + draft[:MAX_RESULT] + "\n```", "image": None,
               "state": "before"})
    seg, seg_t = turn_events(seg_dir / "store", "segment")
    ev += seg
    final = (seg_dir / "tree_final.txt").read_text() if (seg_dir / "tree_final.txt").exists() else ""
    ev.append({"stage": "segment", "kind": "result", "tool": "frame tree",
               "text": "the frame tree the segmentation left:\n```\n" + final[:MAX_RESULT] + "\n```",
               "image": None, "state": "before"})
    keep, keep_t = turn_events(keep_dir / "store", "reconcile")
    if keep_t is None:
        ev.append({"stage": "reconcile", "kind": "init", "tool": "IWM initialised by code",
                   "text": "first capture: no agent reconcile; the frame tree was copied in, its root became "
                           "the object (e05). Anton's rule since 2026-10-12: an object root first, the frame "
                           "tree as its child.", "image": None, "state": "before"})
    else:
        ev.append({"stage": "reconcile", "kind": "start", "tool": "reconcile starts (e05 keep agent)",
                   "text": f"task: {keep_t.get('task', '')}\n\nmodel: {keep_t.get('model', '')}",
                   "prompt": keep_t.get("prompt", ""), "image": None, "state": "before"})
        ev += keep
    ch = diff(prev, cur, fid)
    ev.append({"stage": "commit", "kind": "commit", "tool": "IWM committed",
               "text": (f"new {len(ch['new'])} · extended {len(ch['extended'])} · moved {len(ch['moved'])} · "
                        f"renamed {len(ch['renamed'])} · gone {len(ch['gone'])} · conflicts in V "
                        f"{len(cur.conflicts)}"),
               "changes": ch, "image": None, "state": "after"})
    meta = {"fid": fid, "step": step, "seg_model": (seg_t or {}).get("model"), "keep_model": (keep_t or {}).get("model"),
            "seg_prompt": (seg_t or {}).get("prompt", "")}
    return ev, ch, meta
