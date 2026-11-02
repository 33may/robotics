"""e07: the IWM inspect loop, actually executed on a recorded run, every turn recorded.

Per captured step, in capture order (Anton, 2026-10-12):
  1. capture arrives; its depth is fused into V (code, no agent). Step 0 = the survey: V only.
  2. masks for the frame (UnSAMv2 ladder + SAM 3 text, e01's sets; generated when missing),
     the draft tree by containment (e01/e02 rule).
  3. agentic segmentation, BLIND to the IWM: e02's agent (VlmAgent + Gemini ER-2, verbs
     tree/show/find/name/group/drop) builds the frame tree; unnamed nodes are dropped.
  4. reconcile with the IWM: the first inspect hangs the frame tree under the object root (code);
     later inspects run e05's keep agent (tree/existing/show/show_existing/same/under/drop).
  5. commit: the decisions applied to the STV (P, T, C after), invariants checked.
Every tool call is recorded with its full result, the image it returned, and the state right after
it (the frame tree for segmentation, the decisions so far for reconcile). Every mask any node ever
had is lifted into the V of that step (L3) so the viewer can draw it in 3D.

V grows: one TSDF, the step integrated before its agents run; the STV's rows are re-indexed onto
the new surface after each fusion (rows are voxels; a voxel that left the surface is dropped).

Record: inspection/data/iwm/e07_update_loop/loops/<run>_<stamp>/
  loop.json                                 run, steps, settings, timings
  step_<id>/events.json                     the inspect block (see EVENT KINDS in the viewer)
  step_<id>/frame/{masks.npz, ids.json}     the mask pool of the frame (node id -> mask, upright)
  step_<id>/frame/lift.npz                  node id -> surface voxel centres (L3, this step's V)
  step_<id>/V_display.ply                   V after this step, thinned for display
  step_<id>/iwm/{parts.json, rows.npz, centres.npz}   the STV after the commit
  step_<id>/seg_store/, keep_store/         the agents' own stores (transcripts, images)
  step_<id>/feedback.json                   Anton's notes on the step (--pause, written by the viewer)

--pause (Anton 2026-10-15, behaviour tuning): one command; the viewer (port 8084) runs inside this
process and follows the loop. After every inspect step's commit the loop WAITS; type feedback in the
viewer and press Enter -> step_<id>/feedback.json, the loop continues. The feedback is only stored; it
is applied in the session after the run, never injected into the agents' prompts.

Run (robo env, GPU, GEMINI_API_KEY set):
  /home/anton/miniconda3/envs/robo/bin/python -u -m inspection.investigation.iwm.e07_update_loop.loop \
      --run 2209-box-ray-demo1 [--steps 0 1 2 3 4] [--mm 1] [--pause]
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
from PIL import Image

from inspection.cell.geometry import MAX_RANGE_M
from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e01_all_masks import tree as T
from inspection.investigation.iwm.e02_agent_tree import agent_tree as A2          # installs e02's verbs
from inspection.investigation.iwm.e02_agent_tree.iwm_tools import DEFAULT_SETS, IwmTree
from inspection.investigation.iwm.e04_voxel_scene import build as B
from inspection.investigation.iwm.e05_parts_in_v import common as C5
from inspection.investigation.iwm.e05_parts_in_v import keep as K
from inspection.investigation.iwm.e05_parts_in_v import lift as L
from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Part, Surface

OUT = C.ROOT / "inspection" / "data" / "iwm" / "e07_update_loop"
LIVE = {"path": None, "lines": []}       # <loop>/live.json: the viewer's live progress panel


def say(msg, **_):
    """print + append to <loop>/live.json (last 14 lines) so the viewer shows progress during a step"""
    print(msg, flush=True)
    if LIVE["path"] is None:
        return
    LIVE["lines"] = (LIVE["lines"] + [f"{datetime.now():%H:%M:%S} {msg.strip()}"])[-14:]
    tmp = LIVE["path"].with_suffix(".tmp")
    json.dump({"lines": LIVE["lines"]}, open(tmp, "w"), ensure_ascii=False)
    tmp.replace(LIVE["path"])
MASKS = OUT / "masks"
UNSAM_G = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]
DISPLAY_M = 0.002

K.install_agent()                       # e05's verbs on top of e02's (each falls through to the other)
VA = A2.VA
_INNER = VA._dispatch
REC = {"log": None, "snap": None}


def _recording_dispatch(name, args, tools, verbs, writer, img, fmt_box):
    """Every tool call with its full result and the state right after it."""
    result, new_img = _INNER(name, args, tools, verbs, writer, img, fmt_box)
    if REC["log"] is not None:
        REC["log"].append({"tool": name, "args": args, "result": str(result), "image": None,
                           "snap": REC["snap"]() if REC["snap"] else None})
    return result, new_img


VA._dispatch = _recording_dispatch


def _robust_id(x):
    """'#2', 'P5', 4, and '#2 connector panel' (the agent sometimes appends the name) -> the number."""
    import re
    m = re.search(r"\d+", str(x))
    if m is None:
        raise ValueError(f"no id in {x!r}")
    return int(m.group())


A2._id = K._id = _robust_id

SEG_STRUCTURE = """

OBJECT STRUCTURE (rule for box-like objects): #0 is the whole object. Depth 1 = the object's FACES,
one node per visible face (the top, each side). A bevel / chamfer band is NOT a face: make no node
for it; a detail on it goes under the side it belongs to (Anton 2026-10-15).
Every detail (sticker, label, port, group of ports, logo, text) goes under the face it physically
sits on, never directly under #0. The draft tree is given in the task below, so you do not need to
call tree first."""

KEEP_STRUCTURE = """

OBJECT STRUCTURE (rule, overrides the #0 line above): P0 is the object and its children are FACES
only. #0 is always same{0, P0}. A face of the new tree is either the same as an existing face
(same{node, P<face>}) or a NEW face under P0 (under{node, P0, name}). Details go under a face
(P<face> or #<new face>), never directly under P0. Face names must be unique in the model: never
give a new face the name of an existing face. The new tree and the model's overlap are given in the
task below, so you do not need to call tree or existing first."""


def fmt_box_of(model, rgb):
    h, w = rgb.shape[:2]
    to_model = getattr(model, "to_model_box", None)
    return (lambda b: str(list(to_model(b, w, h)))) if to_model else (lambda b: str(tuple(int(v) for v in b)))


def enforce_structure(decisions, view, stv):
    """Details never directly under the object. Returns the corrected decisions and the list of
    corrections made (shown in the record).

    A node the segmenter left under #0 is NOT assumed to be a face (2026-10-15: that rule moved the
    agent's correct 'model label' -> chamfer and 'antenna ports' -> P7 up to the object, P9/P24);
    the reconcile agent's placement of it stands."""
    nodes, root, fixes = view["nodes"], stv.root, []
    out = dict(decisions)
    if out.get(0) != ("same", root):
        fixes.append(f"#0 {out.get(0)} -> same P{root}")
        out[0] = ("same", root)
    for nid, d in list(out.items()):
        nd = nodes.get(nid)
        if nd is None or nid == 0 or d[0] != "under":
            continue
        fparent = nd["parent"]
        if fparent not in (None, 0) and d[1] == root:           # a detail: hangs on its face
            pd = out.get(fparent)
            new_parent = pd[1] if pd and pd[0] == "same" else ("node", fparent)
            fixes.append(f"#{nid} detail under P{root} -> under {new_parent}")
            out[nid] = ("under", new_parent, d[2])
    return out, fixes


# ====================================================================== V and the STV
def new_grid(size, dev):
    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=size, block_resolution=B.BLOCK_RES,
        block_count=8000, device=dev)            # grows by doubling on demand; 200000 pre-reserved 16 GB


def fuse(vbg, st, intr, scale, size, dev):
    K_ = B.intrinsic_tensor(intr)
    tm = max(B.TRUNC_VOXELS, B.MIN_TRUNC_M / size)
    d = B.clean_depth(st["depth"], scale, None)
    depth = o3d.t.geometry.Image(o3c.Tensor(d.astype(np.uint16))).to(dev)
    color = o3d.t.geometry.Image(o3c.Tensor(st["rgb"].astype(np.uint8))).to(dev)
    ext = o3c.Tensor(np.linalg.inv(st["T"]), o3c.float64)
    coords = vbg.compute_unique_block_coordinates(depth, K_, ext, 1.0 / scale, MAX_RANGE_M, trunc_voxel_multiplier=tm)
    vbg.integrate(coords, depth, color, K_, K_, ext, 1.0 / scale, MAX_RANGE_M, trunc_voxel_multiplier=tm)


def remap(stv, surf):
    """The STV onto a new surface (V grew): rows are voxels, re-indexed by their (i, j, k)."""
    t = STV.__new__(STV)
    t.s, t.meta, t._next, t.root = surf, stv.meta, stv._next, stv.root
    old = stv.s

    def move(rows):
        if len(rows) == 0:
            return np.empty(0, np.int64)
        r = surf.rows_at(old.centres(rows))
        return np.unique(r[r >= 0]).astype(np.int64)

    t.parts = {p: Part(q.id, q.name, q.text, q.parent, list(q.children), move(q.rows), list(q.detections), list(q.aliases))
               for p, q in stv.parts.items()}
    t.label = np.full(len(surf), -1, np.int32)
    lab_rows = np.nonzero(stv.label != -1)[0]
    if len(lab_rows):
        nr = surf.rows_at(old.centres(lab_rows))
        ok = nr >= 0
        t.label[nr[ok]] = stv.label[lab_rows[ok]]
    t.conflicts = []
    for r, a, b in stv.conflicts:
        nr = int(surf.rows_at(old.centres([r]))[0])
        if nr >= 0:
            t.conflicts.append((nr, a, b))
    return t


def display_mesh(vbg, run, path):
    mesh = vbg.extract_triangle_mesh(weight_threshold=B.MESH_WEIGHT).to_legacy()
    cloud = np.load(C.RUNS_DIR / run / "fused" / "cloud.npy")
    box = o3d.geometry.AxisAlignedBoundingBox(cloud.min(0) - [B.SCENE_MARGIN_M, B.SCENE_MARGIN_M, 0.03],
                                              cloud.max(0) + [B.SCENE_MARGIN_M, B.SCENE_MARGIN_M, 0.10])
    mesh = mesh.crop(box).simplify_vertex_clustering(voxel_size=DISPLAY_M)
    o3d.io.write_triangle_mesh(str(path), mesh)


# ====================================================================== masks and the draft tree
def e01_id(run, step):
    for f in json.load(open(C.EVIDENCE / "frames_selected.json"))["frames"]:
        if f["run"] == run and f["step"] == step:
            return f["id"]
    return None


def ensure_masks(run, step, text_backend):
    """The e01 mask sets of a frame: e01's own when it made them, else generated here."""
    fid = e01_id(run, step)
    if fid and all((C.DATA / "masks" / fid / s / "masks.json").exists() for s in DEFAULT_SETS):
        return C.DATA / "masks" / fid, "e01"
    d = MASKS / run / f"{step:03d}"
    img = C.RUNS_DIR / run / "steps" / f"{step:03d}" / "rgb.png"
    for g in UNSAM_G:
        o = d / f"unsam_plus_g{g}"
        if not (o / "masks.json").exists():
            for device in ("cuda", "cpu"):
                cmd = (f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsam && cd {C.ROOT} && "
                       f"{'CUDA_VISIBLE_DEVICES= ' if device == 'cpu' else ''}"
                       f"python inspection/investigation/iwm/e01_all_masks/unsamv2_masks.py --image {img} "
                       f"--granularity {g} --out {o} --variant plus --device {device}")
                r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
                if r.returncode == 0:
                    break
                print(f"    unsam g{g} on {device} failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}", flush=True)
            else:
                raise RuntimeError(f"UnSAMv2 failed on {img} g{g}")
    o = d / "sam3_text"
    if not (o / "masks.json").exists():
        rgb = np.array(Image.open(img).convert("RGB"))
        masks, recs = [], []
        for noun in C.PLANNER_NOUNS:
            for box, score, m in text_backend.instances(rgb, noun):
                if score >= 0.3 and m.sum() > 0:
                    masks.append(m)
                    recs.append(C.mask_record(m, score=round(score, 3), phrase=noun))
        C.save_masks(o, masks, recs, {"strategy": "text", "nouns": C.PLANNER_NOUNS, "min_score": 0.3})
    return d, "generated"


def draft_tree(frame, root):
    """e02's IwmTree.from_e01 with the mask root as a parameter (same sets, same containment rule)."""
    rgb = C.load_rgb(frame)
    masks, recs = [], []
    for s in DEFAULT_SETS:
        d = root / s
        if not (d / "masks.json").exists():
            continue
        arr, meta = C.load_masks(d)
        g = meta["settings"].get("granularity")
        for m, r in zip(arr, meta["masks"]):
            masks.append(m)
            recs.append({**r, "source": s, **({"g": g} if g is not None else {})})
    obj, how = C.recorded_object(frame)
    if obj is None:
        arr, _ = C.load_masks(root / "unsam_plus_g0.9")
        areas = [m.sum() for m in arr]
        lim = 0.8 * rgb.shape[0] * rgb.shape[1]
        big = max(range(len(arr)), key=lambda i: areas[i] if areas[i] < lim else -1)
        obj, how = arr[big], "largest mask of unsam_plus_g0.9"
    ms, nodes, conflicts = T.build(masks, recs, obj, contain=0.9, dedupe_iou=0.8, ratio=1.25)
    t = IwmTree(rgb)
    for nd, m in zip(nodes, ms):
        t.nodes[nd["id"]] = {"parent": nd["parent"] if nd["id"] else None, "children": [], "name": None, "text": None,
                             "area": int(nd["area"]), "bbox": nd["bbox"],
                             "source": nd.get("phrase") or nd.get("source", ""), "mask": m}
    for i, nd in t.nodes.items():
        if nd["parent"] is not None:
            t.nodes[nd["parent"]]["children"].append(i)
    t._next = max(t.nodes) + 1
    t.object_from = how
    t.collapse_single_children()
    return t


def tree_snap(tree):
    return {i: {"parent": nd["parent"], "children": list(nd["children"]), "name": nd["name"], "text": nd["text"],
                "bbox": [int(v) for v in nd["bbox"]], "source": nd.get("source", "")} for i, nd in tree.nodes.items()}


# ====================================================================== agents
def run_agent(spec, run_obj, rec, store_dir, model, verbs, task, snap):
    from inspection.eyes.store import FindingWriter, RunStore
    from inspection.eyes.tools import ViewTools
    store = RunStore.create(store_dir, h_bins=12, v_elevs=(10.0, 40.0, 70.0), r=0.3)
    writer = FindingWriter(store)
    tools = ViewTools(run_obj, writer=writer)
    log = []
    REC["log"], REC["snap"] = log, snap
    t0 = time.time()

    def on_turn(n, max_turns, event, **fields):
        if event == "image" and log:
            log[-1]["image"] = fields.get("image")
        if event in ("tool", "done", "exhausted"):
            say(f"    t{n:02d} {event:9s} {fields.get('tool', '')} {json.dumps(fields.get('args', ''), default=str)[:70]} -> "
                  f"{str(fields.get('result', fields.get('answer', '')))[:90]}", flush=True)

    try:
        finding = spec.run(tools, verbs, writer, model, task=task, view_rec=rec, on_turn=on_turn)
    finally:
        REC["log"], REC["snap"] = None, None
    tr = json.load(open(Path(store_dir) / finding.transcript_rel)) if finding.transcript_rel else {}
    attach_images(log, tr, Path(store_dir))
    final = next((t for t in reversed(tr.get("turns", [])) if t["type"] == "reply" and "tool" not in t), {})
    return finding, log, {k: v for k, v in final.items() if k != "type"}, tr.get("prompt", ""), round(time.time() - t0, 1)


def attach_images(log, tr, store_dir):
    """The agent saves a tool's image as artifacts/<step>_<turn index>_<tool>.png without putting the path in
    the transcript; match each dispatched call (log, in order) to its tool_result turn by the result text."""
    turns = tr.get("turns", [])
    j = 0
    for x in log:
        while j < len(turns):
            t = turns[j]
            j += 1
            if t["type"] == "tool_result" and t.get("tool") == x["tool"] and str(t.get("result", "")).startswith(x["result"][:60]):
                p = store_dir / "artifacts" / f"{int(tr.get('step_id', 0)):03d}_{j - 1:02d}_{x['tool']}.png"
                if p.exists():
                    x["image"] = str(p.relative_to(store_dir))
                break


def make_view(run, step, st, intr, scale, tree):
    """e05's view dict from the live frame tree (raw orientation like the pose)."""
    sj = json.load(open(C.RUNS_DIR / run / "steps" / f"{step:03d}" / "step.json"))
    rot = int(sj.get("rgb_rotation_deg", 0) or 0)
    flip = (lambda a: np.ascontiguousarray(a[::-1, ::-1])) if rot == 180 else (lambda a: a)
    tj = tree.to_json()
    return {"id": f"s{step}", "run": run, "step": step, "T": st["T"], "K": C5.K_of(intr), "intr": intr, "scale": scale,
            "depth": st["depth"], "rgb": st["rgb"], "rot": rot, "tree": tj, "nodes": {n["id"]: n for n in tj["nodes"]},
            "masks": {i: flip(nd["mask"]) for i, nd in tree.nodes.items()}, "cam": st["T"][:3, 3].copy()}


def first_inspect(stv, view, rows):
    """Anton 2026-10-12: the object root first; the first inspect's frame tree hangs under it."""
    m = {}
    for nid in L.node_order(view["tree"]):
        nd = view["nodes"][nid]
        r = rows.get(nid, np.empty(0, np.int64))
        if nd["parent"] is None:                                 # the frame root IS the object (bug fixed 2026-10-12)
            stv.assign(stv.root, r, detection=(view["id"], nid), alias=nd["name"])
            m[nid] = stv.root
        else:
            m[nid] = stv.add(m[nd["parent"]], nd["name"] or f"#{nid}", rows=r, text=nd.get("text"),
                             detection=(view["id"], nid))
    stv.meta[view["id"]] = m
    return {f"#{k}": ["under", f"P{v}"] for k, v in m.items()}


# ====================================================================== the loop
def wait_feedback(out, sd, sid, info):
    """--pause: block until the viewer marks this step's feedback done."""
    info["waiting"] = sid
    json.dump(info, open(out / "loop.json", "w"), indent=1, default=str)
    say(f"step {sid}: PAUSED - review it in the viewer, type feedback + Enter to continue", flush=True)
    fb = sd / "feedback.json"
    while True:
        try:
            if json.load(open(fb)).get("done"):
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    info["waiting"] = None
    json.dump(info, open(out / "loop.json", "w"), indent=1, default=str)
    say(f"step {sid}: continuing ({len(json.load(open(fb)).get('notes', []))} notes)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2209-box-ray-demo1")
    ap.add_argument("--steps", nargs="*", type=int)
    ap.add_argument("--mm", type=int, default=1)
    ap.add_argument("--pause", action="store_true", help="live viewer in-process; wait for feedback after every inspect step")
    ap.add_argument("--port", type=int, default=8084)
    args = ap.parse_args()
    from inspection.eyes.models import GeminiVlm
    from inspection.eyes.tools import ViewRecord
    from inspection.eyes.verbs_local import LocalVerbs, Sam3Backend, SamOcrBackend
    from inspection.record.run import Run

    run, size = args.run, args.mm / 1000.0
    dev = B.device()
    steps, intr, scale = B.load_steps(run)
    if args.steps:
        steps = [s for s in steps if s["id"] in args.steps]
    out = OUT / "loops" / f"{run}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    LIVE["path"] = out / "live.json"     # set before the masks so their progress shows too
    text = Sam3Backend()
    backend = SamOcrBackend(sam3=text)
    backend.instances = text.instances          # find{} needs instances(); SamOcrBackend only delegates detect
    model, verbs = GeminiVlm(), LocalVerbs(backend)
    run_obj = Run.load(C.RUNS_DIR / run)
    mask_roots = {}
    for st in steps:                     # POC: masks made up front (a live loop makes them when the capture arrives)
        if st["address"] is not None:
            t0 = time.time()
            mask_roots[st["id"]] = ensure_masks(run, st["id"], text)
            say(f"masks step {st['id']}: {mask_roots[st['id']][1]} {time.time() - t0:.0f}s", flush=True)
    vbg = new_grid(size, dev)
    stv = None
    info = {"run": run, "mm": args.mm, "steps": [s["id"] for s in steps], "started": datetime.now().isoformat(timespec="seconds"),
            "segmentation": "blind (e02 agent)", "reconcile": "e05 keep agent", "lift": "L3", "pause": args.pause,
            "waiting": None, "per_step": []}
    json.dump(info, open(out / "loop.json", "w"), indent=1)
    say(f"loop -> {out.name}")
    if args.pause:                       # the viewer lives in this process; it waits for the first step by itself
        import threading
        from inspection.investigation.iwm.e07_update_loop.loop_viewer import serve
        threading.Thread(target=serve, args=(out, args.port), daemon=True).start()
    for st in steps:
        sid = st["id"]
        sd = out / f"step_{sid:03d}"
        sd.mkdir()
        t_step = time.time()
        ev = [{"stage": "capture", "kind": "capture", "tool": "capture arrives",
               "text": f"step {sid} · view {st['address']}"}]
        say(f"step {sid}: capture arrives ({st['address'] or 'survey'}), fusing depth")
        fuse(vbg, st, intr, scale, size, dev)
        surf = Surface.from_vbg(vbg, size, C5.trunc_m(size))
        if stv is not None:
            stv = remap(stv, surf)
        display_mesh(vbg, run, sd / "V_display.ply")
        ev.append({"stage": "V", "kind": "fuse", "tool": "depth fused into V",
                   "text": f"code: step {sid}'s depth integrated; surface voxels now {len(surf):,}"})
        rec_info = {"step": sid, "surface_rows": len(surf)}
        if st["address"] is None:                                # the survey boots V, no agent
            ev.append({"stage": "V", "kind": "survey", "tool": "survey: V only", "text": "no tree yet; the first inspect creates it"})
            json.dump(ev, open(sd / "events.json", "w"), indent=1)
            info["per_step"].append({**rec_info, "seconds": round(time.time() - t_step, 1)})
            json.dump(info, open(out / "loop.json", "w"), indent=1)
            say(f"step {sid}: survey, V {len(surf):,} rows", flush=True)
            continue
        frame = {"id": f"s{sid}", "run": run, "step": sid}
        mroot, mhow = mask_roots[sid]
        tree = draft_tree(frame, mroot)
        pool = {i: nd["mask"] for i, nd in tree.nodes.items()}
        ev.append({"stage": "segment", "kind": "draft", "tool": "draft tree from masks",
                   "text": f"masks: {mhow} ({mroot.relative_to(C.ROOT)}); object from {tree.object_from}; "
                           f"{len(tree.nodes)} draft nodes", "tree": tree_snap(tree)})
        # ---- segmentation (blind)
        A2.CURRENT["tree"], K.CURRENT["keeper"] = tree, None
        seg_spec = VA.VlmAgent(rules=A2.RULES + SEG_STRUCTURE, kind="inspect",
                               tool_names=(A2._ORIG_TOOLS | A2.IWM_VERBS) - {"detect", "segment"},
                               emit=("evidence", "reasoning", "answer", "tree"), extras={"tree": A2.normalise_tree}, max_turns=24)

        def seg_snap():
            for i, nd in tree.nodes.items():
                pool.setdefault(i, nd["mask"])
            return tree_snap(tree)

        _run, rec = A2.step_record(frame)
        say(f"step {sid}: segmenting ({len(tree.nodes)} draft nodes)", flush=True)
        task = "Build the levelled tree of this view: sides, the things on them, their parts. Name every node you keep, drop the noise."
        fmt_box = fmt_box_of(model, tree.rgb)
        task_full = task + "\n\nThe draft tree (what tree{} returns):\n" + tree.tree_text(fmt_box)
        finding, log, final, prompt, dt = run_agent(seg_spec, run_obj, rec, sd / "seg_store", model, verbs, task_full, seg_snap)
        A2.CURRENT["tree"] = None
        ev.append({"stage": "segment", "kind": "start", "tool": "segmentation agent starts (blind to the IWM)",
                   "text": f"task: {task}", "prompt": prompt, "tree": tree_snap(tree) if not log else None})
        for x in log:
            snap = x.pop("snap")
            ev.append({"stage": "segment", "kind": "tool", "store": "seg_store", **x, "tree": snap})
        for k_, v in (finding.extras.get("tree") or {}).items():
            i = int(k_)
            if i in tree.nodes and not tree.nodes[i]["name"]:
                tree.nodes[i]["name"] = v
        ev.append({"stage": "segment", "kind": "final", "tool": "segmentation final reply", "fields": final,
                   "seconds": dt, "tree": tree_snap(tree)})
        unnamed = [i for i, n in tree.nodes.items() if i != 0 and not n["name"]]
        tree.drop(unnamed)
        tree.nodes[0]["name"] = tree.nodes[0]["name"] or "object"
        for i, nd in tree.nodes.items():
            pool.setdefault(i, nd["mask"])
        ev.append({"stage": "segment", "kind": "result", "tool": "frame tree (unnamed nodes dropped)",
                   "text": f"{len(unnamed)} unnamed nodes dropped, {len(tree.nodes)} kept", "tree": tree_snap(tree)})
        # ---- lift every mask the frame ever had into this step's V
        view = make_view(run, sid, st, intr, scale, tree)
        pool_view = {**view, "masks": {i: (np.ascontiguousarray(m[::-1, ::-1]) if view["rot"] == 180 else m) for i, m in pool.items()}}
        rows_all, _ = L.lift_L3(pool_view, surf, vbg, size)
        fd = sd / "frame"
        fd.mkdir()
        ids = sorted(pool)
        C.save_packed(fd / "masks.npz", [pool[i] for i in ids])
        json.dump(ids, open(fd / "ids.json", "w"))
        cen = [surf.centres(rows_all.get(i, np.empty(0, np.int64))).astype(np.float32) for i in ids]
        np.savez_compressed(fd / "lift.npz", ids=np.array(ids), offsets=np.cumsum([0] + [len(c) for c in cen]),
                            pts=np.concatenate(cen) if cen else np.zeros((0, 3), np.float32))
        rows = {i: rows_all[i] for i in tree.nodes}
        # ---- reconcile
        if stv is None:
            stv = STV(surf)
            dec = first_inspect(stv, view, rows)
            ev.append({"stage": "reconcile", "kind": "init", "tool": "first inspect: frame tree under the object root (code)",
                       "text": "Anton 2026-10-12: T starts with an object root; the first inspect's tree is its child.",
                       "decisions": dec})
        else:
            keeper = K.Keeper(stv, vbg, size, view, rows)
            K.CURRENT["keeper"] = keeper
            say(f"step {sid}: reconciling ({len(keeper.visible)} model parts visible)", flush=True)
            ktask = ("Place every node of this frame's tree in the model: same for parts the model already "
                     "holds, under for new parts, drop for noise.")
            kspec = K.make_spec()
            kspec.rules = K.RULES + KEEP_STRUCTURE
            fmt_box = fmt_box_of(model, keeper.rgb_up)
            ktask_full = (ktask + "\n\nThe new tree (what tree{} returns):\n" + keeper.tree.tree_text(fmt_box)
                          + "\n\nThe model here (what existing{} returns):\n" + keeper.existing(fmt_box))
            kfind, klog, kfinal, kprompt, kdt = run_agent(kspec, run_obj, rec, sd / "keep_store", model, verbs, ktask_full,
                                                          lambda: {f"#{k_}": [str(x) for x in v] for k_, v in keeper.decisions.items()})
            K.CURRENT["keeper"] = None
            ev.append({"stage": "reconcile", "kind": "start", "tool": "reconcile agent starts (sees the IWM projected)",
                       "text": f"task: {ktask}", "prompt": kprompt})
            for x in klog:
                snap = x.pop("snap")
                ev.append({"stage": "reconcile", "kind": "tool", "store": "keep_store", **x, "decisions": snap})
            final_dec = K.normalise_decisions(kfind.extras.get("decisions") or {})
            decisions = K.merge_final(keeper.decisions, final_dec, view, stv)
            ev.append({"stage": "reconcile", "kind": "final", "tool": "reconcile final reply", "fields": kfinal, "seconds": kdt,
                       "decisions": {f"#{k_}": [str(x) for x in v] for k_, v in decisions.items()}})
            decisions, fixes = enforce_structure(decisions, view, stv)
            ev.append({"stage": "reconcile", "kind": "enforce", "tool": "structure rule enforced (code)",
                       "text": ("\n".join(fixes) or "no corrections needed"),
                       "decisions": {f"#{k_}": [str(x) for x in v] for k_, v in decisions.items()}})
            res = K.apply_decisions(stv, view, rows, decisions)
            rec_info["applied"] = res
        stv.check()
        stv.save(sd / "iwm")
        ids_p = sorted(stv.parts)
        cen = [surf.centres(stv.parts[p].rows).astype(np.float32) for p in ids_p]
        lab_rows = np.nonzero(stv.label != -1)[0]
        np.savez_compressed(sd / "iwm" / "centres.npz", ids=np.array(ids_p), offsets=np.cumsum([0] + [len(c) for c in cen]),
                            pts=np.concatenate(cen) if cen else np.zeros((0, 3), np.float32),
                            lab_pts=surf.centres(lab_rows).astype(np.float32), lab=stv.label[lab_rows])
        ev.append({"stage": "commit", "kind": "commit", "tool": "IWM committed",
                   "text": f"parts {len(stv.parts)} · labelled voxels {len(lab_rows):,} · conflicts {len(stv.conflicts)}"})
        json.dump(ev, open(sd / "events.json", "w"), indent=1, default=str)
        info["per_step"].append({**rec_info, "parts": len(stv.parts), "conflicts": len(stv.conflicts),
                                 "seconds": round(time.time() - t_step, 1)})
        json.dump(info, open(out / "loop.json", "w"), indent=1, default=str)
        say(f"step {sid}: committed, {len(stv.parts)} parts, {len(stv.conflicts)} conflicts, "
              f"{time.time() - t_step:.0f}s", flush=True)
        if args.pause:
            wait_feedback(out, sd, sid, info)
    info["finished"] = datetime.now().isoformat(timespec="seconds")
    json.dump(info, open(out / "loop.json", "w"), indent=1, default=str)
    print("LOOP-DONE", out, flush=True)
    if args.pause:                       # keep the viewer up after the last step
        print("viewer stays up - Ctrl+C to quit", flush=True)
        while True:
            time.sleep(10)


if __name__ == "__main__":
    main()
