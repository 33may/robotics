"""e07 loop viewer: step through a REAL loop record (loop.py) capture by capture and turn by turn.

Per event you see the state as it was at that moment, nothing from later:
  3D       V after this step's fusion (grey); the IWM's labelled voxels (before the commit: the state
           the capture started from; at the commit: the new state), coloured per part; the frame
           tree's masks lifted into this step's V, coloured per node (deepest node per voxel, or one
           level); the focused node in white; capture cameras.
  sidebar  navigation; the frame with the current frame tree's masks; the trace event (tool, args,
           full result, the image the tool returned, the final reply, the prompt); the frame tree
           and the IWM tree as node-link diagrams (decisions and change tags next to the nodes).

Feedback (loop.py --pause, Anton 2026-10-15): loop.py starts this viewer in its own process and the
viewer follows it (each new step opens by itself). When the loop pauses, type in the feedback box and
press Enter: the note is saved to step_<id>/feedback.json (tied to the event you are looking at) and
the loop continues (an empty Enter just continues). A note is one line. Notes are only stored.

Run:  /home/anton/miniconda3/envs/robo/bin/python -m inspection.investigation.iwm.e07_update_loop.loop_viewer \
          [--loop inspection/data/iwm/e07_update_loop/loops/<dir>] [--run 2209-box-ray-demo1] [--port 8084]
"""
import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import viser
import viser.transforms as vtf
from PIL import Image

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e04_voxel_scene import build as B
from inspection.investigation.iwm.e07_update_loop import draw as D
from inspection.investigation.iwm.e07_update_loop.loop import OUT

POINT = 0.0014
IMG_W = 600
STAGES = ["capture", "V", "segment", "reconcile", "commit"]
PART_SALT = 3
EMPTY = (np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8))


def small(img, w=IMG_W):
    img = np.asarray(img)
    if img.shape[1] <= w:
        return img
    return np.asarray(Image.fromarray(img).resize((w, int(round(img.shape[0] * w / img.shape[1]))), Image.BILINEAR))


def grey_trimesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    F = np.asarray(mesh.triangles)
    grey = (np.asarray(mesh.vertex_colors) * 255 * 0.45 + 40).clip(0, 255).astype(np.uint8)
    return trimesh.Trimesh(np.asarray(mesh.vertices), np.concatenate([F, F[:, ::-1]]), vertex_colors=grey, process=False)


def intkeys(d):
    return {int(k): v for k, v in d.items()} if d else d


def load_iwm(d):
    if not (d / "parts.json").exists():
        return None
    j = json.load(open(d / "parts.json"))
    z = np.load(d / "centres.npz")
    parts = {p["id"]: p for p in j["parts"]}
    off = z["offsets"]
    pts = {int(i): z["pts"][off[k]:off[k + 1]] for k, i in enumerate(z["ids"])}
    return {"root": j["root"], "parts": parts, "lab_pts": z["lab_pts"], "lab": z["lab"], "pts": pts,
            "conflicts": len(j["conflicts"])}


def load(loop_dir):
    info = json.load(open(loop_dir / "loop.json"))
    run = info["run"]
    steps, _, _ = B.load_steps(run)
    st_of = {s["id"]: s for s in steps}
    blocks, prev_iwm = [], None
    for sd in sorted(loop_dir.glob("step_*")):
        if not (sd / "events.json").exists():
            continue                                            # the step the loop is still on
        sid = int(sd.name.split("_")[1])
        sj = json.load(open(C.RUNS_DIR / run / "steps" / f"{sid:03d}" / "step.json"))
        ev = json.load(open(sd / "events.json"))
        for e in ev:
            for k in ("tree",):
                if e.get(k):
                    e[k] = intkeys(e[k])
        b = {"step": sid, "dir": sd, "events": ev, "T": st_of[sid]["T"], "rgb_raw": st_of[sid]["rgb"],
             "rot": int(sj.get("rgb_rotation_deg", 0) or 0), "address": sj["view"].get("address"),
             "rgb_up": np.array(Image.open(C.RUNS_DIR / run / "steps" / f"{sid:03d}" / "rgb.png").convert("RGB")),
             "mesh": grey_trimesh(sd / "V_display.ply"), "prev": prev_iwm, "masks": {}, "lift": {}}
        fd = sd / "frame"
        if fd.exists():
            ids = json.load(open(fd / "ids.json"))
            b["masks"] = dict(zip(ids, C.load_packed(fd / "masks.npz")))
            z = np.load(fd / "lift.npz")
            b["lift"] = {int(i): z["pts"][z["offsets"][k]:z["offsets"][k + 1]] for k, i in enumerate(z["ids"])}
        cur = load_iwm(sd / "iwm")
        b["cur"] = cur if cur is not None else prev_iwm
        prev_iwm = b["cur"]
        blocks.append(b)
    return info, blocks


def read_feedback(sd):
    try:
        return json.load(open(sd / "feedback.json"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"notes": [], "done": False}


def write_feedback(sd, fb):
    tmp = sd / "feedback.json.tmp"
    json.dump(fb, open(tmp, "w"), indent=1, ensure_ascii=False)
    tmp.replace(sd / "feedback.json")                          # atomic: the loop never reads half a file


def n_done(loop_dir):
    return sum(1 for sd in loop_dir.glob("step_*") if (sd / "events.json").exists())


def depth_of(tree, i):
    d = 0
    while tree[i]["parent"] is not None and tree[i]["parent"] in tree:
        i = tree[i]["parent"]
        d += 1
    return d


def frame_tree_of(block, k):
    """The frame tree the event at k sees: its own snapshot, else the latest one before it."""
    for e in reversed(block["events"][:k + 1]):
        if e.get("tree"):
            return e["tree"]
    return None


def decisions_of(block, k):
    for e in reversed(block["events"][:k + 1]):
        if e.get("decisions"):
            return e["decisions"]
    return None


def event_md(e, k, n):
    head = f"**{k + 1}/{n} · {e['stage']} · {e['tool']}**"
    if e["kind"] == "tool":
        args = json.dumps(e.get("args", {}), default=str)
        return f"{head}  \nargs `{args}`\n\n```\n{str(e.get('result', ''))[:5000]}\n```"
    if e["kind"] == "final":
        f = e.get("fields", {})
        parts = [head + (f" · {e.get('seconds')} s" if e.get("seconds") else "")]
        for key, v in f.items():
            parts.append(f"**{key}**\n\n```\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)}\n```")
        return "\n\n".join(parts)
    return f"{head}\n\n{e.get('text', '')}"


def serve(loop_dir, port=8084):
    """Build the viewer on loop_dir and return (the server runs in its own threads)."""
    loop_dir = Path(loop_dir)
    while n_done(loop_dir) == 0:                               # the loop is still making masks / on its first step
        print(f"waiting for the first step in {loop_dir} ...", flush=True)
        time.sleep(5)
    info, blocks = load(loop_dir)
    run = info["run"]
    steps, intr, _ = B.load_steps(run)
    fov = 2 * np.arctan(intr["height"] / 2 / intr["fy"])
    aspect = intr["width"] / intr["height"]

    server = viser.ViserServer(port=port)
    server.gui.configure_theme(dark_mode=True, control_width="large", show_logo=False)
    server.scene.set_up_direction("+z")

    with server.gui.add_folder("Navigate"):
        pos_md = server.gui.add_markdown("")
        cap_bg = server.gui.add_button_group("capture", ["◀ prev capture", "next capture ▶"])
        ev_bg = server.gui.add_button_group("event", ["◀ stage", "◀", "▶", "stage ▶"])
        ev_sl = server.gui.add_slider("event", 0, max(1, len(blocks[0]["events"]) - 1), 1, 0)
        reload_bt = server.gui.add_button("reload record (loop still running)")
    with server.gui.add_folder("Feedback"):
        fb_status = server.gui.add_markdown("")
        fb_text = server.gui.add_text("feedback", "", multiline=True,
                                      hint="one line; Enter = save + continue (empty Enter = just continue)")
        follow_cb = server.gui.add_checkbox("follow the loop (jump to each new step)", initial_value=True)
        live_md = server.gui.add_markdown("")
        fb_md = server.gui.add_markdown("")
    with server.gui.add_folder("Frame + current masks"):
        frame_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8))
        frame_md = server.gui.add_markdown("")
    with server.gui.add_folder("Agent trace"):
        ev_md = server.gui.add_markdown("")
        ev_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8), visible=False)
        with server.gui.add_folder("prompt of this stage", expand_by_default=False):
            prompt_md = server.gui.add_markdown("")
    with server.gui.add_folder("Frame tree"):
        ftree_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8))
        focus_dd = server.gui.add_dropdown("focus node", ["-"], initial_value="-")
    with server.gui.add_folder("IWM tree"):
        itree_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8))
        itree_md = server.gui.add_markdown("")
    with server.gui.add_folder("3D layers"):
        mesh_cb = server.gui.add_checkbox("V mesh", initial_value=True)
        iwm_cb = server.gui.add_checkbox("IWM voxels (per part)", initial_value=True)
        fr_cb = server.gui.add_checkbox("frame masks in 3D (per node)", initial_value=True)
        level_dd = server.gui.add_dropdown("frame tree level", ["deepest", "1", "2", "3", "4", "5"], initial_value="deepest")
        look_bt = server.gui.add_button("look from this capture")

    S = {"b": 0, "e": 0, "busy": False, "mesh_b": None, "mesh_h": None, "blocks": blocks}
    lock = threading.RLock()

    def waiting():
        try:
            return json.load(open(loop_dir / "loop.json")).get("waiting")
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def refresh_feedback():
        b = B_()
        fb = read_feedback(b["dir"])
        w = waiting()
        if w is None:
            fb_status.content = "loop **running** (or finished) · Enter saves a note to the step you are on"
        else:
            fb_status.content = (f"### ⏸ PAUSED after step {w}\ntype feedback, **Enter** = save + continue"
                                 + ("" if w == b["step"] else f" (you are looking at step {b['step']})"))
        lines = [f"- **ev {n['event']}** · {n['stage']} · {n['tool']}"
                 + (f" · focus {n['focus']}" if n.get("focus") else "") + f"  \n  {n['text']}" for n in fb["notes"]]
        fb_md.content = (f"**notes on step {b['step']}** ({len(lines)})" + (" · done" if fb.get("done") else "")
                         + "\n\n" + "\n".join(lines)) if lines else f"_no notes on step {b['step']} yet_"

    def B_():
        return S["blocks"][S["b"]]

    def iwm_state():
        b = B_()
        e = b["events"][S["e"]]
        if e["stage"] == "commit":
            return b["cur"], f"IWM after step {b['step']}", True
        return b["prev"], (f"IWM before step {b['step']}" if b["prev"] else "IWM empty (no tree yet)"), False

    def draw_3d(tree, focus):
        b = B_()
        if S["mesh_b"] != S["b"]:
            S["mesh_b"] = S["b"]
            S["mesh_h"] = server.scene.add_mesh_trimesh("/V", b["mesh"], cast_shadow=False, receive_shadow=False,
                                                        visible=mesh_cb.value)
        iwm, _, _ = iwm_state()
        if iwm is not None and iwm_cb.value and len(iwm["lab"]):
            cols = np.array([D.colour(int(l), PART_SALT) for l in iwm["lab"]], np.uint8)
            server.scene.add_point_cloud("/iwm", iwm["lab_pts"], cols, point_size=POINT, point_shape="rounded")
        else:
            server.scene.add_point_cloud("/iwm", *EMPTY, point_size=1e-6)
        if tree and fr_cb.value and b["lift"]:
            lv = level_dd.value
            ids = sorted((i for i in tree if i in b["lift"]), key=lambda i: depth_of(tree, i))
            if lv != "deepest":
                ids = [i for i in ids if depth_of(tree, i) == int(lv)]
            key_node = {}
            pts_all = {}
            for i in ids:                                       # parent first: deeper nodes overwrite
                P = b["lift"][i]
                if not len(P):
                    continue
                keys = np.round(P / 0.0005).astype(np.int64)
                for kk, p in zip(map(tuple, keys), P):
                    key_node[kk] = i
                    pts_all[kk] = p
            if key_node:
                P = np.array(list(pts_all.values()), np.float32)
                cols = np.array([D.colour(key_node[k]) for k in pts_all], np.uint8)
                server.scene.add_point_cloud("/frame", P + np.float32([0, 0, 0.0005]), cols, point_size=POINT * 1.3)
            else:
                server.scene.add_point_cloud("/frame", *EMPTY, point_size=1e-6)
        else:
            server.scene.add_point_cloud("/frame", *EMPTY, point_size=1e-6)
        if focus is not None and focus in b["lift"] and len(b["lift"][focus]):
            P = b["lift"][focus]
            server.scene.add_point_cloud("/focus", P + np.float32([0, 0, 0.001]),
                                         np.tile(np.array([[255, 255, 255]], np.uint8), (len(P), 1)), point_size=POINT * 1.8)
        else:
            server.scene.add_point_cloud("/focus", *EMPTY, point_size=1e-6)
        for j, bl in enumerate(S["blocks"]):
            cur = j == S["b"]
            server.scene.add_camera_frustum(f"/cams/{bl['step']}", fov=fov, aspect=aspect, scale=0.05 if cur else 0.03,
                                            wxyz=vtf.SO3.from_matrix(bl["T"][:3, :3]).wxyz, position=bl["T"][:3, 3],
                                            image=bl["rgb_raw"][::6, ::6] if cur else None,
                                            color=(255, 200, 0) if cur else (140, 140, 140), visible=j <= S["b"])

    def refresh():
        b = B_()
        k = S["e"]
        evs = b["events"]
        e = evs[k]
        pos_md.content = (f"**{run}** · loop `{loop_dir.name}` · capture **{S['b'] + 1}/{len(S['blocks'])}** step {b['step']} "
                          f"({'survey' if b['address'] is None else b['address']}) · event **{k + 1}/{len(evs)}**  \n"
                          + " → ".join(f"**{s}**" if s == e["stage"] else s for s in STAGES))
        tree = frame_tree_of(b, k) if e["stage"] in ("segment", "reconcile", "commit") else None
        focus = None if focus_dd.value == "-" else int(focus_dd.value.split()[0][1:])
        if tree is not None:
            opts = ["-"] + [f"#{i} {tree[i]['name'] or '?'}" for i in sorted(tree, key=lambda i: (depth_of(tree, i), i))]
            if list(focus_dd.options) != opts:
                focus_dd.options = opts
                if focus is not None and focus not in tree:
                    focus_dd.value, focus = "-", None
            ms = {i: b["masks"][i] for i in sorted(tree, key=lambda i: depth_of(tree, i)) if i in b["masks"] and i != 0}
            frame_img.image = small(D.overlay(b["rgb_up"], ms, focus))
            frame_md.content = f"frame tree: {len(tree)} nodes (masks coloured per node, parent first; white = focus)"
        else:
            focus_dd.options = ["-"]
            frame_img.image = small(b["rgb_up"])
            frame_md.content = f"step {b['step']} · camera z {b['T'][2, 3]:.3f} m · rgb rotation {b['rot']}"
        ev_md.content = event_md(e, k, len(evs))
        img = e.get("image")
        p = (b["dir"] / e.get("store", "") / img) if img else None
        if p is not None and p.exists():
            ev_img.image = small(np.asarray(Image.open(p).convert("RGB")))
            ev_img.visible = True
        else:
            ev_img.visible = False
        pr = next((x.get("prompt") for x in evs if x["stage"] == e["stage"] and x.get("prompt")), None)
        prompt_md.content = "```\n" + pr[:8000] + "\n```" if pr else "_no prompt at this stage_"
        if tree is not None:
            dec = decisions_of(b, k) if e["stage"] in ("reconcile", "commit") else None
            tag = (lambda i: " ".join(dec.get(f"#{i}", [])) or None) if dec else (lambda i: tree[i]["source"][:14] if not tree[i]["name"] else None)
            ftree_img.image = D.tree_image(tree, 0 if 0 in tree else min(tree), lambda i: f"#{i} {tree[i]['name'] or '?'}"
                                           + (f" \"{tree[i]['text']}\"" if tree[i].get("text") else ""), tag=tag, focus=focus,
                                           title=f"frame tree · {e['tool']}")
        else:
            ftree_img.image = np.full((40, 300, 3), 22, np.uint8)
        iwm, label, after = iwm_state()
        if iwm is not None:
            parts = iwm["parts"]
            sname = f"s{b['step']}"
            prev_ids = set(b["prev"]["parts"]) if b["prev"] else set()

            def itag(p):
                if not after:
                    return None
                if p not in prev_ids:
                    return "NEW"
                if any(d[0] == sname for d in parts[p]["detections"]):
                    return "+view"
                return None
            itree_img.image = D.tree_image(parts, iwm["root"], lambda p: f"P{p} {parts[p]['name']} ({parts[p]['n_rows']})",
                                           tag=itag, salt=PART_SALT, title=label)
            itree_md.content = f"{label} · {len(parts)} parts · {len(iwm['lab']):,} labelled voxels · {iwm['conflicts']} conflicts"
        else:
            itree_img.image = np.full((40, 300, 3), 22, np.uint8)
            itree_md.content = f"**{label}**"
        draw_3d(tree, focus)
        refresh_feedback()
        if ev_sl.value != k:
            S["busy"] = True
            ev_sl.value = k
            S["busy"] = False

    def go(bi, k):
        bi = int(np.clip(bi, 0, len(S["blocks"]) - 1))
        if bi != S["b"]:
            ev_sl.max = max(1, len(S["blocks"][bi]["events"]) - 1)
        S["b"] = bi
        S["e"] = int(np.clip(k, 0, len(S["blocks"][bi]["events"]) - 1))
        refresh()

    def stage_jump(direction):
        evs = B_()["events"]
        cur = evs[S["e"]]["stage"]
        rng = range(S["e"] + 1, len(evs)) if direction > 0 else range(S["e"] - 1, -1, -1)
        for i in rng:
            if evs[i]["stage"] != cur:
                if direction < 0:
                    s = evs[i]["stage"]
                    while i > 0 and evs[i - 1]["stage"] == s:
                        i -= 1
                go(S["b"], i)
                return
        go(S["b"] + direction, 0)

    @cap_bg.on_click
    def _(_):
        go(S["b"] + (1 if "next" in cap_bg.value else -1), 0)

    @ev_bg.on_click
    def _(_):
        v = ev_bg.value
        n = len(B_()["events"])
        if v == "◀":
            go(S["b"] - 1, 10 ** 6) if S["e"] == 0 and S["b"] > 0 else go(S["b"], S["e"] - 1)
        elif v == "▶":
            go(S["b"] + 1, 0) if S["e"] == n - 1 and S["b"] < len(S["blocks"]) - 1 else go(S["b"], S["e"] + 1)
        elif v == "◀ stage":
            stage_jump(-1)
        else:
            stage_jump(+1)

    @ev_sl.on_update
    def _(_):
        if not S["busy"] and int(ev_sl.value) != S["e"]:
            go(S["b"], int(ev_sl.value))

    for h in (focus_dd, iwm_cb, fr_cb, level_dd):
        h.on_update(lambda _: refresh())

    @mesh_cb.on_update
    def _(_):
        if S["mesh_h"] is not None:
            S["mesh_h"].visible = mesh_cb.value

    def reload(jump=False):
        with lock:
            _, bl = load(loop_dir)
            S["blocks"] = bl
            S["mesh_b"] = None
            if jump:
                ev_sl.max = max(1, len(bl[-1]["events"]) - 1)
                go(len(bl) - 1, 0)
            else:
                go(S["b"], S["e"])

    @reload_bt.on_click
    def _(_):
        reload()

    def save_note(sd, b, k, txt):
        e = b["events"][k] if b is not None else {}
        fb = read_feedback(sd)
        fb["step"] = int(sd.name.split("_")[1])
        if txt:
            fb["notes"].append({"event": k + 1 if b is not None else None, "stage": e.get("stage"), "tool": e.get("tool"),
                                "focus": None if focus_dd.value == "-" else focus_dd.value, "text": txt,
                                "at": datetime.now().isoformat(timespec="seconds")})
        return fb

    @fb_text.on_update
    def _(_):
        v = fb_text.value
        if not v.endswith("\n"):
            return                                              # still typing (each keystroke arrives here)
        txt = v.strip()
        with lock:
            fb_text.value = ""
            b, k, w = B_(), S["e"], waiting()
            if w is not None:                                   # paused: the note belongs to the paused step
                sd = loop_dir / f"step_{w:03d}"
                fb = save_note(sd, b if b["step"] == w else None, k, txt)
                fb.update(done=True, done_at=datetime.now().isoformat(timespec="seconds"))
                write_feedback(sd, fb)
            elif txt:
                write_feedback(b["dir"], save_note(b["dir"], b, k, txt))
            refresh_feedback()

    def live():
        try:
            lines = json.load(open(loop_dir / "live.json"))["lines"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return
        txt = "**live** (the loop right now)\n```\n" + "\n".join(l[:110] for l in lines) + "\n```"
        if live_md.content != txt:
            live_md.content = txt

    def follow():
        seen, last_w = n_done(loop_dir), waiting()
        while True:
            time.sleep(2)
            live()
            n, w = n_done(loop_dir), waiting()
            if n != seen:
                seen = n
                reload(jump=follow_cb.value)
            elif w != last_w:
                with lock:
                    if w is not None and follow_cb.value:       # paused: show the paused step
                        bi = next((j for j, bl in enumerate(S["blocks"]) if bl["step"] == w), None)
                        if bi is not None and bi != S["b"]:
                            ev_sl.max = max(1, len(S["blocks"][bi]["events"]) - 1)
                            go(bi, 0)
                    refresh_feedback()
            last_w = w

    threading.Thread(target=follow, daemon=True).start()

    @look_bt.on_click
    def _(event):
        if event.client is None:
            return
        b = B_()
        T = b["T"].copy()
        if b["rot"] == 180:
            T[:3, :3] = T[:3, :3] @ np.diag([-1, -1, 1])
        R = T[:3, :3]
        with event.client.atomic():
            event.client.camera.fov = fov
            event.client.camera.position = T[:3, 3]
            event.client.camera.wxyz = vtf.SO3.from_matrix(R).wxyz
            event.client.camera.look_at = T[:3, 3] + R[:, 2] * 0.2
            event.client.camera.up_direction = -R[:, 1]

    go(len(blocks) - 1, 0)                                      # the newest step (the paused one in a live run)
    print(f"e07 loop viewer on http://localhost:{port}  {loop_dir}  {len(blocks)} steps", flush=True)
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop")
    ap.add_argument("--run", default="2209-box-ray-demo1")
    ap.add_argument("--port", type=int, default=8084)
    args = ap.parse_args()
    serve(Path(args.loop) if args.loop else sorted((OUT / "loops").glob(f"{args.run}_*"))[-1], args.port)
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
