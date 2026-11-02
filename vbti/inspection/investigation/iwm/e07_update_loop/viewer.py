"""e07 review viewer: step through a run's captures and each capture's inspect block, to see what
the agent saw, did and decided, and what the IWM looked like before and after.

Left (3D): V's mesh in grey, the IWM's labelled voxels coloured by their deepest part (the state
the current event sees: before the capture, or after its commit), the capture cameras (current one
yellow, carrying its image), changed parts marked white at the commit.
Sidebar: navigation (capture / event / stage), the incoming frame, the agent trace event (tool,
args, result, the image the tool returned, the final reply's evidence and reasoning), the IWM tree
with this capture's change tags, a part highlighter.

Run:  /home/anton/miniconda3/envs/robo/bin/python -m inspection.investigation.iwm.e07_update_loop.viewer \
          [--run 2209-compute-ray-demo5] [--mm 1] [--port 8084]
"""
import argparse
import json
import time

import numpy as np
import open3d as o3d
import trimesh
import viser
import viser.transforms as vtf
from PIL import Image

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e04_voxel_scene.build import OUT as E04_OUT
from inspection.investigation.iwm.e05_parts_in_v import common as C5
from inspection.investigation.iwm.e05_parts_in_v import lift as L
from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface
from inspection.investigation.iwm.e07_update_loop import grow as G
from inspection.investigation.iwm.e07_update_loop import trace as Tr

POINT = 0.0014
IMG_W = 560
STAGES = ["capture", "V", "segment", "reconcile", "commit"]


def small(img, w=IMG_W):
    img = np.asarray(img)
    if img.shape[1] <= w:
        return img
    h = int(round(img.shape[0] * w / img.shape[1]))
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def grey_trimesh(mesh, display_m=0.002):
    """Display copy: vertex clustering at 2 mm (the full 1 mm surface is 1-4 M triangles per step)."""
    mesh = mesh.simplify_vertex_clustering(voxel_size=display_m)
    F = np.asarray(mesh.triangles)
    grey = (np.asarray(mesh.vertex_colors) * 255 * 0.45 + 40).clip(0, 255).astype(np.uint8)
    return trimesh.Trimesh(np.asarray(mesh.vertices), np.concatenate([F, F[:, ::-1]]), vertex_colors=grey, process=False)


def load(run, mm):
    """One block per captured step, in capture order; V grows step by step."""
    vbg, size = C5.V(run, mm)
    surf = Surface.from_vbg(vbg, size, C5.trunc_m(size))
    steps, intr, scale = C5.run_steps(run)
    meshes = G.grow(run, mm, steps, intr, scale)
    fids = C5.ORDER[run]
    fid_of = {C5.FRAMES[f][1]: f for f in fids}
    keep = C5.DATA / run / f"{mm}mm" / "keep_agent"
    blocks, state = [], None
    for st in steps:
        sd = C.RUNS_DIR / run / "steps" / f"{st['id']:03d}"
        sj = json.load(open(sd / "step.json"))
        rot = int(sj.get("rgb_rotation_deg", 0) or 0)
        fid = fid_of.get(st["id"])
        prev = state
        if fid is not None:
            state = STV.load(keep / f"after_{fid}", surf)
            ev, ch, meta = Tr.block(run, fid, mm, prev, state)
        else:
            ev, ch, meta = Tr.plain_block(run, st["id"], st["address"], fids), None, {}
        meta["step"] = st["id"]
        blocks.append({"step": st["id"], "fid": fid, "events": ev, "changes": ch, "meta": meta, "prev": prev,
                       "cur": state, "T": st["T"], "rot": rot, "rgb_raw": st["rgb"],
                       "rgb_up": np.array(Image.open(sd / "rgb.png").convert("RGB")),
                       "mesh": grey_trimesh(meshes[st["id"]])})
    ids = sorted(set().union(*[set(b["cur"].parts) for b in blocks if b["cur"] is not None]))
    return surf, blocks, intr, ids


def event_md(e, k, n):
    head = f"**{k + 1}/{n} · {e['stage']} · {e['tool']}**"
    if e["kind"] == "tool":
        return f"{head}  \nargs `{e['args']}`\n\n```\n{e['result']}\n```"
    if e["kind"] == "final":
        f = e["fields"]
        parts = [head]
        for key in ("evidence", "reasoning", "answer", "decisions", "tree"):
            if key in f:
                parts.append(f"**{key}**\n\n```\n{f[key]}\n```")
        rest = [k2 for k2 in f if k2 not in ("evidence", "reasoning", "answer", "decisions", "tree")]
        parts += [f"**{k2}** {f[k2]}" for k2 in rest]
        return "\n\n".join(parts)
    return f"{head}\n\n{e.get('text', '')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2209-compute-ray-demo5")
    ap.add_argument("--mm", type=int, default=1)
    ap.add_argument("--port", type=int, default=8084)
    args = ap.parse_args()

    surf, blocks, intr, ids = load(args.run, args.mm)
    pal_l = C._palette(len(ids), seed=3)
    pal = {p: np.array(pal_l[i], np.uint8) for i, p in enumerate(ids)}
    fov = 2 * np.arctan(intr["height"] / 2 / intr["fy"])
    aspect = intr["width"] / intr["height"]

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(dark_mode=True, control_layout="fixed", control_width="large", show_logo=False)
    server.scene.set_up_direction("+z")
    mesh_h = {"h": None, "b": None}

    with server.gui.add_folder("Navigate"):
        pos_md = server.gui.add_markdown("")
        cap_bg = server.gui.add_button_group("capture", ["◀ prev capture", "next capture ▶"])
        ev_bg = server.gui.add_button_group("event", ["◀ stage", "◀", "▶", "stage ▶"])
        ev_sl = server.gui.add_slider("event", 0, max(1, len(blocks[0]["events"]) - 1), 1, 0)
    with server.gui.add_folder("Incoming frame"):
        frame_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8))
        frame_md = server.gui.add_markdown("")
    with server.gui.add_folder("Agent trace"):
        ev_md = server.gui.add_markdown("")
        ev_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8), visible=False)
        with server.gui.add_folder("prompt of this stage", expand_by_default=False):
            prompt_md = server.gui.add_markdown("")
    with server.gui.add_folder("IWM tree"):
        tree_md = server.gui.add_markdown("")
        part_dd = server.gui.add_dropdown("highlight part", ["-"], initial_value="-")
    with server.gui.add_folder("3D"):
        mesh_cb = server.gui.add_checkbox("V mesh", initial_value=True)
        mark_cb = server.gui.add_checkbox("mark changed parts at commit", initial_value=True)
        look_bt = server.gui.add_button("look from this capture")

    st = {"b": 0, "e": 0, "shown": None, "busy": False}

    def shown_state():
        """(stv or None, label) the current event sees."""
        bl = blocks[st["b"]]
        e = bl["events"][st["e"]]
        name = bl["fid"] or f"step {bl['step']}"
        if e["state"] == "after":
            return bl["cur"], f"IWM after {name}"
        if bl["prev"] is None:
            return None, "IWM empty (no tree yet)"
        return bl["prev"], f"IWM before {name}"

    def draw_mesh():
        if mesh_h["b"] == st["b"]:
            return
        mesh_h["b"] = st["b"]
        mesh_h["h"] = server.scene.add_mesh_trimesh("/V", blocks[st["b"]]["mesh"], cast_shadow=False, receive_shadow=False,
                                                    visible=mesh_cb.value)

    def draw_points(stv, key):
        if st["shown"] == key:
            return
        st["shown"] = key
        if stv is None:
            server.scene.add_point_cloud("/iwm", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), point_size=1e-6)
            return
        rows = np.nonzero(stv.label != -1)[0]
        lab = stv.label[rows]
        cols = np.stack([pal[int(l)] for l in lab]) if len(lab) else np.zeros((0, 3), np.uint8)
        server.scene.add_point_cloud("/iwm", surf.centres(rows).astype(np.float32), cols, point_size=POINT, point_shape="rounded")

    def draw_marks(stv, e):
        ch = e.get("changes")
        if stv is None or ch is None or not mark_cb.value:
            server.scene.add_point_cloud("/marks", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), point_size=1e-6)
            return
        parts = [p for p in ch["new"] + ch["extended"] if p != stv.root]
        if not parts:
            server.scene.add_point_cloud("/marks", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), point_size=1e-6)
            return
        rows = np.unique(np.concatenate([stv.parts[p].rows for p in parts]))
        server.scene.add_point_cloud("/marks", surf.centres(rows).astype(np.float32),
                                     np.tile(np.array([[255, 255, 255]], np.uint8), (len(rows), 1)), point_size=POINT * 1.5)

    def draw_cams():
        for i, bl in enumerate(blocks):
            vis = i <= st["b"]
            cur = i == st["b"]
            server.scene.add_camera_frustum(f"/cams/{bl['step']}", fov=fov, aspect=aspect, scale=0.05 if cur else 0.03,
                                            wxyz=vtf.SO3.from_matrix(bl["T"][:3, :3]).wxyz, position=bl["T"][:3, 3],
                                            image=bl["rgb_raw"][::6, ::6] if cur else None,
                                            color=(255, 200, 0) if cur else (140, 140, 140), visible=vis)

    def refresh():
        b, k = st["b"], st["e"]
        bl = blocks[b]
        evs = bl["events"]
        e = evs[k]
        stv, label = shown_state()
        pos_md.content = (f"**{args.run}** · capture **{b + 1}/{len(blocks)}** step {bl['step']}"
                          + (f" = frame **{bl['fid']}**" if bl["fid"] else " (no agent trace)") + " · "
                          f"event **{k + 1}/{len(evs)}** · stage **{e['stage']}**  \n"
                          + " → ".join(f"**{s}**" if s == e["stage"] else s for s in STAGES))
        frame_img.image = small(bl["rgb_up"])
        frame_md.content = (f"step {bl['step']}" + (f" · frame {bl['fid']}" if bl["fid"] else "")
                            + f" · camera z {bl['T'][2, 3]:.3f} m · rgb rotation {bl['rot']}")
        ev_md.content = event_md(e, k, len(evs))
        if e.get("image") is not None:
            ev_img.image = small(np.asarray(Image.open(e["image"]).convert("RGB")))
            ev_img.visible = True
        else:
            ev_img.visible = False
        if e["stage"] == "segment":
            prompt_md.content = "```\n" + (bl["meta"].get("seg_prompt") or "(not logged)")[:6000] + "\n```"
        elif e.get("prompt"):
            prompt_md.content = "```\n" + e["prompt"][:6000] + "\n```"
        elif e["stage"] == "reconcile":
            first = next((x for x in evs if x.get("prompt")), None)
            prompt_md.content = "```\n" + (first["prompt"][:6000] if first else "(no agent at this stage)") + "\n```"
        else:
            prompt_md.content = "_no prompt at this stage_"
        changes = e.get("changes") if e["state"] == "after" else None
        if stv is None:
            tree_md.content = f"**{label}**"
            part_dd.options = ["-"]
        else:
            tree_md.content = f"**{label}**\n\n```\n" + "\n".join(Tr.tree_lines(stv, changes)) + "\n```"
            opts = ["-"] + [f"P{p} {stv.parts[p].name}" for p in sorted(stv.parts)]
            if list(part_dd.options) != opts:
                part_dd.options = opts
                part_dd.value = "-"
        draw_mesh()
        draw_points(stv, label)
        draw_marks(stv, e)
        draw_cams()
        if ev_sl.value != k:
            st["busy"] = True
            ev_sl.value = k
            st["busy"] = False

    def go(b, k):
        b = int(np.clip(b, 0, len(blocks) - 1))
        if b != st["b"]:
            ev_sl.max = max(1, len(blocks[b]["events"]) - 1)
        st["b"] = b
        st["e"] = int(np.clip(k, 0, len(blocks[b]["events"]) - 1))
        refresh()

    def stage_jump(direction):
        evs = blocks[st["b"]]["events"]
        cur = evs[st["e"]]["stage"]
        rng = range(st["e"] + 1, len(evs)) if direction > 0 else range(st["e"] - 1, -1, -1)
        for i in rng:
            if evs[i]["stage"] != cur:
                if direction < 0:          # to the first event of that earlier stage
                    s = evs[i]["stage"]
                    while i > 0 and evs[i - 1]["stage"] == s:
                        i -= 1
                go(st["b"], i)
                return
        if direction > 0 and st["b"] < len(blocks) - 1:
            go(st["b"] + 1, 0)
        elif direction < 0 and st["b"] > 0:
            go(st["b"] - 1, 0)

    @cap_bg.on_click
    def _(event):
        go(st["b"] + (1 if "next" in cap_bg.value else -1), 0)

    @ev_bg.on_click
    def _(event):
        v = ev_bg.value
        if v == "◀":
            if st["e"] == 0 and st["b"] > 0:
                go(st["b"] - 1, len(blocks[st["b"] - 1]["events"]) - 1)
            else:
                go(st["b"], st["e"] - 1)
        elif v == "▶":
            if st["e"] == len(blocks[st["b"]]["events"]) - 1 and st["b"] < len(blocks) - 1:
                go(st["b"] + 1, 0)
            else:
                go(st["b"], st["e"] + 1)
        elif v == "◀ stage":
            stage_jump(-1)
        else:
            stage_jump(+1)

    @ev_sl.on_update
    def _(_):
        if not st["busy"] and int(ev_sl.value) != st["e"]:
            go(st["b"], int(ev_sl.value))

    @part_dd.on_update
    def _(_):
        stv, _l = shown_state()
        if part_dd.value == "-" or stv is None:
            server.scene.add_point_cloud("/selected", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), point_size=1e-6)
            return
        p = int(part_dd.value.split()[0][1:])
        if p not in stv.parts:
            return
        rows = stv.parts[p].rows
        server.scene.add_point_cloud("/selected", surf.centres(rows).astype(np.float32),
                                     np.tile(np.array([[255, 0, 255]], np.uint8), (len(rows), 1)), point_size=POINT * 1.8)

    @mesh_cb.on_update
    def _(_):
        if mesh_h["h"] is not None:
            mesh_h["h"].visible = mesh_cb.value

    @mark_cb.on_update
    def _(_):
        refresh()

    @look_bt.on_click
    def _(event):
        if event.client is None:
            return
        bl = blocks[st["b"]]
        T = bl["T"].copy()
        if bl["rot"] == 180:
            T[:3, :3] = T[:3, :3] @ np.diag([-1, -1, 1])
        R = T[:3, :3]
        with event.client.atomic():
            event.client.camera.fov = fov
            event.client.camera.position = T[:3, 3]
            event.client.camera.wxyz = vtf.SO3.from_matrix(R).wxyz
            event.client.camera.look_at = T[:3, 3] + R[:, 2] * 0.2
            event.client.camera.up_direction = -R[:, 1]

    go(0, 0)
    n_ev = sum(len(b["events"]) for b in blocks)
    print(f"e07 viewer on http://localhost:{args.port}  run {args.run}  {args.mm} mm  "
          f"{len(blocks)} captures, {n_ev} events", flush=True)
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
