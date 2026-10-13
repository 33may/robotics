"""e02: the IWM tree state and the verbs the vision agent gets over it. Throwaway.

An IwmTree is built from e01's mask sets (UnSAMv2 ladder + SAM 3 text) with the e01
containment rule, then two judgement-free pre-filters. The agent then edits it with
name / group / drop / find and looks at it with tree / show.
"""
import json
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e01_all_masks import tree as T

DEFAULT_SETS = ["unsam_plus_g0.9", "unsam_plus_g0.7", "unsam_plus_g0.5",
                "unsam_plus_g0.3", "unsam_plus_g0.2", "unsam_plus_g0.1", "sam3_text"]
COLORS = T.DEPTH_COLORS


class IwmTree:
    """Nodes: id -> {parent, children, name, text, area, bbox, source, mask}. 0 = object."""

    def __init__(self, rgb):
        self.rgb = rgb
        self.nodes = {}
        self.log = []                       # every edit, for the report
        self._next = 0

    # ------------------------------------------------------------ building
    @classmethod
    def from_e01(cls, frame, sets=DEFAULT_SETS, contain=0.9):
        rgb = C.load_rgb(frame)
        masks, recs = [], []
        for s in sets:
            d = C.DATA / "masks" / frame["id"] / s
            if not (d / "masks.json").exists():
                continue
            arr, meta = C.load_masks(d)
            g = meta["settings"].get("granularity")
            for m, r in zip(arr, meta["masks"]):
                masks.append(m)
                rr = {**r, "source": s}
                if g is not None:
                    rr["g"] = g
                recs.append(rr)
        obj, how = C.recorded_object(frame)
        if obj is None:
            coarse = C.DATA / "masks" / frame["id"] / "unsam_plus_g0.9"
            arr, _ = C.load_masks(coarse)
            areas = [m.sum() for m in arr]
            lim = 0.8 * rgb.shape[0] * rgb.shape[1]
            big = max(range(len(arr)), key=lambda i: areas[i] if areas[i] < lim else -1)
            obj, how = arr[big], "largest mask of unsam_plus_g0.9"
        ms, nodes, conflicts = T.build(masks, recs, obj, contain=contain,
                                       dedupe_iou=0.8, ratio=1.25)
        t = cls(rgb)
        for nd, m in zip(nodes, ms):
            t.nodes[nd["id"]] = {"parent": nd["parent"] if nd["id"] else None,
                                 "children": [], "name": None, "text": None,
                                 "area": int(nd["area"]), "bbox": nd["bbox"],
                                 "source": nd.get("phrase") or nd.get("source", ""),
                                 "mask": m}
        for i, nd in t.nodes.items():
            if nd["parent"] is not None:
                t.nodes[nd["parent"]]["children"].append(i)
        t._next = max(t.nodes) + 1
        t.object_from = how
        t.conflicts = conflicts
        t.collapse_single_children()
        return t

    def collapse_single_children(self, frac=0.9):
        """A node covering >= frac of its parent is the same region twice: keep the
        parent, hand the node's children up. (The first c5.6 run had #1 = 99 % of
        #0 holding both faces; the agent named it 'front face' and dropped the faces.)"""
        changed = True
        n = 0
        while changed:
            changed = False
            for i, nd in list(self.nodes.items()):
                if i == 0 or i not in self.nodes:
                    continue
                p = nd["parent"]
                if nd["area"] >= frac * self.nodes[p]["area"]:
                    self._remove(i, keep_children=True)
                    n += 1
                    changed = True
                    break
        if n:
            self.log.append({"op": "collapse", "n": n})
        return n

    # ------------------------------------------------------------ helpers
    def depth(self, i):
        d, p = 0, self.nodes[i]["parent"]
        while p is not None:
            d, p = d + 1, self.nodes[p]["parent"]
        return d

    def max_depth(self):
        return max(self.depth(i) for i in self.nodes)

    def _remove(self, i, keep_children=True):
        nd = self.nodes.pop(i)
        p = nd["parent"]
        if p is not None and i in self.nodes[p]["children"]:
            self.nodes[p]["children"].remove(i)
        for c in nd["children"]:
            if keep_children:
                self.nodes[c]["parent"] = p
                if p is not None:
                    self.nodes[p]["children"].append(c)
            else:
                self._remove(c, keep_children=False)

    def _contain_frac(self, a, b):
        """|a & b| / |b| on 2x downsampled masks."""
        A, B = a[::2, ::2], b[::2, ::2]
        bs = B.sum()
        return 0.0 if bs == 0 else float((A & B).sum()) / bs

    def _iou(self, a, b):
        A, B = a[::2, ::2], b[::2, ::2]
        u = (A | B).sum()
        return 0.0 if u == 0 else float((A & B).sum()) / u

    def _insert(self, mask, source, contain=0.9):
        """Place a new mask: parent = smallest node containing it; existing nodes
        it contains become its children. Returns id, or an existing id if duplicate."""
        area = int(mask.sum())
        for i, nd in self.nodes.items():
            if abs(nd["area"] - area) < 0.6 * max(nd["area"], area) and self._iou(mask, nd["mask"]) >= 0.8:
                return i, False
        cands = [i for i, nd in self.nodes.items()
                 if nd["area"] >= 1.25 * area and self._contain_frac(nd["mask"], mask) >= contain]
        parent = min(cands, key=lambda i: self.nodes[i]["area"]) if cands else 0
        i = self._next
        self._next += 1
        self.nodes[i] = {"parent": parent, "children": [], "name": None, "text": None,
                         "area": area, "bbox": C.mask_record(mask)["bbox"],
                         "source": source, "mask": mask}
        # adopt children of the parent that lie inside the new node
        for c in list(self.nodes[parent]["children"]):
            if c != i and area >= 1.25 * self.nodes[c]["area"] and \
                    self._contain_frac(mask, self.nodes[c]["mask"]) >= contain:
                self.nodes[parent]["children"].remove(c)
                self.nodes[c]["parent"] = i
                self.nodes[i]["children"].append(c)
        self.nodes[parent]["children"].append(i)
        return i, True

    def _common_parent(self, ids):
        chains = []
        for i in ids:
            ch, p = [], self.nodes[i]["parent"]
            while p is not None:
                ch.append(p)
                p = self.nodes[p]["parent"]
            chains.append(ch)
        for p in chains[0]:
            if all(p in ch for ch in chains):
                return p
        return 0

    # ------------------------------------------------------------ verbs
    def tree_text(self, fmt_box, root=0):
        lines = []

        def walk(i, ind):
            nd = self.nodes[i]
            p = nd["parent"]
            share = "" if p is None else f" {100 * nd['area'] / max(self.nodes[p]['area'], 1):.0f}% of #{p}"
            name = nd["name"] or "?"
            txt = f' text="{nd["text"]}"' if nd.get("text") else ""
            lines.append(f"{'  ' * ind}#{i} {name}{txt} box {fmt_box(nd['bbox'])}{share}")
            for c in sorted(nd["children"], key=lambda k: -self.nodes[k]["area"]):
                walk(c, ind + 1)
        walk(root, 0)
        return "\n".join(lines)

    def show(self, i, img, margin=0.08, max_w=1100):
        """Crop of node i with its children outlined and numbered. (text, ViewImage)."""
        if i not in self.nodes:
            raise KeyError(f"no node #{i}")
        nd = self.nodes[i]
        h, w = self.rgb.shape[:2]
        x0, y0, x1, y1 = C.expand_box(nd["bbox"], (h, w), margin)
        sub = self.rgb[y0:y1, x0:x1]
        kids = sorted(nd["children"], key=lambda k: -self.nodes[k]["area"])
        if kids:
            ms = [self.nodes[k]["mask"][y0:y1, x0:x1] for k in kids]
            cols = [COLORS[j % len(COLORS)] for j in range(len(kids))]
            out = C.outlines(sub, ms, cols, thick=2, dim=0.15)
        else:
            out = sub.copy()
        im = Image.fromarray(out)
        d = ImageDraw.Draw(im)
        font = C._font(16)
        for j, k in enumerate(kids):
            bx0, by0, bx1, by1 = self.nodes[k]["bbox"]
            tx, ty = max(0, bx0 - x0), max(0, by0 - y0 - 18)
            label = f"#{k}" + (f" {self.nodes[k]['name']}" if self.nodes[k]["name"] else "")
            tb = d.textbbox((tx, ty), label, font=font)
            d.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=COLORS[j % len(COLORS)])
            d.text((tx, ty), label, fill="white", font=font)
        out = np.array(im)
        if out.shape[1] > max_w:
            out = C.resize(out, max_w)
        kid_txt = ", ".join(
            f"#{k}" + (f" {self.nodes[k]['name']}" if self.nodes[k]["name"] else "")
            + f" ({100 * self.nodes[k]['area'] / max(nd['area'], 1):.0f}%)" for k in kids)
        text = (f"show #{i}" + (f" {nd['name']}" if nd["name"] else "")
                + f": region {x0},{y0}-{x1},{y1}; "
                + (f"{len(kids)} children outlined and numbered: {kid_txt}" if kids
                   else "no children — if you see things here, use find{phrase}"))
        view = replace(img, rgb=out, box=(x0, y0, x1, y1),
                       text=f"{img.text} · show #{i} {x0},{y0}-{x1},{y1}")
        return text, view

    def find(self, phrase, verbs, img, min_score=0.3, max_n=12):
        inst = [d for d in verbs.instances(img, phrase) if d.score >= min_score][:max_n]
        obj = self.nodes[0]["mask"]
        new, old = [], []
        for d in inst:
            m = d.mask & obj
            if m.sum() < 40 or m.sum() < 0.5 * d.mask.sum():
                continue
            i, created = self._insert(m, f"find:{phrase}")
            (new if created else old).append(i)
        self.log.append({"op": "find", "phrase": phrase, "new": new, "existing": old})
        if not new and not old:
            return f"find '{phrase}': nothing found on the object"
        return (f"find '{phrase}': " + (f"new nodes {['#%d' % i for i in new]}" if new else "")
                + (f"; already in the tree {['#%d' % i for i in old]}" if old else "")
                + ". Call tree to see where they sit.")

    def name(self, i, name, text=None):
        if i not in self.nodes:
            return f"name: no node #{i}"
        self.nodes[i]["name"] = str(name).strip()
        if text:
            self.nodes[i]["text"] = str(text).strip()
        self.log.append({"op": "name", "id": i, "name": name, "text": text})
        return f"name: #{i} = {name!r}" + (f' text="{text}"' if text else "")

    def group(self, ids, name):
        ids = [i for i in ids if i in self.nodes and i != 0]
        if len(ids) < 2:
            return "group: give at least two existing node ids"
        parent = self._common_parent(ids)
        mask = np.zeros_like(self.nodes[0]["mask"])
        for i in ids:
            mask |= self.nodes[i]["mask"]
        g = self._next
        self._next += 1
        self.nodes[g] = {"parent": parent, "children": [], "name": str(name), "text": None,
                         "area": int(mask.sum()), "bbox": C.mask_record(mask)["bbox"],
                         "source": "group", "mask": mask}
        for i in ids:
            p = self.nodes[i]["parent"]
            if p is not None and i in self.nodes[p]["children"]:
                self.nodes[p]["children"].remove(i)
            self.nodes[i]["parent"] = g
            self.nodes[g]["children"].append(i)
        self.nodes[parent]["children"].append(g)
        self.log.append({"op": "group", "ids": ids, "name": name, "id": g})
        return f"group: #{g} {name!r} under #{parent} with children {['#%d' % i for i in ids]}"

    def drop(self, ids):
        done = []
        for i in ids:
            if i in self.nodes and i != 0:
                self._remove(i, keep_children=True)
                done.append(i)
        self.log.append({"op": "drop", "ids": done})
        return f"drop: removed {['#%d' % i for i in done]} (their children moved up)" if done \
            else "drop: no such nodes"

    # ------------------------------------------------------------ output
    def to_json(self):
        out = []
        for i, nd in self.nodes.items():
            out.append({"id": i, "parent": nd["parent"], "depth": self.depth(i),
                        "name": nd["name"], "text": nd["text"], "area": nd["area"],
                        "bbox": nd["bbox"], "source": nd["source"],
                        "children": list(nd["children"])})
        return {"nodes": sorted(out, key=lambda n: n["id"]), "log": self.log,
                "object_from": getattr(self, "object_from", None)}

    def as_text(self):
        return self.tree_text(lambda b: str(list(b)))

    def save(self, d, tag):
        d.mkdir(parents=True, exist_ok=True)
        json.dump(self.to_json(), open(d / f"tree_{tag}.json", "w"), indent=1)
        (d / f"tree_{tag}.txt").write_text(self.as_text() + "\n")
        ids = sorted(self.nodes)
        C.save_packed(d / f"masks_{tag}.npz", [self.nodes[i]["mask"] for i in ids])
        json.dump(ids, open(d / f"masks_{tag}_ids.json", "w"))

    def render_levels(self, title, named_only=False):
        obj = self.nodes[0]["mask"]
        x0, y0, x1, y1 = T.crop_region(obj)
        maxd = self.max_depth()
        panels = []
        for d in range(1, maxd + 1):
            ids = [i for i in self.nodes if self.depth(i) == d and (not named_only or self.nodes[i]["name"])]
            sub = self.rgb[y0:y1, x0:x1]
            ms = [self.nodes[i]["mask"][y0:y1, x0:x1] for i in ids]
            ov = C.overlay(sub, ms, alpha=0.5, seed=d) if ms else sub.copy()
            im = Image.fromarray(ov)
            dr = ImageDraw.Draw(im)
            font = C._font(15)
            for i in ids:
                if self.nodes[i]["name"]:
                    bx0, by0 = self.nodes[i]["bbox"][:2]
                    tx, ty = max(0, bx0 - x0), max(0, by0 - y0 - 17)
                    lab = f"#{i} {self.nodes[i]['name']}"
                    tb = dr.textbbox((tx, ty), lab, font=font)
                    dr.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=(0, 0, 0))
                    dr.text((tx, ty), lab, fill="white", font=font)
            panels.append(C.label(np.array(im), f"depth {d}: {len(ids)} nodes", size=20))
        cols = 1 if len(panels) <= 3 else 2
        img = C.grid(panels, cols)
        return C.title_band(img, title)
