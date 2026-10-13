"""S / T / V: the surface rows of V, the parts (S) in a tree (T), one deepest label per row.

Surface: the voxels of the TSDF grid that lie on the surface (weight > 0.5, |tsdf| < one
voxel), one row each; V's geometry is indexed, not copied.
Part: a node of T; it holds a sorted array of rows; children's rows lie inside the parent's
(assign adds to every ancestor). label[row] = the deepest part holding the row, -1 if none;
a row claimed by two unrelated parts keeps its first label and the pair is recorded in
`conflicts` (merging is not decided here).

Real-V check:  p inspection/investigation/iwm/e05_parts_in_v/stv.py 2209-compute-ray-demo5 1
"""
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from inspection.investigation.iwm.e04_voxel_scene.build import BLOCK_RES

_OFF = 1 << 20


def _pack(ijk):
    q = np.asarray(ijk, np.int64) + _OFF
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


class Surface:
    def __init__(self, ijk, tsdf, weight, size):
        # canonical row order = ascending (i, j, k) key: a row number names one voxel in every
        # process (Open3D's active_buf_indices() order is not stable across loads)
        key = _pack(np.asarray(ijk, np.int64))
        order = np.argsort(key, kind="stable")
        self.ijk = np.ascontiguousarray(np.asarray(ijk)[order], dtype=np.int32)
        self.tsdf = np.asarray(tsdf, np.float32)[order]
        self.weight = np.asarray(weight, np.float32)[order]
        self.size = float(size)
        self._sorted = key[order]
        self._order = np.arange(len(self._sorted))
        self._kd = None

    def __len__(self):
        return len(self.ijk)

    def fingerprint(self):
        """Cheap identity of the row set: count, size, and a hash of the sorted keys."""
        h = int(np.bitwise_xor.reduce(self._sorted * np.int64(0x9E3779B1) ^ np.arange(len(self._sorted), dtype=np.int64)))
        return {"rows": int(len(self)), "size": self.size, "hash": h}

    SHELL_VOXELS = 1.0      # a row is a voxel within this many voxels of the zero crossing (set by the real-V check)

    @classmethod
    def from_vbg(cls, vbg, size_m, trunc_m, index="round", shell_voxels=None):
        shell = cls.SHELL_VOXELS if shell_voxels is None else shell_voxels
        hm = vbg.hashmap()
        buf = hm.active_buf_indices()
        keys = hm.key_tensor()[buf].cpu().numpy().astype(np.int64)                 # (N, 3) block keys
        tsdf = vbg.attribute("tsdf")[buf].cpu().numpy()[..., 0]                    # (N, 16, 16, 16) stored [z][y][x]
        w = vbg.attribute("weight")[buf].cpu().numpy()[..., 0]
        tsdf = np.transpose(tsdf, (0, 3, 2, 1))                                    # [n][x][y][z]
        w = np.transpose(w, (0, 3, 2, 1))
        sel = (w > 0.5) & (np.abs(tsdf) * trunc_m < shell * size_m)
        n, x, y, z = np.nonzero(sel)
        ijk = keys[n] * BLOCK_RES + np.stack([x, y, z], 1)
        s = cls(ijk, tsdf[sel], w[sel], size_m)
        s.index = index
        return s

    # ------------------------------------------------------------ lookups
    def _grid(self, pts):
        q = np.asarray(pts, np.float64) / self.size
        return (np.floor(q) if getattr(self, "index", "round") == "floor" else np.rint(q)).astype(np.int64)

    def rows_at(self, pts):
        """Point -> its own voxel's row, -1 if that voxel is not a surface row."""
        key = _pack(self._grid(pts))
        pos = np.searchsorted(self._sorted, key)
        pos_c = np.minimum(pos, len(self._sorted) - 1)
        ok = (pos < len(self._sorted)) & (self._sorted[pos_c] == key)
        return np.where(ok, self._order[pos_c], -1)

    @property
    def kd(self):
        if self._kd is None:
            self._kd = cKDTree(self.centres(np.arange(len(self))))
        return self._kd

    def rows_near(self, pts, max_voxels=1.5):
        """Own voxel's row, else the nearest row within max_voxels * size, else -1."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        r = self.rows_at(pts)
        miss = r == -1
        if miss.any():
            d, i = self.kd.query(pts[miss], distance_upper_bound=max_voxels * self.size)
            r[miss] = np.where(i < len(self), i, -1)
        return r

    def centres(self, rows):
        return self.ijk[np.asarray(rows, np.int64)].astype(np.float64) * self.size


@dataclass
class Part:
    id: int
    name: str
    text: str | None = None
    parent: int | None = None
    children: list = field(default_factory=list)
    rows: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    detections: list = field(default_factory=list)
    aliases: list = field(default_factory=list)       # names later views gave the same part


class STV:
    def __init__(self, surface):
        self.s = surface
        self.parts = {}
        self.label = np.full(len(surface), -1, np.int32)
        self.conflicts = []
        self.meta = {}                      # view id -> {e02 node id: part id}
        self._next = 0
        self.root = self.add(None, "object")

    # ------------------------------------------------------------ tree
    def add(self, parent, name, rows=(), text=None, detection=None):
        pid = self._next
        self._next += 1
        self.parts[pid] = Part(pid, name, text, parent)
        if parent is not None:
            self.parts[parent].children.append(pid)
        if len(rows):
            self.assign(pid, rows, detection)
        elif detection is not None:
            self.parts[pid].detections.append(tuple(detection))
        return pid

    def ancestors(self, part):
        chain, q = [], self.parts[part].parent
        while q is not None:
            chain.append(q)
            q = self.parts[q].parent
        return chain

    def descendants(self, part):
        out, todo = [], list(self.parts[part].children)
        while todo:
            c = todo.pop()
            out.append(c)
            todo.extend(self.parts[c].children)
        return out

    def is_desc_or_self(self, a, b):
        """a is b or lies below b."""
        return a == b or b in self.ancestors(a)

    def depth(self, part):
        return len(self.ancestors(part))

    def path(self, part):
        return " / ".join(self.parts[p].name for p in reversed([part] + self.ancestors(part)))

    # ------------------------------------------------------------ rows
    def assign(self, part, rows, detection=None, alias=None):
        rows = np.unique(np.asarray(rows, np.int64))
        rows = rows[rows >= 0]
        p = self.parts[part]
        added = int(len(np.setdiff1d(rows, p.rows, assume_unique=True)))
        for q in [part] + self.ancestors(part):
            self.parts[q].rows = np.union1d(self.parts[q].rows, rows)
        if detection is not None and tuple(detection) not in p.detections:
            p.detections.append(tuple(detection))
        if alias and alias != p.name and alias not in p.aliases:
            p.aliases.append(alias)
        old = self.label[rows]
        take = (old == -1) | (old == part) | np.isin(old, self.ancestors(part))
        rest = ~take
        n_conf = 0
        if rest.any():
            deeper = np.isin(old[rest], self.descendants(part))          # already a child's: keep
            for r, l in zip(rows[rest][~deeper], old[rest][~deeper]):
                self.conflicts.append((int(r), int(l), int(part)))
                n_conf += 1
        relabelled = int((take & (old != part)).sum())
        self.label[rows[take]] = part
        return {"added": added, "relabelled": relabelled, "conflicts": n_conf}

    extend = assign

    def voxels_of(self, part):
        return self.parts[part].rows

    def part_of(self, row):
        l = int(self.label[row])
        return [] if l == -1 else [l] + self.ancestors(l)

    def search(self, query):
        q = query.lower().strip()
        hits = [p for p in self.parts
                if q in self.parts[p].name.lower() or (self.parts[p].text and q in self.parts[p].text.lower())
                or any(q in a.lower() for a in self.parts[p].aliases) or q in self.path(p).lower()]
        return sorted(hits, key=lambda p: (self.depth(p), p))

    # ------------------------------------------------------------ invariants
    def check(self):
        labelled = np.nonzero(self.label != -1)[0]
        assert np.isin(labelled, self.parts[self.root].rows).all(), "labelled rows outside the root"
        conflict_rows = np.array(sorted({c[0] for c in self.conflicts}), np.int64)   # the recorded exceptions
        for p in self.parts.values():
            if p.parent is not None:
                assert np.isin(p.rows, self.parts[p.parent].rows).all(), f"part {p.id} has rows outside parent {p.parent}"
            rows = p.rows[~np.isin(p.rows, conflict_rows)]
            l = self.label[rows]
            l = l[l != -1]
            ok = np.isin(l, [p.id] + self.descendants(p.id))
            assert ok.all(), f"row of part {p.id} labelled with unrelated part {int(l[~ok][0])}"

    # ------------------------------------------------------------ io
    def save(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        ids = sorted(self.parts)
        parts = [{"id": i, "name": self.parts[i].name, "text": self.parts[i].text, "parent": self.parts[i].parent,
                  "children": list(self.parts[i].children), "detections": [list(t) for t in self.parts[i].detections],
                  "aliases": list(self.parts[i].aliases),
                  "n_rows": int(len(self.parts[i].rows)), "depth": self.depth(i), "path": self.path(i)} for i in ids]
        json.dump({"root": self.root, "next": self._next, "parts": parts, "conflicts": self.conflicts,
                   "meta": {v: {str(k): int(p) for k, p in m.items()} for v, m in self.meta.items()},
                   "surface_rows": int(len(self.s)), "voxel_size_m": self.s.size, "surface": self.s.fingerprint()},
                  open(d / "parts.json", "w"), indent=1)
        rows = [self.parts[i].rows for i in ids]
        off = np.cumsum([0] + [len(r) for r in rows])
        np.savez_compressed(d / "rows.npz", rows=np.concatenate(rows) if rows else np.empty(0, np.int64),
                            offsets=off, ids=np.array(ids), label=self.label)

    @classmethod
    def load(cls, d, surface):
        d = Path(d)
        j = json.load(open(d / "parts.json"))
        assert j["surface_rows"] == len(surface), "surface does not match the saved parts (row count)"
        assert j.get("surface", surface.fingerprint()) == surface.fingerprint(), "surface does not match the saved parts (keys)"
        z = np.load(d / "rows.npz")
        t = cls.__new__(cls)
        t.s = surface
        t.parts = {}
        for k, pj in enumerate(j["parts"]):
            t.parts[pj["id"]] = Part(pj["id"], pj["name"], pj["text"], pj["parent"], list(pj["children"]),
                                     z["rows"][z["offsets"][k]:z["offsets"][k + 1]].astype(np.int64),
                                     [tuple(x) for x in pj["detections"]], list(pj.get("aliases", [])))
        t.label = z["label"].astype(np.int32)
        t.conflicts = [tuple(c) for c in j["conflicts"]]
        t.meta = {v: {int(k): p for k, p in m.items()} for v, m in j["meta"].items()}
        t._next = j["next"]
        t.root = j["root"]
        return t

    def summary(self):
        return {"parts": len(self.parts), "rows_labelled": int((self.label != -1).sum()), "surface_rows": len(self.s),
                "conflicts": len(self.conflicts),
                "bytes": int(self.label.nbytes + sum(p.rows.nbytes for p in self.parts.values()))}


def _real_check(run, mm):
    from inspection.investigation.iwm.e05_parts_in_v import common as C5
    vbg, size = C5.V(run, mm)
    out = {"run": run, "voxel_mm": mm}
    for index in ("round", "floor"):
        t0 = time.time()
        s = Surface.from_vbg(vbg, size, C5.trunc_m(size), index=index)
        out[f"build_s_{index}"] = round(time.time() - t0, 2)
        pc = vbg.extract_point_cloud(weight_threshold=0.5).point.positions.cpu().numpy()
        r = s.rows_near(pc, max_voxels=1.0)
        out[f"rows_{index}"] = len(s)
        out[f"cloud_points"] = len(pc)
        out[f"cloud_on_rows_{index}"] = round(float((r != -1).mean()), 4)
        out[f"cloud_exact_{index}"] = round(float((s.rows_at(pc) != -1).mean()), 4)
        out[f"bytes_{index}"] = int(s.ijk.nbytes + s.tsdf.nbytes + s.weight.nbytes)
    print(json.dumps(out, indent=1))
    C5.EVID.mkdir(exist_ok=True)
    p = C5.EVID / "surface_stats.json"
    prev = json.load(open(p)) if p.exists() else []
    prev = [x for x in prev if not (x["run"] == run and x["voxel_mm"] == mm)] + [out]
    json.dump(prev, open(p, "w"), indent=1)


if __name__ == "__main__":
    _real_check(sys.argv[1], int(sys.argv[2]))
