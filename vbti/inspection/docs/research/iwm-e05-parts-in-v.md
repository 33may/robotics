# IWM e05 — parts in V — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native: one session,
> the way this harness runs work). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build the S/T/V structure over e04's V, colour it from the e02 trees of several
views (the agent placing each new view's nodes), and compute the view for a part and for a set
of parts — producing the numbers that answer RQ1–RQ6 of the spec.

**Architecture:** one module `stv.py` holds the structure (surface-row table over the TSDF
grid, parts with sorted row arrays, one deepest label per row, invariants). Everything else is
a measurement script that uses it: `lift.py` (2D mask → rows, RQ1), `pairs.py` (RQ2),
`keep.py` (sequential views with the agent, RQ4), `views.py` (view for a part / set, RQ5–6),
`viewer.py` (Viser). Scripts write to `inspection/data/iwm/e05_parts_in_v/` (heavy) and
`e05_parts_in_v/evidence/` (what the report shows), each with a `run.json`.

**Tech Stack:** Python 3 (`robo` conda env, run as `p <script>` = `~/miniconda3/envs/robo/bin/python
-u -m ...` from the repo root), Open3D 0.19 `VoxelBlockGrid` (CUDA), numpy, scipy cKDTree, PIL,
Viser 1.1.1, the in-loop `VlmAgent` + Gemini ER-2 (e02 harness).

**Spec:** `inspection/investigation/iwm/e05_parts_in_v/NOTE.md`

## Global Constraints

- Recorded runs under `inspection/data/runs/2209-*` are read-only.
- Heavy outputs go to `inspection/data/iwm/e05_parts_in_v/<run>/<mm>mm/`; figures and small
  result files to `inspection/investigation/iwm/e05_parts_in_v/evidence/`; never `/tmp`.
- Every script run writes `run.json` beside its outputs (`C.run_record`).
- Report numbers are read from files in `evidence/`.
- The e02 trees are inputs; no new tree-building runs.
- V is never corrected for the per-view offset.
- RANSAC fits use a fixed seed (`o3d.utility.random.seed(0)` before each `segment_plane`).
- No git commits unless Anton asks.
- Resolution 1 mm main, 2 mm check; both lifts measured in RQ1.
- Orientation rule (e03/e04): rgb / depth / masks on disk are upright; `T_base_cam` and the
  intrinsics are raw; on steps with `rgb_rotation_deg == 180` flip images `[::-1, ::-1]`
  before any geometry.

## Review Focus

1. **A mask pixel whose ray hits nothing in V** (hole on dark print) — must be counted in the
   hit share, never crash and never land on row 0. Test in Task 3.
2. **A hit point that rounds to a grid index that is not a surface row** (between samples) —
   must fall back to the nearest row within 1.5 voxels or be counted as a miss. Test in Task 3.
3. **Assigning rows to a part whose ancestor already labels them, and the reverse** — labels
   must end as the deepest part, with no conflict recorded. Test in Task 2.
4. **A view whose root is a face, not the object** (b1.4) — the keeper must let the agent place
   the root; the default for an undecided root is `same{root, object}`. Test in Task 6.
5. **A part on a 180°-rotated step** (c5.6 is one) — the lift and the label render must use the
   raw orientation, or the colouring lands mirrored. Test in Task 3 (c5.6 IoU must be > 0.5).

---

### Task 1: common — frames, V, poses, ray cast to rows

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/common.py`
- Create: `inspection/investigation/iwm/e05_parts_in_v/__init__.py` (empty)

**Interfaces:**
- Consumes: `e01_all_masks.common` (`frames`, `frame_paths`, `load_rgb`, `load_packed`,
  `run_record`, `outlines`, `grid`, `label`); `e04_voxel_scene.build` (`load_steps`,
  `intrinsic_tensor`, `device`, `OUT`, `TRUNC_VOXELS`, `MIN_TRUNC_M`, `BLOCK_RES`);
  `e04_voxel_scene.render.load_vbg`.
- Produces:
  - `FRAMES = {"c5.2": ("2209-compute-ray-demo5", 2), "c5.6": (..., 6), "c5.7": (..., 7), "b1.1": ("2209-box-ray-demo1", 1), "b1.4": (..., 4)}`
  - `ORDER = {"2209-compute-ray-demo5": ["c5.2", "c5.6", "c5.7"], "2209-box-ray-demo1": ["b1.1", "b1.4"]}`
  - `DATA = ROOT/inspection/data/iwm/e05_parts_in_v`, `EVID = HERE/evidence`
  - `load_view(fid) -> dict(id, run, step, T (4x4), K (3x3 np), intr, scale, depth (raw orientation), rgb, rot, tree (json), masks {id: bool HxW raw orientation})`
  - `load_tree(fid) -> (tree_json, masks_upright)`
  - `V(run, mm) -> (vbg, size_m)`
  - `cast(vbg, size_m, T, K, w, h) -> (depth HxW float m, normal HxWx3)` full-resolution ray cast (as e04 `cast` but any K/w/h, returns raw arrays)
  - `hits_to_points(depth, K, T) -> (pts (M,3) base frame, px (M,2) (v,u))`
  - `W, H = 1280, 720`

- [ ] **Step 1: write `common.py`** with the above; `load_view` flips depth/rgb/masks to raw when `rot == 180` (masks via `load_packed` + ids json, as e03 `load_tree`). `cast` = e04's call with `render_attributes=["depth","normal"]`, `weight_threshold=0.5` (so single-view voxels render, matching the mesh), returns `depth[...,0]` and `normal`.
- [ ] **Step 2: smoke test** — `p inspection/investigation/iwm/e05_parts_in_v/common.py c5.6` prints: view id, rot, number of nodes, mask count, V rows allocated, and the share of depth>0 pixels in the cast from the view's own pose (expect > 0.3).

### Task 2: stv — the structure

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/stv.py`
- Test: `inspection/investigation/iwm/e05_parts_in_v/tests/test_stv.py`

**Interfaces:**
- Produces:
  ```python
  class Surface:                       # the surface rows of V
      ijk: np.ndarray (N,3) int32      # grid index, position = ijk * size
      tsdf: (N,) float32; weight: (N,) float32; size: float
      def __init__(self, ijk, tsdf, weight, size)
      @classmethod from_vbg(cls, vbg, size_m, trunc_m) -> Surface   # weight > 0.5 and |tsdf| * trunc_m < size_m
      def rows_at(self, pts (M,3)) -> (M,) int64        # round(p/size) -> row, -1 if not a row
      def rows_near(self, pts, max_voxels=1.5) -> (M,) int64   # exact, else nearest row within max_voxels*size, else -1
      def centres(self, rows) -> (k,3)
      kd: cKDTree over centres (built lazily)

  class Part:  id:int; name:str; text:str|None; parent:int|None; children:list[int]
               rows: np.ndarray int64 sorted unique; detections: list[tuple[str,int]]   # (view id, e02 node id)

  class STV:
      def __init__(self, surface: Surface)
      parts: dict[int, Part]; root: int; label: np.ndarray (N,) int32 (-1 none); conflicts: list[tuple[int,int,int]]
      def add(self, parent: int|None, name, rows=(), text=None, detection=None) -> int
      def assign(self, part: int, rows, detection=None) -> dict(added=int, relabelled=int, conflicts=int)
      extend = assign
      def ancestors(self, part) -> list[int]      # part's chain up to the root, excluding part
      def is_desc_or_self(self, a, b) -> bool     # a is b or below b
      def depth(self, part) -> int
      def voxels_of(self, part) -> np.ndarray
      def part_of(self, row) -> list[int]         # [deepest, ..., root] or []
      def path(self, part) -> str                 # "object / front face / LAN ports / LAN 1G port"
      def search(self, query) -> list[int]        # case-insensitive substring over name, text and path; sorted by depth then id
      def check(self) -> None                     # raises AssertionError with the first violation
      def save(self, d: Path); @classmethod load(cls, d, surface) -> STV
      def summary(self) -> dict                   # parts, rows labelled, conflicts, bytes
  ```
- `assign` rule: for each row r in rows: add r to part and all its ancestors (set union, stays sorted); `l = label[r]`; if `l == -1` or `is_desc_or_self(part, l)` (l is an ancestor of part or part itself): `label[r] = part`; elif `is_desc_or_self(l, part)` (l is deeper, under part): keep; else: keep `l`, append `(r, l, part)` to conflicts.
- `check`: for every part p with parent q: `set(p.rows) <= set(q.rows)`; for every row r with label l != -1: for every part p holding r, `is_desc_or_self(l, p)`; the root's rows == the union of all labelled rows ∪ its own.

- [ ] **Step 1: write the failing tests** (toy surface: a 10×10×1 grid of rows, size 0.001)

```python
import numpy as np, pytest
from inspection.investigation.iwm.e05_parts_in_v.stv import Surface, STV

def toy():
    g = np.stack(np.meshgrid(np.arange(10), np.arange(10), [0], indexing="ij"), -1).reshape(-1, 3)
    return Surface(g.astype(np.int32), np.zeros(len(g), np.float32), np.ones(len(g), np.float32), 0.001)

def test_rows_at_and_near():
    s = toy()
    assert s.rows_at(np.array([[0.0031, 0.0049, 0.0]]))[0] == s.rows_at(np.array([[0.003, 0.005, 0.0]]))[0]
    assert s.rows_at(np.array([[0.02, 0.0, 0.0]]))[0] == -1
    assert s.rows_near(np.array([[0.0094, 0.0, 0.0]]))[0] == s.rows_at(np.array([[0.009, 0.0, 0.0]]))[0]
    assert s.rows_near(np.array([[0.012, 0.0, 0.0]]), max_voxels=1.5)[0] == -1

def test_children_inside_parent_and_deepest_label():
    s = toy(); t = STV(s)
    obj = t.root
    face = t.add(obj, "face", rows=np.arange(0, 50))
    port = t.add(face, "port", rows=np.arange(10, 14))
    assert set(t.parts[port].rows) <= set(t.parts[face].rows) <= set(t.parts[obj].rows)
    assert (t.label[10:14] == port).all() and (t.label[0:10] == face).all()
    t.assign(face, np.arange(10, 14))           # parent re-assigned over the child: child keeps the label
    assert (t.label[10:14] == port).all() and not t.conflicts
    t.assign(port, np.arange(14, 16))            # child extended into parent-only rows: relabelled, no conflict
    assert (t.label[14:16] == port).all() and not t.conflicts
    t.check()

def test_conflict_recorded_first_wins():
    s = toy(); t = STV(s)
    a = t.add(t.root, "A", rows=np.arange(0, 5)); b = t.add(t.root, "B", rows=np.arange(3, 8))
    assert (t.label[3:5] == a).all() and len(t.conflicts) == 2 and t.conflicts[0] == (3, a, b)
    assert set(np.arange(3, 5)) <= set(t.parts[b].rows)      # B still holds the rows
    t.check()

def test_part_of_path_search_roundtrip(tmp_path):
    s = toy(); t = STV(s)
    face = t.add(t.root, "front face", rows=np.arange(0, 50))
    grp = t.add(face, "LAN ports", rows=np.arange(10, 20))
    p1 = t.add(grp, "LAN 1G port", rows=np.arange(10, 14), text="1G")
    assert t.part_of(12) == [p1, grp, face, t.root] and t.part_of(60) == []
    assert t.path(p1) == "object / front face / LAN ports / LAN 1G port"
    assert t.search("lan") == [grp, p1] and t.search("1g") == [p1]
    t.save(tmp_path); u = STV.load(tmp_path, s)
    assert u.path(p1) == t.path(p1) and (u.label == t.label).all() and u.conflicts == t.conflicts
```

- [ ] **Step 2: run** `~/miniconda3/envs/robo/bin/python -m pytest inspection/investigation/iwm/e05_parts_in_v/tests -q` — expect ImportError.
- [ ] **Step 3: write `stv.py`** per the interface. `Surface.from_vbg`: `buf = vbg.hashmap().active_buf_indices()`; `keys = vbg.hashmap().key_tensor()[buf].cpu().numpy()` (N,3); `tsdf = vbg.attribute("tsdf")[buf]` → `(N,16,16,16,1)` → `np.transpose(arr[...,0], (0,3,2,1))` so the array is `[n][x][y][z]`; same for weight; mask `(w > 0.5) & (abs(tsdf) * trunc_m < size_m)`; `ijk = keys[n]*16 + (x,y,z)`. Hash: `np.ravel_multi_index` is unsafe for negatives → use a dict-free lookup: sort rows by a packed int64 key `(i+2**20)*2**42 + (j+2**20)*2**21 + (k+2**20)` and `np.searchsorted`. `rows_near`: exact first, then `kd.query` on the misses with `distance_upper_bound = max_voxels*size`.
- [ ] **Step 4: run the tests** — expect 4 passed.
- [ ] **Step 5: real-V check** — `p .../stv.py 2209-compute-ray-demo5 1` builds the surface at 1 mm and prints rows, bytes, build seconds, and the share of `vbg.extract_point_cloud(weight_threshold=0.5)` points whose `rows_near(max_voxels=1.0)` is not −1 (expect > 0.95; if < 0.9, the index convention is off by half a voxel: try `floor` instead of `round` and keep the better, recording which in `evidence/surface_stats.json`).

### Task 3: lift — a view's tree into rows, and the label render (RQ1)

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/lift.py`
- Test: `inspection/investigation/iwm/e05_parts_in_v/tests/test_lift.py` (integration, skipped without the data)

**Interfaces:**
- Consumes: Task 1 `load_view`, `V`, `cast`, `hits_to_points`; Task 2 `Surface`, `STV`.
- Produces:
  ```python
  def lift_L1(view, surface, snap_m=0.003) -> (rows_by_node: dict[int, np.ndarray], stats: dict[int, dict])
      # deproject(depth, intr, scale, stride=1, return_px=True) -> base pts; per node: pts under the mask -> rows_near(max_voxels=snap_m/size); stats: n_px, n_pts (px with depth), n_rows, unmatched share
  def lift_L2(view, surface, vbg, size_m) -> (rows_by_node, stats)
      # cast from view.T, view.K at W×H; per node: hit points under the mask -> rows_near(1.5); stats: n_px, hit share, n_rows, miss share
  def colour(stv, view, rows_by_node) -> None
      # for node in tree order (parents first): part = stv.add(parent_part, name, rows=rows_by_node[id], text, detection=(view.id, id)); node 0 -> stv.root (rename root if unnamed); keeps a map node id -> part id on stv.meta[view.id]
  def label_render(stv, vbg, size_m, T, K, w=W, h=H) -> (labels HxW int32 (-1 none, -2 hit but no row), depth, normal)
      # cast -> hits -> rows_near(1.5) -> stv.label
  def part_pixels(labels, stv, part) -> bool HxW      # pixels whose label is part or a descendant
  def smear(surface, rows, ref_rows) -> float          # share of rows > 8 mm off the RANSAC plane of ref_rows (the node's face = its depth-1 ancestor's rows; for depth-1 nodes their own)
  def iou(a, b) -> (iou, precision, recall)
  ```
- Script `p .../lift.py --frames c5.6 b1.4 --sizes 1 2` → per frame, size, lift: builds a fresh STV, colours it, renders the labels from the view's pose, and writes:
  - `evidence/lift.json`: per (frame, mm, lift, node): name, depth, n_px, hit/depth share, n_rows, iou, precision, recall, smear, extent_mm; plus per (frame, mm, lift) medians split by node width (< 20 mm, ≥ 20 mm).
  - `evidence/lift_<frame>_<mm>mm_<lift>.jpg`: a grid, one tile per node: left the frame crop with the mask outline, right the label render crop (rendered colour image, part's pixels outlined), title "#id name IoU 0.xx smear x %".
  - `DATA/<run>/<mm>mm/single_<frame>_<lift>/` the STV (save) + `run.json`.

- [ ] **Step 1: write the integration test**

```python
import numpy as np, pytest
from pathlib import Path
from inspection.investigation.iwm.e05_parts_in_v import common as C5
pytestmark = pytest.mark.skipif(not (C5.DATA.parent / "e04_voxel_scene" / "2209-compute-ray-demo5" / "scene" / "2mm" / "vbg.npz").exists(), reason="needs e04 V")

def test_l2_roundtrip_c56_2mm():
    from inspection.investigation.iwm.e05_parts_in_v import lift as L
    from inspection.investigation.iwm.e05_parts_in_v.stv import Surface, STV
    view = C5.load_view("c5.6"); vbg, size = C5.V(view["run"], 2)
    surf = Surface.from_vbg(vbg, size, max(4 * size, 0.004))
    rows, stats = L.lift_L2(view, surf, vbg, size)
    assert all(s["hit_share"] > 0.5 for s in stats.values())          # holes exist but never dominate
    assert all((r >= 0).all() for r in rows.values())                  # never row -1 inside a node
    t = STV(surf); L.colour(t, view, rows); t.check()
    lab, _, _ = L.label_render(t, vbg, size, view["T"], view["K"])
    for nid, part in t.meta["c5.6"].items():
        if nid == 0: continue
        i, _, _ = L.iou(L.part_pixels(lab, t, part), view["masks"][nid])
        assert i > 0.5, (nid, i)                                       # catches a mirrored (180°) colouring
```

- [ ] **Step 2: run** — expect ImportError on `lift`.
- [ ] **Step 3: write `lift.py`** (functions above + `main` with argparse `--frames`, `--sizes`, `--lifts L1 L2`). Face for `smear`: the node's ancestor at depth 1 (depth-0/1 nodes use themselves). Figure tiles via `C.outlines`, `C.grid`, `C.label` (e01 common); crops expanded 8 % like `IwmTree.show`.
- [ ] **Step 4: run the test** — expect PASS. If IoU < 0.5 on c5.6 (rotated step), the orientation is wrong somewhere: print the mask centroid vs the part's pixel centroid and fix before continuing.
- [ ] **Step 5: run the script** on c5.6 and b1.4 at 1 and 2 mm, both lifts. Look at every figure (Read the jpg). Record in `evidence/lift.json` which lift goes downstream (`"chosen": "L2"` or `"L1"`), by the median IoU of small nodes and the smear.

### Task 4: truth pairs (RQ2 / RQ4 ground truth)

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/evidence/truth_pairs.json`

- [ ] **Step 1: view the e02 final figures** — Read `e02_agent_tree/evidence/{c5.2,c5.6,c5.7,b1.1,b1.4}_full_final.png` and the `tree_final.txt` of each.
- [ ] **Step 2: write the file** as `{"author": "claude, 2026-10-12, for Anton's check", "pairs": [{"a": "c5.2", "a_id": 35, "b": "c5.6", "b_id": 12, "what": "ethernet ports group = LAN ports group"}, ...], "none": [{"view": "c5.6", "id": 7, "what": "antenna port 1, not in c5.2"}, ...]}` covering c5.2↔c5.6 (ports, groups, sticker, audio, faces), c5.2↔c5.7 (object, the top face if it is the same), b1.1↔b1.4 (front face ↔ root, the label(s)). Every node of the later view gets either a pair or a `none` entry.

### Task 5: pairs — cross-view overlap and offset (RQ2)

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/pairs.py`

**Interfaces:**
- Consumes: Task 3 lift (chosen), Task 4 truth pairs.
- Produces `evidence/pairs.json`, `evidence/pairs_<run>_<mm>mm.jpg`, `evidence/pairs_render_<later view>.jpg`.

- [ ] **Step 1: write `pairs.py`**: for each run and mm: STV coloured from the first view; the later view lifted separately (rows per node, not assigned). Per truth pair: `iou_rows` (|A∩B|/|A∪B|), centroid offset vector (mm) and norm; the view's rigid vector = the median of the pair offset vectors over pairs wider than 20 mm; `iou_rows_shifted` after moving the later view's rows by −vector (via `rows_near` of the shifted centres); width of the part (max PCA extent). Render test: `label_render` of the first-view STV from the later view's pose vs the later view's masks: IoU and pixel centroid offset per pair.
- [ ] **Step 2: figures**: 3D: the two row sets of each pair in two colours, rendered from the later view's pose on the grey V render (one tile per pair); 2D: the later view's frame with its mask outline and the projected first-view part outline.
- [ ] **Step 3: run** for both runs at 1 and 2 mm; read the numbers against H2 (small parts < 50 % overlap?).

### Task 6: keep — sequential views, the agent places the nodes (RQ3, RQ4)

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/keep.py`
- Test: `inspection/investigation/iwm/e05_parts_in_v/tests/test_keep.py`

**Interfaces:**
- Consumes: e02 harness pattern (`VA.TOOLS`, `VA._dispatch` wrap, `VA.VlmAgent(rules, kind="inspect", tool_names, emit, extras, max_turns)`, `ViewTools`, `RunStore`, `FindingWriter`, `GeminiVlm`, `LocalVerbs(SamOcrBackend())`), `IwmTree` (constructed with `IwmTree(rgb)` and `nodes` filled from `tree_final.json` + masks: keys `parent, children, name, text, area, bbox, source, mask`), Task 3 lift + label_render, Task 2 STV.
- Produces:
  ```python
  class Keeper:               # one new view against the current STV
      def __init__(self, stv, vbg, size_m, view, rows_by_node)
      def existing(self) -> str          # visible existing parts in this view: "#p name (path) px box; new nodes overlapping: #n xx%"
      def show_existing(self, node, img) -> (text, ViewImage)   # the node's crop with existing parts outlined + named
      def overlap(self, node) -> list[(part, share)]            # share of the node's rows held by each existing part, deepest-first
      def decide_same(self, node, part); def decide_under(self, node, part, name)
      decisions: dict[int, tuple]        # node -> ("same", part) | ("under", part, name) | ("drop",)
      def resolve_defaults(self)         # undecided root -> same(root, stv.root); undecided node -> under(resolved parent part, own name)
      def apply(self) -> dict            # top-down: same -> stv.extend(part, rows, detection); under -> stv.add(...); returns counts + conflicts
  def baseline_decisions(stv, view, rows_by_node) -> dict      # same to the child of the resolved parent with share >= 0.5, else under
  def run_view(stv, vbg, size_m, view, model, verbs, mode="agent"|"baseline") -> summary
  def score(decisions, truth, view) -> dict(recovered, wrong, invented, missed, by_size={"<20mm":..., ">=20mm":...})
  ```
- Verbs added for the keeper (process-local, like e02): `existing{}`, `show_existing{node}`, `same{node, part}`, `under{node, part, name}`, plus e02's `tree`, `show`, `drop`, `crop`, `read_text`, `note`. RULES text: the task described in the spec stage 4 (one tool per reply; decide every node incl. #0; `same` only when it is the same physical thing; `under` with the name it should carry; final reply `{"evidence","reasoning","answer","decisions": {"<node>": "same:<part>" | "under:<part>:<name>" | "drop"}}`).
- Script: `p .../keep.py --run 2209-compute-ray-demo5 --mm 1 --mode agent` → order from `ORDER`; first view coloured by `lift.colour`; each later view through `run_view`; after each view: `stv.check()`, STV saved to `DATA/<run>/<mm>mm/keep_<mode>/after_<view>/`, transcripts under the store; `evidence/agent_vs_truth.json` (per view: decisions, scores, turns, tool calls, seconds, conflicts after), `evidence/conflicts.json` (per view and mode), `evidence/keep_<run>_<mm>mm_<mode>.jpg` (the label render from each view's pose after all views, with a legend).

- [ ] **Step 1: write the test** (no model: baseline and defaults only)

```python
import numpy as np, pytest
from inspection.investigation.iwm.e05_parts_in_v.stv import Surface, STV

def toy():
    g = np.stack(np.meshgrid(np.arange(20), np.arange(20), [0], indexing="ij"), -1).reshape(-1, 3)
    return Surface(g.astype(np.int32), np.zeros(len(g), np.float32), np.ones(len(g), np.float32), 0.001)

def test_baseline_same_and_under_and_face_root():
    from inspection.investigation.iwm.e05_parts_in_v import keep as K
    s = toy(); t = STV(s)
    face = t.add(t.root, "front face", rows=np.arange(0, 200)); port = t.add(face, "port", rows=np.arange(40, 48))
    view = {"id": "x", "tree": {"nodes": [{"id": 0, "parent": None, "depth": 0, "name": "front face", "text": None, "children": [1, 2]},
                                          {"id": 1, "parent": 0, "depth": 1, "name": "a port", "text": None, "children": []},
                                          {"id": 2, "parent": 0, "depth": 1, "name": "a label", "text": None, "children": []}]}}
    rows = {0: np.arange(0, 200), 1: np.arange(41, 49), 2: np.arange(100, 110)}
    d = K.baseline_decisions(t, view, rows)
    assert d[0] == ("same", face)                 # the root is a face here, and the geometry says so
    assert d[1] == ("same", port)                 # 7 of 8 rows on the port
    assert d[2] == ("under", face, "a label")     # nothing holds it at the child level
```

- [ ] **Step 2: run** — ImportError.
- [ ] **Step 3: write `keep.py`**; `baseline_decisions` resolves top-down: node 0 → the existing part with the largest share among {root} ∪ root's children if share ≥ 0.5 else `same(root)`; node n with resolved parent P → candidates = P's children; best share ≥ 0.5 → same; else under(P, name).
- [ ] **Step 4: run the test** — PASS.
- [ ] **Step 5: run baseline** on both runs at 1 mm (fast); check `agent_vs_truth.json` scores and conflicts.
- [ ] **Step 6: run agent** on both runs at 1 mm; read each transcript's decisions; re-run a view once if a decision looks like noise (keep both, mark in the json). Then 2 mm baseline only.
- [ ] **Step 7: figures**; read them.

### Task 7: views — the view for a part and for a set (RQ5, RQ6)

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/views.py`
- Test: `inspection/investigation/iwm/e05_parts_in_v/tests/test_views.py`

**Interfaces:**
- Consumes: the final STV of each run (`keep_agent/after_<last view>`), Task 1 `cast`, `hits_to_points`, Task 2 Surface.
- Produces:
  ```python
  FOV_H = 2*atan(W/(2*fx)); D_MIN, D_MAX = 0.07, 0.50; FILL = 0.5; R16 = 0.016
  def normal_of(stv, part, toward=None) -> (n, face_part, inlier_ratio)   # face = nearest ancestor (self included) with >= 300 rows and RANSAC(3 mm, seed 0) inlier ratio >= 0.7; fit over the face's rows within R16 of the part centre; flipped toward `toward` (default: the mean camera position of the part's detecting steps)
  def frame_of(stv, part) -> dict(centre, normal, extent (3,), axes (3,3), face)   # PCA box of the part's rows
  def pose_for(centre, normal, up_axis, dist) -> T_base_cam (4x4)   # cam z = -normal, cam x = up_axis projected, y = z × x
  def standoff(extent) -> float      # clip(max in-plane extent / (2*FILL*tan(FOV_H/2)), D_MIN, D_MAX)
  def candidates(stv, part) -> list[dict(name, T)]   # "normal"; "tilt±15/30 {x,y}" (rotate about the centre around each in-plane axis); "pivot k/8 @30°"; "step <id>" for the run's captured steps
  def score(stv, vbg, size_m, part, T, K=K_scaled 0.5, w=640, h=360) -> dict(visible, occluded, outside, centring_x, centring_y, incidence_med_deg, px_w, px_h, dist_m)
      # visible = share of the part's rows within 1 voxel of a hit point (kd on hits); outside = rows projecting off-frame; occluded = in-frame rows whose depth > rendered depth at their pixel + 2*size; incidence = median angle(ray, rendered normal) over the part's pixels; centring = |centroid - centre| / (W, H)
  def rank(scored) -> list      # lexicographic: visible >= 0.9 first (then by visible), then incidence asc, then max(centring) asc
  def view_for(stv, vbg, size_m, part) -> dict(best, all)
  def view_for_set(stv, vbg, size_m, parts) -> dict   # union rows; normal from RANSAC over the union (seed 0) if inlier ratio >= 0.7 else the mean of the parts' normals; standoff from the union extent; score = min visible over the parts, then union centring
  ```
- Script `p .../views.py --run ... --mm 1`: the per-part table `evidence/views.json` (every part: frame_of, standoff, the detecting step's real distance and its score, the best candidate and its score, the normal pose's score, the tilt scores); `evidence/sets.json` (the six sets of the spec by `search` names); figures `evidence/view_<run>_<part>.jpg` for the eight chosen parts (a row: normal pose render | best tilt | pivot ring thumbnails | the detecting step's render | the step's real frame crop; scores in the titles), `evidence/pivot_<part>.jpg`, `evidence/set_<name>.jpg`, `evidence/poses_<run>.jpg` (proposed camera frustums among the step cameras, drawn on the side render).

- [ ] **Step 1: write the unit tests** (pose geometry only, no V)

```python
import numpy as np
from inspection.investigation.iwm.e05_parts_in_v import views as Vw

def test_pose_looks_along_minus_normal_and_is_right_handed():
    T = Vw.pose_for(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), 0.1)
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.4]) and np.allclose(T[:3, 2], [0, 0, -1])
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)

def test_standoff_clamps():
    assert Vw.standoff(np.array([0.005, 0.005, 0.001])) == Vw.D_MIN
    assert Vw.D_MIN < Vw.standoff(np.array([0.2, 0.1, 0.01])) < Vw.D_MAX
    assert Vw.standoff(np.array([2.0, 2.0, 0.1])) == Vw.D_MAX
```

- [ ] **Step 2: run** — ImportError. **Step 3: write `views.py`.** **Step 4: tests PASS.**
- [ ] **Step 5: run** on both runs at 1 mm; read `views.json` against H5 (computed ≥ step for > 80 %?), the eight figures, the pivot ring, the sets.

### Task 8: viewer

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/viewer.py` (port 8083), starting from `e04_voxel_scene/viewer.py` (meshes per size drawn with both windings, frustums, "look from step" per client, click → `RaycastingScene`).

- [ ] **Step 1:** mesh vertex colours = palette[label of `rows_near(vertex)`] (deepest), grey where −1; GUI: run, mm, **level slider** (recolour by the ancestor at that depth), **view filter** dropdown (all / one view: dim parts without a detection from it), **search** text + button (hits listed as buttons; clicking one selects the part), **set** multi-select (`add_gui_multi_select` if present in viser 1.1.1, else checkboxes) + "view for set" button.
- [ ] **Step 2:** click → nearest row → `part_of` → right panel: name, path, detections (view ids), the source frame crop with the mask (e03 style), the part's voxels highlighted (point cloud of its row centres), its `view_for` best pose as a frustum; "look from proposed" moves `event.client.camera`.
- [ ] **Step 3:** launch `nohup ~/miniconda3/envs/robo/bin/python -u -m inspection.investigation.iwm.e05_parts_in_v.viewer --port 8083 > <scratchpad>/viewer_e05.log &`, check the log, and screenshot via the static label renders (`evidence/labels_<run>_<mm>mm.jpg`: each capture pose + side + top with legend) since browser automation is unreliable here.

### Task 9: report and README

**Files:**
- Create: `inspection/investigation/iwm/e05_parts_in_v/REPORT.md`
- Modify: `inspection/investigation/iwm/README.md` (the e05 row: status running, the answer line)
- Modify: `inspection/investigation/iwm/e04_voxel_scene/normals/REPORT.md` (rename "M5 own face only" → "region fit, parent-face points only", promised earlier)

- [ ] **Step 1:** REPORT.md per the IWM template: Answer (RQ1–RQ6 in six lines), What was run, What we saw (figures), Numbers (tables read from `evidence/*.json`), Limits, What changes; plus "Assumed answers" copied from the NOTE for alignment round v2.
- [ ] **Step 2:** README row; normals rename.
- [ ] **Step 3:** memory: save the durable knowledge (what held, what failed, quirks found) via `mem.py` — not progress.

## Self-review

- Spec coverage: stages 1–8 → Tasks 3, 2, 5, 6, (search in 2 + evidence in 7's run via `search` queries → add to Task 7's script: `evidence/queries.json` with the 15 queries and precision/recall), 7, 7, 8; acceptance criteria AC-1…AC-9 → Tasks 3, 2/6, 5, 6, 7, 7, 7, 8, all.
- Types: `rows_by_node: dict[int, np.ndarray]` everywhere; `view` dict keys as in Task 1; `decisions` tuples as in Task 6.
- Review Focus 1–5 have tests in Tasks 3, 3, 2, 6, 3.
