# Image Machinery (MAY-186) Implementation Plan

> **Execution mode (Anton, 2026-08-24):** single-agent inline TDD in-session
> (superpowers:executing-plans), with running commentary so the build is fully
> observable. No subagent dispatch. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the eyes of the inspection system — the run structure, tool
API, and two-tier agent machinery that answer questions from captured views
(design: `inspection/2026-08-20-image-machinery-design.md`).

**Architecture:** A deterministic view store replayed from captured runs, a
small geometry tool API on top, local perception verbs (detect/segment/OCR),
then a cloud-VLM inspection subagent and an orchestrator agent. Two tiers so
the orchestrator never accumulates raw pixels.

**Tech Stack:** Python 3.11 (conda `robo`), numpy, JSON-on-disk persistence.
Later stages: SAM 3 + PP-OCRv6 local on the 5090, cloud VLM APIs behind a thin
swappable interface.

## Global Constraints

- No question-specific logic in the module — the agent owns the question.
- No robot motion from the image tier, in any form. `inspection/eyes/` must
  never import from `inspection/motion/` or `inspection/run/` (enforced by a
  test).
- No model writes geometric ground truth (visited/poses/coverage) — enforced
  by writer-class construction, not convention.
- No pruning/summarisation/context management in v1. Runs are 5–30 steps.
- Images used as captured, 848×480. No second high-res capture path in v1.
- Absolute imports everywhere; repo is pip-installed editable. No sys.path hacks.
- Tests are plain assert functions + `main()` runner, run as
  `p inspection/tests/test_X.py` (no pytest). House style: `tests/synth.py`.
- Module data lives under `inspection/data/` — the store writes inside the
  run's own folder (`data/runs/<id>/eyes/`).
- Export names: two words, grep-unique (`RunStore`, `FactWriter`, …).
- Commit after every green task.

## The waterfall

| # | Stage | Delivers | Exit gate (no arm needed) |
|---|---|---|---|
| 1 | **Run structure / view store** | `eyes/store.py` + `eyes/replay.py`: three trust-level writers; loader replaying a captured `data/runs/` sweep | Real run loads; visited/coverage/poses queryable; model writers can't touch geometry |
| 2 | **Geometry tool API** | `view_at, views_near, get_view, crop, coverage, note`; neighbour-step with logged misses; pixels+text returns | Verbs exercised against the replayed dataset, deterministic tests |
| 3 | **Local tool models** | SAM 3 detect/segment + PP-OCRv6 as subagent verbs on the 5090 | Each verb runs standalone on captured images |
| 4 | **Inspection subagent** | `inspect(cell, task)`: cloud VLM behind thin model interface, verbs from #3, camera-centric prompt contract, verbatim transcript → store | One-view inspect on the dataset; transcript lands in the run structure |
| 5 | **Layer-1 readout bench** | Per-view GT labels + subagent scoring (accuracy, yes-bias, grounding) | Subagent model settled on evidence (ER-2 vs gemini flash/pro) |
| 6 | **Orchestrator agent** | Verbs + `inspect()`, injected pose/object context, answer policy (no vote-counting; coverage argues absence only) | Answers the logo question over a full replayed sweep |
| 7 | **Live wiring** | Machinery inside loop-v2 (`view/request`→`confirm` seam untouched) | Real run: capture → inspect → answer; MAY-187 handoff point |

Stages 2–7 get their own task breakdowns when reached (house rule: tasks
module-by-module, never all upfront). **Only Stage 1 is detailed below.**

---

## Stage 1 — Run structure / view store

The design's "main deliverable": the data structure that aggregates everything
the system knows, plus the loader that fills it from a captured run.

**Data layout on disk** (inside an existing run dir, e.g. `data/runs/2108-d/`):

```
data/runs/<id>/
  run.json            # written by the data engine (q_survey, r, turns[])
  000/ 001/ …         # captures: rgb.png, depth_*.npy, meta.json (pose_id, T_base_cam)
  eyes/
    store.json        # the run structure (this stage)
    transcripts/      # verbatim subagent transcripts (stage 4 fills)
```

**`store.json` schema:**

```json
{
  "grid": {"h_bins": 12, "v_elevs": [10.0, 40.0, 70.0], "r": 0.35},
  "views": [
    {"cell": [3, 1], "pose_id": 2, "cap_dir": "002", "t": 1787316180.16,
     "T_base_cam": [[…], […], […], [0, 0, 0, 1]]}
  ],
  "notes": [{"who": "plan", "cell": null, "text": "…", "t": 0.0}],
  "hypothesis": null,
  "plan": null,
  "findings": [
    {"cell": [3, 1], "summary": "…", "transcript": "transcripts/f001.json", "t": 0.0}
  ]
}
```

`cell` is `[h, v]`, or `null` for the survey view. Flush to disk on **every**
write (same rule as `run/machine.py:_save` — state on disk at all times).

**Capture↔cell join** (verified against `run/machine.py:566`): capture
`pose_id k` (k ≥ 1) is the k-th **non-survey** turn in `run.json` `turns[]`
order; `pose_id 0` / dir `000` is the survey. Turn `step` and `pose_id` are
deliberately different namespaces — do not conflate. A failed capture leaves
no dir; the loader trusts what is on disk.

### Task 1: `RunStore` + `FactWriter` (deterministic facts)

**Files:**
- Create: `inspection/eyes/__init__.py` (empty)
- Create: `inspection/eyes/store.py`
- Test: `inspection/tests/test_eyes_store.py`

**Interfaces (later tasks/stages rely on these exact names):**
- Produces: `RunStore.create(run_dir, h_bins, v_elevs, r)`,
  `RunStore.open(run_dir)`, `store.views(cell=None) -> list[ViewRecord]`,
  `store.visited() -> set[tuple]`, `store.coverage() -> np.ndarray` (bool,
  shape `(len(v_elevs), h_bins)`), `store.path` (the `eyes/` dir).
- `ViewRecord`: frozen dataclass — `cell: tuple | None`, `pose_id: int`,
  `cap_dir: str`, `t: float`, `T_base_cam: np.ndarray (4,4)`.
- `FactWriter(store).add_view(cell, pose_id, cap_dir, T_base_cam, t)`.

- [x] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""View store: facts tier. Run: p inspection/tests/test_eyes_store.py"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.store import RunStore, FactWriter, ViewRecord

T = np.eye(4); T[:3, 3] = [0.1, 0.2, 0.3]


def _fresh(tmp):
    return RunStore.create(Path(tmp), h_bins=12, v_elevs=(10.0, 40.0, 70.0), r=0.35)


def test_add_view_and_query():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh(tmp)
        facts = FactWriter(store)
        facts.add_view(cell=None, pose_id=0, cap_dir="000", T_base_cam=T, t=1.0)
        facts.add_view(cell=(3, 1), pose_id=1, cap_dir="001", T_base_cam=T, t=2.0)
        assert store.visited() == {(3, 1)}          # survey is not a cell
        assert len(store.views()) == 2
        v = store.views(cell=(3, 1))[0]
        assert isinstance(v, ViewRecord) and v.cap_dir == "001"
        assert v.T_base_cam.shape == (4, 4)
        cov = store.coverage()
        assert cov.shape == (3, 12) and cov[1, 3] and cov.sum() == 1


def test_flush_and_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh(tmp)
        FactWriter(store).add_view(cell=(0, 2), pose_id=1, cap_dir="001",
                                   T_base_cam=T, t=1.0)
        again = RunStore.open(Path(tmp))            # fresh object, disk only
        assert again.visited() == {(0, 2)}
        assert np.allclose(again.views()[0].T_base_cam, T)
        assert (Path(tmp) / "eyes" / "store.json").exists()


def test_no_motion_imports():
    import inspection.eyes.store as m
    src = Path(m.__file__).read_text()
    assert "inspection.motion" not in src and "inspection.run" not in src


def main():
    test_add_view_and_query(); test_flush_and_reopen(); test_no_motion_imports()
    print("OK test_eyes_store")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run it — must fail** with `ModuleNotFoundError: inspection.eyes`.
  Run: `p inspection/tests/test_eyes_store.py`
- [x] **Step 3: Implement minimal `store.py`**

```python
#!/usr/bin/env python3
"""Run structure for the image machinery — the facts tier.

Three trust levels (design 2026-08-20): deterministic code writes geometry
facts via FactWriter; agents get their own writers (Task 2) that CANNOT
touch views/visited/coverage. Flushed to eyes/store.json on every write.
Never imports motion/run — the image tier does not move the robot.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ViewRecord:
    cell: tuple | None
    pose_id: int
    cap_dir: str
    t: float
    T_base_cam: np.ndarray


class RunStore:
    """Owns the dict + disk flush. Read API only — writes go through writers."""

    def __init__(self, run_dir: Path, data: dict):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "eyes"
        self._d = data

    @classmethod
    def create(cls, run_dir, h_bins, v_elevs, r):
        d = {"grid": {"h_bins": h_bins, "v_elevs": list(v_elevs), "r": r},
             "views": [], "notes": [], "hypothesis": None, "plan": None,
             "findings": []}
        store = cls(run_dir, d)
        store._flush()
        return store

    @classmethod
    def open(cls, run_dir):
        p = Path(run_dir) / "eyes" / "store.json"
        return cls(run_dir, json.loads(p.read_text()))

    def _flush(self):
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "store.json").write_text(
            json.dumps(self._d, indent=1) + "\n")

    def views(self, cell=None):
        out = []
        for v in self._d["views"]:
            c = tuple(v["cell"]) if v["cell"] is not None else None
            if cell is None or c == tuple(cell):
                out.append(ViewRecord(c, v["pose_id"], v["cap_dir"], v["t"],
                                      np.asarray(v["T_base_cam"], float)))
        return out

    def visited(self):
        return {v.cell for v in self.views() if v.cell is not None}

    def coverage(self):
        g = self._d["grid"]
        cov = np.zeros((len(g["v_elevs"]), g["h_bins"]), dtype=bool)
        for (h, v) in self.visited():
            cov[v, h] = True
        return cov


class FactWriter:
    """Deterministic-code tier: the ONLY writer of geometric ground truth."""

    def __init__(self, store: RunStore):
        self._s = store

    def add_view(self, cell, pose_id, cap_dir, T_base_cam, t):
        self._s._d["views"].append(
            {"cell": list(cell) if cell is not None else None,
             "pose_id": int(pose_id), "cap_dir": str(cap_dir),
             "t": float(t),
             "T_base_cam": np.asarray(T_base_cam, float).tolist()})
        self._s._flush()
```

- [x] **Step 4: Run test — must pass.** `p inspection/tests/test_eyes_store.py`
- [x] **Step 5: Commit** — `eyes: RunStore + FactWriter — the facts tier of the run structure`

### Task 2: `PlanWriter` + `FindingWriter` (agent tiers, locked out of geometry)

**Files:**
- Modify: `inspection/eyes/store.py` (append two classes)
- Test: `inspection/tests/test_eyes_store.py` (append tests)

**Interfaces:**
- Consumes: `RunStore`, `FactWriter` from Task 1.
- Produces: `PlanWriter(store)` — `.set_plan(text)`, `.set_hypothesis(text)`,
  `.note(text, cell=None)`; `FindingWriter(store)` — `.add_finding(cell,
  summary, transcript_text)` (writes `eyes/transcripts/fNNN.json`, returns its
  relative path), `.note(text, cell=None)`. Store reads: `store.notes()`,
  `store.findings()`, `store.hypothesis`, `store.plan` (properties).

- [x] **Step 1: Append the failing tests**

```python
def test_agent_writers():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh(tmp)
        plan, finder = PlanWriter(store), FindingWriter(store)
        plan.set_hypothesis("logo likely on far wall")
        plan.set_plan("sweep h=3..5 at v=1")
        plan.note("view 12 shows a fragment at right edge")
        rel = finder.add_finding((3, 1), "partial logo, right edge",
                                 transcript_text='{"turns": []}')
        finder.note("wanted neighbour (4,1) — not captured", cell=(3, 1))
        again = RunStore.open(Path(tmp))
        assert again.hypothesis == "logo likely on far wall"
        assert again.plan.startswith("sweep")
        assert [n["who"] for n in again.notes()] == ["plan", "finding"]
        f = again.findings()[0]
        assert f["cell"] == [3, 1] and (Path(tmp) / "eyes" / rel).exists()


def test_trust_levels_by_construction():
    # The model-facing writers must PHYSICALLY lack geometry verbs.
    for cls in (PlanWriter, FindingWriter):
        api = {m for m in dir(cls) if not m.startswith("_")}
        assert "add_view" not in api, cls
    assert {m for m in dir(PlanWriter) if not m.startswith("_")} == \
        {"set_plan", "set_hypothesis", "note"}
    assert {m for m in dir(FindingWriter) if not m.startswith("_")} == \
        {"add_finding", "note"}
```

  (extend the import line with `PlanWriter, FindingWriter`; call both from `main()`)

- [x] **Step 2: Run — must fail** (`ImportError: PlanWriter`).
- [x] **Step 3: Implement** — append to `store.py`:

```python
class PlanWriter:
    """Orchestrator tier: reasoning state only. No geometry verbs exist here."""

    def __init__(self, store: RunStore):
        self._s = store

    def set_plan(self, text):
        self._s._d["plan"] = str(text); self._s._flush()

    def set_hypothesis(self, text):
        self._s._d["hypothesis"] = str(text); self._s._flush()

    def note(self, text, cell=None):
        _note(self._s, "plan", text, cell)


class FindingWriter:
    """Inspection-subagent tier: per-view findings + verbatim transcripts."""

    def __init__(self, store: RunStore):
        self._s = store

    def add_finding(self, cell, summary, transcript_text):
        tdir = self._s.path / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        rel = f"transcripts/f{len(self._s._d['findings']):03d}.json"
        (self._s.path / rel).write_text(transcript_text)
        self._s._d["findings"].append(
            {"cell": list(cell), "summary": str(summary), "transcript": rel,
             "t": time.time()})
        self._s._flush()
        return rel

    def note(self, text, cell=None):
        _note(self._s, "finding", text, cell)


def _note(store, who, text, cell):
    store._d["notes"].append(
        {"who": who, "cell": list(cell) if cell is not None else None,
         "text": str(text), "t": time.time()})
    store._flush()
```

  Plus on `RunStore`: `notes()` / `findings()` returning the lists,
  `hypothesis` / `plan` as `@property`; add `import time` at top.

- [x] **Step 4: Run test — must pass.**
- [x] **Step 5: Commit** — `eyes: agent writers — trust levels enforced by construction`

### Task 3: Replay loader — a captured run becomes a store

**Files:**
- Create: `inspection/eyes/replay.py`
- Test: `inspection/tests/test_eyes_replay.py`

**Interfaces:**
- Consumes: `RunStore`, `FactWriter` from Task 1.
- Produces: `load_run(run_dir, v_elevs=(10.0, 40.0, 70.0), h_bins=12)
  -> RunStore` — reads `run.json` + `NNN/meta.json`, fills the store via
  `FactWriter`, idempotent (re-load rebuilds `eyes/store.json` views from
  disk truth; agent-tier content is preserved). Stage 2's tools take this
  `RunStore` as their substrate.

**Join rule** (from `run/machine.py:566`): dir `000` = survey; dir `NNN`
(N ≥ 1) has `meta.json:pose_id == N`, and its cell is the N-th non-survey
entry of `run.json:turns[]` (`target` field, `[h, v]`). Trust dirs on disk;
skip turns without a dir (failed captures).

- [x] **Step 1: Write the failing test** — build a synthetic run dir, load it:

```python
#!/usr/bin/env python3
"""Replay a captured run into a RunStore. Run: p inspection/tests/test_eyes_replay.py"""
import json
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.replay import load_run
from inspection.eyes.store import PlanWriter, RunStore

T = np.eye(4)


def _synth_run(root):
    """run.json with survey + 3 cell turns (one failed -> no dir)."""
    turns = [{"step": 1, "target": "survey", "t": 1.0, "result": "ok"},
             {"step": 2, "target": [3, 1], "t": 2.0, "result": "+100 pts"},
             {"step": 3, "target": [4, 1], "t": 3.0,
              "result": "capture failed: no frames"},          # no dir 002
             {"step": 4, "target": [5, 2], "t": 4.0, "result": "+90 pts"}]
    (root / "run.json").write_text(json.dumps(
        {"q_survey": [0.0] * 6, "r": 0.35, "turns": turns}))
    for pose_id, name in [(0, "000"), (1, "001"), (3, "003")]:
        d = root / name; d.mkdir()
        (d / "meta.json").write_text(json.dumps(
            {"pose_id": pose_id, "timestamp": 10.0 + pose_id,
             "joints_rad": [0.0] * 6, "T_base_flange": T.tolist(),
             "T_base_cam": T.tolist()}))
        (d / "rgb.png").write_bytes(b"png")


def test_load_synthetic_run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _synth_run(root)
        store = load_run(root)
        assert store.visited() == {(3, 1), (5, 2)}    # failed cell absent
        assert [v.cell for v in store.views()] == [None, (3, 1), (5, 2)]
        assert store.views(cell=(5, 2))[0].cap_dir == "003"   # pose_id 3 = 3rd cell turn


def test_reload_preserves_agent_tiers():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _synth_run(root)
        PlanWriter(load_run(root)).set_hypothesis("h1")
        store = load_run(root)                         # idempotent re-load
        assert store.hypothesis == "h1"
        assert len(store.views()) == 3                 # not duplicated


def test_real_run_if_present():
    real = Path("inspection/data/runs/2108-d")
    if not (real / "run.json").exists():
        print("  (skip: 2108-d not present)"); return
    store = load_run(real)
    assert len(store.views()) == 5 and len(store.visited()) == 4
    assert store.views()[0].cell is None               # survey first


def main():
    test_load_synthetic_run(); test_reload_preserves_agent_tiers()
    test_real_run_if_present()
    print("OK test_eyes_replay")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run — must fail** (`ModuleNotFoundError: inspection.eyes.replay`).
- [x] **Step 3: Implement `replay.py`**

```python
#!/usr/bin/env python3
"""Replay a captured data-engine run into a RunStore.

Join rule (run/machine.py: turn step vs capture pose_id are different
namespaces): dir 000 = survey; dir NNN's cell is the N-th non-survey turn
of run.json. Disk is truth — turns without a dir (failed captures) are
skipped. Debug CLI: p inspection/eyes/replay.py <run_dir>
"""
import json
import sys
from pathlib import Path

from inspection.eyes.store import FactWriter, RunStore


def load_run(run_dir, v_elevs=(10.0, 40.0, 70.0), h_bins=12):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    cells = [t["target"] for t in run["turns"] if t["target"] != "survey"]

    if (run_dir / "eyes" / "store.json").exists():
        store = RunStore.open(run_dir)
        store._d["views"] = []                 # views: rebuilt from disk truth
    else:
        store = RunStore.create(run_dir, h_bins=h_bins, v_elevs=v_elevs,
                                r=run.get("r", 0.35))
    facts = FactWriter(store)
    for d in sorted(p for p in run_dir.iterdir()
                    if p.is_dir() and p.name.isdigit()):
        meta = json.loads((d / "meta.json").read_text())
        pid = meta["pose_id"]
        cell = None if pid == 0 else tuple(cells[pid - 1])
        facts.add_view(cell=cell, pose_id=pid, cap_dir=d.name,
                       T_base_cam=meta["T_base_cam"], t=meta["timestamp"])
    return store


if __name__ == "__main__":
    s = load_run(sys.argv[1])
    print(f"views {len(s.views())}  visited {sorted(s.visited())}")
    print(s.coverage().astype(int))
```

- [x] **Step 4: Run test — must pass** (real-run assert included: 2108-d has
  5 captures, 4 cells).
- [x] **Step 5: Commit** — `eyes: replay loader — captured runs become the store's substrate`

### Task 4: Module contract doc

**Files:**
- Create: `inspection/eyes/AGENTS.md` — house style, mirrors other modules:
  what the module is (the design doc, condensed), the three writers and who
  may hold each, the join rule, the never-imports-motion rule, how to run
  tests and the debug CLI.

- [x] **Step 1: Write it** (content distilled from this plan's Stage 1 header —
  no new decisions).
- [x] **Step 2: Commit** — `eyes: module contract (AGENTS.md)`

**Stage 1 exit gate:** both test files green; `p inspection/eyes/replay.py
inspection/data/runs/2108-d` prints 5 views / 4 cells and the coverage
bitmap; `PlanWriter`/`FindingWriter` API-locked out of geometry.

---

## Stage 2–7 stubs (detailed when reached)

- **Stage 2** — `eyes/tools.py`: verbs over `RunStore` + capture dirs.
  `get_view` resolves `cap_dir/rgb.png`; `views_near` walks the grid with
  wraparound in h, clamped v; uncaptured neighbour → `FindingWriter.note`
  miss + empty return. Pixels+text both.
- **Stage 3** — `eyes/verbs_local.py`: SAM 3 (detect+segment), PP-OCRv6
  (transformers engine, polygon output), crop chain. cu128 wheels, 5090 only.
- **Stage 4** — `eyes/inspect_agent.py` + `eyes/models.py` (thin swappable
  model interface): subagent run loop, camera-frame prompt rules,
  evidence→reasoning→answer JSON, verbatim transcript via `FindingWriter`.
- **Stage 5** — `eyes/bench/`: GT labels per view (logo visible: y/n/partial),
  layer-1 readout scoring. Settles the subagent model.
- **Stage 6** — `eyes/orchestrator.py`: agent harness, injected pose/object
  context, answer policy. Resolves open D8 (lifecycle) — Anton decides then.
- **Stage 7** — wire into `run/` behind the loop-v2 seam. MAY-187 starts here.
