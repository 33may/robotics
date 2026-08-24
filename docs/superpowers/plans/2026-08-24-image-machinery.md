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

**Reference dataset:** `data/runs/2408-seeded` (Anton, 2026-08-24) — the first
real sweep: 26 turns, 25 captures, survey + **24 of 36 cells**, 58 MB, one
object. Stage 2+ exit gates use it; `2108-d` (4 cells) stays as the small
fixture in `test_eyes_replay.py`. It validates the join rule on real data:
`pose_id 3` has no dir (step 4 was a software stop on cell `[10, 0]`), and the
retry at step 5 landed in dir `004` — a stopped turn consumes an ordinal and
leaves no directory. Every dir's name equals its `meta.json:pose_id`; rgb is
848×480 as assumed.

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

## Stage 2 — Geometry tool API

**Decision (Anton, 2026-08-24):** the vocabulary both AI tiers need — index
math, visited bitmap + ASCII, egocentric gloss, action listing — lives in a
new light `inspection/view/grid.py`, NOT in `eyes/` and not in a new `state/`
package (`RunStore` is already the state; this is a *language over the grid*).
`viewsphere.py` imports its constants back from it. Legality of a move comes
from motion, so `grid.py` takes the feasible set as an argument and never
imports motion/IK.

### Task 5: `view/grid.py` — the shared grid vocabulary

**Files:**
- Create: `inspection/view/grid.py`
- Modify: `inspection/view/viewsphere.py:37-39` (constants → import from grid)
- Modify: `inspection/run/decider.py:19,57-79` (drop `_dh`/`gloss`, import them)
- Test: `inspection/tests/test_grid.py`

**Interfaces:**
- Produces: `H_BINS=12`, `V_ELEVATIONS=(10.0, 40.0, 70.0)`, `DEFAULT_R=0.35`;
  `step_delta(cur_h, h, h_bins=H_BINS) -> int` (signed shortest azimuth steps,
  `+1` = one step right); `cell_gloss(cur, cell, h_bins=H_BINS,
  elevations=V_ELEVATIONS) -> str` (wording identical to today's
  `decider.gloss`); `neighbors(cell, h_bins=H_BINS, n_v=3) -> list[tuple]`
  (4-connected: h wraps, v clamps, excludes self); `coverage_map(cov, cur=None,
  elevations=V_ELEVATIONS) -> str`; `moves_from(cur, allowed, ...) -> list[Move]`
  where `Move` is a frozen dataclass `(cell: tuple, gloss: str)`, nearest first.
- Consumed by: `eyes/tools.py` (Task 6) and the MAY-187 `AgentDecider`.
- `run/decider.py` keeps re-exporting the name `gloss` so
  `tests/test_decider.py` passes untouched — that test is the refactor's net.

- [x] **Step 1: Write the failing test** — `inspection/tests/test_grid.py`

```python
#!/usr/bin/env python3
"""Grid vocabulary — index math, gloss, coverage ASCII.
Run: p inspection/tests/test_grid.py"""
import sys
from pathlib import Path

import numpy as np

from inspection.view.grid import (H_BINS, V_ELEVATIONS, cell_gloss,
                                  coverage_map, moves_from, neighbors,
                                  step_delta)


def test_step_delta_wraps():
    assert step_delta(5, 6) == 1
    assert step_delta(11, 0) == 1            # wraps forward past h=0
    assert step_delta(0, 11) == -1
    assert abs(step_delta(5, 11)) == H_BINS // 2      # opposite side


def test_cell_gloss_matches_decider_wording():
    assert cell_gloss((5, 1), (6, 1)) == "one step right, same height"
    assert cell_gloss((5, 1), (3, 1)) == "two steps left, same height"
    assert cell_gloss((11, 0), (0, 0)) == "one step right, same height"
    assert cell_gloss((5, 1), (11, 2)) == "opposite side, higher"
    assert cell_gloss((5, 1), (5, 0)) == "same side, lower"
    assert cell_gloss(None, (3, 2)) == "elevation 70 deg"


def test_neighbors_wrap_in_h_clamp_in_v():
    assert set(neighbors((0, 0))) == {(11, 0), (1, 0), (0, 1)}   # v floor
    assert len(neighbors((0, 1))) == 4
    assert set(neighbors((0, 2))) == {(11, 2), (1, 2), (0, 1)}   # v ceiling
    assert (0, 0) not in neighbors((0, 0))


def test_coverage_map_ascii():
    cov = np.zeros((3, 12), dtype=bool)
    cov[0, 2] = True; cov[1, 3] = True
    txt = coverage_map(cov, cur=(3, 1))
    assert txt.count("#") == 1 and txt.count("@") == 1   # cur overrides its mark
    assert txt.splitlines()[0].startswith("v2")          # highest elevation first


def test_moves_from_nearest_first():
    mv = moves_from((5, 1), {(6, 1), (4, 1), (5, 2), (11, 2)})
    assert mv[0].cell == (4, 1)                  # tie at distance 1, tuple order
    assert mv[0].gloss == "one step left, same height"
    assert mv[-1].cell == (11, 2)                # farthest last


def test_grid_is_light():
    import inspection.view.grid as g
    src = Path(g.__file__).read_text()
    assert "inspection.motion" not in src and "inspection.cell" not in src
    assert "pinocchio" not in sys.modules        # importing grid must stay cheap


def main():
    test_step_delta_wraps(); test_cell_gloss_matches_decider_wording()
    test_neighbors_wrap_in_h_clamp_in_v(); test_coverage_map_ascii()
    test_moves_from_nearest_first(); test_grid_is_light()
    print("OK test_grid")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run — must fail** (`ModuleNotFoundError: inspection.view.grid`).
  Run: `p inspection/tests/test_grid.py`
- [x] **Step 3: Implement `inspection/view/grid.py`**

```python
#!/usr/bin/env python3
"""Grid vocabulary for the viewsphere — index math and the words for it.

The light half of the viewsphere: cell addressing (h wraps, v clamps), the
egocentric gloss both AI tiers speak, and the coverage bitmap's rendering.
numpy only — importing this must NEVER pull in IK/motion/pinocchio, because
the image tier (eyes/) and the decider both live on it (Anton 2026-08-24).

Move legality comes from motion (`viewsphere.reachability`), so `moves_from`
takes the feasible set as an argument rather than computing it.
"""
from dataclasses import dataclass

import numpy as np

H_BINS = 12                          # 30 deg each, h=0 faces the robot base
V_ELEVATIONS = (10.0, 40.0, 70.0)    # deg above the table (Anton 2026-08-18)
DEFAULT_R = 0.35                     # camera-to-center distance

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


@dataclass(frozen=True)
class Move:
    cell: tuple
    gloss: str


def step_delta(cur_h, h, h_bins=H_BINS):
    """Signed shortest azimuth steps cur -> h; +1 = one step right."""
    return (h - cur_h + h_bins // 2) % h_bins - h_bins // 2


def cell_gloss(cur, cell, h_bins=H_BINS, elevations=V_ELEVATIONS):
    """Egocentric label for cell relative to cur (D4 menu gloss)."""
    if cur is None:
        return f"elevation {elevations[cell[1]]:.0f} deg"
    dh = step_delta(cur[0], cell[0], h_bins)
    dv = cell[1] - cur[1]
    if abs(dh) == h_bins // 2:
        side = "opposite side"
    elif dh == 0:
        side = "same side"
    else:
        n = abs(dh)
        side = f"{_WORDS[n]} step{'s' if n > 1 else ''} " \
               f"{'right' if dh > 0 else 'left'}"
    height = "same height" if dv == 0 else ("higher" if dv > 0 else "lower")
    return f"{side}, {height}"


def neighbors(cell, h_bins=H_BINS, n_v=len(V_ELEVATIONS)):
    """4-connected neighbours: h wraps around the ring, v clamps at the ends."""
    h, v = cell
    out = [((h - 1) % h_bins, v), ((h + 1) % h_bins, v)]
    if v - 1 >= 0:
        out.append((h, v - 1))
    if v + 1 < n_v:
        out.append((h, v + 1))
    return out


def coverage_map(cov, cur=None, elevations=V_ELEVATIONS):
    """ASCII bitmap of a (n_v, h_bins) coverage array, highest ring first."""
    cov = np.asarray(cov, dtype=bool)
    n_v, n_h = cov.shape
    rows = []
    for v in range(n_v - 1, -1, -1):
        line = ""
        for h in range(n_h):
            if cur is not None and (int(cur[0]), int(cur[1])) == (h, v):
                line += "@"
            else:
                line += "#" if cov[v, h] else "."
        rows.append(f"v{v} {elevations[v]:>2.0f}deg |{line}|")
    rows.append(" " * 9 + "h" + "".join(str(h % 10) for h in range(n_h)))
    return "\n".join(rows)


def moves_from(cur, allowed, h_bins=H_BINS, elevations=V_ELEVATIONS):
    """Describe an allowed cell set from cur, nearest first. Feasibility is
    the caller's business — this only orders and names the options."""
    def dist(cell):
        if cur is None:
            return (0, cell[0], cell[1])
        return (abs(step_delta(cur[0], cell[0], h_bins)) + abs(cell[1] - cur[1]),
                cell[0], cell[1])

    return [Move(tuple(c), cell_gloss(cur, c, h_bins, elevations))
            for c in sorted(allowed, key=dist)]
```

- [x] **Step 4: Run test — must pass.** `p inspection/tests/test_grid.py`
- [x] **Step 5: Point the old homes at it.** In `viewsphere.py`, delete the
  three constant definitions and import them instead (keeps
  `from inspection.view.viewsphere import H_BINS` working everywhere):

```python
from inspection.view.grid import H_BINS, V_ELEVATIONS, DEFAULT_R
```

  In `decider.py`, delete `_WORDS`, `_dh` and `gloss` and import replacements
  (the module keeps exporting the name `gloss`, so `test_decider.py` is
  untouched):

```python
from inspection.view.grid import (H_BINS, V_ELEVATIONS, cell_gloss as gloss,
                                  step_delta as _dh)
```

- [x] **Step 6: Run the neighbours' tests — must still pass.**
  `p inspection/tests/test_decider.py` and `p inspection/tests/test_machine.py`
- [x] **Step 7: Commit** — `view: grid.py — the shared cell vocabulary, split out of the viewsphere`

### Task 6: `eyes/tools.py` — the orchestrator's verbs

**Files:**
- Create: `inspection/eyes/tools.py`
- Test: `inspection/tests/test_eyes_tools.py`

**Interfaces:**
- Consumes: `RunStore`/`ViewRecord` (Task 1), the writers (Task 2),
  `view/grid.py` (Task 5).
- Produces: `ViewTools(store, writer=None)` with `view_at(cell)`,
  `views_near(cell)`, `get_view(target)`, `crop(img, box)`, `coverage()`,
  `note(text, cell=None)`; `ViewImage` frozen dataclass
  `(cell, cap_dir, rgb: np.ndarray, text: str, box: tuple | None)`.
- Stage 3 hangs detect/segment/OCR off `ViewImage`; Stage 4's subagent is
  handed a `ViewTools` bound to a `FindingWriter`.

**Contracts fixed here:**
- `view_at` returns the **newest** view of a cell (`store.views(cell)` returns
  all, oldest first) — a retry or revisit supersedes the earlier look.
- `views_near` = `grid.neighbors` (4-connected, h wraps, v clamps). Captured
  neighbours come back as records; uncaptured ones come back as `None` **and**
  write a miss note through `writer` when one is bound. That note is a
  next-move hint that cost no motion — the design's whole reason for the verb.
- `get_view` reads `cap_dir/rgb.png` with cv2 and converts **BGR→RGB**
  (`perception/capture.py:44` writes it through `RGB2BGR`; not inverting here
  hands the VLM colour-swapped images). cv2 is imported inside the function so
  the module stays importable and cheap for pixel-free callers.
- Both pixels and text on every return (design: ship both, decide by
  measurement). Text is deterministic provenance built from grid + store —
  never a model's words.

- [x] **Step 1: Write the failing test** — `inspection/tests/test_eyes_tools.py`

```python
#!/usr/bin/env python3
"""Verb surface over the run structure. Run: p inspection/tests/test_eyes_tools.py"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.store import FactWriter, FindingWriter, RunStore
from inspection.eyes.tools import ViewImage, ViewTools

T = np.eye(4)


def _rig(tmp):
    """Store with two views of (3,1) — a revisit — plus a neighbour at (4,1)."""
    root = Path(tmp)
    store = RunStore.create(root, h_bins=12, v_elevs=(10.0, 40.0, 70.0), r=0.35)
    facts = FactWriter(store)
    for pid, (cell, name) in enumerate([((3, 1), "001"), ((4, 1), "002"),
                                        ((3, 1), "003")], start=1):
        d = root / name; d.mkdir()
        img = np.zeros((480, 848, 3), np.uint8)
        img[:, :, 0] = pid * 10                      # RED channel marks the dir
        import cv2
        cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        facts.add_view(cell=cell, pose_id=pid, cap_dir=name, T_base_cam=T,
                       t=float(pid))
    return store


def test_view_at_returns_newest():
    with tempfile.TemporaryDirectory() as tmp:
        tools = ViewTools(_rig(tmp))
        assert len(tools._store.views(cell=(3, 1))) == 2      # both kept
        assert tools.view_at((3, 1)).cap_dir == "003"         # newest wins
        assert tools.view_at((9, 0)) is None                  # never captured


def test_views_near_hits_and_logs_misses():
    with tempfile.TemporaryDirectory() as tmp:
        store = _rig(tmp)
        tools = ViewTools(store, writer=FindingWriter(store))
        near = dict(tools.views_near((3, 1)))
        assert set(near) == {(2, 1), (4, 1), (3, 0), (3, 2)}  # 4-connected
        assert near[(4, 1)].cap_dir == "002"                  # captured
        assert near[(2, 1)] is None                           # not captured
        misses = [n for n in store.notes() if "not captured" in n["text"]]
        assert len(misses) == 3 and misses[0]["who"] == "finding"


def test_get_view_pixels_are_rgb_not_bgr():
    with tempfile.TemporaryDirectory() as tmp:
        img = ViewTools(_rig(tmp)).get_view((4, 1))
        assert isinstance(img, ViewImage) and img.rgb.shape == (480, 848, 3)
        assert img.rgb[0, 0, 0] == 20 and img.rgb[0, 0, 2] == 0   # RED, not blue
        assert "cell [4, 1]" in img.text and "dir 002" in img.text


def test_crop_clips_and_records():
    with tempfile.TemporaryDirectory() as tmp:
        tools = ViewTools(_rig(tmp))
        c = tools.crop(tools.get_view((4, 1)), (800, 400, 900, 500))  # over edge
        assert c.rgb.shape == (80, 48, 3)          # clipped to 848x480
        assert c.box == (800, 400, 848, 480) and "crop" in c.text


def test_coverage_text():
    with tempfile.TemporaryDirectory() as tmp:
        txt = ViewTools(_rig(tmp)).coverage(cur=(3, 1))
        assert "@" in txt and txt.count("#") == 1  # (4,1) seen, (3,1) is cur
        assert "2/36" in txt


def test_seeded_run_if_present():
    real = Path("inspection/data/runs/2408-seeded")
    if not (real / "run.json").exists():
        print("  (skip: 2408-seeded not present)"); return
    from inspection.eyes.replay import load_run
    tools = ViewTools(load_run(real))
    assert len(tools._store.views()) == 25 and len(tools._store.visited()) == 24
    cell = sorted(tools._store.visited())[0]
    assert tools.get_view(cell).rgb.shape == (480, 848, 3)
    print(tools.coverage(cur=cell))


def main():
    test_view_at_returns_newest(); test_views_near_hits_and_logs_misses()
    test_get_view_pixels_are_rgb_not_bgr(); test_crop_clips_and_records()
    test_coverage_text(); test_seeded_run_if_present()
    print("OK test_eyes_tools")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run — must fail** (`ModuleNotFoundError: inspection.eyes.tools`).
- [x] **Step 3: Implement `inspection/eyes/tools.py`** — `ViewImage` dataclass
  plus `ViewTools` as specified above; text gloss built from
  `grid.cell_gloss`-style facts (cell, azimuth degrees from `h`, elevation from
  `v_elevs`, radius, dir), coverage text from `grid.coverage_map` plus an
  `N/36 cells seen` line and a legend.
- [x] **Step 4: Run test — must pass** (seeded smoke prints the real bitmap).
- [x] **Step 5: Commit** — `eyes: the verb surface — views, neighbours, pixels, coverage`

**Stage 2 exit gate:** `test_eyes_tools.py` green including the `2408-seeded`
smoke; the printed coverage bitmap shows 24 of 36 cells; misses logged as notes.

**A cell may hold more than one view.** A revisit or a retry after a failed
capture can produce two dirs for the same `[h, v]` (in `2408-seeded` the retry
of `[10, 0]` produced only one, because the stopped turn never wrote a dir —
but the store must not assume that). Contract: `store.views(cell)` returns
**all**, oldest first; `view_at(cell)` returns the **newest** — a later look
supersedes an earlier one at the same pose. Task 6 adds a store-level
regression test for the two-views-one-cell case, plus a `2408-seeded` smoke
assertion (25 views, 24 cells, no view for the un-captured `pose_id 3`).

## Stage 3 — Local tool models (detect / segment / OCR)

**Decisions (Anton, 2026-08-24):** the model stack installs **into `robo`**, not
a separate env — Stage 7 wires the machinery into the live loop, so the process
holding pinocchio/ur_rtde must import the verbs in-process. And **SAM 3 with the
HF gate accepted**, not the Apache fallback.

Research already settled (memory `perception/local-tool-models-detect-segment-ocr`,
2026-08-21) — do not re-litigate:
- `facebook/sam3` is ONE checkpoint for detect+segment, transformers-native.
- **API trap:** `Sam3Model.input_boxes` are *concept exemplars*, NOT "segment
  this box". Box→mask must go through `Sam3TrackerModel`.
- SAM 3 finds `"cup"`, not `"the logo on the cup"` — simple noun phrases only.
  The pipeline is therefore **detect object → crop → OCR reads the logo**.
- Input is resized to square 1008, so our 848×480 is *upscaled* — helps small
  logos, costs latency. If <500 ms is needed: `config.image_size` 1008→560.
- OCR: **PP-OCRv6_medium via `engine="transformers"`**, DB detector set to
  **polygon** output (quad boxes lose on a curved cup). Never the native paddle
  engine on Blackwell — `cudaErrorLaunchFailure` after several images.
- Blackwell: verify with `get_device_capability() == (12, 0)` **plus a real
  matmul**, not `is_available()` — CUDA 12.8/12.9 shipped without
  `libnvptxcompiler.so`, so runtime PTX JIT can fail while AOT works.

Env baseline recorded before the install: torch absent, transformers absent,
numpy 2.4.6, open3d 0.19.0, pin 4.0.0, ur-rtde 1.6.5, 152 packages,
RTX 5090 / driver 580.178.04 / 32 GB, 586 GB free.

### Task 7: the environment and its verifier

**Anton runs** (the install and the gate need his account; snapshot first so a
bad interaction with the robot stack is one `pip install -r` away from undone):

```bash
pip freeze > ~/robo-before-torch.txt          # rollback point
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5.15" accelerate
hf auth login                                  # after accepting the terms at
                                               # huggingface.co/facebook/sam3
```

**Files:**
- Create: `inspection/eyes/env_check.py` (script, not a test — it needs the GPU)

**Interfaces:**
- Produces: `check_env() -> dict` with keys `torch`, `cuda`, `capability`,
  `matmul_ok`, `transformers`, `hf_token`, `sam3_available`; CLI
  `p inspection/eyes/env_check.py` printing a pass/fail line each.

- [ ] **Step 1: Write `env_check.py`** — every check is one the memory says has
  actually bitten someone: capability must be `(12, 0)`; a real bf16 matmul must
  run on device (PTX JIT gotcha); `transformers >= 5.0` for `Sam3Model`; token
  present; `facebook/sam3` config fetchable (proves the gate was accepted).
- [ ] **Step 2: Run it** — `p inspection/eyes/env_check.py`. Expected before the
  install: every line FAIL but no traceback. After: all PASS.
- [ ] **Step 3: Commit** — `eyes: environment verifier for the local verb stack`

### Task 8: `eyes/verbs_local.py` — detect + segment behind a backend seam

**Files:**
- Create: `inspection/eyes/verbs_local.py`
- Test: `inspection/tests/test_eyes_verbs.py`

**Interfaces:**
- Consumes: `ViewImage` from `eyes/tools.py`.
- Produces: `Detection` frozen dataclass `(box: tuple[int,int,int,int], score:
  float, label: str)`; `Segment` frozen dataclass `(mask: np.ndarray bool
  (H,W), box, score)`; `LocalVerbs(backend)` with `detect(img: ViewImage,
  phrase: str, min_score=0.3) -> list[Detection]` (highest score first, boxes
  clipped to frame) and `segment(img: ViewImage, box) -> Segment`.
- Backends: `StubBackend` (deterministic, no torch — what the tests use) and
  `Sam3Backend(device="cuda", dtype="bfloat16")` (lazy-loads both checkpoints:
  `Sam3Model` for text→boxes, `Sam3TrackerModel` for box→mask, because
  `input_boxes` are exemplars).
- Stage 4's subagent gets `LocalVerbs` alongside its `ViewTools`.

- [ ] **Step 1: Write the failing test** — stub-backed, so it runs with no GPU:

```python
#!/usr/bin/env python3
"""Local verbs over a stub backend. Run: p inspection/tests/test_eyes_verbs.py"""
import numpy as np

from inspection.eyes.tools import ViewImage
from inspection.eyes.verbs_local import (Detection, LocalVerbs, Segment,
                                         StubBackend)

IMG = ViewImage(cell=(3, 1), cap_dir="003",
                rgb=np.zeros((480, 848, 3), np.uint8), text="cell [3, 1]")


def test_detect_sorts_and_clips():
    backend = StubBackend(boxes=[((10, 10, 100, 100), 0.4, "cup"),
                                 ((-20, 5, 900, 500), 0.9, "cup")])
    dets = LocalVerbs(backend).detect(IMG, "cup")
    assert [d.score for d in dets] == [0.9, 0.4]          # highest first
    assert dets[0].box == (0, 5, 848, 480)                # clipped to frame
    assert isinstance(dets[0], Detection) and dets[0].label == "cup"


def test_detect_drops_low_scores():
    backend = StubBackend(boxes=[((0, 0, 10, 10), 0.1, "cup")])
    assert LocalVerbs(backend).detect(IMG, "cup", min_score=0.3) == []


def test_segment_returns_mask_of_frame_size():
    seg = LocalVerbs(StubBackend()).segment(IMG, (10, 10, 100, 100))
    assert isinstance(seg, Segment)
    assert seg.mask.shape == (480, 848) and seg.mask.dtype == bool
    assert seg.mask[50, 50] and not seg.mask[400, 700]     # inside vs outside


def test_module_has_no_torch_at_import():
    import sys
    assert "torch" not in sys.modules      # backends load lazily; stub needs none


def main():
    test_detect_sorts_and_clips(); test_detect_drops_low_scores()
    test_segment_returns_mask_of_frame_size(); test_module_has_no_torch_at_import()
    print("OK test_eyes_verbs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run — must fail** (`ModuleNotFoundError: inspection.eyes.verbs_local`).
- [ ] **Step 3: Implement `verbs_local.py`** — `LocalVerbs` owns the clipping,
  score filter and ordering (backend-independent policy, testable with no GPU);
  `Sam3Backend` owns only model I/O and imports torch inside its methods.
- [ ] **Step 4: Run test — must pass.**
- [ ] **Step 5: Real-weights smoke** (needs Task 7 green) — CLI
  `p inspection/eyes/verbs_local.py inspection/data/runs/2408-seeded 3 1`:
  detect `"cup"` on that cell, print boxes+scores and the segment's mask area,
  and **measure latency** (memory: claims diverge 40×, ours is unverified).
- [ ] **Step 6: Commit** — `eyes: detect and segment verbs behind a backend seam`

### Task 9: `read_text` (PP-OCRv6)

Detailed when reached. Shape: `read_text(img) -> list[TextLine]` with
`TextLine(text, score, polygon)`; polygon output, not quads. Installs
`paddleocr` and runs it through `engine="transformers"`. The chain the design
implies is `detect("cup") → crop → read_text`, so the test asserts a crop of a
`ViewImage` survives into OCR with its provenance text intact.

**Stage 3 exit gate:** `env_check.py` all-PASS; `test_eyes_verbs.py` green on
the stub; the real-weights CLI finds a cup in a `2408-seeded` view and prints a
measured latency; OCR reads text off a crop.

## Stage 4–7 stubs (detailed when reached)
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
