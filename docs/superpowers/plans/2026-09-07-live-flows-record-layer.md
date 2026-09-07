# Live Flows Through the Record Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the live inspection loop (Flow A: AI run) and a new data-collection sweep (Flow B) through the record schema — one `Run` read object for the whole application, `RunWriter`/`AIRunWriter` as the only writers, verified end-to-end with headless playwright.

**Architecture:** The Supervisor state-machine core is untouched; we extract a settle pipeline and a Recorder seam around it. Supervisor threads own motion truth (`run.json`, `steps/**`, `session.json`); the brain thread owns `ai/**` via its own writer. All readers stand on a new `record/run.py:Run` object; `eyes/replay.py` dies.

**Tech Stack:** Python 3.11 (conda env `robo`), pydantic v2 (`inspection/record/schema.py`), pytest, porthole bus, React frontend (`inspection/ui`), playwright (headless chromium), FakeRig for all hardware-free tests.

**Spec:** `inspection/docs/data-engine/flows-design.md` (read it first; also `inspection/docs/data-engine/management-user-stories.md` for the stories referenced).

## Global Constraints

- Python: ALWAYS `~/miniconda3/envs/robo/bin/python` (Anton's alias `p`). Tests: `~/miniconda3/envs/robo/bin/python -m pytest inspection/tests/<file> -q`.
- Real archive `inspection/data/runs/**` is READ-ONLY. Tests write only under `tmp_path` or `/tmp`.
- NEVER command the UR5e or touch `192.168.2.50`. All integration tests use `FakeRig` / `ui/mock.py`.
- No `sys.path` hacks; imports are `from inspection. ...` run from the repo root `~/projects/robotics`.
- The 119+ existing tests in `inspection/tests/` must stay green after every task (`-m pytest inspection/tests -q`).
- Every commit ends with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Branch: `data-engine/record-layer` (already pushed; keep committing there).
- Schema models are the contract — extend via `inspection/record/schema.py` + a RED test in `inspection/tests/test_record_schema.py`, never with ad-hoc dicts.
- Run-dir layout produced by this plan:

```
<run>/run.json  config.json  session.json  events.jsonl  manifest.json
<run>/steps/NNN/{step.json, view_state.json, rgb.png, depth_aligned.npy, mask.png, chain_*.png}
<run>/fused/{cloud.npy, colors.npy}
<run>/ai/SSS/{airun.json, trace.jsonl, store.json, transcripts/tNNN.json,
              menu/{menu_def.json, inputs.jsonl}, answer.json}     # SSS = seq, 000 = live
```

---

### Task 1: `Run`/`Step` core — the read door (records only)

**Files:**
- Create: `inspection/record/run.py`
- Test: `inspection/tests/test_record_run.py`

**Interfaces:**
- Consumes: `inspection/record/schema.py` models; `inspection/tests/record_fixtures.py:make_run(root, run_id=..., object=...)` (existing fixture that writes a valid new-schema run via `RunWriter`).
- Produces (later tasks rely on these exact names):
  - `Run.load(run_dir: Path) -> Run` (raises `FileNotFoundError` if no `run.json`)
  - `run.record: RunRecord`, `run.path: Path`, `run.provenance: str` ("native"|"legacy")
  - `run.steps: list[Step]` (ordered by id), `run.step(step_id: int) -> Step` (KeyError if absent)
  - `run.survey: Step | None` (step 0), `run.captured: list[Step]` (outcome=="captured")
  - `run.at(address) -> Step | None` — latest captured step whose `view.address == list(address)`
  - `run.session: SessionRecord | None`, `run.config: ConfigSnapshot | None`
  - `run.events: list[OperatorEvent]` (parsed events.jsonl, [] if absent)
  - `run.fused() -> tuple[np.ndarray, np.ndarray | None] | None` (points, colors|None); reads `fused/cloud.npy` + `fused/colors.npy`, falls back to legacy `fused_cloud.npy`/`fused_colors.npy`
  - `run.validate(deep=False) -> Report` (delegates to `record.validate.validate_run`)
  - `run.derived(method: str, variant: str | None = None) -> Path | None` (existing dir under `inspection/data/derived/<id>/fits/<method>[ -<variant>]` or `derived/<id>/<method>`, else None)
  - `Step.id: int`, `Step.record: StepRecord`, `Step.dir: Path`, `Step.view_state: ViewState | None`, `Step.T_base_cam: np.ndarray | None` (4x4 float array)

- [ ] **Step 1: Write the failing tests**

```python
#!/usr/bin/env python3
"""Run — the one read door over a recorded run (flows-design.md §4)."""
import numpy as np
import pytest

from inspection.record.run import Run
from inspection.tests.record_fixtures import make_run


def test_load_exposes_records(tmp_path):
    make_run(tmp_path, run_id="0709-box1", object="box")
    run = Run.load(tmp_path / "0709-box1")
    assert run.record.id == "0709-box1"
    assert run.provenance == "native"
    assert run.session is not None and run.config is not None


def test_steps_and_survey(tmp_path):
    make_run(tmp_path, run_id="0709-box1")
    run = Run.load(tmp_path / "0709-box1")
    assert [s.id for s in run.steps] == run.record.steps
    assert run.survey is not None and run.survey.id == 0
    assert run.step(run.steps[-1].id).record.step_id == run.steps[-1].id
    with pytest.raises(KeyError):
        run.step(999)


def test_at_returns_latest_capture_for_address(tmp_path):
    make_run(tmp_path, run_id="0709-box1")   # fixture captures cell [3, 0]
    run = Run.load(tmp_path / "0709-box1")
    hit = run.at((3, 0))
    assert hit is not None and hit.record.view.address == [3, 0]
    assert run.at((9, 9)) is None


def test_T_base_cam_is_4x4_array(tmp_path):
    make_run(tmp_path, run_id="0709-box1")
    run = Run.load(tmp_path / "0709-box1")
    T = run.captured[0].T_base_cam
    assert isinstance(T, np.ndarray) and T.shape == (4, 4)


def test_fused_and_events_and_validate(tmp_path):
    make_run(tmp_path, run_id="0709-box1")
    run = Run.load(tmp_path / "0709-box1")
    pts, colors = run.fused() or (None, None)
    assert pts is None or pts.ndim == 2          # fixture may not write fused
    assert isinstance(run.events, list)
    assert run.validate().run_id == "0709-box1"


def test_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Run.load(tmp_path / "nope")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
```

Note: open `inspection/tests/record_fixtures.py` first and adjust the
assertions to what `make_run` actually writes (which cell it captures, whether
it writes `fused/`). If `make_run` captures a different cell than `[3, 0]`,
use that cell — do not change the fixture.

- [ ] **Step 2: Run tests — verify they fail** with `ModuleNotFoundError: inspection.record.run`.
- [ ] **Step 3: Implement `inspection/record/run.py`.** Skeleton:

```python
"""Run — the one read object over a recorded run (spec: flows-design.md §4).

Read-only and lazy: records validated eagerly (small JSON), binaries only on
call. Files are the truth; this object never writes a byte into the run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inspection.record.schema import (ConfigSnapshot, OperatorEvent, RunRecord,
                                      SessionRecord, StepRecord, ViewState)


class Step:
    def __init__(self, record: StepRecord, step_dir: Path):
        self.record, self.dir = record, step_dir
        self.id = record.step_id

    @property
    def T_base_cam(self):
        t = self.record.T_base_cam
        return None if t is None else np.asarray(t, float)

    @property
    def view_state(self):
        p = self.dir / "view_state.json"
        return ViewState.model_validate_json(p.read_text()) if p.exists() else None


class Run:
    def __init__(self, path: Path, record: RunRecord, steps: dict[int, Step],
                 provenance: str = "native"):
        self.path, self.record, self.provenance = path, record, provenance
        self._steps = steps

    @classmethod
    def load(cls, run_dir: Path) -> "Run":
        run_dir = Path(run_dir)
        rp = run_dir / "run.json"
        if not rp.exists():
            raise FileNotFoundError(rp)
        record = RunRecord.model_validate_json(rp.read_text())
        steps = {}
        for sid in record.steps:
            sp = run_dir / "steps" / f"{sid:03d}" / "step.json"
            if sp.exists():
                steps[sid] = Step(StepRecord.model_validate_json(sp.read_text()),
                                  sp.parent)
        return cls(run_dir, record, steps)

    # steps -----------------------------------------------------------------
    @property
    def steps(self):
        return [self._steps[i] for i in sorted(self._steps)]

    def step(self, step_id: int) -> Step:
        return self._steps[step_id]

    @property
    def survey(self):
        return self._steps.get(0)

    @property
    def captured(self):
        return [s for s in self.steps if s.record.outcome == "captured"]

    def at(self, address):
        addr = list(address)
        hits = [s for s in self.captured if s.record.view.address == addr]
        return hits[-1] if hits else None

    # run-level records -----------------------------------------------------
    # session/config/events as lazy properties over the obvious files;
    # fused() prefers fused/cloud.npy+colors.npy, falls back to
    # fused_cloud.npy/fused_colors.npy, returns None if neither exists,
    # colors only when row-aligned. validate() delegates to
    # record.validate.validate_run(self.path, deep=deep). derived() probes
    # inspection/data/derived/<id>/fits/<method> then derived/<id>/<method>.
```

Fill the elided properties exactly as described in Interfaces (they are
one-liners over `json.loads`/`np.load`).

- [ ] **Step 4: Run tests — verify they pass.** Also `-m pytest inspection/tests -q` (everything green).
- [ ] **Step 5: Commit** `record: the Run read object — records, steps, addresses`.

---

### Task 2: `Step` binaries + the one rotation policy

**Files:**
- Modify: `inspection/record/run.py`
- Test: `inspection/tests/test_record_run.py` (append)

**Interfaces:**
- Produces:
  - `Step.rgb(upright: bool = True) -> np.ndarray | None` — RGB order. When `upright=True` and `record.rgb_rotation_deg == 180` **and the stored png is raw**, rotate 180°. IMPORTANT: check the Conventions note in `inspection/record/schema.py` and the module docstring of `inspection/investigation/step_replay.py` first — post-2026-08-24 captures store rgb.png UPRIGHT already while the pose stays raw; `upright=False` must return the frame in the pose's RAW orientation (rotating BACK when the file is stored upright). Encode whichever polarity the Conventions record declares; the test below pins behavior via the fixture's conventions.
  - `Step.depth() -> np.ndarray | None` (`depth_aligned.npy`)
  - `Step.cloud() -> np.ndarray | None` (`cloud.ply` via open3d, else None)
  - `Step.mask() -> np.ndarray | None` (`mask.png` grayscale bool array, else None)

- [ ] **Step 1: Write the failing tests**

```python
def test_rgb_upright_and_raw_differ_only_when_rotated(tmp_path):
    make_run(tmp_path, run_id="0709-rot")
    run = Run.load(tmp_path / "0709-rot")
    step = run.captured[0]
    import cv2
    marked = np.zeros((4, 6, 3), np.uint8); marked[0, 0] = (255, 0, 0)
    cv2.imwrite(str(step.dir / "rgb.png"), cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
    up, raw = step.rgb(upright=True), step.rgb(upright=False)
    if step.record.rgb_rotation_deg == 180:
        assert not np.array_equal(up, raw)
        assert np.array_equal(np.rot90(up, 2), raw)
    else:
        assert np.array_equal(up, raw)


def test_depth_and_missing_binaries_are_none(tmp_path):
    make_run(tmp_path, run_id="0709-bin")
    step = Run.load(tmp_path / "0709-bin").captured[0]
    np.save(step.dir / "depth_aligned.npy", np.ones((4, 6), np.uint16))
    assert step.depth().shape == (4, 6)
    assert step.cloud() is None and step.mask() is None
```

- [ ] **Step 2: Run — fail** (`AttributeError: rgb`).
- [ ] **Step 3: Implement** the four loaders (cv2 for png, BGR→RGB; the rotation rule with a comment citing the convention source line).
- [ ] **Step 4: Run — pass**; whole suite green.
- [ ] **Step 5: Commit** `record: Step binary loaders — the one rotation policy lives here`.

---

### Task 3: legacy below the door + `refresh()`

**Files:**
- Modify: `inspection/record/run.py`
- Test: `inspection/tests/test_record_run.py` (append)

**Interfaces:**
- Consumes: `inspection/record/legacy.py:is_legacy(run_dir)`, `adapt_run(run_dir) -> AdaptedRun{run, steps: dict[int, StepRecord], path}`.
- Produces: `Run.load` on a pre-schema dir returns a `Run` with `provenance == "legacy"`, steps living in the OLD layout (`<run>/NNN/`, so `Step.dir` points there); `run.refresh() -> Run` (self, mutated) re-reads `run.json` and picks up steps written since load.

- [ ] **Step 1: Write the failing tests** (copy the `_legacy_run` helper from `inspection/tests/test_record_legacy.py` — do not import it, tests stay standalone):

```python
def test_legacy_run_loads_through_the_same_door(tmp_path):
    d = _legacy_run(tmp_path, "2408-old")        # helper copied from test_record_legacy
    run = Run.load(d)
    assert run.provenance == "legacy"
    assert run.survey is not None
    assert run.at((3, 0)) is not None            # the turns-join address
    assert run.captured[0].dir.name.isdigit()    # old layout: <run>/NNN/


def test_refresh_picks_up_new_steps(tmp_path):
    from inspection.record.writer import RunWriter
    from inspection.tests.record_fixtures import make_config
    w = RunWriter.create(tmp_path, run_id="0709-live", name="live",
                         source="live", config=make_config())
    run = Run.load(tmp_path / "0709-live")
    assert run.steps == []
    sid, sdir = w.begin_step({"method": "vs", "address": None})
    w.write_capture(sid, t_captured=1.0, joints_rad=[0.0] * 6,
                    T_base_flange=np.eye(4).tolist(),
                    T_base_cam=np.eye(4).tolist(), rgb_rotation_deg=0)
    run.refresh()
    assert [s.id for s in run.steps] == [0]
```

(If `record_fixtures` has no `make_config`, add one there returning a minimal
valid `ConfigSnapshot` — look at how `make_run` builds its config and extract.)

- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement.** In `Run.load`: `if is_legacy(run_dir): a = adapt_run(run_dir); steps = {sid: Step(rec, run_dir / f"{sid:03d}") for sid, rec in a.steps.items()}; return cls(run_dir, a.run, steps, provenance="legacy")`. `refresh()`: re-run the loading body against `self.path`, replace `self.record`/`self._steps`, return self. Legacy `view_state`/`session` degrade to None gracefully.
- [ ] **Step 4: Run — pass**; suite green.
- [ ] **Step 5: Commit** `record: Run reads legacy runs below the door; refresh() for mid-run readers`.

---

### Task 4: `run.ai` — the AIRun read side (plural from day one)

**Files:**
- Modify: `inspection/record/run.py`
- Test: `inspection/tests/test_record_run.py` (append)

**Interfaces:**
- Consumes: schema models `AIRunRecord`, `TranscriptRecord`, `AnswerRecord`, `MenuDef`, `MenuInput`.
- Produces:
  - `run.ai: list[AIRun]` — scans `ai/*/airun.json`, sorted by seq; `[]` when absent (legacy: also probe old `eyes/answer.json` into a synthetic read-only view is NOT done here — legacy AI stays reachable via `record/story.py`).
  - `AIRun.seq: int`, `AIRun.dir: Path`, `AIRun.record: AIRunRecord`
  - `AIRun.transcripts: list[TranscriptRecord]` (from `transcripts/t*.json`, sorted by t)
  - `AIRun.answer: AnswerRecord | None`, `AIRun.menu: MenuDef | None`, `AIRun.menu_inputs: list[MenuInput]` (from `menu/inputs.jsonl`)
  - `AIRun.trace_path: Path` (`dir / "trace.jsonl"`)

- [ ] **Step 1: Write the failing test** — build the `ai/000/` tree by hand with minimal valid model JSON (`AIRunRecord(...).model_dump_json()` etc. — open `inspection/record/schema.py` for required fields and construct real instances in the test), then:

```python
def test_ai_runs_scan_and_parse(tmp_path):
    make_run(tmp_path, run_id="0709-ai")
    _write_airun(tmp_path / "0709-ai" / "ai" / "000")   # helper in this test file
    run = Run.load(tmp_path / "0709-ai")
    assert len(run.ai) == 1
    a = run.ai[0]
    assert a.seq == 0 and a.record.orchestrator_model
    assert len(a.transcripts) == 1 and a.transcripts[0].step_id == 1
    assert a.answer is not None and a.menu is not None
```

- [ ] **Step 2: Run — fail.** **Step 3: Implement** (pure scanning + `model_validate_json`). **Step 4: pass + suite green.** **Step 5: Commit** `record: run.ai — AIRuns are plural from day one`.

---

### Task 5: port the brain's readers to `Run`; delete `eyes/replay.py`

**Files:**
- Modify: `inspection/eyes/tools.py` (ViewTools consumes `Run`), `inspection/brain/loop.py`, `inspection/brain/live.py` (`has_survey`, `SupervisorMover._captures`, `refresh_views`), `inspection/record/show.py` + `inspection/record/story.py` (switch to `Run.load`)
- Delete: `inspection/eyes/replay.py`
- Test: `inspection/tests/test_record_run_port.py`

**Interfaces:**
- Consumes: Task 1–4 `Run` API.
- Produces: `ViewTools(run: Run, writer=None)` — same public methods as today (`view_at(cell)`, `views_near`, `coverage(cur=None)`, `get_view(target)`, `crop`, `note`, `survey_bearing`); internally `store.views()` becomes `run.captured` (+ survey), a "view record" adapter keeps `.cell/.pose_id/.cap_dir/.T_base_cam/.t` names so `get_view`/`survey_bearing` bodies stay small diffs. `brain/live.py:refresh_views(run)` → `run.refresh()`. `has_survey(run_dir)` → `Run.load` + `run.survey is not None` (tolerating `FileNotFoundError` → False).

Key semantic mapping (write it as a comment in tools.py): old `RunStore` views were `{cell: None|tuple, pose_id, cap_dir, T_base_cam, t}`; new: `cell = None if step.id == 0 else tuple(step.record.view.address)`, `pose_id = step.id`, `cap_dir = step.dir` (a full Path now — old code joined `run_dir / cap_dir`; fix the join sites), `t = step.record.t_captured`. `RunStore` itself SURVIVES for now as the brain's mutable state (plan/hypothesis/findings) — only its `views` role moves to `Run`; `FactWriter` dies with replay.py.

- [ ] **Step 1: Write the failing tests**

```python
#!/usr/bin/env python3
"""The brain's readers stand on Run (flows-design §4 'consumers ported')."""
import numpy as np
import pytest

from inspection.record.run import Run
from inspection.tests.record_fixtures import make_run


def test_viewtools_over_run(tmp_path):
    make_run(tmp_path, run_id="0709-vt")
    run = Run.load(tmp_path / "0709-vt")
    from inspection.eyes.tools import ViewTools
    vt = ViewTools(run)
    cell = tuple(run.captured[-1].record.view.address or (0, 0))
    assert vt.view_at(cell) is not None
    assert vt.view_at((9, 9)) is None


def test_has_survey_over_run(tmp_path):
    from inspection.brain.live import has_survey
    assert has_survey(tmp_path / "absent") is False
    make_run(tmp_path, run_id="0709-hs")
    assert has_survey(tmp_path / "0709-hs") is True


def test_replay_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        import inspection.eyes.replay  # noqa: F401
```

- [ ] **Step 2: Run — fail** (ViewTools still wants a store; replay imports fine).
- [ ] **Step 3: Port.** Order: (a) ViewTools + the view-record adapter; (b) `brain/loop.py` `Brain.__init__` — `self.run = Run.load(run_dir)`, `self.tools = ViewTools(self.run, writer=self.plan_writer)`; `self.store` stays ONLY for plan/hypothesis/findings (`RunStore.open/create` without views); `move` verb: `refresh_views(self.store, ...)` → `self.run.refresh()`; `_counts`/`answer` sites that call `self.store.visited()` → `len({tuple(s.record.view.address) for s in self.run.captured if s.id != 0})`; (c) `brain/live.py` `has_survey`, `_captures` → `len(Run.load(...).captured)` under try/except → 0; (d) `record/show.py`/`story.py` swap `load_any` for `Run.load` where trivial; (e) `git rm inspection/eyes/replay.py`; grep the tree for `eyes.replay` / `eyes/replay` — every hit must die (`ui/mock.py` included if it imports it).
- [ ] **Step 4: Run — pass. Whole suite green** (this is the task most likely to break existing brain tests — fix call sites, never the semantics).
- [ ] **Step 5: Commit** `brain+record: all readers stand on Run; eyes/replay.py deleted`.

---

### Task 6: extract the settle pipeline (`run/settle.py`)

**Files:**
- Create: `inspection/run/settle.py`
- Modify: `inspection/run/machine.py:578-648` (`_exec_worker` capture leg)
- Test: `inspection/tests/test_settle.py`

**Interfaces:**
- Consumes: `inspection/run/segmenter.py:object_view(cap, intr, intr_color, depth_scale, points, segmenter, on_fallback)`, `machine._plane_error(plane)`, `CloudAccumulator.add`.
- Produces:

```python
@dataclass
class SettleResult:
    ok: bool
    detail: str            # "" on success; rejection/failure reason otherwise
    view: dict | None      # object_view's dict (points, plane, centroid, ...)
    seg: object | None     # segmenter result or None
    npts: int
    dropped: int

def settle_capture(cap, rig, segmenter, acc, is_survey: bool,
                   on_warn=lambda msg: None) -> SettleResult
```

Behavior (moved VERBATIM from `_exec_worker`, same order): `object_view` → `_plane_error` gate (reject before anything is kept) → survey-with-no-centroid rejection → `acc.add(points, colors)` with the detached-points warning through `on_warn`. NO publishing, NO writing, NO recenter — those stay in the machine.

- [ ] **Step 1: Write the failing tests** — drive `settle_capture` directly with a `FakeRig` capture:

```python
def _fake_cap(tmp_path):
    from inspection.run.rigs import FakeRig
    import threading
    rig = FakeRig(None, threading.Event(), tmp_path)   # match FakeRig.__init__ (read rigs.py:17)
    return rig, rig.capture(0)

def test_settle_accepts_a_good_capture(tmp_path):
    from inspection.cell.geometry import CloudAccumulator
    from inspection.run.settle import settle_capture
    rig, cap = _fake_cap(tmp_path)
    acc = CloudAccumulator()
    res = settle_capture(cap, rig, None, acc, is_survey=True)
    assert res.ok and res.npts > 0 and len(acc.points) > 0

def test_settle_rejects_bad_plane(tmp_path, monkeypatch):
    from inspection.cell.geometry import CloudAccumulator
    from inspection.run import settle
    rig, cap = _fake_cap(tmp_path)
    monkeypatch.setattr(settle, "_plane_error", lambda plane: "table at z=90mm")
    res = settle.settle_capture(cap, rig, None, CloudAccumulator(), is_survey=False)
    assert not res.ok and "rejected" in res.detail
```

(Adjust `FakeRig` construction to its real `__init__` signature — read `rigs.py:17-40` first. If the fake depth yields no object for `is_survey=True`, use `is_survey=False` in the accept test; the assertion that matters is ok+points-fused.)

- [ ] **Step 2: Run — fail.** **Step 3: Implement + rewire** — `_exec_worker` between the `"capturing"` phase event and the `settle_done` put becomes: call `settle_capture`, then on ok: `_recenter()`, reach refresh, `_publish_chain`, `publish_capture`, event; on not ok: put `settle_done ok=False detail=res.detail`. `_plane_error` stays in machine.py, imported by settle (or moved — pick ONE home, `settle.py`, and re-export from machine for its existing test if one exists).
- [ ] **Step 4: Run — pass; whole suite green** (any existing machine tests must pass unchanged — behavior parity is the point).
- [ ] **Step 5: Commit** `run: extract the settle pipeline from _exec_worker — no behavior change`.

---

### Task 7: writer verbs — schema additions + `AIRunWriter`

**Files:**
- Modify: `inspection/record/writer.py`, `inspection/record/schema.py` (only if a field is missing — RED test first)
- Create: `inspection/record/ai_writer.py`
- Test: `inspection/tests/test_record_writer.py` (append), `inspection/tests/test_ai_writer.py`

**Interfaces:**
- Produces (RunWriter additions):
  - `RunWriter.set_view_method_params(method_id: str, **params) -> None` — merge into `run.view_methods[i].params` (e.g. the survey-derived `r`), flush run.json.
  - `RunWriter.set_question(q: str) -> None` — set + flush (Flow A's ask arrives after create).
  - `RunWriter.write_session(session: SessionRecord) -> None` — for rigs whose camera opens after create.
- Produces (AIRunWriter, owned by the brain thread; writes ONLY `ai/SSS/**` + shared `events.jsonl`):

```python
class AIRunWriter:
    @classmethod
    def create(cls, run_dir: Path, *, seq: int = 0, mode: str = "live",
               orchestrator_model: str, menu_id: str, menu_hash: str,
               opening_prompt: str | None = None) -> "AIRunWriter"
        # mkdir ai/SSS + transcripts/ + menu/; write airun.json
    @property
    def dir(self) -> Path                      # ai/SSS
    def transcript(self, rec: "TranscriptRecord") -> Path   # transcripts/t{n:03d}.json, collision-proof counter
    def menu_def(self, m: "MenuDef") -> None                # menu/menu_def.json (once per version)
    def menu_input(self, mi: "MenuInput") -> None           # menu/inputs.jsonl append
    def answer(self, a: "AnswerRecord | dict") -> None      # answer.json
    def usage(self, **fields) -> None                       # merge into airun.json usage
    def event(self, kind: str, step_id=None, detail=None) -> None
        # SHARED file: append OperatorEvent line to <run>/events.jsonl,
        # open(..., "a") per call — O_APPEND atomicity, same as RunWriter.event
```

- [ ] **Step 1: Write the failing tests** (both files; ai_writer test builds real schema instances — read `schema.py` for `TranscriptRecord`/`MenuDef`/`MenuInput`/`AnswerRecord` required fields):

```python
def test_set_view_method_params_merges_r(tmp_path):
    w = _writer(tmp_path)   # existing helper in test_record_writer.py
    w.set_view_method_params("vs", r=0.24)
    run = json.loads((w.dir / "run.json").read_text())
    assert run["view_methods"][0]["params"]["r"] == 0.24

def test_ai_writer_full_round_trip(tmp_path):
    from inspection.record.ai_writer import AIRunWriter
    from inspection.record.run import Run
    make_run(tmp_path, run_id="0709-aiw")
    a = AIRunWriter.create(tmp_path / "0709-aiw", orchestrator_model="opus",
                           menu_id="viewsphere-menu", menu_hash="x" * 64)
    a.transcript(_transcript(step_id=1))       # helper building a valid TranscriptRecord
    a.answer({"verdict": "yes", "reasoning": "r", "evidence": ""})
    a.event("approved", step_id=1)
    run = Run.load(tmp_path / "0709-aiw")
    assert run.ai[0].answer.verdict == "yes"
    assert run.ai[0].transcripts[0].step_id == 1
    assert run.events[-1].kind == "approved"    # landed in the SHARED events.jsonl
```

- [ ] **Step 2: Run — fail.** **Step 3: Implement** (every JSON through `_write_json`; `answer` accepts a dict and validates into `AnswerRecord`). If `AnswerRecord`/`TranscriptRecord` lack a field the brain produces (compare against `brain/loop.py:_answer` payload: reasoning/verdict/evidence/evidence_images/views_inspected/coverage), add it to the schema with a RED schema test first.
- [ ] **Step 4: Run — pass; suite green.** **Step 5: Commit** `record: AIRunWriter — the brain's own pen, ai/ subtree only`.

---

### Task 8: Flow A wiring — machine/rigs/app write through `RunWriter`

**Files:**
- Modify: `inspection/run/machine.py`, `inspection/run/rigs.py` (`FakeRig.capture`, `RealRig.capture`, `perception/capture.py:save_bundle`), `inspection/run/app.py`, `inspection/ui/mock.py`, `inspection/brain/live.py` (mover events + `decider`), `inspection/brain/trace.py` (trace path → `ai/SSS/trace.jsonl`), `inspection/eyes/store.py` (store path → `ai/SSS/store.json`), `inspection/brain/loop.py` + `make_ask_handler` (create/use `AIRunWriter`)
- Test: `inspection/tests/test_flow_a.py`

**Interfaces:**
- Consumes: everything above.
- Produces the exact write-path table of flows-design §3. The load-bearing edits:

1. `Supervisor.__init__(..., writer: RunWriter)`; delete `self.turns`, `_record_turn`, `_save`, `_next_cell_step` (the UI `captures[target]["step"]` badge takes the new dense `step_id`).
2. `view/request` carries `decider` (default `"operator"`; `SupervisorMover.request` sends `"ai"`; Task 9's sweep sends `"sweep-shortest"`); Supervisor stashes it as `self._decider` at `_start_planning` for the step's `view_state`.
3. `_exec_worker`: after the pose handoff — `sid, sdir = writer.begin_step({"method": "vs", "address": None if target == "survey" else list(target)})`; `cap = self.rig.capture(sid, sdir)`; on settle ok → `writer.write_capture(sid, t_captured=..., joints_rad=cap["q"], T_base_flange=..., T_base_cam=..., rgb_rotation_deg=..., segmentation=...)`; on rejection/failure → `writer.mark_step(sid, outcome="rejected"|"failed", detail=...)`.
4. `_on_settle_done` ok-path → `writer.write_fused(sid, geometry, view_state)` with `GeometryStats` from the settle result and `ViewState(step_id=sid, t=..., candidates=<from sphere.cells() × _reach/visited/blocked>, centroid/extent/r=<from acc/sphere>, chosen=..., decider=self._decider)`; after the survey's `_recenter` derives `r` → `writer.set_view_method_params("vs", r=self.r)`.
5. `_on_exec_done` stopped/fault → `writer.event("stopped"|"fault", step_id=...)`. Mover's `on_event` beats → `ai_writer.event(state, ...)` (they already flow through `make_ask_handler.on_event`).
6. `rigs`: `capture(pose_id, out_dir: Path | None = None)` — binaries into `out_dir` when given; **no `meta.json`**; `save_bundle` gains `write_meta: bool = True` and rigs call it with `False` + the step dir. `RealRig.__init__` stops writing `session.json` — it builds a `SessionRecord` and the app hands it to the writer.
7. `app.py run()`: build `ConfigSnapshot` (`git_sha` via `subprocess.run(["git", "rev-parse", "HEAD"])`, model labels from `cognition.label`, calib/constants — mirror the fields `make_config` uses); `writer = RunWriter.create(outdir.parent, run_id=outdir.name, name=..., source="live", object=..., question=None, view_methods=[viewsphere def], q_survey=...)` — note `create` takes `(root, run_id)`, so pass the split, and add `--name/--object` CLI params defaulting name to the outdir leaf. `finally:` fused arrays → `outdir/"fused"/cloud.npy|colors.npy` then `writer.close("completed" if (outdir/"ai"/"000"/"answer.json").exists() else "aborted")`.
8. `make_ask_handler(sup, pub, run_dir, cognition, writer)`: creates `AIRunWriter` per ask (seq = next free `ai/SSS`), passes to `Brain` (trace + store + transcripts + answer all through it), calls `writer.set_question(q)`.
9. `ui/mock.py`: same wiring with `FakeRig` (this is what Task 11 drives).

- [ ] **Step 1: Write the failing integration test** — a full mock turn against the real Supervisor:

```python
#!/usr/bin/env python3
"""Flow A end-to-end on the FakeRig: one approved turn writes a valid run."""
import threading, time
import pytest

def test_one_approved_turn_writes_schema_run(tmp_path):
    from inspection.ui.mock import start_mock          # adapt to its real signature
    from inspection.record.run import Run
    bus, pub = _mock_bus()                             # copy the pattern mock.py uses
    sup = start_mock(bus, pub, tmp_path / "0709-mock")
    t = threading.Thread(target=sup.run, daemon=True); t.start()
    sup.events.put({"cmd": "view/request", "target": "survey"})
    _wait(lambda: sup.phase == "previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    _wait(lambda: sup.phase == "idle")
    sup.request_shutdown(); t.join(5)
    run = Run.load(tmp_path / "0709-mock")
    assert run.survey is not None and run.survey.record.phase == "fused"
    assert run.record.source == "live"
    assert not (tmp_path / "0709-mock" / "000").exists()   # old layout is DEAD
    rep = run.validate()
    assert rep.ok, rep.problems
```

- [ ] **Step 2: Run — fail.** **Step 3: Implement the wiring** in the numbered order above; run the record suite after each numbered item (the writer tests catch protocol misuse early).
- [ ] **Step 4: Full suite green** — expect and fix fallout in machine/brain tests that assumed `turns`/`meta.json`; the fix direction is always toward the schema, never a compat shim.
- [ ] **Step 5: Commit** `run: the live loop writes through RunWriter/AIRunWriter — old writes deleted`.

---

### Task 9: Flow B — the sweep driver + `collect` entry

**Files:**
- Create: `inspection/run/collect.py`
- Modify: `inspection/run/app.py` (add `collect` command), `inspection/ui/mock.py` (add `collect=True` variant)
- Test: `inspection/tests/test_collect.py`

**Interfaces:**
- Consumes: `SupervisorMover` (Task 8's `decider="sweep-shortest"` on its requests when constructed with `decider=...`), `sphere.plan_to_cell`, `RunWriter`.
- Produces:

```python
def joint_l1(path) -> float:
    """Cost of a planned path: sum of |dq| over consecutive waypoints."""

class SweepDriver(threading.Thread):
    def __init__(self, sup, run_dir, on_event=None, planner=None):
        # planner: injectable (world, ik, q_now, cell) -> (path | None, cost);
        # default builds its OWN RobotCell + UR5eIK (the machine's world is
        # single-threaded property — never touch sup.world from this thread)
        # and mirrors the object box from sup.acc.aabb() before each ranking.
    def run(self):
        # survey first via mover.request("survey"); then loop:
        #   cands = [c for c, r in sup._reach.items()
        #            if r is not None and c not in sup.visited and c not in sup.blocked]
        #   ranked = sorted((planner(...q_now..., c) for c in cands), by cost, None-cost last)
        #   ok, reason = mover.request(best)     # blocks on the human gate
        #   refusal -> next candidate; redirect -> continue from wherever the arm is
        # exit when no candidates remain; self.finished = True
    finished: bool
```

- [ ] **Step 1: Write the failing tests** — ordering is pure logic, test it with a fake planner:

```python
def test_ranking_picks_cheapest_and_skips_unplannable():
    from inspection.run.collect import rank_candidates
    costs = {(0, 0): 3.0, (1, 0): 1.0, (2, 0): None}
    ranked = rank_candidates([(0, 0), (1, 0), (2, 0)],
                             planner=lambda c: costs[c])
    assert ranked == [(1, 0), (0, 0)]            # None never ranks

def test_joint_l1():
    from inspection.run.collect import joint_l1
    import numpy as np
    path = np.array([[0.0] * 6, [1.0] + [0.0] * 5, [1.0, 2.0] + [0.0] * 4])
    assert joint_l1(path) == pytest.approx(3.0)
```

Plus an integration test mirroring Task 8's, with an auto-approving operator thread (`view/confirm` whenever `phase == "previewing"`), asserting: `run.record.source == "data-engine"`, every FakeRig-reachable cell visited, every non-survey `view_state.decider == "sweep-shortest"`, `run.validate().ok`.

- [ ] **Step 2: Run — fail.** **Step 3: Implement** `collect.py` (keep `rank_candidates(cands, planner)` a pure function — the thread calls it). `app.py collect(outdir, name, object, ...)`: identical composition to `run()` minus `real_cognition`/`make_ask_handler`, plus `SweepDriver`; writer `source="data-engine"`, `question=None`; close status `"completed" if driver.finished else "aborted"`.
- [ ] **Step 4: Run — pass; suite green.** **Step 5: Commit** `run: Flow B — data-collection sweep, true shortest path, same human gate`.

---

### Task 10: UI mode — `run/meta` topic, reduced collect UI

**Files:**
- Modify: `inspection/ui/publisher.py` (declare + publish retained `run/meta`), `inspection/run/app.py` + `inspection/ui/mock.py` (publish at boot), `inspection/ui/src/InspectionApp.tsx` + `inspection/ui/src/panelDefinitions.ts` (hide the trace/Ask UI when `source == "data-engine"`)
- Test: `inspection/tests/test_ui_meta.py` (backend half; frontend asserted by Task 11)

**Interfaces:**
- Produces: retained topic `run/meta` payload `{"source": "live"|"data-engine", "name": str, "object": str|null, "question": str|null}`; publisher method `pub.publish_run_meta(source=..., name=..., object=..., question=...)` called once at boot (and again when the question is set). Frontend: read `run/meta` where other retained topics are subscribed (grep `TracePanel`/`panelDefinitions` for the subscription pattern); when `source === "data-engine"`, the trace panel and Ask input are not rendered.

- [ ] **Step 1: failing backend test** — publish + assert the bus retains it (copy the publish/subscribe pattern from an existing publisher test if present; else assert via `InspectionPublisher` API surface that the topic is declared and the payload validates).
- [ ] **Step 2: fail → Step 3: implement backend, then frontend (`npm run build` in `inspection/ui` — check `ui/app.py:_require_build` for the expected dist path).**
- [ ] **Step 4: backend test green; suite green.** **Step 5: Commit** `ui: run/meta topic — collect mode renders without the AI panel`.

---

### Task 11: headless playwright e2e — both flows

**Files:**
- Create: `inspection/tests/e2e/test_flows_e2e.py`, `inspection/tests/e2e/conftest.py`
- Test: itself (marked `@pytest.mark.e2e`, excluded from the default run via `-m "not e2e"` in `pytest.ini`/`pyproject` if no marker config exists yet — add it)

**Interfaces:**
- Consumes: `ui/mock.py` app (both modes), the built frontend, playwright.
- Produces: the spec §8.3 verification, screenshots under `/tmp/e2e-flows/`.

- [ ] **Step 1: Install** (one-time): `~/miniconda3/envs/robo/bin/pip install playwright pytest-playwright && ~/miniconda3/envs/robo/bin/python -m playwright install chromium`. Record versions in the commit message.
- [ ] **Step 2: Write the e2e (it IS the failing test first — run it, watch it fail on selectors, fix selectors against the real DOM):**

```python
import pytest, subprocess, time
pytestmark = pytest.mark.e2e

def test_flow_a_ask_approve_answer(mock_app_live, page):   # fixtures in conftest
    page.goto(mock_app_live.url)
    page.fill("[data-testid=ask-input]", "is there a logo?")   # adapt selectors to the real DOM
    page.click("[data-testid=ask-send]")
    page.wait_for_selector("[data-testid=confirm-button]", timeout=30_000)
    page.screenshot(path="/tmp/e2e-flows/a-preview.png")
    page.click("[data-testid=confirm-button]")
    page.wait_for_selector("text=approved", timeout=30_000)
    page.screenshot(path="/tmp/e2e-flows/a-trace.png")
    mock_app_live.shutdown()
    from inspection.record.run import Run
    run = Run.load(mock_app_live.run_dir)
    assert run.record.source == "live" and run.validate().ok

def test_flow_b_no_ai_panel_sweep(mock_app_collect, page):
    page.goto(mock_app_collect.url)
    assert page.locator("[data-testid=trace-panel]").count() == 0
    for _ in range(3):
        page.wait_for_selector("[data-testid=confirm-button]", timeout=30_000)
        page.click("[data-testid=confirm-button]")
    page.screenshot(path="/tmp/e2e-flows/b-sweep.png")
    mock_app_collect.shutdown()
    from inspection.record.run import Run
    run = Run.load(mock_app_collect.run_dir)
    assert run.record.source == "data-engine" and run.validate().ok
```

`conftest.py` fixtures launch `ui/mock.py` (live / collect) as a subprocess on a free port with `--no_window`-equivalent serving, yield `{url, run_dir, shutdown()}`, and always terminate in teardown. Add `data-testid` attributes to the frontend where selectors need them (that is a frontend edit + rebuild, part of this task).
- [ ] **Step 3: Run e2e — green:** `-m pytest inspection/tests/e2e -m e2e -q`. Default suite still green and still fast (e2e excluded).
- [ ] **Step 4: Read the screenshots** with the Read tool — visually confirm Flow A shows the trace panel and Flow B does not. This is Anton's explicit acceptance criterion.
- [ ] **Step 5: Commit** `tests: headless playwright e2e over both flows`.

---

## Final gate (after Task 11)

- [ ] `-m pytest inspection/tests -q` — all green.
- [ ] `p -m inspection.record ls` over `inspection/data/runs` — 26 legacy cards still render (read paths untouched the archive).
- [ ] `p -m inspection.record show 2408-cup3 --no-open` still builds the workspace.
- [ ] Present Anton: the e2e screenshots + a `record card` of one mock Flow A run and one Flow B run.

## Self-review notes

- Spec coverage: §2→Tasks 6,8; §3→Tasks 7,8; §4→Tasks 1–5; §5→Task 8; §6→Task 9; §7→Task 10; §8→every task's test cycle + Task 11; build order §9 = task order. Out-of-scope §10 respected (no replay harness, no auto-confirm, no migration).
- Types cross-checked: `RunWriter.create(root, run_id=...)` split (Task 8.7 notes it); `capture(pose_id, out_dir)` consistent between Tasks 6 (reads cap dict) and 8.6; `decider` threaded Task 8.2 → Task 9.
- Known discovery points called out in-task (FakeRig ctor, make_run fixture cell, DOM selectors) rather than invented.
