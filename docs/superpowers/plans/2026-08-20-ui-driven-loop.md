# Loop v2 — UI-Driven Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the terminal-driven inspection loop with a UI-driven command state machine: two-press (request → looping preview → confirm) motions, live camera, live pose mirror, safe stop/exit — per `inspection/2026-08-20-ui-driven-loop-design.md` (read it first; it is the authority on every behavior below).

**Architecture:** One process (`inspection/run/app.py`): a never-blocking dispatcher (main thread) drives a state machine; one action worker at a time does plan/preview-loop/execute/settle; camera and pose streamer threads publish continuously; the porthole bus carries topics out and commands in. The frontend gains an ActionsPanel that is pure IO.

**Tech Stack:** Python 3 (conda env `robo`), pinocchio, ur_rtde, pyrealsense2, porthole bus (`~/projects/porthole`), React + TypeScript + vite (`inspection/ui`).

## Global Constraints

- Run Python as `conda run -n robo python …` from `/home/anton/projects/robotics` (alias `p` = `python`, env `robo`).
- Nothing in `inspection/run/` may import from `inspection/ui/` except `inspection.ui.publisher` (the seam); `publisher.py` must keep importing nothing from `inspection.run`.
- The publisher and every streamer swallow exceptions — a dead UI/bus must never abort a run (design §Safety).
- Only the dispatcher mutates run state; only the action worker touches planning/`arm.execute`; `scene/poses` has exactly one producer per phase.
- `stop_event` set by dispatcher or SIGINT handler only; cleared only by the dispatcher between actions.
- Commands are untrusted: validate `cmd`, `target`, and phase before acting; log and drop anything else.
- Frontend: no local button state — render from `views/state` + `run/status` only. No `window.pywebview.api` ever.
- Commit after every green test cycle. Frontend rebuild: `cd ~/projects/porthole && npm run build -w @porthole/framework` (only if porthole changed), then `cd inspection/ui && npm run build`.
- Existing tests that must stay green: `p inspection/tests/test_decider.py` (decider parks, its tests stay), `npm run check` in `inspection/ui`.

## File Structure

| File | Responsibility |
|---|---|
| `inspection/run/machine.py` (new) | `Supervisor`: state machine, dispatcher, action worker. No hardware imports. |
| `inspection/run/rigs.py` (new) | `FakeRig` (moved+extended from tests), `CameraWorker`, `PoseStreamer`, `RealRig` (moved from loop.py, camera/pose threads added). Hardware imports lazy. |
| `inspection/run/app.py` (new) | Composition root + CLI: `run`, `teach`. SIGINT wrapper, shutdown sequence. |
| `inspection/ui/publisher.py` (modify) | New cell states + colours, `survey` entry in `views/state`, new phase vocabulary. |
| `inspection/ui/mock.py` (rewrite) | Drives the real `Supervisor` with `FakeRig` on a real bus. |
| `inspection/ui/app.py` (modify) | `mock` runs Supervisor-based mock; `serve` unchanged. |
| `inspection/ui/src/panels/ActionsPanel.tsx` (new) | Survey + grid + Stop + Exit. Pure IO. |
| `inspection/ui/src/panels/CloudInspectPanel.tsx` (modify) | Two new marker states. |
| `inspection/ui/src/panelDefinitions.ts`, `src/InspectionApp.tsx` (modify) | Register + place ActionsPanel. |
| `inspection/tests/test_machine.py` (new) | Supervisor tests with FakeRig + injected commands. |
| `inspection/run/loop.py`, `inspection/tests/test_loop_fake.py` (delete at end) | Superseded. `teach()` moves to app.py; `decider.py` stays parked. |

---

### Task 1: Publisher v2 — new states, survey entry, phase vocabulary

**Files:**
- Modify: `inspection/ui/publisher.py`
- Test: `inspection/tests/test_publisher_v2.py` (new)

**Interfaces:**
- Consumes: existing `InspectionPublisher`.
- Produces (later tasks rely on these exact signatures):
  - `STATE_COLOR` gains `"pending": "#b0a0f0"`, `"previewing": "#f08bd4"` (cosmetic; Anton restyles with his grid drawing later).
  - `publish_views(sphere, reach, visited=(), blocked=(), current=None, captures=None, glosses=None, pending=None, previewing=None, survey="visited")` — `pending`/`previewing` are single cells `(h, v) | None` taking precedence over other states; `survey` ∈ state vocabulary, emitted as top-level `"survey": {"state": ...}` in the `views/state` payload.
  - `publish_survey_only(state)` — pre-sphere payload: `{"survey": {"state": state}, "center": None, "radius": None, "current": None, "cells": []}` on `views/state` (and no 3D markers). Used before boot capture.
  - `status(**fields)` unchanged; callers use `phase` ∈ `idle|planning|previewing|executing|capturing|fusing|fault|done` and `target` (`"survey"` or `[h, v]` or `None`).

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""publisher v2: pending/previewing/survey in views/state. Run: p inspection/tests/test_publisher_v2.py"""
import numpy as np
from inspection.ui.publisher import InspectionPublisher, STATE_COLOR


class BusSpy:
    def __init__(self):
        self.published = []          # (topic, payload)
    def publish(self, topic, payload):
        self.published.append((topic, payload))
    def declare(self, *a, **kw):
        pass
    @staticmethod
    def array_payload(a):
        return a
    @staticmethod
    def jpeg_payload(a, quality=80):
        return a
    def last(self, topic):
        return [p for t, p in self.published if t == topic][-1]


class FakeSphere:
    center = np.array([0.4, 0.0, 0.1])
    r = 0.35
    elevations = [20.0, 45.0, 70.0]
    def cells(self):
        return [(h, v) for v in range(2) for h in range(3)]
    def cell_dir(self, h, v):
        d = np.array([np.cos(h), np.sin(h), 0.5 + v])
        return d / np.linalg.norm(d)


def test_states_and_survey():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    sphere = FakeSphere()
    reach = {c: 0.0 for c in sphere.cells()}
    reach[(2, 1)] = None                                   # unreachable
    pub.publish_views(sphere, reach, visited=[(0, 0)], blocked=[(1, 0)],
                      current=(0, 0), pending=(1, 1), previewing=None,
                      survey="visited")
    views = bus.last("views/state")
    st = {(c["h"], c["v"]): c["state"] for c in views["cells"]}
    assert st[(1, 1)] == "pending"
    assert st[(1, 0)] == "blocked"
    assert st[(2, 1)] == "unreachable"
    assert views["survey"] == {"state": "visited"}
    assert "pending" in STATE_COLOR and "previewing" in STATE_COLOR


def test_previewing_precedence_and_survey_only():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    sphere = FakeSphere()
    reach = {c: 0.0 for c in sphere.cells()}
    pub.publish_views(sphere, reach, visited=[(1, 1)], previewing=(1, 1),
                      survey="visited")
    views = bus.last("views/state")
    st = {(c["h"], c["v"]): c["state"] for c in views["cells"]}
    assert st[(1, 1)] == "previewing"                      # beats visited

    bus2 = BusSpy()
    pub2 = InspectionPublisher(bus2)
    pub2.publish_survey_only("pending")
    views2 = bus2.last("views/state")
    assert views2 == {"survey": {"state": "pending"}, "center": None,
                      "radius": None, "current": None, "cells": []}


def main():
    test_states_and_survey()
    test_previewing_precedence_and_survey_only()
    print("OK test_publisher_v2")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it — must fail** (`conda run -n robo python inspection/tests/test_publisher_v2.py`; expect `TypeError: publish_views() got an unexpected keyword argument 'pending'`).

- [ ] **Step 3: Implement.** In `publisher.py`:
  - Add the two `STATE_COLOR` entries above.
  - Extend `publish_views` signature with `pending=None, previewing=None, survey="visited"`. In `_cell_state`, order: `previewing` (arg match) > `pending` (arg match) > `current` > `visited` > unreachable > blocked > available. Pass the two cells through (tuple-compare like `current`). Marker radius for `previewing` = 0.02 like `current`. Add `"survey": {"state": survey}` to the `TOPIC_VIEWS` payload.
  - Add `publish_survey_only(self, state)` publishing exactly the payload asserted above (wrapped in the usual try/except).

- [ ] **Step 4: Run test — PASS.** Also run `conda run -n robo python inspection/tests/test_decider.py` (must stay green).

- [ ] **Step 5: Commit** — `git add inspection/ui/publisher.py inspection/tests/test_publisher_v2.py && git commit -m "publisher v2: pending/previewing states, survey entry, pre-sphere payload"`

---

### Task 2: FakeRig v2 + CameraWorker + PoseStreamer (`rigs.py`)

**Files:**
- Create: `inspection/run/rigs.py`
- Test: `inspection/tests/test_rigs.py` (new)

**Interfaces:**
- Consumes: `inspection.tests.synth.synth_capture` (existing; returns `(depth, intr, scale, T_bc, …)`).
- Produces:
  - `FakeRig(q0, stop_event=None, dt=0.01, speed=3.0)` with: `q() -> np.ndarray`; `move(path) -> {"waypoints_done", "s", "final_err_deg", "stopped"}` — interpolates joints at `speed` rad/s in `dt` ticks, honours `stop_event` (set → stopJ-like halt, `stopped: True`, `q` frozen mid-path); `capture(pose_id) -> {"dir", "rgb", "depth_raw", "T_base_cam", "q"}` (synth depth, `rgb` = 64×64×3 uint8 gradient); `frame() -> np.ndarray` (live rgb); `close()`. Attributes `intr`, `depth_scale`, `moves` (list of path lens).
  - `CameraWorker(grab, publish=None, hz=10.0)` (`threading.Thread`, daemon): `grab() -> dict | None` (keys at least `rgb`); `publish(rgb)` called per tick if given; `.fresh_bundle(min_new=3, timeout=3.0) -> dict` — returns the first bundle at least `min_new` grabs after the call; `.latest() -> (count, bundle | None)`; `.stop()`.
  - `PoseStreamer(q_fn, publish_pose, hz=30.0)` (`threading.Thread`, daemon): `.active` (`threading.Event`) — publishes `publish_pose(q_fn())` per tick only while set; `.stop()`.

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""rigs: FakeRig interpolation+stop, CameraWorker freshness, PoseStreamer gating.
Run: p inspection/tests/test_rigs.py"""
import threading
import time

import numpy as np

from inspection.run.rigs import CameraWorker, FakeRig, PoseStreamer


def test_fake_move_interpolates_and_stops():
    stop = threading.Event()
    rig = FakeRig(np.zeros(6), stop_event=stop, dt=0.005, speed=2.0)
    goal = np.zeros(6); goal[0] = 0.4
    rep = rig.move([np.zeros(6), goal])
    assert rep["stopped"] is False and np.allclose(rig.q(), goal)

    far = np.zeros(6); far[0] = 50.0                       # long move
    t = threading.Thread(target=lambda: (time.sleep(0.05), stop.set()))
    t.start()
    rep = rig.move([goal, far])
    t.join()
    assert rep["stopped"] is True
    assert rig.q()[0] < 49.0                               # froze mid-path


def test_camera_worker_freshness_and_publish():
    calls = []
    n = [0]
    def grab():
        n[0] += 1
        return {"rgb": np.full((4, 4, 3), n[0] % 255, np.uint8)}
    cam = CameraWorker(grab, publish=lambda rgb: calls.append(1), hz=200.0)
    cam.start()
    c0, _ = cam.latest()
    b = cam.fresh_bundle(min_new=3, timeout=2.0)
    c1, _ = cam.latest()
    assert c1 >= c0 + 3 and b["rgb"] is not None
    cam.stop()
    assert calls, "publish was never called"


def test_pose_streamer_gated_by_active():
    got = []
    ps = PoseStreamer(lambda: np.zeros(6), lambda q: got.append(1), hz=200.0)
    ps.start()
    time.sleep(0.05)
    assert not got, "published while inactive"
    ps.active.set()
    time.sleep(0.05)
    ps.active.clear()
    assert got, "did not publish while active"
    ps.stop()


def main():
    test_fake_move_interpolates_and_stops()
    test_camera_worker_freshness_and_publish()
    test_pose_streamer_gated_by_active()
    print("OK test_rigs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run — must fail** (`ModuleNotFoundError: inspection.run.rigs`).

- [ ] **Step 3: Implement `inspection/run/rigs.py`** (FakeRig adapted from `inspection/tests/test_loop_fake.py`; module docstring: "Rig implementations behind the rig contract: q/move/capture/frame/close."):

```python
class FakeRig:
    def __init__(self, q0, stop_event=None, dt=0.01, speed=3.0):
        self._q = np.asarray(q0, dtype=float)
        self.stop_event = stop_event or threading.Event()
        self.dt, self.speed = dt, speed
        from inspection.tests.synth import synth_capture
        depth, intr, scale, T_bc, _ = synth_capture()
        self._depth, self._T_bc = depth, T_bc
        self.intr, self.depth_scale = intr, scale
        self.moves = []

    def q(self): return self._q.copy()

    def move(self, path):
        self.moves.append(len(path))
        t0, done = time.perf_counter(), 0
        for q_goal in [np.asarray(q, float) for q in path[1:]]:
            while True:
                if self.stop_event.is_set():
                    return {"waypoints_done": done, "s": time.perf_counter() - t0,
                            "final_err_deg": 0.0, "stopped": True}
                delta = q_goal - self._q
                dist = np.abs(delta).max()
                if dist <= self.speed * self.dt:
                    self._q = q_goal.copy(); break
                self._q = self._q + delta / dist * self.speed * self.dt
                time.sleep(self.dt)
            done += 1
        return {"waypoints_done": done, "s": time.perf_counter() - t0,
                "final_err_deg": 0.0, "stopped": False}

    def frame(self):
        rgb = np.zeros((64, 64, 3), np.uint8)
        rgb[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        return rgb

    def capture(self, pose_id):
        return {"dir": f"(fake {pose_id:03d})", "rgb": self.frame(),
                "depth_raw": self._depth, "T_base_cam": self._T_bc,
                "q": self._q.copy()}

    def close(self): pass
```

`CameraWorker`: loop `while not self._stop.is_set()`: `b = grab()`; on non-None, under a `threading.Condition`: `self._count += 1; self._latest = b; notify_all()`; call `publish(b["rgb"])` inside try/except; sleep `1/hz`. `fresh_bundle`: record `target = self._count + min_new` under the condition, `wait_for(lambda: self._count >= target, timeout)`, raise `RuntimeError("camera produced no fresh frames")` on timeout, else return `self._latest`. `latest()` returns `(self._count, self._latest)` under the condition. `PoseStreamer`: loop: `if self.active.is_set(): try: publish_pose(q_fn()) except: log`; sleep `1/hz`.

- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** — `"rigs: FakeRig v2 (stoppable interpolating move), CameraWorker, PoseStreamer"`

---

### Task 3: Supervisor — request → planning → previewing (+ supersede, blocked)

**Files:**
- Create: `inspection/run/machine.py`
- Test: `inspection/tests/test_machine.py` (new)

**Interfaces:**
- Consumes: `FakeRig` (Task 2), publisher v2 (Task 1), `RobotCell`, `UR5eIK`, `ViewSphere`, `plan_viewpoint`, `CloudAccumulator`, `object_in_base`, `DEMO_PARK`.
- Produces:
  - `Supervisor(rig, pub, outdir, q_survey, world=None, ik=None, r=0.35, seed=0)` — attrs: `events: queue.Queue` (bus commands and worker events land here), `stop_event: threading.Event` (same object as `rig.stop_event`), `pose_active: threading.Event`, `phase: str`, `target`, `gen: int`, `visited/blocked: set`, `current`, `sphere`, `acc`.
  - `run()` — dispatcher loop; returns when phase becomes `done`. `request_shutdown()` — thread-safe (SIGINT-safe): sets `stop_event`, puts `{"cmd": "run/exit"}`.
  - Internal worker events (dispatcher-validated by `gen`): `{"ev": "plan_done", "gen", "target", "path" | None, "detail"}`, `{"ev": "phase", "gen", "phase"}`, `{"ev": "exec_done", "gen", "target", "outcome": "done"|"stopped"|"fault", "detail"}`, `{"ev": "settle_done", "gen", "target", "ok", "npts", "detail"}`.
  - Targets: `"survey"` or `(h, v)` (bus sends `[h, v]`; normalize with `tuple()`).

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""Supervisor: request->planning->previewing, supersede, blocked.
Run: p inspection/tests/test_machine.py"""
import queue
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from inspection.motion.plan import DEMO_PARK
from inspection.run.machine import Supervisor
from inspection.run.rigs import FakeRig
from inspection.tests.test_publisher_v2 import BusSpy
from inspection.ui.publisher import InspectionPublisher


def wait_for(cond, timeout=30.0, msg=""):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, f"timeout: {msg}"
        time.sleep(0.01)


def make_sup(q_start=None, **kw):
    q_survey = DEMO_PARK.copy()
    rig = FakeRig(q_start if q_start is not None else q_survey.copy())
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus),
                     Path(tempfile.mkdtemp()) / "run", q_survey, **kw)
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()
    return sup, rig, bus, th


def test_survey_request_reaches_previewing():
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    assert sup.phase == "idle"
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    assert sup.target == "survey"
    # preview loop is publishing poses
    n0 = len([1 for t, _ in bus.published if t == "scene/poses"])
    time.sleep(0.2)
    n1 = len([1 for t, _ in bus.published if t == "scene/poses"])
    assert n1 > n0, "preview replay is not looping"
    sup.request_shutdown(); th.join(10)


def test_invalid_commands_dropped():
    sup, rig, bus, th = make_sup()
    sup.events.put({"cmd": "view/confirm", "target": "survey"})   # not previewing
    sup.events.put({"cmd": "nonsense"})
    sup.events.put({"cmd": "view/request", "target": [99, 99]})   # no sphere yet
    time.sleep(0.3)
    assert sup.phase == "idle"
    sup.request_shutdown(); th.join(10)


def test_stale_plan_result_discarded():
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    sup.events.put({"cmd": "view/request", "target": "survey"})
    gen = None
    wait_for(lambda: sup.phase == "planning", msg="planning")
    gen = sup.gen
    # a stale plan_done from a dead generation must be ignored
    sup.events.put({"ev": "plan_done", "gen": gen - 1, "target": "survey",
                    "path": [DEMO_PARK.copy()], "detail": "stale"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    sup.request_shutdown(); th.join(10)


def main():
    test_survey_request_reaches_previewing()
    test_invalid_commands_dropped()
    test_stale_plan_result_discarded()
    print("OK test_machine (task 3)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run — must fail** (`ModuleNotFoundError: inspection.run.machine`).

- [ ] **Step 3: Implement `machine.py`.** Core skeleton (the docstring must reference the design doc):

```python
class Supervisor:
    def __init__(self, rig, pub, outdir, q_survey, world=None, ik=None,
                 r=0.35, seed=0):
        self.rig, self.pub = rig, pub
        self.outdir = Path(outdir); self.q_survey = np.asarray(q_survey, float)
        self.world = world if world is not None else RobotCell()
        self.ik = ik if ik is not None else UR5eIK()
        self.r, self.seed = r, seed
        self.events = queue.Queue()
        self.stop_event = getattr(rig, "stop_event", None) or threading.Event()
        rig.stop_event = self.stop_event
        self.pose_active = threading.Event()
        self.phase, self.target, self.gen = "idle", None, 0
        self.visited, self.blocked = set(), set()
        self.current, self.sphere = None, None
        self.acc = CloudAccumulator()
        self.captures = {}          # cell -> {"step", "dir"}
        self.survey_state = "available"
        self.turns, self.t0 = [], time.time()
        self._preview_cancel = None
        self._path = None           # path of the current preview

    def run(self):
        self._publish_all()
        self._set_phase("idle")
        while self.phase != "done":
            try:
                ev = self.events.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.handle(ev)
            except Exception:
                log.exception("dispatcher error on %s", ev)
        self._save()
```

`handle(ev)` dispatch table — implement exactly this validation:
  - `cmd == "view/request"`: normalize target (`"survey"` or `tuple(ev["target"])`). Reject (log + return) if: phase in `("executing", "capturing", "fusing", "fault")`; target is a cell and (`self.sphere is None` or not reachable or in `self.visited`); target `"survey"` and `survey_state == "visited"`. Else: `self._cancel_preview()`; `self.gen += 1`; mark pending (`survey_state = "pending"` or remember the pending cell); `self._set_phase("planning", target)`; `self._publish_views()`; spawn `threading.Thread(target=self._plan_worker, args=(self.gen, target), daemon=True)`.
  - `cmd == "view/confirm"`: only if `phase == "previewing"` and normalized target equals `self.target`; else log+drop. On accept: `self._cancel_preview()`; `self.stop_event.clear()`; `self.pose_active.set()`; `self._set_phase("executing", self.target)`; spawn `self._exec_worker(self.gen, self.target, self._path)`.
  - `cmd == "run/stop"`: if `phase == "executing"`: `self.stop_event.set()`; else log no-op.
  - `cmd == "run/exit"`: if `phase == "executing"`: `self.stop_event.set()` and remember `self._exit_after = True`, handled when `exec_done` arrives; else `self._shutdown()` (cancel preview, `self._set_phase("done")`).
  - `ev == "plan_done"` (drop if `ev["gen"] != self.gen`): if `path is None`: blocked (cell → `self.blocked`; survey → `survey_state = "available"`), `pub.log("warn", …)`, `self._set_phase("idle", None)`, `self._publish_views()`. Else store `self._path = ev["path"]`, `self._set_phase("previewing", ev["target"])`, `self._publish_views()`, start `self._preview_cancel = threading.Event()` + `threading.Thread(target=self._preview_worker, args=(self.gen, ev["path"], self._preview_cancel), daemon=True)`.
  - `ev == "phase"` (gen-checked): `self._set_phase(ev["phase"], self.target)` (worker-reported `capturing`/`fusing`).
  - `ev == "exec_done"` (gen-checked): `self.pose_active.clear()`. `outcome == "stopped"` → record turn, `self.current = None`, `self.blocked.clear()`, `self.stop_event.clear()` **only here** (design: cleared only by dispatcher between actions), `self._set_phase("idle", None)`; if `self._exit_after`: `self._shutdown()`. `outcome == "fault"` → record, `self._set_phase("fault", None)`, `pub.log("error", "executor halted/refused — check pendant; Exit only")`. `outcome == "done"` → wait for `settle_done` (worker continues by itself).
  - `ev == "settle_done"` (gen-checked): if `ok`: mark visited (cell → `visited`/`captures[cell]`/`current = cell`; survey → `survey_state = "visited"`), `self.blocked.clear()`, republish object+views, record turn. Else: `pub.log("error", detail)`, record turn, **not** visited. Both → `self._set_phase("idle", None)`; `self._save()`.

Workers (each wraps its body in `try/except` that turns any raise into the failure event; each publishes nothing directly except poses/log via `pub`):

```python
    def _plan_worker(self, gen, target):
        try:
            q_now = self.rig.q()
            if target == "survey":
                path, rep = plan_viewpoint(self.world, self.ik, q_now,
                                           self.ik.fk(self.q_survey), seed=self.seed)
                detail = "" if path else str(rep.get("reason", rep))
            else:
                path, prep, roll = self.sphere.plan_to_cell(
                    self.world, self.ik, q_now, *target, seed=self.seed)
                detail = f"tier {prep['tier']}" if path else "plan refused"
        except Exception as e:
            path, detail = None, f"planner error: {e}"
        self.events.put({"ev": "plan_done", "gen": gen, "target": target,
                         "path": path, "detail": detail})

    def _preview_worker(self, gen, path, cancel):
        from inspection.cell.world import DEFAULT_STEP
        dense = self.world.discretize(path, DEFAULT_STEP)
        while not cancel.is_set():
            for q in dense:
                if cancel.is_set():
                    return
                self.pub.publish_pose(self.world, q,
                                      colliding=self.world.is_colliding(q))
                time.sleep(1 / 30.0)

    def _exec_worker(self, gen, target, path):
        try:
            rep = self.rig.move(path)
        except RuntimeError as e:
            self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                             "outcome": "fault", "detail": str(e)})
            return
        if rep.get("stopped"):
            self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                             "outcome": "stopped", "detail": ""})
            return
        self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                         "outcome": "done", "detail": ""})
        self.events.put({"ev": "phase", "gen": gen, "phase": "capturing"})
        try:
            step = len(self.turns) + 1 if target != "survey" else 0
            cap = self.rig.capture(step)
            view = object_in_base(cap["depth_raw"], self.rig.intr,
                                  self.rig.depth_scale, cap["T_base_cam"])
            self.events.put({"ev": "phase", "gen": gen, "phase": "fusing"})
            npts = len(view["points"]) if view["points"] is not None else 0
            if target == "survey" and view["centroid"] is None:
                self.events.put({"ev": "settle_done", "gen": gen, "target": target,
                                 "ok": False, "npts": 0,
                                 "detail": "NO OBJECT above the table"})
                return
            if npts:
                self.acc.add(view["points"])
            self._recenter()
            self.pub.publish_capture(cap)
            self._last_cap_dir = cap.get("dir")
            self.events.put({"ev": "settle_done", "gen": gen, "target": target,
                             "ok": True, "npts": npts, "detail": ""})
        except RuntimeError as e:
            self.events.put({"ev": "settle_done", "gen": gen, "target": target,
                             "ok": False, "npts": 0, "detail": f"capture failed: {e}"})
```

Port `_recenter()` and `_save()` from `loop.py` verbatim (`_recenter` builds `ViewSphere` + `world.set_object`; `_save` writes `run.json` — record entries: `{"step", "target", "t", "result", "stopped"}`). `_set_phase(phase, target=…)` sets attrs and calls `pub.status(phase=…, target=list(target) if isinstance(target, tuple) else target, visited=len(self.visited), total=…)`. `_publish_views()`: if `self.sphere is None` → `pub.publish_survey_only(self.survey_state)`; else compute `reach = self.sphere.reachability(self.world, self.ik)` **in the planning worker, not the dispatcher** — cache the last `reach` on the supervisor (`self._reach`), recompute it in `_plan_worker` before planning and in `_exec_worker` after `_recenter()`; `_publish_views` uses the cache with `pending=…/previewing=…` from phase+target, `survey=self.survey_state`, `captures=self.captures`. `_cancel_preview()`: set + drop `self._preview_cancel` if alive. `_shutdown()`: `self._cancel_preview(); self.pose_active.clear(); self._set_phase("done")`.

- [ ] **Step 4: Run — PASS** (planning is real OMPL; allow the 30 s timeouts).
- [ ] **Step 5: Commit** — `"machine: Supervisor request->planning->previewing with generation guard"`

---

### Task 4: Supervisor — confirm → executing → settling, stop, capture-fail, fault

**Files:**
- Modify: `inspection/run/machine.py` (behavior already sketched in Task 3 — this task drives it green)
- Test: append to `inspection/tests/test_machine.py`

**Interfaces:** consumes/produces as Task 3; adds nothing new.

- [ ] **Step 1: Append failing tests**

```python
def full_boot(sup):
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="boot previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    wait_for(lambda: sup.phase == "idle" and sup.survey_state == "visited",
             timeout=60, msg="boot settled")


def test_full_cycle_and_visited():
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    full_boot(sup)
    assert sup.sphere is not None and len(sup.acc.points)
    views = bus.last("views/state")
    assert views["survey"]["state"] == "visited"
    cells = [c for c in views["cells"] if c["state"] == "available"]
    assert cells, "no reachable cells after boot"
    target = [cells[0]["h"], cells[0]["v"]]
    sup.events.put({"cmd": "view/request", "target": target})
    wait_for(lambda: sup.phase == "previewing", timeout=60, msg="cell preview")
    sup.events.put({"cmd": "view/confirm", "target": target})
    wait_for(lambda: tuple(target) in sup.visited, timeout=60, msg="cell visited")
    assert sup.current == tuple(target)
    sup.request_shutdown(); th.join(10)


def test_stop_mid_execute_returns_to_idle():
    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 30]))          # long boot move
    rig.speed = 0.05                    # slow: ~10 s — time to stop it
    sup.events.put({"cmd": "view/request", "target": "survey"})
    wait_for(lambda: sup.phase == "previewing", msg="previewing")
    sup.events.put({"cmd": "view/confirm", "target": "survey"})
    wait_for(lambda: sup.phase == "executing", msg="executing")
    assert sup.pose_active.is_set()
    sup.events.put({"cmd": "run/stop"})
    wait_for(lambda: sup.phase == "idle", msg="stopped -> idle")
    assert not sup.pose_active.is_set()
    assert not sup.stop_event.is_set(), "dispatcher must clear stop_event"
    assert sup.survey_state != "visited"
    sup.request_shutdown(); th.join(10)


class CaptureFailRig(FakeRig):
    def capture(self, pose_id):
        if pose_id == 1:
            raise RuntimeError("synthetic bad view")
        return super().capture(pose_id)


def test_capture_fail_not_visited():
    q_survey = DEMO_PARK.copy()
    rig = CaptureFailRig(q_survey.copy() + np.radians([0, 0, 0, 0, 0, 8]))
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus),
                     Path(tempfile.mkdtemp()) / "run", q_survey)
    th = threading.Thread(target=sup.run, daemon=True); th.start()
    full_boot(sup)
    views = bus.last("views/state")
    target = [c["h"] for c in views["cells"] if c["state"] == "available"][:1] and \
             [[c["h"], c["v"]] for c in views["cells"] if c["state"] == "available"][0]
    sup.events.put({"cmd": "view/request", "target": target})
    wait_for(lambda: sup.phase == "previewing", timeout=60, msg="preview")
    sup.events.put({"cmd": "view/confirm", "target": target})
    wait_for(lambda: sup.phase == "idle", timeout=60, msg="settle")
    assert tuple(target) not in sup.visited
    sup.request_shutdown(); th.join(10)


class FaultRig(FakeRig):
    def move(self, path):
        if getattr(self, "_boot_done", False):
            raise RuntimeError("safety mode changed mid-path")
        self._boot_done = True
        return super().move(path)


def test_executor_fault_enters_fault_and_exit_works():
    q_survey = DEMO_PARK.copy()
    rig = FaultRig(q_survey.copy() + np.radians([0, 0, 0, 0, 0, 8]))
    bus = BusSpy()
    sup = Supervisor(rig, InspectionPublisher(bus),
                     Path(tempfile.mkdtemp()) / "run", q_survey)
    th = threading.Thread(target=sup.run, daemon=True); th.start()
    full_boot(sup)
    views = bus.last("views/state")
    target = [[c["h"], c["v"]] for c in views["cells"]
              if c["state"] == "available"][0]
    sup.events.put({"cmd": "view/request", "target": target})
    wait_for(lambda: sup.phase == "previewing", timeout=60, msg="preview")
    sup.events.put({"cmd": "view/confirm", "target": target})
    wait_for(lambda: sup.phase == "fault", timeout=60, msg="fault")
    sup.events.put({"cmd": "view/request", "target": target})   # dead in fault
    time.sleep(0.2)
    assert sup.phase == "fault"
    sup.events.put({"cmd": "run/exit"})
    th.join(10); assert not th.is_alive()
    assert (sup.outdir / "run.json").exists()
```

Add all five to `main()`.

- [ ] **Step 2: Run — new tests must fail** (whichever transition is not implemented yet fails its `wait_for`).
- [ ] **Step 3: Implement/complete** the `view/confirm`, `run/stop`, `exec_done`, `settle_done`, fault, and `_save` paths exactly as specified in Task 3's dispatch table.
- [ ] **Step 4: Run — all PASS** (`conda run -n robo python inspection/tests/test_machine.py`).
- [ ] **Step 5: Commit** — `"machine: execute/settle/stop/fault paths complete"`

---

### Task 5: `run/app.py` — composition root, SIGINT, shutdown, teach

**Files:**
- Create: `inspection/run/app.py`
- Test: append `test_sigint_saves_and_exits` to `inspection/tests/test_machine.py`

**Interfaces:**
- Consumes: `Supervisor`, `RealRig`+`FakeRig` (rigs), `PortholeBus`, `serve_ui`/`open_window` (porthole.window), `InspectionPublisher`, `mesh_dir_for`, `preflight`, `SURVEY_POSE_FILE` pattern from loop.py.
- Produces: CLI `p inspection/run/app.py run --outdir=… [--ip --r --max_turns --port --bus_port --no_window]`, `p inspection/run/app.py teach`. Also `install_sigint(sup)` — importable, testable: installs a SIGINT handler calling `sup.request_shutdown()`, restores default handler on second Ctrl-C.

- [ ] **Step 1: Write the failing test** (append; uses `signal.raise_signal` so it exercises the real handler path in-process):

```python
def test_sigint_saves_and_exits():
    import signal
    from inspection.run.app import install_sigint
    sup, rig, bus, th = make_sup()
    install_sigint(sup)
    try:
        signal.raise_signal(signal.SIGINT)
        th.join(15); assert not th.is_alive(), "SIGINT did not shut down"
        assert (sup.outdir / "run.json").exists()
        assert sup.stop_event.is_set() or sup.phase == "done"
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
```

- [ ] **Step 2: Run — fails** (`ModuleNotFoundError: inspection.run.app`).
- [ ] **Step 3: Implement `inspection/run/app.py`:**

```python
#!/usr/bin/env python3
"""Loop v2 composition root. Design: inspection/2026-08-20-ui-driven-loop-design.md.

    p inspection/run/app.py run --outdir=data/runs/r1     # the real thing
    p inspection/run/app.py teach                         # save survey pose
"""
import json, logging, signal, sys, threading, webbrowser
from pathlib import Path
import numpy as np

ROBOT_IP = "192.168.2.50"
SURVEY_POSE_FILE = Path(__file__).resolve().parent / "survey_pose.json"


def install_sigint(sup):
    """First Ctrl-C: safe shutdown (stop arm, save, close). Second: hard exit."""
    def handler(signum, frame):
        signal.signal(signal.SIGINT, signal.default_int_handler)
        sup.request_shutdown()
    signal.signal(signal.SIGINT, handler)


def teach(ip: str = ROBOT_IP):
    # moved verbatim from loop.py
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(ip)
    q = list(r.getActualQ())
    r.disconnect()
    SURVEY_POSE_FILE.write_text(json.dumps({"q_rad": q}, indent=2) + "\n")
    print(f"survey pose saved: {np.round(np.degrees(q), 1).tolist()} deg")


def run(outdir: str, ip: str = ROBOT_IP, r: float = 0.35,
        port: int = 8767, bus_port: int = 8765, no_window: bool = False,
        seed: int = 0, gui: str = "qt"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from porthole import PortholeBus
    from porthole.window import open_window, serve_ui
    from inspection.motion.execute import preflight
    from inspection.run.machine import Supervisor
    from inspection.run.rigs import PoseStreamer, RealRig
    from inspection.ui.app import DIST, _asset_mounts, _require_build
    from inspection.ui.publisher import InspectionPublisher

    pf = preflight(ip)
    if not pf["go"]:
        raise SystemExit(f"preflight NO-GO: {pf}")
    if not SURVEY_POSE_FILE.exists():
        raise SystemExit("no survey pose — run `p inspection/run/app.py teach`")
    if not _require_build():
        raise SystemExit(1)
    q_survey = np.array(json.loads(SURVEY_POSE_FILE.read_text())["q_rad"])

    outdir = Path(outdir)
    bus = PortholeBus(app="inspection", port=bus_port).start()
    pub = InspectionPublisher(bus, run_dir=outdir)
    pub.declare()
    stop_event = threading.Event()
    rig = RealRig(None, stop_event, outdir, ip)     # world set below
    sup = Supervisor(rig, pub, outdir, q_survey, seed=seed, r=r)
    rig.world = sup.world
    pub.publish_world(sup.world)
    rig.start_camera(pub)
    poses = PoseStreamer(rig.q, lambda q: pub.publish_pose(sup.world, q))
    poses.active = sup.pose_active                   # dispatcher-gated
    poses.start()
    install_sigint(sup)

    def pump():
        for c in bus.commands():
            sup.events.put(c)
    threading.Thread(target=pump, name="cmd-pump", daemon=True).start()

    url = serve_ui(DIST, port=port, assets=_asset_mounts(outdir))
    if bus_port != 8765:
        url = f"{url}/?bus={bus_port}"
    print(f"ui   {url}", flush=True)
    ui_thread = threading.Thread(
        target=lambda: webbrowser.open(url) if no_window else open_window(
            url, title="inspection", width=1700, height=1000, gui=gui),
        daemon=True)
    ui_thread.start()
    try:
        sup.run()                                    # blocks until done
    finally:
        # belt-and-braces shutdown, in this order (design §Safety)
        sup.stop_event.set()
        poses.stop()
        try:
            rig.arm.stop()
        except Exception:
            pass
        if len(sup.acc.points):
            np.save(outdir / "fused_cloud.npy", sup.acc.points)
        rig.close()
        bus.stop()
    print(f"run saved: {outdir}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run, "teach": teach})
```

Note: `RealRig` here is the Task 6 version; until Task 6 lands, `run()` simply cannot be invoked (fine — the test only imports `install_sigint`). Keep the import inside `run()` so `install_sigint` is importable standalone.

- [ ] **Step 4: Run the new test — PASS.** Whole file: `conda run -n robo python inspection/tests/test_machine.py` all green.
- [ ] **Step 5: Commit** — `"run/app: composition root, SIGINT safe shutdown, teach moved"`

---

### Task 6: RealRig v2 — camera worker integration, locked recv, fresh-frame capture

**Files:**
- Modify: `inspection/run/rigs.py` (add `RealRig`), `inspection/perception/camera.py` (extract `grab_aligned`)
- Test: append `test_grab_aligned_shape_contract` to `inspection/tests/test_rigs.py` (structure-only; no hardware in CI)

**Interfaces:**
- Consumes: `UR5eArm`, `open_camera`, `session_metadata`, `t_flange_cam`, `save_bundle`, `CameraWorker` (Task 2).
- Produces: `RealRig(world, stop_event, outdir, ip=ROBOT_IP)` — `q()` (recv behind `self._recv_lock`), `move(path)` = `arm.execute(path, world=self.world, stop_event=self.stop_event)`, `start_camera(pub, hz=10.0)`, `capture(pose_id)` (uses `CameraWorker.fresh_bundle(min_new=3)` — the v1 `flush=3` semantics), `close()`. `camera.grab_aligned(pipe, align) -> {"rgb", "depth_raw"}` — the single-frame core, shared by `capture_bundle` and `RealRig`.

- [ ] **Step 1: Failing test** — `grab_aligned` must exist and `capture_bundle` must now delegate to it:

```python
def test_grab_aligned_shape_contract():
    import inspect
    from inspection.perception import camera
    assert callable(getattr(camera, "grab_aligned", None))
    src = inspect.getsource(camera.capture_bundle)
    assert "grab_aligned" in src, "capture_bundle must delegate to grab_aligned"
```

- [ ] **Step 2: Run — fails** (`grab_aligned` missing).
- [ ] **Step 3: Implement.**
  - `camera.py`: extract the existing wait→align→to-numpy body of `capture_bundle` into `grab_aligned(pipe, align)` returning `{"rgb", "depth_raw"}` with identical dtypes/keys; `capture_bundle` keeps its flush/settle loop but builds its return from `grab_aligned`. Behavior identical.
  - `rigs.py` `RealRig` — port from `loop.py`, with these changes:

```python
class RealRig:
    """UR5e + wrist D405. Camera pipe is owned by a CameraWorker; recv is
    shared with the executor's safety polls behind _recv_lock."""

    def __init__(self, world, stop_event, outdir, ip=ROBOT_IP):
        from inspection.motion.execute import UR5eArm
        from inspection.perception.camera import (
            WRIST_SERIAL, grab_aligned, open_camera, session_metadata,
            t_flange_cam)
        self.world, self.stop_event = world, stop_event
        self.outdir = Path(outdir)
        self.arm = UR5eArm(ip)
        self._recv_lock = threading.Lock()
        # wrap the arm's recv reads: same lock for q() and safety polls
        arm_safety = self.arm._safety_ok
        self.arm._safety_ok = lambda: self._locked(arm_safety)
        try:
            self.pipe, profile, self.align, self.depth_scale = open_camera()
            meta = session_metadata(profile, self.depth_scale, WRIST_SERIAL)
            self.outdir.mkdir(parents=True, exist_ok=True)
            (self.outdir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
            self.intr = meta["intrinsics"]["ir_left"]
            self._T_fc = t_flange_cam()
            self._ik = UR5eIK()
            self._grab = lambda: grab_aligned(self.pipe, self.align)
            self.camera = None
        except Exception:
            if hasattr(self, "pipe"):
                try: self.pipe.stop()
                except Exception: pass
            self.arm.close()
            raise

    def _locked(self, fn):
        with self._recv_lock:
            return fn()

    def q(self):
        with self._recv_lock:
            return self.arm.q()

    def start_camera(self, pub, hz=10.0):
        self.camera = CameraWorker(self._grab, publish=pub.publish_frame, hz=hz)
        self.camera.start()

    def move(self, path):
        return self.arm.execute(path, world=self.world, stop_event=self.stop_event)

    def capture(self, pose_id):
        from inspection.perception.capture import save_bundle
        bundle = self.camera.fresh_bundle(min_new=3) if self.camera \
            else self._grab()
        q = self.q()
        T_bf = self._ik.fk(q)
        pose = {"joints_rad": q, "T_base_flange": T_bf,
                "T_base_cam": T_bf @ self._T_fc}
        d = save_bundle(self.outdir, pose_id, bundle, pose)
        return {"dir": str(d), "rgb": bundle["rgb"],
                "depth_raw": bundle["depth_raw"],
                "T_base_cam": pose["T_base_cam"], "q": q}

    def close(self):
        if self.camera: self.camera.stop()
        try: self.pipe.stop()
        except Exception: pass
        self.arm.close()
```

  - Implementation-time verification (design flagged it): search ur_rtde docs for documented thread-safety of `RTDEReceiveInterface` (context7 `/UniversalRobots/RTDE` or web). The lock stays regardless; note the finding in the commit message.
- [ ] **Step 4: Run `test_rigs.py` — PASS.**
- [ ] **Step 5: Commit** — `"rigs: RealRig v2 — CameraWorker owns the pipe, recv behind a lock"`

---

### Task 7: Mock v2 — the real Supervisor on a real bus, no hardware

**Files:**
- Rewrite: `inspection/ui/mock.py` (keep `cup_cloud`, `mock_frame` helpers; delete the scripted `run_mock`)
- Modify: `inspection/ui/app.py` (`mock` command)
- Test: `inspection/tests/test_mock_bus.py` (new) — end-to-end over a real websocket

**Interfaces:**
- Consumes: `Supervisor`, `FakeRig`, `PortholeBus` (+ its `commands()`), `DEMO_PARK`.
- Produces: `inspection.ui.mock.start_mock(bus, pub, outdir, seed=0) -> Supervisor` — wires FakeRig (start q offset from survey so boot needs a move, `speed=0.6` so executing is watchable/stoppable), camera `CameraWorker(grab=lambda: {"rgb": mock_frame(time.time())}, publish=pub.publish_frame)`, `PoseStreamer`, command pump, `publish_world`, and returns the (already `run()`-ing on a daemon thread) supervisor.

- [ ] **Step 1: Write the failing test** — drives the mock through a real websocket client:

```python
#!/usr/bin/env python3
"""End-to-end: commands over a real websocket drive the real Supervisor.
Run: p inspection/tests/test_mock_bus.py"""
import json
import tempfile
import time
from pathlib import Path

from porthole import PortholeBus
from inspection.tests.test_machine import wait_for
from inspection.ui.publisher import InspectionPublisher


def test_ws_command_reaches_previewing():
    from websockets.sync.client import connect
    from inspection.ui.mock import start_mock

    bus = PortholeBus(app="inspection", port=8899).start()
    pub = InspectionPublisher(bus)
    sup = start_mock(bus, pub, Path(tempfile.mkdtemp()) / "run")
    with connect("ws://127.0.0.1:8899") as ws:
        ws.send(json.dumps({"cmd": "view/request", "target": "survey"}))
        wait_for(lambda: sup.phase in ("planning", "previewing"), timeout=30,
                 msg="command never reached the supervisor")
        wait_for(lambda: sup.phase == "previewing", timeout=60, msg="previewing")
        ws.send(json.dumps({"cmd": "run/exit"}))
        wait_for(lambda: sup.phase == "done", timeout=15, msg="exit")
    bus.stop()
    print("OK test_mock_bus")


if __name__ == "__main__":
    test_ws_command_reaches_previewing()
```

- [ ] **Step 2: Run — fails** (`ImportError: start_mock`).
- [ ] **Step 3: Implement.** `mock.py::start_mock`: build `FakeRig(DEMO_PARK + radians([0,0,0,0,0,8]), speed=0.6)`, `Supervisor(rig, pub, outdir, q_survey=DEMO_PARK.copy(), seed=seed)`, `pub.publish_world(sup.world)`, `CameraWorker` + `PoseStreamer` (`poses.active = sup.pose_active`) started, pump thread (`for c in bus.commands(): sup.events.put(c)`), `threading.Thread(target=sup.run, daemon=True).start()`, return `sup`. In `inspection/ui/app.py::mock`: replace the `drive()` scripted thread with `start_mock(bus, pub, Path(tempfile.mkdtemp())/"mock-run")`; drop `turns`/`loop_forever` params; keep ports/window handling as is.
- [ ] **Step 4: Run — PASS.** Then eyeball it for real: `conda run -n robo python inspection/ui/app.py mock --no_window` → click nothing yet (no ActionsPanel), but `scene/description` + camera panels must show the fake cell (browser at the printed URL).
- [ ] **Step 5: Commit** — `"ui/mock v2: real Supervisor + FakeRig behind the real bus"`

---

### Task 8: ActionsPanel + new marker states (frontend)

**Files:**
- Create: `inspection/ui/src/panels/ActionsPanel.tsx`
- Modify: `inspection/ui/src/panels/CloudInspectPanel.tsx` (extend `ViewCellState` + colour map), `inspection/ui/src/panelDefinitions.ts`, `inspection/ui/src/InspectionApp.tsx` (register + dock the panel)

**Interfaces:**
- Consumes: `useBus()` (`BusClient.send({cmd, …}) -> boolean`), `useTopicPayload('views/state')`, `useTopicPayload('run/status')` from `@porthole/framework`; `ViewCellState` from CloudInspectPanel.
- Produces: commands `{"cmd": "view/request", "target": "survey" | [h, v]}`, `{"cmd": "view/confirm", "target": …}`, `{"cmd": "run/stop"}`, `{"cmd": "run/exit"}` — names must match Task 3's dispatcher exactly.

- [ ] **Step 1: Extend states.** In `CloudInspectPanel.tsx`: `ViewCellState` union += `'pending' | 'previewing'`; add to its colour map: `pending: '#b0a0f0'`, `previewing: '#f08bd4'` (same hex as publisher `STATE_COLOR` — grep the file for the existing state→colour table and follow its shape).

- [ ] **Step 2: Write `ActionsPanel.tsx`** (pure IO; no local button state — Global Constraints):

```tsx
/**
 * Operator actions: survey + view grid + Stop + Exit. Pure IO — every button
 * renders from `views/state` + `run/status`; a click sends a command and the
 * truth comes back on the bus. See ../AGENTS.md and the v2 design doc.
 */
import { useBus, useTopicPayload } from '@porthole/framework';
import type { PanelProps } from '@porthole/framework';
import type { ViewCellState } from './CloudInspectPanel';

interface ViewsState {
  survey?: { state: ViewCellState };
  cells: { h: number; v: number; state: ViewCellState }[];
}
interface RunStatus { phase?: string; target?: unknown }

const ACTIONABLE: ViewCellState[] = ['available', 'blocked', 'pending', 'previewing'];

function label(state: ViewCellState): string {
  if (state === 'pending') return '…';
  if (state === 'previewing') return 'preview';
  return '';
}

export function ActionsPanel(_props: PanelProps) {
  const bus = useBus();
  const views = useTopicPayload<ViewsState>('views/state');
  const status = useTopicPayload<RunStatus>('run/status') ?? {};
  const send = (cmd: string, target?: unknown) =>
    bus.send(target === undefined ? { cmd } : { cmd, target });
  const press = (state: ViewCellState, target: 'survey' | [number, number]) =>
    send(state === 'previewing' ? 'view/confirm' : 'view/request', target);

  const cells = views?.cells ?? [];
  const rows = [...new Set(cells.map((c) => c.v))].sort((a, b) => b - a);
  const survey = views?.survey?.state;
  return (
    <div className="actions-panel">
      <div className="actions-status">{String(status.phase ?? '—')}</div>
      {survey && (
        <button
          className={`act act-${survey}`}
          disabled={!ACTIONABLE.includes(survey)}
          onClick={() => press(survey, 'survey')}
        >
          survey {label(survey)}
        </button>
      )}
      <div className="actions-grid">
        {rows.map((v) => (
          <div className="actions-row" key={v}>
            {cells.filter((c) => c.v === v).sort((a, b) => a.h - b.h).map((c) => (
              <button
                key={`${c.h}-${c.v}`}
                className={`act act-${c.state}`}
                disabled={!ACTIONABLE.includes(c.state)}
                onClick={() => press(c.state, [c.h, c.v])}
                title={`h${c.h} v${c.v}: ${c.state}`}
              >
                {c.h},{c.v} {label(c.state)}
              </button>
            ))}
          </div>
        ))}
      </div>
      <div className="actions-footer">
        <button className="act act-stop" disabled={status.phase !== 'executing'}
                onClick={() => send('run/stop')}>STOP</button>
        <button className="act act-exit" onClick={() => send('run/exit')}>exit</button>
      </div>
    </div>
  );
}
```

Add minimal styles to `inspection/ui/src/inspection.css` following its existing class conventions: `.actions-grid` flex column, `.act` compact button, `.act-pending`/`.act-previewing` background = the two new hexes, `.act-stop` red. Data only — **no explanatory text in the panel** (Anton's standing rule). Layout is provisional until Anton's drawing; keep all layout in CSS, not logic.

- [ ] **Step 3: Register.** In `panelDefinitions.ts` add an `actions` entry exactly parallel to the existing four (id, title `actions`, component `ActionsPanel`); in `InspectionApp.tsx` add it to the dock layout beside the log panel (follow how the four existing panels are placed).
- [ ] **Step 4: Build + look.** `cd inspection/ui && npm run build` (must be clean). Then `conda run -n robo python inspection/ui/app.py mock --no_window`, open the URL, and drive one full fake cycle by mouse: survey → preview loops in 3D → survey again → fake arm crawls → grid populates → pick a cell → confirm → cell goes visited, image appears on the sphere panel. Stop button live only during `executing`.
- [ ] **Step 5: Commit** — `"ui: ActionsPanel — two-press flow, stop/exit; pending/previewing states"`

---

### Task 9: `npm run check` — headless proof the panel exists and commands flow

**Files:**
- Modify: `inspection/ui/tools/verify-ui.mjs` (or the file `npm run check` invokes — read `package.json` first)

**Interfaces:** consumes the mock (Task 7) and ActionsPanel (Task 8).

- [ ] **Step 1: Read the existing harness** (`inspection/ui/package.json` → the check script → its file). Understand how the 16 existing checks boot the mock and screenshot.
- [ ] **Step 2: Add two checks, following the harness's existing check pattern verbatim:**
  1. **actions-panel-renders**: after boot, the DOM contains a `.actions-panel` with a `button.act-available` or `button.act-visited` for survey (the mock publishes `survey` from the first `views/state`).
  2. **click-sends-command**: click the survey button; within 5 s the supervisor leaves `idle` (observable UI-side: `run/status.phase` text in `.actions-status` becomes `planning` or `previewing`). This proves the whole path: DOM click → `bus.send` → websocket → pump → dispatcher.
- [ ] **Step 3: Run `npm run check`** — all previous checks + 2 new ones green. If a previous check screenshotted a fixed layout, update its expectation for the fifth panel.
- [ ] **Step 4: Commit** — `"ui/check: actions panel render + command round-trip"`

---### Task 10: Retire v1 — delete loop.py & old test, docs to v2

**Files:**
- Delete: `inspection/run/loop.py`, `inspection/tests/test_loop_fake.py`
- Modify: `inspection/ui/AGENTS.md`, `inspection/AGENTS.md`, `inspection/run/decider.py` (docstring only)

**Interfaces:** none new — this task removes and documents.

- [ ] **Step 1: Verify nothing imports the deleted modules:** `grep -rn "run.loop\|run import loop\|from inspection.run.loop" --include="*.py" .` → only hits inside the two files being deleted (and docs). Fix any stragglers first.
- [ ] **Step 2: Delete** `inspection/run/loop.py` and `inspection/tests/test_loop_fake.py` (`git rm`). `teach` lives in `app.py` since Task 5; `survey_pose.json` stays where it is.
- [ ] **Step 3: Update docs.**
  - `inspection/ui/AGENTS.md`: §"This module publishes. It does not decide." → v2 reality: commands exist, backend validates everything (quote the design doc's stop hierarchy); panels table += actions panel; §3 marker table += `pending`/`previewing` rows with the two hexes; §5 payload += `survey` entry; §6 phase list ← new vocabulary; §8 wiring table ← replaced by a pointer to `machine.py` + the four commands; §9 run commands ← `p inspection/run/app.py run …` and the new mock invocation.
  - `inspection/AGENTS.md`: wherever it describes the run flow/loop, point to `run/app.py` + the design doc.
  - `decider.py` docstring: "parked for v2-AI; not wired since loop v2 — see 2026-08-20 design."
- [ ] **Step 4: Full sweep** — every remaining test green:

```bash
conda run -n robo python inspection/tests/test_publisher_v2.py
conda run -n robo python inspection/tests/test_rigs.py
conda run -n robo python inspection/tests/test_machine.py
conda run -n robo python inspection/tests/test_mock_bus.py
conda run -n robo python inspection/tests/test_decider.py
cd inspection/ui && npm run check
```

- [ ] **Step 5: Commit** — `"retire loop v1: terminal flow deleted, docs point at the v2 machine"`

---

### Task 11: Hardware smoke (Anton present — do not run unattended)

**Files:** none — checklist only. Requires: UR5e in Remote Control, pendant e-stop within reach, D405 plugged.

- [ ] `conda run -n robo python inspection/motion/execute.py preflight` → `go: True`.
- [ ] `conda run -n robo python inspection/run/app.py run --outdir=data/runs/$(date +%d%m)-ui-smoke` — window opens, camera live before any motion.
- [ ] Survey two-press: preview loops → confirm → arm moves with live mirror → capture 000 → sphere + grid appear.
- [ ] One cell two-press full cycle → visited + image on sphere.
- [ ] Start a cell move and press **STOP** mid-motion → arm decelerates, phase back to `idle`, grid usable, chosen cell not visited.
- [ ] Ctrl-C during `idle` → clean shutdown, `run.json` + `fused_cloud.npy` present.
- [ ] Anything unexpected: stop, record symptom, fix before demo claims.

---

## Self-Review (performed at plan-writing time)

- **Spec coverage:** FR1 (retained topics — existing bus behavior + Task 3 `_publish_all`), FR2 (Tasks 2/6/7), FR3 (pose streamer + phase strip, Tasks 2/3/8), FR4–FR7 (Tasks 3/4/8), FR8 (Tasks 4/8), FR9 (Task 4 settle path), FR10 (Tasks 4/5), FR11 (Task 3 validation + Task 7 ws test), FR12 (Task 5). Retirements: Task 10. Hardware proof: Task 11.
- **Placeholder scan:** Task 9 depends on reading the existing harness first by design (its pattern is authoritative); all other steps carry concrete code.
- **Type consistency:** command names, event dicts, `publish_views` kwargs, and `ViewCellState` values cross-checked across Tasks 1/3/4/7/8.
