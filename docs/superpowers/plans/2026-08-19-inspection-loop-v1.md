# Inspection Loop v1 (POC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the D3 inspection loop on the real UR5e with a human in the
model slot: boot to a taught survey pose, seed the object, then repeat
comment → menu → plan → 3D preview → approve → move → capture → fuse.

**Architecture:** `run/loop.py` orchestrates existing modules (viewsphere →
plan_viewpoint → UR5eArm → capture → geometry). `run/decider.py` is the AI
seam (TerminalDecider v1). `motion/execute.py` becomes interruptible
(async moveJ + poll + software stop). Every motion is meshcat-previewed and
terminal-approved before it touches the robot.

**Tech Stack:** Python (conda env `robo`), ur_rtde, pinocchio/meshcat via
`cell/world.py`, OMPL ladder via `motion/plan.py`, pyrealsense2 via
`perception/camera.py`. No new dependencies.

**Spec:** `inspection/2026-08-19-loop-v1-design.md` (parent:
`inspection/2026-08-12-ai-inspection-project-definition.md`, sections D3/D4).

## Global Constraints

- Run commands as `p <file-path>` (user's alias; robo env is already active
  in their shell). Never `python`, never `-m`. For YOUR verification runs
  use `~/miniconda3/envs/robo/bin/python <file>` — same interpreter.
- NEVER run anything that contacts the robot (192.168.2.50) or the camera —
  no RTDE/dashboard/realsense constructors. Anton runs those commands
  himself. Headless tests must run without robot or camera attached.
- No sys.path hacks — `inspection` is pip-installed editable.
- Tests are plain `p`-runnable assert scripts (no pytest in the env): each
  test file ends with a `main()` that runs all test functions and prints OK.
- Do not commit `inspection/run/survey_pose.json` (taught on hardware later;
  it does not exist yet).
- Files `inspection/view/viewsphere.py`, `inspection/motion/{plan,rrt,
  direct,smooth,bench,ik}.py`, `inspection/cell/{world,viewer}.py`,
  `inspection/{cell,motion,view}/AGENTS.md` carry uncommitted parallel work.
  You may IMPORT from them freely but do not edit or commit them except the
  two AGENTS.md edits in Task 6, and never `git add -A`.
- Speed caps stay at SPEED_RAD_S=0.25, ACC_RAD_S2=0.5, SPEED_SLIDER=0.25.

## Existing interfaces you will consume (verified 2026-08-19)

```python
# inspection/motion/plan.py
plan_viewpoint(world, ik, q_from, T_target, seed=0, n_branches=3,
               step=DEFAULT_STEP, tiers=(1, 2)) -> (path | None, report_dict)
# report has: "tier", "ms", and on refusal "reason"

# inspection/view/viewsphere.py
ViewSphere(center, r=0.35, elevations=(10.0, 40.0, 70.0))   # center: base frame
sphere.reachability(world, ik) -> {(h, v): roll | None}      # ~33 ms
sphere.plan_to_cell(world, ik, q_now, h, v, seed=0, step=DEFAULT_STEP)
    -> (path | None, report, roll)
map_str(reach, elevations) -> str                            # ascii coverage map
H_BINS = 12; V_ELEVATIONS = (10.0, 40.0, 70.0)

# inspection/cell/world.py
RobotCell()                       # collision world, headless-safe
world.path_valid(path, step=DEFAULT_STEP, padding=None) -> (ok, bad_segment)
world.set_object(name, dims, pose, parent="base")   # pose = [x,y,z,r,p,y]
world.init_viewer(); world.show(q, label=""); world.replay(path, dt=0.03)

# inspection/motion/ik.py
UR5eIK().fk(q) -> 4x4 T_base_flange

# inspection/perception/camera.py
open_camera(serial=WRIST_SERIAL) -> (pipe, profile, align, depth_scale)
session_metadata(profile, depth_scale, serial) -> dict   # ["intrinsics"]["ir_left"]
capture_bundle(pipe, align) -> {"rgb","ir_left","ir_right","depth_raw",
                                "depth_aligned","timestamp"}
t_flange_cam() -> 4x4  # calibrated, OpenCV convention, left-eye frame

# inspection/perception/capture.py
save_bundle(outdir, pose_id, bundle, pose_dict_or_None) -> Path  # NNN/ dir

# inspection/cell/geometry.py
deproject(depth_u16, intr, depth_scale, ...) -> (N,3) cam-frame points
cam_to_base(points_cam, T_base_cam) -> (N,3)
crop_workspace(points); fit_table(points) -> plane[4]; above_table(points, plane)
largest_cluster(points); CloudAccumulator()  # .add(pts) .points .centroid .aabb()

# inspection/motion/execute.py (this plan modifies it)
preflight(ip) -> dict with "go"; UR5eArm(ip)  # ctor raises on NO-GO
arm.q() -> np.ndarray; arm.execute(path, world=None, step=None) -> report

# ur_rtde async API (verified against installed package)
ctrl.moveJ(list_q, speed, acc, True)      # asynchronous — returns immediately
ctrl.getAsyncOperationProgressEx().isAsyncOperationRunning() -> bool
ctrl.stopJ(2.0)                            # decelerating software stop
```

---

### Task 1: Decider data model — actions, gloss, menu, parsing

**Files:**
- Create: `inspection/run/decider.py`
- Create: `inspection/tests/__init__.py` (empty)
- Test: `inspection/tests/test_decider.py`

**Interfaces:**
- Consumes: `H_BINS`, `V_ELEVATIONS` from `inspection/view/viewsphere.py`
- Produces (Tasks 2/5 rely on exact names): dataclasses `Look(h, v)`,
  `Answer(text)`, `Quit()`, `MenuItem(h, v, gloss, visited=False)`,
  `Ctx(question, step, current_cell, map_ascii, menu, comments)`;
  functions `gloss(cur, cell, n_h=12, elevations=V_ELEVATIONS) -> str`,
  `build_menu(reach, visited, cur, elevations=V_ELEVATIONS) -> list[MenuItem]`,
  `parse_command(line) -> Look | Answer | Quit | None`

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""Tests for the decider data model. Run: p inspection/tests/test_decider.py"""
from inspection.run.decider import (Look, Answer, Quit, MenuItem, gloss,
                                    build_menu, parse_command)


def test_gloss():
    # +1 azimuth step = "one step right" (D4 example: h=5 -> h=6)
    assert gloss((5, 1), (6, 1)) == "one step right, same height"
    assert gloss((5, 1), (3, 1)) == "two steps left, same height"
    assert gloss((11, 0), (0, 0)) == "one step right, same height"  # wrap
    assert gloss((5, 1), (11, 2)) == "opposite side, higher"        # |dh|=6
    assert gloss((5, 1), (5, 0)) == "same side, lower"
    assert gloss(None, (3, 2)) == "elevation 70 deg"                # survey


def test_build_menu():
    reach = {(0, 0): 0.0, (1, 0): 0.5, (6, 0): 0.0, (2, 0): None}
    menu = build_menu(reach, visited={(0, 0)}, cur=(0, 0))
    cells = [(it.h, it.v) for it in menu]
    assert (2, 0) not in cells          # unreachable filtered
    assert (0, 0) not in cells          # visited filtered
    assert cells[0] == (1, 0)           # nearest first
    assert cells[-1] == (6, 0)          # opposite side last
    assert menu[0].gloss == "one step right, same height"


def test_parse_command():
    assert parse_command("look 6 1") == Look(6, 1)
    assert parse_command("l 6 1") == Look(6, 1)
    assert parse_command("answer no logo visible") == Answer("no logo visible")
    assert parse_command("a yes") == Answer("yes")
    assert parse_command("quit") == Quit()
    assert parse_command("look six 1") is None
    assert parse_command("look 6") is None
    assert parse_command("") is None
    assert parse_command("garbage") is None


def main():
    test_gloss(); test_build_menu(); test_parse_command()
    print("OK test_decider")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_decider.py`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (decider.py absent).

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Decider seam for the v1 loop — the AI slot from the project definition.

D3/D4 (inspection/2026-08-12-ai-inspection-project-definition.md): the
decider READs the newest capture (a comment) and DECIDEs look(h, v) or
answer(text). v1 is Anton at the terminal; BusDecider (UI toolkit) and
AgentDecider swap in later without touching loop.py.
"""
import queue
import sys
import threading
from dataclasses import dataclass, field

from inspection.view.viewsphere import H_BINS, V_ELEVATIONS

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


@dataclass(frozen=True)
class Look:
    h: int
    v: int


@dataclass(frozen=True)
class Answer:
    text: str


@dataclass(frozen=True)
class Quit:
    pass


@dataclass
class MenuItem:
    h: int
    v: int
    gloss: str
    visited: bool = False


@dataclass
class Ctx:
    """Everything the decider sees — D4: 'injected every turn, not tools'."""
    question: str
    step: int
    current_cell: tuple | None      # None before the first look (survey)
    map_ascii: str
    menu: list                      # [MenuItem], nearest first
    comments: list = field(default_factory=list)   # prior READs, oldest first


def _dh(cur_h, h, n_h):
    """Signed shortest azimuth steps cur -> cell; +1 = one step right."""
    return (h - cur_h + n_h // 2) % n_h - n_h // 2


def gloss(cur, cell, n_h=H_BINS, elevations=V_ELEVATIONS):
    """Egocentric label for cell relative to cur (D4 menu gloss)."""
    if cur is None:
        return f"elevation {elevations[cell[1]]:.0f} deg"
    dh = _dh(cur[0], cell[0], n_h)
    dv = cell[1] - cur[1]
    if abs(dh) == n_h // 2:
        side = "opposite side"
    elif dh == 0:
        side = "same side"
    else:
        n = abs(dh)
        side = f"{_WORDS[n]} step{'s' if n > 1 else ''} " \
               f"{'right' if dh > 0 else 'left'}"
    height = "same height" if dv == 0 else ("higher" if dv > 0 else "lower")
    return f"{side}, {height}"


def build_menu(reach, visited, cur, elevations=V_ELEVATIONS):
    """Feasible unvisited cells, nearest to cur first."""
    items = [MenuItem(h, v, gloss(cur, (h, v), elevations=elevations))
             for (h, v), roll in sorted(reach.items())
             if roll is not None and (h, v) not in visited]

    def key(it):
        if cur is None:
            return (it.v, it.h, 0)
        dh = abs(_dh(cur[0], it.h, H_BINS))
        return (dh + abs(it.v - cur[1]), dh, it.h)

    return sorted(items, key=key)


def parse_command(line):
    """'look 6 1'/'l 6 1' -> Look; 'answer <text>'/'a <text>' -> Answer;
    'quit'/'q' -> Quit; None on anything unparseable."""
    toks = line.strip().split()
    if not toks:
        return None
    cmd = toks[0].lower()
    if cmd in ("look", "l") and len(toks) == 3:
        try:
            return Look(int(toks[1]), int(toks[2]))
        except ValueError:
            return None
    if cmd in ("answer", "a") and len(toks) >= 2:
        return Answer(line.strip().split(None, 1)[1])
    if cmd in ("quit", "q") and len(toks) == 1:
        return Quit()
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_decider.py`
Expected: `OK test_decider`

- [ ] **Step 5: Commit**

```bash
git add inspection/run/decider.py inspection/tests/__init__.py inspection/tests/test_decider.py
git commit -m "loop v1: decider data model — actions, egocentric gloss, menu, parsing"
```

---

### Task 2: Console (stdin owner + software-stop arming) and TerminalDecider

**Files:**
- Modify: `inspection/run/decider.py` (append)
- Test: `inspection/tests/test_decider.py` (append)

**Interfaces:**
- Consumes: Task 1 definitions in the same file.
- Produces (Tasks 5/6 rely on): `Console(stream=None)` with
  `.readline(prompt="") -> str`, `.arm_stop()`, `.disarm_stop()`,
  `.stop_event` (threading.Event); `TerminalDecider(console)` with
  `.read(cap: dict) -> str` and `.decide(ctx: Ctx) -> Look | Answer | Quit`;
  `show_capture(rgb)` (cv2 stopgap viewer).

- [ ] **Step 1: Append the failing tests**

```python
import queue
import time

from inspection.run.decider import Console, TerminalDecider, Ctx, MenuItem


class FeedStream:
    """Line source the test controls in real time (stdin stand-in)."""
    def __init__(self):
        self.q = queue.Queue()

    def feed(self, line):
        self.q.put(line + "\n")

    def __iter__(self):
        while True:
            line = self.q.get()
            if line is None:
                return
            yield line


def test_console_stop_arming():
    fs = FeedStream()
    con = Console(stream=fs)
    fs.feed("first")
    assert con.readline() == "first"
    con.arm_stop()                      # robot "moving" now
    fs.feed("anything")                 # any line while armed = STOP
    assert con.stop_event.wait(timeout=1.0)
    con.disarm_stop()
    fs.feed("after")
    assert con.readline() == "after"    # queue not polluted by the stop line


def test_terminal_decider_parses_until_valid():
    fs = FeedStream()
    dec = TerminalDecider(Console(stream=fs))
    ctx = Ctx(question="logo?", step=1, current_cell=(0, 0),
              map_ascii="(map)", menu=[MenuItem(1, 0, "one step right")])
    fs.feed("nonsense")                 # rejected, re-prompts
    fs.feed("look 1 0")
    from inspection.run.decider import Look
    assert dec.decide(ctx) == Look(1, 0)
```

Add both to `main()`:
```python
    test_console_stop_arming(); test_terminal_decider_parses_until_valid()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_decider.py`
Expected: FAIL — `ImportError: cannot import name 'Console'`.

- [ ] **Step 3: Append the implementation**

```python
class Console:
    """Single owner of terminal input. A daemon thread reads lines; while
    the stop is ARMED (robot in motion) any line fires stop_event instead
    of queueing — ENTER is the software stop button. The hardware e-stop
    is unaffected and always available."""

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._q = queue.Queue()
        self.stop_event = threading.Event()
        self._armed = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self._stream:
            if self._armed:
                self.stop_event.set()
            else:
                self._q.put(line.rstrip("\n"))

    def arm_stop(self):
        self.stop_event.clear()
        self._armed = True

    def disarm_stop(self):
        self._armed = False

    def readline(self, prompt=""):
        if prompt:
            print(prompt, end="", flush=True)
        return self._q.get()


def show_capture(rgb):
    """Stopgap viewer until the UI toolkit lands: one reused cv2 window."""
    import cv2
    cv2.imshow("newest capture", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(200)        # render once; window persists between turns


class TerminalDecider:
    """Anton in the model slot. The printed ctx IS the future model prompt."""

    def __init__(self, console):
        self.console = console

    def read(self, cap):
        if cap.get("rgb") is not None:
            show_capture(cap["rgb"])
        print(f"\n[READ] capture: {cap.get('dir', '(fake)')}")
        return self.console.readline("comment> ")

    def decide(self, ctx):
        print(f"\n[DECIDE] step {ctx.step}   question: {ctx.question!r}")
        print(ctx.map_ascii)
        cur = (f"h={ctx.current_cell[0]} v={ctx.current_cell[1]}"
               if ctx.current_cell else "survey pose")
        print(f"you are at: {cur}")
        for it in ctx.menu[:12]:
            print(f"  look {it.h} {it.v}    {it.gloss}")
        if len(ctx.menu) > 12:
            print(f"  ... {len(ctx.menu) - 12} more cells on the map above")
        while True:
            act = parse_command(
                self.console.readline("look H V | answer TEXT | quit > "))
            if act is not None:
                return act
            print("could not parse — try 'look 6 1', 'answer no logo', 'quit'")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_decider.py`
Expected: `OK test_decider`

- [ ] **Step 5: Commit**

```bash
git add inspection/run/decider.py inspection/tests/test_decider.py
git commit -m "loop v1: Console with ENTER software-stop arming + TerminalDecider"
```

---

### Task 3: Interruptible executor — async moveJ, poll, software stop

**Files:**
- Modify: `inspection/motion/execute.py` (execute() body + new module fn)
- Test: `inspection/tests/test_execute_wait.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (Task 5/6 rely on): `_wait_async(running_fn, safety_fn,
  stop_event=None, poll_s=0.05, grace_s=0.3) -> "done"|"stopped"|"unsafe"`;
  `UR5eArm.execute(path, world=None, step=None, stop_event=None)` — report
  dict gains `"stopped": bool`. A stopped run RETURNS (stopped=True), it
  does not raise; "unsafe" still raises.

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""_wait_async poll logic, no robot. Run: p inspection/tests/test_execute_wait.py"""
import threading

from inspection.motion.execute import _wait_async


def _seq(values, after=False):
    """running_fn that yields `values` then `after` forever."""
    it = iter(values)
    return lambda: next(it, after)


def test_done_after_motion():
    r = _wait_async(_seq([True, True, False]), lambda: True, poll_s=0.001)
    assert r == "done"


def test_grace_prevents_premature_done():
    # controller hasn't registered the op yet: not-running at first poll,
    # then running, then finished — must NOT return done on the first poll
    r = _wait_async(_seq([False, False, True, False]), lambda: True,
                    poll_s=0.001, grace_s=10.0)
    assert r == "done"


def test_never_ran_times_out_via_grace():
    r = _wait_async(lambda: False, lambda: True, poll_s=0.001, grace_s=0.02)
    assert r == "done"          # grace expired, op never registered


def test_stopped():
    ev = threading.Event(); ev.set()
    r = _wait_async(lambda: True, lambda: True, stop_event=ev, poll_s=0.001)
    assert r == "stopped"


def test_unsafe():
    r = _wait_async(lambda: True, lambda: False, poll_s=0.001)
    assert r == "unsafe"


def main():
    test_done_after_motion(); test_grace_prevents_premature_done()
    test_never_ran_times_out_via_grace(); test_stopped(); test_unsafe()
    print("OK test_execute_wait")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_execute_wait.py`
Expected: FAIL — `ImportError: cannot import name '_wait_async'`.

- [ ] **Step 3: Implement**

Add module-level function (below the constants in execute.py):

```python
def _wait_async(running_fn, safety_fn, stop_event=None,
                poll_s=0.05, grace_s=0.3):
    """Poll an asynchronous move. Returns 'done' | 'stopped' | 'unsafe'.

    grace_s covers the window between issuing moveJ(async) and the
    controller reporting the op as running — without it a fast first poll
    sees 'not running' and declares done before the arm ever moves.
    """
    t0 = time.monotonic()
    seen_running = False
    while True:
        if stop_event is not None and stop_event.is_set():
            return "stopped"
        if not safety_fn():
            return "unsafe"
        running = running_fn()
        seen_running = seen_running or running
        if not running and (seen_running
                            or time.monotonic() - t0 > grace_s):
            return "done"
        time.sleep(poll_s)
```

Replace the motion section of `UR5eArm.execute` (keep all preconditions —
start tolerance, safety, world re-validation — exactly as they are; change
the signature line and the `try` block):

```python
    def execute(self, path, world=None, step=None, stop_event=None) -> dict:
        """Run a validated waypoint path. Returns an execution report.

        Refuses (raises, no motion) when: start mismatch, world says the
        path is no longer valid, or safety is not NORMAL. Motion is
        asynchronous per waypoint so it can be interrupted: stop_event set
        -> stopJ deceleration, report {"stopped": True} (a normal outcome,
        not an error). Safety change mid-path still raises.
        """
```

and the loop body:

```python
        t0 = time.perf_counter()
        done, stopped = 0, False
        try:
            for q_goal in path[1:]:
                if not self._safety_ok():
                    raise RuntimeError("safety mode changed mid-path")
                self.ctrl.moveJ(list(q_goal), self.speed, self.acc, True)
                outcome = _wait_async(
                    lambda: (self.ctrl.getAsyncOperationProgressEx()
                             .isAsyncOperationRunning()),
                    self._safety_ok, stop_event)
                if outcome == "stopped":
                    self.ctrl.stopJ(2.0)
                    stopped = True
                    break
                if outcome == "unsafe":
                    raise RuntimeError("safety mode changed mid-path")
                done += 1
        except Exception:
            self.stop()
            raise
        err = np.degrees(np.abs(self.q() - path[-1]).max())
        return {"waypoints_done": done, "s": time.perf_counter() - t0,
                "final_err_deg": float(err), "stopped": stopped}
```

Update the module docstring's "sequential blocking moveJ" phrase to
"sequential async moveJ (interruptible — software stop via stop_event)".

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_execute_wait.py`
Expected: `OK test_execute_wait`

- [ ] **Step 5: Verify the demo path still compiles (no robot)**

Run: `~/miniconda3/envs/robo/bin/python -c "import inspection.motion.execute as e; print(e._wait_async(lambda: False, lambda: True, grace_s=0.01))"`
Expected: `done`

- [ ] **Step 6: Commit**

```bash
git add inspection/motion/execute.py inspection/tests/test_execute_wait.py
git commit -m "execute: interruptible async moveJ with software stop (stop_event -> stopJ)"
```

---

### Task 4: Base-frame object seeding — `object_in_base`

**Files:**
- Modify: `inspection/cell/geometry.py` (one new function, after
  `object_from_view`)
- Create: `inspection/tests/synth.py` (shared synthetic scene helper)
- Test: `inspection/tests/test_geometry_seed.py`

**Interfaces:**
- Consumes: existing `deproject`, `cam_to_base`, `crop_workspace`,
  `fit_table`, `above_table`, `largest_cluster` from the same module.
- Produces (Task 5 relies on): `object_in_base(depth_u16, intr, depth_scale,
  T_base_cam) -> {"points","centroid","plane","n_scene"}` — everything in
  the BASE frame, no table rectification (the viewsphere center and the
  collision box must live where IK/collision live; `object_from_view`'s
  rectified frame is for accumulation/display, wrong for planning).
  Also `inspection/tests/synth.py`: `synth_capture() ->
  (depth_u16, intr, depth_scale, T_base_cam, expected_centroid)`.

- [ ] **Step 1: Write the shared synthetic scene**

```python
#!/usr/bin/env python3
"""Synthetic overhead D405 scene: flat table + one box. Shared by the
geometry seed test and the loop smoke test — no camera, no robot."""
import numpy as np

INTR = {"fx": 400.0, "fy": 400.0, "ppx": 424.0, "ppy": 240.0}
DEPTH_SCALE = 0.001            # 1 mm per unit, like the D405
# camera 0.5 m above the table at x=0.30, looking straight down (cv frame)
T_BASE_CAM = np.array([[1.0, 0, 0, 0.30],
                       [0, -1.0, 0, 0.00],
                       [0, 0, -1.0, 0.50],
                       [0, 0, 0, 1.0]])


def synth_capture():
    """Returns (depth_u16, intr, depth_scale, T_base_cam, expected_centroid).

    Table plane fills the frame at 0.5 m; an 80x80 px box sits 5 cm proud
    at the principal point -> object centroid ~ (0.30, 0.00, ~0.025-0.05)
    in the base frame (top face at z=0.05).
    """
    depth = np.full((480, 848), 500, dtype=np.uint16)      # 0.5 m table
    depth[200:280, 384:464] = 450                          # box top at 0.45 m
    return depth, INTR, DEPTH_SCALE, T_BASE_CAM, np.array([0.30, 0.0, 0.05])
```

- [ ] **Step 2: Write the failing test**

```python
#!/usr/bin/env python3
"""object_in_base on a synthetic scene. Run: p inspection/tests/test_geometry_seed.py"""
import numpy as np

from inspection.cell.geometry import object_in_base
from inspection.tests.synth import synth_capture


def test_seed_centroid_in_base_frame():
    depth, intr, scale, T_bc, expected = synth_capture()
    view = object_in_base(depth, intr, scale, T_bc)
    c = view["centroid"]
    assert c is not None and len(view["points"]) > 100
    assert abs(c[0] - expected[0]) < 0.02          # x ~ 0.30
    assert abs(c[1] - expected[1]) < 0.02          # y ~ 0.00
    assert 0.03 < c[2] < 0.07                      # box top face at 0.05
    # plane normal points up and the plane sits near z=0 in base frame
    assert view["plane"][2] > 0.9
    assert abs(view["plane"][3]) < 0.02


def main():
    test_seed_centroid_in_base_frame()
    print("OK test_geometry_seed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_geometry_seed.py`
Expected: FAIL — `ImportError: cannot import name 'object_in_base'`.

- [ ] **Step 4: Implement in geometry.py**

```python
def object_in_base(depth_u16: np.ndarray, intr: dict, depth_scale: float,
                   T_base_cam: np.ndarray) -> dict:
    """One view -> object candidate in the BASE frame (no rectification).

    The loop plans in the base frame: the viewsphere center and the
    collision box must not live in the table-rectified frame that
    object_from_view returns. UR5e FK is mm-accurate, so skipping the
    per-view tilt correction is safe here (it existed for SO-101 flex).
    """
    pts_cam = deproject(depth_u16, intr, depth_scale)
    pts = crop_workspace(cam_to_base(pts_cam, T_base_cam))
    if len(pts) < 100:
        raise RuntimeError(f"only {len(pts)} workspace points — bad view?")
    plane = fit_table(pts)
    above = above_table(pts, plane)
    obj = largest_cluster(above) if len(above) else above
    return {
        "points": obj,
        "centroid": obj.mean(axis=0) if len(obj) else None,
        "plane": plane,
        "n_scene": len(pts),
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_geometry_seed.py`
Expected: `OK test_geometry_seed`

- [ ] **Step 6: Commit**

```bash
git add inspection/cell/geometry.py inspection/tests/synth.py inspection/tests/test_geometry_seed.py
git commit -m "geometry: object_in_base — base-frame seeding for the loop (no rectification)"
```

---

### Task 5: Loop core — boot, turn machine, run record (fake rig)

**Files:**
- Create: `inspection/run/loop.py` (core only; CLI + real rig are Task 6)
- Test: `inspection/tests/test_loop_fake.py`

**Interfaces:**
- Consumes: Tasks 1/2 (`Ctx`, `build_menu`, actions, `Console`), Task 4
  (`object_in_base`, `synth.py`), plus `plan_viewpoint`, `ViewSphere`,
  `map_str`, `RobotCell`, `UR5eIK`, `CloudAccumulator`.
- Produces (Task 6 relies on): `Loop(rig, decider, console, outdir,
  q_survey, r=0.35, max_turns=12, question="", seed=0, world=None,
  ik=None)` with `.run()`; the rig contract (duck-typed, five members):
  `rig.q() -> np.ndarray`, `rig.preview(path)`, `rig.move(path) -> report`
  (report has `"stopped"`), `rig.capture(pose_id) -> {"dir","rgb",
  "depth_raw","T_base_cam","q"}`, attrs `rig.intr` (ir_left dict) and
  `rig.depth_scale`.

- [ ] **Step 1: Write the failing smoke test**

```python
#!/usr/bin/env python3
"""Headless dry-run of the whole loop: real world+IK+planner, fake arm and
camera, scripted decider. Run: p inspection/tests/test_loop_fake.py"""
import json
import tempfile
from pathlib import Path

import numpy as np

from inspection.motion.plan import DEMO_PARK
from inspection.run.decider import Console, Look, Answer
from inspection.run.loop import Loop
from inspection.tests.synth import synth_capture
from inspection.tests.test_decider import FeedStream


class FakeRig:
    """Arm+camera stand-in honouring the five-member rig contract."""

    def __init__(self, q0):
        self._q = np.asarray(q0, dtype=float)
        depth, intr, scale, T_bc, _ = synth_capture()
        self._depth, self._T_bc = depth, T_bc
        self.intr, self.depth_scale = intr, scale
        self.moves = []

    def q(self):
        return self._q.copy()

    def preview(self, path):
        pass                                    # meshcat replay in real rig

    def move(self, path):
        self.moves.append(len(path))
        self._q = np.asarray(path[-1], dtype=float)
        return {"waypoints_done": len(path) - 1, "s": 0.0,
                "final_err_deg": 0.0, "stopped": False}

    def capture(self, pose_id):
        return {"dir": f"(fake {pose_id:03d})", "rgb": None,
                "depth_raw": self._depth, "T_base_cam": self._T_bc,
                "q": self._q.copy()}


class ScriptDecider:
    """READ: canned comment. DECIDE: first menu cell once, then answer."""

    def __init__(self):
        self.decisions = 0

    def read(self, cap):
        return f"scripted comment {self.decisions}"

    def decide(self, ctx):
        self.decisions += 1
        if self.decisions == 1:
            assert ctx.menu, "no reachable cells offered"
            return Look(ctx.menu[0].h, ctx.menu[0].v)
        return Answer("scripted: no logo")


def test_full_fake_run():
    q_survey = DEMO_PARK.copy()
    q_start = q_survey.copy()
    q_start[5] += np.radians(8)             # boot must plan+move to survey
    fs = FeedStream()
    con = Console(stream=fs)
    for _ in range(4):                      # approvals: boot move + one look
        fs.feed("y")
    outdir = Path(tempfile.mkdtemp()) / "run1"
    loop = Loop(FakeRig(q_start), ScriptDecider(), con, outdir,
                q_survey=q_survey, max_turns=5, question="logo?")
    loop.run()

    rec = json.loads((outdir / "run.json").read_text())
    assert rec["question"] == "logo?"
    acts = [t["action"] for t in rec["turns"]]
    assert any(a.startswith("look") for a in acts)
    assert acts[-1].startswith("answer")
    assert rec["turns"][0]["comment"] == "scripted comment 0"
    assert loop.acc.centroid is not None    # cloud fused


def main():
    test_full_fake_run()
    print("OK test_loop_fake")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_loop_fake.py`
Expected: FAIL — `ModuleNotFoundError: inspection.run.loop`.

- [ ] **Step 3: Implement the loop core**

```python
#!/usr/bin/env python3
"""Inspection loop v1 (POC) — the D3 orchestrator with a human decider.

Spec: inspection/2026-08-19-loop-v1-design.md. Boot: plan to the taught
survey pose (approved, like every motion), capture 000, seed the object.
Turn: READ comment -> DECIDE look/answer -> plan -> meshcat preview ->
approve -> move (ENTER = software stop) -> capture -> fuse -> re-center.

The Decider is the AI seam; the rig bundles arm+camera so tests can run
the full loop headless (see tests/test_loop_fake.py).
"""
import json
import time
from pathlib import Path

import numpy as np

from inspection.cell.geometry import CloudAccumulator, object_in_base
from inspection.cell.world import RobotCell
from inspection.motion.ik import UR5eIK
from inspection.motion.plan import plan_viewpoint
from inspection.run.decider import Answer, Ctx, Look, Quit, build_menu
from inspection.view.viewsphere import ViewSphere, map_str

START_TOL_RAD = 0.02        # same gate as execute.py


class Loop:
    def __init__(self, rig, decider, console, outdir, q_survey, r=0.35,
                 max_turns=12, question="", seed=0, world=None, ik=None):
        self.rig, self.decider, self.console = rig, decider, console
        self.outdir = Path(outdir)
        self.q_survey = np.asarray(q_survey, dtype=float)
        self.r, self.max_turns = r, max_turns
        self.question, self.seed = question, seed
        self.world = world if world is not None else RobotCell()
        self.ik = ik if ik is not None else UR5eIK()
        self.acc = CloudAccumulator()
        self.sphere = None
        self.current = None                 # (h, v) after the first look
        self.visited = set()
        self.plan_failed = set()            # cleared after every real move
        self.turns = []
        self.latest_cap = None
        self.t0 = time.time()

    # ---------------------------------------------------------------- boot
    def boot(self):
        q_now = self.rig.q()
        if np.abs(q_now - self.q_survey).max() > START_TOL_RAD:
            print("boot: planning current -> survey pose")
            path, rep = plan_viewpoint(self.world, self.ik, q_now,
                                       self.ik.fk(self.q_survey),
                                       seed=self.seed)
            if path is None:
                print(f"boot REFUSED: {rep.get('reason', rep)}")
                return False
            if not self._approve_and_move(path, "boot -> survey"):
                return False
        cap = self.rig.capture(0)
        self.latest_cap = cap
        view = object_in_base(cap["depth_raw"], self.rig.intr,
                              self.rig.depth_scale, cap["T_base_cam"])
        if view["centroid"] is None:
            print("boot: NO OBJECT above the table")
            return False
        self.acc.add(view["points"])
        self._recenter()
        print(f"boot ok: object at {np.round(self.sphere.center, 3).tolist()}, "
              f"{len(view['points'])} pts")
        return True

    # ---------------------------------------------------------------- turn
    def turn(self):
        step = len(self.turns) + 1
        comment = self.decider.read(self.latest_cap)
        reach = self.sphere.reachability(self.world, self.ik)
        ctx = Ctx(question=self.question, step=step,
                  current_cell=self.current,
                  map_ascii=map_str(reach, self.sphere.elevations),
                  menu=build_menu(reach, self.visited | self.plan_failed,
                                  self.current,
                                  elevations=self.sphere.elevations),
                  comments=[t["comment"] for t in self.turns] + [comment])
        act = self.decider.decide(ctx)
        rec = {"step": step, "comment": comment, "t": time.time() - self.t0}
        self.turns.append(rec)

        if isinstance(act, (Answer, Quit)):
            rec["action"] = ("answer " + act.text) if isinstance(act, Answer) \
                else "quit"
            return False

        rec["action"] = f"look {act.h} {act.v}"
        if reach.get((act.h, act.v)) is None:
            rec["result"] = "unreachable cell — pick again"
            print(rec["result"])
            return True
        path, prep, roll = self.sphere.plan_to_cell(
            self.world, self.ik, self.rig.q(), act.h, act.v, seed=self.seed)
        if path is None:
            self.plan_failed.add((act.h, act.v))
            rec["result"] = "plan refused — cell dropped from menu this round"
            print(rec["result"])
            return True
        rec["tier"] = prep["tier"]
        if not self._approve_and_move(path, rec["action"]):
            rec["result"] = "not executed (refused or software stop)"
            return True
        self.visited.add((act.h, act.v))
        self.current = (act.h, act.v)
        self.plan_failed.clear()            # new q — refused cells may work now

        cap = self.rig.capture(step)
        self.latest_cap = cap
        view = object_in_base(cap["depth_raw"], self.rig.intr,
                              self.rig.depth_scale, cap["T_base_cam"])
        if len(view["points"]):
            self.acc.add(view["points"])
        self._recenter()
        rec["result"] = (f"+{len(view['points'])} pts, "
                         f"fused {len(self.acc.points)}")
        self._save()                        # crash-safe: record every turn
        return True

    # ------------------------------------------------------------- helpers
    def _approve_and_move(self, path, label):
        self.rig.preview(path)
        ans = self.console.readline(
            f"{label}: {len(path)} waypoints previewed — approve? [y/n] ")
        if ans.strip().lower() != "y":
            print("refused — nothing sent to the robot")
            return False
        print("executing (ENTER = software stop)")
        self.console.arm_stop()
        try:
            rep = self.rig.move(path)
        finally:
            self.console.disarm_stop()
        if rep.get("stopped"):
            print("SOFTWARE STOP — arm halted; back to the menu "
                  "(plans restart from wherever it stopped)")
            return False
        return True

    def _recenter(self):
        center = self.acc.centroid
        self.sphere = ViewSphere(center, r=self.r)
        mn, mx = self.acc.aabb()
        dims = np.maximum(mx - mn + 0.04, 0.05)         # 2 cm margin each side
        mid = (mn + mx) / 2
        self.world.set_object("object", dims.tolist(),
                              [*mid.tolist(), 0.0, 0.0, 0.0], parent="base")

    def _save(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "run.json").write_text(json.dumps(
            {"question": self.question, "q_survey": self.q_survey.tolist(),
             "r": self.r, "turns": self.turns}, indent=2) + "\n")

    # ----------------------------------------------------------------- run
    def run(self):
        ok = self.boot()
        if ok:
            while len(self.turns) < self.max_turns:
                if not self.turn():
                    break
        self._save()
        if len(self.acc.points):
            np.save(self.outdir / "fused_cloud.npy", self.acc.points)
        print(f"run saved: {self.outdir}  ({len(self.turns)} turns, "
              f"{len(self.acc.points)} fused pts)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_loop_fake.py`
Expected: `OK test_loop_fake` (first run is slow — OMPL may plan the boot
move; the synthetic object sits at x=0.30 y=0.0 where reachability is known
good, 29/36 cells).

Debug notes if it fails:
- `set_object` called twice with the same name: check
  `inspection/cell/world.py` `set_object` — if it appends rather than
  replaces, remove the old body first (read the file; there is a matching
  removal or replace mechanism; if truly absent, name objects
  `object` once and update via the same pinocchio geometry handle — fix
  inside `_recenter`, smallest change that makes re-seeding idempotent).
- No menu cells: print `map_str` output — if all blocked, the synthetic
  centroid may be outside the calibrated reachable zone; move the synthetic
  box (synth.py T_BASE_CAM x-offset) toward the cell's known-good
  CUP_POS = (0.09, -0.28) and update the expected centroid in both tests.

- [ ] **Step 5: Re-run the other tests (no regressions)**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_decider.py && ~/miniconda3/envs/robo/bin/python inspection/tests/test_execute_wait.py && ~/miniconda3/envs/robo/bin/python inspection/tests/test_geometry_seed.py`
Expected: three OK lines.

- [ ] **Step 6: Commit**

```bash
git add inspection/run/loop.py inspection/tests/test_loop_fake.py
git commit -m "loop v1: orchestrator core — boot/turn machine, approval gate, run record"
```

---

### Task 6: CLI + RealRig + teach; retire explore.py

**Files:**
- Modify: `inspection/run/loop.py` (append CLI section)
- Delete: `inspection/run/explore.py` (SO-101 era, imports vbti — superseded)
- Modify: `inspection/run/AGENTS.md` (rewrite Files section)
- Modify: `inspection/perception/AGENTS.md` (no change needed — skip unless
  capture contract changed; it did not)

**Interfaces:**
- Consumes: everything above; `preflight`, `UR5eArm` from execute.py;
  `open_camera`, `session_metadata`, `capture_bundle`, `t_flange_cam`,
  `WRIST_SERIAL` from camera.py; `save_bundle` from capture.py.
- Produces: `p inspection/run/loop.py teach` and
  `p inspection/run/loop.py run --outdir=... --question="..."` (Anton-run).

- [ ] **Step 1: Append RealRig + CLI to loop.py**

```python
# ── real rig + CLI ──────────────────────────────────────────────────────────

ROBOT_IP = "192.168.2.50"
SURVEY_POSE_FILE = Path(__file__).resolve().parent / "survey_pose.json"


class RealRig:
    """UR5e + wrist D405 behind the rig contract. Owns session.json."""

    def __init__(self, world, console, outdir, ip=ROBOT_IP):
        from inspection.motion.execute import UR5eArm
        from inspection.perception.camera import (
            WRIST_SERIAL, capture_bundle, open_camera, session_metadata,
            t_flange_cam)
        self._capture_bundle = capture_bundle
        self.world, self.console = world, console
        self.outdir = Path(outdir)
        self.arm = UR5eArm(ip)
        self.pipe, profile, self.align, self.depth_scale = open_camera()
        meta = session_metadata(profile, self.depth_scale, WRIST_SERIAL)
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "session.json").write_text(
            json.dumps(meta, indent=2) + "\n")
        self.intr = meta["intrinsics"]["ir_left"]
        self._T_fc = t_flange_cam()
        self._ik = UR5eIK()

    def q(self):
        return self.arm.q()

    def preview(self, path):
        self.world.replay(path)

    def move(self, path):
        return self.arm.execute(path, world=self.world,
                                stop_event=self.console.stop_event)

    def capture(self, pose_id):
        from inspection.perception.capture import save_bundle
        bundle = self._capture_bundle(self.pipe, self.align)
        q = self.arm.q()
        T_bf = self._ik.fk(q)
        pose = {"joints_rad": q, "T_base_flange": T_bf,
                "T_base_cam": T_bf @ self._T_fc}
        d = save_bundle(self.outdir, pose_id, bundle, pose)
        return {"dir": str(d), "rgb": bundle["rgb"],
                "depth_raw": bundle["depth_raw"],
                "T_base_cam": pose["T_base_cam"], "q": q}

    def close(self):
        self.pipe.stop()
        self.arm.close()


def teach(ip: str = ROBOT_IP):
    """Freedrive the arm to the survey pose, then run this. Read-only."""
    from rtde_receive import RTDEReceiveInterface
    q = list(RTDEReceiveInterface(ip).getActualQ())
    SURVEY_POSE_FILE.write_text(json.dumps({"q_rad": q}, indent=2) + "\n")
    print(f"survey pose saved: {np.round(np.degrees(q), 1).tolist()} deg "
          f"-> {SURVEY_POSE_FILE}")


def run(outdir: str, ip: str = ROBOT_IP, r: float = 0.35,
        max_turns: int = 12, question: str = "", seed: int = 0):
    """The v1 loop. Requires: bringup done, Remote Control, taught survey."""
    from inspection.motion.execute import preflight
    from inspection.run.decider import Console, TerminalDecider

    pf = preflight(ip)
    if not pf["go"]:
        raise SystemExit(f"preflight NO-GO: {pf}")
    if not SURVEY_POSE_FILE.exists():
        raise SystemExit("no survey pose — freedrive there and run "
                         "`p inspection/run/loop.py teach` first")
    q_survey = np.array(json.loads(SURVEY_POSE_FILE.read_text())["q_rad"])

    world = RobotCell()
    world.init_viewer()                    # meshcat preview window
    console = Console()
    rig = RealRig(world, console, outdir, ip)
    try:
        Loop(rig, TerminalDecider(console), console, outdir,
             q_survey=q_survey, r=r, max_turns=max_turns,
             question=question, seed=seed, world=world).run()
    finally:
        rig.close()


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"teach": teach, "run": run})
```

- [ ] **Step 2: Import smoke test (no robot, no camera)**

Run: `~/miniconda3/envs/robo/bin/python -c "import inspection.run.loop as L; print(L.SURVEY_POSE_FILE.name, callable(L.teach), callable(L.run))"`
Expected: `survey_pose.json True True` (RealRig only touches hardware in
its constructor, which nothing calls at import).

- [ ] **Step 3: Delete explore.py, update run/AGENTS.md**

```bash
git rm inspection/run/explore.py
```

Replace the `## Files` section of `inspection/run/AGENTS.md` with:

```markdown
## Files
- `loop.py` — the v1 inspection loop (POC): boot to the taught survey pose,
  seed the object, then READ -> DECIDE -> plan -> meshcat preview ->
  approve -> move -> capture -> fuse. Every motion is human-approved;
  ENTER during motion is the software stop. Subcommands: `teach`
  (save freedrive survey pose), `run`. Spec:
  `inspection/2026-08-19-loop-v1-design.md`.
- `decider.py` — the AI seam. Decider.read/.decide contract, egocentric
  menu glosses (D4), Console (stdin owner + stop arming), TerminalDecider.
  BusDecider (UI toolkit) and AgentDecider slot in here later.
- `survey_pose.json` — taught on hardware, not committed until it exists.
```

and update its `## Contracts & decisions` VLM line to:

```markdown
- The AI slot: it replaces exactly the Decider (read + decide), nothing
  else. The orchestrator owns budget, approval, and the software stop.
```

- [ ] **Step 4: Re-run all tests**

Run: `for t in test_decider test_execute_wait test_geometry_seed test_loop_fake; do ~/miniconda3/envs/robo/bin/python inspection/tests/$t.py || break; done`
Expected: four OK lines.

- [ ] **Step 5: Commit**

```bash
git add inspection/run/loop.py inspection/run/AGENTS.md
git rm -q --cached inspection/run/explore.py 2>/dev/null; git add -u inspection/run/
git commit -m "loop v1: CLI + RealRig + teach; retire SO-101 explore.py"
```

---

### Task 7: Live bring-up (Anton at the robot — no agent executes this)

This task is a checklist handed to Anton; the agent's job is only to
confirm the commands below are printed at the end of execution.

- [ ] 1. Pendant: Remote Control ON. Then `p inspection/motion/execute.py bringup`
- [ ] 2. Switch pendant to Local, freedrive the arm to a camera-down survey
      pose over the table center, back to Remote, then
      `p inspection/run/loop.py teach`
- [ ] 3. Place the duck alone on the table.
- [ ] 4. `p inspection/run/loop.py run --outdir=data/inspection/loop_run1 --question="is there a logo on the duck?"`
- [ ] 5. Boot move → watch preview, approve, press ENTER mid-motion → verify
      the arm decelerates and the CLI re-plans from the stopped q and offers
      approval again; approve and complete.
- [ ] 5b. Repeat the ENTER stop test on the FIRST look move — verify it
      returns to the menu with the turn recorded (`approved` true,
      `stopped` true).
- [ ] 6. Run 3-5 turns, answer, then inspect `run.json` + `fused_cloud.npy`.
- [ ] 7. Eyeball the seeded object box in meshcat after boot, before
      approving the first look.
- [ ] 8. Teach the survey pose with the camera 0.15-0.55 m from the table
      (deproject clips at 0.13/0.60 m).
- [ ] 9. At 10-degree cells, watch the first fuse quality.

---

## Self-review (done at write time)

- **Spec coverage**: survey teach ✓ (T6), boot plan+approve ✓ (T5),
  approval gate every motion ✓ (T5 `_approve_and_move`), software stop ✓
  (T2 Console + T3 executor + T7 live test), stop -> menu ✓ (T5),
  single-object seed ✓ (T4), terminal decider + cv2 stopgap ✓ (T2),
  meshcat preview ✓ (T6 RealRig.preview), run.json ✓ (T5), bundles
  unchanged ✓ (T6 reuses save_bundle), plan-fail -> blocked cell ✓ (T5),
  max_turns ✓, out-of-scope list untouched ✓.
- **Placeholder scan**: clean — no TBDs, every code step carries the code.
- **Type consistency**: rig contract identical in FakeRig (T5) and RealRig
  (T6); `Ctx`/`build_menu` signatures match between T1, T2 and T5;
  `execute(..., stop_event)` matches T3 and T6 `RealRig.move`.
- **Known risk, flagged in T5 debug notes**: `world.set_object` replace-vs-
  append semantics unverified; test catches it.
