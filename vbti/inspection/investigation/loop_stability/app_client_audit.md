# Network-client audit — `inspection/` app (offline code read, 2026-08-24)

Scope: every `RTDEReceiveInterface`, `RTDEControlInterface`, `RTDEIOInterface`
and `DashboardClient` in the repo; their lifetimes; the thread inventory; and a
ranked defect list. No code was run against the robot. Offline unit tests were
run: `76 passed, 1 failed` (`tests/test_grid.py::test_grid_is_light` — an
import-order artifact of running the whole suite in one process, unrelated).

Ground truth about ur_rtde 1.6.5 below was taken from the **shipped binary**
(`~/miniconda3/envs/robo/lib/python3.11/site-packages/librtde.so.1.6.5`),
disassembled, not from memory:

| fact | evidence |
|---|---|
| `RTDEControlInterface` ctor opens **three** sockets | in `ur_rtde::RTDEControlInterface::RTDEControlInterface(...)`: `movl $0x7534,0x10(%rbx)` (RTDE, **30004**), `mov $0x7533,%r8d` → `ScriptClient::ScriptClient(string,uint,uint,int,bool)` (**30003**), `mov $0x752f,%edx` → `DashboardClient::DashboardClient(string,int,bool)` (**29999**) |
| `RTDEControlInterface::disconnect()` closes all three | calls `RTDE::disconnect(bool)`, `ScriptClient::disconnect()`, `DashboardClient::disconnect()` |
| `RTDEReceiveInterface` ctor / `RTDEIOInterface` ctor use only 30004 | `movl $0x7534,...` in each ctor |
| `RTDEReceiveInterface::reconnect()` never calls `RTDE::disconnect()` | its call list is `RTDE::connect` → `negotiateProtocolVersion` → `getControllerVersion` → `setupRecipes` → `new RobotState` → `sendStart` → `boost::thread::interrupt/joinable/join_noexcept` → **new `boost::thread`**. The old fd is closed implicitly when `RTDE::connect()` resets the socket → **active close → TIME-WAIT** |
| `RTDEIOInterface::~RTDEIOInterface()` does disconnect | destructor calls `RTDE::disconnect(bool)` when connected |
| binding defaults | `RTDEReceiveInterface(hostname, frequency=-1.0, variables=[], verbose=False, use_upper_range_registers=False, rt_priority=0)`; `RTDEControlInterface(hostname, frequency=-1.0, flags=FLAG_UPLOAD_SCRIPT, ur_cap_port=50002, rt_priority=0)`. `frequency=-1` → **500 Hz** on an e-Series. `RTDEControlInterface` has **no** `setSpeedSlider` in 1.6.5 (checked `dir()`), so `RTDEIOInterface` genuinely is required for the slider. |

---

## 1. Complete client table

### 1a. On the `run/app.py run` path (the real run)

| # | file:line | class | port(s) | created when | lifetime / teardown | threads that touch it | synchronised? |
|---|---|---|---|---|---|---|---|
| C1 | `motion/execute.py:73` | `RTDEReceiveInterface` | 30004 | `preflight()` from `run/app.py:95`, before anything else exists | `r.disconnect()` at `execute.py:88`, **not** in a `try/finally` | main thread only | n/a (sole owner) |
| C2 | `motion/execute.py:74` | `DashboardClient` | 29999 | same call | `dash.disconnect()` at `execute.py:84`, **not** in a `try/finally` | main thread only | n/a |
| C3 | `motion/execute.py:73` | `RTDEReceiveInterface` | 30004 | **second** `preflight()`, called from inside `UR5eArm.__init__` (`execute.py:185`), reached via `RealRig.__init__` (`run/rigs.py:172`) ← `run/app.py:115` | as C1 | main thread only | n/a |
| C4 | `motion/execute.py:74` | `DashboardClient` | 29999 | same second `preflight()` | as C2 | main thread only | n/a |
| **C5** | `motion/execute.py:188` | `RTDEReceiveInterface` (`self.recv`) | 30004 | `UR5eArm.__init__`, unconditional | **long-lived**, whole run. Closed by `close()` (`execute.py:329`) ← `RealRig.close()` (`rigs.py:232`) ← `app.py:157`. Also **replaced in place** by `recv.reconnect()` at `execute.py:244` | main (dispatcher `_publish_real_pose`), `PoseStreamer` @30 Hz, `plan-*` worker, `exec-*` worker (q + 20 Hz safety poll), plus ur_rtde's own C++ receive thread | **yes** — `RealRig._recv_lock`, via the two monkey-patched wrappers (`rigs.py:178-181`). **Except** `recv.disconnect()` at teardown, which is *not* taken under the lock |
| **C6** | `motion/execute.py:189` | `RTDEControlInterface` (`self.ctrl`) | **30004 + 30003 + 29999** (three sockets, one object) | `UR5eArm.__init__`, immediately after C5 | long-lived; `ctrl.disconnect()` at `execute.py:328` closes all three | `exec-*` worker (`moveJ`, `getAsyncOperationProgressEx` @20 Hz, `stopJ`), **main thread** (`rig.arm.stop()` `app.py:151`, `arm.close()` `app.py:157`), plus ur_rtde's own C++ receive thread | **NO lock at all** — see D8 |
| **C7** | `motion/execute.py:190` | `RTDEIOInterface` | 30004 | `UR5eArm.__init__`, `RTDEIOInterface(ip).setSpeedSlider(slider)` — an unnamed temporary | destroyed at end of statement by CPython refcounting; the C++ dtor disconnects. **Not** explicit | main thread only | n/a |
| C8 | ur_rtde internal | `DashboardClient` | 29999 | inside C6's constructor | lives for the whole control session; closed by `ctrl.disconnect()` | ur_rtde internals | n/a |
| C9 | ur_rtde internal | `ScriptClient` | **30003** | inside C6's constructor (uploads `rtde_control.urscript`) | whole control session | ur_rtde internals, **write-only — never read** | n/a |

### 1b. Not on the run path (CLI / diagnostics only)

| file:line | class | entered from | lifetime |
|---|---|---|---|
| `run/app.py:77` | `RTDEReceiveInterface` | `app.py teach` CLI | `disconnect()` at :79, no `try/finally` |
| `motion/execute.py:101,107` | `DashboardClient`, `RTDEReceiveInterface` | `execute.py bringup` CLI | dashboard disconnected at :112; **the receive interface is never disconnected** (D14) |
| `perception/capture.py:63` | `RTDEReceiveInterface` | `capture.py snap` → `read_pose()` | **created per call, never disconnected** (D7) |
| `perception/handeye.py:109` | `RTDEReceiveInterface` | `handeye.py collect` | **never disconnected**, held for the whole interactive session |
| `calib/frame_check.py:38` | `RTDEReceiveInterface` | `frame_check.py check` | correct: `try/finally: receive.disconnect()`. One per sample, `n` samples |
| `ui/tap.py:83` | `RTDEReceiveInterface` | `tap.py --rtde`, **separate process** | **never disconnected**; the thread `return`s on error and the interface is abandoned |
| `motion/rtde_soak.py:87-88,122,143,166,191,202,204,306` | all four kinds | soak matrix CLI | test harness, mostly explicit disconnects |
| `motion/execute.py:339` | `UR5eArm` | `execute.py demo` CLI | context manager → `close()` |

---

## 2. Peak simultaneous connections

**Peak = 5 live TCP connections to 192.168.2.50, at `motion/execute.py:190`**, for the
duration of the `setSpeedSlider` call inside `UR5eArm.__init__`:

| socket | owner | port |
|---|---|---|
| 1 | C5 `self.recv` | 30004 |
| 2 | C6 `self.ctrl`'s internal `RTDE` | 30004 |
| 3 | **C7** the temporary `RTDEIOInterface` | 30004 |
| 4 | C9 `self.ctrl`'s internal `ScriptClient` | 30003 |
| 5 | C8 `self.ctrl`'s internal `DashboardClient` | 29999 |

So **three simultaneous RTDE (30004) clients** at that instant. Add up to **four
sockets in TIME-WAIT** at the same moment (C1–C4, both preflights, TIME-WAIT is
60 s and the two preflights are seconds apart) → **9 kernel entries toward the
controller** during startup.

**Steady state during a run = 4 live**: C5 (30004), C6 (30004), C9 (30003), C8 (29999).

Transient extras:
- every `recv.reconnect()` (`execute.py:244`) leaves the previous C5 socket in
  TIME-WAIT while the new one is ESTABLISHED → 4 live + 1 TIME-WAIT on 30004;
- `ui/tap.py --rtde` run from another terminal adds a 5th 30004 client from a
  **different pid**.

### Reading this against the captured socket set

Observed at the moment of death: 30004 ×2 (1 ESTABLISHED, 1 TIME-WAIT), 30003 ×1
with 91304 B in Recv-Q, 29999 ×1 ESTABLISHED + ×2 TIME-WAIT.

- **The two TIME-WAIT dashboard sockets are the app's own two `preflight()` calls.**
  `preflight()` runs **twice** per startup — once at `app.py:95` and again inside
  `UR5eArm.__init__` at `execute.py:185`. Both disconnect correctly. Because
  TIME-WAIT is 60 s, their presence **bounds the death to within ~60 s of process
  start**.
- **The still-ESTABLISHED dashboard socket is not one the app opened.** It is
  `RTDEControlInterface`'s internal `DashboardClient` (C8), created at
  `execute.py:189` and released only by `ctrl.disconnect()`. Fully explained,
  **not a leak** — but the app's own code is not the place to look for it, which
  is why it looked anomalous.
- **The 91 KB Recv-Q on 30003 is `ScriptClient` (C9).** ur_rtde only ever writes
  on that socket; the UR RT interface pushes a ~1 KB state packet every 2 ms, so
  the receive queue fills within a fraction of a second of `execute.py:189` and
  stays permanently full/zero-windowed for the rest of the run. **Nothing
  app-side can drain it** (the socket lives inside `librtde.so`).
- **Discrepancy worth chasing (D12):** only ONE ESTABLISHED socket on 30004 is
  visible, but a live `RTDEControlInterface` must hold one *in addition to* C5's.
  Either the capture is partial, or the control session's RTDE link was already
  gone — in which case `UR5eArm.stop()` → `ctrl.stopJ(2.0)` was a **no-op** and
  the abort path in `_assert_fresh` could not have stopped the arm.

---

## 3. Is any interface created per-call or in a loop?

- **Not on the run path.** C5–C7 are created exactly once each per process.
- **`recv.reconnect()` is the exception** and it is effectively in a loop: see D2.
  Every failing `q()`/`_safety_ok()` re-runs it, and callers swallow the
  resulting `RuntimeError`, so a 30 Hz thread will re-attempt forever.
- **Off the run path**: `perception/capture.py:63` (`read_pose`) creates one
  `RTDEReceiveInterface` per call and never disconnects it. `calib/frame_check.py`
  creates one per sample but disconnects correctly.

## 4. Does anything create an RTDE/dashboard client while the long-lived recv is alive?

Yes — three moments, all inside `UR5eArm.__init__`, all *after* `self.recv`
(C5) exists:

1. `execute.py:189` — `RTDEControlInterface(ip)` opens 3 new sockets (30004,
   30003, 29999) **and uploads a URScript to the controller** while C5 is
   streaming at 500 Hz. `rtde_soak.ctrl`/`full_mix` tested exactly this and it
   survived.
2. `execute.py:190` — the `RTDEIOInterface` temporary opens and closes a 3rd
   30004 client within one statement.
3. `execute.py:244` — `recv.reconnect()` re-opens C5's own socket at an arbitrary
   later moment, potentially many times (D2).

Nothing else in the run path opens a client after startup. `ui/tap.py` and the
soak scenarios do, but only from other processes.

## 5. Is `ui/tap.py` in-process?

**Separate process, always.** Nothing imports `inspection.ui.tap` — grep over
`*.py`/`*.md`/`*.ts*` finds only the module's own docstring and its `fire`
entry point. It is a stand-alone CLI (`p inspection/ui/tap.py`).

**But it can absolutely be active during a normal run** — the docstring
explicitly instructs the operator to "run it in another terminal while a run is
going", and `--rtde` then adds a **6th connection to 30004 from a second pid**
polling `getActualQ()`+`getTimestamp()` at 30 Hz (`tap.py:83`, `_ArmProbe`).
It never disconnects that interface. If any hardware run that died was
diagnosed with `--rtde` attached, the client count on 30004 was higher than the
in-process analysis suggests. **Check the run logs for `arm probe:` lines before
concluding anything about client counts.**

---

## 6. Thread inventory

### Loop process (`p inspection/run/app.py run`)

| thread | started at | rate | robot / camera resources | notes |
|---|---|---|---|---|
| `MainThread` (dispatcher) | `app.py:131` `sup.run()` | 20 Hz queue poll (`timeout=0.05`) | `rig.q()` → **C5** (locked) on every transition to `idle`/`fault` (`machine.py:623`); at teardown `ctrl.stopJ` + `ctrl.disconnect` + `recv.disconnect` (**unlocked**) | also the only signal-handling thread; SIGINT handler installed at `app.py:117` |
| `porthole-bus` (daemon) | `porthole/python/porthole/bus.py:132` via `app.py:112` | asyncio, event-driven | none | websockets server on :8765 |
| `cmd-pump` (daemon) | `app.py:128` | blocks on `queue.get()` | none | forwards UI commands into `sup.events` |
| `PoseStreamer` (daemon) | `app.py:123`, class `rigs.py:235` | **30 Hz**, only while `pose_active` (i.e. during `executing`) | `rig.q()` → **C5**, locked; then pinocchio FK + msgpack publish | swallows every exception (`rigs.py:276`) — this is what makes D2 an unbounded retry loop |
| `CameraWorker` (daemon) | `app.py:119`, class `rigs.py:77` | **10 Hz** | D405 pipeline (`grab_aligned`), JPEG encode + publish | no robot access |
| `plan-<target>` (daemon) | `machine.py:293` | one-shot | `rig.q()` → **C5**, locked; then OMPL (`plan_viewpoint`), can take seconds | at most one action worker alive at a time |
| `exec-<target>` (daemon) | `machine.py:315` | safety poll **20 Hz** (`_wait_async`, `poll_s=0.05`) | `q()`+`_safety_ok()` → **C5** locked; `moveJ`/`getAsyncOperationProgressEx`/`stopJ` → **C6 unlocked**; then `CameraWorker.fresh_bundle` + `rig.q()` + disk I/O + `world.set_object()` | the only motion producer |
| `preview-<target>` (daemon) | `machine.py:375` | 30 Hz | none (poses precomputed) | joined, bounded 1 s, in `_cancel_preview` |
| ur_rtde receive thread (C++) | inside C5's ctor, **re-created by every `reconnect()`** | 500 Hz socket read | **C5**'s 30004 socket | not a Python thread; invisible to `threading.enumerate()` in `stream_autopsy` |
| ur_rtde control receive thread (C++) | inside C6's ctor | continuous | **C6**'s 30004 socket | ditto |
| librealsense worker threads | `pipe.start()` in `camera.open_camera` | driver-internal | D405 USB | several |
| `ThreadingHTTPServer` + per-request threads | `porthole/window.py:117` — **only when `--no_window`** (`app.py:46-49`) | per request | none | with a window these live in the child process instead |

### UI child process (`inspection/ui/app.py serve`, spawned at `app.py:130`)

Qt/pywebview main thread + `ThreadingHTTPServer` thread + per-request threads.
**No robot access; no bus.** Forked *after* the camera pipeline and all four
robot sockets are open (`close_fds=True` is the `Popen` default, so the sockets
are not inherited).

---

## 7. Are there RTDE reads NOT protected by `_recv_lock`?

**Exhaustive trace result: on the run path, every `RTDEReceiveInterface`
*read* is correctly locked. The discipline holds.** Proof:

- The only long-lived recv is `UR5eArm.recv` (C5). The only methods that read it
  are `UR5eArm.q()` (`execute.py:258`), `UR5eArm._safety_ok()` (`:262`) and
  `UR5eArm._assert_fresh()`/`_ts_advancing()` (`:201-256`) — and `_assert_fresh`
  is only ever reached *from* `q()`/`_safety_ok()`.
- `RealRig.__init__` replaces both entry points with locked wrappers
  (`rigs.py:178-181`). These are **instance** attributes, so `UR5eArm.execute()`'s
  internal `self.q()`, `self._safety_ok()` and the `safety_fn` passed to
  `_wait_async` (`execute.py:283, 288, 301, 306, 317`) all resolve to the
  wrappers. Verified by reading every call site.
- Every rig read outside `execute()` goes through `RealRig.q()` (`rigs.py:203`):
  `machine.py:450` (`_plan_worker`), `machine.py:623` (`_publish_real_pose`),
  `rigs.py:219` (`capture`), `app.py:120` (`PoseStreamer(rig.q, ...)`).
  A grep for `rig.`/`.arm` across `run/`, `view/`, `motion/plan.py` and `cell/`
  finds no other reader.
- The lock is non-reentrant and no wrapped method calls another wrapped method,
  so there is no deadlock.

**What *is* unprotected:**

1. **`self.ctrl` (C6) has no synchronisation of any kind** and is touched from
   two threads — see D8.
2. **`recv.disconnect()` at teardown is not taken under `_recv_lock`**
   (`app.py:157` → `rigs.py:232` → `execute.py:329`) — see D8b.
3. `_assert_fresh` holds the lock for a *very* long time (up to `stale_window` +
   an `ss` subprocess + a full RTDE reconnect) — see D1.
4. Off the run path, every other interface is its own object with a single
   reader; no sharing, so no lock needed.

---

## 8. Ranked defects

### DEFINITELY A BUG

---

**D1 — `_ts_advancing()` can declare a healthy stream dead after a single comparison.**
`motion/execute.py:201-213`

```python
t0 = self.recv.getTimestamp()
deadline = time.monotonic() + self.stale_window          # 0.10 s
while time.monotonic() < deadline:
    if self.recv.getTimestamp() != t0:
        return True
    time.sleep(0.002)
return False
```

The loop can execute **exactly one** comparison: read `t0`, read again
immediately (at 500 Hz these land on the same 2 ms packet — the docstring says
so itself), `time.sleep(0.002)` returns after >100 ms because the thread was
descheduled or the GIL was held, the `while` condition is now false, and the
function returns `False` **without ever re-reading the timestamp**. The
function's whole stated purpose ("a single pair compared once is not enough —
so poll for an advance") is defeated by its own loop shape.

This is the single strongest app-side explanation for "healthy in 23 isolated
repros, dead in every assembled run": the isolated repros ran on an idle box;
the assembled app runs a 30 Hz pinocchio publisher, a 10 Hz JPEG encoder, OMPL,
librealsense and a software-Vulkan Chromium child (`publisher.py:527-541`
documents that this box's compositor is already saturated). And the "recovery"
is destructive — it tears down a working RTDE socket.

*Minimal fix:* always do a final read after the loop, and treat a long
scheduling gap as inconclusive rather than dead:

```python
t0 = self.recv.getTimestamp()
deadline = time.monotonic() + self.stale_window
while time.monotonic() < deadline:
    time.sleep(0.002)
    if self.recv.getTimestamp() != t0:
        return True
return self.recv.getTimestamp() != t0      # one last read, always
```

Consider also widening `stale_window` to 0.5 s — `rtde_soak.DEAD_AFTER_S` is
already 0.5 s and calls that "unambiguous".

---

**D2 — the freshness recovery has no latch, no backoff and no attempt limit.**
`motion/execute.py:235-256`, callers at `rigs.py:275-277` and `machine.py:623-627`

Every failing `q()`/`_safety_ok()` independently runs: a 100 ms probe → a full
`stream_autopsy()` (which forks `ss`) → `recv.reconnect()`. `reconnect()` is
verified to close the old 30004 socket (TIME-WAIT) and open a new one,
renegotiate the protocol, resend the recipe, `sendStart()` and spawn a new
boost thread. The `RuntimeError` raised on failure is swallowed by
`PoseStreamer.run` (`rigs.py:276`) and by `_publish_real_pose` (`machine.py:626`),
so the 30 Hz pose thread retries **forever**: a connect storm on port 30004,
several attempts per second, each leaving a TIME-WAIT socket, for the rest of
the run. A single transient hiccup is thereby converted into sustained
controller-side connection churn.

*Minimal fix:* store `self._last_reconnect = 0.0` in `__init__`; in
`_assert_fresh`, skip the autopsy+reconnect and raise immediately if
`time.monotonic() - self._last_reconnect < 2.0`; log the autopsy only on the
first failure.

---

**D3 — `stream_autopsy()` forks `ss` (5 s timeout) while holding `_recv_lock`, during motion.**
`motion/execute.py:146-148` called from `:242`; lock at `run/rigs.py:199-201`

`_assert_fresh` runs inside the locked wrapper. During a move, the `PoseStreamer`
(30 Hz) and `_wait_async`'s `safety_fn` (20 Hz) both serialise on `_recv_lock`.
If the pose thread enters the autopsy, `execute()`'s safety poll blocks for up
to **5 seconds with the arm in motion**, and because `_wait_async` re-checks
`stop_event` only at the top of the loop, a `run/stop` (and Ctrl-C) cannot halt
the arm for that whole period. Forking a process that has pinocchio, open3d,
librealsense and two ur_rtde boost threads mapped in is itself a hazard on the
failure path of a moving robot.

*Minimal fix:* build the autopsy string **outside** the lock (hand `_assert_fresh`
a callback the caller invokes after releasing), or at minimum drop the `ss`
timeout to 0.5 s and skip the autopsy entirely when called from `_safety_ok`.

---

**D4 — `RTDEIOInterface` is an unnamed temporary; its teardown depends on refcounting.**
`motion/execute.py:190` — `RTDEIOInterface(ip).setSpeedSlider(slider)`

The C++ destructor does disconnect (verified), so under CPython this is a
create-use-destroy within one statement rather than a permanent leak. But: if
`setSpeedSlider` raises, the traceback keeps the frame — and the temporary —
alive, leaving a 3rd RTDE client on 30004 for the life of the exception; and
nothing in the code says "this must be disconnected". The comment in
`rtde_soak.py:135` ("the app's exact leak") is based on this line.

*Minimal fix:*
```python
io = RTDEIOInterface(ip)
try:
    io.setSpeedSlider(slider)
finally:
    io.disconnect()
```

---

**D5 — `UR5eArm.close()` skips `recv.disconnect()` if `ctrl.disconnect()` raises; and a raising `rig.close()` orphans the UI child and the bus.**
`motion/execute.py:327-329`, `run/rigs.py:228-232`, `run/app.py:154-160`

```python
def close(self):
    self.ctrl.disconnect()      # if this raises...
    self.recv.disconnect()      # ...this never runs
```

And in `app.py`'s `finally`:
```python
if sup is not None and len(sup.acc.points):
    np.save(outdir / "fused_cloud.npy", sup.acc.points)   # unguarded
if rig is not None:
    rig.close()                                            # unguarded
_close_ui(child)                                           # skipped if either raises
if bus is not None:
    bus.stop()                                             # skipped too
```
A raise from `np.save` (disk full) or from `ctrl.disconnect()` leaves the RTDE
interfaces open, the **UI child process orphaned** and the bus port bound —
and replaces the original exception. `rig.arm.stop()` just above *is* guarded,
which shows the intent.

*Minimal fix:* `try/except Exception: log.exception(...)` around each of the
three teardown steps, and around each disconnect in `UR5eArm.close()`.

---

**D6 — `UR5eArm.__init__` leaks `self.recv` on partial construction.**
`motion/execute.py:186-190`, `run/rigs.py:172`

If `RTDEControlInterface(ip)` raises (pendant left in Local, script upload
rejected — the common first-run failure), `self.recv` is already connected. The
half-built `UR5eArm` is dropped; `RealRig.__init__` builds the arm at
`rigs.py:172`, **outside** its own `try/except` (which starts at `:182`), so the
`self.arm.close()` in its handler never runs; and `app.run`'s `finally` sees
`rig is None`. The 30004 socket survives until GC.

*Minimal fix:* wrap lines 188-190 in `try/except` that disconnects whatever was
built and re-raises. Symmetrically, move `self.arm = UR5eArm(ip)` inside
`RealRig.__init__`'s existing `try`.

---

**D7 — `read_pose()` creates an RTDE client per call and never disconnects it.**
`perception/capture.py:59-63`

```python
q = np.array(RTDEReceiveInterface(robot_ip).getActualQ())
```
Not on the run path (`RealRig.capture` computes the pose itself, `rigs.py:219-222`)
— it is only reached from the `snap` CLI. Same pattern, worse, at
`perception/handeye.py:109`, where `rtde` is held for a whole interactive
collection session and never disconnected, and at `ui/tap.py:83`.
`calib/frame_check.py:36-42` does it correctly with `try/finally` — that is the
template.

*Minimal fix:* `r = RTDEReceiveInterface(robot_ip); try: ... finally: r.disconnect()`.

---

**D8 — `RTDEControlInterface` (C6) is used from two threads with no lock.**
`motion/execute.py:302-308, 321-329` vs `run/app.py:151, 157`

ur_rtde documents `RTDEControlInterface` as **not thread-safe**. The `exec-*`
worker calls `moveJ`/`getAsyncOperationProgressEx`/`stopJ` on it; the main
thread calls `rig.arm.stop()` → `ctrl.stopJ(2.0)` at `app.py:151` and
`rig.close()` → `ctrl.disconnect()` at `app.py:157`. That overlap is reachable
whenever `sup.join_workers(5.0)` returns `False` — which `machine.py:162-183`
explicitly anticipates ("Returns False if anything was still running"), and
which an OMPL plan that "can take seconds" (`machine.py:266`) makes realistic.

**D8b, same class:** `recv.disconnect()` at `execute.py:329` is not taken under
`_recv_lock`, while `PoseStreamer.stop()` only joins with a 2 s timeout
(`rigs.py:297-303`) and logs "did not exit" rather than blocking. A pose tick
inside `getActualQ()` when the interface is destroyed is a C++ use-after-free,
not an exception — exactly the hazard `machine.py:165-170` was written to avoid,
left open on the last hop.

*Minimal fix:* in `RealRig.close()`, `with self._recv_lock: self.arm.close()`;
and give `UR5eArm` a `self._ctrl_lock` taken by `execute()`, `stop()` and
`close()`.

---

**D14 — `bringup()` never disconnects its receive interface.**
`motion/execute.py:106-113`

`r = RTDEReceiveInterface(ip)` is polled for up to 20 s and then abandoned;
`bringup` then calls `preflight(ip)`, opening a *second* recv and a dashboard
while `r` is still connected. Also `dash.disconnect()` is skipped if the poll
loop raises. CLI-only, run rarely, hence the low rank — but it is the same
class of bug as D7.

### SUSPICIOUS — CANNOT PROVE IT MATTERS

---

**D9 — the long-lived recv runs at 500 Hz with the full default variable set and `rt_priority=0`.**
`motion/execute.py:188` — `RTDEReceiveInterface(ip)`

`frequency=-1.0` resolves to **500 Hz** on an e-Series; `variables=[]` means the
full default recipe; `rt_priority=0` means the C++ receive thread gets no
real-time scheduling. The same process then runs a 30 Hz pinocchio FK publisher,
a 10 Hz JPEG encoder, OMPL, librealsense and (via the child) a software-Vulkan
browser that `publisher.py:527-541` documents as already saturating this box.
A subscriber that stops draining is the classic way to lose an RTDE stream.

Note the shape of the evidence: outside a move the app barely reads the recv at
all (only `_publish_real_pose` on phase transitions and one read per plan), so
the *Python* read pattern is not plausibly the killer — which matches
`rtde_soak.threads` surviving. What differs in the assembled app is **CPU
pressure on ur_rtde's own C++ receive thread**, and no soak scenario reproduced
that: `ui_launch` had no bus/publisher, `camera` had no bus/publisher, `threads`
had neither camera nor UI. `rtde_soak.py:28` already advertises
`--frequency=125` as the "H3 probe" but nothing in the matrix combines it with
app-level load.

*Cheapest experiment / fix:* `RTDEReceiveInterface(ip, frequency=125.0,
variables=["timestamp","actual_q","safety_status_bits","robot_mode","safety_mode"])`
— a ~4× cut in packet rate and a large cut in payload. Optionally
`rt_priority=90` (needs `CAP_SYS_NICE`).

---

**D10 — the 91 KB Recv-Q on 30003 is ur_rtde's `ScriptClient`, and it is permanently zero-windowed.**
consequence of `motion/execute.py:189`

Verified from the binary: `RTDEControlInterface`'s ctor builds
`ScriptClient(host, maj, min, 30003, verbose)`. ur_rtde only writes on that
socket; the UR RT interface pushes a ~1 KB state packet every 2 ms, so the
kernel receive queue fills within a fraction of a second of `execute.py:189`
and the TCP window stays closed for the whole run. This **fully explains the
91304 bytes** and means it is *not* an app defect — but it does mean the
controller's RT-interface writer has a permanently backed-up send queue toward
us for the entire run, which is the one wire-level abnormality present in every
real run. `rtde_soak.ctrl`/`full_mix` reproduced this and survived, so it is not
sufficient on its own.

*No app-side fix exists.* Only options: patch/replace ur_rtde, or construct
`RTDEControlInterface` with `flags` that skip the script upload (requires the
ExternalControl URCap on 50002).

---

**D11 — `preflight()` runs twice per startup.**
`run/app.py:95` and `motion/execute.py:185`

Four extra connect/disconnect cycles (2× 30004, 2× 29999) in the first seconds,
for information `app.py` already has. Harmless in isolation
(`rtde_soak.preflight_churn` survived) but it is pure churn against the
controller at exactly the moment C5/C6 are being established.

*Minimal fix:* give `UR5eArm.__init__` a `preflight_report=None` parameter and
pass `pf` down from `app.run`.

---

**D12 — the captured socket set is missing an ESTABLISHED 30004 socket.**

A live `RTDEControlInterface` must hold one in addition to C5's. Either the
`ss` output was truncated/summarised, or the control session's RTDE link was
already dead — in which case `stop()`/`stopJ()` were no-ops and the
"abort on stale stream" path could not actually have stopped the arm. **Worth
resolving before anything else: it changes whether the abort path is real.**
Re-run the autopsy with the raw `ss` output preserved verbatim.

---

**D13 — forking a heavily-threaded process while four robot sockets are open.**
`run/app.py:55` (`subprocess.Popen`) and `motion/execute.py:146` (`subprocess.run`)

`_open_ui` is called at `app.py:130`, i.e. after the D405 pipeline is running,
after `PoseStreamer` has started, and with C5/C6 (and their two C++ receive
threads) live. `close_fds=True` is the default so the sockets are not inherited,
but `fork()` from a multithreaded process is a known hazard, and
`rtde_soak.ui_launch` tested the `Popen` with **only a recv open** — never with
`ctrl` + camera + bus in the same address space.

---

**D15 — an operator-attached `tap.py --rtde` is an invisible 6th RTDE client.**
`ui/tap.py:83`

The module docstring tells the operator to run it during a live run. It opens
its own 30004 subscriber at 30 Hz from another pid and never disconnects it.
Before drawing conclusions from client counts, check whether the dying runs had
a tap attached (look for `arm probe: read-only RTDE on ...` in the terminal
history).

---

## 9. Suggested order of work

1. **D12** — recover the raw `ss` output. It decides whether the abort path can
   actually stop the arm.
2. **D1** — 3-line fix, removes an entire class of spurious "stream DEAD".
3. **D2 + D3** — latch + backoff + get the `ss` fork out from under the lock.
   Together these stop a hiccup from becoming a connect storm and stop a
   5-second blind spot during motion.
4. **D9** — one-line experiment (`frequency=125`, explicit `variables`) that
   directly tests the "CPU pressure starves the receive thread" hypothesis, and
   is the only untested cell in the elimination matrix.
5. **D5, D6, D8, D4** — teardown/lifetime hygiene.
6. **D7, D14, D11** — off-path leaks and startup churn.
