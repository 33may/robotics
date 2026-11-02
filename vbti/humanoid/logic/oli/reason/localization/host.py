"""localization/host.py — the in-brain localization host (locbench design.md D6).

"No internal logic outside the brain" (Anton, 2026-07-13): the candidate module runs INSIDE
the brain process, on this host's side thread. The brain becomes the frame consumer — it owns
its senses. Flow per camera tick:

    frame source (CameraStreamReader) ──▶ newest unprocessed frames
    latest Observation/Intent (fed by the control loop via `on_tick`) ─┐
                                                                       ▼
                        LocalizationIn (frame-paced) ──▶ module.step() on THIS thread
                                                                       ▼
                        latest LocalizationOut ──▶ telemetry (est) / HostLocalizer (Stage 2)

Contract with the control loop: `on_tick` / `latest` / `request_start` / `request_stop` are
cheap and NEVER block on the module. A slow module skips frames (latest-wins — coverage/rate
measure the cost); a slow `start()` runs here too (lifecycle commands are queued). A raising
or contract-breaking module marks the episode `crashed` (verify_module_contract semantics),
its teardown is attempted, and the host survives to run the next episode with a FRESH module
instance (warm start, D4 — no state leaks across episodes).

Accepted risks, eyes open (D6): a GIL-holding binding still stalls Python — that shows up in
the brain-loop rate metric, measured not hidden; a hard segfault kills the process — the
evaluator marks the run crashed and reboots the brain. Pure: stdlib + contracts only.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from typing import Callable, Dict, List, Optional, Protocol

from ...contracts import CameraFrame, Intent, Observation
from .contracts import LocalizationIn, LocalizationOut, LocalizationSetup
from .module import LocalizationModule

_KEEP = object()   # "don't touch this field" sentinel for _set()


class FrameSource(Protocol):
    """What the host reads frames from — `comm.camera_stream.CameraStreamReader`'s shape."""

    def read(self, name: str) -> Optional[CameraFrame]: ...
    def stream_names(self) -> List[str]: ...


class LocalizationHost:
    """Own a candidate's lifecycle + stepping on a side thread; expose latest-state reads."""

    def __init__(
        self,
        module_factory: Callable[[], LocalizationModule],
        frames: FrameSource,
        poll_dt: float = 0.002,
    ) -> None:
        self._factory = module_factory
        self._frames = frames
        self._poll_dt = poll_dt

        self._lock = threading.Lock()          # guards everything below (all cheap)
        self._commands: deque = deque()        # ("start", setup) | ("stop",)
        self._state = "idle"                   # idle|starting|running|stopping|crashed
        self._latest_out: Optional[LocalizationOut] = None
        self._obs: Optional[Observation] = None
        self._intent: Optional[Intent] = None
        self._last_error: Optional[str] = None
        self._steps = 0

        self._module: Optional[LocalizationModule] = None   # host-thread-only
        # PER-STREAM watermarks (host-thread-only). One shared stamp starved whichever
        # stream the World writes last: chest@t consumed → head@t (equal stamp) is never
        # `>` the watermark again (cuvslam run 20260714-130101 — est frozen at warm start).
        self._last_stamps: Dict[str, int] = {}

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── control-loop surface (cheap, never blocks on the module) ────────────

    def on_tick(self, obs: Observation, intent: Optional[Intent]) -> None:
        """Feed the latest in-process observation/intent (recorder-hook cadence)."""
        with self._lock:
            self._obs = obs
            self._intent = intent

    def request_start(self, setup: LocalizationSetup) -> None:
        """Enqueue a fresh episode lifecycle (executes on the host thread — a heavy
        `start()` must not stall the control loop)."""
        with self._lock:
            self._commands.append(("start", setup))

    def request_stop(self) -> None:
        with self._lock:
            self._commands.append(("stop", None))

    def latest(self) -> Optional[LocalizationOut]:
        with self._lock:
            return self._latest_out

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    @property
    def steps(self) -> int:
        with self._lock:
            return self._steps

    def diagnostics(self):
        """Display-only: the module's cached diagnostics dict, if it offers one.

        `_module` is host-thread-owned; this cross-thread read is a deliberate exception —
        it only dereferences an attribute holding an immutable dict ref (GIL-atomic swap,
        see the cuvslam module's `_latest_diag`). Worst case: None or one frame stale.
        Never part of the LocalizationModule contract; absent method → None.
        """
        module = self._module
        if module is None:
            return None
        get = getattr(module, "diagnostics", None)
        try:
            return get() if callable(get) else None
        except Exception:  # noqa: BLE001 — display-only, never let it bite the host
            return None

    # ── lifecycle of the host itself ─────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="loc-host", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._teardown_module()

    # ── host thread ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            cmd = None
            with self._lock:
                if self._commands:
                    cmd = self._commands.popleft()
            if cmd is not None:
                self._execute(cmd)
                continue
            if self.state == "running":
                stepped = self._maybe_step()
                if stepped:
                    continue
            time.sleep(self._poll_dt)

    def _execute(self, cmd) -> None:
        op, setup = cmd
        if op == "start":
            self._teardown_module()            # restart-while-running = fresh episode
            self._set(state="starting", latest=None, error=None)
            self._last_stamps = {}
            try:
                self._module = self._factory() # FRESH instance per episode (D4)
                self._module.start(setup)
                self._set(state="running")
            except Exception as exc:  # noqa: BLE001 — any candidate failure = crashed episode
                self._crash(f"start() failed: {exc}")
        elif op == "stop":
            self._set(state="stopping")
            self._teardown_module()
            self._set(state="idle", latest=None)

    def _maybe_step(self) -> bool:
        """Assemble the newest unprocessed frame bundle and step the module once.

        "Unprocessed" is judged PER STREAM: each stream's frame enters the bundle iff it
        is newer than what the module last saw FROM THAT STREAM. Streams publish
        back-to-back with equal stamps, so a shared watermark permanently starves the
        late-written one (see test_equal_stamp_multi_stream_frames_all_reach_the_module).
        """
        bundle: Dict[str, CameraFrame] = {}
        for name in self._frames.stream_names():
            f = self._frames.read(name)
            if f is not None and f.stamp_ns > self._last_stamps.get(name, -1):
                bundle[name] = f
        if not bundle:
            return False
        with self._lock:
            obs, intent = self._obs, self._intent
        if obs is None:
            return False                        # LocalizationIn requires an Observation
        for name, f in bundle.items():
            self._last_stamps[name] = f.stamp_ns
        newest = max(f.stamp_ns for f in bundle.values())
        assert self._module is not None  # state=="running" ⇒ module exists (host thread only)
        try:
            out = self._module.step(LocalizationIn(
                stamp_ns=newest, frames=bundle, observation=obs, intent=intent))
            if not isinstance(out, LocalizationOut):
                raise TypeError(
                    f"step returned {type(out).__name__}, not LocalizationOut")
            with self._lock:
                self._latest_out = out
                self._steps += 1
        except Exception as exc:  # noqa: BLE001 — contract violation or candidate crash
            self._crash(f"step() failed: {exc}\n{traceback.format_exc(limit=3)}")
        return True

    def _crash(self, message: str) -> None:
        self._teardown_module()
        self._set(state="crashed", error=message, latest=None)

    def _teardown_module(self) -> None:
        if self._module is not None:
            try:
                self._module.stop()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
            self._module = None

    def _set(self, state: Optional[str] = None, latest: object = _KEEP,
             error: object = _KEEP) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            if latest is not _KEEP:
                self._latest_out = latest  # type: ignore[assignment]
            if error is not _KEEP:
                self._last_error = error  # type: ignore[assignment]


class HostLocalizer:
    """The ~10-line `Localizer` over the host's latest verdict — the Stage-2 seam.

    Stage 1 (shadow): constructed but dormant — Nav drives on GT. Stage 2 (`--localizer
    <name>`): injected into Nav, which then drives on the candidate's pose. LOST (or no
    verdict yet) → None → Nav's zero-velocity hold; never a stale pose. Same `Localizer`
    protocol as `GroundTruthLocalizer` — Nav cannot tell the difference (that is the point).
    """

    def __init__(self, host: LocalizationHost) -> None:
        self._host = host

    def estimate(self, observation, camera_frame=None):
        out = self._host.latest()
        if out is None or out.pose is None:
            return None
        return out.pose
