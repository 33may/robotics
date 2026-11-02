"""locbench/evaluator.py — the episode loop, a pure client of the brain seam (D1, D3, D4).

Per episode:

  1. PLACE AT SPAWN (unscored): TELEPORT when the World offers the bench wire (Anton,
     14-07-2026 — the module is stopped between episodes, so a base snap can corrupt
     nothing; transit was ~half of a run's wall time), else walk it as a transit goal
     (D3's original path — also the automatic fallback when the teleport is unconfirmed).
  2. WARM START (D4): read the LIVE GT pose (position and whatever heading the follower
     arrived with) as `initial_pose`; command `start` over loc-ctrl; wait for the host to
     report `running` on telemetry (or `crashed`).
  3. SCORED LEG: send the goal; collect the candidate's est stream (telemetry `est`, deduped
     by stamp — telemetry is latest-wins, so polling faster than the module steps just
     re-reads) and the GT stream, until GT arrival ≤ `arrival_tol_m` or `timeout_s`.
  4. TEARDOWN: `stop` + clear goal, always.

A crashed host (loc_state) or a dead stack (`stack_alive()` — the launcher subprocess) marks
the episode `crashed`; `run()` continues with the next episode regardless — a run's report
must cover every episode, whatever happened to individual ones.

All I/O is injected as callables (the real run wires GoalChannelClient / LocCtrlClient /
TelemetryClient / DebugPoseClient; tests wire fakes) — the loop logic stays sim-free.
Timeouts are wall-clock: glide runs ~real-time (lockstep), and a stalled sim SHOULD time an
episode out — that is a real failure of the run, not measurement noise.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from ..oli.reason.localization import LocalizationOut
from .episodes import Episode
from .pairs import GtSample


@dataclass(frozen=True)
class EvalConfig:
    # timeout_s / transit_timeout_s are SIM seconds, measured on the GT stamp stream: the
    # sim runs far below real time under camera load (RTF ~0.03 observed 13-07), so a
    # wall-clock episode budget would falsely time out every long episode. The wall
    # backstop only catches a FROZEN sim (stamps stop advancing).
    timeout_s: float = 90.0            # scored-leg budget, SIM seconds (D2)
    transit_timeout_s: float = 120.0   # getting to the spawn, SIM seconds
    teleport_timeout_s: float = 10.0   # teleport → GT-at-spawn confirmation, SIM seconds
    start_timeout_s: float = 30.0      # host lifecycle start→running (WALL — sim-independent)
    wall_backstop_s: float = 3600.0    # hard wall ceiling per phase (frozen-sim guard)
    arrival_tol_m: float = 0.3         # GT distance that counts as "arrived" (D2)
    poll_dt: float = 0.02              # client poll cadence [s]


@dataclass
class EpisodeResult:
    episode: Episode
    outcome: str                                   # "arrived" | "timeout" | "crashed"
    ests: List[LocalizationOut] = field(default_factory=list)
    gts: List[GtSample] = field(default_factory=list)
    episode_start_ns: int = 0                      # scored-leg start (sim clock)
    error: Optional[str] = None


class Evaluator:
    """Drive frozen episodes through the live stack and collect the raw streams."""

    def __init__(
        self,
        *,
        send_goal: Callable[..., None],
        clear_goal: Callable[[], None],
        send_start: Callable[..., None],
        send_stop: Callable[[], None],
        gt_latest: Callable[[], Optional[GtSample]],
        telemetry_latest: Callable[[], object],
        stack_alive: Callable[[], bool] = lambda: True,
        teleport: Optional[Callable[[float, float, float], None]] = None,
        map_dir: str,
        calibration: Optional[dict] = None,
        config: EvalConfig = EvalConfig(),
        log: Callable[[str], None] = print,
        viewer=None,
    ) -> None:
        self._send_goal = send_goal
        self._clear_goal = clear_goal
        self._send_start = send_start
        self._send_stop = send_stop
        self._gt = gt_latest
        self._telemetry = telemetry_latest
        self._alive = stack_alive
        self._teleport = teleport
        self._map_dir = map_dir
        self._calibration = dict(calibration or {})
        self._cfg = config
        self._log = log
        # Optional live observability (`--live-view`): duck-typed `LiveView` hooks, every
        # one exception-proof on its side — the window can never change a run's outcome.
        self._viewer = viewer

    def run(self, episodes) -> List[EpisodeResult]:
        results = []
        for ep in episodes:
            self._log(f"[locbench] episode {ep.id}: S{ep.spawn} → G{ep.goal}")
            res = self.run_episode(ep)
            self._log(f"[locbench] episode {ep.id}: {res.outcome}"
                      + (f" ({res.error})" if res.error else ""))
            results.append(res)
        return results

    def run_episode(self, ep: Episode) -> EpisodeResult:
        res = EpisodeResult(episode=ep, outcome="crashed")
        if self._viewer is not None:
            self._viewer.on_episode(ep)
        try:
            # 1) place at the spawn (unscored): teleport wire when offered, else walk
            gt = self._place_at_spawn(ep)
            if gt is None:
                res.outcome = "crashed" if not self._alive() else "timeout"
                res.error = "transit to spawn failed"
                return res

            # 2) warm start from the LIVE GT pose
            self._send_start(map_dir=self._map_dir,
                             initial_pose=(gt[1], gt[2], gt[3]),
                             calibration=self._calibration)
            state = self._wait_loc_state("running", self._cfg.start_timeout_s)
            if state != "running":
                res.outcome = "crashed"
                res.error = self._loc_error() or f"host state {state!r} after start"
                return res

            # 3) the scored leg
            res.episode_start_ns = gt[0]
            self._send_goal(*ep.goal)
            outcome, error = self._collect(ep, res)
            res.outcome, res.error = outcome, error
            return res
        finally:
            # 4) teardown, always — the next episode must start clean
            try:
                self._send_stop()
                self._clear_goal()
            except OSError:
                pass  # dead stack: nothing to tear down

    # ── internals ─────────────────────────────────────────────────────────────

    def _place_at_spawn(self, ep: Episode) -> Optional[GtSample]:
        """Teleport to the spawn (facing the goal — deterministic warm-start heading) and
        confirm via GT; fall back to the walked transit if the wire is absent/unconfirmed."""
        if self._teleport is not None:
            yaw = math.atan2(ep.goal[1] - ep.spawn[1], ep.goal[0] - ep.spawn[0])
            self._teleport(ep.spawn[0], ep.spawn[1], yaw)
            gt = self._wait_arrival(ep.spawn, self._cfg.teleport_timeout_s)
            if gt is not None:
                return gt
            if not self._alive():
                return None
            self._log("[locbench] teleport unconfirmed — falling back to walk transit "
                      "(World booted without --teleport?)")
        self._send_goal(*ep.spawn)
        return self._wait_arrival(ep.spawn, self._cfg.transit_timeout_s)

    def _collect(self, ep: Episode, res: EpisodeResult) -> Tuple[str, Optional[str]]:
        wall_deadline = time.monotonic() + self._cfg.wall_backstop_s
        sim_t0: Optional[int] = None          # SIM-time budget starts at the first GT stamp
        last_gt_stamp = 0
        last_est_stamp = 0
        while time.monotonic() < wall_deadline:
            if not self._alive():
                return ("crashed", "stack died mid-episode")
            gt = self._gt()
            if gt is not None and gt[0] > last_gt_stamp:
                last_gt_stamp = gt[0]
                res.gts.append(gt)
                if self._viewer is not None:
                    self._viewer.on_gt(gt)
                if sim_t0 is None:
                    sim_t0 = gt[0]
            snap = self._telemetry()
            if snap is not None:
                if getattr(snap, "loc_state", None) == "crashed":
                    return ("crashed", getattr(snap, "loc_error", None) or "host crashed")
                est = getattr(snap, "est", None)
                if est is not None and est.stamp_ns > last_est_stamp:
                    last_est_stamp = est.stamp_ns
                    res.ests.append(est)
                    if self._viewer is not None:
                        self._viewer.on_est(est)
            if gt is not None and self._dist(gt, ep.goal) <= self._cfg.arrival_tol_m:
                return ("arrived", None)
            if sim_t0 is not None and (last_gt_stamp - sim_t0) / 1e9 > self._cfg.timeout_s:
                return ("timeout", f"no GT arrival within {self._cfg.timeout_s} sim-s")
            if self._cfg.poll_dt:
                time.sleep(self._cfg.poll_dt)
        return ("timeout", f"wall backstop {self._cfg.wall_backstop_s}s hit — sim frozen?")

    def _wait_arrival(self, target, timeout_s: float) -> Optional[GtSample]:
        """Wait for GT to reach `target` within `timeout_s` SIM seconds (wall backstop)."""
        wall_deadline = time.monotonic() + self._cfg.wall_backstop_s
        sim_t0: Optional[int] = None
        while time.monotonic() < wall_deadline:
            if not self._alive():
                return None
            gt = self._gt()
            if gt is not None:
                if self._viewer is not None:
                    self._viewer.on_transit_gt(gt)
                if sim_t0 is None:
                    sim_t0 = gt[0]
                if self._dist(gt, target) <= self._cfg.arrival_tol_m:
                    return gt
                if (gt[0] - sim_t0) / 1e9 > timeout_s:
                    return None
            if self._cfg.poll_dt:
                time.sleep(self._cfg.poll_dt)
        return None

    def _wait_loc_state(self, want: str, timeout_s: float) -> Optional[str]:
        deadline = time.monotonic() + timeout_s
        state = None
        while time.monotonic() < deadline:
            snap = self._telemetry()
            state = getattr(snap, "loc_state", None) if snap is not None else None
            if state == want or state == "crashed":
                return state
            if not self._alive():
                return "crashed"
            if self._cfg.poll_dt:
                time.sleep(self._cfg.poll_dt)
        return state

    def _loc_error(self) -> Optional[str]:
        snap = self._telemetry()
        return getattr(snap, "loc_error", None) if snap is not None else None

    @staticmethod
    def _dist(gt: GtSample, target) -> float:
        return math.dist((gt[1], gt[2]), (target[0], target[1]))
