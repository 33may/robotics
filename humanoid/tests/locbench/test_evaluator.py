"""TDD for the evaluator episode loop (logic/locbench/evaluator.py) — D1/D3/D4, tasks 6.1.

The evaluator is a PURE CLIENT of the brain service seam: per episode it (1) sends the spawn
as a transit goal and watches GT until arrival (unscored, D3 — no teleport), (2) reads the
live GT pose as the warm-start hint and commands `start` over loc-ctrl (D4), (3) sends the
scored goal and collects the est stream (telemetry, deduped by stamp) + GT samples until GT
arrival ≤0.3 m or timeout, (4) commands `stop`. A crashed host / dead stack marks the
episode `crashed`; the loop continues with the next episode. All fakes — no sockets, no sim;
wall-clock timeouts shrunk. `brain` env.
"""

from typing import List, Optional

import pytest

from humanoid.logic.locbench.episodes import Episode
from humanoid.logic.locbench.evaluator import EvalConfig, Evaluator
from humanoid.logic.oli.reason.localization import (
    LocalizationOut,
    LocalizationStatus,
    RobotPose,
)
from humanoid.logic.oli.service.protocol import TelemetrySnapshot

pytestmark = pytest.mark.brain

S = 1_000_000_000


class FakeStack:
    """World + brain + shadow host in one object: GT glides toward the active goal at 1 m/s
    per poll step; the shadow est echoes GT (+bias) while 'running'. The evaluator's client
    surfaces (goal/loc-ctrl senders, telemetry/GT readers) all point here."""

    def __init__(self, bias=0.0, start_behavior="ok", die_after_polls=None):
        self.t_ns = 0
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.goal: Optional[tuple] = None
        self.loc_state = "idle"
        self.started_with: List = []
        self.stops = 0
        self.bias = bias
        self.start_behavior = start_behavior      # "ok" | "crash"
        self.die_after_polls = die_after_polls
        self.polls = 0
        self.alive = True

    # ── the evaluator's client surfaces ──────────────────────────────────────
    def send_goal(self, x, y, yaw=None):
        self.goal = (x, y)

    def clear_goal(self):
        self.goal = None

    def send_start(self, *, map_dir, initial_pose=None, calibration=None):
        self.started_with.append((map_dir, initial_pose, calibration))
        self.loc_state = "crashed" if self.start_behavior == "crash" else "running"

    def send_stop(self):
        self.stops += 1
        self.loc_state = "idle"

    def gt_latest(self):
        self._advance()
        return (self.t_ns, self.x, self.y, self.yaw)

    def telemetry_latest(self):
        est = None
        if self.loc_state == "running":
            est = LocalizationOut(
                stamp_ns=self.t_ns,
                pose=RobotPose(self.t_ns, self.x + self.bias, self.y, self.yaw),
                status=LocalizationStatus.TRACKING)
        return TelemetrySnapshot(stamp_ns=self.t_ns, est=est, loc_state=self.loc_state,
                                 loc_error="boom" if self.loc_state == "crashed" else None)

    def stack_alive(self):
        return self.alive

    # ── world dynamics: one step per GT poll ─────────────────────────────────
    def _advance(self):
        self.polls += 1
        if self.die_after_polls is not None and self.polls >= self.die_after_polls:
            self.alive = False
        self.t_ns += int(0.1 * S)                 # 10 Hz GT stream
        if self.goal is not None:
            gx, gy = self.goal
            dx, dy = gx - self.x, gy - self.y
            d = (dx * dx + dy * dy) ** 0.5
            step = min(0.1, d)                    # 1 m/s at 10 Hz
            if d > 1e-9:
                self.x += dx / d * step
                self.y += dy / d * step


def _evaluator(stack, **cfg_over):
    # timeouts are SIM seconds (GT stamps); FakeStack advances 0.1 sim-s per poll
    base = dict(timeout_s=5.0, transit_timeout_s=5.0, start_timeout_s=1.0,
                wall_backstop_s=5.0, arrival_tol_m=0.3, poll_dt=0.0)
    base.update(cfg_over)
    cfg = EvalConfig(**base)
    return Evaluator(
        send_goal=stack.send_goal, clear_goal=stack.clear_goal,
        send_start=stack.send_start, send_stop=stack.send_stop,
        gt_latest=stack.gt_latest, telemetry_latest=stack.telemetry_latest,
        stack_alive=stack.stack_alive,
        map_dir="/maps/x", calibration={"k": 1}, config=cfg)


_EP = Episode(id=0, spawn=(1.0, 0.0), goal=(3.0, 0.0), route_m=2.0)


def test_full_episode_transit_start_drive_stop():
    stack = FakeStack()
    res = _evaluator(stack).run_episode(_EP)

    assert res.outcome == "arrived"
    # warm start used the LIVE GT pose at the spawn, not a frozen yaw
    map_dir, pose, calib = stack.started_with[0]
    assert map_dir == "/maps/x" and calib == {"k": 1}
    assert abs(pose[0] - 1.0) < 0.35 and abs(pose[1] - 0.0) < 0.35
    assert stack.stops == 1                       # lifecycle closed
    assert stack.goal is None                     # goal cleared after the episode
    # both streams were collected and deduped
    assert len(res.ests) > 5
    assert len(res.gts) > len(res.ests) / 2
    assert len({e.stamp_ns for e in res.ests}) == len(res.ests)
    # episode_start marks the SCORED leg, not the transit
    assert res.episode_start_ns > 0
    assert all(g[0] >= res.episode_start_ns for g in res.gts)


def test_timeout_marks_and_moves_on():
    class Stuck(FakeStack):
        def _advance(self):
            self.t_ns += int(0.1 * S)            # clock runs, robot never moves

    stuck = Stuck()
    res = _evaluator(stuck, timeout_s=0.5, transit_timeout_s=0.5).run_episode(_EP)
    assert res.outcome == "timeout"
    assert stuck.stops <= 1                       # teardown still attempted


def test_crashed_host_marks_crashed():
    stack = FakeStack(start_behavior="crash")
    res = _evaluator(stack).run_episode(_EP)
    assert res.outcome == "crashed"
    assert "boom" in (res.error or "")


def test_dead_stack_marks_crashed():
    stack = FakeStack(die_after_polls=15)
    res = _evaluator(stack).run_episode(_EP)
    assert res.outcome == "crashed"


def test_run_continues_past_a_crashed_episode():
    stack = FakeStack(start_behavior="crash")
    eps = [Episode(id=0, spawn=(1.0, 0.0), goal=(3.0, 0.0), route_m=2.0),
           Episode(id=1, spawn=(1.0, 1.0), goal=(3.0, 1.0), route_m=2.0)]
    results = _evaluator(stack).run(eps)
    assert [r.outcome for r in results] == ["crashed", "crashed"]
    assert len(stack.started_with) == 2           # episode 1 was still attempted


def test_est_bias_flows_through_untouched():
    stack = FakeStack(bias=0.2)
    res = _evaluator(stack).run_episode(_EP)
    assert res.outcome == "arrived"
    # the E1 regime survives the collection layer: est = GT + 0.2 on x
    est = res.ests[-1]
    gt_by_stamp = {g[0]: g for g in res.gts}
    g = gt_by_stamp[est.stamp_ns]
    assert est.pose.x - g[1] == pytest.approx(0.2)
