"""deploybench/deploy_stats.py — the deploy answer from a run's two raw streams (FR5).

A run yields the algorithm's `LocalizationOut` stream and the GT sample stream. This module
turns them into the numbers that answer the operator's question:

  - **arrived** — did the robot physically reach the goal (the evaluator's `outcome`)?
  - **localization error** — reused verbatim from locbench (`pairs.associate` + `stats`): raw
    map-frame pos/yaw mean/p95/max + coverage. No alignment (a constant bias must stay visible).
  - **time-to-first-fix (TTFF)** — for KIDNAPPED: sim-seconds from spawn to the first
    map-anchored (`TRACKING`) fix. ~0 for a KNOWN start that snaps immediately; the recovery
    cost for a kidnapped one; None if it never localizes.
  - **longest dead-reckon gap** — the longest continuous span (sim-s) the pose ran without a
    fresh `TRACKING` fix. This is the E2 "how long is the estimate safe between anchors" budget.

Pure numpy/stdlib → `brain`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..locbench.pairs import GtSample, associate
from ..locbench.stats import EpisodeStats, compute_stats
from ..oli.reason.localization import LocalizationOut, LocalizationStatus
from .scenario import Scenario


@dataclass(frozen=True)
class DeployResult:
    """One scenario's end-to-end verdict + localization statistics."""

    scenario_name: str
    start_mode: str
    outcome: str                       # "arrived" | "timeout" | "crashed"
    arrived: bool
    stats: EpisodeStats                # locbench raw-frame error/coverage (reused)
    ttff_s: Optional[float]            # sim-s spawn → first TRACKING fix; None if never
    longest_dead_reckon_s: float       # longest span with no fresh TRACKING fix [sim-s]
    final_goal_dist_m: Optional[float]  # GT distance to the goal at run end [m]


def compute_deploy_result(
    scenario: Scenario,
    estimates: Sequence[LocalizationOut],
    gt_samples: Sequence[GtSample],
    *,
    outcome: str,
    episode_start_ns: int,
) -> DeployResult:
    ests = sorted(estimates, key=lambda e: e.stamp_ns)
    gts = sorted(gt_samples)

    pairs = associate(ests, gts, episode_start_ns=episode_start_ns)
    stats = compute_stats(pairs, episode_id=0, outcome=outcome)

    ttff_s = _time_to_first_fix(ests, episode_start_ns)
    dead_reckon_s = _longest_dead_reckon(ests, episode_start_ns)
    final_dist = (math.dist((gts[-1][1], gts[-1][2]), scenario.goal) if gts else None)

    return DeployResult(
        scenario_name=scenario.name,
        start_mode=scenario.start_mode.value,
        outcome=outcome,
        arrived=(outcome == "arrived"),
        stats=stats,
        ttff_s=ttff_s,
        longest_dead_reckon_s=dead_reckon_s,
        final_goal_dist_m=final_dist,
    )


def _time_to_first_fix(ests: List[LocalizationOut], start_ns: int) -> Optional[float]:
    for e in ests:
        if e.status is LocalizationStatus.TRACKING:
            return max(0.0, (e.stamp_ns - start_ns) / 1e9)
    return None


def _longest_dead_reckon(ests: List[LocalizationOut], start_ns: int) -> float:
    """Longest sim-time span with no fresh map-anchored (TRACKING) fix, measured at each
    estimate. Seeded at spawn: a slow first fix counts as dead-reckoning from t=0."""
    last_fix = start_ns
    worst = 0.0
    for e in ests:
        worst = max(worst, (e.stamp_ns - last_fix) / 1e9)
        if e.status is LocalizationStatus.TRACKING:
            last_fix = e.stamp_ns
    return worst


def summary_line(res: DeployResult) -> str:
    """One-glance human summary of the run."""
    pos = res.stats
    if res.outcome != "arrived":
        head = f"NOT-ARRIVED ({res.outcome})"
    else:
        head = "ARRIVED"
    err = "—" if math.isnan(pos.pos_mean) else f"{pos.pos_mean:.3f}/{pos.pos_max:.3f} m mean/max"
    ttff = "n/a" if res.ttff_s is None else f"{res.ttff_s:.2f} s"
    return (f"[{res.scenario_name} · {res.start_mode}] {head} | "
            f"ATE {err} | cov {pos.coverage:.0%} | TTFF {ttff} | "
            f"worst dead-reckon {res.longest_dead_reckon_s:.2f} s")
