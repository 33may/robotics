"""locbench/verdict.py — the two-tier gates (design.md D11).

| tier   | mean pos | max pos | yaw mean | coverage | meaning                              |
|--------|----------|---------|----------|----------|--------------------------------------|
| PASS   | <0.10 m  | <0.15 m | <10°     | ≥95%     | measured E1/E2 nav budget — works    |
| DEPLOY | <0.07 m  | <0.12 m | <7°      | ≥98%     | ~30% sim→real margin — take to ph. 2 |

A run's tier is the highest tier EVERY episode clears — one lost aisle = a failed demo, so
there is no averaging across episodes. Timeout/crashed episodes fail both tiers regardless
of their numbers. Failures are reported as human-readable strings naming the gate that broke
(the dev loop reads them out of report.json to know what to fix). DEPLOY numbers proposed,
Anton to confirm (design.md Open Questions). Pure stdlib → `brain` env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .stats import EpisodeStats


@dataclass(frozen=True)
class Tier:
    name: str
    mean_pos_m: float
    max_pos_m: float
    yaw_mean_deg: float
    min_coverage: float


PASS = Tier("PASS", mean_pos_m=0.10, max_pos_m=0.15, yaw_mean_deg=10.0, min_coverage=0.95)
DEPLOY = Tier("DEPLOY", mean_pos_m=0.07, max_pos_m=0.12, yaw_mean_deg=7.0, min_coverage=0.98)


@dataclass(frozen=True)
class EpisodeVerdict:
    episode_id: int
    tier: str                      # "DEPLOY" | "PASS" | "FAIL"
    failures: Tuple[str, ...]      # why the episode missed PASS (empty when PASS+)

    @property
    def passed(self) -> bool:
        return self.tier in ("PASS", "DEPLOY")

    @property
    def deployable(self) -> bool:
        return self.tier == "DEPLOY"


@dataclass(frozen=True)
class RunVerdict:
    tier: str
    failed_episodes: Tuple[int, ...]

    @property
    def passed(self) -> bool:
        return self.tier in ("PASS", "DEPLOY")

    @property
    def deployable(self) -> bool:
        return self.tier == "DEPLOY"


def episode_verdict(st: EpisodeStats) -> EpisodeVerdict:
    failures = _gate_failures(st, PASS)
    if failures:
        return EpisodeVerdict(st.episode_id, "FAIL", tuple(failures))
    if _gate_failures(st, DEPLOY):
        return EpisodeVerdict(st.episode_id, "PASS", ())
    return EpisodeVerdict(st.episode_id, "DEPLOY", ())


def run_verdict(verdicts: Sequence[EpisodeVerdict]) -> RunVerdict:
    """Every episode must clear the tier — the run is only as good as its worst episode."""
    if not verdicts:
        return RunVerdict("FAIL", ())
    failed = tuple(v.episode_id for v in verdicts if not v.passed)
    if failed:
        return RunVerdict("FAIL", failed)
    if all(v.deployable for v in verdicts):
        return RunVerdict("DEPLOY", ())
    return RunVerdict("PASS", ())


def _gate_failures(st: EpisodeStats, tier: Tier) -> List[str]:
    """The gates an episode misses at `tier` — named, so the dev loop knows what to fix."""
    fails: List[str] = []
    if st.outcome != "arrived":
        fails.append(f"episode {st.outcome} (must arrive within the time budget)")
    if st.n_ticks == 0:
        fails.append("no scored ticks (dead episode)")
        return fails
    if st.coverage < tier.min_coverage:
        fails.append(f"coverage {st.coverage:.1%} < {tier.min_coverage:.0%}")
    if math.isnan(st.pos_mean):
        fails.append("no answered ticks (all LOST)")
        return fails
    if st.pos_mean >= tier.mean_pos_m:
        fails.append(f"mean pos {st.pos_mean:.3f} m ≥ {tier.mean_pos_m} m")
    if st.pos_max >= tier.max_pos_m:
        fails.append(f"max pos {st.pos_max:.3f} m ≥ {tier.max_pos_m} m")
    if st.yaw_mean_deg >= tier.yaw_mean_deg:
        fails.append(f"mean yaw {st.yaw_mean_deg:.1f}° ≥ {tier.yaw_mean_deg}°")
    return fails
