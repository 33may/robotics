"""TDD for per-episode statistics (logic/locbench/stats.py) — D10.

Raw map-frame compare per pair: position error [m], yaw error [deg, wrapped], coverage
(% ticks answered — LOST pairs count against it), pose rate. Aggregates per episode:
mean/median/p95/max. NO alignment anywhere — a constant bias must arrive intact at the
verdict layer. Pure numpy/stdlib → `brain` env.
"""

import math

import pytest

from humanoid.logic.locbench.pairs import PosePair
from humanoid.logic.locbench.stats import EpisodeStats, compute_stats
from humanoid.logic.oli.reason.localization import LocalizationStatus

pytestmark = pytest.mark.brain

S = 1_000_000_000


def _pair(t_s, est, gt, status=LocalizationStatus.TRACKING):
    return PosePair(stamp_ns=int(t_s * S), est=est, gt=gt, status=status, assoc_dt_ns=0)


def _track(n=20, bias=(0.0, 0.0), yaw_err=0.0, lost_every=0):
    """n pairs at 10 Hz walking along +x; est = gt + bias (constant — the E1 regime)."""
    pairs = []
    for i in range(n):
        gt = (i * 0.1, 0.0, 0.0)
        lost = lost_every and (i % lost_every == 0)
        est = None if lost else (gt[0] + bias[0], gt[1] + bias[1], yaw_err)
        pairs.append(_pair(10 + i * 0.1, est, gt,
                           LocalizationStatus.LOST if lost else LocalizationStatus.TRACKING))
    return pairs


def test_perfect_track_scores_zero():
    st = compute_stats(_track(), episode_id=0, outcome="arrived")
    assert isinstance(st, EpisodeStats)
    assert st.n_ticks == 20 and st.n_answered == 20
    assert st.coverage == 1.0
    assert st.pos_mean == 0.0 and st.pos_max == 0.0
    assert st.yaw_mean_deg == 0.0
    assert st.outcome == "arrived"


def test_constant_bias_arrives_intact():
    # THE E1 check at the stats layer: a 0.2 m constant offset must show up as exactly
    # 0.2 m in mean AND max — any alignment/anchoring would zero it out.
    st = compute_stats(_track(bias=(0.2, 0.0)), episode_id=0, outcome="arrived")
    assert st.pos_mean == pytest.approx(0.2)
    assert st.pos_max == pytest.approx(0.2)
    assert st.pos_p95 == pytest.approx(0.2)


def test_yaw_error_wraps():
    # est yaw −179° vs gt +179° is a 2° error, not 358°.
    p = _pair(10, (0.0, 0.0, math.radians(-179)), (0.0, 0.0, math.radians(179)))
    st = compute_stats([p], episode_id=0, outcome="arrived")
    assert st.yaw_mean_deg == pytest.approx(2.0, abs=1e-6)


def test_lost_ticks_hit_coverage_not_errors():
    st = compute_stats(_track(n=20, lost_every=5), episode_id=0, outcome="arrived")
    assert st.n_ticks == 20 and st.n_answered == 16
    assert st.coverage == pytest.approx(0.8)
    assert st.pos_mean == 0.0   # error stats over ANSWERED ticks only


def test_pose_rate_from_answered_ticks():
    st = compute_stats(_track(n=20), episode_id=0, outcome="arrived")
    # 20 answered ticks over 1.9 s of estimates ≈ 10 Hz
    assert st.pose_rate_hz == pytest.approx(20 / 1.9, rel=0.01)


def test_empty_pairs_is_a_dead_episode():
    st = compute_stats([], episode_id=3, outcome="timeout")
    assert st.n_ticks == 0 and st.coverage == 0.0
    assert math.isnan(st.pos_mean) and math.isnan(st.pos_max)
