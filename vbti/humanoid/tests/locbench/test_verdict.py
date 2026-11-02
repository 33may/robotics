"""TDD for the two-tier verdict (logic/locbench/verdict.py) — D11.

| tier   | mean pos | max pos | yaw mean | coverage |
| PASS   | <0.10 m  | <0.15 m | <10°     | ≥95%     |  the measured E1/E2 nav budget
| DEPLOY | <0.07 m  | <0.12 m | <7°      | ≥98%     |  ~30% sim→real margin

A candidate's run tier = the highest tier EVERY episode clears (one lost aisle = failed
demo — no averaging across episodes). Timeout/crashed episodes fail both tiers. The harness
acceptance triplet lives on these gates: clean → PASS; 0.2 m bias → fails on max-pos;
20% dropout → fails on coverage. Pure → `brain` env.
"""

import pytest

from humanoid.logic.locbench.pairs import PosePair
from humanoid.logic.locbench.stats import compute_stats
from humanoid.logic.locbench.verdict import (
    DEPLOY,
    PASS,
    episode_verdict,
    run_verdict,
)
from humanoid.logic.oli.reason.localization import LocalizationStatus

pytestmark = pytest.mark.brain

S = 1_000_000_000


def _track_stats(episode_id=0, n=100, bias=0.0, yaw_err=0.0, lost_every=0,
                 outcome="arrived"):
    pairs = []
    for i in range(n):
        gt = (i * 0.1, 0.0, 0.0)
        lost = lost_every and (i % lost_every == 0)
        est = None if lost else (gt[0] + bias, gt[1], yaw_err)
        pairs.append(PosePair(
            stamp_ns=(10 + i) * S, est=est, gt=gt,
            status=LocalizationStatus.LOST if lost else LocalizationStatus.TRACKING,
            assoc_dt_ns=0))
    return compute_stats(pairs, episode_id=episode_id, outcome=outcome)


# ── episode gates ────────────────────────────────────────────────────────────


def test_clean_episode_reaches_deploy():
    v = episode_verdict(_track_stats(bias=0.02))
    assert v.tier == "DEPLOY" and v.passed and v.deployable
    assert v.failures == ()


def test_bias_between_tiers_is_pass_not_deploy():
    v = episode_verdict(_track_stats(bias=0.08))     # >0.07 DEPLOY mean, <0.10 PASS mean
    assert v.tier == "PASS" and v.passed and not v.deployable


def test_02m_bias_fails_on_max_pos():
    # THE harness self-check (D13 triplet): a constant 0.2 m offset must fail, and the
    # report must SAY it was position error — max 0.2 > 0.15 (and mean 0.2 > 0.10).
    v = episode_verdict(_track_stats(bias=0.2))
    assert v.tier == "FAIL" and not v.passed
    assert any("max pos" in f for f in v.failures)
    assert any("mean pos" in f for f in v.failures)


def test_dropout_fails_on_coverage():
    v = episode_verdict(_track_stats(lost_every=5))  # 80% coverage < 95%
    assert v.tier == "FAIL"
    assert any("coverage" in f for f in v.failures)


def test_yaw_gate():
    import math
    v = episode_verdict(_track_stats(yaw_err=math.radians(12.0)))
    assert v.tier == "FAIL"
    assert any("yaw" in f for f in v.failures)


def test_timeout_and_crash_fail_both_tiers():
    for outcome in ("timeout", "crashed"):
        v = episode_verdict(_track_stats(bias=0.0, outcome=outcome))
        assert v.tier == "FAIL"
        assert any(outcome in f for f in v.failures)


def test_dead_episode_fails():
    from humanoid.logic.locbench.stats import compute_stats as cs
    v = episode_verdict(cs([], episode_id=0, outcome="crashed"))
    assert v.tier == "FAIL"


# ── run verdict: every episode must pass ─────────────────────────────────────


def test_run_tier_is_the_weakest_episode():
    stats = [_track_stats(episode_id=0, bias=0.02),   # DEPLOY-grade
             _track_stats(episode_id=1, bias=0.08),   # PASS-grade
             _track_stats(episode_id=2, bias=0.02)]
    rv = run_verdict([episode_verdict(s) for s in stats])
    assert rv.tier == "PASS"          # one PASS-grade episode caps the run at PASS


def test_one_failed_episode_fails_the_run():
    stats = [_track_stats(episode_id=i, bias=0.02) for i in range(9)]
    stats.append(_track_stats(episode_id=9, bias=0.2))
    rv = run_verdict([episode_verdict(s) for s in stats])
    assert rv.tier == "FAIL" and not rv.passed
    assert rv.failed_episodes == (9,)


def test_all_clean_run_deploys():
    rv = run_verdict([episode_verdict(_track_stats(episode_id=i, bias=0.02))
                      for i in range(10)])
    assert rv.tier == "DEPLOY" and rv.passed and rv.deployable


def test_thresholds_are_the_documented_numbers():
    # Freeze the D11 table in code — a silent threshold edit must go red here.
    assert (PASS.mean_pos_m, PASS.max_pos_m, PASS.yaw_mean_deg, PASS.min_coverage) == \
        (0.10, 0.15, 10.0, 0.95)
    assert (DEPLOY.mean_pos_m, DEPLOY.max_pos_m, DEPLOY.yaw_mean_deg, DEPLOY.min_coverage) == \
        (0.07, 0.12, 7.0, 0.98)
