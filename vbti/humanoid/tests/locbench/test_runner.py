"""TDD for the runner's pure parts (logic/locbench/runner.py) — tasks 6.2-6.4.

`write_run_artifacts` (results → pairs.csv + report.json + plots), `score_run_dir`
(recompute offline from stored pairs — same numbers), `board` (MD scoreboard from the
latest report per candidate). The live `run_bench` spawns the launcher and is validated by
the §8 acceptance triplet, not unit tests. `brain` env.
"""

import json

import numpy as np
import pytest

from humanoid.logic.locbench.episodes import Episode, EpisodeSet
from humanoid.logic.locbench.evaluator import EpisodeResult
from humanoid.logic.locbench.runner import board, score_run_dir, write_run_artifacts
from humanoid.logic.oli.reason.localization import (
    LocalizationOut,
    LocalizationStatus,
    RobotPose,
)
from humanoid.logic.oli.reason.mapping import OccupancyGrid

pytestmark = pytest.mark.brain

S = 1_000_000_000


def _grid(n=40):
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return OccupancyGrid(occ, 0.5)


def _episode_set():
    eps = (Episode(id=0, spawn=(2.0, 5.0), goal=(12.0, 5.0), route_m=10.0),
           Episode(id=1, spawn=(5.0, 2.0), goal=(5.0, 12.0), route_m=10.0))
    return EpisodeSet(scene="testscene", seed=7, map_dir="maps/test",
                      constraints=(("n_episodes", 2.0),), episodes=eps, coverage_goals=())


def _result(ep, bias=0.0, n=60, outcome="arrived"):
    t0 = 10 * S
    ests, gts = [], []
    for i in range(n):
        t = t0 + int(i * 0.1 * S)
        x = 2.0 + i * 0.15
        gts.append((t, x, 5.0, 0.0))
        ests.append(LocalizationOut(stamp_ns=t, pose=RobotPose(t, x + bias, 5.0, 0.0),
                                    status=LocalizationStatus.TRACKING))
    return EpisodeResult(episode=ep, outcome=outcome, ests=ests, gts=gts,
                         episode_start_ns=t0)


def test_artifacts_written_and_verdict_correct(tmp_path):
    es = _episode_set()
    results = [_result(es.episodes[0], bias=0.02), _result(es.episodes[1], bias=0.2)]
    doc = write_run_artifacts(tmp_path / "run1", candidate="ref", episode_set=es,
                              results=results, grid=_grid(),
                              provenance={"adapter_git": "abc"})
    assert doc["run"]["tier"] == "FAIL" and doc["run"]["failed_episodes"] == [1]
    d = tmp_path / "run1"
    for f in ("report.json", "ep0_pairs.csv", "ep1_pairs.csv",
              "run_sheet.png", "error_timeline.png", "error_cdf.png"):
        assert (d / f).exists(), f
    assert json.loads((d / "report.json").read_text())["provenance"]["adapter_git"] == "abc"


def test_score_recomputes_identical_numbers(tmp_path):
    es = _episode_set()
    results = [_result(es.episodes[0], bias=0.05), _result(es.episodes[1], bias=0.05)]
    doc1 = write_run_artifacts(tmp_path / "r", candidate="ref", episode_set=es,
                               results=results)
    doc2 = score_run_dir(tmp_path / "r", es)
    assert doc2["run"] == doc1["run"]
    assert [e["stats"]["pos_mean"] for e in doc2["episodes"]] == \
        [e["stats"]["pos_mean"] for e in doc1["episodes"]]


def test_board_shows_latest_run_per_candidate(tmp_path):
    es = _episode_set()
    write_run_artifacts(tmp_path / "ref" / "20260713-1000", candidate="ref",
                        episode_set=es, results=[_result(es.episodes[0], bias=0.2)])
    write_run_artifacts(tmp_path / "ref" / "20260713-1100", candidate="ref",
                        episode_set=es, results=[_result(es.episodes[0], bias=0.02)])
    md = board(tmp_path)
    assert "| ref | 20260713-1100 |" in md      # the latest run, not the first
    assert "DEPLOY" in md
    assert md.count("| ref |") == 1
