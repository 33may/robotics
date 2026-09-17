"""derive_r_floor pick rule — the count profiles are the measured sweeps
(block_report --rs, 2026-09-15) that motivated the per-run floor."""
import numpy as np
import pytest

from inspection.view.viewsphere import derive_r_floor


def counter(profile):
    """count_at stub from {radius: reachable_count}, tracking probes."""
    probed = []

    def count_at(r):
        probed.append(round(r, 4))
        return profile[round(r, 4)]

    return count_at, probed


def test_want_above_static_floor_is_untouched():
    count_at, probed = counter({})
    assert derive_r_floor(None, None, None, 0.31, count_at=count_at) == 0.31
    assert probed == []          # no probing when the floor is not in the way


def test_box_profile_floor_drops_to_020():
    # bench box 1509: flat 23/36 down to 0.20, cliff at 0.18
    count_at, _ = counter({0.24: 23, 0.22: 23, 0.20: 23, 0.19: 19})
    assert derive_r_floor(None, None, None, 0.19, count_at=count_at) == 0.20


def test_cup_profile_keeps_static_floor():
    # cup 2408-cup1: 27/36 at 0.24, cliff straight to 18 at 0.22
    count_at, probed = counter({0.24: 27, 0.22: 18})
    assert derive_r_floor(None, None, None, 0.20, count_at=count_at) == 0.24
    assert probed == [0.24, 0.22]   # stops at the first losing radius


def test_want_exactly_on_a_flat_profile_is_granted():
    count_at, _ = counter({0.24: 23, 0.22: 23, 0.20: 23})
    assert derive_r_floor(None, None, None, 0.20, count_at=count_at) == 0.20
