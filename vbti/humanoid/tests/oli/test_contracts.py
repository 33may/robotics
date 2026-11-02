"""Tests for the deployment-invariant contracts (humanoid.logic.oli.contracts).

These lock the three canonical-PR contracts and — critically — the invariant that
importing the brain pulls in no world SDK. Brain-pure: runs in the `brain` env.
"""

import sys

import numpy as np
import pytest

from humanoid.logic.oli import (
    NUM_JOINTS,
    PR_ORDER,
    Intent,
    Mode,
    Observation,
    PolicyIn,
    PolicyOut,
)

pytestmark = pytest.mark.brain


# ── PR_ORDER / Mode ──────────────────────────────────────────────────────────

def test_pr_order_is_31_unique_joints():
    assert len(PR_ORDER) == NUM_JOINTS == 31
    assert len(set(PR_ORDER)) == 31, "PR_ORDER has duplicate joint names"


def test_mode_members_are_ints():
    assert int(Mode.STAND) == 0
    assert int(Mode.WALK) == 1


# ── Observation ──────────────────────────────────────────────────────────────

def _obs(**overrides):
    kw = dict(
        stamp_ns=123,
        q=[0.0] * 31, dq=[0.0] * 31, tau=[0.0] * 31,
        acc=[0.0, 0.0, -9.8], gyro=[0.0, 0.0, 0.0], quat_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    kw.update(overrides)
    return Observation(**kw)


def test_observation_coerces_lists_to_float32_pr_vectors():
    o = _obs()
    assert o.q.dtype == np.float32 and o.q.shape == (31,)
    assert o.acc.shape == (3,) and o.gyro.shape == (3,)
    assert o.quat_wxyz.shape == (4,)
    assert isinstance(o.stamp_ns, int)


def test_observation_is_frozen():
    o = _obs()
    with pytest.raises((AttributeError, TypeError)):
        o.q = np.zeros(31)


@pytest.mark.parametrize("field,bad", [
    ("q", [0.0] * 30), ("dq", [0.0] * 32), ("tau", [0.0] * 1),
    ("acc", [0.0, 0.0]), ("gyro", [0.0] * 4), ("quat_wxyz", [1.0, 0.0, 0.0]),
])
def test_observation_rejects_wrong_length(field, bad):
    with pytest.raises(ValueError):
        _obs(**{field: bad})


# ── Intent / PolicyIn ────────────────────────────────────────────────────────

def test_intent_velocity_fields_default_to_zero():
    i = Intent(mode=Mode.STAND)
    assert (i.v_x, i.v_y, i.w_z) == (0.0, 0.0, 0.0)


def test_policy_in_carries_observation_and_intent():
    o = _obs()
    pin = PolicyIn(observation=o, intent=Intent(mode=Mode.WALK, v_x=0.5))
    assert pin.intent.mode == Mode.WALK
    assert pin.intent.v_x == 0.5
    assert pin.observation is o


# ── PolicyOut ────────────────────────────────────────────────────────────────

def _pout(**overrides):
    kw = dict(
        stamp_ns=1,
        q_des=[0.0] * 31, dq_des=[0.0] * 31, tau_ff=[0.0] * 31,
        kp=[10.0] * 31, kd=[1.0] * 31, mode=[0] * 31,
    )
    kw.update(overrides)
    return PolicyOut(**kw)


def test_policy_out_resolves_to_31_pr_vectors():
    po = _pout()
    for name in ("q_des", "dq_des", "tau_ff", "kp", "kd"):
        v = getattr(po, name)
        assert v.dtype == np.float32 and v.shape == (31,), name
    assert po.mode.dtype == np.int32 and po.mode.shape == (31,)


def test_policy_out_is_frozen():
    po = _pout()
    with pytest.raises((AttributeError, TypeError)):
        po.kp = np.zeros(31)


def test_policy_out_rejects_wrong_length_gains():
    with pytest.raises(ValueError):
        _pout(kp=[10.0] * 30)


# ── The load-bearing invariant ───────────────────────────────────────────────

def test_brain_import_graph_has_no_world_sdk():
    """Importing the brain must never pull in isaacsim or limxsdk."""
    assert "isaacsim" not in sys.modules, "isaacsim leaked into the brain import graph"
    assert "limxsdk" not in sys.modules, "limxsdk leaked into the brain import graph"
