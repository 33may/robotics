"""Drift guard for the injected joint armature (rotor inertia).

Isaac's HU_D04 USD ships armature=0 on every joint; the walk policy was tuned in
IsaacLab against the real rotor inertia from the HU_D04_01 MJCF. `ARMATURE_PR` is the
PR-ordered vector the World injects via `Oli.set_armature`. This pins each joint to the
authoritative MJCF group value so a future joint reorder / typo can't silently
re-introduce a massless rotor (which buzzes the stiff leg drives into divergence).

Pure (imports sim_world_main, which defers all isaacsim imports), so runs in `brain`.
"""

import pytest

from humanoid.logic.oli.contracts import NUM_JOINTS, PR_ORDER
from humanoid.logic.simulation.isaacsim import sim_world_main as w

pytestmark = pytest.mark.brain

# Authoritative HU_D04_01.xml MJCF rotor inertias (kg·m²).
_HIP_KNEE = 0.14125
_ANKLE_WAIST = 0.1845504
_SHOULDER_ELBOW = 0.0886706
_HEAD_WRIST = 0.0153218


def test_armature_vector_shape_and_order():
    assert len(w.ARMATURE_PR) == NUM_JOINTS
    # built straight from PR_ORDER, so index alignment is guaranteed
    assert w.ARMATURE_PR == [w._armature_for(n) for n in PR_ORDER]


def test_no_joint_is_massless():
    # the entire point: every joint must carry rotor inertia, none left at 0
    assert all(v > 0.0 for v in w.ARMATURE_PR)


def test_each_joint_maps_to_its_mjcf_group():
    expected = {
        "hip": _HIP_KNEE, "knee": _HIP_KNEE,
        "ankle": _ANKLE_WAIST, "waist": _ANKLE_WAIST,
        "shoulder": _SHOULDER_ELBOW, "elbow": _SHOULDER_ELBOW,
        "head": _HEAD_WRIST, "wrist": _HEAD_WRIST,
    }
    for name, value in zip(PR_ORDER, w.ARMATURE_PR):
        key = next(k for k in expected if k in name)
        assert value == pytest.approx(expected[key]), f"{name} armature wrong"


def test_ankle_differs_from_hip_within_a_leg():
    # the serial ankle inherits the heavier achilles A/B rotor inertia, not the hip's —
    # this asymmetry is the subtle bit a positional typo would flip
    i_hip = PR_ORDER.index("left_hip_pitch_joint")
    i_ankle = PR_ORDER.index("left_ankle_pitch_joint")
    assert w.ARMATURE_PR[i_hip] == pytest.approx(_HIP_KNEE)
    assert w.ARMATURE_PR[i_ankle] == pytest.approx(_ANKLE_WAIST)
    assert w.ARMATURE_PR[i_hip] != w.ARMATURE_PR[i_ankle]
