"""Conformance gate for THIS candidate — runs inside its own `bench-cuvslam` env:

    conda run -n bench-cuvslam python -m pytest logic/oli/reason/localization/realizations/cuvslam/

Deliberately NOT part of the repo's brain-marked suite (playbook §Conformance): dep-heavy
candidates must never be importable from the brain env. Green here = phase-4 "refine" done on
the contract; accuracy is locbench's verdict, never this file's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from humanoid.logic.oli.contracts import CameraFrame, CameraIntrinsics, Observation
from humanoid.logic.oli.reason.localization import (
    LocalizationIn,
    LocalizationModule,
    LocalizationSetup,
    RobotPose,
)
from humanoid.logic.oli.reason.localization.testing import verify_module_contract

# /loc-new rewrites `_template` and the class name below. ABSOLUTE import on purpose: a
# relative `.module` makes standalone pytest runs import the package under a second namespace
# (`logic.…` vs `humanoid.logic.…`) and isinstance checks then fail on identity.
from humanoid.logic.oli.reason.localization.realizations.cuvslam.module import (
    CuvslamModule,
)


def _obs(stamp_ns=1):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _loc_in(stamp_ns):
    frame = CameraFrame(
        stamp_ns=stamp_ns, name="head",
        rgb=np.zeros((3, 4, 3), dtype=np.uint8), depth=np.ones((3, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=4, height=3, fx=2.0, fy=2.0, cx=2.0, cy=1.5),
    )
    return LocalizationIn(stamp_ns=stamp_ns, frames={"head": frame}, observation=_obs(stamp_ns))


def test_module_satisfies_the_protocol():
    assert isinstance(CuvslamModule(), LocalizationModule)


def test_module_passes_the_conformance_checker(tmp_path: Path):
    outs = verify_module_contract(
        CuvslamModule(),
        LocalizationSetup(map_dir=tmp_path, initial_pose=RobotPose(stamp_ns=0, x=0.0, y=0.0)),
        [_loc_in(10), _loc_in(20), _loc_in(30)],
    )
    assert len(outs) == 3


def test_stop_tolerates_failed_or_absent_start():
    CuvslamModule().stop()


# ── map_relative mode (it-3) ──────────────────────────────────────────────────────


def _map_relative_module():
    return CuvslamModule({"mode": "map_relative"})


def test_map_relative_satisfies_the_protocol():
    assert isinstance(_map_relative_module(), LocalizationModule)


def test_map_relative_bad_map_degrades_to_lost_not_crash(tmp_path: Path):
    """Empty map_dir (no registration_gt.json) = dead episode: every step answers LOST,
    nothing raises — the re-hint path relies on this (host boots a fresh instance)."""
    outs = verify_module_contract(
        _map_relative_module(),
        LocalizationSetup(map_dir=tmp_path, initial_pose=RobotPose(stamp_ns=0, x=0.0, y=0.0)),
        [_loc_in(10), _loc_in(20), _loc_in(30)],
    )
    assert len(outs) == 3
    assert all(o.pose is None for o in outs)


def test_map_relative_stop_tolerates_failed_start():
    _map_relative_module().stop()


# ── periodic self-relocalize (it-8) — real dump + map, GPU ────────────────────────

_HUMANOID = Path(__file__).resolve().parents[6]
_DUMP = _HUMANOID / "data/coverage_drives/teleop_v1_demo"
_MAP = (_HUMANOID / "data/maps/teleop_v1_demo/"
        "2026-07-16_14-25-28_teleop_v1_demo_bag/pycuvslam_map")
_needs_data = pytest.mark.skipif(
    not (_DUMP.is_dir() and _MAP.is_dir()),
    reason="teleop_v1_demo dump/map not on disk")


def _run_dump_frames(period_s: float, n_frames: int = 150, registration="gt", hint=None):
    """Replay the first `n_frames` dump ticks (30 Hz ≈ 5 s) through map_relative with
    localization.relocalize_period_s overridden; returns (outs, final diagnostics, gt).

    registration="gt" = config default (W-frame emission, GT hint);
    registration=None  = identity (M IS the world — the 0-GT demo path); pass `hint`
    as the M-frame start pose then (the bake's start_pose.json, (0,0,0) by construction)."""
    import json as _json

    import yaml
    from PIL import Image

    rig = _json.loads((_DUMP / "rig.json").read_text())

    def _intr(name):
        i = rig["cameras"][name]["intrinsics"]
        return CameraIntrinsics(width=i["width"], height=i["height"],
                                fx=i["fx"], fy=i["fy"], cx=i["cx"], cy=i["cy"])

    gt = {}
    for line in open(_DUMP / "poses.jsonl"):
        r = _json.loads(line)
        if r["cam"] == "base":
            gt[r["stamp_ns"]] = (r["x"], r["y"], r["yaw"])
    stamps = sorted(int(p.stem) for p in (_DUMP / "frames/head_left").glob("*.jpg"))
    stamps = [s for s in stamps if s in gt][:n_frames]

    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text()) or {}
    cfg.setdefault("localization", {})["relocalize_period_s"] = period_s
    if registration != "gt":
        cfg["localization"]["registration_file"] = registration
    module = CuvslamModule(cfg)
    h = hint if hint is not None else gt[stamps[0]]
    module.start(LocalizationSetup(
        map_dir=_MAP,
        initial_pose=RobotPose(stamp_ns=stamps[0], x=h[0], y=h[1], yaw=h[2])))
    outs = []
    for ts in stamps:
        frames = {
            name: CameraFrame(
                stamp_ns=ts, name=name,
                rgb=np.asarray(Image.open(_DUMP / f"frames/{name}/{ts:019d}.jpg")),
                depth=None, intrinsics=_intr(name))
            for name in ("head_left", "head_right")
        }
        outs.append(module.step(LocalizationIn(stamp_ns=ts, frames=frames,
                                               observation=_obs(ts))))
    diag = module.diagnostics()
    module.stop()
    return outs, diag, {ts: gt[ts] for ts in stamps}


@_needs_data
def test_periodic_relocalize_fires_and_tracking_survives():
    """Tiny period (2 s over a 5 s replay) → at least one mid-tracking self-reloc happens,
    every step still answers TRACKING, nothing crashes (corpus: LocalizeInMap mid-tracking
    is non-destructive)."""
    outs, diag, _ = _run_dump_frames(period_s=2.0)
    assert diag is not None and diag["reloc_ok"] >= 1
    assert all(o.status.name == "TRACKING" for o in outs)


@_needs_data
def test_periodic_relocalize_period_zero_is_off():
    """relocalize_period_s: 0 = feature off — counters stay 0, behavior parity with it-7."""
    outs, diag, _ = _run_dump_frames(period_s=0.0)
    assert diag is not None and diag["reloc_ok"] == 0 and diag["reloc_fail"] == 0
    assert all(o.status.name == "TRACKING" for o in outs)


@_needs_data
def test_identity_registration_emits_m_frame_poses():
    """registration_file: null → M IS the world (the 0-GT bake-time-alignment demo path):
    hint = the bake's start_pose (M origin), emitted poses come out raw — start ≈ (0,0,0)
    and per-window displacement magnitude matches GT's (frame-independent scalar)."""
    outs, _, gt = _run_dump_frames(period_s=0.0, registration=None, hint=(0.0, 0.0, 0.0))
    assert all(o.status.name == "TRACKING" for o in outs)
    p0, pn = outs[0].pose, outs[-1].pose
    assert np.hypot(p0.x, p0.y) < 0.3, "first pose should sit at the M-origin dock"
    g = list(gt.values())
    d_gt = np.hypot(g[-1][0] - g[0][0], g[-1][1] - g[0][1])
    d_m = np.hypot(pn.x - p0.x, pn.y - p0.y)
    assert abs(d_m - d_gt) < 0.3, f"M displacement {d_m:.2f} vs GT {d_gt:.2f}"


def test_registration_round_trip(tmp_path: Path):
    """W↔M SE(2) bridge must invert exactly — pose emission uses w_from_m, hint uses m_from_w."""
    import json as _json
    import math as _math

    from humanoid.logic.oli.reason.localization.realizations.cuvslam.module import (
        _Registration,
    )

    theta = 0.0115  # ~0.66° — the real teleop_v1_demo registration scale
    (tmp_path / "registration_gt.json").write_text(_json.dumps({
        "R": [[_math.cos(theta), -_math.sin(theta)],
              [_math.sin(theta), _math.cos(theta)]],
        "t": [0.156, -0.032],
        "theta_rad": theta,
    }))
    reg = _Registration.load(tmp_path / "registration_gt.json")
    x, y, yaw = 3.2, -1.7, 0.8
    xm, ym, yawm = reg.m_from_w(x, y, yaw)
    xw, yw, yaww = reg.w_from_m(xm, ym, yawm)
    assert abs(xw - x) < 1e-9 and abs(yw - y) < 1e-9 and abs(yaww - yaw) < 1e-9
    # and the transform is not identity (it actually moves points)
    assert abs(xm - x) > 1e-3 or abs(ym - y) > 1e-3
