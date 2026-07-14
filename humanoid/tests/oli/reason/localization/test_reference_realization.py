"""TDD for the reference candidate (realizations/reference/) — locbench D13, tasks 8.1.

GT replay through the FULL hosting path with injectable failure modes: the module binds a
bench-only GT feed socket (path via `Setup.calibration["gt_feed_socket"]` — the evaluator
republishes GT there; real candidates ignore the key) and answers each step with the newest
GT sample, degraded per config: constant bias, Gaussian noise (seeded), dropout (seeded),
delay. It validates host + wire + scorer with zero SLAM dependencies — and it is the
exemplar realization the playbook points at. Pure (stdlib/numpy) → `brain` env, per the
conformance-split in realizations/AGENTS.md.
"""

import numpy as np
import pytest

from humanoid.logic.oli.comm.debug_pose import DebugPoseServer
from humanoid.logic.oli.contracts import CameraFrame, CameraIntrinsics, Observation
from humanoid.logic.oli.reason.localization import (
    LocalizationIn,
    LocalizationModule,
    LocalizationSetup,
    LocalizationStatus,
    load_realization,
)
from humanoid.logic.oli.reason.localization.realizations.reference.module import (
    ReferenceModule,
    build,
)
from humanoid.logic.oli.reason.localization.testing import verify_module_contract

pytestmark = pytest.mark.brain

MS = 1_000_000

# static canary — declaration-site conformance (reason/AGENTS.md checklist)
_static_reference: LocalizationModule = build({})


def _obs(stamp_ns):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.zeros(3, dtype=np.float32), gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
    )


def _loc_in(stamp_ns):
    frame = CameraFrame(
        stamp_ns=stamp_ns, name="head",
        rgb=np.zeros((3, 4, 3), dtype=np.uint8), depth=np.ones((3, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=4, height=3, fx=2.0, fy=2.0, cx=2.0, cy=1.5),
    )
    return LocalizationIn(stamp_ns=stamp_ns, frames={"head": frame},
                          observation=_obs(stamp_ns))


@pytest.fixture()
def rig(tmp_path):
    """Per-module feed sockets: a datagram path has ONE reader, so each module gets its own
    (server, path) pair — exactly how the evaluator runs it (one candidate per run)."""
    servers, modules = [], []
    counter = [0]

    def make(config=None):
        counter[0] += 1
        path = str(tmp_path / f"gt-feed-{counter[0]}.sock")
        m = build(config or {})
        m.start(LocalizationSetup(map_dir="/tmp/nomap",
                                  calibration={"gt_feed_socket": path}))
        server = DebugPoseServer(path)
        servers.append(server)
        modules.append(m)
        return m, server

    yield make
    for m in modules:
        m.stop()
    for s_ in servers:
        s_.close()


def test_clean_reference_echoes_gt(rig):
    m, server = rig()
    server.publish(90 * MS, 1.5, -2.0, 0.3)
    out = m.step(_loc_in(100 * MS))
    assert out.status is LocalizationStatus.TRACKING
    assert out.stamp_ns == 100 * MS                       # answers the INPUT stamp
    assert (out.pose.x, out.pose.y, out.pose.yaw) == (1.5, -2.0, 0.3)


def test_no_gt_yet_is_honest_lost(rig):
    m, _ = rig()
    out = m.step(_loc_in(10 * MS))
    assert out.status is LocalizationStatus.LOST and out.pose is None


def test_constant_bias_injection(rig):
    m, server = rig({"inject": {"bias_x_m": 0.2}})
    server.publish(90 * MS, 1.0, 1.0, 0.0)
    out = m.step(_loc_in(100 * MS))
    assert out.pose.x == pytest.approx(1.2)
    assert out.pose.y == pytest.approx(1.0)


def test_noise_is_seeded_and_bounded(rig):
    m1, s1 = rig({"inject": {"noise_sigma_m": 0.01, "seed": 5}})
    m2, s2 = rig({"inject": {"noise_sigma_m": 0.01, "seed": 5}})
    s1.publish(90 * MS, 1.0, 1.0, 0.0)
    s2.publish(90 * MS, 1.0, 1.0, 0.0)
    o1 = m1.step(_loc_in(100 * MS))
    o2 = m2.step(_loc_in(100 * MS))
    assert o1.pose.x == o2.pose.x                          # same seed → same draw
    assert abs(o1.pose.x - 1.0) < 0.1


def test_dropout_goes_lost_at_the_configured_rate(rig):
    m, server = rig({"inject": {"dropout": 0.5, "seed": 7}})
    lost = 0
    for i in range(200):
        server.publish((90 + i * 10) * MS, 1.0, 1.0, 0.0)
        out = m.step(_loc_in((100 + i * 10) * MS))
        lost += out.status is LocalizationStatus.LOST
    assert 60 < lost < 140                                 # ≈50%, seeded


def test_passes_the_conformance_checker(tmp_path):
    # verify_module_contract drives the whole lifecycle itself; the GT publish must land
    # AFTER start() binds the feed socket → publish lazily from the input generator.
    path = str(tmp_path / "gt-feed.sock")
    server = DebugPoseServer(path)

    def loc_ins():
        for i in range(5):
            server.publish((5 + i * 10) * MS, 0.1 * i, 0.0, 0.0)
            yield _loc_in((10 + i * 10) * MS)

    outs = verify_module_contract(
        build({}),
        LocalizationSetup(map_dir="/tmp/nomap",
                          calibration={"gt_feed_socket": path}),
        loc_ins(),
    )
    server.close()
    assert len(outs) == 5
    assert outs[-1].status is LocalizationStatus.TRACKING


def test_loads_via_the_registry_with_overrides():
    m = load_realization("reference", overrides={"inject": {"bias_x_m": 0.2}})
    assert isinstance(m, ReferenceModule)
    assert m.config["inject"]["bias_x_m"] == 0.2


def test_stop_tolerates_failed_start():
    m = build({})
    try:
        m.start(LocalizationSetup(map_dir="/x"))           # no gt_feed_socket key
    except Exception:
        pass
    m.stop()                                               # must not raise
