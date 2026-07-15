"""Isaac-env test for Oli's opt-in head stereo pair (MAY-173 locdev T1).

Boots a real headless Isaac on a TEMP COPY of the asset tree (stereo prims baked in
by the fixture — the committed asset may or may not be re-baked yet), then checks
the attach contract:

  - `stereo_cameras=True` exposes exactly head_left/head_right via
    `stereo_camera_names` + `read_camera_rgb` (RGB-only: no depth annotator cost);
  - the stereo names never leak into `camera_names` — that's the RGBD table the
    CameraPublisher iterates with `read_camera_rgbd`, which stereo can't serve.

Marked `isaac` → run via `conda run -n isaac pytest`.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.isaac

_ASSET_USD = Path(__file__).resolve().parents[3] / "assets" / "oli" / "usd"


@pytest.fixture(scope="module")
def isaac_world():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from isaacsim.core.api import World  # noqa: E402 (must follow SimulationApp)
    world = World(stage_units_in_meters=1.0)
    yield world
    app.close()


@pytest.fixture(scope="module")
def stereo_oli(isaac_world, tmp_path_factory):
    from humanoid.logic.simulation.isaacsim.build_camera_usd import bake_cameras
    from humanoid.logic.simulation.isaacsim.oli import Oli

    dst = tmp_path_factory.mktemp("asset") / "usd"
    shutil.copytree(_ASSET_USD, dst)
    bake_cameras(dst / "configuration" / "HU_D04_01_sensor.usd")

    oli = Oli(
        isaac_world,
        usd_path=dst / "HU_D04_01.usd",
        cameras=True,
        stereo_cameras=True,
        camera_resolution=(640, 360),  # small: this test checks wiring, not optics
    )
    # Isaac annotators need render ticks before the first frame is readable.
    for _ in range(8):
        isaac_world.step(render=True)
    return oli


def test_stereo_pair_attached_and_readable(stereo_oli):
    assert stereo_oli.stereo_camera_names == ["head_left", "head_right"]
    for name in stereo_oli.stereo_camera_names:
        rgb = stereo_oli.read_camera_rgb(name)
        assert rgb.shape == (360, 640, 3)
        assert rgb.dtype == np.uint8


def test_stereo_names_do_not_leak_into_rgbd_table(stereo_oli):
    # camera_names is the CameraPublisher's iteration set (read_camera_rgbd) —
    # stereo cams have no depth annotator and must never appear there.
    assert set(stereo_oli.camera_names) == {"chest", "head"}


def test_stereo_off_by_default(isaac_world, tmp_path_factory):
    from humanoid.logic.simulation.isaacsim.build_camera_usd import bake_cameras
    from humanoid.logic.simulation.isaacsim.oli import Oli

    dst = tmp_path_factory.mktemp("asset_nostereo") / "usd"
    shutil.copytree(_ASSET_USD, dst)
    bake_cameras(dst / "configuration" / "HU_D04_01_sensor.usd")

    oli = Oli(
        isaac_world,
        prim_path="/World/OliNoStereo",
        usd_path=dst / "HU_D04_01.usd",
        cameras=True,
        camera_resolution=(640, 360),
    )
    assert oli.stereo_camera_names == []
