"""reason/localization/ — the pose-source module (change `may-173-reason-module-separation`).

Owns the localization contracts (`LocalizationIn` → `LocalizationOut`), the `LocalizationModule`
protocol candidate algorithms implement, and the thin in-brain `Localizer` seam with its GT/debug
realizations. Algorithm-private 3D maps live INSIDE this module (opaque `map_dir`) until a
`world_representation` split is warranted. Pure: numpy/stdlib only.
"""

from .contracts import (
    LocalizationIn,
    LocalizationOut,
    LocalizationSetup,
    LocalizationStatus,
    RobotPose,
)
from .host import HostLocalizer, LocalizationHost
from .localizer import DebugPoseLocalizer, GroundTruthLocalizer, Localizer
from .module import LocalizationModule
from .registry import list_realizations, load_realization

__all__ = [
    "HostLocalizer",
    "LocalizationHost",
    "list_realizations",
    "load_realization",
    "DebugPoseLocalizer",
    "GroundTruthLocalizer",
    "LocalizationIn",
    "LocalizationModule",
    "LocalizationOut",
    "LocalizationSetup",
    "LocalizationStatus",
    "Localizer",
    "RobotPose",
]
