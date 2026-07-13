"""humanoid.logic.oli.reason.nav — the navigation reasoning stack (2D PoC).

Composes a `Localizer` (pose source) + costmap + planner into a Nav reason module that emits a
base-velocity `Intent` toward a goal — reusing the existing glide path (`GlideAction` →
`GLIDE_CMD`) unchanged. Everything here is world-invariant (no isaacsim/limxsdk), guarded by the
`brain` pytest marker. See docs/architecture/architecture.md §6–7 and reason/AGENTS.md.
"""

from .arm import ArmedNav
from .controller import PurePursuit
from .costmap import OccupancyGrid
from .localizer import DebugPoseLocalizer, GroundTruthLocalizer, Localizer
from .nav import Nav
from .occupancy_io import convert_ros_map, load_occupancy, occupancy_from_image, save_occupancy
from .planner import plan_path
from .types import GoalCoordinate, RobotPose

__all__ = [
    "RobotPose",
    "GoalCoordinate",
    "Localizer",
    "GroundTruthLocalizer",
    "DebugPoseLocalizer",
    "OccupancyGrid",
    "plan_path",
    "PurePursuit",
    "Nav",
    "ArmedNav",
    "load_occupancy",
    "save_occupancy",
]
