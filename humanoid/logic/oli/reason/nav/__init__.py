"""humanoid.logic.oli.reason.nav — the navigation reasoning stack (2D PoC).

Composes a `Localizer` (pose source) + costmap + planner into a Nav reason module that emits a
base-velocity `Intent` toward a goal — reusing the existing glide path (`GlideAction` →
`GLIDE_CMD`) unchanged. Everything here is world-invariant (no isaacsim/limxsdk), guarded by the
`brain` pytest marker. See docs/architecture/architecture.md §6–7 and reason/AGENTS.md.
"""

from .arm import ArmedNav
from .controller import PurePursuit
from .nav import Nav
from .planner import Planner, plan_path
from .types import GoalCoordinate

__all__ = [
    "GoalCoordinate",
    "Planner",
    "plan_path",
    "PurePursuit",
    "Nav",
    "ArmedNav",
]
