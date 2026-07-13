"""nav/factory.py — the one tuned Nav recipe, shared by every brain host.

The robot/policy knobs below were tuned live in the glide demo (MAY-173 nav bring-up) and
used to live in dev_app's `brain_link.py`. They are properties of the ROBOT and its planner
policy — not of any particular host — so they live here and every host (`brain_main
--service`, dev_app's BrainLink, later the real robot stack) calls `build_nav(...)` and gets
the identical Nav. Change a knob HERE and every host follows; never re-tune per host.

`speed_scale`: GlideAction multiplies commanded velocities by its `speed_scale` downstream,
so the controller caps are PRE-DIVIDED by the same factor — the product (what Oli actually
does when armed) stays at `_NAV_SPEED_MS` / `_NAV_YAW_RS` regardless of the glide demo's
stick-gain. Non-glide hosts pass the default 1.0. Pure: no isaacsim/limxsdk.
"""

from __future__ import annotations

from ..localization import Localizer
from ..mapping import MappingModule
from .controller import PurePursuit
from .nav import Nav
from .planner import Planner

# Robot/policy planning knobs (footprint + clearance) — tuned live, see module docstring.
_ROBOT_RADIUS_M = 0.30       # hard footprint: cells within this of a wall are impassable
_INFLATION_RADIUS_M = 1.0    # soft clearance reach (> robot radius) — path prefers this much gap
_CLEARANCE_WEIGHT = 3.0      # how hard to trade path length for clearance (0 = shortest path)
_HEURISTIC_WEIGHT = 1.2      # weighted A*: ~30× fewer nodes on open routes, near-lossless clearance
_HORIZON_M = 2.0             # local re-plan only re-solves this far ahead; the far tail is reused
_NAV_SPEED_MS = 1.0          # armed autonomy target forward speed [m/s] (after glide rescale)
_NAV_YAW_RS = 1.2            # armed autonomy target yaw rate [rad/s] (after glide rescale)


def build_nav(
    mapping: MappingModule,
    localizer: Localizer,
    *,
    speed_scale: float = 1.0,
) -> Nav:
    """Compose the tuned Nav: `mapping` + `localizer` behind their seams, knobs pinned here."""
    return Nav(
        mapping,
        localizer,
        controller=PurePursuit(
            max_lin=_NAV_SPEED_MS / speed_scale,
            max_wz=_NAV_YAW_RS / speed_scale,
        ),
        planner=Planner(
            robot_radius_m=_ROBOT_RADIUS_M,
            inflation_radius_m=_INFLATION_RADIUS_M,
            clearance_weight=_CLEARANCE_WEIGHT,
            heuristic_weight=_HEURISTIC_WEIGHT,
            horizon_m=_HORIZON_M,
        ),
    )
