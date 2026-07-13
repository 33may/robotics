"""reason/mapping/ — the world-truth module (change `may-173-reason-module-separation`).

Owns the map: the `OccupancyGrid` type, the baked-artifact IO, and the versioned `Map` snapshot
contract behind `MappingModule.latest()`. World truth only — footprint/clearance live in the
planner's robot layer. Pure: numpy/stdlib (PIL/yaml lazy in `occupancy_io`).
"""

from .contracts import Map
from .costmap import OccupancyGrid
from .module import MappingModule, StaticMapping
from .occupancy_io import convert_ros_map, load_occupancy, occupancy_from_image, save_occupancy

__all__ = [
    "Map",
    "MappingModule",
    "OccupancyGrid",
    "StaticMapping",
    "convert_ros_map",
    "load_occupancy",
    "occupancy_from_image",
    "save_occupancy",
]
