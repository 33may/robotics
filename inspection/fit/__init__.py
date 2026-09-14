"""Scene-conditioned primitive guards and the slim fit contract.

The X1 generic fitter bank (planar patches / cylinder RANSAC / EMS
superquadrics) lived here and lost: on partial step clouds, fitting bare
points cannot beat baking in what the scene already knows — the probed
table anchors the bottom face, gravity fixes the axis, the object class
supplies the topology, and the extents are the outer fuzz envelope.
`boxguard` is the recipe that won the completion bench.

What lives here now:

    box.py       boxguard — the cuboid guard (mesh + Decomposition faces)
    cyl.py       cylguard — the cylinder guard (next)
    contract.py  the slim output contract: `Primitive`, `Decomposition`,
                 closed kinds, lossless JSON round trip
    common.py    cloud helpers shared with `viewplan/`
"""
from inspection.fit.box import (boxguard, boxguard_core, boxguard_fit,
                                boxmesh)
from inspection.fit.contract import (KINDS, Decomposition, Primitive,
                                     abstained, decomposition_from_json,
                                     decomposition_to_json, frame_from_axis,
                                     pose_from)

__all__ = ["boxguard", "boxguard_core", "boxguard_fit", "boxmesh",
           "KINDS", "Decomposition", "Primitive", "abstained",
           "decomposition_from_json", "decomposition_to_json",
           "frame_from_axis", "pose_from"]
