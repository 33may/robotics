"""mapping/ — World-side offline mapping tooling (MAY-173 locdev).

The data-collection flow for map baking: a swappable motion source
(`waypoint_follower`) drives the base, a motion-agnostic `recorder` dumps
frames + rendered camera poses to disk. Offline tooling — GT is allowed here
(the bake is not the runtime path). Pure numpy/stdlib(+PIL); never isaacsim.
"""
