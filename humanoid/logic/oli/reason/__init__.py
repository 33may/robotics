"""humanoid.logic.oli.reason — the Reasoning layer.

Reasoning is the single producer of `PolicyIn` (D5): it consumes an `Observation` plus
an external operator signal and emits `PolicyIn` directly (no separate Reason→Command
seam). The foundation realization is `Teleop` + `JoystickAdapter`; richer reasoning
(SLAM/VLA/autonomy) plugs in here later behind the same `→ PolicyIn` contract.
"""
