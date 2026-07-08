"""sim_comm.py — back-compat shim.

The engine-agnostic World-side Comm server moved to `humanoid.logic.oli.comm.world`
as `WorldComm` (so the MuJoCo/limx World can share it). The Isaac World keeps importing
it here under its historical `SimComm`/`SimCommError` names. New code should import
`WorldComm`/`WorldCommError` from `humanoid.logic.oli.comm.world` directly.
"""

from humanoid.logic.oli.comm.world import WorldComm as SimComm  # noqa: F401
from humanoid.logic.oli.comm.world import WorldCommError as SimCommError  # noqa: F401

__all__ = ["SimComm", "SimCommError"]
