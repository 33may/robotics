"""humanoid.logic.oli.comm — the Communication boundary (brain side).

`Communication` is the only world-aware layer (D4). On the brain side it is the
client that connects to the World server, sends `PolicyOut`, and reads the latest
`Observation`. The wire (`protocol`) and the contract↔wire mapping (`codec`) are
PURE (no isaacsim/limxsdk); the world realisations live elsewhere:
`SimComm` under `logic/simulation/isaacsim`, `RealComm` (deferred) in the py3.8 edge.

Kept import-light so the py3.8 `RealComm` and the py3.11 brain can both pull just
the wire + codec without dragging in socket-client machinery.
"""
