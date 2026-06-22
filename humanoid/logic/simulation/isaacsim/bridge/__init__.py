"""
bridge package for the Isaac ↔ limxsdk MROS sim peer.

Phase 3 (current): `protocol.py` defines the wire format.
Phase 4: `sidecar.py` — Py 3.8 process owning `limxsdk.Robot`.
Phase 6: `OliBridge` — Py 3.11 client + sidecar lifecycle manager.

See `humanoid/openspec/changes/may-147-isaac-limx-sdk-bridge/design.md`.
"""
