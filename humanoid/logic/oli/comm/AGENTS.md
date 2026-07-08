# comm/ — Communication (the World edge)

`Comm` is the **only world-aware layer** and the **Robot↔World edge**. It translates between the invariant Robot contracts and whatever World is attached. It is **not** an internal hub — internal modules talk over the bus, not through here.

## Rules

- **Owns world-specific translation** — joint reorder, unit / PR-space mapping, per-World encode/decode. World-specifics live here and nowhere else.
- **No decision logic, no timing** — no policy, no action buffer, no control-rate loop (that's `PolicyRunner`). Comm only prepares + delivers.
- **Two channels** — control (`AF_UNIX` SEQPACKET, fixed-size, `protocol.py`) + frame (`SOCK_STREAM`, variable, `frame_protocol.py`).
- **Wire is canonical & pure** — `protocol.py` / `frame_protocol.py` are stdlib-only `struct.pack` layouts (no numpy / isaacsim / limxsdk), byte-identical across py3.8 (`RealComm`) and py3.11 (`SimComm`). Keep them pure.
- **New messages are additive + versioned** — classify each as spine (invariant) or flagged auxiliary (like `GLIDE_CMD`).

See `docs/architecture/architecture.md` §4–§5.
