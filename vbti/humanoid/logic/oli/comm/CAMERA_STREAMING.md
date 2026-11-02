# Oli camera streaming (World → consumer)

The camera surface of the Communication boundary: how the World renders Oli's cameras and
ships RGBD to a consumer (dev app, brain/SLAM) over a dedicated channel, engine-invariantly.
This is the PRODUCER + transport side; the consumer-side contract the dev app implements is in
[`devapp/sources/README.md`](../devapp/sources/README.md).

Sibling of the control channel: cameras never travel the 1 kHz `SEQPACKET` control socket — a
720p RGB frame (~2.8 MB) far exceeds a datagram, and frames are slow (~render rate) vs control.

## Data flow

```
Oli (Isaac render)  ->  CameraPublisher  ->  [AF_UNIX SOCK_STREAM]  ->  CameraStreamReader  ->  CameraFrame
   read_camera_rgbd      encode (uint16 mm)     per-name framing         demux by name           (consumer)
```

## Contract — `CameraFrame` (`logic/oli/contracts.py`)

| field | type | notes |
|---|---|---|
| `stamp_ns` | int | World sim time (design D8) |
| `name` | str | `"chest"` \| `"head"` (≤16 ASCII on the wire) |
| `rgb` | uint8 `(H,W,3)` | RGB |
| `depth` | float32 `(H,W)` | meters, planar Z; 0 = invalid |
| `intrinsics` | `CameraIntrinsics(width,height,fx,fy,cx,cy)` | **no extrinsics** — pose is FK(Observation, mount table) brain-side |

Streams + intrinsics + static mounts come from `logic/oli/camera_mounts.py` (`CAMERAS`,
`rgb_intrinsics`) — pure, shared by the USD bake and the brain FK. D435i default 1280×720.

## Entry points

| role | module / class | key API |
|---|---|---|
| World publisher | `comm/camera_publisher.py` · `CameraPublisher` | `CameraPublisher(body, socket_path="/tmp/oli-world-frames.sock", every=1)`; `publish(tick, stamp_ns=None)`; `close()` |
| consumer reader | `comm/camera_stream.py` · `CameraStreamReader` | `connect(timeout)`; `read(name) -> CameraFrame\|None` (non-blocking, non-consuming); `stream_names()`; `close()` |
| wire framing | `comm/frame_protocol.py` | `pack/unpack_camera_frame`, `payload_lengths(header)`, `frame_name(header)` |
| codec | `comm/codec.py` | `encode_camera_frame(frame, seq)` (depth→uint16 mm), `decode_camera_frame(buf)` |
| transport | `comm/frame_channel.py` | `FrameChannelServer` (per-name mailbox) / `FrameChannelClient`; `recv_frame(conn)` |
| Isaac body | `simulation/isaacsim/oli.py` · `Oli` | `Oli(cameras=True, camera_resolution=(w,h))`; `camera_names`; `read_camera_rgbd(name)`; `camera_intrinsics(name)` |

Body protocol `CameraPublisher` duck-types (engine-agnostic — Isaac Oli, MuJoCo, or a fake):
`body.camera_names`, `body.read_camera_rgbd(name) -> (rgb, depth)`, `body.camera_intrinsics(name)`.

## CLI

```bash
# World with cameras (headless), self-terminating:
python humanoid/logic/simulation/isaacsim/sim_world_main.py --cameras --headless --duration 30 \
    [--camera-socket /tmp/oli-world-frames.sock] [--camera-res 1280 720]
# brain to pace the lockstep loop so it renders:
python humanoid/logic/oli/brain_main.py --mode stand --duration 28
# read + save frames (§8.3 smoke):
python humanoid/logic/oli/frame_smoke.py --n 5
```

## Behavior / invariants

- **Latest-wins, per camera name, both ends.** A single-slot mailbox clobbers with 2 cameras;
  the server keeps a `{name: bytes}` mailbox (peeks the wire name) and the reader demuxes into
  per-stream slots. Two cameras never overwrite each other.
- **Never blocks the World.** `publish()` is an O(1) mailbox drop; a background thread does the
  blocking `sendall`. An absent/slow consumer drops frames, never stalls control (design D6/D7).
- **Render sub-tick.** The World publishes only on `world.step(render=True)` ticks (the
  `--render-every` gate) — camera cadence decoupled from the 1 kHz control step.
- **`read()` is non-consuming**: a display feed re-reads the latest frame until a newer one
  replaces it; returns `None` only until the first frame for that stream arrives.

## Failure modes

| symptom | cause | handling |
|---|---|---|
| World dies right after first render, no traceback | Isaac camera annotator empty on the first render tick → read throws → `app.close()` hard-exits | `CameraPublisher` guards each read (warn-once, skip); World does an 8-tick warmup before serving. See memory `isaac-camera-first-render-not-ready`. |
| consumer `read(name)` always `None` | World not launched with `--cameras`, wrong socket path, or nothing pacing the lockstep loop (no renders) | run a brain (`--mode stand`) so the World steps + renders |
| only one stream arrives | consumer used raw `FrameChannelClient.read_latest()` (single-slot) | use `CameraStreamReader` (demuxes by name) |
| second consumer can't connect | `FrameChannelServer` accepts ONE client | one reader per frame socket |

## Tests

`tests/oli/comm/test_camera_stream.py` (reader + server↔reader loopback),
`tests/oli/world/test_camera_publisher.py` (publisher, throttle, not-ready survival),
`tests/oli/comm/test_frame_codec.py`, `tests/oli/comm/test_frame_channel.py`. Isaac end-to-end:
`logic/oli/frame_smoke.py` + `logic/simulation/isaacsim/camera_smoke.py` (pose/FOV cross-check).
