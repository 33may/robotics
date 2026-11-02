# Porthole Bus Protocol v1

**Status:** stable contract. Breaking changes require a version bump and a migration note.

This document is the integration contract between a Python backend and a Porthole UI.
Anything not written here is not guaranteed. If you need something this document does
not describe, change this document first.

---

## 1. Transport

A single WebSocket endpoint.

```
ws://127.0.0.1:8765
```

- The Python process is the **server**. It is the only long-lived process.
- The UI is a **client**. Zero or more clients may connect at any time
  (a native pywebview window and a browser tab on another machine can coexist).
- Bind address defaults to `127.0.0.1`. Binding to `0.0.0.0` is opt-in and exists
  only for demos over a trusted network (e.g. Tailscale).
- No authentication. The bus assumes a trusted host. See §8.

Two frame types are used, and they are never mixed up:

| Direction | Frame type | Encoding |
|---|---|---|
| Python → UI | **binary** | msgpack |
| UI → Python | **text** | JSON |

A client that receives a text frame from the server, or a server that receives a
binary frame from a client, must treat it as a protocol error and log it.

---

## 2. Python → UI: the envelope

Every server→client message is one msgpack-encoded map:

```jsonc
{
  "topic":   "camera/wrist",   // str, required — see §4
  "seq":     1043,             // uint, required — per-topic, monotonic from 0
  "ts":      1755600000.123,   // float, required — seconds since epoch, publisher clock
  "payload":  <any>            // required — see §3
}
```

Field rules:

- `seq` counts **per topic**, not globally. A gap in `seq` means the UI missed a
  message: either it connected late, or a lossy topic dropped frames (§6).
  Panels may use gaps to show a "dropped" indicator; they must not treat a gap as
  an error.
- `ts` is the publisher's wall clock at the moment the data was produced — not the
  moment it was sent. For sensor data, stamp it when the sensor produced it.
- `payload` may be any msgpack-representable value, including the binary forms in §3.

### Reserved topics

Topics beginning with `$/` are reserved for the protocol itself.

**`$/hello`** — sent by the server to each client immediately on connect, before
any other message:

```jsonc
{
  "topic": "$/hello",
  "seq": 0,
  "ts": 1755600000.0,
  "payload": {
    "protocol": 1,                       // this document's version
    "app": "inspection",                 // app identifier, free-form
    "server": "porthole 0.1.0",
    "topics": [                          // declared topics, may grow later
      {"topic": "camera/wrist", "qos": "stream", "kind": "image/jpeg"},
      {"topic": "log/events",   "qos": "event",  "kind": "json"}
    ]
  }
}
```

`topics` is advisory. It lets a UI build generic panels and lets a developer see
what a backend offers without reading its source. A backend may publish to topics
it never declared; the UI must tolerate that.

**`$/bye`** — optional, best-effort, sent before a clean server shutdown so the UI
can show "backend exited" rather than "connection lost".

---

## 3. Payload encoding

`payload` is ordinary msgpack. Two **tagged maps** carry binary data and may appear
at any depth inside a payload, including nested in lists and maps.

### 3.1 Numeric arrays — `$nd`

```jsonc
{
  "$nd": {
    "dtype": "float32",     // see table below
    "shape": [12043, 3],    // row-major
    "data":  <msgpack bin>  // raw little-endian buffer, C-contiguous
  }
}
```

Supported `dtype` values: `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`,
`int32`, `uint32`, `int64`, `uint64`.

- Always **little-endian**. Always **C-contiguous**. The publisher is responsible
  for converting; the UI does not handle Fortran order or big-endian.
- Decoders produce the matching JS `TypedArray`. `shape` is carried alongside; the
  UI does not reshape into nested arrays.
- `int64`/`uint64` decode to `BigInt64Array`/`BigUint64Array`. Avoid them for data
  that feeds three.js or uPlot, which cannot consume BigInt.

**Prefer `float32` for geometry.** A 12k-point cloud is 144 KB as float32 and
288 KB as float64, at a precision no display can resolve.

### 3.2 Images — `$img`

```jsonc
{
  "$img": {
    "mime": "image/jpeg",   // "image/jpeg" | "image/png"
    "w": 848,               // pixels, informational — decoders may ignore
    "h": 480,
    "data": <msgpack bin>   // encoded image bytes, not raw pixels
  }
}
```

Raw pixel buffers must **not** be sent as images. Encode to JPEG on the Python
side. An 848×480 RGB frame is 1.2 MB raw and ~40 KB as JPEG q80; at 15 fps that
is the difference between 18 MB/s and 600 KB/s over the socket.

### 3.3 Everything else

Plain msgpack: maps, lists, strings, numbers, booleans, `null`. Use ordinary
JSON-shaped structures for state, logs, and metadata.

### 3.4 Composite payloads

Tagged maps compose freely. A scene update may carry several arrays at once:

```jsonc
{
  "frame": "base",
  "points": {"$nd": {"dtype": "float32", "shape": [12043, 3], "data": "<bin>"}},
  "colors": {"$nd": {"dtype": "uint8",   "shape": [12043, 3], "data": "<bin>"}},
  "n_dropped": 0
}
```

---

## 4. Topic naming

Topics are `/`-separated lowercase paths. First segment is the **domain**, the rest
is free.

```
camera/wrist            camera/wrist/depth
scene/cloud             scene/robot          scene/viewsphere
state/loop              state/planner
log/events
decision/request
```

Rules:

- Lowercase, `[a-z0-9_]` per segment, `/` as separator. No leading or trailing `/`.
- A topic carries **one kind of thing**. Do not multiplex two payload shapes onto
  one topic and discriminate with a `type` field — split the topic instead. Panels
  subscribe to topics, and a panel that must branch on payload shape is a panel
  that should have been two subscriptions.
- Topic names are part of your app's contract. Renaming one breaks saved layouts.

---

## 5. Retained state (late joiners)

The server **retains recent messages per topic** and replays them to each newly
connected client, immediately after `$/hello`.

This is what makes a browser tab opened ten minutes into a run show the current
scene instead of an empty canvas. It is not optional, and it is why a panel can be
written as a pure function of the messages it has received.

How much is retained depends on QoS (§6):

| QoS | Retained | Why |
|---|---|---|
| `stream` | the newest message only | an old camera frame is worthless; a stale one is misleading |
| `event` | a bounded backlog, default **256** | a log panel must open with history, not one orphaned line |

- Replayed messages keep their **original** `seq` and `ts`. A client can therefore
  tell that a retained frame is stale by comparing `ts` to now.
- Backlogs are per topic and bounded. Retention is a convenience for late joiners,
  not a database — if you need durable history, write it to disk.
- A topic may opt out with `retain=False` when replaying it would be wrong — for
  example a one-shot notification.

Consequence for backends: **publish state, not deltas.** A topic whose payload only
makes sense applied on top of previous messages cannot be retained meaningfully, and
a late client will be permanently wrong. If you need deltas for bandwidth, publish a
full snapshot periodically on a separate topic.

`event` topics are the exception to "publish state, not deltas": they are
append-only streams, which is exactly why they get a backlog instead of a single
retained message. A client that connects late has still missed anything older than
the backlog, so a backend that needs durable history must write it to disk itself.

---

## 6. Quality of service

Each topic has one of two QoS classes, declared by the publisher.

| QoS | Behaviour when a client is slow | Use for |
|---|---|---|
| `stream` | **Drop oldest.** Only the newest undelivered message per topic is kept per client. | camera frames, point clouds, joint states, anything sampled |
| `event` | **Buffer and deliver in order.** Never dropped. | log lines, decision requests, state transitions, anything a human must not miss |

`stream` is the default and the right choice for most data. The rule of thumb: if
receiving message N makes message N−1 worthless, it is a `stream`.

If an `event` client's buffer exceeds its limit (default 4096 messages), the server
**disconnects that client** rather than dropping an event silently or growing without
bound. A disconnect is visible; a silently dropped decision request is not.

Backpressure is per-client. A stalled browser tab on a slow link must not slow down
the native window on localhost.

---

## 7. UI → Python: commands

Client→server frames are **text**, containing one JSON object per frame.

```jsonc
{
  "cmd": "decision/answer",   // str, required — command name
  "id":  "c-17",              // str, optional — echoed back by the app if it replies
  "ts":  1755600000.9,        // float, optional — client clock
  ...                         // any additional JSON fields, command-specific
}
```

- `cmd` names follow the same path convention as topics but are a **separate
  namespace**. `decision/answer` as a command is unrelated to `decision/request`
  as a topic, beyond the app's own convention of pairing them.
- Commands land on a queue the backend drains. The bus itself never interprets a
  command, never validates its fields, and never acts on one.
- Delivery is at-most-once and unacknowledged at the protocol level. If your app
  needs a reply, publish one on a topic and correlate with `id`.

### The frontend is IO only

This is the load-bearing rule of the whole system, so it is stated here rather than
implied:

> **A command is a statement of intent by a human, not an instruction to hardware.**
> All authority, validation, sequencing and safety live in the Python backend. The
> UI may send any command at any time, including nonsense, including twice, including
> while the robot is moving. A backend that would be unsafe if that happened is
> incorrect, and no amount of UI discipline fixes it.

The bus deliberately provides no permission model, no command classes, and no
confirm round-trip. Adding one would create the illusion that the UI is part of the
safety story. It is not.

---

## 8. Security posture

There is none, by design, and this is only safe because of the deployment shape:

- Default bind is loopback. Nothing outside the machine can connect.
- Binding to a routable address is an explicit, per-run opt-in for demos.
- Any client that connects can send any command. See §7 — that is acceptable
  precisely because commands are inert without a backend that chooses to act.

Do not expose a Porthole bus to an untrusted network and do not put one behind a
public reverse proxy.

---

## 9. Reference: a minimal exchange

```
client                                   server
  |------------- ws connect ------------->|
  |<-- bin  $/hello           seq=0 -------|
  |<-- bin  scene/cloud       seq=87 ------|   retained, ts is 4s old
  |<-- bin  camera/wrist      seq=1201 ----|   retained
  |<-- bin  camera/wrist      seq=1202 ----|   live from here on
  |<-- bin  camera/wrist      seq=1203 ----|
  |--- txt  {"cmd":"view/select","cell":[3,1]} -->|
  |<-- bin  scene/viewsphere  seq=12 ------|
  |<-- bin  log/events        seq=44 ------|
```

---

## 10. Conformance checklist

A backend conforms if:

- [ ] It sends `$/hello` first, on every connection.
- [ ] `seq` is per-topic and monotonic; `ts` is the production time.
- [ ] All arrays are C-contiguous little-endian with a declared `dtype`.
- [ ] Images are encoded (JPEG/PNG), never raw pixel buffers.
- [ ] Every retained topic carries complete state, not a delta.
- [ ] `event` topics are used for anything a human must not miss.
- [ ] It is safe against a hostile client sending arbitrary commands.

A UI conforms if:

- [ ] It renders correctly from retained messages alone, with no live traffic.
- [ ] It tolerates `seq` gaps, unknown topics, and unknown payload fields.
- [ ] It never assumes a topic exists before its first message arrives.
- [ ] It uses no platform API that a browser does not have.
