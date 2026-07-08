# Connecting real cameras to the dev app — handoff for MAY-149

The dev app's **Camera panel renders any `CameraSource`**. Today it uses
`SyntheticCameraSource` (animated fake RGBD). To show the real Isaac (and later real-robot)
streams, implement a `CameraSource` backed by your SOCK_STREAM frame channel and **swap one
line** — the panel, shell, and tests do not change.

You own the whole data path up to this protocol boundary: Isaac cameras → `CameraFrame` →
SOCK_STREAM channel → a brain-side reader → **your `CameraSource`**. We own everything above it.

---

## 1. The interface you implement

From `logic/oli/devapp/sources/camera_source.py`:

```python
class CameraSource(Protocol):
    def stream_names(self) -> list[str]: ...              # e.g. ["chest", "head"]
    def read(self, name: str) -> Optional[CameraFrame]: ...  # NON-BLOCKING, latest-wins
    def close(self) -> None: ...
```

## 2. The frame you return

```python
@dataclass(frozen=True)
class CameraFrame:
    stamp_ns: int              # sim-time ns — SAME clock as Observation.stamp_ns
    name: str                  # "chest" | "head"
    rgb: np.ndarray            # (H, W, 3) uint8, channel order RGB, C-contiguous
    depth: np.ndarray          # (H, W) float32, METRES (planar Z); non-finite = invalid
    intrinsics: CameraIntrinsics | None

@dataclass(frozen=True)
class CameraIntrinsics:
    width: int; height: int
    fx: float; fy: float; cx: float; cy: float
```

## 3. Requirements (non-negotiable — this is where it breaks if ignored)

1. **`read()` MUST be non-blocking + latest-wins.** It runs on the **UI thread every frame
   (~60 Hz)**. Never block on the socket. Buffer incoming frames on YOUR reader thread and
   return the newest buffered frame instantly, or `None` if none has arrived. Your SOCK_STREAM
   design is already latest-wins — just expose it this way.
2. **Exact dtypes / conventions:** `rgb` = `(H,W,3)` uint8 **RGB** (not BGR); `depth` =
   `(H,W)` float32 **metres** (your `distance_to_image_plane` planar Z is exactly right);
   both `np.ascontiguousarray`. The panel colourises depth with a jet map over `[near,far]`
   and shows the intrinsics line, so metres + correct channel order matter.
3. **`stream_names()`** returns the streams you serve; the panel draws one RGB+depth row per
   name. Keep names stable ("chest", "head").
4. **`close()`** tears down your socket + reader thread.

## 4. One `CameraFrame`, please (reconcile the contract)

You are landing `CameraFrame` as the 4th invariant contract in `logic/oli/contracts.py`. Our
`sources/camera_source.py` currently holds a **local stand-in** with the fields above. **Make
your `contracts.CameraFrame` match this spec exactly** — field names `stamp_ns / name / rgb /
depth / intrinsics`, dtypes and conventions as in §2 — and we will replace our stand-in with
`from ...contracts import CameraFrame` so there is ONE definition. If you must diverge (e.g.
`stamp` vs `stamp_ns`, a 3×3 K matrix instead of fx/fy/cx/cy, mm instead of m), that's fine —
just tell us and we absorb it in the adapter rather than in the panel. Matching is cleaner.

## 5. Wiring — one line

Drop your class in `sources/isaac_camera_source.py`, then in
`logic/oli/devapp/__main__.py::build_registry`:

```python
- reg.register(CameraPanel(SyntheticCameraSource()))
+ reg.register(CameraPanel(IsaacCameraSource(frame_socket=args.frame_socket)))
```

Add a `--frame-socket` arg if your frame channel uses a path separate from the control socket
(`--socket`). Nothing else in the app moves.

## 6. Verifying it

- **Unit:** test your source in isolation — `read()` returns a `CameraFrame` with the right
  shapes/dtypes and returns fast even when no frame has arrived (non-blocking). Pure, no GUI.
- **Visual:** the app is validated headless via `capture.py` + Xvfb — boot the app pointed at a
  live World and screenshot it (see `devapp` agent-memory `devapp_build_and_validation`):
  ```
  xvfb-run -a -s "-screen 0 1600x1000x24" \
    /home/may33/miniconda3/envs/brain/bin/python -m humanoid.logic.oli.devapp \
    --socket /tmp/oli-world.sock --frame-socket <yours> --screenshot /tmp/cams.png
  ```

Questions on the boundary → ping the dev-app side. The synthetic source in
`synthetic_camera_source.py` is a working reference implementation of this exact protocol.
