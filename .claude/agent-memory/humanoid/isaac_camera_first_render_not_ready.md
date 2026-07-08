---
name: isaac-camera-first-render-not-ready
description: Isaac camera get_rgba/depth returns an empty buffer on the FIRST render tick — reading it crashes the whole World (Isaac app.close hard-exits, no traceback). Fix = warmup + a guard.
metadata:
  type: reference
---

**Isaac `isaacsim.sensors.camera.Camera` annotators are not populated on the first render.**
Right after `cam.initialize()` + `cam.add_distance_to_image_plane_to_frame()`, the render product
needs a few `world.step(render=True)` ticks before `get_rgba()` / `get_current_frame()
["distance_to_image_plane"]` return valid buffers. On tick 0 `get_rgba()` yields an empty/unshaped
array, so `rgba[:, :, :3]` (or the depth reshape) throws.

**Why it bit hard (MAY-149):** the crash happened *inside the World loop* on the first camera publish.
An uncaught exception there → `finally: app.close()` runs and Isaac **hard-exits the process before
the Python traceback flushes** → the log shows only "Simulation App Shutting Down", no traceback, and
the brain just sees "World closed the connection". Debugged by correlating the death time with the
first render (the `omni:rtx:material:db:flattener` camera-population warnings mark it).

**Fix (both, belt-and-suspenders):**
1. **Warm the cameras** before serving: `for _ in range(8): world.step(render=True)` once cameras are
   attached (`camera_smoke.py` dodged this by reading only after 60 warmup steps).
2. **Guard the publisher**: `CameraPublisher.publish` wraps each camera's read+encode in try/except,
   warns once, and skips a not-ready camera — a camera hiccup must NEVER kill the World's loop.

General rule: any code that reads an Isaac render sensor inside the authoritative sim loop must
tolerate a not-ready frame; never let a sensor read propagate an exception into the loop.
Relates to [[oli-perception-camera-design]], [[isaac_oli_smoke_loader]].
