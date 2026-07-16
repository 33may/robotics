"""recorder_main.py — the standalone Robot-side recording process (MAY-173 1.4).

Attaches a SINK to the already-flowing World channels: drains the camera frame
channel (`CameraStreamReader`) + the GT debug-pose channel (`DebugPoseClient`)
into a bake-grade coverage dump via `RecordingSession`. Spawned/monitored by the
dev_app RecordPanel (ProcessLauncher), but deliberately its OWN process:

  crash-safety   dev_app dying cannot touch the recording — this process keeps
                 draining; every row/frame is on disk the moment it's written
  graceful stop  SIGTERM/SIGINT → drain the encoder queue → finalize → exit 0
  world death    no fresh frame for --idle-timeout s → finalize + exit 0
                 ("the World hung up" is a SAVED recording, not a crash)

Status: `<out>/status.json` rewritten atomically ~1 Hz — state + session stats —
so the panel (or a human with `cat`) can monitor without any socket coupling.

    conda run -n brain python -m humanoid.logic.oli.recording.recorder_main \
        --out data/coverage_drives/teleop_zone_v1 [--codec jpeg] \
        [--camera-socket /tmp/oli-world-frames.sock] [--pose-socket /tmp/oli-debug-pose.sock]

Brain env; imports comm + recording only — no isaacsim/limxsdk.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.comm.camera_stream import CameraStreamReader  # noqa: E402
from humanoid.logic.oli.comm.debug_pose import DebugPoseClient  # noqa: E402
from humanoid.logic.oli.recording.session import RecordingSession  # noqa: E402

_STATUS_PERIOD_S = 1.0


def _write_status(path: Path, state: str, stats: dict) -> None:
    """Atomic rewrite (tmp + rename) — a reader never sees a torn file."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"state": state, **stats}, indent=2))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="dump directory (created)")
    ap.add_argument("--codec", choices=("png", "jpeg"), default="jpeg",
                    help="RGB codec — jpeg q95 is cuVSLAM-proven (cell test 16-07); "
                         "depth is always lossless PNG")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--camera-socket", default="/tmp/oli-world-frames.sock")
    ap.add_argument("--pose-socket", default="/tmp/oli-record-pose.sock",
                    help="debug-pose channel to BIND (the World publishes GT here; the "
                         "launcher's --stereo-cameras adds this path automatically)")
    ap.add_argument("--idle-timeout", type=float, default=10.0,
                    help="finalize after N s without a fresh frame (0 = wait forever)")
    ap.add_argument("--connect-timeout", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        print(f"[recorder] REFUSING to record into non-empty {args.out}", flush=True)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    status_path = args.out / "status.json"
    _write_status(status_path, "connecting", {})

    poses = DebugPoseClient(args.pose_socket)
    reader = CameraStreamReader(socket_path=args.camera_socket)
    try:
        reader.connect(timeout=args.connect_timeout)
    except Exception as exc:
        print(f"[recorder] camera channel connect failed: {exc}", flush=True)
        _write_status(status_path, "failed-connect", {"error": str(exc)})
        poses.close()
        return 1
    print(f"[recorder] connected: cameras={args.camera_socket} poses={args.pose_socket} "
          f"→ {args.out} (codec {args.codec})", flush=True)

    stop = {"flag": False, "why": ""}

    def _sig(signum, _frame):
        stop["flag"], stop["why"] = True, signal.Signals(signum).name
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    session = None            # created on the FIRST frame (res from its intrinsics)
    last_pose_stamp = -1
    last_frame_wall = time.monotonic()
    last_status_wall = 0.0
    t0 = time.monotonic()

    try:
        while not stop["flag"]:
            p = poses.latest()
            if p is not None and p[0] != last_pose_stamp:
                last_pose_stamp = p[0]
                if session is not None:
                    session.feed_base(p[0], x=p[1], y=p[2], yaw=p[3])
            got = False
            for name in reader.stream_names() or []:
                frame = reader.read(name)
                if frame is None:
                    continue
                if session is None:
                    res = (frame.intrinsics.width, frame.intrinsics.height)
                    session = RecordingSession(
                        args.out, camera_res=res,
                        streams=list(reader.stream_names() or [name]),
                        codec=args.codec, quality=args.quality, workers=args.workers)
                    if p is not None:  # seed the pose buffer before the first frame
                        session.feed_base(p[0], x=p[1], y=p[2], yaw=p[3])
                    print(f"[recorder] recording started: {res[0]}x{res[1]}, "
                          f"streams {reader.stream_names()}", flush=True)
                if session.feed_frame(frame):
                    got = True
            now = time.monotonic()
            if got:
                last_frame_wall = now
            elif (args.idle_timeout and session is not None
                    and now - last_frame_wall > args.idle_timeout):
                stop["flag"], stop["why"] = True, "idle-timeout (World gone?)"
            if now - last_status_wall >= _STATUS_PERIOD_S:
                last_status_wall = now
                st = session.stats() if session is not None else {}
                st["elapsed_s"] = round(now - t0, 1)
                _write_status(status_path, "recording" if session else "waiting-frames", st)
            time.sleep(0.002)
    finally:
        why = stop["why"] or "loop-exit"
        print(f"[recorder] stopping ({why}) — draining writer…", flush=True)
        final = session.stop() if session is not None else {}
        _write_status(status_path, "saved", final)
        reader.close()
        poses.close()
        print(f"[recorder] SAVED {final.get('frames', 0)} frames "
              f"({final.get('stamps', 0)} stamps, gaps {final.get('gaps', 0)}) "
              f"→ {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
