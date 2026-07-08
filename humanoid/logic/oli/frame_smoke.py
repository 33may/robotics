"""frame_smoke.py — brain-side camera-frame smoke consumer (§8.3, MAY-149).

Connects ONLY the World's camera frame channel (the SOCK_STREAM, separate from the
control socket) via `CameraStreamReader`, reads N frames per stream, and saves RGB +
depth to disk. Proves RGBD crosses the process boundary from a REAL Isaac World — the
control loop is paced by a separate brain (`brain_main.py --mode stand`).

    # terminal 1 (World):  cameras on, headless, self-terminating
    conda run -n isaac python humanoid/logic/simulation/isaacsim/sim_world_main.py \
        --cameras --headless --pin-root --duration 30
    # terminal 2 (brain):  stand policy paces the lockstep loop → renders → frames
    conda run -n brain python humanoid/logic/oli/brain_main.py --mode stand --duration 28
    # terminal 3 (this):   read + save the frames
    conda run -n brain python humanoid/logic/oli/frame_smoke.py --n 5

Runs in the `brain` env (CameraStreamReader is pure stdlib + numpy — no isaacsim).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.camera_mounts import CAMERAS  # noqa: E402
from humanoid.logic.oli.comm.camera_stream import CameraStreamReader  # noqa: E402


def _save_png(path: Path, arr: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(arr).save(str(path))
    except Exception:  # pragma: no cover - PIL absent
        np.save(str(path.with_suffix(".npy")), arr)


def _save_depth_png(path: Path, depth: np.ndarray) -> None:
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d) & (d > 0)
    norm = np.zeros_like(d)
    if finite.any():
        lo, hi = np.percentile(d[finite], [5, 95])
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    norm[~finite] = 0.0
    _save_png(path, (norm * 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-socket", default="/tmp/oli-world-frames.sock")
    ap.add_argument("--out", type=Path, default=Path("/tmp/oli_frame_smoke"))
    ap.add_argument("--n", type=int, default=5, help="distinct frames to save per stream")
    ap.add_argument("--timeout", type=float, default=40.0,
                    help="wall seconds to wait for all streams to deliver --n frames")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    expected = [m.name for m in CAMERAS]
    print(f"[frame-smoke] connecting {args.frame_socket}; expecting streams {expected}",
          flush=True)
    reader = CameraStreamReader(socket_path=args.frame_socket)
    reader.connect(timeout=args.timeout)

    # Per stream: collect the first --n frames with DISTINCT stamps (latest-wins means we
    # re-read the same frame until a newer one arrives, so dedup by stamp).
    saved: dict[str, int] = {n: 0 for n in expected}
    last_stamp: dict[str, int] = {n: -1 for n in expected}
    deadline = time.monotonic() + args.timeout
    ok = True
    try:
        while time.monotonic() < deadline and any(saved[n] < args.n for n in expected):
            for name in expected:
                if saved[name] >= args.n:
                    continue
                frame = reader.read(name)
                if frame is None or frame.stamp_ns == last_stamp[name]:
                    continue
                last_stamp[name] = frame.stamp_ns
                idx = saved[name]
                _save_png(args.out / f"{name}_{idx}_rgb.png", frame.rgb)
                _save_depth_png(args.out / f"{name}_{idx}_depth.png", frame.depth)
                if idx == 0:
                    np.save(args.out / f"{name}_depth.npy", frame.depth)
                d = frame.depth
                finite = np.isfinite(d) & (d > 0)
                cy, cx = d.shape[0] // 2, d.shape[1] // 2
                center = float(d[cy, cx]) if finite[cy, cx] else float("nan")
                intr = frame.intrinsics
                print(f"[frame-smoke] {name:5s} #{idx} stamp={frame.stamp_ns} "
                      f"rgb={frame.rgb.shape} depth_finite={finite.mean():.2f} "
                      f"center={center:.3f}m {intr.width}×{intr.height}", flush=True)
                saved[name] += 1
            time.sleep(0.005)
    finally:
        reader.close()

    for name in expected:
        got = saved[name]
        ok = ok and got > 0
        print(f"[frame-smoke] {name}: {got}/{args.n} frames saved", flush=True)
    print(f"[frame-smoke] VERDICT: {'PASS' if ok else 'FAIL'} — frames in {args.out}",
          flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
