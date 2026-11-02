"""python -m humanoid.logic.oli.service — the hand tool for the brain service seam.

Smoke/debug surface for W4/W5 (the locbench evaluator uses the classes directly):

    p -m humanoid.logic.oli.service send 3.0 2.0 [--yaw 1.57]   # goal → the brain
    p -m humanoid.logic.oli.service clear                       # drop the goal
    p -m humanoid.logic.oli.service watch                       # tail telemetry (Ctrl-C stops)

Run from `humanoid/` with `p` (repo convention) or any CWD with the repo root on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the `humanoid` namespace package importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.service.goal_channel import (  # noqa: E402
    DEFAULT_GOAL_SOCKET,
    GoalChannelClient,
)
from humanoid.logic.oli.service.telemetry import (  # noqa: E402
    DEFAULT_TELEMETRY_SOCKET,
    TelemetryClient,
)


def _fmt(v, spec=".2f"):
    return "—" if v is None else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser(prog="oli.service")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_send = sub.add_parser("send", help="send a GoalCoordinate to the brain")
    p_send.add_argument("x", type=float)
    p_send.add_argument("y", type=float)
    p_send.add_argument("--yaw", type=float, default=None)
    p_send.add_argument("--goal-socket", default=DEFAULT_GOAL_SOCKET)

    p_clear = sub.add_parser("clear", help="clear the brain's goal")
    p_clear.add_argument("--goal-socket", default=DEFAULT_GOAL_SOCKET)

    p_watch = sub.add_parser("watch", help="tail the telemetry stream (Ctrl-C stops)")
    p_watch.add_argument("--telemetry-socket", default=DEFAULT_TELEMETRY_SOCKET)
    p_watch.add_argument("--hz", type=float, default=5.0, help="print rate")

    args = ap.parse_args()

    if args.cmd in ("send", "clear"):
        client = GoalChannelClient(args.goal_socket)
        try:
            if args.cmd == "send":
                client.send_goal(args.x, args.y, args.yaw)
                print(f"goal → ({args.x}, {args.y}, yaw={args.yaw})")
            else:
                client.clear_goal()
                print("goal cleared")
        except OSError as exc:
            sys.exit(f"no brain listening on {args.goal_socket} ({exc}) — "
                     "is `--service` up?")
        finally:
            client.close()
        return

    # watch
    client = TelemetryClient(args.telemetry_socket)
    print(f"watching {args.telemetry_socket} (Ctrl-C stops)...", flush=True)
    try:
        while True:
            snap = client.latest()
            if snap is not None:
                pose = (f"({_fmt(snap.pose[0])}, {_fmt(snap.pose[1])}, "
                        f"{_fmt(snap.pose[2])})" if snap.pose else "—")
                goal = (f"({snap.goal.x:.2f}, {snap.goal.y:.2f})" if snap.goal else "—")
                est = (f"{snap.est.status.name}" if snap.est else "—")
                n_wp = len(snap.path) if snap.path else 0
                intent = (f"vx {snap.intent[0]:+.2f} wz {snap.intent[2]:+.2f}"
                          if snap.intent else "—")
                print(f"[{snap.stamp_ns/1e9:9.3f}s] pose {pose}  goal {goal}  "
                      f"path {n_wp:3d} wp  intent {intent}  est {est}  "
                      f"loop {_fmt(snap.loop_hz, '.0f')} Hz", flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
