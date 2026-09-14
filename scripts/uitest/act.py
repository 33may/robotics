#!/usr/bin/env python
"""Send REAL input events at pixel coordinates.

    python act.py click 120 400
    python act.py click 120 400 --button right
    python act.py dblclick 120 400
    python act.py move 800 500
    python act.py drag 1400 883 400 883            # x1 y1 x2 y2
    python act.py drag 1400 883 400 883 --steps 40
    python act.py wheel 800 500 --dy -240
    python act.py key ArrowRight
    python act.py text "hello"

Coordinates are in SCREENSHOT pixels == CSS pixels (launch.py pins
deviceScaleFactor=1, so the two are identical - see README "Coordinates").

Events are delivered with CDP Input.dispatchMouseEvent, i.e. they are *trusted*
browser input, hit-tested exactly like a human's mouse.  This is what makes a
WASM/<canvas> app (egui, Flutter, Unity) driveable at all - there is no DOM to
address.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cdp as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def xy(p):
        p.add_argument("x", type=float)
        p.add_argument("y", type=float)
        return p

    c = xy(sub.add_parser("click"))
    c.add_argument("--button", default="left", choices=["left", "right", "middle"])
    c.add_argument("--settle", type=float, default=0.25)

    d = xy(sub.add_parser("dblclick"))
    d.add_argument("--settle", type=float, default=0.25)

    xy(sub.add_parser("move"))

    g = sub.add_parser("drag")
    g.add_argument("x1", type=float); g.add_argument("y1", type=float)
    g.add_argument("x2", type=float); g.add_argument("y2", type=float)
    g.add_argument("--steps", type=int, default=24)
    g.add_argument("--settle", type=float, default=0.35)

    w = xy(sub.add_parser("wheel"))
    w.add_argument("--dx", type=float, default=0.0)
    w.add_argument("--dy", type=float, default=-120.0)

    k = sub.add_parser("key")
    k.add_argument("key")
    k.add_argument("--repeat", type=int, default=1)

    t = sub.add_parser("text")
    t.add_argument("value")

    sub.add_parser("reload", help="reload the page to get back to a known state")

    args = ap.parse_args()
    sess = C.Session.load()

    with C.CDP(sess.ws_url) as cd:
        if args.cmd == "click":
            cd.click(args.x, args.y, args.button, 1, args.settle)
        elif args.cmd == "dblclick":
            cd.click(args.x, args.y, "left", 2, args.settle)
        elif args.cmd == "move":
            cd.move(args.x, args.y)
        elif args.cmd == "drag":
            cd.drag(args.x1, args.y1, args.x2, args.y2, args.steps, args.settle)
        elif args.cmd == "wheel":
            cd.wheel(args.x, args.y, args.dx, args.dy)
        elif args.cmd == "key":
            for _ in range(args.repeat):
                cd.key(args.key)
        elif args.cmd == "text":
            cd.text(args.value)
        elif args.cmd == "reload":
            from launch import wait_ready
            cd.send("Page.reload", {"ignoreCache": False})
            wait_ready(cd)

    print(json.dumps({"ok": True, "action": vars(args)}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
