#!/usr/bin/env python
"""Launch the app-under-test (Rerun web viewer) + an isolated Chrome, and wait
until it is actually ready to be poked at.

    python launch.py /tmp/rrd/_toy.rrd
    python launch.py /tmp/rrd/_toy.rrd --width 1600 --height 1000 --headful

Prints a JSON blob with the viewer URL, the PIDs, and the CDP websocket URL.
Everything it starts is appended to $UITEST_DIR/pids so cleanup.py can reap it.

Readiness is three-stage, and all three matter:
  1. the HTTP port accepts connections            (server up)
  2. the page reports a sized <canvas>            (WASM booted)
  3. two screenshots 0.6 s apart stop differing   (data streamed + first paint)
Stopping at stage 1 gives you screenshots of a blank/loading viewer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cdp as C  # noqa: E402
from imgdiff import diff as img_diff  # noqa: E402

RERUN_BIN = os.environ.get(
    "UITEST_RERUN", "/home/anton/miniconda3/envs/robo/bin/rerun"
)


def start_rerun(rrd: Path, web_port: int, grpc_port: int) -> tuple[int, str]:
    C.ART.mkdir(parents=True, exist_ok=True)
    log = (C.ART / "rerun.log").open("ab")
    args = [
        RERUN_BIN,
        str(rrd),
        # --serve-web, NOT --web-viewer: the latter also launches the user's
        # *default desktop browser* onto the viewer URL, hijacking their real
        # session.  --serve-web only hosts.
        "--serve-web",
        "--web-viewer-port", str(web_port),
        "--port", str(grpc_port),
        "--hide-welcome-screen",
    ]
    env = dict(os.environ, BROWSER="/bin/true")  # belt & braces against auto-open
    p = subprocess.Popen(args, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                         env=env, start_new_session=True)
    C.track_pid(p.pid, "rerun")
    if not C.wait_port(web_port, 40):
        raise RuntimeError(f"rerun web port {web_port} never opened; see {C.ART/'rerun.log'}")
    url = (f"http://127.0.0.1:{web_port}/?url="
           f"rerun%2Bhttp%3A%2F%2F127.0.0.1%3A{grpc_port}%2Fproxy")
    return p.pid, url


def wait_ready(c: "C.CDP", timeout: float = 90.0) -> dict:
    """Stage 2 + 3 of readiness. Returns a small report."""
    end = time.time() + timeout
    canvas = None
    while time.time() < end:
        try:
            canvas = c.eval(
                "(() => {const e=document.querySelector('canvas');"
                "return e ? JSON.stringify({w:e.width,h:e.height}) : null;})()"
            )
        except Exception:
            canvas = None
        if canvas:
            break
        time.sleep(0.4)
    if not canvas:
        raise RuntimeError("no <canvas> appeared - WASM viewer failed to boot")

    a = C.ART / "_ready_a.png"
    b = C.ART / "_ready_b.png"
    stable_since = None
    while time.time() < end:
        c.screenshot(a)
        time.sleep(0.6)
        c.screenshot(b)
        r = img_diff(a, b)
        if r["changed_frac"] < 0.002:
            if stable_since is None:
                stable_since = time.time()
            if time.time() - stable_since > 0.3:
                # re-query: at first paint the canvas still reports its 300x150
                # HTML default, so the early value is meaningless.
                canvas = c.eval(
                    "(() => {const e=document.querySelector('canvas');"
                    "return e ? JSON.stringify({w:e.width,h:e.height}) : null;})()"
                )
                return {"canvas": json.loads(canvas), "settle_frac": r["changed_frac"]}
        else:
            stable_since = None
    raise RuntimeError("viewer never settled (still animating after timeout)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rrd", type=Path, nargs="?", help=".rrd to serve (omit to reuse a server)")
    ap.add_argument("--web-port", type=int, default=0)
    ap.add_argument("--grpc-port", type=int, default=0)
    ap.add_argument("--width", type=int, default=C.DEFAULT_W)
    ap.add_argument("--height", type=int, default=C.DEFAULT_H)
    ap.add_argument("--headful", action="store_true",
                    help="show a real Chrome window (default is headless=new)")
    ap.add_argument("--url", help="skip rerun; point the browser at this URL instead")
    args = ap.parse_args()

    C.ART.mkdir(parents=True, exist_ok=True)

    if args.url:
        url, rerun_pid = args.url, None
    else:
        if not args.rrd or not args.rrd.exists():
            ap.error("need an existing .rrd (or --url)")
        wp = args.web_port or C.free_port()
        gp = args.grpc_port or C.free_port()
        rerun_pid, url = start_rerun(args.rrd, wp, gp)

    sess = C.launch_chrome(args.width, args.height, headless=not args.headful)
    with C.CDP(sess.ws_url) as c:
        c.prepare(args.width, args.height)
        c.navigate(url)
        rep = wait_ready(c)
        shot = c.screenshot(C.ART / "launch_ready.png")

    out = {
        "url": url,
        "rerun_pid": rerun_pid,
        "cdp_port": sess.port,
        "ws_url": sess.ws_url,
        "width": args.width,
        "height": args.height,
        "canvas": rep["canvas"],
        "ready_screenshot": str(shot),
        "session_file": str(C.SESSION_FILE),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
