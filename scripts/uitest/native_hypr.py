#!/usr/bin/env python3
"""Launch a native app on Hyprland at a known size and screenshot it.

The other half of the UI debug loop. `launch.py`/`cdp.py` drive the app in
a browser because that is where real synthetic input is cheap; this drives
the NATIVE window, because that is what the user actually double-clicks.
Its job is the *verify* half only — observe, at a size that is stable
enough to compare two runs — not act. There is no input injection here on
purpose: this desktop has no ydotool/wtype/xdotool (see README) and the
native viewer is not the harness, it is the deliverable.

Why the window has to be floated. Hyprland tiles by default, so a viewer
launched next to a browser gets whatever slot is left — on this machine a
1600x1000 request came back as an 848x1335 portrait sliver and the
blueprint's horizontal split had nowhere to go. Floating + an explicit
`resizewindowpixel` is the only way two captures on different days are the
same picture.

    p scripts/uitest/native_hypr.py --rrd /tmp/rrd/2408-cup3.rrd \
        --out /tmp/rrd_ui_test/native_cup3.png

Leaves nothing running: the window is killed after the capture unless
`--keep` is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time

HYPRCTL = "hyprctl"


def _clients() -> list[dict]:
    return json.loads(subprocess.run([HYPRCTL, "clients", "-j"],
                                     capture_output=True, text=True,
                                     check=True).stdout)


def _dispatch(*args: str) -> None:
    subprocess.run([HYPRCTL, "dispatch", *args], capture_output=True,
                   text=True, check=False)


def launch(cmd: list[str], match: str, timeout: float = 60.0) -> dict:
    """Start `cmd`, return the hyprctl client dict of the window it opened.

    Matches on a window that was NOT there before, so a second viewer does
    not get confused for the first one.
    """
    before = {c["address"] for c in _clients()}
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        for c in _clients():
            if c["address"] not in before and (
                    match in c["class"].lower() or match in c["title"].lower()):
                c["_pid"] = proc.pid
                return c
        if proc.poll() is not None:
            raise SystemExit(f"{cmd[0]} exited before opening a window")
        time.sleep(0.4)
    raise SystemExit(f"no window matching {match!r} within {timeout}s")


def place(addr: str, x: int, y: int, w: int, h: int, tries: int = 6) -> dict:
    """Float the window and pin it to an exact rectangle. Verified, retried.

    Hyprland does not honour this on the first ask, and it silently undoes
    it later: a window placed at 2200x1300 came back as 1366x1068 the
    moment another client opened and the layout was recomputed. So the
    placement is asserted rather than requested, and the caller is
    expected to call this AGAIN right before capturing.
    """
    a = f"address:{addr}"
    for _ in range(tries):
        _dispatch("setfloating", a)
        _dispatch("resizewindowpixel", f"exact {w} {h},{a}")
        _dispatch("movewindowpixel", f"exact {x} {y},{a}")
        time.sleep(0.5)
        c = next((c for c in _clients() if c["address"] == addr), None)
        if c is None:
            raise SystemExit("window vanished while being placed")
        if c["floating"] and c["size"] == [w, h] and c["at"] == [x, y]:
            _dispatch("focuswindow", a)
            time.sleep(0.3)
            return c
    raise SystemExit(f"could not pin window to {w}x{h}+{x}+{y}; "
                     f"got {c['size']} at {c['at']} floating={c['floating']}")


def grim(client: dict, out: str) -> str:
    """Full-output grab, cropped to the window — and grabbed twice.

    Two traps, both learned the hard way on this desktop:

    `grim -g` on a region that was just resized comes back TORN — the left
    two thirds of a 1400x900 capture were the previous frame's black, the
    right third the new one. Grabbing the whole output and cropping in
    PIL hits a single composited frame instead of a region readback.

    And the first grab after a move is still stale even then, so the first
    one is thrown away. Cheap insurance; a screenshot assertion that fails
    intermittently is worse than no assertion.
    """
    from PIL import Image
    x, y = client["at"]
    w, h = client["size"]
    s = _scale()
    tmp = out + ".full.png"
    for _ in range(2):
        subprocess.run(["grim", tmp], check=True)
        time.sleep(0.7)
    im = Image.open(tmp)
    box = tuple(int(round(v * s)) for v in (x, y, x + w, y + h))
    im.crop(box).save(out)
    os.remove(tmp)
    return out


def _scale() -> float:
    """Physical pixels per logical point. On this desktop it is 1.25.

    The third trap of the day and the nastiest, because it fails
    *plausibly*: `hyprctl clients` reports `at`/`size` in LOGICAL points
    (a 3440x1440 monitor at scale 1.25 is a 2752x1152 coordinate space)
    while `grim` writes the PHYSICAL 3440x1440 framebuffer. Cropping the
    one with the other's numbers gives a picture that is the right shape,
    off by 25%, and looks fine until you notice the left panel is cut in
    half. `grim -g` does not save you — it takes logical geometry and
    returns physical pixels, so the output is a size you did not ask for.
    """
    mons = json.loads(subprocess.run([HYPRCTL, "monitors", "-j"],
                                     capture_output=True, text=True,
                                     check=True).stdout)
    return float(mons[0].get("scale", 1.0)) if mons else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrd", help="recording to open (implies the rerun cmd)")
    ap.add_argument("--cmd", nargs=argparse.REMAINDER,
                    help="arbitrary command instead of rerun")
    ap.add_argument("--match", default="rerun",
                    help="substring of the window class/title to wait for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--geometry", default="60,60,2000,1260",
                    help="x,y,w,h the floating window is pinned to")
    ap.add_argument("--settle", type=float, default=14.0,
                    help="seconds to let the app load before the capture; a "
                         "60 MB .rrd needs about ten")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    cmd = a.cmd or [os.path.expanduser(
        "~/miniconda3/envs/robo/bin/rerun"), a.rrd]
    client = launch(cmd, a.match.lower())
    x, y, w, h = (int(v) for v in a.geometry.split(","))
    client = place(client["address"], x, y, w, h)
    print(f"window {client['address']} class={client['class']} "
          f"at={client['at']} size={client['size']} pid={client['pid']}")
    time.sleep(a.settle)
    # Re-assert: loading a 80 MB recording takes long enough for the layout
    # to be recomputed underneath us at least once.
    client = place(client["address"], x, y, w, h)
    print("captured", grim(client, a.out), client["size"])
    if not a.keep:
        os.kill(client["pid"], signal.SIGTERM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
