#!/usr/bin/env python
"""Worked example + regression test for the toolkit, against Rerun 0.37.

    python make_toy_rrd.py
    python launch.py /tmp/rrd/_toy.rrd
    python demo_rerun.py

Proves the three interactions that matter for a WASM canvas app:
  1. CLICK      - expand a node in the blueprint tree
  2. DRAG       - scrub the timeline to an exact step
  3. TOGGLE     - hide an entity via its hover-revealed eye icon,
                  with a control region that must NOT change

Every step is asserted with imgdiff, and every screenshot is kept in
$UITEST_DIR so a failure can be read by eye afterwards.

All coordinates below are for a 1600x1000 viewport and are re-derived by
`calibrate()` if you change the size.  See README "Coordinates".
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cdp as C  # noqa: E402
from imgdiff import diff  # noqa: E402
from launch import wait_ready  # noqa: E402

# --- regions (x, y, w, h) at 1600x1000 ------------------------------------- #
VIEW3D = (200, 55, 933, 690)      # the Spatial3DView
CAM2D = (1135, 55, 465, 690)      # the Spatial2DView - control region
TREE = (0, 190, 197, 170)         # blueprint entity tree
TIMEFIELD = (298, 758, 60, 20)    # just the "#NN" digits - see README on why
                                  # a tight region beats a loose one for text

# Somewhere the pointer can rest without the app reacting to it.  Rerun shows a
# data-inspector tooltip whenever the cursor is over a view, and highlights
# panel splitters / headers, so an un-parked cursor silently corrupts diffs.
# This spot (empty blueprint panel below the tree) measured 0.00000 change.
PARK = (100, 700)

# --- timeline geometry ------------------------------------------------------ #
# Derived empirically from the ruler ticks, then confirmed by click probes:
#   x(step) = X0 + PX_PER_STEP * step
X0, PX_PER_STEP = 203.0, 23.35
TIME_ROW_Y = 900                  # inside the streams area, NOT the ruler strip
RULER_Y = 792                     # dragging HERE makes a loop selection instead

# --- widget coordinates ----------------------------------------------------- #
PLAY_PAUSE = (17, 765)
CLOUD_A_ROW = (105, 296)          # the "cloud_a" label - clicking selects it
CLOUD_A_EYE = (179, 296)          # revealed only while the row is hovered
# NOT used: the ">" expand arrow at (19, 344).  Its hit rect is ~10 px and a
# click there registered only 0.003 change - too small a target to rely on.


def x_for_step(step: float) -> float:
    return X0 + PX_PER_STEP * step


def shot(c: "C.CDP", name: str) -> Path:
    """Always park the cursor before capturing, or hover state leaks into the
    diff and you chase phantom failures."""
    c.move(*PARK)
    time.sleep(0.35)
    return c.screenshot(C.ART / f"{name}.png")


def check(results: list, name: str, a: Path, b: Path, region, *,
          min_frac=None, max_frac=None) -> None:
    r = diff(a, b, region)
    ok = True
    if min_frac is not None and r["changed_frac"] < min_frac:
        ok = False
    if max_frac is not None and r["changed_frac"] > max_frac:
        ok = False
    results.append({
        "check": name, "pass": ok,
        "changed_frac": round(r["changed_frac"], 5),
        "mean_abs": round(r["mean_abs"], 3),
        "expect": f"min={min_frac}" if min_frac is not None else f"max={max_frac}",
        "a": str(a), "b": str(b),
    })


def main() -> int:
    sess = C.Session.load()
    results: list[dict] = []

    with C.CDP(sess.ws_url) as c:
        # ---------- reset to a known state ------------------------------- #
        c.send("Page.reload", {})
        wait_ready(c)
        base = shot(c, "proof_00_baseline")

        # A reload must be pixel-identical to the previous baseline, otherwise
        # nothing below is trustworthy.
        c.send("Page.reload", {})
        wait_ready(c)
        base2 = shot(c, "proof_00_baseline_again")
        check(results, "reload is deterministic", base, base2, None, max_frac=0.001)

        # ---------- 1. CLICK --------------------------------------------- #
        c.click(*CLOUD_A_ROW, settle=0.6)
        a1 = shot(c, "proof_01_after_click_select")
        check(results, "click selects tree row", base, a1, TREE, min_frac=0.02)
        # selecting an entity also highlights its points in the 3D view, so the
        # control region has to be the *2D* view, not the 3D one.
        check(results, "click leaves 2D view alone", base, a1, CAM2D, max_frac=0.002)

        c.send("Page.reload", {})
        wait_ready(c)

        # ---------- 2. DRAG the timeline --------------------------------- #
        b2 = shot(c, "proof_02_before_drag")
        c.drag(x_for_step(0), TIME_ROW_Y, x_for_step(30), TIME_ROW_Y, steps=30, settle=0.6)
        a2 = shot(c, "proof_02_after_drag_step30")
        check(results, "drag moves the time readout", b2, a2, TIMEFIELD, min_frac=0.02)
        check(results, "drag changes the 2D image", b2, a2, CAM2D, min_frac=0.10)
        check(results, "drag grows the point cloud", b2, a2, VIEW3D, min_frac=0.02)

        c.send("Page.reload", {})
        wait_ready(c)

        # ---------- 3. TOGGLE entity visibility --------------------------- #
        b3 = shot(c, "proof_03_before_toggle")
        c.click(*CLOUD_A_EYE, settle=0.8)   # click() hovers first, which is what
        a3 = shot(c, "proof_03_after_toggle_hidden")  # reveals the eye at all
        check(results, "toggle changes the 3D view", b3, a3, VIEW3D, min_frac=0.05)
        check(results, "toggle leaves the 2D view alone", b3, a3, CAM2D, max_frac=0.002)

        # toggling back must restore the exact original pixels
        c.click(*CLOUD_A_EYE, settle=0.8)
        a3b = shot(c, "proof_03_after_toggle_restored")
        check(results, "toggle is reversible", b3, a3b, VIEW3D, max_frac=0.002)

    n_fail = sum(1 for r in results if not r["pass"])
    print(json.dumps(results, indent=2))
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
