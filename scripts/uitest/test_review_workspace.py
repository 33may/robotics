#!/usr/bin/env python
"""T1-T6: drive the Rerun review workspace with real input, assert on pixels.

    python launch.py /tmp/rrd/2408-cup3.rrd --width 1600 --height 1000
    python test_review_workspace.py
    python cleanup.py

This is the *verify* end of the loop in `README.md`, written against the
workspace `inspection/record/rr/workspace.py` produces. Every action
is a real trusted mouse event through CDP; every assertion is a measurement
on a screenshot, not on the SDK.

  T1  the blueprint arrives arranged        3 panels are non-blank + distinct
  T2  the slider grows the cloud            step 0 -> 15 -> 29, 3D region diff
                                            + cloud-coloured pixels increase
  T3  the chain panel tracks the slider     chain region diff over the same
  T4  the placeholder AABB toggles off      3D diff on, restore diff ~= 0
  T5  the stored/replayed overlay toggles   3D diff on + the two overlay
                                            colours appear where they were absent
  T6  cameras toggle off in one click       no camera shape survives, and
                                            the cloud comes out from under it

Coordinates are for a 1600x1000 viewport with THIS blueprint. They are
constants rather than a search because the blueprint is baked into the
`.rrd`: the panels cannot move between runs, which is the whole reason the
layout is saved in the file. `--calibrate` re-derives the timeline mapping
if that ever stops being true.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cdp as C  # noqa: E402
from imgdiff import diff as img_diff  # noqa: E402

PY = sys.executable
ART = C.ART

# ---------------------------------------------------------------- geometry
R3D = (205, 55, 1060, 690)        # the 3D view, inside its frame
RCHAIN = (1270, 30, 330, 715)     # the 2x2 chain grid
RTREE = (0, 190, 200, 560)        # blueprint entity tree
RTIME = (200, 755, 1400, 245)     # time panel + streams

#: Clicking the STREAMS band scrubs; clicking the ruler 100 px higher makes a
#: loop selection instead (learned the hard way, see README).
SCRUB_Y = 900
#: step -> x, calibrated on 2408-cup3 (30 steps): six probe clicks read back
#: #0 #4 #11 #17 #23 #29. Good to the step across the whole span.
STEP_X0, STEP_DX = 205.0, 47.2

#: The eye toggle only exists while the row is hovered; it sits at the right
#: edge of the blueprint panel, on the row's own y.
EYE_X = 179
#: The `>` triangles that open the scene view and then `world`. Both are
#: needed: expanding the view alone leaves `world` shut, and the rows that
#: follow are then `chain`'s, so a "toggle" quietly hides the chain panel
#: instead of an entity. Each level is indented ~12 px further right.
SCENE_EXPANDER = (24, 240)
WORLD_EXPANDER = (36, 264)
#: Rows are 24 px apart, in schema order once both expanders are open.
#: Their ABSOLUTE y is not fixed — it rides on the height of the Sources
#: panel above, which grows with the number of loaded recordings, and an
#: 8 px shift is enough to turn "hide the AABB" into "hide the chain
#: panel". So only the ORDER is a constant here; `find_rows()` measures
#: the origin at runtime.
ROW_ORDER = ["scene", "world", "cam", "cloud", "fresh", "overlay",
             "replayed", "stored", "prim", "table", "chain"]
ROW_PITCH = 24

#: Somewhere with no widget under it. Rerun draws a hover highlight, and a
#: cursor left on a row is worth ~0.08 changed-fraction of pure noise.
PARK = (100, 700)

#: rr/log.py's two overlay channels. The assertion for T5 is not "something
#: changed" but "these two colours are now on screen", which a stray repaint
#: cannot fake.
C_STORED = (60, 170, 255)
C_REPLAYED = (255, 74, 138)
C_CAM = (120, 205, 255)


# ------------------------------------------------------------------ helpers
def run(*args: str) -> None:
    subprocess.run([PY, *args], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def shot(name: str, settle: float = 0.9) -> Path:
    """Park the cursor, wait, capture. The parking is not optional."""
    run("act.py", "move", str(PARK[0]), str(PARK[1]))
    time.sleep(settle)
    p = ART / f"{name}.png"
    run("observe.py", str(p))
    return p


def click(x: float, y: float) -> None:
    run("act.py", "click", str(int(x)), str(int(y)))
    time.sleep(0.45)


def seek(step: int) -> None:
    click(STEP_X0 + STEP_DX * step, SCRUB_Y)


def ink_rows(img: Path) -> int:
    """How many 24 px rows of the blueprint tree have text in them."""
    a = region_px(img, RTREE).astype(int)
    n = 0
    for i in range(RTREE[3] // ROW_PITCH):
        band = a[i * ROW_PITCH:(i + 1) * ROW_PITCH]
        if band.std() > 12:
            n += 1
    return n


def find_rows(img: Path) -> dict[str, int]:
    """Locate the tree's rows by measuring, not by remembering.

    Bands of ink in the blueprint panel are the text rows; the panel header
    above them is ink too, so the rows are picked out as the LONGEST run of
    bands spaced exactly `ROW_PITCH` apart. That run is the tree, its first
    element is the view row, and everything else follows from the schema
    order — which is the only part that is genuinely fixed.
    """
    a = region_px(img, RTREE).astype(int)
    prof = a.std(axis=(1, 2))
    ink = prof > 10
    bands, run = [], None
    for y, v in enumerate(ink):
        if v and run is None:
            run = y
        elif not v and run is not None:
            if y - run >= 4:
                bands.append((run + y) // 2)
            run = None
    # Anchor on the row we already know we hit: `SCENE_EXPANDER` opened the
    # view, so the view's row is the band nearest it. Taking the longest
    # 24-pitch chain instead is one row too generous — the panel's own
    # "Blueprint" header sits exactly one pitch above the first row and
    # joins the run, which shifts every name down by one and turns
    # "hide /world/cam" into "hide /world". (Observed: a 0.99 changed_frac
    # and a blank white view, which is what an off-by-one looks like when
    # your only feedback is pixels.)
    anchor = SCENE_EXPANDER[1] - RTREE[1]
    if not bands:
        raise SystemExit("no rows found in the blueprint panel")
    start = min(range(len(bands)), key=lambda i: abs(bands[i] - anchor))
    chain = [bands[start]]
    for b in bands[start + 1:]:
        if abs(b - (chain[-1] + ROW_PITCH)) <= 3:
            chain.append(b)
    if len(chain) < len(ROW_ORDER):
        raise SystemExit(f"only found {len(chain)} tree rows, need "
                         f"{len(ROW_ORDER)} — is the tree expanded?")
    return {name: RTREE[1] + chain[i] for i, name in enumerate(ROW_ORDER)}


def _expand(x: int, y: int, tag: str) -> tuple[int, int]:
    """Click an expander and make sure the tree got LONGER, not shorter."""
    n0 = ink_rows(shot(f"_tree_{tag}0", settle=0.35))
    click(x, y)
    n1 = ink_rows(shot(f"_tree_{tag}1", settle=0.35))
    if n1 < n0:
        click(x, y)
        n1 = ink_rows(shot(f"_tree_{tag}2", settle=0.35))
    return n0, n1


def ensure_tree_expanded() -> None:
    """Open the tree down to the entity rows, whatever state it opened in.

    The expansion state of the blueprint panel is NOT part of the saved
    blueprint and NOT deterministic between viewer sessions — the same
    `.rrd` opened twice gave an expanded tree once and a collapsed one the
    next time. That silently turns every ROW_Y constant into a click on the
    wrong row: with `world` shut, y=288 is `chain`, and "hide the AABB"
    hides the whole chain panel instead, which even changes enough pixels
    to look like it worked. So the suite ESTABLISHES the state rather than
    assuming it, and does it per level.
    """
    a = _expand(*SCENE_EXPANDER, "scene")
    b = _expand(*WORLD_EXPANDER, "world")
    ROW_Y.update(find_rows(shot("_tree_final", settle=0.4)))
    print(f"  tree rows with content: {a[0]} -> {a[1]} -> {b[1]}")
    print(f"  rows located: " + ", ".join(f"{k}@{v}" for k, v in ROW_Y.items()))


def toggle(row: str) -> None:
    """Hover the row so its eye appears, then click the eye."""
    y = ROW_Y[row]  # populated by ensure_tree_expanded()
    run("act.py", "move", str(EYE_X), str(y))
    time.sleep(0.35)
    click(EYE_X, y)


def frac(a: Path, b: Path, region) -> float:
    return img_diff(a, b, region=region)["changed_frac"]


def region_px(img: Path, region) -> np.ndarray:
    x, y, w, h = region
    return np.asarray(Image.open(img).convert("RGB"),
                      dtype=np.int16)[y:y + h, x:x + w]


def color_count(img: Path, region, rgb, tol: int = 40) -> int:
    a = region_px(img, region)
    return int((np.abs(a - np.array(rgb, dtype=np.int16)).max(2) <= tol).sum())


def cloud_px(img: Path, region) -> int:
    """Points, not grid: anything clearly brighter than the dark backdrop."""
    a = region_px(img, region)
    return int((a.mean(2) > 110).sum())


ROW_Y: dict[str, int] = {}
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# --------------------------------------------------------------------- tests
def t1() -> None:
    print("T1  layout arrives arranged")
    base = shot("T1_open", settle=1.6)
    a = region_px(base, R3D)
    c = region_px(base, RCHAIN)
    t = region_px(base, RTIME)
    tr = region_px(base, RTREE)
    for label, arr, floor in (("3D view", a, 8.0), ("chain grid", c, 8.0),
                              ("time panel", t, 3.0), ("entity tree", tr, 8.0)):
        sd = float(arr.std())
        check(f"T1 {label} rendered", sd > floor, f"pixel std {sd:.1f} > {floor}")
    check("T1 3D and chain are different panels",
          float(np.abs(a[:300, :300].astype(int)
                       - c[:300, :300].astype(int)).mean()) > 5,
          "the two regions are not the same picture")
    print(f"  evidence {base}")


def t2_t3() -> None:
    print("T2/T3  scrub 0 -> 15 -> 29")
    shots, clouds = [], []
    for s in (0, 15, 29):
        seek(s)
        p = shot(f"T2_step{s:02d}")
        shots.append(p)
        clouds.append(cloud_px(p, R3D))
        print(f"  step {s:>2}: {p}  cloud-ish px in 3D region = {clouds[-1]}")
    d01 = frac(shots[0], shots[1], R3D)
    d12 = frac(shots[1], shots[2], R3D)
    d02 = frac(shots[0], shots[2], R3D)
    check("T2 3D changes 0->15", d01 > 0.01, f"changed_frac {d01:.4f} > 0.01")
    check("T2 3D changes 15->29", d12 > 0.01, f"changed_frac {d12:.4f} > 0.01")
    check("T2 3D changes 0->29", d02 > 0.02, f"changed_frac {d02:.4f} > 0.02")
    check("T2 cloud grows monotonically",
          clouds[0] < clouds[1] < clouds[2],
          f"bright px {clouds[0]} -> {clouds[1]} -> {clouds[2]}")
    c01 = frac(shots[0], shots[1], RCHAIN)
    c12 = frac(shots[1], shots[2], RCHAIN)
    check("T3 chain panel changes 0->15", c01 > 0.02,
          f"changed_frac {c01:.4f} > 0.02")
    check("T3 chain panel changes 15->29", c12 > 0.02,
          f"changed_frac {c12:.4f} > 0.02")


def t4() -> None:
    print("T4  placeholder AABB toggles off")
    seek(29)
    before = shot("T4_before_prim_on")
    toggle("prim")
    off = shot("T4_after_prim_off")
    d = frac(before, off, R3D)
    # A wireframe box is ~1100 px of a 1060x690 region, i.e. 0.0015. Asking
    # for more than that is asking the test to fail on a passing feature —
    # thresholds are set from what the geometry can produce, not from a
    # round number. The colour count below is the real assertion.
    check("T4 hiding /world/prim changes the 3D view", d > 0.0008,
          f"changed_frac {d:.4f} > 0.0008")
    # The box is the only yellow thing in the scene, so its disappearance is
    # measurable as a colour and not merely as "some pixels moved".
    y0 = color_count(before, R3D, (255, 214, 61), tol=60)
    y1 = color_count(off, R3D, (255, 214, 61), tol=60)
    check("T4 the yellow AABB is gone", y1 < y0 * 0.25,
          f"aabb-coloured px {y0} -> {y1}")
    toggle("prim")
    back = shot("T4_after_prim_restored")
    d2 = frac(before, back, R3D)
    check("T4 restoring it returns the same picture", d2 < 0.004,
          f"changed_frac vs before {d2:.4f} < 0.004")


def t5() -> None:
    print("T5  stored/replayed overlay toggles on")
    before = shot("T5_before_overlay_off")
    s0 = color_count(before, R3D, C_STORED)
    r0 = color_count(before, R3D, C_REPLAYED)
    toggle("overlay")
    on = shot("T5_after_overlay_on")
    d = frac(before, on, R3D)
    s1 = color_count(on, R3D, C_STORED)
    r1 = color_count(on, R3D, C_REPLAYED)
    # The 100 mm cup is a small part of a 1060x690 view of a 0.6 m cell, so
    # repainting all of it is well under 1% of the region.
    check("T5 the 3D view changes", d > 0.004, f"changed_frac {d:.4f} > 0.004")
    # Scored on the STORED channel alone. The replayed magenta has a dirty
    # baseline — the per-camera axis arrows are red enough to land inside a
    # 40-level tolerance of it, ~1100 px before the overlay is even on — and
    # an assertion that has to be explained is not an assertion.
    check("T5 a second colour appears", s1 > max(5 * s0, s0 + 500),
          f"stored-coloured px {s0} -> {s1} (replayed {r0} -> {r1})")
    toggle("overlay")
    off = shot("T5_after_overlay_off_again")
    d2 = frac(before, off, R3D)
    check("T5 turning it back off restores the picture", d2 < 0.004,
          f"changed_frac vs before {d2:.4f} < 0.004")


def t6() -> None:
    """Anton's ask: ONE click leaves no camera anything on screen."""
    print("T6  cameras toggle off in one click, completely")
    seek(29)
    before = shot("T6_before_cam_on")
    toggle("cam")
    off = shot("T6_after_cam_off")
    d = frac(before, off, R3D)
    check("T6 hiding /world/cam changes the 3D view", d > 0.05,
          f"changed_frac {d:.4f} > 0.05")
    # Scored on WARM pixels, not bright ones. Everything synthetic in this
    # scene is neutral or cyan — grid, frusta, cloud, table — and the only
    # warm thing is photographed plywood, so "R clearly above B" isolates
    # the previews from the wireframe that must survive them. A plain
    # brightness count only fell 34% here and would need a threshold picked
    # to fit rather than to mean something.
    # Nothing camera-shaped may be left — the literal ask.
    f0 = color_count(before, R3D, C_CAM, tol=45)
    f1 = color_count(off, R3D, C_CAM, tol=45)
    check("T6 no blue camera shapes remain", f1 < f0 * 0.02,
          f"frustum-coloured px {f0} -> {f1}")
    # And the point of doing it: the thing under the dome becomes reviewable.
    # Scored on the AABB, which is one flat colour nothing else in the scene
    # shares, so "more of it is on screen" cannot be faked by a repaint.
    #
    # Deliberately NOT scored as "the photo area went away". Three area
    # metrics were tried and all three lied, because the dome OCCLUDES a
    # brighter grid: warm pixels went UP (it was hiding the red axis
    # gizmos), non-background pixels went UP 1.55x, and eroded mid-tone
    # area only fell 28% because the table patch is also a filled surface.
    # A metric that needs a threshold chosen after seeing the answer is not
    # evidence.
    y0 = color_count(before, R3D, (255, 214, 61), tol=60)
    y1 = color_count(off, R3D, (255, 214, 61), tol=60)
    check("T6 the cloud and its box come out from under the dome",
          y1 > y0 * 3, f"aabb-coloured px {y0} -> {y1}")
    toggle("cam")
    back = shot("T6_after_cam_restored")
    check("T6 restoring brings them back", frac(before, back, R3D) < 0.004,
          f"changed_frac vs before {frac(before, back, R3D):.4f} < 0.004")


def calibrate() -> None:
    """Re-derive STEP_X0/STEP_DX by clicking and reading the step field back."""
    print("calibration: click x -> crop of the step field")
    for x in range(210, 1600, 150):
        click(x, SCRUB_Y)
        p = shot(f"_cal_{x}", settle=0.4)
        subprocess.run([PY, "crop.py", str(p), "--region", "296,755,120,22",
                        "--out", str(ART / f"_calc_{x}.png"), "--scale", "3"],
                       cwd=HERE, check=True, stdout=subprocess.DEVNULL)
        print(f"  x={x} -> {ART / f'_calc_{x}.png'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma list of t1,t2,t4,t5,t6")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
        return 0
    want = set((a.only or "t1,t2,t4,t5,t6").split(","))
    ensure_tree_expanded()
    if "t1" in want:
        t1()
    if "t2" in want:
        t2_t3()
    if "t4" in want:
        t4()
    if "t5" in want:
        t5()
    if "t6" in want:
        t6()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  FAILED {n}: {d}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
