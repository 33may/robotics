#!/usr/bin/env python
"""F1-F5: the fit channels of the review workspace, driven and measured.

    python launch.py <derived>/<run>/rrd/<run>.rrd --width 1600 --height 1000
    python test_fit_channels.py --run 2408-cup3
    python cleanup.py

The claim under test is the one `workspace.py` makes about the primitive
decompositions it sweeps out of `inspection/data/derived/<run>/fits/*/`:

  F1  every swept method is a channel        its row exists in the entity tree,
                                             even when it never draws (planar
                                             abstains on all 30 steps of cup3)
  F2  a channel paints its own primitives    toggling it off removes the
                                             method-coloured pixels and only
                                             those; toggling it back on is
                                             pixel-reversible
  F3  the pixels agree with the JSON         a step where the decomposition has
                                             primitives shows them; a step where
                                             it abstains shows NOTHING — which is
                                             the latest-at retirement working
  F4  scrubbing changes the primitives       the method-coloured mask at a mid
                                             step is not the mask at the last one
  F5  a channel is not the whole scene       the chain panel and the other
                                             methods survive one method's toggle

Everything the suite expects is read from the same fit JSONs the producer
read (`_expectations`), so this is the pixels being held against the data,
not against a constant somebody typed twice.

Three things here are deliberate and hard-won; changing them will produce
green-looking nonsense.

1. ROWS ARE FOUND WITH THE BLUEPRINT SEARCH BOX, not counted from the top.
   Which subtrees the viewer auto-expands is not part of the saved blueprint
   (README trap 13) and it now varies with the DATA — `cyl` names its
   cylinders by geometric content, so cup3 has six label rows under it and
   cup5 has a different six. Typing a method name into the blueprint filter
   collapses the tree to the path chain that matches, and the target is then
   at a fixed depth: view > world > prim > <method>.

2. AFTER TYPING, THE PANEL SHIFTS 8 px. The search field changes height when
   it loses focus, which moves every row. A click aimed at a row measured
   before the shift lands between two rows: it selected the entity instead of
   hitting its eye, and the 3D view did not change by a single pixel — a
   silent no-op that looks exactly like "the toggle does not work". So the
   filter is applied, focus is dropped, and only THEN are rows measured.

3. THE METRIC IS A HUE, NOT AN AREA. Each channel is scored by the number of
   pixels whose CHROMA DIRECTION matches the method's colour (cosine >= 0.99
   after removing the grey component), which is invariant to the translucent
   superquadric shell being blended over whatever is behind it. Measured on
   the real workspace: amber(ems) reads 8668 px with the channel on and
   exactly 0 with it off, while the fresh-point orange 10 degrees away in hue
   and the plywood in the camera previews both score 0. An area metric would
   not survive this scene: hiding a channel UNCOVERS what it occluded, and
   the violet cylinder count goes UP when the amber shell is hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cdp as C  # noqa: E402
from imgdiff import diff as img_diff  # noqa: E402

from inspection.record.rr import log as rr_log  # noqa: E402

PKG = HERE.parents[1]  # vbti/inspection
DERIVED = PKG / "data" / "derived"
ART = C.ART

# ---------------------------------------------------------------- geometry
R3D = (205, 55, 1060, 690)        # the 3D view, inside its frame
RCHAIN = (1270, 30, 330, 715)     # the 2x2 chain grid (the control region)
RTREE = (0, 216, 200, 534)        # blueprint tree, BELOW the search field
PARK = (100, 700)                 # inert: no widget, no hover highlight
SEARCH = (158, 200)               # magnifier when idle, clear-X when filtering
EYE_X = 179                       # the per-row eye, drawn only while hovered
ROW_PITCH = 24
SCRUB_Y = 900                     # the streams band; the ruler above it
#                                   makes loop selections instead (trap 3)
#: The playhead is dark on the light time panel and runs the FULL height of
#: it; the ruler ticks only occupy the top ~20 px. Measuring over the whole
#: height is what separates them (a full line scores 213, a tick ~20).
PLAY_BAND = (782, 995)
#: Starts at 196, not 202: at step 0 the playhead sits at x=199, ON the
#: panel's left border, and a scan that begins after it reports the nearest
#: ruler tick instead — a wrong step that looks like a right one.
PLAY_X = (196, 1580)

#: rr/log.py's frustum colour. The cameras are hidden for the duration of
#: the measurements (F0) and this is how that is confirmed.
C_CAM = (120, 205, 255)

#: A channel counts as ON SCREEN above this many method-coloured pixels and
#: as ABSENT below the floor. The gap between them is three orders of
#: magnitude in every measurement taken here (0 or 1 px when absent, >1000
#: when present), so neither number is doing any work.
PRESENT_PX = 200
ABSENT_PX = 60


# ------------------------------------------------------------------ metric
def region(img: Path, reg) -> np.ndarray:
    x, y, w, h = reg
    return np.asarray(Image.open(img).convert("RGB"),
                      dtype=np.float64)[y:y + h, x:x + w]


def hue_mask(a: np.ndarray, rgb, cos_min: float = 0.99,
             mag_min: float = 45.0) -> np.ndarray:
    """Pixels painted in `rgb`'s hue, whatever they were painted over.

    Chroma = colour minus its own grey component. Blending a translucent
    shell over a neutral backdrop scales the chroma but does not rotate it,
    so the mask survives the superquadric being drawn at alpha 110 over the
    cloud, the table and the void alike. Magnitude gates out the greys
    (cloud, grid, table), direction gates out the other hues.
    """
    t = np.asarray(rgb, dtype=np.float64)
    tc = t - t.min()
    tc = tc / np.linalg.norm(tc)
    c = a - a.min(2, keepdims=True)
    mag = np.linalg.norm(c, axis=2)
    ok = mag >= mag_min
    cos = np.zeros_like(mag)
    cos[ok] = (c[ok] @ tc) / mag[ok]
    return ok & (cos >= cos_min)


def method_px(img: Path, method: str) -> int:
    return int(hue_mask(region(img, R3D), rr_log.method_color(method)).sum())


def method_mask(img: Path, method: str) -> np.ndarray:
    return hue_mask(region(img, R3D), rr_log.method_color(method))


def scoped_diff(a: Path, b: Path, method: str) -> tuple[int, float]:
    """How much of the method's own ink differs between two frames.

    Returned as (XOR pixels, XOR / union) so a scrub that merely nudges the
    fit and one that replaces it read differently.
    """
    ma, mb = method_mask(a, method), method_mask(b, method)
    xor = int((ma ^ mb).sum())
    union = int((ma | mb).sum())
    return xor, (xor / union if union else 0.0)


def frac(a: Path, b: Path, reg) -> float:
    return img_diff(a, b, region=reg)["changed_frac"]


# ---------------------------------------------------------------------- UI
class UI:
    def __init__(self, c: C.CDP):
        self.c = c
        self.filtered = False
        self.cur = 0

    # -- observe
    def shot(self, name: str, settle: float = 0.9) -> Path:
        self.c.move(*PARK)
        time.sleep(settle)
        return self.raw_shot(name)

    def raw_shot(self, name: str) -> Path:
        """Capture WITHOUT parking — for looking at hover state itself."""
        p = ART / f"{name}.png"
        self.c.screenshot(p)
        return p

    def click(self, x: float, y: float, settle: float = 0.4) -> None:
        self.c.click(x, y)
        time.sleep(settle)

    def hover_click(self, x: float, y: float) -> None:
        """For controls that only exist while their row is hovered."""
        self.c.move(x, y)
        time.sleep(0.35)
        self.c.click(x, y)
        time.sleep(0.45)

    # -- the entity tree, addressed by search
    def filter_tree(self, term: str) -> None:
        self.clear_filter()
        self.click(*SEARCH)
        self.c.text(term)
        time.sleep(0.4)
        # Drop focus BEFORE measuring: the field is 8 px taller while it has
        # the caret, and every row below it moves when that changes.
        self.click(*PARK)
        time.sleep(0.3)
        self.filtered = True

    def clear_filter(self) -> None:
        """Escape closes the filter AND returns keyboard focus to the app.

        Focus matters beyond the tree: with the caret still in the search
        field, the arrow keys that scrub the timeline would move a text
        cursor instead, and the seek would silently not happen.
        """
        self.c.key("Escape")
        time.sleep(0.35)
        self.filtered = False

    def rows(self, tag: str) -> tuple[list[int], list[float]]:
        """(row centres, ink per row) — the tree's 24 px ladder, phased now.

        Found by CORRELATING the ink profile with a 24 px comb rather than
        by looking for gaps between rows. Gaps are not reliable: filtering
        on `cam` shows thirty camera subtrees, whose indent guides and
        selected-row background keep the profile off the floor the whole
        way down, and a gap-based reading merged all of them into one band.
        The pitch is the one thing that never varies.
        """
        a = region(self.shot(f"_rows_{tag}", settle=0.35), RTREE)
        prof = a.std(axis=(1, 2))
        k = np.ones(7) / 7.0
        prof = np.convolve(prof, k, mode="same")
        phase = max(range(ROW_PITCH),
                    key=lambda o: float(prof[o::ROW_PITCH].sum()))
        ys = list(range(phase, RTREE[3] - 2, ROW_PITCH))
        return ([RTREE[1] + y for y in ys], [float(prof[y]) for y in ys])

    def target_row(self, term: str, depth: int, tag: str) -> int:
        """The y of `term`'s own row, after filtering the tree to it.

        `depth` counts the rows above it in the filtered tree: the panel's
        "Viewport" caption, the view, and each ancestor entity — so
        `/world/prim/ems` is 4 and `/world/cam` is 3. Getting this wrong is
        the failure mode that hid the chain panel and still looked like a
        pass, so the whole path chain above the target must have ink in it
        before the row is returned, and the caller's assertion is on a
        colour only that entity paints.
        """
        self.filter_tree(term)
        ys, ink = self.rows(tag)
        if len(ys) <= depth:
            raise SystemExit(f"filter {term!r}: only {len(ys)} rows")
        # An absolute floor, not a relative one: filtering on `cam` leaves
        # thirty inked rows and their median is what a FULL row looks like,
        # so a relative floor called a perfectly good row empty. Background
        # rows measure ~0-2 here and text rows 30-60.
        weak = [(ys[i], round(ink[i], 1)) for i in range(depth + 1)
                if ink[i] < 8.0]
        if weak:
            raise SystemExit(f"filter {term!r}: rows {weak} are empty — the "
                             f"filtered path chain is not {depth + 1} deep")
        return ys[depth]

    def toggle(self, y: int, tag: str) -> None:
        """Hover the row, PROVE the hover landed, then click its eye.

        The eye is only drawn while its row is hovered, so a hover that
        missed produces a click on nothing — which changed not one pixel of
        the 3D view and reads exactly like a broken feature. The hover is
        therefore checked against the row's own pixels before the press.
        """
        self.c.move(*PARK)
        time.sleep(0.3)
        before = region(self.raw_shot(f"_hov0_{tag}"), RTREE)
        self.c.move(EYE_X, y)
        time.sleep(0.4)
        after = region(self.raw_shot(f"_hov1_{tag}"), RTREE)
        i = y - RTREE[1]
        band = float(np.abs(after[i - 8:i + 8] - before[i - 8:i + 8]).mean())
        if band < 3.0:
            raise SystemExit(f"hover at y={y} did not light the row "
                             f"(mean delta {band:.1f}) — the eye is not there")
        self.c.click(EYE_X, y)
        time.sleep(0.5)

    def reset(self) -> None:
        """Back to the file's own state: the blueprint, nothing toggled.

        Reloading the page is what makes running the suite three times mean
        anything — a run that started on the leftovers of the previous one
        would drift, and drift is exactly what rule 4 is meant to catch.
        """
        from launch import wait_ready
        self.c.send("Page.reload", {"ignoreCache": False})
        wait_ready(self.c)
        self.filtered = False
        time.sleep(1.0)

    # -- the timeline
    def playhead(self) -> tuple[int, int]:
        """(x, height) of the playhead — the full-height dark line.

        Scored as "how many rows of this column are dark", not as "which
        column is darkest": the ruler's #10/#20 ticks are dark too and win
        an average, and the answer was then a plausible wrong step. The
        playhead is the only mark that runs the whole height of the panel.
        """
        p = ART / "_playhead.png"
        self.c.move(*PARK)
        time.sleep(0.25)
        self.c.screenshot(p)
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64).mean(2)
        dark = (a[PLAY_BAND[0]:PLAY_BAND[1], PLAY_X[0]:PLAY_X[1]] < 170).sum(0)
        i = int(np.argmax(dark))
        return i + PLAY_X[0], int(dark[i])

    def press(self, key: str, times: int, settle: float = 0.15) -> None:
        """Send an arrow `times` times — knowing some of them will not stick.

        Rerun advances at most ONE step per rendered frame (egui's
        `key_pressed` is a per-frame boolean, not a count), and the frame
        rate here is the software rasterizer's: measured at ~10 steps/s
        early in a run and ~2.5 steps/s at the end, where thirty textured
        camera previews are in the scene. So 40 presses at 10 Hz moved the
        playhead by 10 steps. Nothing that counts presses can be trusted;
        everything above reads the position back instead.
        """
        self.clear_filter()
        for _ in range(times):
            self.c.key(key, settle=settle)
        time.sleep(0.4)

    def _end(self, key: str, n_steps: int, tag: str) -> int:
        """Hold an arrow until the playhead stops moving: an exact endpoint.

        CLAMPING is what makes this exact. A calibration built from "I
        pressed right ten times so I am at step 10" came back wrong by two
        steps and perfectly self-consistent about it. The end of the
        timeline cannot be missed by a step — the extra presses do nothing
        — and two rounds that read the same position are the proof it was
        reached rather than merely approached.
        """
        prev, x, h = None, -1, 0
        for _ in range(15):
            self.press(key, max(6, n_steps // 2))
            x, h = self.playhead()
            if h >= 150 and prev == x:
                return x
            prev = x
        raise SystemExit(f"{tag}: playhead still moving at {x}(h{h}) — never "
                         f"reached the end of the timeline")

    def calibrate(self, n_steps: int) -> tuple[float, float]:
        """x(step) from the two ENDS of the timeline, which cannot be missed."""
        x0 = self._end("ArrowLeft", n_steps, "rewind")
        xn = self._end("ArrowRight", n_steps, "fast forward")
        self.cur = n_steps - 1
        return x0, (xn - x0) / (n_steps - 1)

    def where(self, cal) -> int:
        """The step under the playhead, read off the panel in absolute terms."""
        x, h = self.playhead()
        if h < 150:
            raise SystemExit(f"playhead not found (h={h})")
        return int(round((x - cal[0]) / cal[1]))

    def seek(self, cal, step: int) -> int:
        """Arrow to the step and keep going until the panel agrees.

        A closed loop rather than a count, because the count is what was
        wrong before: it presses, reads the ABSOLUTE position back off the
        playhead, and corrects. Five rounds is far more than a dropped
        keypress ever needs.
        """
        for _ in range(5):
            cur = self.where(cal)
            if cur == step:
                self.cur = step
                return cur
            d = step - cur
            self.press("ArrowRight" if d > 0 else "ArrowLeft", abs(d))
        raise SystemExit(f"seek {step}: stuck at {self.where(cal)}")


# ------------------------------------------------------------------- input
def _expectations(run: str) -> tuple[list[int], dict[str, dict[int, int]]]:
    """(pose id per step index, {method: {step index: n primitives}}).

    Read from the derived tree, i.e. from the producer's own inputs. If the
    JSON says planar abstained at step 20, the screen has to be empty of
    green at step 20; that is the assertion.
    """
    steps = json.loads((DERIVED / run / "steps" / "summary.json").read_text())
    pose_ids = [s["pose_id"] for s in steps["steps"]]
    methods: dict[str, dict[int, int]] = {}
    for d in sorted(p for p in (DERIVED / run / "fits").iterdir() if p.is_dir()):
        files = {int(f.stem.split("_")[1]): f for f in d.glob("step_*.json")}
        if not files:
            continue
        methods[d.name] = {
            k: len(json.loads(files[pid].read_text())["primitives"])
            for k, pid in enumerate(pose_ids) if pid in files}
    return pose_ids, methods


RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ------------------------------------------------------------------- tests
def run_suite(run: str, ui: UI) -> None:
    ui.reset()
    pose_ids, methods = _expectations(run)
    n = len(pose_ids)
    mid, last = n // 2, n - 1
    print(f"{run}: {n} steps, methods {list(methods)}, probing steps "
          f"{mid} (view {pose_ids[mid]:03d}) and {last} (view "
          f"{pose_ids[last]:03d})")

    # F0 --------------------------------------------------------------
    # The camera dome is hidden for the whole measurement, and this is not
    # cosmetic: its 30 rgb previews are photographs of plywood (warm, i.e.
    # near the ems amber) and every frustum carries a green +Y axis arrow
    # (near the planar green). With the dome up, an empty planar channel
    # still scored 122 "planar" pixels — the metric would have been
    # measuring the cameras. It is also what a reviewer does before looking
    # at a fit, which is why `/world/cam` is one click in the first place.
    print("F0  the camera dome is out of the way")
    cam_on = ui.shot(f"{run}_cams_on")
    cyan_on = int(hue_mask(region(cam_on, R3D), C_CAM).sum())
    cam_y = ui.target_row("cam", 3, f"{run}_cam")
    ui.toggle(cam_y, f"{run}_cam_off")
    cam_off = ui.shot(f"{run}_cams_off")
    cyan_off = int(hue_mask(region(cam_off, R3D), C_CAM).sum())
    check("F0 hiding /world/cam leaves no camera on screen",
          cyan_off < 0.02 * cyan_on and cyan_on > 500,
          f"frustum-coloured px {cyan_on} -> {cyan_off}")
    ui.clear_filter()

    # Calibrated AFTER the dome is down: with thirty textured previews on
    # screen the viewer renders at ~2.5 FPS and swallows most of the arrow
    # keys that drive the timeline.
    cal = ui.calibrate(n)
    print(f"  timeline x(step) = {cal[0]:.0f} + {cal[1]:.2f} * step")

    # F1 --------------------------------------------------------------
    print("F1  every swept method has a channel in the tree")
    rows = {}
    for m in methods:
        rows[m] = ui.target_row(m, 4, f"{run}_{m}")
        drawn = sum(methods[m].values())
        check(f"F1 {m} channel exists", rows[m] > 0,
              f"row at y={rows[m]}, {drawn} primitives over {n} steps"
              + (" (abstains everywhere — the channel is the evidence it ran)"
                 if drawn == 0 else ""))
    ui.clear_filter()

    # F2/F3/F5 --------------------------------------------------------
    shots: dict[tuple[str, int], Path] = {}
    for step in (mid, last):
        ui.seek(cal, step)
        base = ui.shot(f"{run}_step{step:02d}_all_on")
        counts = {m: method_px(base, m) for m in methods}
        print(f"\nF2/F3  step {step} (view {pose_ids[step]:03d})  "
              f"colour counts {counts}")
        print(f"  evidence {base}")
        for m in methods:
            shots[(m, step)] = base
            want = methods[m].get(step, 0)
            # F3: the pixels have to say what the JSON says.
            if want:
                check(f"F3 {m}@{step} JSON says {want} primitive(s), screen agrees",
                      counts[m] >= PRESENT_PX,
                      f"{counts[m]} {m}-coloured px >= {PRESENT_PX}")
            else:
                check(f"F3 {m}@{step} JSON abstains, screen is empty of it",
                      counts[m] <= ABSENT_PX,
                      f"{counts[m]} {m}-coloured px <= {ABSENT_PX} "
                      f"(nothing bled forward from an earlier step)")

        for m in methods:
            if not methods[m].get(step, 0):
                continue  # nothing to toggle; F3 already asserted the void
            y = ui.target_row(m, 4, f"{run}_{m}_s{step}")
            ui.toggle(y, f"{run}_{m}_s{step}_off")
            off = ui.shot(f"{run}_step{step:02d}_{m}_off")
            c_off = {k: method_px(off, k) for k in methods}
            xor, share = scoped_diff(base, off, m)
            check(f"F2 {m}@{step} its own colour goes away",
                  c_off[m] <= max(ABSENT_PX, 0.1 * counts[m]),
                  f"{m}-coloured px {counts[m]} -> {c_off[m]}")
            check(f"F2 {m}@{step} the scoped diff is the primitives",
                  xor >= PRESENT_PX and share > 0.8,
                  f"{xor} px differ, {share:.2f} of the union")
            others = [k for k in methods if k != m and counts[k] >= PRESENT_PX]
            check(f"F5 {m}@{step} the other channels survive",
                  all(c_off[k] >= 0.5 * counts[k] for k in others),
                  f"others {[f'{k}:{counts[k]}->{c_off[k]}' for k in others]}"
                  or "no other channel drawn at this step")
            ch = frac(base, off, RCHAIN)
            check(f"F5 {m}@{step} the chain panel is untouched", ch < 0.005,
                  f"chain changed_frac {ch:.4f} < 0.005 (a wrong row would "
                  f"have hidden a panel)")
            ui.toggle(y, f"{run}_{m}_s{step}_on")
            back = ui.shot(f"{run}_step{step:02d}_{m}_back_on")
            d = frac(base, back, R3D)
            check(f"F2 {m}@{step} toggling back on restores the pixels",
                  d < 0.004, f"3D changed_frac vs baseline {d:.4f} < 0.004")
            ui.clear_filter()

    # F4 --------------------------------------------------------------
    print("\nF4  scrubbing steps changes the primitives")
    for m in methods:
        if not (methods[m].get(mid, 0) and methods[m].get(last, 0)):
            check(f"F4 {m} mid vs last", True,
                  f"skipped — {m} draws at "
                  f"{'mid' if methods[m].get(mid) else 'neither'}"
                  f"{' and last' if methods[m].get(last) else ''} step only")
            continue
        xor, share = scoped_diff(shots[(m, mid)], shots[(m, last)], m)
        check(f"F4 {m} the fit at step {mid} is not the fit at step {last}",
              share > 0.15, f"{xor} px of {m} differ, {share:.2f} of the union")

    # F0 again: leave the workspace as it was found ---------------------
    ui.seek(cal, last)
    cam_y = ui.target_row("cam", 3, f"{run}_cam_back")
    ui.toggle(cam_y, f"{run}_cam_on")
    ui.clear_filter()
    back = ui.shot(f"{run}_cams_back_on")
    d = frac(cam_on, back, R3D)
    check("F0 restoring /world/cam restores the picture", d < 0.004,
          f"3D changed_frac vs the opening frame {d:.4f} < 0.004")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    _, c = C.attach()
    with c:
        ui = UI(c)
        run_suite(a.run, ui)
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  FAILED {n}: {d}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
