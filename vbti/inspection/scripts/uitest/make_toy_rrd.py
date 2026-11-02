#!/usr/bin/env python
"""Build a deterministic toy .rrd for UI-loop smoke tests.

Designed so that three different UI interactions are each verifiable by a pixel diff:

  * timeline drag   -> `/world/cloud_a` grows monotonically with `step`, and
                       `/cam/image` changes colour every step.
  * entity toggling -> `/world/cloud_a` and `/world/cloud_b` occupy disjoint
                       screen regions, so hiding one blanks a known area.
  * panel clicks    -> a fixed blueprint means panel geometry is reproducible.

Usage:
    python make_toy_rrd.py [--out /tmp/rrd/_toy.rrd] [--steps 60]

Verified against rerun-sdk 0.37.0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

TIMELINE = "step"
IMG_W, IMG_H = 128, 96


def build(out: Path, steps: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0xC0FFEE)

    # Pre-generate the full point set once so the cloud *grows* rather than
    # jitters: at step k we log the first (k+1)*per_step points.
    per_step = 12
    total = steps * per_step
    # cloud_a lives in x in [-3,-1] (screen-left-ish), cloud_b in x in [1,3].
    a_all = rng.uniform([-3.0, -1.0, -1.0], [-1.0, 1.0, 1.0], size=(total, 3))
    b_all = rng.uniform([1.0, -1.0, -1.0], [3.0, 1.0, 1.0], size=(256, 3))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="world",
                background=[16, 16, 24],
                line_grid=False,
            ),
            rrb.Spatial2DView(origin="/cam", name="cam"),
            column_shares=[2.0, 1.0],
        ),
        rrb.BlueprintPanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        # play_state="paused" is ESSENTIAL for a deterministic UI test: by
        # default the viewer auto-plays the recording, so every screenshot
        # catches a different `step` and every assertion is flaky.
        rrb.TimePanel(state="expanded", timeline=TIMELINE, play_state="paused"),
        auto_layout=False,
        auto_views=False,
    )

    rr.init("uitest_toy", recording_id="uitest-toy-fixed", spawn=False)

    # Static: cloud_b never changes, so the only reason it can disappear from a
    # screenshot is an explicit visibility toggle.
    rr.set_time(TIMELINE, sequence=0)
    rr.log(
        "/world/cloud_b",
        rr.Points3D(b_all, colors=[60, 130, 255], radii=0.035),
        static=True,
    )

    for k in range(steps):
        rr.set_time(TIMELINE, sequence=k)

        n = (k + 1) * per_step
        rr.log("/world/cloud_a", rr.Points3D(a_all[:n], colors=[255, 80, 60], radii=0.035))

        # Image: horizontal band sweeps down + hue ramps, so consecutive steps
        # differ a lot in mean-abs-diff terms.
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        img[:, :, 0] = np.uint8((k * 255) // max(steps - 1, 1))
        img[:, :, 2] = np.uint8(255 - (k * 255) // max(steps - 1, 1))
        band = (k * IMG_H) // max(steps, 1)
        img[band : band + 12, :, 1] = 255
        rr.log("/cam/image", rr.Image(img, color_model="RGB"))

        rr.log("/world/counter", rr.TextLog(f"step={k} n_points={n}"))

    rr.save(out, default_blueprint=blueprint)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/rrd/_toy.rrd"))
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    p = build(args.out, args.steps)
    print(f"{p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
