"""Generate id_single_cup_30.json — 30 single-cup ID trials inside the green ID zone.

Design:
  - 6 cup positions × 5 trials = 30. Cups reuse id_scale_60's exact positions
    (they sit in the cup_strip, ABOVE the green ID zone), so this protocol is a
    half-density spatial sibling of id_scale_60.
  - 5 Halton-jittered duck positions per cup, sampled ONLY inside the green ID
    zone (the "small green square" = table_mapping.json id_zone bbox), ≥35 px
    from the cup, and OUTSIDE the central-bottom block zone (DUCK_NOGO_BBOX:
    horizontal middle third × bottom third of the zone — robot-sweep region).
  - duck_dir_deg is FULLY RANDOM (uniform 0–360°) — no pointing-at-cup balancing.
    This isolates spatial generalization and tests robustness to arbitrary duck
    orientation. (Contrast id_scale_60 / id_smoke_15, which balance yes/no
    pointing per cup.)
  - Trials grouped by cup_group so the operator only repositions the cup 6 times.
  - Legacy schema (duck_px / cup_px / duck_dir_deg / cup_group) — renders and
    slots into the eval harness identically to id_scale_60.
  - Seed=13, deterministic, matching the id_scale family.

Run:
  python vbti/logic/inference/protocols/generators/make_id_single_cup_30.py
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

from vbti.logic.inference.protocols.render_protocol import render

PROTO_DIR = Path(__file__).resolve().parent.parent  # protocols/
OUT_PATH  = PROTO_DIR / "id_single_cup_30.json"

# Cup positions reused verbatim from id_scale_60 — they live in the cup_strip
# (y=117 / y=173), above the green ID zone where the ducks are sampled.
CUP_POSITIONS = [
    [221, 117], [271, 173], [321, 117],
    [371, 173], [421, 117], [472, 173],
]

# The "small green square" — id_zone bbox, read from table_mapping.json so the
# duck sampling region stays the single source of truth.
_TABLE_MAP     = json.loads((PROTO_DIR / "table_mapping.json").read_text())
WORKSPACE_BBOX = tuple(_TABLE_MAP["id_zone"]["bbox"])  # (270, 191, 424, 313)

# Duck no-go zone — block the central-bottom band of the green ID zone:
# horizontal middle third × bottom third vertically. The robot base + gripper
# sweep up into this region, so ducks spawned here get awkward pick approaches.
# Computed from WORKSPACE_BBOX (thirds) so it tracks the id_zone if recalibrated.
# Same reject-if-inside-bbox pattern as DUCK_NOGO_BBOX in make_dual_cup_30.py.
_wx0, _wy0, _wx1, _wy1 = WORKSPACE_BBOX
_ww, _wh = _wx1 - _wx0, _wy1 - _wy0
DUCK_NOGO_BBOX = (
    round(_wx0 + _ww / 3), round(_wy0 + 2 * _wh / 3),   # x: middle third, y: bottom third
    round(_wx0 + 2 * _ww / 3), _wy1,
)

N_PER_CUP       = 5
MIN_DUCK_TO_CUP = 35.0
SEED            = 13
PROTOCOL_NAME   = "id_single_cup_30"


def halton(idx: int, base: int) -> float:
    f, r = 1.0, 0.0
    while idx > 0:
        f /= base
        r += f * (idx % base)
        idx //= base
    return r


def _in_bbox(px, bbox) -> bool:
    """True if px=(x, y) lies inside bbox=(x0, y0, x1, y1)."""
    bx0, by0, bx1, by1 = bbox
    return bx0 <= px[0] <= bx1 and by0 <= px[1] <= by1


def position_pool(cup, n_want, seed_offset):
    """Halton-spread duck positions inside the green ID zone, ≥MIN_DUCK_TO_CUP
    from the cup and outside DUCK_NOGO_BBOX. seed_offset = cup_idx * 173 keeps
    each cup's pool stable and distinct (same offset scheme as id_scale_60)."""
    x0, y0, x1, y1 = WORKSPACE_BBOX
    out, i, tries = [], 1 + seed_offset, 0
    while len(out) < n_want and tries < 400:
        u, v = halton(i, 2), halton(i, 3)
        x = x0 + u * (x1 - x0)
        y = y0 + v * (y1 - y0)
        i += 1
        tries += 1
        rx, ry = round(x), round(y)           # check constraints on stored coords
        if math.hypot(rx - cup[0], ry - cup[1]) < MIN_DUCK_TO_CUP:
            continue
        if _in_bbox((rx, ry), DUCK_NOGO_BBOX):  # reject the central-bottom block
            continue
        out.append((rx, ry))
    if len(out) < n_want:
        raise RuntimeError(f"Halton pool too small: {len(out)}/{n_want}")
    return out


def main():
    random.seed(SEED)
    trials, tid = [], 0

    for cup_idx, cup in enumerate(CUP_POSITIONS):
        positions = position_pool(cup, N_PER_CUP, cup_idx * 173)
        random.shuffle(positions)

        for px in positions:
            duck_dir = random.uniform(0.0, 360.0)   # fully random orientation
            trials.append({
                "trial_id": tid, "zone": "ID",
                "duck_px": list(px), "cup_px": cup,
                "duck_dir_deg": round(duck_dir, 1),
                "cup_group": cup_idx,
            })
            tid += 1

    proto = {
        "name": PROTOCOL_NAME,
        "version": "v1",
        "description": (
            "30 single-cup ID trials. 6 cup positions × 5 trials. Ducks "
            "Halton-spread inside the green ID zone (id_zone bbox), ≥35 px from "
            "the cup, with fully random duck_dir_deg (no pointing-at-cup "
            "balancing). Half-density spatial sibling of id_scale_60. Seed=13."
        ),
        "task": "pick up the duck and place it in the cup",
        "total_trials": len(trials),
        "id_count": len(trials),
        "ood_count": 0,
        "trials": trials,
        "cup_positions": CUP_POSITIONS,
        "cup_groups": [N_PER_CUP] * len(CUP_POSITIONS),
        "workspace_bbox": list(WORKSPACE_BBOX),
        "duck_nogo_bbox": list(DUCK_NOGO_BBOX),   # central-bottom block, ducks excluded
        "constraints": {
            "min_duck_to_cup": MIN_DUCK_TO_CUP,
            "duck_nogo_for":   ["duck"],
        },
        "_generated_by": "vbti/logic/inference/protocols/generators/make_id_single_cup_30.py",
    }

    OUT_PATH.write_text(json.dumps(proto, indent=2))
    print(f"Wrote {OUT_PATH}  ({len(trials)} trials)")

    # Console sanity: duck-direction octant histogram (should look ~uniform).
    octants = [int(((t["duck_dir_deg"] + 22.5) % 360) // 45) for t in trials]
    oct_names = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    print("\nDuck-direction octant histogram (8 bins, expect ~uniform):")
    for i in range(8):
        n = Counter(octants).get(i, 0)
        print(f"  {i} {oct_names[i]:2}  {n:2}  {'█' * n}")
    print(f"\ncup_groups: {[N_PER_CUP] * len(CUP_POSITIONS)}  (operator repositions cup 6×)")

    render(PROTOCOL_NAME)   # writes renders/id_single_cup_overview.png + _positions.png


if __name__ == "__main__":
    main()
