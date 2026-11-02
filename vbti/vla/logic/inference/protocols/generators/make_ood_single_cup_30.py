"""Generate ood_single_cup_30.json — OOD sibling of id_single_cup_30.

Design (parallel to make_id_single_cup_30.py / make_ood_smoke_15.py):
  - Same 6 cup positions as id_single_cup_30 / id_scale_60.
  - 5 trials per cup × 6 cups = 30.
  - Ducks spawn in the LEFT/RIGHT side bands flanking the green square, never
    inside it. Per cup the L/R split alternates 3L/2R (even cups) and 2L/3R
    (odd cups) → 15 LEFT / 15 RIGHT total.
  - duck_dir_deg is FULLY RANDOM — matches id_single_cup_30 for a clean ID↔OOD
    comparison (contrast ood_smoke_15's 3-yes/2-no pointing balance).
  - BLOCK ZONES at the left/right table extremes keep the duck reachable: ducks
    are Halton-sampled across the FULL table flank (board edge ↔ green edge),
    then rejected if they land inside the extreme block zone. This is the same
    reject-if-inside-bbox approach as the central DUCK_NOGO_BBOX in
    make_id_single_cup_30.py, applied to a zone on each side. Allowed reach =
    ~10.6 cm (~81 px) outward from the green edge (7 cm base + half again).
  - Vertical extent matches the green ID zone's y-range, so the only axis the
    policy hasn't seen is horizontal displacement — clean x-axis OOD ablation.
  - Seed=13, matching the id/ood single-cup + smoke family.

Spatial bounds (from protocols/table_mapping.json, user-confirmed 2026-05-04):
    table.bbox   = (160,  80, 540, 480)  ← full wooden board
    id_zone.bbox = (270, 191, 424, 313)  ← the green square (20 × 16 cm physical)

Pixel-to-cm scale, from id_zone: 154 px / 20 cm ≈ 7.7 px/cm. Reach = 81 px
(7 cm base + half again ≈ 10.6 cm) outward from the green edge.

    LEFT  spawn band: x ∈ [189, 270], y ∈ [191, 313]   (green edge − 81 px)
    RIGHT spawn band: x ∈ [424, 505], y ∈ [191, 313]   (green edge + 81 px)
    LEFT  block zone: x ∈ [160, 189]   (table extreme, ducks rejected)
    RIGHT block zone: x ∈ [505, 540]   (table extreme, ducks rejected)

Run:
  python vbti/logic/inference/protocols/generators/make_ood_single_cup_30.py
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

from vbti.logic.inference.protocols.render_protocol import render_positions

PROTO_DIR = Path(__file__).resolve().parent.parent  # protocols/
OUT_PATH  = PROTO_DIR / "ood_single_cup_30.json"

# Same cup family as id_single_cup_30 / id_scale_60 — direct cross-protocol
# comparison. Cups sit in the cup_strip above the green ID zone.
CUP_POSITIONS = [
    [221, 117], [271, 173], [321, 117],
    [371, 173], [421, 117], [472, 173],
]

# Spatial references, read from table_mapping.json (single source of truth).
_TABLE_MAP     = json.loads((PROTO_DIR / "table_mapping.json").read_text())
WORKSPACE_BBOX = tuple(_TABLE_MAP["id_zone"]["bbox"])  # green square (270,191,424,313)
TABLE_BBOX     = tuple(_TABLE_MAP["table"]["bbox"])    # wooden board (160,80,540,480)

# Sideways reach outward from the green edge. Base is 7 cm (~54 px), broadened
# by +50% (half the base width added toward the table edge) → ~81 px (~10.6 cm).
PX_PER_CM      = 7.66
_BASE_REACH_PX = round(7.0 * PX_PER_CM)                # = 54 px (original 7 cm)
BAND_REACH_PX  = _BASE_REACH_PX + _BASE_REACH_PX // 2  # = 81 px (54 + 27)
BAND_REACH_CM  = round(BAND_REACH_PX / PX_PER_CM, 1)   # = 10.6 cm (reported)

_gx0, _gy0, _gx1, _gy1 = WORKSPACE_BBOX
_tx0, _, _tx1, _ = TABLE_BBOX          # only the table's x-extent matters here

# Sampling supersets — the full table flank on each side (board edge ↔ green).
LEFT_FLANK_BBOX  = (_tx0, _gy0, _gx0, _gy1)            # (160,191,270,313)
RIGHT_FLANK_BBOX = (_gx1, _gy0, _tx1, _gy1)            # (424,191,540,313)

# Block zones at the table extremes — ducks landing here are rejected so they
# never spawn too far from the workspace (same approach as DUCK_NOGO_BBOX).
LEFT_BLOCK_BBOX  = (_tx0,                 _gy0, _gx0 - BAND_REACH_PX, _gy1)  # (160,191,216,313)
RIGHT_BLOCK_BBOX = (_gx1 + BAND_REACH_PX, _gy0, _tx1,                _gy1)  # (478,191,540,313)

# Net allowed spawn bands (flank minus block) — recorded in JSON for clarity.
LEFT_BAND_BBOX   = (_gx0 - BAND_REACH_PX, _gy0, _gx0, _gy1)  # (216,191,270,313)
RIGHT_BAND_BBOX  = (_gx1, _gy0, _gx1 + BAND_REACH_PX, _gy1)  # (424,191,478,313)

N_PER_CUP       = 5
MIN_DUCK_TO_CUP = 35.0
SEED            = 13
PROTOCOL_NAME   = "ood_single_cup_30"


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


def position_pool_in_bbox(cup, n_want, seed_offset, bbox, block_bbox):
    """Halton draw inside `bbox`, rejecting points too close to the cup OR
    inside `block_bbox` (the table-extreme block zone). Same reject-in-bbox
    pattern as make_id_single_cup_30.position_pool, parameterised per side."""
    x0, y0, x1, y1 = bbox
    out, i, tries = [], 1 + seed_offset, 0
    while len(out) < n_want and tries < 400:
        u, v = halton(i, 2), halton(i, 3)
        x = x0 + u * (x1 - x0)
        y = y0 + v * (y1 - y0)
        i += 1
        tries += 1
        rx, ry = round(x), round(y)             # check constraints on stored coords
        if math.hypot(rx - cup[0], ry - cup[1]) < MIN_DUCK_TO_CUP:
            continue
        if _in_bbox((rx, ry), block_bbox):       # reject table-extreme block zone
            continue
        out.append((rx, ry))
    if len(out) < n_want:
        raise RuntimeError(f"Halton pool too small for bbox {bbox}: {len(out)}/{n_want}")
    return out


def main():
    random.seed(SEED)
    trials, tid = [], 0

    for cup_idx, cup in enumerate(CUP_POSITIONS):
        # Alternate the majority side per cup so totals land at 15 LEFT / 15 RIGHT.
        n_left, n_right = (3, 2) if cup_idx % 2 == 0 else (2, 3)

        # Distinct seed offsets per band so left/right draws don't share a Halton
        # sub-sequence (mirrors make_ood_smoke_15's +11 / +97 offsets).
        left_pos  = position_pool_in_bbox(cup, n_left,  cup_idx * 173 + 11,
                                          LEFT_FLANK_BBOX,  LEFT_BLOCK_BBOX)
        right_pos = position_pool_in_bbox(cup, n_right, cup_idx * 173 + 97,
                                          RIGHT_FLANK_BBOX, RIGHT_BLOCK_BBOX)

        tagged = [(p, "LEFT") for p in left_pos] + [(p, "RIGHT") for p in right_pos]
        random.shuffle(tagged)

        for px, band in tagged:
            duck_dir = random.uniform(0.0, 360.0)   # fully random orientation
            trials.append({
                "trial_id": tid, "zone": "OOD", "subzone": band,
                "duck_px": list(px), "cup_px": cup,
                "duck_dir_deg": round(duck_dir, 1),
                "cup_group": cup_idx,
            })
            tid += 1

    proto = {
        "name": PROTOCOL_NAME,
        "version": "v1",
        "description": (
            "30 single-cup OOD trials — sibling of id_single_cup_30. 6 cup "
            "positions × 5 trials. Ducks spawn in the LEFT/RIGHT side bands "
            "flanking the green ID zone (15 LEFT / 15 RIGHT), with fully random "
            "duck_dir_deg. Block zones at the table extremes cap the sideways "
            f"reach at {BAND_REACH_CM:.1f} cm (≈{BAND_REACH_PX} px @ "
            f"{PX_PER_CM:.2f} px/cm) so the duck never spawns too far. Vertical "
            "extent matches the ID zone → clean horizontal-displacement OOD. "
            "Seed=13."
        ),
        "task": "pick up the duck and place it in the cup",
        "total_trials": len(trials),
        "id_count": 0,
        "ood_count": len(trials),
        "trials": trials,
        "cup_positions": CUP_POSITIONS,
        "cup_groups": [N_PER_CUP] * len(CUP_POSITIONS),
        "workspace_bbox": list(WORKSPACE_BBOX),
        "ood_bands": {"LEFT": list(LEFT_BAND_BBOX),
                      "RIGHT": list(RIGHT_BAND_BBOX)},
        "duck_block_bands": {"LEFT": list(LEFT_BLOCK_BBOX),
                             "RIGHT": list(RIGHT_BLOCK_BBOX)},
        "band_reach_cm": BAND_REACH_CM,
        "band_reach_px": BAND_REACH_PX,
        "px_per_cm": PX_PER_CM,
        "constraints": {"min_duck_to_cup": MIN_DUCK_TO_CUP,
                        "duck_block_for": ["duck"]},
        "_generated_by": "vbti/logic/inference/protocols/generators/make_ood_single_cup_30.py",
    }

    OUT_PATH.write_text(json.dumps(proto, indent=2))
    print(f"Wrote {OUT_PATH}  ({len(trials)} trials)")

    # Console sanity: duck-direction octants + band split.
    octants, by_band = [], Counter()
    for t in trials:
        octants.append(int(((t["duck_dir_deg"] + 22.5) % 360) // 45))
        by_band[t["subzone"]] += 1

    oct_names = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    print("\nDuck-direction octant histogram (8 bins, expect ~uniform):")
    for i in range(8):
        n = Counter(octants).get(i, 0)
        print(f"  {i} {oct_names[i]:2}  {n:2}  {'█' * n}")
    print(f"\nband split: LEFT={by_band['LEFT']}  RIGHT={by_band['RIGHT']}  (target 15L / 15R)")
    print(f"cup_groups: {[N_PER_CUP] * len(CUP_POSITIONS)}  (operator repositions cup 6×)")

    # OOD ducks sit OUTSIDE the green square, so the per-trial overview (zoomed
    # to the workspace) isn't informative — render only the full-frame positions
    # plot, where the side bands + empty blocked extremes are visible.
    render_positions(PROTOCOL_NAME)


if __name__ == "__main__":
    main()
