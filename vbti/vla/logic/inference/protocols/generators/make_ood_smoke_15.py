"""Generate ood_smoke_15.json — OOD sibling of id_smoke_15.

Design (parallel to make_id_smoke_15.py):
  - Same 3 cup positions (C0/C2/C4 — top row of id_scale_60).
  - Same 5 trials per cup × 3 cups = 15.
  - Same yes/no direction balance (3 yes / 2 no per cup).
  - DIFFERENT spawn zone: ducks spawn LEFT or RIGHT of the green workspace,
    never inside it. Per cup, 3 left-band + 2 right-band → 9L/6R total.
  - Vertical extent of OOD bands matches the ID zone's y-range so the only
    axis the policy hasn't seen is horizontal displacement — clean ablation.
  - Seed=13, matching the id_smoke_15 / id_scale_60 family.

Spatial bounds (from protocols/table_mapping.json, user-confirmed 2026-05-04):
    table.bbox  = (160,  80, 540, 480)  ← full wooden board
    id_zone.bbox = (270, 191, 424, 313)  ← the green square (20 × 16 cm physical)

Pixel-to-cm scale, derived from id_zone: 154 px / 20 cm = 7.70 px/cm horizontal,
                                          122 px / 16 cm = 7.625 px/cm vertical
                                          → use 7.66 px/cm → 3 cm ≈ 23 px

A TABLE_MARGIN_PX inset of 23 px keeps ducks ≥3 cm from the wooden board edge
so the operator never has to place the duck where it would fall off. Only the
table-touching side of each band is inset; the workspace-facing side is
unchanged (it's already 110+ px clear of any table edge).

    LEFT  OOD band: x ∈ [183, 270], y ∈ [191, 313]   (87 × 122 px)
    RIGHT OOD band: x ∈ [424, 517], y ∈ [191, 313]   (93 × 122 px)

Run:
  python vbti/logic/inference/protocols/generators/make_ood_smoke_15.py
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

PROTO_DIR = Path(__file__).resolve().parent.parent  # protocols/
OUT_PATH  = PROTO_DIR / "ood_smoke_15.json"

# Same cup family as id_smoke_15 so cross-protocol comparison is direct.
ALL_CUP_POSITIONS = [
    [221, 117], [271, 173], [321, 117],
    [371, 173], [421, 117], [472, 173],
]
SMOKE_CUP_INDICES = [0, 2, 4]                          # top-row cups
CUP_POSITIONS     = [ALL_CUP_POSITIONS[i] for i in SMOKE_CUP_INDICES]

# Workspace stays as reference (drawn green in the live overlay), but ducks
# spawn in the two flanking bands instead of inside it.
WORKSPACE_BBOX    = (270, 191, 424, 313)               # x0, y0, x1, y1

# 3 cm safety margin from the wooden board edge — ID zone calibration:
# 154 px / 20 cm ≈ 7.7 px/cm horizontally → 3 cm ≈ 23 px. Only the
# table-touching side of each band is inset (workspace-facing side already
# has plenty of clearance).
PX_PER_CM         = 7.66
TABLE_MARGIN_CM   = 3.0
TABLE_MARGIN_PX   = round(TABLE_MARGIN_CM * PX_PER_CM)  # = 23 px

LEFT_BAND_BBOX    = (160 + TABLE_MARGIN_PX, 191, 270,                       313)
RIGHT_BAND_BBOX   = (424,                   191, 540 - TABLE_MARGIN_PX,     313)

N_PER_CUP         = 5
N_LEFT_PER_CUP    = 3
N_RIGHT_PER_CUP   = 2
N_YES_PER_CUP     = 3
N_NO_PER_CUP      = 2
MIN_DUCK_TO_CUP   = 35.0
SEED              = 13
PROTOCOL_NAME     = "ood_smoke_15"


def halton(idx: int, base: int) -> float:
    f, r = 1.0, 0.0
    while idx > 0:
        f /= base
        r += f * (idx % base)
        idx //= base
    return r


def position_pool_in_bbox(cup, n_want, seed_offset, bbox):
    """Halton draw inside an arbitrary bbox, with MIN_DUCK_TO_CUP rejection.

    Identical logic to id_smoke_15's `position_pool`, but parameterised on
    the bbox so we can draw separately from LEFT vs RIGHT bands.
    """
    x0, y0, x1, y1 = bbox
    out, i, tries = [], 1 + seed_offset, 0
    while len(out) < n_want and tries < 200:
        u, v = halton(i, 2), halton(i, 3)
        x = x0 + u * (x1 - x0)
        y = y0 + v * (y1 - y0)
        i += 1
        tries += 1
        if math.hypot(x - cup[0], y - cup[1]) < MIN_DUCK_TO_CUP:
            continue
        out.append((round(x), round(y)))
    if len(out) < n_want:
        raise RuntimeError(f"Halton pool too small for bbox {bbox}: {len(out)}/{n_want}")
    return out


def fold(angle):
    a = angle % 360.0
    return a - 360.0 if a > 180.0 else a


def main():
    random.seed(SEED)
    trials, tid = [], 0

    for cup_orig_idx, cup in zip(SMOKE_CUP_INDICES, CUP_POSITIONS):
        # Distinct seed offsets per band so left/right draws don't overlap each
        # other's sequences. Offsets large enough to be visibly different from
        # id_smoke_15's `cup_idx * 173`.
        left_pos  = position_pool_in_bbox(cup, N_LEFT_PER_CUP,
                                          cup_orig_idx * 173 + 11, LEFT_BAND_BBOX)
        right_pos = position_pool_in_bbox(cup, N_RIGHT_PER_CUP,
                                          cup_orig_idx * 173 + 97, RIGHT_BAND_BBOX)

        # Tag each position with its band so we can record `subzone` later,
        # then shuffle the combined pool and assign the first 3 as YES, last 2
        # as NO (mirrors id_smoke_15's structure).
        tagged = [(p, "LEFT") for p in left_pos] + [(p, "RIGHT") for p in right_pos]
        random.shuffle(tagged)
        yes_pool = tagged[:N_YES_PER_CUP]
        no_pool  = tagged[N_YES_PER_CUP:N_YES_PER_CUP + N_NO_PER_CUP]

        for px, band in yes_pool:
            d2c = math.degrees(math.atan2(cup[1] - px[1], cup[0] - px[0])) % 360.0
            duck_dir = (d2c + random.uniform(-30.0, 30.0)) % 360.0
            trials.append({
                "trial_id": tid, "zone": "OOD", "subzone": band,
                "duck_px": list(px), "cup_px": cup,
                "duck_dir_deg": round(duck_dir, 1),
                "cup_group": cup_orig_idx,
            })
            tid += 1

        for px, band in no_pool:
            d2c = math.degrees(math.atan2(cup[1] - px[1], cup[0] - px[0])) % 360.0
            mag  = random.uniform(60.0, 180.0)
            sign = random.choice([-1, 1])
            duck_dir = (d2c + sign * mag) % 360.0
            trials.append({
                "trial_id": tid, "zone": "OOD", "subzone": band,
                "duck_px": list(px), "cup_px": cup,
                "duck_dir_deg": round(duck_dir, 1),
                "cup_group": cup_orig_idx,
            })
            tid += 1

    proto = {
        "name": PROTOCOL_NAME,
        "version": "v1",
        "description": (
            "15-trial OOD smoke — siblings of id_smoke_15. Ducks spawn in two "
            "side bands flanking the workspace (3 LEFT + 2 RIGHT per cup); cups "
            "and prompt unchanged. Tests horizontal-displacement OOD with y "
            f"matched to the ID zone so failure modes isolate the x-axis. "
            f"{TABLE_MARGIN_CM:.0f} cm safety margin from wooden board edge "
            f"(≈{TABLE_MARGIN_PX} px @ {PX_PER_CM:.2f} px/cm). Seed=13."
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
        "table_margin_cm": TABLE_MARGIN_CM,
        "table_margin_px": TABLE_MARGIN_PX,
        "px_per_cm": PX_PER_CM,
        "_generated_by": "vbti/logic/inference/protocols/generators/make_ood_smoke_15.py",
    }

    OUT_PATH.write_text(json.dumps(proto, indent=2))
    print(f"Wrote {OUT_PATH}  ({len(trials)} trials)")

    # Console sanity histograms
    octants, pterrs, by_band = [], [], Counter()
    for t in trials:
        d2c = math.degrees(math.atan2(
            t["cup_px"][1] - t["duck_px"][1],
            t["cup_px"][0] - t["duck_px"][0],
        )) % 360.0
        pterrs.append(fold(t["duck_dir_deg"] - d2c))
        octants.append(int(((t["duck_dir_deg"] + 22.5) % 360) // 45))
        by_band[t["subzone"]] += 1

    oct_names = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    print("\nOctant histogram (8 bins):")
    for i in range(8):
        n = Counter(octants).get(i, 0)
        print(f"  {i} {oct_names[i]:2}  {n:2}  {'█' * n}")
    yes = sum(1 for p in pterrs if abs(p) <= 45)
    print(f"\npointing_at_cup: {yes}/{len(pterrs)} yes  (target=9, enforced 3/cup)")
    print(f"band split: LEFT={by_band['LEFT']}  RIGHT={by_band['RIGHT']}  (target 9L / 6R)")


if __name__ == "__main__":
    main()
