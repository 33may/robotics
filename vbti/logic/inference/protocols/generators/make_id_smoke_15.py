"""Generate id_smoke_15.json — 15-trial smoke version of id_scale_60.

Design:
  - 3 cup positions (C0, C2, C4 — the top row of id_scale_60) × 5 trials = 15.
  - Halton-jittered duck positions inside workspace bbox, ≥35 px from cup.
  - Per cup: 3 trials with pointing_at_cup=yes (|pterr| ≤ 30°),
            2 trials with pointing_at_cup=no  (|pterr| ∈ [60°, 180°]).
  - Cup indices in the Halton seed-offset are the *original* id_scale_60 indices
    (0, 2, 4), not (0, 1, 2). This makes the smoke's duck positions a true
    pixel-space subset of id_scale_60 — passing the smoke is a real predictor
    for the corresponding subset of the full 60-trial eval.
  - Seed=13, matching id_scale_60.

Run:
  python vbti/logic/inference/protocols/generators/make_id_smoke_15.py
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

PROTO_DIR = Path(__file__).resolve().parent.parent  # protocols/
OUT_PATH  = PROTO_DIR / "id_smoke_15.json"

# Subset of id_scale_60's CUP_POSITIONS — keep original indices so the Halton
# seed_offset (cup_idx * 173) reproduces a subset of id_scale_60's duck pool.
ALL_CUP_POSITIONS = [
    [221, 117], [271, 173], [321, 117],
    [371, 173], [421, 117], [472, 173],
]
SMOKE_CUP_INDICES = [0, 2, 4]                          # top-row cups: C0, C2, C4
CUP_POSITIONS     = [ALL_CUP_POSITIONS[i] for i in SMOKE_CUP_INDICES]

WORKSPACE_BBOX    = (270, 191, 424, 313)               # x0, y0, x1, y1
N_PER_CUP         = 5
N_YES_PER_CUP     = 3
N_NO_PER_CUP      = 2
MIN_DUCK_TO_CUP   = 35.0
SEED              = 13
PROTOCOL_NAME     = "id_smoke_15"


def halton(idx: int, base: int) -> float:
    f, r = 1.0, 0.0
    while idx > 0:
        f /= base
        r += f * (idx % base)
        idx //= base
    return r


def position_pool(cup, n_want, seed_offset):
    x0, y0, x1, y1 = WORKSPACE_BBOX
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
        raise RuntimeError(f"Halton pool too small: {len(out)}/{n_want}")
    return out


def fold(angle):
    a = angle % 360.0
    return a - 360.0 if a > 180.0 else a


def main():
    random.seed(SEED)
    trials, tid = [], 0

    for cup_orig_idx, cup in zip(SMOKE_CUP_INDICES, CUP_POSITIONS):
        # Use the ORIGINAL cup index in the seed offset so we draw from the
        # same Halton sequence id_scale_60 used for this cup.
        positions = position_pool(cup, N_PER_CUP, cup_orig_idx * 173)
        random.shuffle(positions)
        yes_pos = positions[:N_YES_PER_CUP]
        no_pos  = positions[N_YES_PER_CUP:N_YES_PER_CUP + N_NO_PER_CUP]

        for px in yes_pos:
            d2c = math.degrees(math.atan2(cup[1] - px[1], cup[0] - px[0])) % 360.0
            duck_dir = (d2c + random.uniform(-30.0, 30.0)) % 360.0
            trials.append({
                "trial_id": tid, "zone": "ID",
                "duck_px": list(px), "cup_px": cup,
                "duck_dir_deg": round(duck_dir, 1),
                "cup_group": cup_orig_idx,        # keep original group label
            })
            tid += 1

        for px in no_pos:
            d2c = math.degrees(math.atan2(cup[1] - px[1], cup[0] - px[0])) % 360.0
            mag  = random.uniform(60.0, 180.0)
            sign = random.choice([-1, 1])
            duck_dir = (d2c + sign * mag) % 360.0
            trials.append({
                "trial_id": tid, "zone": "ID",
                "duck_px": list(px), "cup_px": cup,
                "duck_dir_deg": round(duck_dir, 1),
                "cup_group": cup_orig_idx,
            })
            tid += 1

    proto = {
        "name": PROTOCOL_NAME,
        "version": "v1",
        "description": (
            "15-trial smoke eval — top-row subset of id_scale_60. "
            "3 cup positions (C0/C2/C4) × 5 trials. Halton duck positions "
            "(same sequence as id_scale_60 for these cups), 3 yes / 2 no per cup. "
            "Seed=13."
        ),
        "task": "pick up the duck and place it in the cup",
        "total_trials": len(trials),
        "id_count": len(trials),
        "ood_count": 0,
        "trials": trials,
        "cup_positions": CUP_POSITIONS,
        "cup_groups": [N_PER_CUP] * len(CUP_POSITIONS),
        "_generated_by": "vbti/logic/inference/protocols/generators/make_id_smoke_15.py",
    }

    OUT_PATH.write_text(json.dumps(proto, indent=2))
    print(f"Wrote {OUT_PATH}  ({len(trials)} trials)")

    # Console sanity histograms
    octants, pterrs = [], []
    for t in trials:
        d2c = math.degrees(math.atan2(
            t["cup_px"][1] - t["duck_px"][1],
            t["cup_px"][0] - t["duck_px"][0],
        )) % 360.0
        pterrs.append(fold(t["duck_dir_deg"] - d2c))
        octants.append(int(((t["duck_dir_deg"] + 22.5) % 360) // 45))

    oct_names = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    print("\nOctant histogram (8 bins):")
    for i in range(8):
        n = Counter(octants).get(i, 0)
        print(f"  {i} {oct_names[i]:2}  {n:2}  {'█' * n}")
    yes = sum(1 for p in pterrs if abs(p) <= 45)
    print(f"\npointing_at_cup: {yes}/{len(pterrs)} yes  (target=9, enforced 3/cup)")


if __name__ == "__main__":
    main()
