"""Generate id_scale_60.json — 60 ID trials, balanced for v013 scale eval.

Design:
  - 6 cup positions × 10 trials = 60.
  - Halton-jittered duck positions inside workspace bbox, ≥35 px from cup.
  - Per cup: 5 trials with pointing_at_cup=yes (|pterr| ≤ 30°),
            5 trials with pointing_at_cup=no  (|pterr| ∈ [60°, 180°]).
  - Trials grouped by cup_group (operator only repositions cup 5 times).
  - Seed=13 for v013, deterministic.

Run:
  python vbti/logic/inference/protocols/generators/make_id_scale_60.py
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

PROTO_DIR = Path(__file__).resolve().parent.parent  # protocols/
OUT_PATH  = PROTO_DIR / "id_scale_60.json"

CUP_POSITIONS = [
    [221, 117], [271, 173], [321, 117],
    [371, 173], [421, 117], [472, 173],
]
WORKSPACE_BBOX   = (270, 191, 424, 313)  # x0, y0, x1, y1
N_PER_CUP        = 10
N_YES_PER_CUP    = 5
N_NO_PER_CUP     = 5
MIN_DUCK_TO_CUP  = 35.0
SEED             = 13
PROTOCOL_NAME    = "id_scale_60"


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

    for cup_idx, cup in enumerate(CUP_POSITIONS):
        positions = position_pool(cup, N_PER_CUP, cup_idx * 173)
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
                "cup_group": cup_idx,
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
                "cup_group": cup_idx,
            })
            tid += 1

    proto = {
        "name": PROTOCOL_NAME,
        "version": "v1",
        "description": (
            "60 ID trials for v013 scale baseline. 6 cup positions × 10 trials. "
            "Halton-spread duck positions, balanced pointing_at_cup (5 yes / 5 no per cup). "
            "Seed=13."
        ),
        "task": "pick up the duck and place it in the cup",
        "total_trials": len(trials),
        "id_count": len(trials),
        "ood_count": 0,
        "trials": trials,
        "cup_positions": CUP_POSITIONS,
        "cup_groups": [N_PER_CUP] * len(CUP_POSITIONS),
        "_generated_by": "vbti/logic/inference/protocols/generators/make_id_scale_60.py",
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
    print(f"\npointing_at_cup: {yes}/{len(pterrs)} yes  (target=30, enforced 5/cup)")


if __name__ == "__main__":
    main()
