"""Generate dual_cup_15.json — 15-trial checkpoint-sweep variant of dual_cup_30.

Same workspace, schema, and constraints as dual_cup_30. Used to compare multiple
checkpoints (e.g. v019 @ 15k / 20k / 25k / 35k / 50k) cheaply: 15 trials × 5 ckpts
= 75 trials, ~30–45 min on the robot.

Scene mix (15 total):
  - 5 trials: ONLY red cup     (single-cup baseline, prompt = red)
  - 5 trials: BOTH cups        (dual-cup language disambiguation)
  - 5 trials: ONLY black cup   (single-cup baseline, prompt = black)

Dual-cup cell allocation (5 trials — counterbalance is approximate at n=5):
  - target_color: 3 red / 2 black
  - target_side : 2 left / 3 right
  - target_closer: 3 True / 2 False
That's the closest split to 50/50 achievable with 5 cells.

Run:
  python vbti/logic/inference/protocols/generators/make_dual_cup_15.py
"""

import json
import math
import random
from collections import Counter
from pathlib import Path
from vbti.logic.inference.protocols.render_protocol import render

PROTO_DIR = Path(__file__).resolve().parent.parent
OUT_PATH  = PROTO_DIR / "dual_cup_15.json"

# Read canonical table bbox from table_mapping.json — single source of truth
_TABLE_MAP   = json.loads((PROTO_DIR / "table_mapping.json").read_text())
_TABLE_BBOX  = _TABLE_MAP["table"]["bbox"]
_INSET_PCT   = 0.10


def _inset_bbox(bbox, pct: float) -> tuple:
    x0, y0, x1, y1 = bbox
    pad_x = round((x1 - x0) * pct)
    pad_y = round((y1 - y0) * pct)
    return (x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y)


WORKSPACE_BBOX  = _inset_bbox(_TABLE_BBOX, _INSET_PCT)

_DUCK_BOTTOM_PAD_PCT = 0.15
_CUP_BOTTOM_PAD_PCT  = 0.20
_ws_x0, _ws_y0, _ws_x1, _ws_y1 = WORKSPACE_BBOX
_ws_h = _ws_y1 - _ws_y0
DUCK_BBOX = (_ws_x0, _ws_y0, _ws_x1,
             _ws_y0 + round(_ws_h * (1.0 - _DUCK_BOTTOM_PAD_PCT)))
CUP_BBOX  = (_ws_x0, _ws_y0, _ws_x1,
             _ws_y0 + round(_ws_h * (1.0 - _CUP_BOTTOM_PAD_PCT)))

ROBOT_NOGO_BBOX = (270, 250, 430, 480)
DUCK_NOGO_BBOX  = (300, 360, 400, 480)
MIN_CUP_TO_CUP  = 80.0
MIN_DUCK_TO_CUP = 35.0
SEED            = 15
PROTOCOL_NAME   = "dual_cup_15"
TASK_TEMPLATE   = "Pick up the duck and place it in the {color} cup"

# Dual-cup subset (5 trials) — best 50/50 balance possible at n=5.
# (target_color, target_side, target_closer): count
DUAL_CELLS = [
    ("red",   "left",  True),
    ("red",   "right", False),
    ("black", "left",  False),
    ("black", "right", True),
    ("red",   "right", True),
]
assert len(DUAL_CELLS) == 5

N_SINGLE_RED   = 5
N_SINGLE_BLACK = 5
N_DUAL         = 5
assert N_SINGLE_RED + N_SINGLE_BLACK + N_DUAL == 15


# ── Sampling helpers ────────────────────────────────────────────────────────

def _rand_in_bbox(rng, bbox=WORKSPACE_BBOX):
    x0, y0, x1, y1 = bbox
    return float(round(rng.uniform(x0, x1))), float(round(rng.uniform(y0, y1)))


def _in_bbox(px, bbox) -> bool:
    rx0, ry0, rx1, ry1 = bbox
    return rx0 <= px[0] <= rx1 and ry0 <= px[1] <= ry1


def _in_robot_zone(px) -> bool:
    return _in_bbox(px, ROBOT_NOGO_BBOX)


def _in_duck_zone(px) -> bool:
    return _in_bbox(px, DUCK_NOGO_BBOX)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sample_single(rng, max_tries: int = 1000):
    for _ in range(max_tries):
        cup  = _rand_in_bbox(rng, CUP_BBOX)
        if _in_robot_zone(cup): continue
        duck = _rand_in_bbox(rng, DUCK_BBOX)
        if _in_duck_zone(duck): continue
        if _dist(duck, cup) < MIN_DUCK_TO_CUP:
            continue
        return cup, duck, rng.uniform(0.0, 360.0)
    raise RuntimeError("Failed to sample single-cup layout")


def _sample_dual(rng, target_side: str, target_closer: bool,
                 max_tries: int = 2000):
    for _ in range(max_tries):
        cup1 = _rand_in_bbox(rng, CUP_BBOX)
        if _in_robot_zone(cup1): continue
        cup2 = _rand_in_bbox(rng, CUP_BBOX)
        if _in_robot_zone(cup2): continue
        duck = _rand_in_bbox(rng, DUCK_BBOX)
        if _in_duck_zone(duck): continue
        if _dist(cup1, cup2) < MIN_CUP_TO_CUP:
            continue
        if _dist(duck, cup1) < MIN_DUCK_TO_CUP or _dist(duck, cup2) < MIN_DUCK_TO_CUP:
            continue
        if target_side == "left":
            tgt, dis = (cup1, cup2) if cup1[0] < cup2[0] else (cup2, cup1)
        else:
            tgt, dis = (cup1, cup2) if cup1[0] > cup2[0] else (cup2, cup1)
        if target_closer and _dist(duck, tgt) >= _dist(duck, dis):
            continue
        if (not target_closer) and _dist(duck, tgt) <= _dist(duck, dis):
            continue
        return tgt, dis, duck, rng.uniform(0.0, 360.0)
    raise RuntimeError(
        f"Failed dual-cup sample: side={target_side} closer={target_closer}")


# ── Trial builders ──────────────────────────────────────────────────────────

def _make_single_trial(tid: int, rng, target_color: str) -> dict:
    cup, duck, ddir = _sample_single(rng)
    return {
        "trial_id": tid,
        "zone":     "ID",
        "entities": [
            {
                "name":    "duck", "kind": "duck", "color": "yellow",
                "px":      [round(duck[0]), round(duck[1])],
                "dir_deg": round(ddir, 1),
            },
            {
                "name":  "A", "kind": "cup", "color": target_color,
                "px":    [round(cup[0]), round(cup[1])],
            },
        ],
        "target": "A",
        "task":   TASK_TEMPLATE.format(color=target_color),
        "tags": {
            "scene":         f"single_{target_color}",
            "target_color":  target_color,
        },
    }


def _make_dual_trial(tid: int, rng, target_color: str,
                     target_side: str, target_closer: bool) -> dict:
    distractor_color = "black" if target_color == "red" else "red"
    tgt, dis, duck, ddir = _sample_dual(rng, target_side, target_closer)
    return {
        "trial_id": tid,
        "zone":     "ID",
        "entities": [
            {
                "name":    "duck", "kind": "duck", "color": "yellow",
                "px":      [round(duck[0]), round(duck[1])],
                "dir_deg": round(ddir, 1),
            },
            {
                "name":  "A", "kind": "cup", "color": target_color,
                "px":    [round(tgt[0]), round(tgt[1])],
            },
            {
                "name":  "B", "kind": "cup", "color": distractor_color,
                "px":    [round(dis[0]), round(dis[1])],
            },
        ],
        "target": "A",
        "task":   TASK_TEMPLATE.format(color=target_color),
        "tags": {
            "scene":         "both",
            "target_color":  target_color,
            "target_side":   target_side,
            "target_closer": target_closer,
        },
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)
    trials = []
    tid = 0

    # Cup-swap-minimising order:
    #   1. single_red  — only red cup on table
    #   2. both        — operator ADDS the black cup (1 swap-op)
    #   3. single_black — operator REMOVES the red cup (1 swap-op)

    for _ in range(N_SINGLE_RED):
        trials.append(_make_single_trial(tid, rng, "red"))
        tid += 1

    for cell in DUAL_CELLS:
        target_color, target_side, target_closer = cell
        trials.append(_make_dual_trial(
            tid, rng, target_color, target_side, target_closer))
        tid += 1

    for _ in range(N_SINGLE_BLACK):
        trials.append(_make_single_trial(tid, rng, "black"))
        tid += 1

    proto = {
        "name":          PROTOCOL_NAME,
        "version":       "v1",
        "schema":        "entities",
        "task_template": TASK_TEMPLATE,
        "description": (
            f"15 trials, mixed scenes: {N_SINGLE_RED} only-red, "
            f"{N_SINGLE_BLACK} only-black, {N_DUAL} both-cups. "
            "Checkpoint-sweep variant of dual_cup_30 — same workspace, schema, "
            "and constraints, smaller n for cheap repeated robot eval. "
            "Dual-cup cells approximately balance color×side×closer at n=5. "
            f"Seed={SEED}."
        ),
        "task":          "VARIES_PER_TRIAL — see trial.task",
        "total_trials":  len(trials),
        "id_count":      len(trials),
        "ood_count":     0,
        "trials":        trials,
        "workspace_bbox":  list(WORKSPACE_BBOX),
        "cup_bbox":        list(CUP_BBOX),
        "duck_bbox":       list(DUCK_BBOX),
        "robot_nogo_bbox": list(ROBOT_NOGO_BBOX),
        "duck_nogo_bbox":  list(DUCK_NOGO_BBOX),
        "constraints": {
            "min_cup_to_cup":      MIN_CUP_TO_CUP,
            "min_duck_to_cup":     MIN_DUCK_TO_CUP,
            "cup_bottom_pad_pct":  _CUP_BOTTOM_PAD_PCT,
            "duck_bottom_pad_pct": _DUCK_BOTTOM_PAD_PCT,
            "robot_nogo_for":      ["cup"],
            "duck_nogo_for":       ["duck"],
        },
        "scene_counts": {
            "single_red":   N_SINGLE_RED,
            "single_black": N_SINGLE_BLACK,
            "both":         N_DUAL,
        },
        "_generated_by": "vbti/logic/inference/protocols/generators/make_dual_cup_15.py",
    }

    OUT_PATH.write_text(json.dumps(proto, indent=2))
    print(f"Wrote {OUT_PATH}  ({len(trials)} trials)")

    scene_h = Counter(t["tags"]["scene"]        for t in trials)
    color_h = Counter(t["tags"]["target_color"] for t in trials)
    print(f"  scene         : {dict(scene_h)}")
    print(f"  target_color  : {dict(color_h)}")

    duals = [t for t in trials if t["tags"]["scene"] == "both"]
    side_h  = Counter(t["tags"]["target_side"]   for t in duals)
    close_h = Counter(t["tags"]["target_closer"] for t in duals)
    color_dual_h = Counter(t["tags"]["target_color"] for t in duals)
    print(f"  dual side     : {dict(side_h)}")
    print(f"  dual closer   : {dict(close_h)}")
    print(f"  dual color    : {dict(color_dual_h)}")

    render("dual_cup_15")


if __name__ == "__main__":
    main()
