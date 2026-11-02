"""Generate checkpoint_sweep.json — 20-trial balanced two-cup checkpoint selector.

Scene mix:
  - 5 only-red trials
  - 5 both-cups, target red
  - 5 both-cups, target black
  - 5 only-black trials
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

from vbti.logic.inference.protocols.render_protocol import render


PROTO_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROTO_DIR / "checkpoint_sweep.json"
TASK_TEMPLATE = "Pick up the duck and place it in the {color} cup"
PROTOCOL_NAME = "checkpoint_sweep"
SEED = 21

_TABLE_MAP = json.loads((PROTO_DIR / "table_mapping.json").read_text())
_TABLE_BBOX = _TABLE_MAP["table"]["bbox"]
_INSET_PCT = 0.10


def _inset_bbox(bbox, pct):
    x0, y0, x1, y1 = bbox
    return (
        x0 + round((x1 - x0) * pct),
        y0 + round((y1 - y0) * pct),
        x1 - round((x1 - x0) * pct),
        y1 - round((y1 - y0) * pct),
    )


WORKSPACE_BBOX = _inset_bbox(_TABLE_BBOX, _INSET_PCT)
_ws_x0, _ws_y0, _ws_x1, _ws_y1 = WORKSPACE_BBOX
_ws_h = _ws_y1 - _ws_y0
DUCK_BBOX = (_ws_x0, _ws_y0, _ws_x1, _ws_y0 + round(_ws_h * 0.85))
CUP_BBOX = (_ws_x0, _ws_y0, _ws_x1, _ws_y0 + round(_ws_h * 0.80))
ROBOT_NOGO_BBOX = (270, 250, 430, 480)
DUCK_NOGO_BBOX = (300, 360, 400, 480)
MIN_CUP_TO_CUP = 80.0
MIN_DUCK_TO_CUP = 35.0

DUAL_CELLS = [
    ("red", "left", True),
    ("red", "left", False),
    ("red", "right", True),
    ("red", "right", False),
    ("red", "right", True),
    ("black", "left", True),
    ("black", "left", False),
    ("black", "right", True),
    ("black", "right", False),
    ("black", "left", False),
]


def _rand_in_bbox(rng, bbox):
    x0, y0, x1, y1 = bbox
    return float(round(rng.uniform(x0, x1))), float(round(rng.uniform(y0, y1)))


def _in_bbox(px, bbox):
    return bbox[0] <= px[0] <= bbox[2] and bbox[1] <= px[1] <= bbox[3]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sample_single(rng):
    for _ in range(1000):
        cup = _rand_in_bbox(rng, CUP_BBOX)
        duck = _rand_in_bbox(rng, DUCK_BBOX)
        if _in_bbox(cup, ROBOT_NOGO_BBOX) or _in_bbox(duck, DUCK_NOGO_BBOX):
            continue
        if _dist(duck, cup) >= MIN_DUCK_TO_CUP:
            return cup, duck, rng.uniform(0.0, 360.0)
    raise RuntimeError("Failed to sample single-cup layout")


def _sample_dual(rng, target_side, target_closer):
    for _ in range(3000):
        cup1 = _rand_in_bbox(rng, CUP_BBOX)
        cup2 = _rand_in_bbox(rng, CUP_BBOX)
        duck = _rand_in_bbox(rng, DUCK_BBOX)
        if _in_bbox(cup1, ROBOT_NOGO_BBOX) or _in_bbox(cup2, ROBOT_NOGO_BBOX) or _in_bbox(duck, DUCK_NOGO_BBOX):
            continue
        if _dist(cup1, cup2) < MIN_CUP_TO_CUP or _dist(duck, cup1) < MIN_DUCK_TO_CUP or _dist(duck, cup2) < MIN_DUCK_TO_CUP:
            continue
        target, distractor = (cup1, cup2) if (cup1[0] < cup2[0]) == (target_side == "left") else (cup2, cup1)
        if target_closer != (_dist(duck, target) < _dist(duck, distractor)):
            continue
        return target, distractor, duck, rng.uniform(0.0, 360.0)
    raise RuntimeError("Failed to sample dual-cup layout")


def _single_trial(tid, rng, color):
    cup, duck, direction = _sample_single(rng)
    return {
        "trial_id": tid,
        "zone": "ID",
        "entities": [
            {"name": "duck", "kind": "duck", "color": "yellow", "px": [round(duck[0]), round(duck[1])], "dir_deg": round(direction, 1)},
            {"name": "A", "kind": "cup", "color": color, "px": [round(cup[0]), round(cup[1])]},
        ],
        "target": "A",
        "task": TASK_TEMPLATE.format(color=color),
        "tags": {"scene": f"single_{color}", "target_color": color},
    }


def _dual_trial(tid, rng, color, side, closer):
    target, distractor, duck, direction = _sample_dual(rng, side, closer)
    distractor_color = "black" if color == "red" else "red"
    return {
        "trial_id": tid,
        "zone": "ID",
        "entities": [
            {"name": "duck", "kind": "duck", "color": "yellow", "px": [round(duck[0]), round(duck[1])], "dir_deg": round(direction, 1)},
            {"name": "A", "kind": "cup", "color": color, "px": [round(target[0]), round(target[1])]},
            {"name": "B", "kind": "cup", "color": distractor_color, "px": [round(distractor[0]), round(distractor[1])]},
        ],
        "target": "A",
        "task": TASK_TEMPLATE.format(color=color),
        "tags": {"scene": f"both_{color}", "target_color": color, "target_side": side, "target_closer": closer},
    }


def main():
    rng = random.Random(SEED)
    trials = []
    for _ in range(5):
        trials.append(_single_trial(len(trials), rng, "red"))
    for color, side, closer in DUAL_CELLS:
        trials.append(_dual_trial(len(trials), rng, color, side, closer))
    for _ in range(5):
        trials.append(_single_trial(len(trials), rng, "black"))

    OUT_PATH.write_text(json.dumps({
        "name": PROTOCOL_NAME,
        "version": "v4",
        "schema": "entities",
        "task_template": TASK_TEMPLATE,
        "description": "20-trial balanced checkpoint sweep: 5 only-red, 5 both target-red, 5 both target-black, 5 only-black. Random positions with dual-cup balance on color, side, and closer. Seed=21.",
        "task": "VARIES_PER_TRIAL — see trial.task",
        "total_trials": len(trials),
        "id_count": len(trials),
        "ood_count": 0,
        "trials": trials,
        "workspace_bbox": list(WORKSPACE_BBOX),
        "cup_bbox": list(CUP_BBOX),
        "duck_bbox": list(DUCK_BBOX),
        "robot_nogo_bbox": list(ROBOT_NOGO_BBOX),
        "duck_nogo_bbox": list(DUCK_NOGO_BBOX),
        "scene_counts": dict(Counter(t["tags"]["scene"] for t in trials)),
        "_generated_by": "vbti/logic/inference/protocols/generators/make_checkpoint_sweep.py",
    }, indent=2))
    print(f"Wrote {OUT_PATH} ({len(trials)} trials)")
    print("scene", dict(Counter(t["tags"]["scene"] for t in trials)))
    print("color", dict(Counter(t["tags"]["target_color"] for t in trials)))
    duals = [t for t in trials if t["tags"]["scene"].startswith("both_")]
    print("dual side", dict(Counter(t["tags"]["target_side"] for t in duals)))
    print("dual closer", dict(Counter(t["tags"]["target_closer"] for t in duals)))
    render(PROTOCOL_NAME)


if __name__ == "__main__":
    main()
