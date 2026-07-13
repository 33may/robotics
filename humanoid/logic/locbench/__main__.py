"""python -m humanoid.logic.locbench — the locbench CLI (design.md D14, the agent surface).

    p -m humanoid.logic.locbench episodes <scene> [--seed N] [--freeze]   # sample → render → freeze
    (run / score / board / map / env land with §6-§9)

`episodes` samples a candidate set on the scene's baked map and renders the approval picture;
`--freeze` additionally writes `logic/locbench/episodes/<scene>.json` (versioned, committed)
AFTER Anton approves the render. Idempotent: same seed → same set → same file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the `humanoid` namespace package importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.locbench.episodes import (  # noqa: E402
    sample_episode_set,
    save_episode_set,
)

_HERE = Path(__file__).resolve().parent
_EPISODES_DIR = _HERE / "episodes"

# Scene registry: name → the baked map + sampling defaults tuned to that scene.
SCENES = {
    "warehouse": {
        "map_dir": "assets/envs/warehouse_nvidia/nav_maps/v1",
        "origin_xy": (0.0, 0.0),      # the glide World's boot spawn (USD origin)
        "n_episodes": 10,
        "min_separation_m": 8.0,
        "min_route_m": 10.0,
        "clearance_m": 0.5,
        "n_coverage": 8,
        # ~70% of episodes live between the shelving rails (Anton, 13-07-2026): spawn AND
        # goal inside the rack block, so the route threads the aisles — the localization-
        # interesting regime (repetitive geometry, narrow passages).
        "zones": [{"rect": (-26.5, 6.5, 0.0, 28.5), "n": 7}],
    },
}


def _cmd_episodes(args) -> int:
    if args.scene not in SCENES:
        sys.exit(f"unknown scene {args.scene!r} — known: {', '.join(SCENES)}")
    cfg = dict(SCENES[args.scene])
    map_dir = cfg.pop("map_dir")

    from humanoid.logic.locbench.render import render_episode_set
    from humanoid.logic.oli.reason.mapping import StaticMapping

    grid = StaticMapping(str(_REPO_ROOT / "humanoid" / map_dir)).latest().grid
    es = sample_episode_set(grid, scene=args.scene, map_dir=map_dir,
                            seed=args.seed, **cfg)
    for ep in es.episodes:
        print(f"  ep {ep.id}: S({ep.spawn[0]:6.2f},{ep.spawn[1]:6.2f}) → "
              f"G({ep.goal[0]:6.2f},{ep.goal[1]:6.2f})   route {ep.route_m:5.1f} m")

    png = render_episode_set(grid, es, _EPISODES_DIR / f"{args.scene}_episodes.png")
    print(f"render → {png}")

    if args.freeze:
        out = _EPISODES_DIR / f"{args.scene}.json"
        save_episode_set(es, out)
        print(f"FROZEN → {out}")
    else:
        print("(dry sample — pass --freeze after the render is approved)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="locbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ep = sub.add_parser("episodes", help="sample + render (+ --freeze) a scene's episode set")
    p_ep.add_argument("scene")
    p_ep.add_argument("--seed", type=int, default=33)
    p_ep.add_argument("--freeze", action="store_true",
                      help="write episodes/<scene>.json (do this after render approval)")
    p_ep.set_defaults(fn=_cmd_episodes)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
