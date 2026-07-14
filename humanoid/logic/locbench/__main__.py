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
        "scene_usd": "assets/envs/warehouse_nvidia/Isaac/Environments/Simple_Warehouse/"
                     "full_warehouse.usd",
        "camera_every": 16,           # ≈60 Hz frames — SLAM-friendly cadence
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


_RUNS_ROOT = _HERE / "runs"
_REALIZATIONS_DIR = (_REPO_ROOT / "humanoid" / "logic" / "oli" / "reason" /
                     "localization" / "realizations")


def _scene_cfg(name: str) -> dict:
    if name not in SCENES:
        sys.exit(f"unknown scene {name!r} — known: {', '.join(SCENES)}")
    return dict(SCENES[name])


def _cmd_run(args) -> int:
    from humanoid.logic.locbench.runner import run_bench

    cfg = _scene_cfg(args.scene)
    if cfg.get("scene_usd"):
        cfg["scene_usd"] = str(_REPO_ROOT / "humanoid" / cfg["scene_usd"])
    episodes_file = _EPISODES_DIR / f"{args.scene}.json"
    if not episodes_file.exists():
        sys.exit(f"no frozen episode set for {args.scene!r} — run "
                 f"`locbench episodes {args.scene} --freeze` (Anton approves the render)")
    return run_bench(
        candidate=args.candidate, scene_cfg=cfg, episodes_file=episodes_file,
        runs_root=_RUNS_ROOT, n_episodes=args.smoke or args.episodes,
        headless=args.headless, timeout_s=args.timeout,
        shadow_config=args.shadow_config, live_view=args.live_view,
        teleport=not args.walk_transit)


def _cmd_score(args) -> int:
    from humanoid.logic.locbench.episodes import load_episode_set
    from humanoid.logic.locbench.runner import score_run_dir
    from humanoid.logic.oli.reason.mapping import StaticMapping

    run_dir = Path(args.run_dir)
    import json as _json
    scene = _json.loads((run_dir / "report.json").read_text())["scene"]
    es = load_episode_set(_EPISODES_DIR / f"{scene}.json")
    grid = StaticMapping(str(_REPO_ROOT / "humanoid" / es.map_dir)).latest().grid
    doc = score_run_dir(run_dir, es, grid=grid)
    print(f"recomputed: {doc['run']['tier']} "
          f"(failed episodes: {doc['run']['failed_episodes'] or 'none'})")
    return 0 if doc["run"]["passed"] else 1


def _cmd_board(args) -> int:
    from humanoid.logic.locbench.runner import board

    print(board(_RUNS_ROOT))
    return 0


def _cmd_env(args) -> int:
    from humanoid.logic.locbench.envs import EnvError, env_create, env_remove

    rdir = _REALIZATIONS_DIR / args.candidate
    try:
        if args.action == "create":
            if not rdir.is_dir():
                sys.exit(f"no realization {args.candidate!r} at {rdir} — "
                         f"scaffold it first (loc-new)")
            lock = env_create(args.candidate, rdir, force=args.force)
            print(f"created bench-{args.candidate}; solve frozen → {lock}")
        else:  # remove
            env_remove(args.candidate)
            print(f"removed bench-{args.candidate} (lock.yml kept as the committed record)")
    except EnvError as e:
        sys.exit(str(e))
    return 0


def _cmd_episodes(args) -> int:
    cfg = _scene_cfg(args.scene)
    cfg.pop("scene_usd", None)     # sampling needs only the baked map
    cfg.pop("camera_every", None)
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

    p_run = sub.add_parser("run", help="evaluate a candidate on a scene's frozen episodes")
    p_run.add_argument("candidate", help="realizations/<name>")
    p_run.add_argument("--scene", default="warehouse")
    p_run.add_argument("--episodes", type=int, default=None, help="cap episode count")
    p_run.add_argument("--smoke", type=int, nargs="?", const=3, default=None,
                       help="quick grind: first N episodes (default 3)")
    p_run.add_argument("--timeout", type=float, default=90.0, help="scored-leg budget [s]")
    p_run.add_argument("--headless", action="store_true")
    p_run.add_argument("--live-view", action="store_true",
                       help="open a live window: GT vs candidate estimate on the map")
    p_run.add_argument("--walk-transit", action="store_true",
                       help="walk the unscored transit legs (D3 original) instead of the "
                            "default bench teleport to each spawn")
    p_run.add_argument("--shadow-config", default=None,
                       help="JSON file of config overrides for the candidate")
    p_run.set_defaults(fn=_cmd_run)

    p_sc = sub.add_parser("score", help="recompute report + plots from a run's stored pairs")
    p_sc.add_argument("run_dir")
    p_sc.set_defaults(fn=_cmd_score)

    p_bd = sub.add_parser("board", help="MD scoreboard — latest report per candidate")
    p_bd.set_defaults(fn=_cmd_board)

    p_env = sub.add_parser("env", help="create/remove a candidate's disposable bench-<name> env")
    p_env.add_argument("action", choices=["create", "remove"])
    p_env.add_argument("candidate", help="realizations/<name>")
    p_env.add_argument("--force", action="store_true", help="recreate if it already exists")
    p_env.set_defaults(fn=_cmd_env)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
