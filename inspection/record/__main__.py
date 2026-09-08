"""CLI over the record layer. Run: p -m inspection.record <cmd>

  ls        [--data | --all] [--object O] [--status S] [--source S] [--rig R]
  card ID   [--root R]
  validate  [ID | --all] [--deep] [--root R]
  show ID   [--out F.rrd] [--root R]
  story ID  [--root R]

Two archives, one CLI: `ls` reads data/runs/ (real inspection runs), `ls
--data` reads data/datasets/ (collection sweeps), `ls --all` reads both.
Commands that take an ID look it up in both roots, so `card 0809-box1` works
no matter which world the run lives in; `--root` overrides everything for a
folder that lives elsewhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inspection.record.catalog import DATASETS_ROOT, RUNS_ROOT


def _run_dir(a) -> Path:
    """Resolve an ID to its directory: explicit --root wins, else whichever
    of the two archive roots actually holds it (runs/ searched first)."""
    if a.root:
        return Path(a.root) / a.id
    for root in (RUNS_ROOT, DATASETS_ROOT):
        if (root / a.id / "run.json").exists():
            return root / a.id
    return RUNS_ROOT / a.id  # missing everywhere -> let the command report it


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inspection.record")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls")
    p_ls.add_argument("--object"), p_ls.add_argument("--status")
    p_ls.add_argument("--source"), p_ls.add_argument("--rig")
    p_ls.add_argument("--data", action="store_true",
                      help="list data-collection sweeps (data/datasets/)")
    p_ls.add_argument("--all", action="store_true",
                      help="list both runs and datasets")
    p_card = sub.add_parser("card")
    p_card.add_argument("id")
    p_val = sub.add_parser("validate")
    p_val.add_argument("id", nargs="?")
    p_val.add_argument("--all", action="store_true")
    p_val.add_argument("--deep", action="store_true")
    p_show = sub.add_parser("show")
    p_show.add_argument("id")
    p_show.add_argument("--out")
    p_show.add_argument("--no-open", action="store_true",
                        help="write the rrd only, don't launch the viewer")
    p_story = sub.add_parser("story")
    p_story.add_argument("id")
    for p in (p_ls, p_card, p_val, p_show, p_story):
        p.add_argument("--root", default=None)

    a = ap.parse_args(argv)

    if a.cmd == "ls":
        from inspection.record.catalog import runs
        if a.root:
            roots = [Path(a.root)]
        elif a.all:
            roots = [RUNS_ROOT, DATASETS_ROOT]
        else:
            roots = [DATASETS_ROOT if a.data else RUNS_ROOT]
        for root in roots:
            for c in runs(root, object=a.object, status=a.status,
                          source=a.source, rig=a.rig):
                kind = "data" if c.source == "data-engine" else "run"
                print(f"{c.id:22s} {kind:4s} {c.object or '-':8s} "
                      f"{c.effective_status:9s} {c.rig:4s} "
                      f"steps={c.n_steps:3d} {c.size_bytes // 1024:6d}K "
                      f"verdict={c.verdict or '-'}")
    elif a.cmd == "card":
        from inspection.record.catalog import card
        d = _run_dir(a)
        c = card(d.parent, d.name)
        for k, v in vars(c).items():
            print(f"  {k}: {v}")
    elif a.cmd == "validate":
        from inspection.record.validate import validate_archive, validate_run
        if a.all:
            roots = [Path(a.root)] if a.root else [RUNS_ROOT, DATASETS_ROOT]
            reports = [r for root in roots
                       for r in validate_archive(root, deep=a.deep)]
        else:
            reports = [validate_run(_run_dir(a), deep=a.deep)]
        for r in reports:
            print(f"{r.run_id:22s} ok={r.ok} status={r.effective_status}")
            for p in r.problems:
                print(f"    [{p.severity}] {p.where}: {p.what}")
        return 0 if all(r.ok for r in reports) else 1
    elif a.cmd == "show":
        import subprocess
        # The full six-story review workspace (record/rr) when the run has
        # derived step clouds; minimal pose/image/cloud projection otherwise.
        try:
            from inspection.record.rr import view as rr_view
            from inspection.record.rr import workspace as rr_ws
            rr_ws._replay_dir(a.id)  # raises SystemExit if no step clouds
            out = rr_ws.build(a.id)
            print(f"wrote {out} (full workspace)")
            if not a.no_open:
                rr_view.show(a.id)
            return 0
        except (Exception, SystemExit):
            pass
        from inspection.record.show import show_run
        out = show_run(_run_dir(a), Path(a.out or f"/tmp/{a.id}.rrd"))
        print(f"wrote {out} (minimal — no derived step clouds; "
              f"run step_replay/derive for the full workspace)")
        if not a.no_open:
            viewer = Path(sys.executable).parent / "rerun"
            subprocess.Popen([str(viewer if viewer.exists() else "rerun"),
                              str(out)], start_new_session=True)
            print("viewer launched")
    elif a.cmd == "story":
        from inspection.record.story import story
        print(story(_run_dir(a)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
