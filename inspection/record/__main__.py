"""CLI over the record layer. Run: p -m inspection.record <cmd>

  ls        [--root R] [--object O] [--status S] [--source S] [--rig R]
  card ID   [--root R]
  validate  [ID | --all] [--deep] [--root R]
  show ID   [--out F.rrd] [--root R]
  story ID  [--root R]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "runs"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inspection.record")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls")
    p_ls.add_argument("--object"), p_ls.add_argument("--status")
    p_ls.add_argument("--source"), p_ls.add_argument("--rig")
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
        p.add_argument("--root", default=str(DEFAULT_ROOT))

    a = ap.parse_args(argv)
    root = Path(a.root)

    if a.cmd == "ls":
        from inspection.record.catalog import runs
        for c in runs(root, object=a.object, status=a.status,
                      source=a.source, rig=a.rig):
            print(f"{c.id:22s} {c.object or '-':8s} {c.effective_status:9s} "
                  f"{c.rig:4s} steps={c.n_steps:3d} {c.size_bytes // 1024:6d}K "
                  f"verdict={c.verdict or '-'}")
    elif a.cmd == "card":
        from inspection.record.catalog import card
        c = card(root, a.id)
        for k, v in vars(c).items():
            print(f"  {k}: {v}")
    elif a.cmd == "validate":
        from inspection.record.validate import validate_archive, validate_run
        reports = validate_archive(root, deep=a.deep) if a.all \
            else [validate_run(root / a.id, deep=a.deep)]
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
        out = show_run(root / a.id, Path(a.out or f"/tmp/{a.id}.rrd"))
        print(f"wrote {out} (minimal — no derived step clouds; "
              f"run step_replay/derive for the full workspace)")
        if not a.no_open:
            viewer = Path(sys.executable).parent / "rerun"
            subprocess.Popen([str(viewer if viewer.exists() else "rerun"),
                              str(out)], start_new_session=True)
            print("viewer launched")
    elif a.cmd == "story":
        from inspection.record.story import story
        print(story(root / a.id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
