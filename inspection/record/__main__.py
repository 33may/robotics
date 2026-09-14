"""CLI over the record layer. Run: p inspection/record/__main__.py <cmd>

  p inspection/record/__main__.py ls [--data | --all] [--object O] [--status S]
                                [--source S] [--rig R]
  p inspection/record/__main__.py card ID [--root R]
  p inspection/record/__main__.py validate [ID | --all] [--deep] [--root R]
  p inspection/record/__main__.py show ID [--out F.rrd] [--noopen] [--root R]
  p inspection/record/__main__.py story ID [--root R]

Two archives, one CLI: `ls` reads data/runs/ (real inspection runs), `ls
--data` reads data/datasets/ (collection sweeps), `ls --all` reads both.
Commands that take an ID look it up in both roots, so `card 0809-box1` works
no matter which world the run lives in; `--root` overrides everything for a
folder that lives elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path


from inspection.record.catalog import DATASETS_ROOT, RUNS_ROOT


def _run_dir(id, root=None) -> Path:
    """Resolve an ID to its directory: explicit --root wins, else whichever
    of the two archive roots actually holds it (runs/ searched first)."""
    id = str(id)  # fire literal-evals: a bare-int id arrives as int
    if root:
        return Path(root) / id
    for r in (RUNS_ROOT, DATASETS_ROOT):
        if (r / id / "run.json").exists():
            return r / id
    return RUNS_ROOT / id  # missing everywhere -> let the command report it


def ls(data=False, all=False, object=None, status=None, source=None,
       rig=None, root=None):
    """List runs (default), data-collection sweeps (--data), or both (--all)."""
    from inspection.record.catalog import runs
    if root:
        roots = [Path(root)]
    elif all:
        roots = [RUNS_ROOT, DATASETS_ROOT]
    else:
        roots = [DATASETS_ROOT if data else RUNS_ROOT]
    for r in roots:
        for c in runs(r, object=object, status=status,
                      source=source, rig=rig):
            kind = "data" if c.source == "data-engine" else "run"
            print(f"{c.id:22s} {kind:4s} {c.object or '-':8s} "
                  f"{c.effective_status:9s} {c.rig:4s} "
                  f"steps={c.n_steps:3d} {c.size_bytes // 1024:6d}K "
                  f"verdict={c.verdict or '-'}")


def card(id, root=None):
    """Print every catalog field of one run."""
    from inspection.record.catalog import card as catalog_card
    d = _run_dir(id, root)
    c = catalog_card(d.parent, d.name)
    for k, v in vars(c).items():
        print(f"  {k}: {v}")


def validate(id=None, all=False, deep=False, root=None):
    """Validate one run (ID) or every run in both archives (--all)."""
    from inspection.record.validate import validate_archive, validate_run
    if all:
        roots = [Path(root)] if root else [RUNS_ROOT, DATASETS_ROOT]
        reports = [r for rt in roots
                   for r in validate_archive(rt, deep=deep)]
    else:
        reports = [validate_run(_run_dir(id, root), deep=deep)]
    for r in reports:
        print(f"{r.run_id:22s} ok={r.ok} status={r.effective_status}")
        for p in r.problems:
            print(f"    [{p.severity}] {p.where}: {p.what}")
    if any(not r.ok for r in reports):
        raise SystemExit(1)


def show(id, out=None, open=True, root=None):
    """Open a run in Rerun (--noopen writes the rrd only)."""
    import subprocess
    id = str(id)
    run_dir = _run_dir(id, root)
    # The full six-story review workspace (record/rr) when the run has
    # step clouds ANYWHERE — the derived tree or recorded steps/NNN/fused
    # pairs; the reduced two-pane projection only when it has neither.
    # Deliberately no blanket try/except around the build: a failing full
    # build is a real error and must say what broke, not masquerade as
    # "no derived step clouds" (Anton, 2026-09-10).
    from inspection.record.rr import view as rr_view
    from inspection.record.rr import workspace as rr_ws
    try:
        rr_ws._replay_dir(id)  # raises SystemExit if no derived clouds
        has_clouds = True
    except SystemExit:
        has_clouds = any(run_dir.glob("steps/*/fused/cloud.npy"))
    if has_clouds:
        p = rr_ws.build(id)
        print(f"wrote {p} (full workspace)")
        if open:
            rr_view.show(id)
        return
    from inspection.record.show import show_run
    p = show_run(_run_dir(id, root), Path(out or f"/tmp/{id}.rrd"))
    print(f"wrote {p} (reduced two-pane — no derived step clouds; "
          f"run step_replay/derive for the full workspace)")
    if open:
        viewer = Path(sys.executable).parent / "rerun"
        # The .rbl sidecar ACTIVATES the two-pane layout over whatever
        # blueprint the viewer persisted for this app-id (rr/log.py:save).
        files = [str(p)]
        if p.with_suffix(".rbl").exists():
            files.append(str(p.with_suffix(".rbl")))
        subprocess.Popen([str(viewer if viewer.exists() else "rerun"),
                          *files], start_new_session=True)
        print("viewer launched")


def story(id, root=None):
    """Print the AI trace of a run as readable text."""
    from inspection.record.story import story as record_story
    print(record_story(_run_dir(id, root)))


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"ls": ls, "card": card, "validate": validate,
               "show": show, "story": story})
