#!/usr/bin/env python3
"""Fit primitive decompositions to a run's step clouds, offline.

Answers "what shape does each method think this object is, and at which step
did it first become a defensible answer" without a robot: the step clouds
are already on disk under `data/derived/<run>/steps/step_NNN.ply`.

Wraps `bank.fit` and nothing else — the research path and the loop path are
the same call, so a verdict formed here transfers.

Writes one `Decomposition` per (method, step) to
`data/derived/<run>/fits/<method>/step_NNN.json` and one score row per
(method, step) to `data/derived/<run>/fits/<method>/scores.jsonl`.

    p inspection/fit/cli.py run --run 2408-cup3 --steps all --methods planar,cyl,ems
    p inspection/fit/cli.py run --run 2408-cup4 --steps final
    p inspection/fit/cli.py run --run 2408-cup5 --steps 0,5,10 --methods cyl
    p inspection/fit/cli.py show --run 2408-cup3 --method cyl --steps final
    p inspection/fit/cli.py methods

Standalone `p inspection/fit/cli.py <cmd> ...` still works.
"""
import json
import sys
from pathlib import Path

import numpy as np

from inspection.fit import bank
from inspection.fit.contract import decomposition_from_json
from inspection.fit.scores import score_row

DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"


def steps_of(run: str):
    d = DERIVED / run / "steps"
    if not d.is_dir():
        raise SystemExit(f"no step clouds at {d}")
    return sorted(d.glob("step_*.ply"))


def select(paths, spec: str):
    if spec == "all":
        return paths
    if spec == "final":
        return paths[-1:]
    want = {int(s) for s in spec.split(",") if s.strip()}
    by_n = {int(p.stem.split("_")[1]): p for p in paths}
    missing = sorted(want - set(by_n))
    if missing:
        raise SystemExit(f"steps {missing} not on disk (have 0..{max(by_n)})")
    return [by_n[n] for n in sorted(want)]


def load(path: Path) -> np.ndarray:
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=float)


def _job(args):
    """One (run, step) -> all methods. Top-level so multiprocessing can pickle it."""
    run, path, methods, seed = args
    P = load(path)
    decs, secs = bank.fit_timed(P, methods, seed=seed)
    n = int(path.stem.split("_")[1])
    rows = []
    for m, dec in decs.items():
        out = DERIVED / run / "fits" / m
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{path.stem}.json").write_text(json.dumps(dec.to_json(), indent=1))
        row = score_row(P, dec)
        row.update(step=n, run=run, seconds=secs[m])
        if m == "planar":
            from inspection.fit import planar
            try:
                row["guard"] = planar.diagnostics(P)
            except Exception as exc:                        # noqa: BLE001
                row["guard"] = {"error": str(exc)}
        rows.append(row)
    return rows


def _strs(value):
    """Comma-list flag value -> list[str], whatever fire made of it.

    fire literal-evals argument values: `--steps 0,5,10` arrives as the
    tuple (0, 5, 10), `--steps 5` as the int 5, while `--run 2408-cup3`
    and `--steps all` stay strings. Everything becomes a list of
    stripped strings; `",".join(_strs(x))` recovers the spec `select`
    understands.
    """
    if isinstance(value, (tuple, list)):
        value = ",".join(str(v) for v in value)
    return [s.strip() for s in str(value).split(",") if s.strip()]


def run(run, steps="final", methods=",".join(bank.DEFAULT_METHODS), seed=0,
        jobs=1):
    """Fit and persist. --run: run id or comma list; --steps: all | final | 0,5,10."""
    methods = _strs(methods)
    for m in methods:
        try:
            bank.get(m)                               # fail fast on a typo
        except KeyError as exc:
            raise SystemExit(str(exc).strip('"')) from None
    runs = _strs(run)
    steps = ",".join(_strs(steps))
    seed, n_jobs = int(seed), int(jobs)

    jobs = [(r, p, methods, seed)
            for r in runs for p in select(steps_of(r), steps)]
    print(f"{len(jobs)} step-clouds x {len(methods)} methods "
          f"({', '.join(methods)}), seed {seed}, {n_jobs} worker(s)")

    if n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_jobs) as pool:
            results = pool.map(_job, jobs)
    else:
        results = [_job(j) for j in jobs]

    per_run = {}
    print(f"\n{'run':>11} {'step':>5} {'N':>6} {'method':>7} {'prim':>4} "
          f"{'cov':>6} {'resid':>7} {'s':>6}  detail")
    for rows in results:
        for r in rows:
            per_run.setdefault((r["run"], r["method"]), []).append(r)
            detail = (r["reason"] if r["abstain"]
                      else " ".join(f"{lb}:{s}" for lb, s
                                    in zip(r["labels"], r["support"])))
            print(f"{r['run']:>11} {r['step']:>5} {r['n_points']:>6} "
                  f"{r['method']:>7} {r['n_primitives']:>4} "
                  f"{r['coverage']:>6.3f} {r['residual_mm']:>6.2f}m "
                  f"{r['seconds']:>6.2f}  {detail[:70]}")

    for (run, method), rows in sorted(per_run.items()):
        out = DERIVED / run / "fits" / method
        out.mkdir(parents=True, exist_ok=True)
        log = out / "scores.jsonl"
        # Merge, never truncate: `--steps 0,5,10` must not delete the rows a
        # previous `--steps all` earned. New rows win on their own step.
        merged = {}
        if log.exists():
            for line in log.read_text().splitlines():
                if line.strip():
                    old = json.loads(line)
                    merged[old["step"]] = old
        for r in rows:
            merged[r["step"]] = r
        rows = [merged[s] for s in sorted(merged)]
        with log.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        fitted = [r for r in rows if not r["abstain"]]
        first = fitted[0]["step"] if fitted else None
        print(f"\n{run}/{method}: {len(fitted)}/{len(rows)} steps fitted, "
              f"first at step {first}, "
              f"final coverage {rows[-1]['coverage']:.3f} "
              f"residual {rows[-1]['residual_mm']:.2f} mm "
              f"-> {out}/scores.jsonl")


def show(run, method, steps="final"):
    """Print a stored decomposition."""
    run, method = str(run), str(method)
    steps = ",".join(_strs(steps))
    for path in select(steps_of(run), steps):
        f = DERIVED / run / "fits" / method / f"{path.stem}.json"
        if not f.exists():
            print(f"{path.stem}: not fitted")
            continue
        d = decomposition_from_json(json.loads(f.read_text()))
        print(f"{path.stem} {d.method} {d.version}")
        print(f"  abstain={d.abstain} reason={d.reason!r}")
        print(f"  coverage={d.coverage:.3f} residual={d.residual_mm:.2f} mm")
        for p in d.primitives:
            print(f"  - {p.label:<14} {p.kind:<12} support={p.support:<5} "
                  f"resid={p.residual_mm:5.2f} mm  params="
                  f"{json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in p.params.items() if k != 'vertices'})}")


def methods():
    """Registered fitters and versions."""
    for m in bank.methods():
        mod = bank.get(m).__module__
        print(f"{m:<8} {sys.modules[mod].VERSION}")


def commands():
    return {"run": run, "show": show, "methods": methods}


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(commands())
