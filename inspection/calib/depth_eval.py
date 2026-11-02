#!/usr/bin/env python3
"""Depth-calibration eval, end to end: survey -> fixed cells -> depth_val -> compare to baseline.

The views are CELLS, not joint configs (Anton 2026-10-15): the object moves between eval runs
and a cell follows it. Same cells as the baseline run 1310-calib-red (survey + 6 cells).
Motion goes through the normal collect path (approval gate auto-confirmed, UI shows each move).

  p inspection/calib/depth_eval.py --label=h240            # object with the yellow dot on the table
  p inspection/calib/depth_eval.py --label=h240 --no_window
  p inspection/calib/depth_eval.py --label=h240 --skip_capture=<run>   # re-score, no motion

Out: data/runs/<DDMM>-depth-eval-<label>/ (a normal run), data/depth_val/<run>/{depth_val.json,
dots.jpg, compare.json}
"""
import hashlib
import json
from datetime import datetime

import numpy as np

from inspection.calib import depth_val

CELLS = [[2, 1], [5, 1], [8, 1], [9, 1], [10, 2], [11, 1]]   # 1310-calib-red steps 1-6, same order
BASELINE = "1310-calib-red"


def calib_sha():
    """Fingerprint of the calibration table on the camera now (which candidate was live)."""
    from inspection.calib.depth_tare import _ac
    return hashlib.sha256(bytes(_ac().get_calibration_table())).hexdigest()[:12]


def spread_mm(res):
    """How far apart the views put the same dot: median pairwise distance of the per-view dots."""
    P = np.array([v["p_base"] for v in res["views"] if "p_base" in v])
    if len(P) < 2:
        return None
    d = [np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P))]
    return round(float(np.median(d)) * 1000, 2)


def summary(res):
    g = lambda k, f: (res.get(k) or {}).get(f)
    return {"dots found": f"{res['n_dots']}/{res['n_views']}",
            "dot spread between views (median mm)": spread_mm(res),
            "dot depth error |median| mm": g("dot_along_mm", "median_abs"),
            "dot depth error |max| mm": g("dot_along_mm", "max_abs"),
            "table height error |median| mm": g("table_h_mm", "median_abs"),
            "table height error |max| mm": g("table_h_mm", "max_abs"),
            "rays miss (depth-free floor) median mm": g("ray_miss_mm", "median")}


def _key(v):
    return tuple(v["address"]) if v.get("address") else "survey"


def main(label: str, object: str = "box", baseline: str = BASELINE, no_window: bool = False,
         skip_capture: str = None):
    """skip_capture=<run>: re-score an existing eval run without moving the robot."""
    if skip_capture:
        run, sha = depth_val.run_dir(skip_capture), None
    else:
        sha = calib_sha()
        run = depth_val.RUNS / f"{datetime.now():%d%m}-depth-eval-{label}"
        from inspection.run.app import collect
        collect(outdir=str(run), name=run.name, object=object, auto=True, cells=CELLS,
                no_window=no_window)
    new = depth_val.main(str(run))
    base = depth_val.main(baseline)
    sb, sn = summary(base), summary(new)
    print(f"\n=== {baseline} (before)  vs  {run.name} ({label}) ===")
    print(f"{'':42s} {'before':>10s} {'after':>10s}")
    for k in sb:
        print(f"{k:42s} {str(sb[k]):>10s} {str(sn[k]):>10s}")
    per = {_key(v): v for v in new["views"]}
    print(f"\nper view, before -> after:  {'cell':>8s} {'dot err mm':>18s} {'table h mm':>18s}")
    for v in base["views"]:
        n = per.get(_key(v), {})
        dot = f"{v.get('along_mm', '-')} -> {n.get('along_mm', '-')}"
        tab = f"{(v.get('table') or {}).get('h_mm', '-')} -> {(n.get('table') or {}).get('h_mm', '-')}"
        print(f"{'':27s} {str(_key(v)):>8s} {dot:>18s} {tab:>18s}")
    out = {"label": label, "calib_sha": sha, "run": run.name, "baseline": baseline,
           "rows": {k: [sb[k], sn[k]] for k in sb}}
    od = depth_val.OUT / run.name
    json.dump(out, open(od / "compare.json", "w"), indent=1)
    print(f"-> {od / 'compare.json'}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
