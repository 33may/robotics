"""e05 (RQ3): what the recorded conflicts are made of.

A conflict = (row, old part, new part): a row claimed by a part that is neither an ancestor nor
a descendant of the row's label. Grouped by (old, new) pair and classified:
  intermediate   the new part's rows CONTAIN most of the old part's rows (a new node slid in
                 above an existing one, e.g. 'lower front panel' over 'ethernet ports')
  reparent       the extended / new part lies mostly INSIDE the old label's part (the two views
                 disagree on the parent, e.g. 'Chort label' under the object vs on the panel)
  boundary       small overlap on both sides (mask edges)
With a re-parent / insert rule the first two kinds would vanish; this script only counts.

Run:  p inspection/investigation/iwm/e05_parts_in_v/conflict_kinds.py [--mm 1] [--mode agent]
Output  evidence/conflict_kinds.json
"""
import argparse
import collections
import json

import numpy as np

from inspection.investigation.iwm.e05_parts_in_v import common as C5

BOUNDARY = 0.1


def classify(d):
    j = json.load(open(d / "parts.json"))
    z = np.load(d / "rows.npz")
    off, ids = z["offsets"], z["ids"].tolist()
    rows = {pid: set(z["rows"][off[k]:off[k + 1]].tolist()) for k, pid in enumerate(ids)}
    names = {p["id"]: p["name"] for p in j["parts"]}
    pairs = collections.Counter((o, n) for _, o, n in j["conflicts"])
    out = []
    for (o, n), cnt in pairs.most_common():
        A, B = rows[o], rows[n]
        share_old_in_new = len(A & B) / max(len(A), 1)
        share_new_in_old = len(A & B) / max(len(B), 1)
        if share_old_in_new >= 0.5 and share_new_in_old < 0.5:
            kind = "intermediate"
        elif share_new_in_old >= 0.5:
            kind = "reparent"
        elif max(share_old_in_new, share_new_in_old) < BOUNDARY:
            kind = "boundary"
        else:
            kind = "mixed"
        out.append({"old": names[o], "new": names[n], "rows": cnt, "kind": kind,
                    "old_in_new": round(share_old_in_new, 2), "new_in_old": round(share_new_in_old, 2)})
    totals = collections.Counter()
    for r in out:
        totals[r["kind"]] += r["rows"]
    return {"conflicts": len(j["conflicts"]), "by_kind": dict(totals), "pairs": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", type=int, default=1)
    ap.add_argument("--mode", default="agent")
    a = ap.parse_args()
    res = {}
    for run, views in C5.ORDER.items():
        d = C5.DATA / run / f"{a.mm}mm" / f"keep_{a.mode}" / f"after_{views[-1]}"
        if not d.exists():
            continue
        res[run] = classify(d)
        print(run, res[run]["conflicts"], res[run]["by_kind"])
        for r in res[run]["pairs"][:8]:
            print(f"   {r['kind']:12s} {r['rows']:5d}  {r['old'][:24]:24s} -> {r['new'][:24]:24s} old_in_new {r['old_in_new']:.2f} new_in_old {r['new_in_old']:.2f}")
    json.dump(res, open(C5.EVID / "conflict_kinds.json", "w"), indent=1)


if __name__ == "__main__":
    main()
