"""e05: the report's tables, read from evidence/*.json (never typed from a terminal).

Run:  p vbti/inspection/investigation/iwm/e05_parts_in_v/numbers.py
"""
import json

import numpy as np

from inspection.investigation.iwm.e05_parts_in_v import common as C5

E = C5.EVID


def j(name):
    p = E / name
    return json.load(open(p)) if p.exists() else None


def lift_table():
    d = j("lift.json")
    print("## RQ1 lift (median over nodes; small = widest extent < 20 mm)\n")
    print("| frame, mm | lift | small n | IoU med / min | recall med | smear med / max | large n | IoU med / min | smear med / max |")
    print("|---|---|---|---|---|---|---|---|---|")
    keys = sorted({k.rsplit("_", 1)[0] for k in d["summary"]})
    for k in keys:
        s, l = d["summary"].get(k + "_small"), d["summary"].get(k + "_large")
        frame, mm, lift = k.split("_")
        print(f"| {frame}, {mm} | {lift} | {s['n'] if s else '-'} | {s['iou_median']:.2f} / {s['iou_min']:.2f} | {s['recall_median']:.2f} | "
              f"{s['smear_median']:.2f} / {s['smear_max']:.2f} | {l['n']} | {l['iou_median']:.2f} / {l['iou_min']:.2f} | {l['smear_median']:.2f} / {l['smear_max']:.2f} |"
              if s and l else f"| {frame}, {mm} | {lift} | - |")
    print(f"\nchosen lift: **{d['chosen']}**; settings {d['settings']}\n")


def pairs_table():
    d = j("pairs.json")
    print("## RQ2 pairs (the same part from two views of one run)\n")
    for r in d["results"]:
        ps = [p for p in r["pairs"] if not p.get("empty")]
        if not ps:
            print(f"- {r['run']} {r['mm']} mm: no scored pairs (b1.1 vs b1.4 share no named part; the face pair was refuted by geometry)")
            continue
        wide = [p for p in ps if p["wide"] and p["a_id"] != 0]
        small = [p for p in ps if not p["wide"]]
        print(f"- {r['run']} {r['mm']} mm, {r['first']} -> {r['second']}: rigid vector {r['rigid_vector_mm']} mm over {len(wide)} wide pairs; "
              f"row IoU median wide {np.median([p['iou_rows'] for p in wide]):.2f} / small {np.median([p['iou_rows'] for p in small]):.2f}; "
              f"after removing the vector {np.median([p['iou_rows_shifted'] for p in wide]):.2f} / {np.median([p['iou_rows_shifted'] for p in small]):.2f}; "
              f"render IoU median {np.median([p['render_iou'] for p in ps]):.2f}")
    print("\n| pair (c5.2 -> c5.6, 1 mm) | width mm | offset mm | row IoU | shifted | render IoU |\n|---|---|---|---|---|---|")
    r = next(x for x in d["results"] if x["mm"] == 1 and x["run"].startswith("2209-compute"))
    for p in r["pairs"]:
        if p.get("empty"):
            continue
        print(f"| #{p['a_id']} -> #{p['b_id']} {p['what'].split('=')[0].strip()[:28]} | {p['width_mm']:.0f} | {p['offset_norm_mm']:.1f} | {p['iou_rows']:.2f} | {p['iou_rows_shifted']:.2f} | {p['render_iou']:.2f} |")
    print()


def keep_table():
    d = j("agent_vs_truth.json")
    print("## RQ3 / RQ4 keep (per later view: decisions vs truth; conflicts added)\n")
    print("| run, mm, mode | view | recovered | missed | wrong | invented | coarse | within ok | new ok | conflicts + | parts | agent turns / s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in sorted(d.items()):
        for v in r["views"]:
            if v["role"] != "later":
                continue
            s = v["agent_score"] or v["baseline_score"]
            a = v.get("agent") or {}
            print(f"| {r['run'].split('-')[1]}, {r['mm']} mm, {r['mode']} | {v['view']} | {s['recovered']} | {s['missed']} | {s['wrong']} | {s['invented']} | "
                  f"{s['coarse']} | {s['within_ok']} | {s['correct_new']} | {v['applied']['conflicts']} | {v['parts']} | "
                  f"{a.get('turns', '-')} / {a.get('seconds', '-')} |")
    print()
    c = j("conflict_kinds.json")
    if c:
        print("conflict kinds (1 mm, agent, after all views):")
        for run, r in c.items():
            print(f"- {run}: {r['conflicts']} rows: {r['by_kind']}; top pairs: " +
                  "; ".join(f"{p['old']} -> {p['new']} {p['rows']} ({p['kind']})" for p in r["pairs"][:4]))
        print()


def views_table():
    d = j("views.json")
    print("## RQ5 views (1 mm, agent STV)\n")
    for run, r in d.items():
        s = r["summary"]
        print(f"- {run}: {s['parts']} parts; computed pose >= detecting step for {s['computed_beats_or_matches_step']}/{s['compared']}; "
              f"normal pose visible >= 0.9 for {s['normal_visible_ge_0.9']}, centred <= 10 % for {s['normal_centred_le_0.1']}; "
              f"median incidence {s['normal_incidence_median']:.0f} deg; a 30 deg tilt costs {100 * s['tilt30_visible_drop_median']:.0f} % visibility (median)")
    print("\n| part | extent mm | normal from | standoff | normal pose: vis / occ / inc / ctr | best | detecting step: vis / inc / ctr / dist |\n|---|---|---|---|---|---|---|")
    for run, r in d.items():
        for p in r["parts"]:
            n, b, st = p["normal_score"], p["best"], p["best_detecting_step"]
            inc = lambda s: f"{s['incidence_med_deg']:.0f}°" if s and s["incidence_med_deg"] is not None else "–"
            print(f"| {p['name'][:26]} | {p['extent_mm'][0]:.0f}×{p['extent_mm'][1]:.0f} | {p['normal_source']} | {p['standoff_m']:.2f} | "
                  f"{n['visible']:.2f} / {n['occluded']:.2f} / {inc(n)} / {max(n['centring_x'], n['centring_y']):.2f} | {b['name']} ({b['visible']:.2f}) | "
                  + (f"{st['name']}: {st['visible']:.2f} / {inc(st)} / {max(st['centring_x'], st['centring_y']):.2f} / {st['dist_m']:.2f} m |" if st else "– |"))
    print()


def sets_table():
    d = j("sets.json")
    print("## RQ6 sets\n\n| set | parts | normal from | standoff | best | min visible | per part | exists > 0.8 | best step |\n|---|---|---|---|---|---|---|---|---|")
    for k, r in d.items():
        per = ", ".join(f"{v:.2f}" for v in r["best"]["per_part"].values())
        print(f"| {k} | {len(r['parts'])} | {r['normal_source']} | {r['standoff_m']:.2f} | {r['best']['name']} | {r['best']['min_visible']:.2f} | {per} | {r['exists_80']} | {r['best_step']['name']} {r['best_step']['min_visible']:.2f} |")
    print()


def queries_table():
    d = j("queries.json")
    print("## search queries (precision / recall)\n")
    for run, q in d.items():
        print(f"- {run}: " + ", ".join(f"{k} {v['precision'] if v['precision'] is not None else '–'}/{v['recall']}" for k, v in q.items()))
    print()


if __name__ == "__main__":
    for f in (lift_table, pairs_table, keep_table, views_table, sets_table, queries_table):
        try:
            f()
        except Exception as e:                      # a missing file must not hide the other tables
            print(f"({f.__name__}: {e})\n")
