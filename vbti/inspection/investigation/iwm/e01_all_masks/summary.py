"""e01 numbers for the report: masks per set, nodes/depth/conflicts per tree.

Writes evidence/tree_stats.json and prints a markdown table.
Run:  p inspection/investigation/iwm/e01_all_masks/summary.py
"""
import json

from inspection.investigation.iwm.e01_all_masks import common as C


def main():
    stats = {}
    for f in C.frames():
        fid = f["id"]
        stats[fid] = {"masks": {}, "trees": {}}
        mdir = C.DATA / "masks" / fid
        if mdir.exists():
            for d in sorted(mdir.iterdir()):
                if (d / "masks.json").exists():
                    stats[fid]["masks"][d.name] = C.load_masks(d)[1]["n_masks"]
        tdir = C.DATA / "trees" / fid
        if tdir.exists():
            for d in sorted(tdir.iterdir()):
                if (d / "tree.json").exists():
                    t = json.load(open(d / "tree.json"))
                    nodes = t["nodes"]
                    maxd = max(n["depth"] for n in nodes)
                    stats[fid]["trees"][d.name] = {
                        "nodes": len(nodes) - 1, "depth": maxd,
                        "per_depth": [sum(n["depth"] == k for n in nodes) for k in range(1, maxd + 1)],
                        "conflicts": len(t["conflicts"]), "object_from": t["object_from"]}
    json.dump(stats, open(C.EVIDENCE / "tree_stats.json", "w"), indent=1)
    print("| frame | tree | nodes | depth | per depth | conflicts |")
    print("|---|---|---|---|---|---|")
    for fid, s in stats.items():
        for name, t in s["trees"].items():
            print(f"| {fid} | {name} | {t['nodes']} | {t['depth']} | {t['per_depth']} | {t['conflicts']} |")
    print()
    print("| frame | " + " | ".join(sorted({k for s in stats.values() for k in s['masks']})) + " |")


if __name__ == "__main__":
    main()
