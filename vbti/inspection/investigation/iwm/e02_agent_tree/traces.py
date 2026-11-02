"""e02: the traces and numbers for the report, from the per-run summary.json files.

Writes evidence/traces.md (one block per frame and variant: numbers, the tool sequence
with results, the final tree as text) and evidence/summary.json.
Run:  p inspection/investigation/iwm/e02_agent_tree/traces.py
"""
import json
from pathlib import Path

from inspection.investigation.iwm.e01_all_masks import common as C

HERE = Path(__file__).resolve().parent
DATA = C.ROOT / "inspection" / "data" / "iwm" / "e02_agent_tree"


def main():
    rows = []
    md = ["# e02 traces\n"]
    for fdir in sorted(DATA.iterdir()):
        for vdir in sorted(fdir.iterdir()):
            s = vdir / "summary.json"
            if not s.exists():
                continue
            sm = json.load(open(s))
            ev = json.load(open(vdir / "events.json"))
            rows.append({k: sm[k] for k in ("frame", "variant", "turns", "tool_calls", "seconds",
                                            "nodes_before", "nodes_after", "nodes_final",
                                            "unnamed_dropped", "exhausted")})
            md.append(f"## {sm['frame']} · {sm['variant']}\n")
            md.append(f"{sm['turns']} turns, {sm['tool_calls']} tool calls, {sm['seconds']} s; "
                      f"nodes {sm['nodes_before']} → {sm['nodes_after']} after the agent → "
                      f"{sm['nodes_final']} named ({sm['unnamed_dropped']} unnamed dropped)"
                      f"{'; EXHAUSTED' if sm['exhausted'] else ''}\n")
            md.append(f"**answer:** {sm['answer']}\n")
            md.append("```")
            for e in ev:
                if e["event"] == "tool":
                    r = str(e.get("result", "")).replace("\n", " ")
                    md.append(f"t{e['turn']:02d} {e['t']:6.1f}s  {e['tool']} {json.dumps(e.get('args', {}))[:70]}")
                    md.append(f"      -> {r[:160]}")
            md.append("```\n")
            t = HERE / "evidence" / f"{sm['frame']}_{sm['variant']}_tree.txt"
            if t.exists():
                md.append("```\n" + t.read_text().rstrip() + "\n```\n")
    (HERE / "evidence" / "traces.md").write_text("\n".join(md))
    json.dump(rows, open(HERE / "evidence" / "summary.json", "w"), indent=1)
    print("| frame | variant | turns | tool calls | s | nodes draft → agent → named |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['frame']} | {r['variant']} | {r['turns']} | {r['tool_calls']} | {r['seconds']:.0f} | "
              f"{r['nodes_before']} → {r['nodes_after']} → {r['nodes_final']}"
              f"{' (exhausted)' if r['exhausted'] else ''} |")


if __name__ == "__main__":
    main()
