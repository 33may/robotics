#!/usr/bin/env python3
"""Per-view ground truth — the thing that makes readout measurable.

The design's known gap: without a human label per view, "did the subagent
read this frame correctly" has no answer, so no tool can be trimmed on
evidence and MAY-187's decider has no scoring function. This is the smallest
thing that closes it — a JSON file beside the run, one entry per CAPTURE
(not per cell: a cell can hold two views, and they can disagree).

Vocabulary is deliberately three-valued. "partial" is not fence-sitting: a
logo half-cut by the frame edge is the exact case that should drive the next
move, so collapsing it into y/n would erase the signal the system exists to
find.

Labels are ground truth ABOUT a run, produced by a human — they are not part
of the run structure's trust levels, which govern what CODE and MODELS may
write. Hence a separate file.

Run: p inspection/eyes/bench/labels.py <run_dir> [question]
"""
import json
from pathlib import Path

LABELS = ("y", "n", "partial", "?")
_MARK = {"y": "y", "n": "n", "partial": "p", "?": "."}


class LabelSet:
    """`labels.json` — {capture dir: {cell, label, note}}."""

    def __init__(self, out_dir, data):
        # Wherever the bench keeps its working set — the same directory the
        # RunStore it was built from flushes to (`RunStore.path`).
        self.dir = Path(out_dir)
        self.path = self.dir / "labels.json"
        self._d = data

    # ------------------------------------------------------------- create
    @classmethod
    def template(cls, store, question):
        """One unlabelled entry per view, in capture order."""
        entries = {}
        for v in sorted(store.views(), key=lambda v: v.t):
            entries[v.cap_dir] = {"cell": list(v.cell) if v.cell else None,
                                  "label": "?", "note": ""}
        labels = cls(store.path, {"question": question, "entries": entries})
        labels._flush()
        return labels

    @classmethod
    def load(cls, out_dir):
        p = Path(out_dir) / "labels.json"
        return cls(out_dir, json.loads(p.read_text()))

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._d, indent=1) + "\n")

    # --------------------------------------------------------------- read
    @property
    def question(self):
        return self._d["question"]

    @property
    def entries(self):
        return self._d["entries"]

    def pending(self):
        return sum(1 for e in self.entries.values() if e["label"] == "?")

    def counts(self):
        c = {k: 0 for k in LABELS}
        for e in self.entries.values():
            c[e["label"]] += 1
        return c

    # -------------------------------------------------------------- write
    def set(self, cap_dir, label, note=""):
        if label not in LABELS:
            raise ValueError(f"label {label!r} not in {LABELS}")
        if cap_dir not in self.entries:
            raise KeyError(f"no view {cap_dir!r} in this run")
        self.entries[cap_dir].update(label=label, note=note)
        self._flush()

    # ------------------------------------------------------------- report
    def report(self, store):
        """Counts plus the labels laid out on the viewsphere grid."""
        c = self.counts()
        head = (f"{c['y']} y · {c['n']} n · {c['partial']} partial · "
                f"{c['?']} unlabelled   ({self.question})")
        g = store._d["grid"]
        n_h, elevs = g["h_bins"], g["v_elevs"]
        marks = {}
        for cap_dir, e in self.entries.items():
            if e["cell"] is not None:
                marks[tuple(e["cell"])] = _MARK[e["label"]]
        rows = []
        for v in range(len(elevs) - 1, -1, -1):
            line = "".join(marks.get((h, v), " ") for h in range(n_h))
            rows.append(f"v{v} {elevs[v]:>2.0f}deg |{line}|")
        rows.append(" " * 9 + "h" + "".join(str(h % 10) for h in range(n_h)))
        return head + "\n" + "\n".join(rows)


def contact_sheet(store, labels):
    """Standalone HTML for eyeballing a whole run — for the HUMAN labeller.

    Explicitly NOT for a model: grid-mosaicking frames wrecks VLM accuracy
    (MMNeedle: 10 separate images 97% -> 4x4 grid 26.9%). Models get one
    frame at a time through get_view.
    """
    cards = []
    for v in sorted(store.views(), key=lambda v: v.t):
        e = labels.entries.get(v.cap_dir, {"label": "?", "note": ""})
        cell = f"cell {list(v.cell)}" if v.cell else "survey"
        cards.append(
            f'<figure><img src="../{v.cap_dir}/rgb.png" width="424">'
            f'<figcaption><b>{v.cap_dir}</b> · {cell} · '
            f'<code>{e["label"]}</code> {e["note"]}</figcaption></figure>')
    return ("<!doctype html><meta charset=utf-8>"
            f"<title>{labels.question}</title>"
            "<style>body{font:14px system-ui;background:#111;color:#eee}"
            "figure{display:inline-block;margin:6px}"
            "figcaption{font-size:12px}code{color:#7cf}</style>"
            f"<h2>{labels.question}</h2>" + "".join(cards))


if __name__ == "__main__":
    import sys

    from inspection.eyes.store import FactWriter, RunStore
    from inspection.record.run import Run

    run_dir = Path(sys.argv[1])
    question = sys.argv[2] if len(sys.argv) > 2 else "is there a logo on this cup?"
    # `replay.py` used to hand this CLI a RunStore pre-populated with views;
    # that module is gone (task-5), so rebuild the same shape here — `Run` is
    # the read side now, `RunStore`/`FactWriter` still exist for exactly this
    # kind of ad-hoc, disk-backed view bag (LabelSet/contact_sheet/report all
    # key off `.views()`/`._d["grid"]`/`.path`).
    run = Run.load(run_dir)
    bench_dir = run_dir / "bench"
    store = RunStore.create(bench_dir, h_bins=12, v_elevs=(10.0, 40.0, 70.0),
                            r=0.35)
    facts = FactWriter(store)
    for s in run.captured:
        cell = None if s.id == 0 else tuple(s.record.view.address)
        facts.add_view(cell=cell, pose_id=s.id, cap_dir=s.dir.name,
                       T_base_cam=s.T_base_cam, t=s.record.t_captured)
    try:
        labels = LabelSet.load(bench_dir)
        print(f"loaded existing labels ({labels.pending()} unlabelled)")
    except FileNotFoundError:
        labels = LabelSet.template(store, question)
        print(f"wrote template: {labels.path}")
    sheet = store.path / "contact.html"
    sheet.write_text(contact_sheet(store, labels))
    print(f"contact sheet: {sheet}")
    print(labels.report(store))
    print(f'\nlabel by editing {labels.path}  ("y" | "n" | "partial")')
