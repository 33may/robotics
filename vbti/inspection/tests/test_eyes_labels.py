#!/usr/bin/env python3
"""Ground-truth labels for a run. Run: p inspection/tests/test_eyes_labels.py"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.bench.labels import LabelSet, contact_sheet
from inspection.eyes.store import FactWriter, RunStore

T = np.eye(4)


def _store(tmp):
    root = Path(tmp)
    store = RunStore.create(root, h_bins=12, v_elevs=(10.0, 40.0, 70.0), r=0.35)
    facts = FactWriter(store)
    facts.add_view(cell=None, pose_id=0, cap_dir="000", T_base_cam=T, t=0.0)
    facts.add_view(cell=(3, 1), pose_id=1, cap_dir="001", T_base_cam=T, t=1.0)
    facts.add_view(cell=(4, 1), pose_id=2, cap_dir="002", T_base_cam=T, t=2.0)
    return store


def test_template_covers_every_view_as_unlabelled():
    with tempfile.TemporaryDirectory() as tmp:
        labels = LabelSet.template(_store(tmp), question="is there a logo?")
        assert set(labels.entries) == {"000", "001", "002"}   # keyed by capture
        assert all(e["label"] == "?" for e in labels.entries.values())
        assert labels.entries["001"]["cell"] == [3, 1]
        assert labels.pending() == 3


def test_roundtrip_and_partial_credit_vocabulary():
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        labels = LabelSet.template(store, question="is there a logo?")
        labels.set("001", "y", note="logo centred")
        labels.set("002", "partial", note="fragment at right edge")
        again = LabelSet.load(store.path)
        assert again.question == "is there a logo?"
        assert again.entries["001"]["label"] == "y"
        assert again.entries["002"]["note"] == "fragment at right edge"
        assert again.pending() == 1                      # only the survey left
        assert (store.path / "labels.json").exists()


def test_rejects_labels_outside_the_vocabulary():
    with tempfile.TemporaryDirectory() as tmp:
        labels = LabelSet.template(_store(tmp), question="q")
        try:
            labels.set("001", "maybe")
        except ValueError as e:
            assert "maybe" in str(e)
        else:
            raise AssertionError("bad label accepted")


def test_report_shows_labels_on_the_grid():
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        labels = LabelSet.template(store, question="q")
        labels.set("001", "y"); labels.set("002", "n")
        txt = labels.report(store)
        assert "1 y · 1 n · 0 partial · 1 unlabelled" in txt
        bitmap = txt.split("\n", 1)[1]
        assert bitmap.count("y") == 1 and bitmap.count("n") == 1


def test_contact_sheet_links_every_view():
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        html = contact_sheet(store, LabelSet.template(store, question="q"))
        assert html.count("<img") == 3
        assert "../001/rgb.png" in html and "cell [3, 1]" in html


def main():
    test_template_covers_every_view_as_unlabelled()
    test_roundtrip_and_partial_credit_vocabulary()
    test_rejects_labels_outside_the_vocabulary()
    test_report_shows_labels_on_the_grid()
    test_contact_sheet_links_every_view()
    print("OK test_eyes_labels")


if __name__ == "__main__":
    main()
