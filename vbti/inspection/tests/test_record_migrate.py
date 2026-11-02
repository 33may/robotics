#!/usr/bin/env python3
"""Migration gate — add-only, recorded, idempotent, dry-run first.
Run: p inspection/tests/test_record_migrate.py
"""
import json

import pytest

from inspection.record.migrate import Migration, migrate_run
from inspection.record.schema import RunRecord
from inspection.tests.record_fixtures import make_run


def _mig(counter):
    def applies(run_dir):
        return not (run_dir / "extra.json").exists()

    def apply(run_dir):
        counter["n"] += 1
        (run_dir / "extra.json").write_text(json.dumps({"new": True}))
        return ["extra.json"]

    return Migration(id="m001-add-extra", description="adds extra.json",
                     applies=applies, apply=apply)


def test_run_record_carries_migrations_list(tmp_path):
    d = make_run(tmp_path)
    run = RunRecord.model_validate_json((d / "run.json").read_text())
    assert run.migrations == []


def test_dry_run_reports_without_touching(tmp_path):
    d = make_run(tmp_path)
    counter = {"n": 0}
    report = migrate_run(d, [_mig(counter)], dry_run=True)
    assert report == [("m001-add-extra", "pending")]
    assert counter["n"] == 0 and not (d / "extra.json").exists()


def test_apply_records_and_is_idempotent(tmp_path):
    d = make_run(tmp_path)
    counter = {"n": 0}
    assert migrate_run(d, [_mig(counter)], dry_run=False) == \
        [("m001-add-extra", "applied")]
    run = RunRecord.model_validate_json((d / "run.json").read_text())
    assert run.migrations == ["m001-add-extra"]
    # second pass: recorded id short-circuits, apply() not called again
    assert migrate_run(d, [_mig(counter)], dry_run=False) == \
        [("m001-add-extra", "skipped")]
    assert counter["n"] == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
