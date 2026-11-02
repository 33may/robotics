#!/usr/bin/env python3
"""Docs/diagrams generated FROM the schema models — no hand-maintained spec.
Run: p inspection/tests/test_record_docgen.py
"""
import pytest

from inspection.record.docgen import generate, main

ENTITIES = ["RunRecord", "ConfigSnapshot", "StepRecord", "ViewState",
            "ViewMethod", "MenuDef", "AIRunRecord", "AnswerRecord",
            "OperatorEvent", "Manifest", "Segmentation", "GeometryStats"]


def test_every_model_gets_a_section_automatically():
    md = generate()
    for name in ENTITIES:
        assert f"### {name}" in md, name


def test_marked_as_generated():
    assert "GENERATED" in generate().splitlines()[0]


def test_literal_types_render_exactly():
    md = generate()
    assert "0 | 180" in md              # rgb_rotation_deg
    assert "'live' | 'data-engine'" in md  # source


def test_mermaid_has_declared_layout_relations():
    md = generate()
    assert 'RunRecord ||--o{ StepRecord' in md
    assert 'StepRecord ||--|| ViewState' in md


def test_mermaid_derives_composition_from_fields():
    md = generate()
    assert 'StepRecord ||--o| Segmentation' in md   # optional field -> o|
    assert 'RunRecord ||--o{ ViewMethod' in md      # list field -> o{
    assert 'Manifest ||--o{ FileEntry' in md        # dict-valued field -> o{
    assert 'SessionRecord ||--o{ Intrinsics' in md


def test_generation_is_deterministic():
    assert generate() == generate()


def test_main_writes_the_generated_doc(tmp_path):
    out = tmp_path / "schema.md"
    main(out)
    assert out.read_text() == generate()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
