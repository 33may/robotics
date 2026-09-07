#!/usr/bin/env python3
"""Generate the schema reference (field tables + mermaid ER diagram) FROM the
pydantic models in `inspection.record.schema` — the models are the only legal
source for schema docs, so this file writes and the humans never edit.

Run: p inspection/record/docgen.py  ->  docs/data-engine/generated/schema.md
"""
from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Annotated, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from inspection.record import schema

OUT_PATH = (Path(__file__).resolve().parents[1]
            / "docs" / "data-engine" / "generated" / "schema.md")


def _models() -> list[type[BaseModel]]:
    """Every model defined in the schema module, in definition order."""
    return [obj for obj in vars(schema).values()
            if isinstance(obj, type) and issubclass(obj, BaseModel)
            and obj.__module__ == schema.__name__
            and obj is not schema.RecordModel]


def _tname(t) -> str:
    """Exact, human-readable type string for the field tables."""
    if t is type(None):
        return "None"
    origin = get_origin(t)
    if origin is Literal:
        return " | ".join(repr(a) for a in get_args(t))
    if origin is Annotated:
        return _tname(get_args(t)[0])
    if origin in (Union, types.UnionType):
        return " | ".join(_tname(a) for a in get_args(t))
    if origin in (list, set, tuple):
        args = get_args(t)
        return f"list[{_tname(args[0])}]" if args else "list"
    if origin is dict:
        k, v = get_args(t)
        return f"dict[{_tname(k)}, {_tname(v)}]"
    if isinstance(t, type) and issubclass(t, BaseModel):
        return t.__name__
    return getattr(t, "__name__", str(t))


def _coarse(t) -> str:
    """Mermaid-safe attribute type token (details live in the tables)."""
    origin = get_origin(t)
    if origin is Literal:
        return "enum"
    if origin is Annotated:
        return _coarse(get_args(t)[0])
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(t) if a is not type(None)]
        return _coarse(args[0]) if args else "json"
    if origin in (list, set, tuple):
        return "list"
    if origin is dict:
        return "dict"
    if isinstance(t, type):
        return t.__name__ if t.__name__.isidentifier() else "json"
    return "json"


def _field_refs(ann) -> list[tuple[type[BaseModel], str]]:
    """Schema models referenced by a field annotation -> (model, cardinality)."""
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return [(ann, "||--||")]
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        args = get_args(ann)
        card = "||--o|" if type(None) in args else "||--||"
        return [(a, card) for a in args
                if isinstance(a, type) and issubclass(a, BaseModel)]
    if origin in (list, set, tuple):
        args = get_args(ann)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return [(args[0], "||--o{")]
    if origin is dict:
        args = get_args(ann)
        if len(args) == 2 and isinstance(args[1], type) \
                and issubclass(args[1], BaseModel):
            return [(args[1], "||--o{")]
    return []


def _mermaid(models: list[type[BaseModel]]) -> str:
    lines = ["```mermaid", "erDiagram"]
    seen: set[tuple[str, str]] = set()
    for a, card, b, label in schema.RELATIONS:
        lines.append(f'    {a} {card} {b} : "{label}"')
        seen.add((a, b))
    for m in models:
        for fname, f in m.model_fields.items():
            for ref, card in _field_refs(f.annotation):
                if (m.__name__, ref.__name__) not in seen:
                    lines.append(f'    {m.__name__} {card} {ref.__name__} : "{fname}"')
                    seen.add((m.__name__, ref.__name__))
    for m in models:
        attrs = [f"        {_coarse(f.annotation)} {fname}"
                 for fname, f in m.model_fields.items()
                 if not _field_refs(f.annotation)]
        if attrs:
            lines.append(f"    {m.__name__} {{")
            lines.extend(attrs)
            lines.append("    }")
    lines.append("```")
    return "\n".join(lines)


def _table(m: type[BaseModel]) -> str:
    rows = ["| field | type | required | default |", "|---|---|---|---|"]
    for fname, f in m.model_fields.items():
        if f.is_required():
            req, default = "yes", "—"
        else:
            req = "no"
            if f.default_factory is not None:
                default = f"`{f.default_factory().__class__.__name__}()`"
            elif f.default is PydanticUndefined:
                default = "—"
            else:
                default = f"`{f.default!r}`"
        rows.append(f"| `{fname}` | `{_tname(f.annotation)}` | {req} | {default} |")
    return "\n".join(rows)


def generate() -> str:
    models = _models()
    parts = [
        "<!-- GENERATED from inspection/record/schema.py — do not edit. "
        "Regenerate: p inspection/record/docgen.py -->",
        "",
        "# Single Recording schema — generated reference",
        "",
        f"`schema_version` = **{schema.SCHEMA_VERSION}**. "
        "Source of truth: `inspection/record/schema.py` "
        "(validated by `inspection/tests/test_record_schema.py`).",
        "",
        "## ER diagram",
        "",
        _mermaid(models),
        "",
        "## Entities",
    ]
    for m in models:
        doc = inspect.cleandoc(m.__doc__ or "")
        parts += ["", f"### {m.__name__}", "", doc, "", _table(m)]
    return "\n".join(parts) + "\n"


def main(out: Path = OUT_PATH) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate())
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_PATH)
