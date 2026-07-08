"""C/C++ header symbol extractor for the structured corpus.

This is a *best-effort* tokenizer that captures top-level type declarations
and their Doxygen docstrings — not a full C++ AST. Goal: make ``find_symbol``
useful for "where is RobotCmd defined and what fields does it have?" without
shipping libclang.

What we extract per header:

* ``struct Name``, ``class Name``, ``enum Name`` / ``enum class Name``
* ``typedef ... Name;`` and ``using Name = ...;``
* Free function prototypes at namespace level (best effort)

For struct/class bodies, the entire body (with comments) is captured as the
signature so an agent can see fields and method prototypes in one query.

Header walking is gated by source roots:

* ``oli-main-2.2.12``: ``install/mbl/include/**/*.{h,hpp}`` and
  ``install/share/*/include/**/*.{h,hpp}``
* ``limxsdk``: every ``.h`` in the vendored include root

Symbols inside ``namespace``/``namespace X { ... }`` blocks are kept; we don't
namespace-qualify the name in the index because the same short name (e.g.
``ImuData``) is what callers grep for. Use ``lib`` + ``source_uri`` to
disambiguate when two libs define the same name.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .structured_schema import DOC_ID_LIMXSDK, DOC_ID_TARBALL

LOG = logging.getLogger(__name__)

DOXYGEN_BLOCK = re.compile(r"/\*\*(?P<body>.*?)\*/", re.DOTALL)
LINE_COMMENT_RUN = re.compile(r"((?:^[ \t]*///[^\n]*\n)+)", re.MULTILINE)

# struct/class/enum at start of line (allowing nesting via brace-counting later)
TYPE_DECL = re.compile(
    r"(?P<kind>struct|class|enum\s+class|enum)\s+"
    r"(?:[A-Z_][A-Z0-9_]*\s+)?"                # optional macro decoration (e.g. EXPORT)
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*:\s*[^{;]+)?"                       # optional inheritance
    r"\s*[{;]"
)

TYPEDEF_DECL = re.compile(
    r"typedef\s+[^;]+?\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;"
)

USING_DECL = re.compile(
    r"using\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=[^;]+;"
)


@dataclass(frozen=True)
class Symbol:
    lib: str
    source_uri: str
    name: str
    kind: str           # class, struct, enum, typedef, using, function
    signature: str
    docstring: str


@dataclass(frozen=True)
class HeaderChunk:
    doc_id: str
    section: str
    heading: str
    body: str


def _strip_block_comments_for_match(text: str) -> str:
    """Remove block comments and string literals so token positions are sane.

    We keep their byte offsets stable by replacing them with same-length
    whitespace, so any subsequent regex match index still maps to the original
    text.
    """
    out = list(text)
    # Block comments
    for m in re.finditer(r"/\*.*?\*/", text, re.DOTALL):
        for i in range(m.start(), m.end()):
            if out[i] != "\n":
                out[i] = " "
    # Line comments
    for m in re.finditer(r"//[^\n]*", text):
        for i in range(m.start(), m.end()):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


def _find_preceding_doc(text: str, decl_start: int) -> str:
    """Return the Doxygen block immediately preceding ``decl_start``, or ``''``."""
    # Look back up to 4 KB
    window = text[max(0, decl_start - 4096):decl_start]
    matches = list(DOXYGEN_BLOCK.finditer(window))
    if matches:
        body = matches[-1].group("body").strip()
        # Strip leading '*' from each line
        cleaned = "\n".join(
            re.sub(r"^\s*\*\s?", "", line) for line in body.splitlines()
        )
        return cleaned.strip()
    # Fall back to a run of '///' comments
    line_runs = list(LINE_COMMENT_RUN.finditer(window))
    if line_runs:
        block = line_runs[-1].group(1)
        cleaned = "\n".join(
            re.sub(r"^\s*///\s?", "", line) for line in block.splitlines() if line.strip()
        )
        return cleaned.strip()
    return ""


def _match_brace_body(text: str, open_brace_pos: int) -> int:
    """Return the index right after the closing brace matching ``text[open_brace_pos] == '{'``."""
    depth = 0
    i = open_brace_pos
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def parse_header(text: str, *, lib: str, source_uri: str) -> list[Symbol]:
    syms: list[Symbol] = []
    sanitized = _strip_block_comments_for_match(text)

    # Pass 1: struct / class / enum (with possible bodies)
    for m in TYPE_DECL.finditer(sanitized):
        name = m.group("name")
        kind = m.group("kind").replace("enum class", "enum").strip()
        # Skip forward decls (;-terminated)
        decl_end = m.end() - 1
        if sanitized[decl_end] == ";":
            signature_end = m.end()
            signature = text[m.start():signature_end].rstrip()
        else:
            # body: brace match
            close = _match_brace_body(sanitized, decl_end)
            if close < 0:
                continue
            signature_end = close
            # Trim huge bodies to avoid 100 KB signatures
            raw = text[m.start():signature_end]
            if len(raw) > 8192:
                raw = raw[:8192].rstrip() + "\n... (truncated)\n}"
            signature = raw
        docstring = _find_preceding_doc(text, m.start())
        syms.append(
            Symbol(
                lib=lib,
                source_uri=source_uri,
                name=name,
                kind=kind,
                signature=signature,
                docstring=docstring,
            )
        )

    # Pass 2: typedefs
    for m in TYPEDEF_DECL.finditer(sanitized):
        name = m.group("name")
        signature = text[m.start():m.end()].strip()
        docstring = _find_preceding_doc(text, m.start())
        syms.append(
            Symbol(
                lib=lib,
                source_uri=source_uri,
                name=name,
                kind="typedef",
                signature=signature,
                docstring=docstring,
            )
        )

    # Pass 3: using alias declarations
    for m in USING_DECL.finditer(sanitized):
        name = m.group("name")
        signature = text[m.start():m.end()].strip()
        docstring = _find_preceding_doc(text, m.start())
        syms.append(
            Symbol(
                lib=lib,
                source_uri=source_uri,
                name=name,
                kind="using",
                signature=signature,
                docstring=docstring,
            )
        )

    return syms


def header_chunk(path: Path, syms: list[Symbol], source_uri: str, doc_id: str, section: str) -> HeaderChunk:
    """Render a markdown chunk listing the symbols defined in this header."""
    lines = [f"# header: {section}", "", f"Source: `{source_uri}`", ""]
    if not syms:
        lines.append("_No top-level symbols extracted._")
    else:
        lines.append("## Symbols")
        lines.append("")
        for s in syms:
            lines.append(f"### `{s.kind} {s.name}`")
            if s.docstring:
                lines.append("")
                # Indent docstring as quote block
                for line in s.docstring.splitlines():
                    lines.append(f"> {line}" if line.strip() else ">")
            lines.append("")
            lines.append("```cpp")
            lines.append(s.signature)
            lines.append("```")
            lines.append("")
    return HeaderChunk(doc_id=doc_id, section=section, heading=f"header: {path.name}", body="\n".join(lines))


def write_symbols(db: sqlite3.Connection, syms: list[Symbol]) -> None:
    if not syms:
        return
    db.executemany(
        """
        INSERT INTO api_symbols(lib, source_uri, symbol, kind, signature, docstring)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(s.lib, s.source_uri, s.name, s.kind, s.signature, s.docstring) for s in syms],
    )


# --------------------------- source root iteration --------------------------- #


def iter_tarball_headers(tarball_root: Path) -> Iterator[Path]:
    for sub in [
        tarball_root / "install" / "mbl" / "include",
    ]:
        if sub.exists():
            for p in sorted(sub.rglob("*.h")):
                yield p
            for p in sorted(sub.rglob("*.hpp")):
                yield p
    share = tarball_root / "install" / "share"
    if share.exists():
        for p in sorted(share.rglob("include/**/*.h")):
            yield p
        for p in sorted(share.rglob("include/**/*.hpp")):
            yield p


def iter_limxsdk_headers(sdk_root: Path) -> Iterator[Path]:
    if not sdk_root.exists():
        return
    for p in sorted(sdk_root.rglob("*.h")):
        yield p
    for p in sorted(sdk_root.rglob("*.hpp")):
        yield p


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def run(
    db: sqlite3.Connection,
    *,
    tarball_root: Path | None,
    limxsdk_root: Path | None,
) -> list[HeaderChunk]:
    """Walk all configured header roots; populate ``api_symbols``; return chunks."""
    chunks: list[HeaderChunk] = []

    if tarball_root is not None and tarball_root.exists():
        for p in iter_tarball_headers(tarball_root):
            try:
                text = _read(p)
            except OSError as exc:
                LOG.error("header read failed: %s: %s", p, exc)
                continue
            section = p.relative_to(tarball_root).as_posix()
            uri = f"oli-corpus://{DOC_ID_TARBALL}#{section}"
            # Derive lib from path: prefer "mbl/<libname>" or "share/<pkg>"
            parts = p.relative_to(tarball_root).parts
            lib = "/".join(parts[:3]) if len(parts) >= 3 else parts[0]
            syms = parse_header(text, lib=lib, source_uri=uri)
            write_symbols(db, syms)
            chunks.append(header_chunk(p, syms, uri, DOC_ID_TARBALL, section))

    if limxsdk_root is not None and limxsdk_root.exists():
        for p in iter_limxsdk_headers(limxsdk_root):
            try:
                text = _read(p)
            except OSError as exc:
                LOG.error("header read failed: %s: %s", p, exc)
                continue
            section = p.relative_to(limxsdk_root).as_posix()
            uri = f"oli-corpus://{DOC_ID_LIMXSDK}#{section}"
            syms = parse_header(text, lib="limxsdk", source_uri=uri)
            write_symbols(db, syms)
            chunks.append(header_chunk(p, syms, uri, DOC_ID_LIMXSDK, section))

    return chunks
