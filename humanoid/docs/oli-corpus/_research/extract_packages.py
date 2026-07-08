"""ROS package.xml extractor for the structured corpus.

Walks every ``package.xml`` under the tarball's ``install/share/`` and similar
roots, populates the ``packages`` and ``pkg_deps`` typed tables, and emits one
markdown chunk per package for FTS.

Dependency tag → ``kind`` normalization:

* ``buildtool_depend``, ``build_depend``       → ``build``
* ``exec_depend``, ``run_depend``              → ``exec``
* ``test_depend``                              → ``test``
* ``depend``                                   → ``depend`` (covers build+exec)

Comment-fenced deps (``<!-- <build_depend>foo</build_depend> -->``) are not
recovered — ElementTree drops them as it should. They were disabled for a
reason.
"""

from __future__ import annotations

import logging
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .structured_schema import DOC_ID_TARBALL

LOG = logging.getLogger(__name__)

# tag → normalized kind
_KIND_MAP = {
    "buildtool_depend": "build",
    "build_depend": "build",
    "exec_depend": "exec",
    "run_depend": "exec",
    "test_depend": "test",
    "depend": "depend",
}


@dataclass(frozen=True)
class Dep:
    pkg: str
    kind: str  # build, exec, test, depend


@dataclass
class PackageDoc:
    name: str
    version: str | None
    description: str
    maintainer: str
    license: str
    source_uri: str
    deps: list[Dep] = field(default_factory=list)


@dataclass(frozen=True)
class PackageChunk:
    doc_id: str
    section: str
    heading: str
    body: str


def _text(elem: ET.Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def parse_package_xml(path: Path, doc_id: str, section: str) -> PackageDoc:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "package":
        raise ValueError(f"{path}: root element is not <package>")
    name = _text(root.find("name"))
    version = _text(root.find("version")) or None
    description = _text(root.find("description"))
    maintainer_elem = root.find("maintainer")
    if maintainer_elem is not None:
        email = maintainer_elem.attrib.get("email", "")
        name_text = _text(maintainer_elem)
        maintainer = f"{name_text} <{email}>" if email else name_text
    else:
        maintainer = ""
    license_text = _text(root.find("license"))
    source_uri = f"oli-corpus://{doc_id}#{section}"

    deps: list[Dep] = []
    for child in root:
        kind = _KIND_MAP.get(child.tag)
        if kind is None:
            continue
        target = _text(child)
        if target:
            deps.append(Dep(pkg=target, kind=kind))

    return PackageDoc(
        name=name,
        version=version,
        description=description,
        maintainer=maintainer,
        license=license_text,
        source_uri=source_uri,
        deps=deps,
    )


def markdown_summary(doc: PackageDoc) -> str:
    lines = [f"# package: {doc.name}", ""]
    if doc.version:
        lines.append(f"**Version**: {doc.version}")
    if doc.license:
        lines.append(f"**License**: {doc.license}")
    if doc.maintainer:
        lines.append(f"**Maintainer**: {doc.maintainer}")
    lines.append(f"**Source**: `{doc.source_uri}`")
    lines.append("")
    if doc.description:
        lines.append(doc.description)
        lines.append("")
    if doc.deps:
        lines.append("## Dependencies")
        lines.append("")
        lines.append("| kind | package |")
        lines.append("|---|---|")
        for d in sorted(doc.deps, key=lambda x: (x.kind, x.pkg)):
            lines.append(f"| {d.kind} | {d.pkg} |")
        lines.append("")
    return "\n".join(lines)


def write_package(db: sqlite3.Connection, doc: PackageDoc) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO packages(name, version, description, maintainer, source_uri)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doc.name, doc.version, doc.description, doc.maintainer, doc.source_uri),
    )
    # Wipe previous deps for this package (we may be re-ingesting from a new tarball).
    db.execute("DELETE FROM pkg_deps WHERE pkg = ?", (doc.name,))
    db.executemany(
        "INSERT OR REPLACE INTO pkg_deps(pkg, dep, kind) VALUES (?, ?, ?)",
        [(doc.name, d.pkg, d.kind) for d in doc.deps],
    )


def iter_package_xmls(source_root: Path) -> Iterator[Path]:
    """Yield every package.xml under the source root, excluding test fixtures."""
    for p in sorted(source_root.rglob("package.xml")):
        rel = p.as_posix()
        if "/test/" in rel or "/colcon_install_layout" in rel:
            continue
        yield p


def relpath_from_root(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def run(db: sqlite3.Connection, tarball_root: Path) -> list[PackageChunk]:
    chunks: list[PackageChunk] = []
    for path in iter_package_xmls(tarball_root):
        section = relpath_from_root(path, tarball_root)
        try:
            doc = parse_package_xml(path, DOC_ID_TARBALL, section)
        except ET.ParseError as exc:
            LOG.error("package.xml parse failed: %s: %s", path, exc)
            continue
        if not doc.name:
            LOG.warning("package.xml missing <name>: %s", path)
            continue
        write_package(db, doc)
        chunks.append(
            PackageChunk(
                doc_id=DOC_ID_TARBALL,
                section=section,
                heading=f"package: {doc.name}",
                body=markdown_summary(doc),
            )
        )
    return chunks
