"""LimX MROS launch-file extractor.

LimX's ``mroslaunch`` uses a YAML format (not ROS XML launch). One launch file
declares the set of nodes to start under ``mrosnode:``, with per-node fields
``path`` (executable name), ``disabled``, ``oneshot``, ``autostart``, ``argv``.
Some files also carry a top-level ``configure: environment:`` block.

Schema-wise we populate ``launch_nodes`` only. ``node_topics`` and
``node_params`` stay empty for these files because the MROS YAML format does
not declare topic remaps or per-node parameters — those live in YAML configs
under ``etc/<pkg>/``.

ROS-style XML launch files (``.launch.xml`` / ``<launch><node ...>``) are NOT
in this tarball; if a future source root includes them, write a sibling
extractor or extend this one.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from .structured_schema import DOC_ID_TARBALL

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaunchNode:
    name: str
    exec: str
    namespace: str | None
    disabled: bool
    oneshot: bool
    autostart: bool
    argv: str


@dataclass
class LaunchDoc:
    source_uri: str
    pkg: str            # derived from path (etc/<pkg>/...)
    environment: dict[str, str] = field(default_factory=dict)
    nodes: list[LaunchNode] = field(default_factory=list)


@dataclass(frozen=True)
class LaunchChunk:
    doc_id: str
    section: str
    heading: str
    body: str


def _bool(v: object, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "on")
    return default


def parse_launch(path: Path, doc_id: str, section: str) -> LaunchDoc:
    # Some launch files include CJK comments and may use non-UTF-8 encodings.
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    data = yaml.safe_load(text) or {}
    source_uri = f"oli-corpus://{doc_id}#{section}"
    # Pkg = first directory after install/etc/ when possible
    rel = section.replace("install/etc/", "", 1) if section.startswith("install/etc/") else section
    pkg = rel.split("/", 1)[0] if "/" in rel else rel
    doc = LaunchDoc(source_uri=source_uri, pkg=pkg)
    configure = data.get("configure", {}) or {}
    env = configure.get("environment", {}) or {}
    if isinstance(env, dict):
        doc.environment = {str(k): str(v) for k, v in env.items()}
    mrosnode = data.get("mrosnode", {}) or {}
    if isinstance(mrosnode, dict):
        for name, fields in mrosnode.items():
            if not isinstance(fields, dict):
                continue
            doc.nodes.append(
                LaunchNode(
                    name=str(name),
                    exec=str(fields.get("path", "")),
                    namespace=fields.get("namespace"),
                    disabled=_bool(fields.get("disabled"), default=False),
                    oneshot=_bool(fields.get("oneshot"), default=False),
                    autostart=_bool(fields.get("autostart"), default=True),
                    argv=str(fields.get("argv", "")),
                )
            )
    return doc


def markdown_summary(doc: LaunchDoc) -> str:
    lines = [f"# launch: {doc.source_uri}", ""]
    lines.append(f"**Package**: {doc.pkg}")
    lines.append("")
    if doc.environment:
        lines.append("## Environment")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        for k, v in sorted(doc.environment.items()):
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")
    if doc.nodes:
        lines.append("## Nodes")
        lines.append("")
        lines.append("| name | executable | disabled | oneshot | autostart | argv |")
        lines.append("|---|---|---|---|---|---|")
        for n in doc.nodes:
            lines.append(
                f"| {n.name} | `{n.exec}` | {str(n.disabled).lower()} | "
                f"{str(n.oneshot).lower()} | {str(n.autostart).lower()} | "
                f"{n.argv or '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_launch(db: sqlite3.Connection, doc: LaunchDoc) -> None:
    db.execute("DELETE FROM launch_nodes WHERE launch_uri = ?", (doc.source_uri,))
    rows = [
        (
            doc.source_uri,
            doc.pkg,
            n.exec,
            n.name,
            n.namespace,
        )
        for n in doc.nodes
    ]
    db.executemany(
        "INSERT OR REPLACE INTO launch_nodes(launch_uri, pkg, exec, name, namespace) VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def iter_launches(source_root: Path) -> Iterator[Path]:
    for p in sorted(source_root.rglob("*.launch")):
        if "/colcon_install_layout" in p.as_posix():
            continue
        yield p


def relpath_from_root(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def run(db: sqlite3.Connection, tarball_root: Path) -> list[LaunchChunk]:
    chunks: list[LaunchChunk] = []
    for path in iter_launches(tarball_root):
        section = relpath_from_root(path, tarball_root)
        try:
            doc = parse_launch(path, DOC_ID_TARBALL, section)
        except (yaml.YAMLError, OSError, UnicodeDecodeError, ValueError) as exc:
            LOG.error("launch parse failed: %s: %s", path, exc)
            continue
        write_launch(db, doc)
        chunks.append(
            LaunchChunk(
                doc_id=DOC_ID_TARBALL,
                section=section,
                heading=f"launch: {path.name}",
                body=markdown_summary(doc),
            )
        )
    return chunks
