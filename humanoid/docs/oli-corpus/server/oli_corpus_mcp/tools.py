from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

import base64
from pathlib import Path

from .index import (
    CORPUS_ROOT,
    EMBEDDING_MODEL,
    INDEX_PATH,
    STRUCTURED_SOURCE_ROOTS,
    VECTORS_PATH,
    repo_relative,
)


@dataclass(frozen=True)
class Citation:
    doc_id: str
    section: str
    part: int | None = None


def serialize_citation(doc_id: str, section: str, part: int | None = None) -> str:
    uri = f"oli-corpus://{doc_id}#{section}"
    if part is not None:
        uri += f"?part={part}"
    return uri


def parse_citation(uri: str) -> Citation:
    match = re.fullmatch(r"oli-corpus://([^#?]+)#([^?]+)(?:\?part=(\d+))?", uri)
    if not match:
        raise ValueError(f"invalid Oli corpus citation: {uri}")
    return Citation(match.group(1), match.group(2), int(match.group(3)) if match.group(3) else None)


def _connect() -> sqlite3.Connection:
    db = sqlite3.connect(INDEX_PATH)
    db.row_factory = sqlite3.Row
    return db


def _snippet(text: str, query: str, limit: int = 280) -> str:
    words = [w.lower() for w in re.findall(r"\w+", query)]
    lower = text.lower()
    pos = min([lower.find(w) for w in words if lower.find(w) >= 0] or [0])
    start = max(0, pos - 80)
    snippet = re.sub(r"\s+", " ", text[start : start + limit]).strip()
    return snippet + ("..." if start + limit < len(text) else "")


def _row_to_result(row: sqlite3.Row, query: str, search_mode: str, score: float | None = None) -> dict:
    part = row["part"]
    if score is None:
        score = row["score"]
    return {
        "doc_id": row["doc_id"],
        "section": row["section"],
        "part": part,
        "heading": row["heading"],
        "snippet": _snippet(row["body"], query),
        "score": score,
        "citation": serialize_citation(row["doc_id"], row["section"], part),
        "path": repo_relative(CORPUS_ROOT / row["path"]),
        "layer": row["layer"],
        "search_mode": search_mode,
    }


def list_docs() -> list[dict]:
    with _connect() as db:
        rows = db.execute(
            """
            SELECT d.doc_id, d.title, d.fetched_at, COUNT(DISTINCT c.section) AS section_count
            FROM docs d
            JOIN chunks c ON c.doc_id = d.doc_id AND c.layer = 'source'
            GROUP BY d.doc_id
            ORDER BY d.doc_id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _fts_search(query: str, top_k: int, doc_id: str | None, include_notes: bool, search_mode: str) -> list[dict]:
    clauses = ["chunks_fts MATCH ?"]
    params: list[object] = [query]
    if doc_id:
        clauses.append("c.doc_id = ?")
        params.append(doc_id)
    if not include_notes:
        clauses.append("c.layer = 'source'")
    params.append(top_k)
    sql = f"""
        SELECT c.*, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE {' AND '.join(clauses)}
        ORDER BY score
        LIMIT ?
    """
    with _connect() as db:
        rows = db.execute(sql, params).fetchall()
    return [_row_to_result(row, query, search_mode) for row in rows]


def _eligible_chunk_ids(doc_id: str | None, include_notes: bool) -> set[int]:
    clauses: list[str] = []
    params: list[object] = []
    if doc_id:
        clauses.append("doc_id = ?")
        params.append(doc_id)
    if not include_notes:
        clauses.append("layer = 'source'")
    sql = "SELECT id FROM chunks"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    with _connect() as db:
        rows = db.execute(sql, params).fetchall()
    return {int(row["id"]) for row in rows}


def _load_vector_index() -> tuple[np.ndarray, np.ndarray]:
    if not VECTORS_PATH.exists():
        raise RuntimeError(f"vector index missing: {VECTORS_PATH}; run docs/oli-corpus/_research/build_index.py")
    data = np.load(VECTORS_PATH)
    return data["chunk_ids"].astype(np.int64), data["vectors"].astype(np.float32)


def _embed_query(query: str) -> np.ndarray:
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except TypeError:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        raise RuntimeError(f"vector model unavailable locally: {EMBEDDING_MODEL}; rebuild/download the index model") from exc
    vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vector, dtype=np.float32)[0]


def _rows_by_scores(scores: list[tuple[int, float]], query: str, search_mode: str) -> list[dict]:
    if not scores:
        return []
    ids = [chunk_id for chunk_id, _ in scores]
    placeholders = ",".join("?" for _ in ids)
    score_by_id = dict(scores)
    with _connect() as db:
        rows = db.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids).fetchall()
    ordered = sorted(rows, key=lambda row: score_by_id[int(row["id"])], reverse=True)
    results = []
    for row in ordered:
        result = _row_to_result(row, query, search_mode, score_by_id[int(row["id"])])
        results.append(result)
    return results


def _vector_search(query: str, top_k: int, doc_id: str | None, include_notes: bool, search_mode: str) -> list[dict]:
    chunk_ids, vectors = _load_vector_index()
    if len(chunk_ids) != vectors.shape[0]:
        raise RuntimeError("vector index is corrupt: chunk id count does not match vector count")
    eligible = _eligible_chunk_ids(doc_id, include_notes)
    if not eligible:
        return []
    query_vector = _embed_query(query)
    sims = vectors @ query_vector
    ranked: list[tuple[int, float]] = []
    for chunk_id, score in zip(chunk_ids.tolist(), sims.tolist(), strict=True):
        if chunk_id in eligible:
            ranked.append((int(chunk_id), float(score)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return _rows_by_scores(ranked[:top_k], query, search_mode)


def _hybrid_search(query: str, top_k: int, doc_id: str | None, include_notes: bool) -> list[dict]:
    fts = _fts_search(query, top_k * 2, doc_id, include_notes, "hybrid")
    vector = _vector_search(query, top_k * 2, doc_id, include_notes, "hybrid")
    fused: dict[str, tuple[dict, float]] = {}
    for result_set in (fts, vector):
        for rank, result in enumerate(result_set, start=1):
            key = result["citation"]
            previous, score = fused.get(key, (result, 0.0))
            fused[key] = (previous, score + 1.0 / (60 + rank))
    ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:top_k]
    results = []
    for result, score in ranked:
        result = dict(result)
        result["score"] = score
        result["search_mode"] = "hybrid"
        results.append(result)
    return results


def search(
    query: str,
    top_k: int = 10,
    doc_id: str | None = None,
    include_notes: bool = False,
    mode: str = "fts",
) -> list[dict]:
    if mode == "fts":
        return _fts_search(query, top_k, doc_id, include_notes, "fts")
    if mode == "vector":
        return _vector_search(query, top_k, doc_id, include_notes, "vector")
    if mode == "hybrid":
        return _hybrid_search(query, top_k, doc_id, include_notes)
    raise ValueError("mode must be one of: fts, vector, hybrid")


def _resolve_chunk(doc_id: str, section: str, part: int | None) -> sqlite3.Row:
    sql = "SELECT * FROM chunks WHERE doc_id = ? AND section = ? AND layer = 'source'"
    params: list[object] = [doc_id, section]
    if part is None:
        sql += " AND part IS NULL"
    else:
        sql += " AND part = ?"
        params.append(part)
    with _connect() as db:
        row = db.execute(sql, params).fetchone()
    if row is None:
        raise ValueError(f"section not found: {serialize_citation(doc_id, section, part)}")
    return row


def _resolve_images(markdown: str) -> str:
    def repl(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2)
        if src.startswith("images/"):
            return f"![{alt}]({repo_relative(CORPUS_ROOT / 'sources' / src)})"
        return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, markdown)


def get_section(doc_id: str, section: str, part: int | None = None) -> dict:
    row = _resolve_chunk(doc_id, section, part)
    return {
        "doc_id": doc_id,
        "section": section,
        "part": part,
        "heading": row["heading"],
        "body": _resolve_images(row["body"]),
        "citation": serialize_citation(doc_id, section, part),
        "path": repo_relative(CORPUS_ROOT / row["path"]),
        "layer": row["layer"],
    }


def cite(doc_id: str, section: str, part: int | None = None) -> dict:
    row = _resolve_chunk(doc_id, section, part)
    path = CORPUS_ROOT / row["path"]
    current_body = path.read_text(encoding="utf-8")
    warning = None
    if row["body"].strip() not in current_body:
        warning = "indexed chunk text differs from current source file; rebuild the corpus index"
    return {
        "uri": serialize_citation(doc_id, section, part),
        "path": repo_relative(path),
        "anchor": f"#{section}",
        "layer": row["layer"],
        "stability_warning": warning,
    }


def citation_stale(doc_id: str, section: str, part: int | None = None) -> bool:
    row = _resolve_chunk(doc_id, section, part)
    path = CORPUS_ROOT / row["path"]
    current = path.read_text(encoding="utf-8")
    return row["body"].strip() not in current


# ===========================================================================
# Structured-corpus tools (URDF / typed-table backed)
# ===========================================================================


def robots() -> list[dict]:
    """List all robots known to the corpus, with their URDF source URI.

    Returns one row per indexed URDF/SRDF file. ``robot_id`` is unique;
    RL-variant URDFs are suffixed with ``_rl`` for disambiguation.
    """
    with _connect() as db:
        rows = db.execute(
            "SELECT robot_id, description, source_uri FROM robots ORDER BY robot_id"
        ).fetchall()
    return [dict(r) for r in rows]


def joints(robot_id: str, include_fixed: bool = False) -> list[dict]:
    """Return the joint table for one robot, ordered by ``urdf_idx``.

    ``urdf_idx`` is declaration order across ALL joints in the URDF.
    ``dof_idx`` is the sequential index across non-fixed joints (matches the
    IsaacLab articulation-builder DoF order). ``sdk_idx`` is NULL until the
    ``extract_sdk_joint_order`` extractor backfills it.

    Set ``include_fixed=True`` to include fixed (welded) joints; by default
    they are filtered out since downstream consumers usually want DoFs.
    """
    sql = (
        "SELECT urdf_idx, sdk_idx, name, type, parent_link, child_link, "
        "axis_x, axis_y, axis_z, lower, upper, effort, velocity, mimic_of, source_uri "
        "FROM joints WHERE robot_id = ?"
    )
    params: list[object] = [robot_id]
    if not include_fixed:
        sql += " AND type != 'fixed'"
    sql += " ORDER BY urdf_idx"
    with _connect() as db:
        rows = db.execute(sql, params).fetchall()
    if not rows:
        with _connect() as db:
            exists = db.execute("SELECT 1 FROM robots WHERE robot_id = ?", (robot_id,)).fetchone()
        if exists is None:
            raise ValueError(
                f"unknown robot: {robot_id!r}. Call robots() to list available robot_ids."
            )
    results = []
    dof_idx = 0
    for r in rows:
        row = dict(r)
        if row["type"] != "fixed":
            row["dof_idx"] = dof_idx
            dof_idx += 1
        else:
            row["dof_idx"] = None
        row["axis"] = (
            None
            if row["axis_x"] is None
            else (row["axis_x"], row["axis_y"], row["axis_z"])
        )
        for k in ("axis_x", "axis_y", "axis_z"):
            row.pop(k)
        results.append(row)
    return results


def links(robot_id: str) -> list[dict]:
    """Return the link table for one robot.

    Includes mass, COM (``com_x/y/z``), inertia tensor (``ixx ixy ixz iyy iyz izz``),
    and ``oli-corpus://`` URIs for visual and collision mesh files. The mesh URIs
    can be passed to ``raw_file()`` to retrieve the actual STL/OBJ bytes.
    """
    with _connect() as db:
        rows = db.execute(
            "SELECT name, mass, com_x, com_y, com_z, ixx, ixy, ixz, iyy, iyz, izz, "
            "visual_mesh_uri, collision_mesh_uri, source_uri "
            "FROM links WHERE robot_id = ? ORDER BY name",
            (robot_id,),
        ).fetchall()
    if not rows:
        with _connect() as db:
            exists = db.execute("SELECT 1 FROM robots WHERE robot_id = ?", (robot_id,)).fetchone()
        if exists is None:
            raise ValueError(
                f"unknown robot: {robot_id!r}. Call robots() to list available robot_ids."
            )
    return [dict(r) for r in rows]


# ---- raw_file: universal escape hatch -------------------------------------


_BINARY_EXTS = {
    ".stl", ".obj", ".dae", ".ply", ".png", ".jpg", ".jpeg", ".gif", ".bmp",
    ".onnx", ".rknn", ".bin", ".so", ".wav", ".mp3", ".pyc", ".7z", ".gz",
    ".tar", ".zip", ".pdf",
}


def _resolve_source_uri(source_uri: str) -> tuple[str, str, Path]:
    """Parse an ``oli-corpus://<doc_id>#<section>`` URI back to an on-disk path.

    Raises ``ValueError`` if the URI is malformed, the doc_id is not a known
    structured source root, the section path escapes the source root, or the
    file does not exist.
    """
    cit = parse_citation(source_uri)
    if cit.part is not None:
        raise ValueError(
            f"raw_file does not accept ?part=N URIs; pass the section-level URI: "
            f"oli-corpus://{cit.doc_id}#{cit.section}"
        )
    root = STRUCTURED_SOURCE_ROOTS.get(cit.doc_id)
    if root is None:
        raise ValueError(
            f"raw_file only resolves URIs whose doc_id is a structured source root "
            f"({', '.join(sorted(STRUCTURED_SOURCE_ROOTS))}); got {cit.doc_id!r}. "
            f"Use get_section() for webpage docs (quick-start, sdk-guide, user-manual)."
        )
    abs_path = (root / cit.section).resolve()
    # Refuse symlink escapes / .. traversal — the resolved path must sit under root.
    try:
        abs_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"raw_file refused: resolved path {abs_path} is not under source root {root}"
        ) from exc
    if not abs_path.exists():
        raise ValueError(
            f"raw_file: file does not exist at resolved path {abs_path}. "
            f"Was the corpus rebuilt after the source moved? Re-run build_index."
        )
    return cit.doc_id, cit.section, abs_path


def raw_file(source_uri: str, max_bytes: int = 1_048_576) -> dict:
    """Return the raw contents of an indexed file by its ``oli-corpus://`` URI.

    Text files (``.urdf``, ``.srdf``, ``.xml``, ``.h``, ``.hpp``, ``.yaml``,
    ``.json``, ``.py``, ``.md``, ``.txt``, ``.launch``, ``.cmake``) come back
    as ``{"encoding": "utf-8", "content": "..."}``.

    Binary files (``.STL``, ``.png``, ``.onnx``, ``.rknn``, …) come back as
    ``{"encoding": "base64", "content": "..."}``.

    ``max_bytes`` bounds the response to avoid pulling a 1 GB mesh through the
    MCP channel by accident; on truncation the response includes a
    ``truncated`` flag.
    """
    doc_id, section, abs_path = _resolve_source_uri(source_uri)
    size = abs_path.stat().st_size
    is_binary = abs_path.suffix.lower() in _BINARY_EXTS
    truncated = size > max_bytes
    with abs_path.open("rb") as f:
        data = f.read(max_bytes)
    if is_binary:
        content = base64.b64encode(data).decode("ascii")
        encoding = "base64"
    else:
        try:
            content = data.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(data).decode("ascii")
            encoding = "base64"
    return {
        "uri": source_uri,
        "doc_id": doc_id,
        "section": section,
        "path": repo_relative(abs_path),
        "size_bytes": size,
        "encoding": encoding,
        "content": content,
        "truncated": truncated,
    }


def nodes(
    launch_uri: str | None = None,
    pkg: str | None = None,
) -> list[dict]:
    """List launch-declared nodes filtered by launch file URI or package.

    Either ``launch_uri`` (an ``oli-corpus://`` URI from ``robots()``/search
    results) OR ``pkg`` (an exec-package name) can be passed; if both are
    provided, both filters apply. With no filters, returns all nodes (148+
    rows).
    """
    clauses: list[str] = []
    params: list[object] = []
    if launch_uri:
        clauses.append("launch_uri = ?")
        params.append(launch_uri)
    if pkg:
        clauses.append("pkg = ?")
        params.append(pkg)
    sql = "SELECT launch_uri, pkg, exec, name, namespace FROM launch_nodes"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY launch_uri, name"
    with _connect() as db:
        rows = db.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def topics(
    node: str | None = None,
    kind: str | None = None,
    topic: str | None = None,
) -> list[dict]:
    """Return topic-graph rows filtered by node name, kind, or topic name.

    ``kind`` ∈ ``{"pub", "sub", "remap", "srv-client", "srv-server"}``.

    Note: LimX's MROS YAML launch format does NOT declare topic remaps or
    pub/sub edges — those are compiled into the node binaries. This table
    will be empty for any node whose launch file is from the MROS tarball.
    It becomes meaningful when a future source root (e.g. a ROS XML launch
    bundle, or runtime topic-graph snapshots) is added.
    """
    clauses: list[str] = []
    params: list[object] = []
    if node:
        clauses.append("node = ?")
        params.append(node)
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if topic:
        clauses.append("topic = ?")
        params.append(topic)
    sql = "SELECT launch_uri, node, kind, topic, remap_from FROM node_topics"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY node, topic"
    with _connect() as db:
        rows = db.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def sdk_joint_order(robot_id: str) -> dict:
    """Return the canonical joint name order for the SDK's ``q``/``dq``/``tau`` arrays.

    The order is recovered from the on-robot deploy reference's
    ``walk_param.yaml`` (comment-annotated arrays) and matched against the URDF.
    Returns ``{"robot_id", "order": [name, ...], "source_uri"}`` on success.

    Raises ``ValueError`` if no canonical order has been resolved for this
    robot — meaning the deploy repo doesn't cover it, or the joint names in the
    deploy config don't match the URDF. We never silently guess the mapping.
    """
    with _connect() as db:
        rows = db.execute(
            "SELECT name, source_uri FROM joints "
            "WHERE robot_id = ? AND sdk_idx IS NOT NULL AND type != 'fixed' "
            "ORDER BY sdk_idx",
            (robot_id,),
        ).fetchall()
        if not rows:
            exists = db.execute("SELECT 1 FROM robots WHERE robot_id = ?", (robot_id,)).fetchone()
            if exists is None:
                raise ValueError(
                    f"unknown robot: {robot_id!r}. Call robots() to list available robot_ids."
                )
            raise ValueError(
                f"SDK joint order is not resolved for {robot_id!r}. "
                f"No matching walk_param.yaml in humanoid-rl-deploy-python/controllers/{robot_id}/, "
                f"or the deploy config's joint names don't match the URDF. "
                f"The mapping must NOT be guessed — verify the deploy repo, then rebuild the index."
            )
    return {
        "robot_id": robot_id,
        "order": [r["name"] for r in rows],
        "source_uri": rows[0]["source_uri"],
    }


def find_symbol(
    query: str,
    kind: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Look up C/C++ symbols by name across all indexed headers.

    ``query`` does an exact match first, falling back to substring match if
    no exact hits. ``kind`` filters by ``class``, ``struct``, ``enum``,
    ``typedef``, ``using``, or ``function``.

    Each row returns the symbol name, kind, lib (path-derived), source URI,
    Doxygen docstring (if any), and the full struct/class body or prototype.
    """
    with _connect() as db:
        # Try exact name first
        sql = "SELECT lib, source_uri, symbol, kind, signature, docstring FROM api_symbols WHERE symbol = ?"
        params: list[object] = [query]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY lib, symbol LIMIT ?"
        params.append(limit)
        rows = db.execute(sql, params).fetchall()
        if rows:
            return [dict(r) for r in rows]
        # Fallback: substring
        sql = (
            "SELECT lib, source_uri, symbol, kind, signature, docstring "
            "FROM api_symbols WHERE symbol LIKE ?"
        )
        params = [f"%{query}%"]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY lib, symbol LIMIT ?"
        params.append(limit)
        rows = db.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def pkg_info(name: str) -> dict:
    """Return package metadata plus dependency edges (both directions).

    The ``deps`` list contains packages this package depends on.
    The ``dependents`` list contains packages that depend on this one — useful
    for impact analysis when changing a low-level package.
    """
    with _connect() as db:
        meta = db.execute(
            "SELECT name, version, description, maintainer, source_uri FROM packages WHERE name = ?",
            (name,),
        ).fetchone()
        if meta is None:
            raise ValueError(
                f"unknown package: {name!r}. Use search() with a partial name to find candidates."
            )
        deps = db.execute(
            "SELECT dep, kind FROM pkg_deps WHERE pkg = ? ORDER BY kind, dep",
            (name,),
        ).fetchall()
        dependents = db.execute(
            "SELECT pkg, kind FROM pkg_deps WHERE dep = ? ORDER BY kind, pkg",
            (name,),
        ).fetchall()
    return {
        **dict(meta),
        "deps": [dict(d) for d in deps],
        "dependents": [dict(d) for d in dependents],
    }
