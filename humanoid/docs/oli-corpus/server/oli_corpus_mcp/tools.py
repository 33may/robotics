from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .index import CORPUS_ROOT, EMBEDDING_MODEL, INDEX_PATH, VECTORS_PATH, repo_relative


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
