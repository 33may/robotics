"""Build the Oli documentation SQLite FTS index."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from . import (
    extract_configs,
    extract_headers,
    extract_launch,
    extract_packages,
    extract_sdk_joint_order,
    extract_urdf,
    structured_schema,
)
from .structured_schema import DOC_ID_LIMXSDK, DOC_ID_TARBALL

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources"
NOTES = ROOT / "notes"
INDEX = ROOT / "index" / "corpus.sqlite"
VECTORS = ROOT / "index" / "vectors.npz"
TOKEN_LIMIT = 500
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Repo root: ROOT is humanoid/docs/oli-corpus; repo root is two parents up.
REPO_ROOT = ROOT.parents[2]

# Source roots for the structured extractors. Each maps doc_id → vendored dir.
STRUCTURED_SOURCE_ROOTS: dict[str, Path] = {
    DOC_ID_TARBALL: REPO_ROOT / "humanoid" / "vendor" / "oli-main-software-2.2.12",
    "limxsdk": REPO_ROOT / "humanoid" / "vendor" / "humanoid-mujoco-sim" / "limxsdk-lowlevel" / "include" / "limxsdk",
    "rl-deploy-python": REPO_ROOT / "humanoid" / "vendor" / "humanoid-rl-deploy-python",
}

DOC_ID_BY_FILE = {
    "LimX_EDU_Quick_Start_Guide.md": "quick-start",
    "Oli_EDU_User_Manual.md": "user-manual",
    "Oli_EDU_SDK_Development_Guide.md": "sdk-guide",
}


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section: str
    part: int | None
    heading: str
    body: str
    layer: str
    path: str
    section_sha: str
    chunk_sha: str


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def heading_section(text: str, fallback: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)*)\b", text)
    return match.group(1) if match else fallback


def markdown_sections(path: Path, doc_id: str, layer: str) -> list[tuple[str, str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: list[tuple[int, int, str, str]] = []
    fallback = 1
    for idx, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            section = heading_section(heading, f"note-{fallback}" if layer == "note" else f"section-{fallback}")
            fallback += 1
            headings.append((idx, level, heading, section))
    if not headings:
        return [("intro", path.stem, "\n".join(lines).strip() + "\n")]

    sections: list[tuple[str, str, str]] = []
    for i, (start, level, heading, section) in enumerate(headings):
        end = len(lines)
        for next_start, next_level, _, _ in headings[i + 1 :]:
            if next_level <= level:
                end = next_start
                break
        body = "\n".join(lines[start:end]).strip() + "\n"
        sections.append((section, heading, body))
    return [s for s in sections if s[2].strip()]


def split_body(body: str) -> list[str]:
    if token_count(body) <= TOKEN_LIMIT:
        return [body]
    paragraphs = re.split(r"\n\s*\n", body.strip())
    chunks: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        candidate = "\n\n".join([*current, para]).strip()
        if current and token_count(candidate) > TOKEN_LIMIT:
            chunks.append("\n\n".join(current).strip() + "\n")
            current = [para]
        else:
            current.append(para)
    if current:
        chunks.append("\n\n".join(current).strip() + "\n")
    return chunks


def chunks_for_file(path: Path, doc_id: str, layer: str) -> list[Chunk]:
    rel = path.relative_to(ROOT).as_posix()
    chunks: list[Chunk] = []
    for section, heading, body in markdown_sections(path, doc_id, layer):
        section_sha = sha(body)
        parts = split_body(body)
        for i, part_body in enumerate(parts, start=1):
            part = i if len(parts) > 1 else None
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    section=section,
                    part=part,
                    heading=heading,
                    body=part_body,
                    layer=layer,
                    path=rel,
                    section_sha=section_sha,
                    chunk_sha=sha(part_body),
                )
            )
    return chunks


def source_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(SOURCES.glob("*.md")):
        doc_id = DOC_ID_BY_FILE.get(path.name)
        if doc_id:
            chunks.extend(chunks_for_file(path, doc_id, "source"))
    return chunks


def note_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    if not NOTES.exists():
        return chunks
    for path in sorted(NOTES.glob("*.md")):
        doc_id = f"note:{path.stem}"
        chunks.extend(chunks_for_file(path, doc_id, "note"))
    return chunks


def _structured_chunk(uchunk: object, source_root: Path) -> Chunk:
    """Convert any extractor's emitted chunk dataclass into a Chunk row.

    All structured extractor chunk types share the same shape (doc_id, section,
    heading, body) so we duck-type on attribute access.
    """
    doc_id: str = uchunk.doc_id  # type: ignore[attr-defined]
    section: str = uchunk.section  # type: ignore[attr-defined]
    heading: str = uchunk.heading  # type: ignore[attr-defined]
    body: str = uchunk.body  # type: ignore[attr-defined]
    rel_to_repo = (source_root / section).relative_to(REPO_ROOT).as_posix()
    section_sha = sha(body)
    return Chunk(
        doc_id=doc_id,
        section=section,
        part=None,
        heading=heading,
        body=body,
        layer="source",
        path=rel_to_repo,
        section_sha=section_sha,
        chunk_sha=section_sha,
    )


def structured_chunks(db: sqlite3.Connection) -> list[Chunk]:
    """Run all structured extractors, return chunks for FTS insertion.

    Each extractor writes its typed tables directly; the returned chunks go
    through the regular chunks/FTS insert path.
    """
    chunks: list[Chunk] = []
    tarball_root = STRUCTURED_SOURCE_ROOTS[DOC_ID_TARBALL]
    limxsdk_root = STRUCTURED_SOURCE_ROOTS[DOC_ID_LIMXSDK]

    if not tarball_root.exists():
        print(f"warning: tarball root not present, skipping tarball extraction: {tarball_root}")
    else:
        for uc in extract_urdf.run(db, tarball_root):
            chunks.append(_structured_chunk(uc, tarball_root))
        for pc in extract_packages.run(db, tarball_root):
            chunks.append(_structured_chunk(pc, tarball_root))
        for lc in extract_launch.run(db, tarball_root):
            chunks.append(_structured_chunk(lc, tarball_root))

    # Headers walk both tarball and limxsdk roots, so call once.
    header_chunks = extract_headers.run(
        db,
        tarball_root=tarball_root if tarball_root.exists() else None,
        limxsdk_root=limxsdk_root if limxsdk_root.exists() else None,
    )
    for hc in header_chunks:
        # Each HeaderChunk knows its own doc_id; map back to its source root.
        source_root = STRUCTURED_SOURCE_ROOTS[hc.doc_id]
        chunks.append(_structured_chunk(hc, source_root))

    # Configs: flat FTS only, no typed table.
    deploy_root = STRUCTURED_SOURCE_ROOTS["rl-deploy-python"]
    config_chunks = extract_configs.run(
        db,
        tarball_root=tarball_root if tarball_root.exists() else None,
        deploy_root=deploy_root if deploy_root.exists() else None,
    )
    for cc in config_chunks:
        source_root = STRUCTURED_SOURCE_ROOTS[cc.doc_id]
        chunks.append(_structured_chunk(cc, source_root))

    # SDK joint order: depends on URDF rows already being in `joints`, so run last.
    extract_sdk_joint_order.run(db, deploy_root)
    return chunks


def doc_metadata() -> list[tuple[str, str, str, str, str]]:
    manifest = yaml.safe_load((SOURCES / "_meta" / "manifest.yaml").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["docs"]:
        rows.append((entry["doc_id"], entry["title"], entry["fetched_at"], entry["filename"], entry["source_url"]))
    return rows


def build_vectors(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT id, heading, body FROM chunks ORDER BY id").fetchall()
    if not rows:
        raise SystemExit("no chunks available for vector index")
    chunk_ids = np.array([row[0] for row in rows], dtype=np.int64)
    texts = [f"{row[1]}\n\n{row[2]}" for row in rows]
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if len(chunk_ids) != vectors.shape[0]:
        raise SystemExit(f"vector/chunk count mismatch: chunks={len(chunk_ids)} vectors={vectors.shape[0]}")
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise SystemExit("vector index contains non-normalized embeddings")
    np.savez_compressed(VECTORS, chunk_ids=chunk_ids, vectors=vectors, model=np.array(EMBEDDING_MODEL))


def build() -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    webpage_rows = source_chunks() + note_chunks()
    with sqlite3.connect(INDEX) as db:
        db.executescript(
            """
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS chunks_fts;
            DROP TABLE IF EXISTS docs;
            DROP TABLE IF EXISTS vector_index;
            CREATE TABLE docs(
              doc_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              fetched_at TEXT NOT NULL,
              filename TEXT NOT NULL,
              source_url TEXT NOT NULL
            );
            CREATE TABLE chunks(
              id INTEGER PRIMARY KEY,
              doc_id TEXT NOT NULL,
              section TEXT NOT NULL,
              part INTEGER,
              heading TEXT NOT NULL,
              body TEXT NOT NULL,
              layer TEXT NOT NULL CHECK(layer IN ('source','note')),
              path TEXT NOT NULL,
              section_sha TEXT NOT NULL,
              chunk_sha TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
              heading,
              body,
              content='chunks',
              content_rowid='id'
            );
            """
        )
        # Install structured typed tables (robots, joints, links, packages, ...)
        structured_schema.install(db)

        # Insert all doc metadata (existing webpages + structured source roots)
        db.executemany("INSERT INTO docs VALUES (?, ?, ?, ?, ?)", doc_metadata())
        structured_schema.insert_structured_doc_metadata(db)

        # Run structured extractors — they populate typed tables AND return FTS chunks.
        structured_rows = structured_chunks(db)
        all_rows = webpage_rows + structured_rows

        db.executemany(
            """
            INSERT INTO chunks(doc_id, section, part, heading, body, layer, path, section_sha, chunk_sha)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(c.doc_id, c.section, c.part, c.heading, c.body, c.layer, c.path, c.section_sha, c.chunk_sha) for c in all_rows],
        )
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        db.execute(
            "CREATE TABLE vector_index(model TEXT NOT NULL, path TEXT NOT NULL, chunk_count INTEGER NOT NULL)"
        )
        build_vectors(db)
        db.execute("INSERT INTO vector_index VALUES (?, ?, ?)", (EMBEDDING_MODEL, VECTORS.name, len(all_rows)))

    by_doc: dict[str, tuple[int, set[str]]] = {}
    for c in all_rows:
        if c.layer != "source":
            continue
        count, sections = by_doc.get(c.doc_id, (0, set()))
        sections.add(c.section)
        by_doc[c.doc_id] = (count + 1, sections)
    for doc_id, (chunk_count, sections) in sorted(by_doc.items()):
        section_count = len(sections)
        if not (section_count <= chunk_count <= section_count * 4):
            raise SystemExit(f"implausible chunk count for {doc_id}: sections={section_count} chunks={chunk_count}")
        print(f"{doc_id}: sections={section_count} chunks={chunk_count}")


if __name__ == "__main__":
    build()
