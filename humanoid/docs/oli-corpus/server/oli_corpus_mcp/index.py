from __future__ import annotations

from pathlib import Path

CORPUS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CORPUS_ROOT.parents[1]
INDEX_PATH = CORPUS_ROOT / "index" / "corpus.sqlite"
VECTORS_PATH = CORPUS_ROOT / "index" / "vectors.npz"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def source_path(repo_path: str) -> Path:
    return REPO_ROOT / repo_path
