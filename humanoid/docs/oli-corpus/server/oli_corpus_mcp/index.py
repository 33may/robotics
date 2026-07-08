from __future__ import annotations

from pathlib import Path

CORPUS_ROOT = Path(__file__).resolve().parents[2]
# CORPUS_ROOT = humanoid/docs/oli-corpus; the repo root is two parents above that
# (.../humanoid is parents[1], the project root is parents[2]).
REPO_ROOT = CORPUS_ROOT.parents[2]
INDEX_PATH = CORPUS_ROOT / "index" / "corpus.sqlite"
VECTORS_PATH = CORPUS_ROOT / "index" / "vectors.npz"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Per-doc-id source root for the structured (code-derived) corpus content.
# Used by raw_file() to resolve oli-corpus:// URIs back to on-disk files.
# Must stay in sync with build_index.STRUCTURED_SOURCE_ROOTS.
STRUCTURED_SOURCE_ROOTS: dict[str, Path] = {
    "oli-main-2.2.12": REPO_ROOT / "humanoid" / "vendor" / "oli-main-software-2.2.12",
    "limxsdk": REPO_ROOT / "humanoid" / "vendor" / "humanoid-mujoco-sim" / "limxsdk-lowlevel" / "include" / "limxsdk",
    "rl-deploy-python": REPO_ROOT / "humanoid" / "vendor" / "humanoid-rl-deploy-python",
}


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def source_path(repo_path: str) -> Path:
    return REPO_ROOT / repo_path
