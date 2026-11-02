"""YAML/JSON config extractor for the structured corpus.

Flat-FTS only — no typed table. Each config file becomes one markdown chunk
whose body is the raw text. Useful for answering questions like:

* "what's the damping_kd for HU_D04_01?"
* "where is the EKF measurement noise tuned?"
* "what's the policy_dir path the walk controller loads?"

Filters: we ingest only configs that humans authored (the on-robot tuning
surface), not the noise that colcon scatters across the install tree.

Tarball includes:
* ``install/share/<pkg>/config/`` — package-shipped configs
* ``install/etc/<pkg>/`` (excluding ``urdf/``, ``meshes/``, ``world/``, ``test/``,
  ``mrosplugins/``) — runtime configs
* ``install/oli/`` — startup configs

Deploy repo includes:
* ``controllers/<robot>/<controller>/`` — controller-specific YAML
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .structured_schema import DOC_ID_TARBALL

LOG = logging.getLogger(__name__)

# Tarball-side: skip these subtrees entirely (binaries, generated env scripts, ROS-internal).
_TARBALL_SKIP_PARTS = (
    "/colcon_install_layout",
    "/__pycache__/",
    "/cmake/",
    "/hook/",
    "/environment/",
)

# Tarball-side: under install/etc/, skip these subtree names (already covered by other extractors
# or not configuration).
_ETC_SKIP_DIRS = {"urdf", "meshes", "world", "test", "mrosplugins", "BT"}


@dataclass(frozen=True)
class ConfigChunk:
    doc_id: str
    section: str
    heading: str
    body: str


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _config_markdown(text: str, lang: str, source_uri: str, kind: str) -> str:
    fence = "```" + lang
    return f"# {kind}: {source_uri}\n\n{fence}\n{text}\n```\n"


def iter_tarball_configs(tarball_root: Path) -> Iterator[Path]:
    install = tarball_root / "install"
    if not install.exists():
        return
    # share/<pkg>/config/
    share = install / "share"
    if share.exists():
        for p in sorted(share.rglob("config/**/*.yaml")):
            yield p
        for p in sorted(share.rglob("config/**/*.json")):
            yield p
    # etc/<pkg>/...
    etc = install / "etc"
    if etc.exists():
        for ext in ("*.yaml", "*.json"):
            for p in sorted(etc.rglob(ext)):
                rel = p.relative_to(etc)
                top = rel.parts[1] if len(rel.parts) >= 2 else ""
                if top in _ETC_SKIP_DIRS:
                    continue
                yield p


def iter_deploy_configs(deploy_root: Path) -> Iterator[Path]:
    controllers = deploy_root / "controllers"
    if not controllers.exists():
        return
    for ext in ("*.yaml", "*.json"):
        for p in sorted(controllers.rglob(ext)):
            yield p


def _should_skip(path: Path) -> bool:
    s = path.as_posix()
    return any(skip in s for skip in _TARBALL_SKIP_PARTS)


def run(
    db: sqlite3.Connection,
    *,
    tarball_root: Path | None,
    deploy_root: Path | None,
) -> list[ConfigChunk]:
    chunks: list[ConfigChunk] = []
    if tarball_root is not None and tarball_root.exists():
        for p in iter_tarball_configs(tarball_root):
            if _should_skip(p):
                continue
            try:
                text = _read(p)
            except OSError as exc:
                LOG.error("config read failed: %s: %s", p, exc)
                continue
            section = p.relative_to(tarball_root).as_posix()
            uri = f"oli-corpus://{DOC_ID_TARBALL}#{section}"
            lang = "yaml" if p.suffix == ".yaml" else "json"
            chunks.append(
                ConfigChunk(
                    doc_id=DOC_ID_TARBALL,
                    section=section,
                    heading=f"config: {p.name}",
                    body=_config_markdown(text, lang, uri, kind="config"),
                )
            )
    if deploy_root is not None and deploy_root.exists():
        for p in iter_deploy_configs(deploy_root):
            try:
                text = _read(p)
            except OSError as exc:
                LOG.error("config read failed: %s: %s", p, exc)
                continue
            section = p.relative_to(deploy_root).as_posix()
            uri = f"oli-corpus://rl-deploy-python#{section}"
            lang = "yaml" if p.suffix == ".yaml" else "json"
            chunks.append(
                ConfigChunk(
                    doc_id="rl-deploy-python",
                    section=section,
                    heading=f"deploy config: {p.name}",
                    body=_config_markdown(text, lang, uri, kind="deploy config"),
                )
            )
    return chunks
