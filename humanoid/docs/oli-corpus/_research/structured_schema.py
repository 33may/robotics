"""Schema additions for the structured Oli corpus.

These tables coexist with the existing ``chunks`` + FTS index. Every row carries
``source_uri`` (an ``oli-corpus://`` URI) so structured answers cite the same way
free-text answers do.

The DDL is intentionally idempotent enough that we DROP-IF-EXISTS and recreate
on every build. Backfilling sdk_idx is handled by a separate extractor.
"""

from __future__ import annotations

import sqlite3

# doc_ids used by the tarball + companion vendored repos.
DOC_ID_TARBALL = "oli-main-2.2.12"
DOC_ID_LIMXSDK = "limxsdk"
DOC_ID_RL_DEPLOY = "rl-deploy-python"

STRUCTURED_DOCS = [
    (
        DOC_ID_TARBALL,
        "Oli Main Software v2.2.12 (EDU Ed.)",
        "2026-06-22",
        "humanoid/vendor/oli-main-software-2.2.12/install/",
        "https://limx.cn/en/products/oli/download",
    ),
    (
        DOC_ID_LIMXSDK,
        "LimX SDK low-level C++ headers",
        "2026-06-22",
        "humanoid/vendor/humanoid-mujoco-sim/limxsdk-lowlevel/include/limxsdk/",
        "https://github.com/limxdynamics/humanoid-mujoco-sim",
    ),
    (
        DOC_ID_RL_DEPLOY,
        "Oli RL deploy reference (Python)",
        "2026-06-22",
        "humanoid/vendor/humanoid-rl-deploy-python/",
        "https://github.com/limxdynamics/humanoid-rl-deploy-python",
    ),
]


DDL = """
DROP TABLE IF EXISTS joints;
DROP TABLE IF EXISTS links;
DROP TABLE IF EXISTS robots;
DROP TABLE IF EXISTS pkg_deps;
DROP TABLE IF EXISTS packages;
DROP TABLE IF EXISTS node_params;
DROP TABLE IF EXISTS node_topics;
DROP TABLE IF EXISTS launch_nodes;
DROP TABLE IF EXISTS api_symbols;
DROP TABLE IF EXISTS file_index;

CREATE TABLE robots(
  robot_id TEXT PRIMARY KEY,
  description TEXT,
  source_uri TEXT NOT NULL
);

CREATE TABLE joints(
  robot_id TEXT NOT NULL,
  urdf_idx INTEGER NOT NULL,
  sdk_idx  INTEGER,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  parent_link TEXT NOT NULL,
  child_link TEXT NOT NULL,
  axis_x REAL, axis_y REAL, axis_z REAL,
  lower REAL, upper REAL,
  effort REAL, velocity REAL,
  mimic_of TEXT,
  source_uri TEXT NOT NULL,
  PRIMARY KEY (robot_id, urdf_idx),
  FOREIGN KEY (robot_id) REFERENCES robots(robot_id)
);
CREATE INDEX joints_name ON joints(name);

CREATE TABLE links(
  robot_id TEXT NOT NULL,
  name TEXT NOT NULL,
  mass REAL,
  com_x REAL, com_y REAL, com_z REAL,
  ixx REAL, ixy REAL, ixz REAL, iyy REAL, iyz REAL, izz REAL,
  visual_mesh_uri TEXT,
  collision_mesh_uri TEXT,
  source_uri TEXT NOT NULL,
  PRIMARY KEY (robot_id, name),
  FOREIGN KEY (robot_id) REFERENCES robots(robot_id)
);

CREATE TABLE packages(
  name TEXT PRIMARY KEY,
  version TEXT,
  description TEXT,
  maintainer TEXT,
  source_uri TEXT NOT NULL
);

CREATE TABLE pkg_deps(
  pkg TEXT NOT NULL,
  dep TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('build','exec','test','depend')),
  PRIMARY KEY (pkg, dep, kind)
);
CREATE INDEX pkg_deps_dep ON pkg_deps(dep);

CREATE TABLE launch_nodes(
  launch_uri TEXT NOT NULL,
  pkg TEXT NOT NULL,
  exec TEXT NOT NULL,
  name TEXT NOT NULL,
  namespace TEXT,
  PRIMARY KEY (launch_uri, name)
);
CREATE INDEX launch_nodes_pkg ON launch_nodes(pkg);

CREATE TABLE node_topics(
  launch_uri TEXT NOT NULL,
  node TEXT NOT NULL,
  kind TEXT NOT NULL,
  topic TEXT NOT NULL,
  remap_from TEXT
);
CREATE INDEX node_topics_topic ON node_topics(topic);
CREATE INDEX node_topics_node ON node_topics(node);

CREATE TABLE node_params(
  launch_uri TEXT NOT NULL,
  node TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT,
  value_kind TEXT
);
CREATE INDEX node_params_node ON node_params(node);

CREATE TABLE api_symbols(
  symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
  lib TEXT NOT NULL,
  source_uri TEXT NOT NULL,
  symbol TEXT NOT NULL,
  kind TEXT NOT NULL,
  signature TEXT,
  docstring TEXT
);
CREATE INDEX api_symbols_name ON api_symbols(symbol);
CREATE INDEX api_symbols_kind ON api_symbols(kind);

-- file_index: every indexed file, regardless of typed-table presence.
-- Powers raw_file(source_uri) by mapping URI → on-disk path.
CREATE TABLE file_index(
  source_uri TEXT PRIMARY KEY,
  abs_path TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  rel_section TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  mime_kind TEXT NOT NULL  -- 'text' or 'binary'
);
"""


def install(db: sqlite3.Connection) -> None:
    """Create the structured tables on a fresh build."""
    db.executescript(DDL)


def insert_structured_doc_metadata(db: sqlite3.Connection) -> None:
    """Ensure the three structured-source doc_ids appear in the docs table.

    Idempotent — uses INSERT OR REPLACE.
    """
    db.executemany(
        "INSERT OR REPLACE INTO docs VALUES (?, ?, ?, ?, ?)",
        STRUCTURED_DOCS,
    )
