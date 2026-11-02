# Oli Main Software v2.2.12 (EDU Ed.)

Vendored LimX-authored on-robot software stack. This directory is one of three source roots ingested by `oli-corpus-mcp` (the others are `humanoid-mujoco-sim/limxsdk-lowlevel/` and `humanoid-rl-deploy-python/`).

## What lives here

- `MANIFEST.tsv` — `<relpath>\t<sha256>` for the 787 indexable files (urdf, srdf, xml, h, hpp, yaml, json, py, package.xml, launch). Committed to the repo, used by the corpus indexer to detect drift.
- `README.md` — this file.
- `install/` — the unpacked colcon ROS2 install workspace. **Gitignored** (2.5 GB on disk). Each developer must download and unpack the tarball themselves; see "How to populate" below.

## What this is

LimX ships the on-robot stack as a wrapper tarball `robot-hu-r-2.2.12.20260508181921.tar.gz` (1.5 GB compressed). The outer tarball contains:
- an inner `robot-hu-r-2.2.12.20260508181921.tar.gz` (1.5 GB) — the actual payload
- a `.md5sum` file with checksum `42227963df309e4f8ae0e43669a2a7da` for the inner tarball

The inner payload is a colcon ROS2 install workspace with 148 packages, organized under `install/`:

| Subdir | Contents |
|---|---|
| `install/share/` | 148 ROS package install dirs (`package.xml`, `launch/`, `config/`, `urdf/`) |
| `install/mbl/include/` | C++ headers for control + teleop libs |
| `install/mbl/bin/`, `install/bin/` | ~700 ELF binaries (controllers, MROS nodes, teleop drivers) |
| `install/lib/`, `install/mbl/lib/` | 120 `.so` shared libs |
| `install/oli/` | `local_cosa-arm` binary + startup wav |
| `install/python/` | `auto_run.py`, `post_analyze/`, `result_processors/` |
| `install/docker/`, `install/etc/` | Runtime configs |
| `install/run_on_local.sh`, `install/run_on_robot.sh`, `install/setup.bash` | Entry points |

**Critical**: zero `.msg` and zero `.srv` files. ROS message IDL is stripped — compiled message types only ship in `.so`. Topic shapes are not recoverable from this tree.

## How to populate

```sh
# 1. Download from LimX (browser required — JS-gated)
#    https://limx.cn/en/products/oli/download
#    → "Oli Main Software (EDU Ed.)" button

# 2. Verify wrapper integrity (sha256 of the inner tarball)
INNER_MD5=42227963df309e4f8ae0e43669a2a7da

# 3. Extract the outer wrapper, then the inner payload, into THIS directory:
cd humanoid/vendor/oli-main-software-2.2.12
tar -xzf ~/Downloads/robot-hu-r-2.2.12.20260508181921.tar.gz -C /tmp
tar -xzf /tmp/robot-hu-r-2.2.12.20260508181921/robot-hu-r-2.2.12.20260508181921.tar.gz

# 4. Verify the MANIFEST matches what you extracted
cut -f1 MANIFEST.tsv | xargs sha256sum > /tmp/local.txt
diff <(awk '{print $2,$1}' MANIFEST.tsv | sort) <(awk '{print $2,$1}' /tmp/local.txt | sort)
# (no output = match)
```

## Why versioned folder name

The folder name encodes the LimX version (`-2.2.12`). When LimX ships a new tarball, vendor it at `oli-main-software-2.2.13/` so the two coexist for diffing. The corpus indexer can then index either version (or both) under its own `oli-corpus://oli-main/2.2.12/...` URI namespace.

## Why gitignored

The payload is 2.5 GB on disk. Committing it would balloon repo size for negligible gain — the tarball is hosted by LimX and the MANIFEST.tsv lets us detect drift. The indexer reads `MANIFEST.tsv` at build time and refuses to proceed if file hashes don't match.

## What gets indexed

The `oli-corpus-mcp` builder walks this directory and extracts:

- URDF/SRDF → `robots`, `joints`, `links` tables (joint axes, limits, mesh URIs)
- `package.xml` → `packages`, `pkg_deps` tables (dependency graph)
- Launch XML → `launch_nodes`, `node_topics`, `node_params` tables
- Headers in `install/mbl/include/` → `api_symbols` table (the wire-level struct definitions LimX strips elsewhere)
- YAML/JSON configs → FTS chunks (controller params, EKF tuning, planner params)

Binaries (`.so`, ELF, `.onnx`, `.rknn`, `.STL`, `.wav`) and colcon environment hooks (`.dsv`, `.sh`, `.bash`, `.zsh`, `.ps1`, `.cmake`) are skipped.

## Related

- `humanoid/docs/oli-corpus/` — the MCP server + indexer
- `humanoid/docs/superpowers/specs/2026-06-22-oli-corpus-structured-ingest-design.md` — the design driving this ingest
