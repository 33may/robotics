# Existing data-engineering / dataset-management skills — prior art survey (GitHub sweep)

Purpose: survey existing Claude Code / agent-skill prior art before designing a robotics
inspection-run "data-engineer" skill. Context for relevance judgments: the target skill governs
a **files-first, single-machine** data engine for multi-modal inspection runs (rgb/depth
pngs+npys, masks, camera poses, fused point clouds, JSON metadata), where a **run is an
immutable, read-only artifact**, derived data lives in a separate tree, and the main use case is
**retrieval for benchmarking and 3D work** — not enterprise big-data pipelines. Every entry below
was fetched and read from the actual `SKILL.md` (via `raw.githubusercontent.com` or GitHub API),
not inferred from a repo README, unless explicitly noted as "README-only" (fetch blocked).

---

## Tier 1 — closest real prior art (sensor-data-to-structured-store + catalog/retrieval)

### 1. NVIDIA/nurec-skills — `ncore` ★★★★★ single best analog found
- URL: https://github.com/NVIDIA/nurec-skills/blob/main/skills/ncore/SKILL.md
- Fetched: full SKILL.md content (not README).
- Covers: converting **any raw sensor recording** (cameras, depth, LiDAR, poses, masks, cuboids,
  point clouds) into a validated "NCore V4" store, for consumption by downstream
  reconstruction/rendering/sim skills. Two paths: bootstrap an existing converter (PAI, Waymo,
  COLMAP/ScanNet++, NuScenes, PandaSet) or author a new one from a 4-method template
  (`get_sequence_ids`, `from_config`, `convert_sequence`, write loop).
- SKILL.md structure (verbatim section order): frontmatter (name, version, license, tools,
  upstream repo, spec-docs link, release tag) → "When to use which path" decision tree →
  "Mental model — the V4 store" (component groups: Poses, Intrinsics, CameraSensor, LidarSensor,
  RadarSensor, Cuboids, Masks, PointClouds) → **non-negotiable frame-of-reference conventions**
  (rig +X forward/+Y left/+Z up; camera +X right/+Y down/+Z optical axis; LiDAR azimuth 0°=+X;
  all timestamps uint64 microseconds) → Path A / Path B instructions → V4 conventions (mandatory
  rules: re-reference all poses to first ego position before float32 cast; cuboid centroid is
  geometric center not bottom-center) → per-format recipes (AV formats, then non-AV: stereo,
  mono+depth RGB-D, mono+LiDAR, IMU+camera VIO, ROS2 bags, aerial drones) → robotics pipeline
  shard (`r2s`) naming conventions → validation commands
  (`ncore_vis`, `ncore_project_pc_to_img`) → **troubleshooting table of 15+ known failure modes**
  (Z-flips, rotated point clouds, timing errors) mapped to fixes → explicit scope-boundary
  "Do NOT use this skill for" list handing off to sibling skills.
- Worth stealing:
  1. **State spatial/temporal conventions as non-negotiable, up front, before any workflow** —
     axis directions, units, re-referencing rules. This is exactly the missing piece for camera
     pose / point-cloud consistency in an inspection-run skill, and no other surveyed skill states
     it this explicitly.
  2. **A troubleshooting table of known failure modes → fixes**, built from real conversion pain,
     as a required section, not an afterthought.

### 2. NVIDIA/nurec-skills — `nurec-index` (router) + `physical-ai-datasets` (catalog) ★★★★
- Router: https://github.com/NVIDIA/nurec-skills/blob/main/skills/nurec-index/SKILL.md
- Catalog: https://github.com/NVIDIA/nurec-skills/blob/main/skills/physical-ai-datasets/SKILL.md
- Fetched: full content of both.
- `nurec-index` is a pure **router**: a "pick a skill" decision table mapping user intent → one
  of 5 sibling skill names (`physical-ai-datasets`, `ncore`, `nre`, `asset-harvester`,
  `nurec-fixer`), explicit refusal to execute anything itself, explicit statement that it's
  hand-curated (won't auto-discover new siblings).
- `physical-ai-datasets` is a **catalog-only discovery skill** (no runtime execution) that routes
  users to ~30 NVIDIA Physical-AI datasets on HF Hub. Structure: prerequisites (HF auth,
  git-lfs) → download recipes (`hf download nvidia/<dataset> --repo-type dataset`, plus filtered
  pulls for huge datasets) → a **task-to-dataset lookup table** (20+ use-cases → recommended
  dataset, one row each) → per-family dataset cards (URL, size, format, sensors/modalities,
  license, gating status, downstream skill) grouped into 8 families (AV, manipulation, GR00T,
  spatial-memory benchmarks, NuRec/sim-ready scenes, healthcare, grasping, material props) → a
  **license decision tree** (CC-BY-4.0 commercial-OK vs CC-BY-NC-4.0 research-only vs gated
  NVIDIA AV license with 12-month expiry) → **cross-skill usage map** (dependency matrix: which
  sibling skill or external tool consumes each dataset) → verification/troubleshooting.
- Worth stealing:
  1. **Family-of-skills-with-a-router pattern**, each sibling stating what it explicitly does
     *not* do and which sibling owns that instead — directly applicable if the inspection
     data-engine skill grows conversion + catalog/retrieval + QA into separate concerns rather
     than one mega-skill.
  2. **Task-to-dataset lookup table** as the retrieval interface — "what's my goal" → "which
     runs/datasets satisfy it" → size/format/license metadata inline. This is the closest thing
     found anywhere to the "retrieval for benchmarking" requirement, even though it's static
     (hand-maintained cards) rather than a live query layer.

### 3. NVIDIA/nurec-skills — `asset-harvester` / `nurec-fixer` (siblings, brief) ★★★
- https://github.com/NVIDIA/nurec-skills/blob/main/skills/asset-harvester/SKILL.md
- https://github.com/NVIDIA/nurec-skills/blob/main/skills/nurec-fixer/SKILL.md
- Fetched: frontmatter + purpose/scope sections of both.
- Not directly about dataset management (they run GPU models: 3D Gaussian-splat asset extraction,
  novel-view diffusion harmonization) but confirm the family's house style: extremely detailed
  frontmatter (`compatibility`: OS/GPU/VRAM/CUDA version/disk space, `dependencies`,
  `time-estimate`, links to upstream repo/paper/HF model/HF dataset/benchmark) and a mandatory
  **"When to Use / When NOT to Use"** section with hard boundaries to sibling skills, repeated
  consistently across every skill in the family.
- Worth stealing: **reproducibility-grade frontmatter** (exact compatibility/dependency/time
  fields) — more rigorous than any other skill family surveyed, useful if the data-engine skill
  ever runs heavy conversion/reconstruction steps itself.

---

## Tier 2 — adjacent ML data-ops skills

### 4. pjt222/agent-almanac — `version-ml-data` ★★★★
- https://github.com/pjt222/agent-almanac/blob/main/skills/version-ml-data/SKILL.md
- Fetched: full SKILL.md (raw).
- Covers: DVC + Git for ML dataset versioning — large files/dirs versioned outside Git via
  content-addressed cache + remote storage (S3/GCS/Azure/SSH/local), lightweight `.dvc` metadata
  tracked in Git, `dvc.yaml` pipelines with declared deps/outs for reproducible
  re-derivation, `dvc push`/`dvc pull` for sharing.
- Structure (repo-wide convention across all `agent-almanac` skills, confirmed across 3 fetched
  files): frontmatter (name, description, license, `allowed-tools`, `metadata:` block with
  author/version/domain/complexity/tags) → "When to Use" bullet list → "Inputs"
  (Required/Optional) → "Procedure" as numbered steps, **each step ending in an explicit
  `**Expected:**` outcome and `**On failure:**` troubleshooting line** → pointer to a
  `references/EXAMPLES.md` for full code (SKILL.md itself keeps snippets truncated with
  `# ... (see EXAMPLES.md for complete implementation)`).
- Worth stealing:
  1. **Per-step `Expected:` / `On failure:`** — more granular than NVIDIA's single end-of-doc
     troubleshooting table; good for a skill with a multi-stage pipeline (e.g. ingest run →
     verify sibling files → build catalog entry → derive artifacts), where each stage can fail
     differently.
  2. **Raw-outside-Git + lightweight-metadata-inside-Git split** is conceptually the closest
     match to "runs are read-only artifacts, derived data + catalog metadata live separately" —
     though DVC's actual mechanism (content-addressed cache, remote push/pull) is overkill for a
     single local workstation with no remote/team-sharing need.

### 5. pjt222/agent-almanac — other ML/data skills (brief, mostly enterprise-shaped) ★★
- `label-training-data`, `orchestrate-ml-pipeline`, `monitor-data-integrity`, `catalog-collection`
  — all fetched (raw) at least partially.
- `label-training-data`: Label Studio setup, inter-annotator agreement (Cohen's kappa), active
  learning sampling — a data-*creation* concern, not retrieval/storage; not directly relevant.
- `orchestrate-ml-pipeline`: Prefect/Airflow DAG orchestration with Kubernetes/MLflow — same
  enterprise-orchestration shape as the generic ETL agents below; noise for single-machine scale.
- `monitor-data-integrity`: GxP/pharma compliance (ALCOA+ principles, audit-trail anomaly
  detection, regulatory escalation matrices) — compliance-domain noise, zero overlap.
- `catalog-collection`: **library science** cataloging (Dewey Decimal / Library of Congress
  classification, MARC records, controlled-vocabulary subject headings) — surprisingly *not* a
  data-catalog skill at all (name is misleading). Skimmed for the idea of controlled-vocabulary
  tagging (1-3 tags per item, "most specific term, not too many") which is a transferable idea
  for tagging inspection runs by defect-type/part/rig, but the skill itself is off-topic.

---

## Tier 3 — Hugging Face ecosystem (Hub-hosted, not local-files)

### 6. huggingface/skills — `huggingface-datasets` ★★★
- https://github.com/huggingface/skills/blob/main/skills/huggingface-datasets/SKILL.md
- Fetched: full SKILL.md.
- Covers: Hugging Face **Dataset Viewer API** workflows — validate a Hub dataset (`/is-valid`),
  resolve subset/split (`/splits`), preview (`/first-rows`), paginate (`/rows`, max 100/req),
  search/filter (`/search`, `/filter`), fetch parquet URLs + size/stats (`/parquet`, `/size`,
  `/statistics`). Also documents uploading agent session traces as HF datasets.
- Structure: frontmatter → numbered core workflow (validate → resolve split → preview →
  paginate → search/filter → parquet/stats) → endpoint reference table → upload workflow
  (browser UI or `hf-cli`).
- Relevance caveat: entirely **Hub-hosted, API-driven** — assumes the dataset already lives on
  huggingface.co in HF Dataset format. Says nothing about local, pre-Hub, heterogeneous
  per-record files, which is the inspection engine's actual starting point.
- Worth stealing: the **numbered "resolve → preview → paginate" discipline before doing anything
  expensive** — cheap ordered inspection steps before loading real data.

### 7. huggingface/lerobot — `AGENT_GUIDE.md` ★★★
- https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md
- Fetched: full content.
- Not a data-engineering skill — a user-facing guide for agents helping someone train a policy
  end-to-end. Frames `LeRobotDataset` as "episode-aware dataset (video/images + actions +
  state)," mutable (episodes can be deleted/merged/trimmed), Hub-or-disk loadable.
- Structure: mandatory pre-action question list (goal, hardware, GPU, dataset status, next step,
  "ask again rather than guess") → framework overview → 3 quickstart paths → data-collection
  principles (50–100 episodes, 20–45s, fixed rig, "start small then extend") → policy selection
  by VRAM → training duration → evaluation.
- No read-only/provenance rules for datasets at all — the opposite of the "run is immutable"
  requirement the new skill needs.
- Worth stealing: the **mandatory pre-action question list** pattern.

---

## Tier 4 — confirmed-noise enterprise data-engineer / ETL skills (read in full to confirm, not just skimmed)

### 8. alirezarezvani/claude-skills — `senior-data-engineer` ★ (confirms noise hypothesis)
- Correct current path (moved since first search hit a 404):
  https://github.com/alirezarezvani/claude-skills/blob/main/engineering-team/skills/senior-data-engineer/SKILL.md
- Fetched: full SKILL.md content.
- Covers: "Trigger Phrases" list → Quick Start (CLI scripts:
  `pipeline_orchestrator.py generate --type airflow`, `data_quality_validator.py validate`,
  `etl_performance_optimizer.py analyze`) → Architecture Decision Framework (batch-vs-streaming
  decision tree, Lambda-vs-Kappa, warehouse-vs-lakehouse comparison tables) → Tech Stack table
  (Spark/Airflow/dbt/Kafka/Snowflake/BigQuery/Great Expectations/Datadog) → pointers to
  `references/*.md` for data modeling (star schema, SCD types 1-6, data vault), DataOps, and
  troubleshooting.
- Confirms the user's framing directly: 100% cloud-warehouse/streaming vocabulary, zero overlap
  with files-on-disk, heterogeneous-per-record, single-workstation reality.
- Worth stealing (structural only, not content): **decision-tree tables for architecture
  choices** (X vs Y, with a "when to choose" bullet list under each) is a clean format for any
  place the new skill needs to justify a design choice to the user.

### 9. rohitg00/awesome-claude-code-toolkit — `data-engineer.md` / `etl-specialist.md` ★★
- https://github.com/rohitg00/awesome-claude-code-toolkit/blob/main/agents/data-ai/data-engineer.md
- https://github.com/rohitg00/awesome-claude-code-toolkit/blob/main/agents/data-ai/etl-specialist.md
- Fetched: full content of `data-engineer.md` (agent definition, not SKILL.md format — frontmatter
  has `tools`, `model: opus` instead of skill-style frontmatter).
- Core principles stated: **idempotency** ("running the same pipeline twice on the same input
  produces the same output without side effects"), validate at ingestion + after transform +
  before delivery, design for schema evolution, prefer ELT over ETL. Sections: Pipeline
  Architecture, Apache Spark, Data Warehousing (medallion Bronze/Silver/Gold), Pipeline
  Orchestration (Airflow/Dagster), Data Quality (Great Expectations), Streaming (Kafka/Flink).
  Ends with a pre-completion checklist (run quality tests, verify idempotency via duplicate runs,
  check partitioning/file sizes, validate DAG structure).
- Worth stealing: **idempotency and "validate at 3 checkpoints" as generalizable principles** even
  though the tooling (Spark/Airflow/Great Expectations) is irrelevant — e.g. "re-ingesting the
  same run directory twice must not corrupt the catalog" is a direct, useful translation.

### 10. gordonmurray/data-engineering-skills — repo-wide structural skeleton ★★
- https://github.com/gordonmurray/data-engineering-skills (repo README + `lance/SKILL.md`
  fetched)
- Covers per-skill: Iceberg, Paimon, Fluss, Flink, Iggy, Lance, Firn, Docker Compose — modern
  streaming/lakehouse data stack, each as its own skill folder.
- `lance/SKILL.md` (fetched in full): covers **Lance format + LanceDB** — an ML-native columnar
  format supporting embeddings/vectors/multimodal blobs with ANN indexing (IVF_PQ, IVF_HNSW_FLAT)
  — the one skill in this repo that touches multimodal storage. Sections: Scope → **"Current
  Facts"** (pinned exact version numbers + release dates, explicitly dated so the skill doesn't
  go stale silently) → **"Inspect First"** checklist (format-or-DB question? check installed
  versions? row count/vector dim/fragment count? local vs object storage? current index type +
  measured recall?) → Decision Rules (index selection) → **Safety** (warns permanent data loss
  from compaction, credential handling). No concrete guidance on storing point clouds/images on a
  local filesystem beyond "Lance supports it" — shallow on the exact multimodal angle needed.
- Repo-wide required-sections skeleton (per gordonmurray's own stated convention): **Scope,
  Inspect First, Safety, Verify, Update Checklist** — every skill in the repo follows this
  regardless of topic.
- Worth stealing:
  1. **"Current Facts" section with pinned dated version numbers** — a good defense against
     skill staleness, worth adopting verbatim.
  2. **"Inspect First" as a named, required section** (not just an implicit habit) — forces
     cheap-check-before-expensive-action discipline explicitly into the skill structure.

---

## Landscape / absence findings

- **anthropics/skills** (official reference repo, https://github.com/anthropics/skills) has **no
  data-engineering, dataset, or ETL skill at all** — current catalog is academy-guide,
  algorithmic-art, brand-guidelines, canvas-design, claude-api, discernment-nudge,
  doc-coauthoring, docx, frontend-design, internal-comms, mcp-builder, pdf, pptx, skill-creator,
  slack-gif-creator, theme-factory, web-artifacts-builder. Data engineering is entirely
  community-contributed; there is no canonical reference to converge toward.
- Large marketplaces/directories scanned for coverage breadth (agentskill.sh — 275k+ skills,
  "Data & Analytics" category ~69k; Agent Almanac / pjt222 repo — 370 skills; karanb192,
  ComposioHQ, BehiSecc, travisvn `awesome-claude-skills` lists; mcpservers.org/agent-skills) —
  all confirm the same pattern: abundant generic CSV/SQL/tabular data-quality skills, abundant
  enterprise ETL/warehouse skills, **zero** hits for point-cloud processing (Open3D/COLMAP),
  zero hits for manufacturing/visual-inspection defect-dataset skills, zero hits for a
  run/episode-as-immutable-folder retrieval catalog. Searched explicitly and came up empty on
  all three axes — treated as a real negative-space finding, not a search-quality failure (each
  search returned plenty of *adjacent* results, just never on-target).
- The one near-miss on "visual inspection": `daymade/claude-code-skills` has a skill diagnosing
  **UI/software rendering defects** (typography, overflow, clipping) — same words, completely
  different domain (frontend QA, not manufacturing/physical inspection).

---

## Gaps — what no existing skill covers that the new one must

1. **"Run" as the atomic unit = a folder of heterogeneous sibling files, inspected/cataloged in
   place — not converted into a uniform store, not a tabular row.** LeRobot/HF Datasets model a
   dataset as uniform parquet+mp4 rows; NCore's answer to sensor heterogeneity is to **convert**
   everything into one archive format (`zarr.itar`) as an active pipeline step. Nothing surveyed
   treats "leave the run's rgb pngs + depth npys + masks + pose JSON + fused point cloud exactly
   as captured, and build tooling that understands that layout without transforming it" as the
   first-class model.
2. **Read-only/immutability as an enforced contract, not a convention.** LeRobot episodes can be
   deleted/merged/trimmed in place; NCore's conversion step mutates raw into V4; DVC's model
   (content-addressed cache) is closest in spirit but still designed around *updating* tracked
   data over time, not "this directory is permanently frozen the moment it's captured." Nothing
   surveyed actively refuses/flags destructive operations on a raw data directory.
3. **Raw vs. derived separation as a first-class, structurally-required rule.** NCore converts in
   place (mixing raw+derived by directory identity); LeRobot/HF have no derived tier at all; DVC
   pipelines (`dvc.yaml` deps/outs) come closest conceptually but assume a shared team/remote
   workflow, not "derived artifacts live in a separate local tree keyed by run id + processing
   version, and old derivations may need to coexist with new ones."
4. **Cross-modal correspondence checks for *spatial* data (not action/state or tabular data).**
   NCore is the only surveyed skill that even attempts this (frame-of-reference conventions +
   troubleshooting table for pose/point-cloud bugs), but it's built for feeding a neural
   renderer, not for verifying an inspection run is internally self-consistent (depth reprojects
   correctly onto the point cloud; mask aligns to rgb; per-camera pose chain agrees). The
   *structure* (state conventions up front, then a troubleshooting table) is reusable; the actual
   checks have to be authored fresh for this domain.
5. **A local retrieval/query layer for benchmarking and 3D-work selection.** HF's dataset skill
   assumes a Hub-hosted API with viewer endpoints; `physical-ai-datasets`'s task-to-dataset
   lookup table is the closest analog but is a **hand-maintained static table over ~30 named
   datasets**, not a queryable index over a continuously-growing local archive of runs (by defect
   type, part, date, camera rig, environmental conditions). This has to be designed from scratch
   — likely a lightweight local index/catalog file, not a database service, matching the
   files-first constraint.
6. **Single-machine storage lifecycle for lab-scale recordings.** DVC's remote push/pull and the
   enterprise ETL agents' cloud lifecycle policies both assume a remote store; nothing surveyed
   addresses "when do we prune redundant pngs once npys exist, when do old runs get
   archived/compressed, how much disk is this archive eating on one workstation" — this
   connective tissue doesn't exist in any surveyed skill.
7. **Multi-file-format-per-record health checks as a routine, repeatable operation over a
   growing archive.** Every surveyed dataset-quality tool (Great Expectations, the generic
   `data-quality-audit`/`monitor-data-integrity` skills) is built for tabular/CSV-style row
   consistency. None check "does every run directory have the expected sibling files present,
   correctly shaped, and mutually consistent" as a repeatable per-run health check — this is
   closest in spirit to NCore's validation commands (`ncore_vis`, `ncore_project_pc_to_img`) but
   those validate a *converted* store, not an untouched raw run directory.

---

## Sources consulted (fetched, not just search-snippet)
- https://github.com/NVIDIA/nurec-skills — `skills/ncore/SKILL.md`, `skills/nurec-index/SKILL.md`,
  `skills/physical-ai-datasets/SKILL.md`, `skills/asset-harvester/SKILL.md`,
  `skills/nurec-fixer/SKILL.md`
- https://github.com/NVIDIA/skills (catalog overview, README-level)
- https://github.com/pjt222/agent-almanac — `skills/version-ml-data/SKILL.md`,
  `skills/label-training-data/SKILL.md`, `skills/orchestrate-ml-pipeline/SKILL.md`,
  `skills/monitor-data-integrity/SKILL.md`, `skills/catalog-collection/SKILL.md`, plus repo
  `skills/` directory listing via GitHub API
- https://github.com/huggingface/skills — `skills/huggingface-datasets/SKILL.md`
- https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md
- https://github.com/alirezarezvani/claude-skills —
  `engineering-team/skills/senior-data-engineer/SKILL.md` (path corrected via GitHub API after
  initial dead link)
- https://github.com/rohitg00/awesome-claude-code-toolkit —
  `agents/data-ai/data-engineer.md`, `agents/data-ai/etl-specialist.md`
- https://github.com/gordonmurray/data-engineering-skills — README + `lance/SKILL.md`
- https://github.com/anthropics/skills (official catalog, checked for absence)
- Marketplace/landscape breadth checks (search only, no single skill worth a dedicated fetch):
  agentskill.sh, https://github.com/pjt222/agent-almanac (Agent Almanac), karanb192/, ComposioHQ/,
  BehiSecc/, travisvn/ `awesome-claude-skills`, mcpservers.org/agent-skills,
  claudeskills.info, lobehub.com/skills, daymade/claude-code-skills
