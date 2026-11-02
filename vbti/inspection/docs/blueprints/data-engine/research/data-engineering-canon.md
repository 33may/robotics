# Data-engineering canon for a robotics dataset-engine skill — research draft

Status: RESEARCH DRAFT — thorough gathering + first distillation, not curated. A senior curation
pass follows. Cross-reference `existing-data-skills.md` (prior-art survey of Claude Code skills)
for the gap analysis this canon fills.

## System this canon governs

An inspection-robot data engine on **one lab machine, GBs not TBs, files-first, no DB decided
yet**. Observed concrete shape (`inspection/data/runs/<run-id>/`): per-view folders (`000/`,
`001/`, …) each with `rgb.png`, `depth_raw.npy`, `depth_aligned.npy`, `mask.png`, `meta.json`,
plus run-level `run.json`, `session.json`, `fused_cloud.npy`, and a VLM-trace `eyes/` subtree.
Derived data already lives in a sibling `data/derived/<run-id>/{fits,steps,rrd}/` tree. Settled
invariants going into this research: recorded runs are **read-only**; derived data lives
separately, keyed by **(run, method, step)**; replay is **bit-exact**; schema docs must be
**generated from code**, never hand-drawn.

## Rule format

**Statement → why (source, cited) → how it applies at our scale.** Rules marked
**[ANTI-PATTERN AT LAB SCALE]** are real, citable industry practice that would be *wrong* to
import wholesale into a one-machine, GB-scale, files-first archive — kept explicit so a later
curation pass rejects them on purpose rather than by omission.

---

## Part 1 — Robotics dataset conventions

### 1.1 Separate calibration-level data from per-frame data
**Statement.** A sensor's mounting/intrinsics (rarely changes) and a rig's live pose (changes
every capture) are different kinds of data and belong in different records, joined by a foreign
key — not flattened together into one per-frame blob.
**Why.** nuScenes' schema puts sensor-to-vehicle extrinsics + camera intrinsics in
`calibrated_sensor`, and vehicle-to-map pose in a separate `ego_pose` table, joined via
`sample_data.calibrated_sensor_token` / `ego_pose_token`
([nuscenes-devkit schema docs](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)).
KITTI does the analogous split with a single `calib.txt` per sequence (`P0..P3`, `R0_rect`,
`Tr_velo_to_cam`) separate from per-frame image/point-cloud files
([KITTI format overview](https://deepdatamininglearning.readthedocs.io/en/latest/KITTI_Tutorial.html)).
**How it applies.** Per-view `meta.json` currently likely bundles intrinsics + pose together.
Worth asking: does intrinsics ever change *within* a run (same physical camera, no re-calibration
mid-run)? If not, intrinsics is calibration-level and should be traceable to a calibration
identity/version recorded once (in `run.json`/`session.json`), even if it's still convenient to
duplicate the numbers into each view's `meta.json` for self-containment. The goal isn't
normalization for its own sake — it's making a mid-session recalibration detectable instead of
silently inconsistent.

### 1.2 Sibling-file completeness is a first-class, routine check
**Statement.** Every step/frame/view in a dataset must carry the *same* set of fields/files — no
per-record schema drift where some views have a mask and others don't.
**Why.** RLDS requires that "all the steps in the same dataset are required to have the same
fields" — optional fields are allowed, but presence must be uniform across the dataset, not
ad hoc per step ([RLDS spec, google-research/rlds](https://github.com/google-research/rlds)).
**How it applies.** This is the exact mechanism to formalize the gap identified in
`existing-data-skills.md` (#7): "does every run directory have the expected sibling files
present, correctly shaped, mutually consistent" as a repeatable per-run health check — modeled
directly on RLDS's uniformity constraint, but checked against the filesystem instead of a
tf.data schema.

### 1.3 State spatial/temporal conventions as non-negotiable, up front, self-describing in the recording itself
**Statement.** Axis directions, units, coordinate-frame conventions, and clock/timestamp meaning
must be recorded *inside* the data (schema version, units, frame convention fields in
`run.json`), not left as tribal knowledge encoded only in the code that produced it.
**Why.** MCAP's foundational design requirement, stated directly in Foxglove's own evaluation:
"the design of robotic systems is often evolving... For a recording file to be useful as
long-term storage, it must be self-describing"
([MCAP evaluation paper](https://mcap.dev/files/evaluation.pdf),
[Introducing the MCAP File Format](https://foxglove.dev/blog/introducing-the-mcap-file-format)).
rosbag/rosbag2's weakness by comparison: messages aren't self-contained, so third-party tools
(and future-you) can't interpret them without the original ROS message definitions
([MCAP vs ROS bag](https://segments.ai/blog/mcap-vs-ros-bag-simplifying-multi-modal-sensor-data-in-robotics/)).
**How it applies.** This is precisely the missing piece flagged as the single best insight from
the prior-art survey (NCore's "non-negotiable frame-of-reference conventions" section). Concrete
translation: `run.json`/`session.json` should carry an explicit schema-version field, units
(meters vs mm), depth convention (raw sensor units vs metric-aligned — note we already have both
`depth_raw.npy` and `depth_aligned.npy`, which is good practice but the *distinction* itself
needs to be documented in-band, not just inferable from filenames).

### 1.4 Timestamp alignment method must be explicit and recorded, not assumed
**Statement.** When modalities are captured at different rates or without hardware sync, the
dataset must record *which* signal is the timing reference and *how* others were aligned to it.
**Why.** Recent multi-sensor robotics datasets converge on this as a hard requirement precisely
because it's easy to get silently wrong: FastUMI-100K uses "a unified ROS clock to assign
consistent timestamps across all data streams" across sensors with different sampling rates
([FastUMI-100K](https://arxiv.org/pdf/2510.08022)); the Rosario Dataset v2 had to build a
post-hoc alignment procedure (detect stationary→moving transitions per sensor, then offset)
precisely because sensors were *not* hardware-synchronized and that had to be corrected for
explicitly, not assumed away
([Rosario Dataset v2](https://arxiv.org/pdf/2508.21635)).
**How it applies.** For fused point clouds and multi-camera views within a run, record (even
briefly) whether captures were simultaneous or sequential-with-known-latency, and which
timestamp field is authoritative. This is cheap now and expensive to reconstruct later.

### 1.5 [ANTI-PATTERN AT LAB SCALE] Sharding many records into few large files
**Statement (as practiced).** LeRobotDataset v3 concatenates many episodes' tabular rows and
video frames into large shared Parquet/MP4 files, resolving individual episode boundaries via
metadata offsets instead of file boundaries — replacing v2's one-file-per-episode layout.
**Why (in its native context).** Explicitly a scale response: "to scale to millions of episodes...
episode-specific views reconstructed via metadata, not file boundaries," motivated by filesystem
pressure and Hub-native streaming for huge datasets
([LeRobotDataset v3.0 docs](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3),
[LeRobot v3 blog](https://huggingface.co/blog/lerobot-datasets-v3)).
**Why this is an anti-pattern at our scale.** We have tens to low-hundreds of runs, not millions
of episodes. One-folder-per-run with plain sibling files is *more* debuggable (every artifact is
independently `ls`-able, diffable, openable) and the filesystem-pressure problem this format
solves does not exist yet at GB scale. Adopting shard consolidation now would trade inspectability
for a scaling headroom we don't need — worth revisiting only if run count crosses into the
thousands.

### 1.6 [PARTIAL ANTI-PATTERN] Standardize by writing an export layer, not by reformatting raw capture
**Statement.** Open X-Embodiment did not ask 21 institutions to recapture their data in a shared
format — it defined RLDS as a target schema and converted each source dataset *into* it as a
separate step, while raw per-institution formats presumably still exist upstream.
**Why.** "All data is standardized using the Reinforcement Learning Datasets (RLDS) framework...
a standardized format that efficiently accommodates the heterogeneous nature of robotics data"
([Open X-Embodiment paper](https://arxiv.org/html/2310.08864v9),
[OXE overview](https://www.emergentmind.com/topics/open-x-embodiment-dataset)). The
standardization is a conversion target, not a capture-time constraint on the source labs.
**How it applies.** For the stated future use "VLA-adjacent dataset export": keep raw runs in
their native inspection-engine layout (rgb/depth/mask/pose as captured) and build an *export*
step that projects into RLDS/LeRobot schema only when actually feeding a training pipeline. Do
not pre-warp the capture format to already look like a training format — that couples capture to
a downstream consumer that may change (RLDS today, something else in 2 years).

---

## Part 2 — ML dataset management practice

### 2.1 Raw is immutable; every downstream artifact is a fresh derivation, never an in-place edit
**Statement.** Treat captured raw data the way DDIA treats batch-processing inputs: "inputs are
immutable... outputs intended to be inputs to another (as yet unknown) program."
**Why.** Kleppmann, *Designing Data-Intensive Applications* — batch processing design principles;
also: "one of the most effective approaches [to safety] is by making all operations idempotent...
preventing faulty code from destroying good (immutable) data"
([DDIA notes/summaries](https://github.com/ahmedhammad97/Designing-Data-Intensive-Applications-Notes),
[DDIA summary](https://danlebrero.com/2021/09/01/designing-data-intensive-applications-summary/)).
DVC's own practitioner guidance converges on the same rule from the tooling side: "raw data
should remain immutable, processed outputs should be written only by scripts... extracts should
be copied into a raw data folder and never edited in place"
([DVC reproducibility summary](https://medium.com/mantisnlp/data-version-control-for-reproducible-analytical-pipelines-5255782d355d)).
**How it applies.** This is already a settled invariant of the system — cited here as
reinforcement with sources, and as the justification for treating *any* code path that would
write into `data/runs/<id>/` after capture as a bug, not a feature, even for "just fixing a typo
in meta.json."

### 2.2 Content hash = identity = integrity check (lightweight, no DVC required)
**Statement.** A cryptographic hash of a file's bytes both identifies it and detects corruption —
compute it once at write time, record it, and it becomes a free integrity check forever after.
**Why.** "The hash serves as a checksum, allowing verification of data integrity — if the data
retrieved doesn't match its hash, corruption is detected"
([content-addressable storage pattern summary](https://nesbitt.io/2026/07/07/content-addressing-in-package-managers.html)).
DVC operationalizes exactly this at the tool level — content hash per tracked file/dir, stored as
a small pointer, with the actual bytes in a cache
([DVC start guide](https://doc.dvc.org/start),
[DVC versioning example](https://doc.dvc.org/example-scenarios/versioning-data-and-models)).
**How it applies (keep the principle, skip the tool — see 2.3).** Record a SHA-256 per raw file
(or per run, as a manifest) at capture time, in `run.json`. This gives cheap corruption detection
on a single spinning/SSD drive with no backup redundancy, and gives derived artifacts something
concrete to reference: "this fit was computed from run X whose fused_cloud.npy hash was Y" — the
verifiable backing for bit-exact replay (see 3.5).

### 2.3 [ANTI-PATTERN AT LAB SCALE] Full DVC-style remote cache + push/pull workflow
**Statement (as practiced).** DVC's actual mechanism is a content-addressed local cache plus a
remote store (S3/GCS/Azure/SSH) with `dvc push`/`dvc pull` for team sharing, plus `dvc.yaml`
DAG-based re-derivation pipelines.
**Why (in its native context).** Solves reproducible, shareable ML pipelines across a *team* with
a *remote* — "auditing the project's immutable history to learn when datasets or models were
approved" implies multi-party review workflows
([DVC pipelines](https://doc.dvc.org/user-guide/pipelines)).
**Why this is an anti-pattern at our scale.** One machine, one user, no remote to push to, no
team review gate. The `existing-data-skills.md` survey already flagged this precisely: DVC's
mechanism is "conceptually the closest match" to raw/derived separation "though DVC's actual
mechanism... is overkill for a single local workstation with no remote/team-sharing need."
Standing up DVC here would add a cache-management layer and a `.dvc`-file indirection with no
corresponding benefit — a plain SHA-256 manifest (2.2) captures the load-bearing idea for free.

### 2.4 Datasheets: record motivation/composition, not just structure
**Statement.** A dataset (or here, a run) should carry human-readable documentation of *why* it
exists and what it contains semantically — not just its file schema.
**Why.** "Datasheets for Datasets" — the seminal metadata-documentation paper — argues the ML
community "currently has no standardized process for documenting datasets," proposing datasheets
covering motivation, composition, collection process, and recommended uses, by analogy to
hardware datasheets ([Gebru et al., arXiv:1803.09010](https://arxiv.org/abs/1803.09010)).
**How it applies.** `run.json`/`session.json` already captures structural/collection metadata.
The gap: a motivation/context field (what defect/part/condition was this run *for*) is what
enables the stated future use case "3D deconstruction benchmarking... retrieval for
benchmarking" — this was flagged as an unfilled gap in the prior-art survey (no surveyed skill
provides a queryable "what's my goal → which runs satisfy it" layer beyond a hand-maintained
table). A datasheet-style motivation field per run is the cheapest first step toward that.

### 2.5 [LOWER PRIORITY, note for later] Croissant / schema.org descriptive layer
**Statement.** Croissant separates dataset metadata into four layers (Dataset Metadata, Resource,
Structure, Semantic), built on schema.org, specifically to make datasets loadable directly by
ML frameworks (TF/PyTorch/JAX) and discoverable via Dataset Search.
**Why.** "The Croissant format doesn't change how the actual data is represented... it provides a
standard way to describe and organize it," now supported by HF/Kaggle/OpenML
([Croissant paper, NeurIPS 2024](https://arxiv.org/pdf/2403.19546),
[Google Research blog](https://research.google/blog/croissant-a-metadata-format-for-ml-ready-datasets/)).
**How it applies.** Not worth adopting now — there's no Hub-publishing or cross-framework-loading
need yet at lab scale. Worth a second look only when/if the VLA-adjacent export target needs to
be consumed by tools that specifically expect Croissant. Recorded here so a future skill revision
doesn't have to re-research it from zero.

### 2.6 [ANTI-PATTERN AT LAB SCALE] WebDataset/tar-shard sequential-I/O optimization
**Statement (as practiced).** WebDataset packs samples into POSIX tar shards specifically so
training loops can read sequentially instead of doing random access across millions of small
files, since "random access to millions of small files... is extremely slow on both local storage
and networked/cloud file systems"
([WebDataset design summary](https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/),
[webdataset/webdataset](https://github.com/webdataset/webdataset)).
**Why this is an anti-pattern at our scale.** This solves an I/O-throughput problem that appears
at "millions of files" / cloud-storage / multi-GPU-cluster scale. On one local machine with
GB-scale data on a fast local disk, plain per-view files are not a bottleneck, and tar-shard
packing would only add an unpacking indirection for zero throughput benefit. If a training export
is ever built (2.5/1.6), *that* export step could emit WebDataset shards as an output format —
but the raw run storage itself should stay as loose files.

### 2.7 [THRESHOLD-GATED, not yet] Zarr-style chunked arrays for large N-D data
**Statement.** Zarr chunks N-dimensional arrays into independently-addressable, compressed blocks
with JSON metadata, enabling parallel/partial reads without loading a whole array.
**Why.** Built specifically for cases where "reading entire files to extract specific data" is
wasteful at scale — a genomics/climate-science-scale need
([Zarr intro, Earthmover](https://www.earthmover.io/blog/what-is-zarr/),
[Zarr storage spec, OGC](https://www.ogc.org/publications/standard/zarr-storage-specification/)).
**How it applies.** Our per-view depth arrays and even `fused_cloud.npy` are single files in the
MB range — plain `.npy` is fine; Zarr's chunking machinery buys nothing until an array is large
enough that partial/parallel reads matter (e.g., a much denser fused reconstruction, or
volumetric occupancy grids). Flag as a size-triggered future consideration, not a rule to adopt
now.

---

## Part 3 — General data-engineering canon (kept narrow — only what earns its place)

### 3.1 Schema-on-write for raw capture; schema-on-read is fine for exploratory derived queries
**Statement.** Because a run is captured once and never rewritten, its schema must be validated
and correct *at write time* — there is no "fix it on next read" option. Exploratory querying over
the growing run archive (for benchmarking/retrieval) can stay schema-on-read, since the questions
being asked will change before the storage does.
**Why.** "Schema-on-write... increased data integrity and minimized inconsistency... ideal for
[cases requiring] strong quality gates," vs. schema-on-read's flexibility for evolving,
not-yet-fully-known query needs
([Dremio: Schema-on-Read vs Schema-on-Write](https://www.dremio.com/wiki/schema-on-read-vs-schema-on-write/)).
**How it applies.** Validate `meta.json`/`run.json` against a schema (pydantic/dataclass, see 3.6)
*during* capture, before the run is considered closed/read-only — catching a malformed record
before it becomes a permanent, immutable artifact is the only chance to catch it at all. The
future retrieval/catalog layer, by contrast, should not be over-designed around today's guessed
query patterns.

### 3.2 Raw/derived is a two-tier split, not medallion's multi-hop refinement
**Statement.** Take medallion architecture's Bronze (immutable raw, single source of truth) /
Gold (curated, purpose-built) split; explicitly skip the Silver "clean up in place" middle tier.
**Why.** Medallion's own definition: "Bronze... serves as the single source of truth... Silver...
data gets fixed of errors, standardized in format, deduplicated... Gold... aggregates results
into tables built for specific consumers"
([Databricks medallion docs](https://docs.databricks.com/aws/en/lakehouse/medallion)). Silver's
entire purpose is progressively *rewriting* copies of the data toward cleanliness.
**Why partially an anti-pattern here.** Runs don't get "cleaned" — they're either valid at capture
time (3.1) or excluded; there's no in-place-deduplicate-and-standardize middle step. Importing a
Silver tier would either (a) violate read-only-ness by rewriting run data, or (b) become a
disguised extra `derived/` output, which is already covered by the existing raw/derived split.
Keep the two-tier model; don't add a third.

### 3.3 Idempotent pipelines: re-running a derivation on unchanged inputs must be a no-op or byte-identical
**Statement.** A derive step (fit/render/replay) run twice against the same (run, method, step)
input must not silently duplicate outputs, and — given the "bit-exact replay" invariant — must
produce identical bytes.
**Why.** Idempotency is Kleppmann's stated defense against faulty-code data corruption (2.1,
reprised here as a pipeline-level rule, not just a raw-data rule); also converges with the
industry data-engineer distillation already surfaced in the prior-art survey: "running the same
pipeline twice on the same input produces the same output without side effects," validated at
multiple checkpoints
([data-engineer.md, rohitg00/awesome-claude-code-toolkit](https://github.com/rohitg00/awesome-claude-code-toolkit/blob/main/agents/data-ai/data-engineer.md)).
**How it applies.** The `(run, method, step)` key is already the right idempotency key — a derive
call with the same key and same code version should either detect "already computed, skip" or
overwrite deterministically, never append a `_2` variant beside a stale one.

### 3.4 Single-writer principle: capture owns the run exclusively; everything else is a reader
**Statement.** Exactly one process is ever the writer of a given run during capture. The moment
capture ends, the run has zero writers, forever — every later consumer (viewers, fit pipelines,
the eyes/VLM trace reader) is read-only by construction, not by convention alone.
**Why.** "One application is typically not permitted to use more than one instance of a
reader/writer... to ensure the reading order of events is not corrupted" — the single-writer
principle from event-driven systems design
([Designing Event-Driven Systems, O'Reilly, ch. 11](https://www.oreilly.com/library/view/designing-event-driven-systems/9781492038252/ch11.html)).
**How it applies.** This is the formal backing for "recorded runs are read-only" — not just a
policy statement but a concurrency-safety argument: read-only-after-capture is what a
single-writer system looks like *after* its one writer has finished. Any code path with two
things claiming write access to the same run (e.g., a live capture process and a "let me patch
this field" script) is the exact failure mode this principle rules out.

### 3.5 Atomic writes: a run must never be observable half-written
**Statement.** Every file that constitutes part of a run's contract (`run.json`, `session.json`,
any manifest) must be written via write-to-temp-in-same-directory → fsync → atomic rename, never
opened for write and updated in place.
**Why.** "The most reliable way to achieve atomic file writes on Linux is the
'write-then-rename' pattern... rename() is atomic with respect to other filesystem operations"
per POSIX ([LWN: A way to do atomic writes](https://lwn.net/Articles/789600/); practical Python
pattern summary at
[bswen.com](https://docs.bswen.com/blog/2026-04-04-atomic-file-writing-python/)). The critical,
often-missed detail: the temp file must be created in the *same* directory as the target so the
rename stays on one filesystem and is genuinely atomic.
**How it applies.** This is the concrete mechanism, not just the policy, behind "runs are
read-only artifacts": if a crash happens mid-write to `run.json`, atomic rename guarantees the
reader either sees the old complete version or the new complete version — never a truncated one.
Any code that currently does `open(path, "w")` directly on a run-contract file should switch to
this pattern.

### 3.6 Docs generated from code schema — because hand-maintained docs empirically drift
**Statement.** Schema documentation (including diagrams) must be generated from the same
dataclass/pydantic model that validates the data at write time — never authored by hand
alongside it.
**Why.** This is not a fussy preference — it's a measured failure mode: "Research from APIContext
found that 75% of production APIs do not conform to their published OpenAPI specifications,
making drift far more the rule than the exception... each piece of the pipeline falls out of sync
independently, as it was built independently"
([Bump.sh: Code-first, generate OpenAPI from code](https://bump.sh/blog/code-first-openapi/)).
The fix converges across ecosystems on one rule: "your OpenAPI spec is the single source of
truth... when you generate documentation directly from that spec, your docs are always accurate."
**How it applies.** This is already a settled invariant here ("schema diagrams must be GENERATED
from code schemas so docs can't drift") — cited with the general-software-engineering evidence
that motivates it as a hard rule, not a nice-to-have: whatever Python type defines `meta.json`'s
shape (dataclass/pydantic/TypedDict) is the only legal source for the diagram-generation script.

### 3.7 Determinism must be scoped explicitly — "bit-exact" is an environment claim, not just an algorithm claim
**Statement.** "Bit-exact replay" needs a stated boundary: same code version, same pinned library
versions, same recorded seed, same op ordering — bitwise reproducibility does not automatically
hold across different library versions, hardware, or compiler flags even with identical logic.
**Why.** From HPC/simulation practice: "bitwise reproducibility... does not extend across
different processors, compilers, or compilation flags," and is deliberately implemented as an
opt-in, performance-costing mode because achieving it requires controlling floating-point
operation order
([GAMER-2 paper](https://arxiv.org/pdf/1712.07070)). Robotics-sim-specific guidance draws the
same line: "determinism means same code, same machine, and same seed produce identical results
(bitwise, ideally)... reliability... refers to performance not collapsing... across machines"
([NVIDIA Isaac Lab reproducibility docs](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html)).
**How it applies.** Whatever guarantees "replay is bit-exact," the run/derived manifests should
record what's pinned to make that claim checkable — library versions used at derive time, and
any seed. Otherwise "bit-exact" is an assumption, not a verified property, and will quietly stop
being true after the next `pip install -U`.

### 3.8 For the undecided local catalog: prefer SQLite (or plain JSON/JSONL) over a server database
**Statement.** At this scale, the retrieval/catalog layer should be a single-file, no-server-process
store, not a client/server RDBMS.
**Why.** "An SQLite database file with a defined schema often makes an excellent application file
format... SQLite is designed to provide local data storage for individual applications" as
opposed to client/server engines "designed for a shared repository of enterprise data"
([SQLite: Appropriate Uses referenced via "Consider SQLite"](https://blog.wesleyac.com/posts/consider-sqlite);
package/O'Reilly framing in the same search set). The explicit trade-off: SQLite "only allows one
write operation at a time" — a limitation that is irrelevant here given the single-writer
principle (3.4) already holds for run capture, and the catalog itself has at most one writer
(the ingest step) at a time too.
**How it applies.** Directly answers the open "no DB decided yet" question raised in the task:
SQLite is the right-sized choice if/when queries (by defect type, part, date, camera rig) get
complex enough that grep-ing JSON files stops being tractable — and it stays a single file that
backs up/copies exactly like every other artifact in this files-first system. Don't reach for
Postgres/MySQL; there's no second machine querying this.

### 3.9 STAC's Item/Catalog split is a reusable *pattern* for the future retrieval layer (not a spec to adopt)
**Statement.** Separate a small, per-record descriptive document (STAC calls it an "Item": one
asset's searchable metadata + links to where its files live) from a lightweight index that groups
Items into browsable Catalogs/Collections — keep the index decoupled from where the underlying
asset bytes are stored.
**Why.** "STAC has been designed to be simple, flexible, and extensible... geospatial assets can
be stored anywhere... the json files... are all indexed in a central place"
([STAC intro tutorial](https://stacspec.org/en/tutorials/intro-to-stac/),
[radiantearth/stac-spec](https://github.com/radiantearth/stac-spec)). STAC explicitly supports
this being backed by a local filesystem, not just cloud APIs (e.g., pygeoapi's FileSystem
Provider).
**How it applies.** This directly targets gap #5 from the prior-art survey (no surveyed skill has
a real queryable local index, only a hand-maintained table). Model: each run gets a small
"Item"-like JSON record (defect type, part, date, camera rig, links to `run.json`/derived paths);
a lightweight index (SQLite table or a single JSONL of these Items, per 3.8) makes them
queryable. Borrow the *separation of concerns* — not STAC's geospatial-specific fields, GeoJSON
geometry, or its REST API layer, none of which apply here.

---

## The 10 rules to bet the skill on (see final response for the raw list + rationale)

Selected as the highest-leverage subset of the above: 3.4 (single-writer → read-only), 3.5
(atomic writes as the mechanism), 3.3 (idempotent derivation keyed by run/method/step), 1.2
(sibling-file completeness as a routine check), 1.3 (self-describing conventions in-band), 1.1
(calibration vs. per-frame separation), 3.7 (scoped determinism claim), 2.1+2.2 (immutable raw +
lightweight content hash, DVC-the-idea not DVC-the-tool), 3.6 (docs generated from code schema,
with the 75%-drift evidence), 3.8+3.9 (SQLite/STAC-pattern for the undecided catalog layer).

---

## Sources consulted

**Robotics dataset conventions**
- LeRobotDataset v3.0 docs: https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3
- LeRobot v3 blog: https://huggingface.co/blog/lerobot-datasets-v3
- LeRobot v3 porting guide: https://huggingface.co/docs/lerobot/en/porting_datasets_v3
- RLDS spec: https://github.com/google-research/rlds
- RLDS overview (Voxel51): https://voxel51.com/glossary/rlds
- RLDS Google Research blog: https://research.google/blog/rlds-an-ecosystem-to-generate-share-and-use-datasets-in-reinforcement-learning/
- Open X-Embodiment paper: https://arxiv.org/html/2310.08864v9
- Open X-Embodiment overview: https://www.emergentmind.com/topics/open-x-embodiment-dataset
- MCAP evaluation paper (Hurliman): https://mcap.dev/files/evaluation.pdf
- MCAP RosCon 2022 talk: http://download.ros.org/downloads/roscon/2022/MCAP%20A%20Next-Generation%20File%20Format%20for%20ROS%20Recording.pdf
- Introducing MCAP (Foxglove): https://foxglove.dev/blog/introducing-the-mcap-file-format
- MCAP vs ROS bag: https://segments.ai/blog/mcap-vs-ros-bag-simplifying-multi-modal-sensor-data-in-robotics/
- nuScenes schema docs: https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md
- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- KITTI format tutorial: https://deepdatamininglearning.readthedocs.io/en/latest/KITTI_Tutorial.html
- FastUMI-100K (ROS clock sync): https://arxiv.org/pdf/2510.08022
- Rosario Dataset v2 (sync without hardware sync): https://arxiv.org/pdf/2508.21635

**ML dataset management practice**
- Datasheets for Datasets (Gebru et al.): https://arxiv.org/abs/1803.09010
- Croissant paper (NeurIPS 2024): https://arxiv.org/pdf/2403.19546
- Croissant overview (Google Research): https://research.google/blog/croissant-a-metadata-format-for-ml-ready-datasets/
- HF Hub dataset versioning: https://huggingface.co/docs/hub/datasets-adding
- DVC start guide: https://doc.dvc.org/start
- DVC pipelines: https://doc.dvc.org/user-guide/pipelines
- DVC versioning example: https://doc.dvc.org/example-scenarios/versioning-data-and-models
- DVC reproducibility summary: https://medium.com/mantisnlp/data-version-control-for-reproducible-analytical-pipelines-5255782d355d
- WebDataset repo: https://github.com/webdataset/webdataset
- WebDataset PyTorch blog: https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/
- Zarr intro (Earthmover): https://www.earthmover.io/blog/what-is-zarr/
- Zarr storage spec (OGC): https://www.ogc.org/publications/standard/zarr-storage-specification/
- Content-addressable storage / package manager hashing: https://nesbitt.io/2026/07/07/content-addressing-in-package-managers.html

**General data-engineering canon**
- DDIA reading notes: https://github.com/ahmedhammad97/Designing-Data-Intensive-Applications-Notes
- DDIA summary: https://danlebrero.com/2021/09/01/designing-data-intensive-applications-summary/
- Schema-on-read vs schema-on-write (Dremio): https://www.dremio.com/wiki/schema-on-read-vs-schema-on-write/
- Medallion architecture (Databricks): https://docs.databricks.com/aws/en/lakehouse/medallion
- Single-writer principle (Designing Event-Driven Systems, O'Reilly): https://www.oreilly.com/library/view/designing-event-driven-systems/9781492038252/ch11.html
- Atomic writes (LWN): https://lwn.net/Articles/789600/
- Atomic file writing in Python: https://docs.bswen.com/blog/2026-04-04-atomic-file-writing-python/
- Code-first OpenAPI / docs drift stat (Bump.sh): https://bump.sh/blog/code-first-openapi/
- GAMER-2 bitwise reproducibility: https://arxiv.org/pdf/1712.07070
- Isaac Lab reproducibility/determinism docs: https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html
- Consider SQLite: https://blog.wesleyac.com/posts/consider-sqlite
- STAC intro tutorial: https://stacspec.org/en/tutorials/intro-to-stac/
- STAC spec repo: https://github.com/radiantearth/stac-spec
- data-engineer.md distillation (idempotency, validate-at-checkpoints), already surfaced in
  `existing-data-skills.md`: https://github.com/rohitg00/awesome-claude-code-toolkit/blob/main/agents/data-ai/data-engineer.md

See also `existing-data-skills.md` in this directory for the Claude Code / agent-skill prior-art
survey (what existing skills cover, and the 7 identified gaps this canon partly addresses).
