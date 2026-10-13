# IWM pass 1, round 1: branch plan

**Date:** 2026-10-06
**Contract:** [`inspection/docs/2026-10-06-iwm-research-foundation.md`](../../../../../inspection/docs/2026-10-06-iwm-research-foundation.md)
**Question of this pass:** which approaches exist, and which are actually used, to
compress a robot's internal representation of the world?

## How every approach is described

One lens for all branches, so the families can be compared side by side.

| Field | Meaning |
|---|---|
| Atomic unit | The smallest thing the structure stores (RQ4) |
| Kept | What survives compression (RQ9) |
| Discarded | What is thrown away, or left outside to be fetched (RQ9) |
| Hierarchy | Levels of granularity, if any (RQ5) |
| Update | How new observations enter, how drift and conflicts are handled, by whom (RQ6, RQ7) |
| Known and unknown | How observation quality and "not looked at" are represented (RQ10) |
| Identity | How repeated sightings become one entry, and how a wrong merge is undone (RQ11) |
| Links | How geometry, image and words are tied together (RQ12) |
| Frame | World, object, robot or viewpoint (RQ13) |
| Reader | Who consumes it and through what interface (RQ8) |
| Cost | Bytes, tokens, latency |
| Training | Needed or not |
| Used where | Shipped systems, open-source stacks, or papers only |

## Branches

| # | Slug | Family | Compresses by |
|---|---|---|---|
| 1 | `geometric-maps` | Occupancy grids, octrees, TSDF and voxel hashing, VDB, surfels, meshes, neural implicit and Gaussian-splat maps | Sparsity, hierarchy, fusion of many measurements into one cell |
| 2 | `learned-representations` | Language-feature fields, 3D tokens for language models, latent world models, memory tokens of feed-forward reconstruction | A learned vector in place of raw data |
| 3 | `scene-graphs-and-hierarchy` | Object maps, 3D scene graphs, part-level and functional graphs, task-driven pruning | Naming things and keeping only nodes and relations |
| 4 | `llm-agent-memory` | Memory of embodied language agents, and memory of language agents in general | Summarising, indexing and retrieving on demand |
| 5 | `task-driven-abstractions` | Keypoints, constraint maps, symbolic predicates, shape primitives, belief state, theory of sufficient representations | Keeping only what the task needs |

## Rules between branches

- Geometry of a neural or Gaussian map goes to branch 1. Language features stored on any
  map go to branch 2.
- Branch 3 owns identity and merging (RQ11) and hierarchy (RQ5). Branch 1 owns
  observation quality per element (RQ10).
- Branch 4 owns who maintains the memory (RQ7) and the read interface (RQ8).
- Branch 5 owns the question of what must be preserved (RQ9).
- View planning is already mapped in `docs/research/view-planning/`. Branch 1 takes from
  it only what state is kept per surface element.

## What we already hold

From the map of 2026-10-06 (memory leaf `perception/llm-facing-3d-structures-prior-art`):

- No training-free structure at the scale of one object's surface, read by a text-only
  language model.
- Observation quality per element is never shown to a language model.
- MIT 2510.16643: query tools beat pasting the graph into context once the graph is
  large; a free Python API was worst.

Branches extend this. They do not repeat it.

## Round 2 method

One mapper per branch. Abstracts and landing pages for breadth, a full read only for
the few anchor papers whose mechanism cannot be told from the abstract. Every number is
read from its source on the day and marked with where it was read. Output: one map per
branch in `round2_map/`.

## Not in this pass

- Group A of the contract (what a robot needs to ask).
- Execution and reachability.
