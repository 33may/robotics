# IWM pass 2: how 3D gets "coloured" with hierarchical semantic labels

**Date:** 2026-10-12
**Question (Anton):** how can a 3D representation be coloured with semantic labels,
preferably hierarchical ones (one place on the object belongs to object > face > sticker
> text line at once)?
**Corpus:** the 11 sources in `docs/research/iwm/papers/`. There are four reader notes in
this folder:
- [octree-family](octree-family.md): OctoMap, Wurm 2011, Larsson 2022
- [open-vocab-3d](open-vocab-3d.md): Octree-Graph, OmniSeg3D, OctLLM
- [scene-graph-survey](scene-graph-survey.md): Rotondi 2026
- [graph-memory-and-tiles](graph-memory-and-tiles.md): GraphPad, ObsGraph, ZoomEye, 3D Tiles

The lead re-read the key numbers and quotes against the paper text (listed at the end).
Everything else is as the readers reported it.

## The short answer

No source stores labels from several levels on one 3D cell. Every source does one of
three things.

| Pattern | Where the label lives | How levels exist | Sources |
|---|---|---|---|
| **1. Label on the cell** | Payload in each voxel: any data type (OctoMap §3.5.1), or a class distribution, top 3 + "other" (Larsson) | Only spatially: a parent cell summarises its 8 children (max/average; Larsson: a weighted mixture of distributions). One label slot per cell | OctoMap, Larsson |
| **2. Label on a node that owns geometry** | On the node; the node points at its piece of geometry (its own octree, a point set, "the subset of metric map points it occupies") | The node tree / graph layers; a cell's levels are its chain of ancestor nodes | Wurm (sub-map tree), Octree-Graph, GraphPad, the survey's general model (V_g, Eq. 1) |
| **3. Label on images and views; geometry kept outside** | On image or crop nodes, linked to geometry by a visibility matrix | Room / view / object layers; a co-visibility cluster tree | ObsGraph, ZoomEye (2D only) |

**Pattern 2 is what the field does.** Three readers found it independently. "Colouring
the 3D" in practice means a labelled node points at a region of geometry. The cell does
not carry the label.

**How one place gets all its levels at once:** never by storing several labels per cell.
It follows from where the place sits:
- Wurm: the chain of sub-maps it lies in.
- 3D Tiles: spatial nesting. A child's content must lie inside its parent's bounding
  volume, and every level carries its own typed metadata and its own geometric error.

## Mechanisms worth keeping, by source

- **Wurm 2011: a node owns its own octree in its own frame.**
  - Node n = (map, 6D transform relative to the parent, parent, children, label).
  - The label sets the resolution: objects 1–16 mm, table 10 mm, background 50 mm (§V).
  - A new segment joins the deepest qualifying ancestor (Eq. 7).
  - Moving a node means updating one transform, with no re-fusing. This is a ready
    mechanism for our per-view offset.
  - Limits: closed set of 3 labels, strict tree ("one object resting on two tables" is
    excluded), no label fusion, accurate poses assumed.
- **Larsson 2022: semantics decide geometric resolution.** Fine cells are kept only where
  the children disagree about classes that matter to the task (Jensen-Shannon divergence).
  The class set is closed and flat.
- **OctoMap: the payload slot.** It takes any data type, and each property needs its own
  fusion rule. It also has the `castRay` visibility query.
- **Octree-Graph:**
  - Each object node holds a fused CLIP feature, a caption feature, a centre, relation
    edges, and an adaptive octree. The octree holds occupancy only; max depth is 4.
  - Its containment ratio |S_m ∩ S_n| / |S_n| is a usable "is inside" test. But the
    paper uses it to delete a segment that contains semantically different sub-segments.
    That is the opposite of what an inward tree needs, where such a segment is a parent.
  - View weighting (IFA, Eq. 1).
- **OmniSeg3D §3.1:** a containment tree per image, without training. SAM masks are split
  into disjoint patches, and a vote matrix counts the masks they share.
  - Lifting it to 3D (§3.2) needs a NeRF trained per scene.
  - Its levels are similarity thresholds, not names. Only 2 levels were evaluated.
- **GraphPad: links both ways between nodes and frames.**
  - Each node lists the keyframes that saw it. A per-keyframe log lists the nodes visible
    in that frame, and that log was the largest single gain in their ablation.
  - Identity is a 2-of-3 vote in code (visual, caption, overlap within 5 cm).
  - Points are downsampled to 2 cm voxels.
  - The VLM writes JSON patches with evidence pointers, and nothing checks what it writes.
- **ObsGraph: a VLM picks the region and the strategy, code computes the pose.** The
  strategies are closer, wider and a different angle. SAM makes the mask, and code turns
  it into a pose, because VLMs "lack accurate coordinate-level reasoning". This is the
  closest thing in the corpus to Anton's "pivot on a detail".
- **3D Tiles:**
  - Typed metadata at every level (tileset, tile, content, feature, down to vertex).
  - `ADD` refinement shows parent and children together; `REPLACE` swaps the parent for
    its children.
  - No rule passes metadata values from parent to child in the core spec. The companion
    metadata specs are not in the corpus.
- **The survey: three challenges that match ours.**
  - Layers are "hard-coded and inflexible".
  - Consistency is "treated almost exclusively as a geometric problem".
  - "Deferred semantics": keep "a limited set of explicit semantic attributes" and
    retrieve the rest on demand.

## What no source covers (the gap)

- Open-vocabulary names on 3–4 nested levels inside ONE object. The deepest found is one
  level below the object (FunGraph-style "is part of" edges, per the survey).
- Any detail at sub-cm scale. Our labels are ~5 mm.
- Fusing labels when the pose error (~10 mm per view, for us) is larger than the detail
  (~5 mm).
- Semantic agreement across views, i.e. whether two views agree on what a thing is.
- Semantics driving where to look next, beyond Larsson's resolution and ObsGraph's
  refinement.

## Fetch next (merged from the four notes, not fetched)

- FOUND-it, 2026 (granularity on demand)
- Search3D (hierarchical open-vocabulary 3D segmentation)
- HOV-SG
- Clio (task-driven granularity)
- FunGraph / OpenFunGraph (part level)
- KeySG (keyframe retrieval)
- Open3DSG (best view per node)
- DC² (a parent summarises its children's descriptions)
- xFLIE (the only inspection paper)
- The 3D Tiles metadata companion specs (`EXT_structural_metadata`, 3D Metadata)

## Re-read by the lead

Checked against the paper text:
- OmniSeg3D 91.3 / 88.9 vs 93.1 / 80.4.
- Octree-Graph L_max = 4 and its containment filter.
- Wurm: 10 mm / 50 mm and "one object resting on two tables".
- Larsson: "three most likely classes".
- GraphPad: 0.02 m voxels, δ_g = 5 cm, "at least two of".
- ObsGraph: "coordinate-level reasoning".
- ZoomEye: "may not align well".
- Survey: "hard-coded and inflexible", "almost exclusively", "limited set of explicit
  semantic attributes".
