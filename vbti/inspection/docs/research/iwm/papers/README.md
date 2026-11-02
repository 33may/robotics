# IWM papers

Papers kept for the IWM research. PDFs are in `pdf/`, markdown versions in `md/`, the
converters in `_tools/`. Fetched and converted on 2026-10-06.

| Paper | File name | Anton's mark, 2026-10-06 |
|---|---|---|
| OctoMap. Hornung, Wurm, Bennewitz, Stachniss, Burgard. Autonomous Robots 2013 | `hornung-2013-octomap` | "good one, we will probably use it as the base". See 3.5.1 (nodes can store extra data) and 3.5.2 (octree hierarchies) |
| Hierarchies of Octrees for Efficient 3D Mapping. Wurm et al. IROS 2011 | `wurm-2011-hierarchies-of-octrees` | Not marked. Added because section 3.5.2 of OctoMap is a summary of this paper |
| Information-theoretic Abstraction of Semantic Octree Models. Larsson et al. arXiv 2209.10035 | `larsson-2022-semantic-octree-abstraction` | "sounds really good for our case but a lot of math to go through" |
| Open-Vocabulary Octree-Graph. Wang et al. ICCV 2025, arXiv 2411.16253 | `wang-2025-octree-graph` | "relevant but applies to scenes, might take some ideas for the inside-object unrolling" |
| 3D Scene Graphs: Open Challenges and Future Directions. Rotondi et al. arXiv 2606.19383 | `rotondi-2026-3d-scene-graphs-survey` | "good to have, will read later" |
| GraphPad. Ali et al. arXiv 2506.01174 | `ali-2025-graphpad` | "this is almost exactly my solution" |
| OmniSeg3D. Ying et al. arXiv 2311.11666 | `ying-2023-omniseg3d` | "approach A is good", "we will trust it, we check how it works with experiments". Only section 3.1 matters: the hierarchy built per image from SAM masks, without training |
| ObsGraph. Lee et al. arXiv 2606.24068 | `lee-2026-obsgraph` | Not marked |
| ZoomEye. arXiv 2411.16044 | `zoomeye-2024` | Not marked |
| Octrees as an Explicit 3D Language (OctLLM). Dan et al. arXiv 2610.02388 | `dan-2026-octllm` | Not marked |
| 3D Tiles specification, core. OGC community standard | `3d-tiles-spec` (markdown only) | Not marked |

Left out on Anton's word: MAP-ADAPT (one object in the scene for now) and Panoptic
Multi-TSDFs (the scene is static).

## How the markdown was made

- **arXiv papers:** from the arXiv HTML page with pandoc. Formulas are the authors' own
  LaTeX. Figures are links to arxiv.org. Complex tables can come out imperfect, so check
  the PDF before quoting a number.
- **OctoMap and Wurm et al.:** no HTML exists, so the text was extracted from the PDF,
  column by column. Formulas, tables and figures are not recovered.
- **3D Tiles:** a light conversion from AsciiDoc. Tables keep their AsciiDoc form.

## Adding a paper

```
# arXiv paper that has an HTML page
curl -sL https://arxiv.org/pdf/<id>  -o pdf/<name>.pdf
curl -sL https://arxiv.org/html/<id> -o /tmp/<name>.html
python3 -I _tools/arxiv_html_to_md.py /tmp/<name>.html md/<name>.md <id> <name>.pdf _tools/clean.lua

# PDF only, two columns
python3 -I _tools/pdf_to_md.py pdf/<name>.pdf md/<name>.md "<title>" "<source url>"
```
