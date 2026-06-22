---
name: feedback_ultrawide_layouts
description: User has ultrawide monitor — prefer wide layouts (more cols, fewer rows) for plots, contact sheets, grids
type: feedback
originSessionId: 672e223a-9b37-4a53-84d3-29a68457f36d
---
For any multi-panel grid/figure (eval-protocol overviews, ablation plots, contact sheets, comparison strips, sensitivity dashboards): default to **wide aspect** (more columns, fewer rows). Target aspect ratio ≈ 3.5 (cols/rows).

**Why:** user has an ultrawide monitor — tall figures waste horizontal real-estate and force scrolling. User said verbatim "in memory I have ultrawide monitor so better to make everything wide not high".

**How to apply:**
- For N panels, prefer grids like 10×3 (N=30), 15×4 (N=60), 10×2 (N=20) over square-ish layouts.
- When implementing auto-layout: pick `rows` minimizing `|cols/rows - 3.5| + small_penalty_for_empty_cells`.
- Single-figure plots: `figsize=(width, height)` with `width/height ≥ 1.6`.
- Applies to: matplotlib subplots, PIL contact sheets, dashboards, `subplots(rows, cols, ...)` choices.
- Does NOT mean every figure must be ultrawide — single-axis plots can be ~1.6 ratio. The rule kicks in for grid layouts where the choice is dictated by `N`.
